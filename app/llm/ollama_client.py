from __future__ import annotations

import asyncio
import json
import random
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional

import httpx

from app.core.errors import AppError
from app.core.logging import get_logger


ApiMode = Literal["generate", "chat", "auto"]


@dataclass
class OllamaResult:
    text: str
    model: str
    endpoint: Literal["generate", "chat"]
    latency_ms: int
    tokens_estimate: int
    raw: Dict[str, Any]


class OllamaClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 120.0,
        api_mode: ApiMode = "generate",
        retry_max_attempts: int = 3,
        retry_backoff_base: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_mode = api_mode
        self.retry_max_attempts = max(1, int(retry_max_attempts))
        self.retry_backoff_base = float(retry_backoff_base)

        timeout = httpx.Timeout(timeout_s, connect=min(10.0, timeout_s))
        self._client = httpx.AsyncClient(timeout=timeout)
        self._log = get_logger("yuki.ollama")

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        if retry_after is not None and retry_after > 0:
            return min(30.0, retry_after)
        base = self.retry_backoff_base
        wait = base * (2 ** (attempt - 1))
        wait += random.random() * 0.1
        return min(10.0, wait)

    def _estimate_tokens(self, raw: Dict[str, Any], text: str, prompt_len: int) -> int:
        pe = raw.get("prompt_eval_count")
        ec = raw.get("eval_count")
        if isinstance(pe, int) or isinstance(ec, int):
            return int((pe or 0) + (ec or 0))
        return max(1, int((prompt_len + len(text)) / 4))

    async def health(self) -> Dict[str, Any]:
        url = self._url("/api/tags")
        try:
            r = await self._client.get(url)
            ok = r.status_code == 200
            data = r.json() if ok else None
            return {"reachable": ok, "status_code": r.status_code, "tags": data}
        except Exception as e:
            return {"reachable": False, "error": str(e)}

    async def complete(
        self,
        *,
        system: str,
        prompt: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
    ) -> OllamaResult:
        if stream:
            raise ValueError("complete() is non-streaming; use stream()")
        mode = self.api_mode
        if mode == "generate":
            return await self._generate(system=system, prompt=prompt)
        if mode == "chat":
            return await self._chat(messages=messages)
        try:
            return await self._chat(messages=messages)
        except AppError as e:
            if e.code == "ollama_not_found":
                return await self._generate(system=system, prompt=prompt)
            raise

    async def stream(
        self,
        *,
        system: str,
        prompt: str,
        messages: List[Dict[str, str]],
    ) -> AsyncGenerator[str, None]:
        mode = self.api_mode
        if mode == "generate":
            async for d in self._generate_stream(system=system, prompt=prompt):
                yield d
            return
        if mode == "chat":
            async for d in self._chat_stream(messages=messages):
                yield d
            return
        try:
            async for d in self._chat_stream(messages=messages):
                yield d
            return
        except AppError as e:
            if e.code == "ollama_not_found":
                async for d in self._generate_stream(system=system, prompt=prompt):
                    yield d
                return
            raise

    async def _post_json_with_retries(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = self._url(path)

        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                t0 = time.perf_counter()
                r = await self._client.post(url, json=payload)
                dt_ms = int((time.perf_counter() - t0) * 1000)

                if r.status_code == 404:
                    raise AppError(
                        detail=f"Ollama endpoint not found: {path}",
                        code="ollama_not_found",
                        status_code=502,
                    )

                if r.status_code in (429,) or 500 <= r.status_code <= 599:
                    retry_after = None
                    if r.status_code == 429:
                        ra = r.headers.get("Retry-After")
                        try:
                            retry_after = float(ra) if ra else None
                        except Exception:
                            retry_after = None

                    if attempt < self.retry_max_attempts:
                        wait = self._backoff(attempt, retry_after=retry_after)
                        self._log.warning(
                            "Ollama retryable response %s on %s (attempt %s/%s, wait=%.2fs, latency_ms=%s)",
                            r.status_code,
                            path,
                            attempt,
                            self.retry_max_attempts,
                            wait,
                            dt_ms,
                        )
                        await asyncio.sleep(wait)
                        continue

                r.raise_for_status()
                data = r.json()
                if isinstance(data, dict) and data.get("error"):
                    raise AppError(
                        detail=str(data.get("error")),
                        code="ollama_error",
                        status_code=502,
                    )
                return data

            except AppError as e:
                if e.code == "ollama_not_found":
                    raise
                if attempt < self.retry_max_attempts:
                    wait = self._backoff(attempt)
                    self._log.warning(
                        "Ollama AppError retry (code=%s, attempt %s/%s, wait=%.2fs): %s",
                        e.code,
                        attempt,
                        self.retry_max_attempts,
                        wait,
                        e.detail,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise

            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.retry_max_attempts:
                    wait = self._backoff(attempt)
                    self._log.warning(
                        "Ollama network/timeout retry (attempt %s/%s, wait=%.2fs): %s",
                        attempt,
                        self.retry_max_attempts,
                        wait,
                        str(e),
                    )
                    await asyncio.sleep(wait)
                    continue
                raise AppError(
                    detail=f"Ollama unreachable: {e}",
                    code="ollama_unavailable",
                    status_code=502,
                )

            except httpx.HTTPStatusError as e:
                body = None
                try:
                    body = e.response.text
                except Exception:
                    body = None
                raise AppError(
                    detail=f"Ollama HTTP error {e.response.status_code}: {body or e}",
                    code="ollama_http_error",
                    status_code=502,
                )

            except Exception as e:
                raise AppError(
                    detail=f"Unexpected Ollama error: {e}",
                    code="ollama_error",
                    status_code=502,
                )

        raise AppError(detail="Ollama request failed", code="ollama_error", status_code=502)

    async def _generate(self, *, system: str, prompt: str) -> OllamaResult:
        payload = {"model": self.model, "system": system, "prompt": prompt, "stream": False}
        t0 = time.perf_counter()
        data = await self._post_json_with_retries("/api/generate", payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = str(data.get("response", ""))
        tokens = self._estimate_tokens(data, text, prompt_len=len(prompt) + len(system))
        return OllamaResult(
            text=text,
            model=self.model,
            endpoint="generate",
            latency_ms=latency_ms,
            tokens_estimate=tokens,
            raw=data,
        )

    async def _chat(self, *, messages: List[Dict[str, str]]) -> OllamaResult:
        payload = {"model": self.model, "messages": messages, "stream": False}
        t0 = time.perf_counter()
        data = await self._post_json_with_retries("/api/chat", payload)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        msg = data.get("message") or {}
        text = str(msg.get("content", ""))
        prompt_len = sum(len(m.get("content", "")) for m in messages)
        tokens = self._estimate_tokens(data, text, prompt_len=prompt_len)
        return OllamaResult(
            text=text,
            model=self.model,
            endpoint="chat",
            latency_ms=latency_ms,
            tokens_estimate=tokens,
            raw=data,
        )

    async def _generate_stream(self, *, system: str, prompt: str) -> AsyncGenerator[str, None]:
        url = self._url("/api/generate")
        payload = {"model": self.model, "system": system, "prompt": prompt, "stream": True}
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code == 404:
                    raise AppError(detail="Ollama /api/generate not found", code="ollama_not_found", status_code=502)
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise AppError(detail=str(chunk.get("error")), code="ollama_error", status_code=502)
                    delta = chunk.get("response")
                    if delta:
                        yield str(delta)
                    if chunk.get("done") is True:
                        break
        except httpx.HTTPStatusError as e:
            raise AppError(detail=f"Ollama streaming HTTP error {e.response.status_code}", code="ollama_http_error", status_code=502)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise AppError(detail=f"Ollama unreachable: {e}", code="ollama_unavailable", status_code=502)

    async def _chat_stream(self, *, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        url = self._url("/api/chat")
        payload = {"model": self.model, "messages": messages, "stream": True}
        try:
            async with self._client.stream("POST", url, json=payload) as resp:
                if resp.status_code == 404:
                    raise AppError(detail="Ollama /api/chat not found", code="ollama_not_found", status_code=502)
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    chunk = json.loads(line)
                    if isinstance(chunk, dict) and chunk.get("error"):
                        raise AppError(detail=str(chunk.get("error")), code="ollama_error", status_code=502)
                    msg = chunk.get("message") or {}
                    delta = msg.get("content")
                    if delta:
                        yield str(delta)
                    if chunk.get("done") is True:
                        break
        except httpx.HTTPStatusError as e:
            raise AppError(detail=f"Ollama streaming HTTP error {e.response.status_code}", code="ollama_http_error", status_code=502)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise AppError(detail=f"Ollama unreachable: {e}", code="ollama_unavailable", status_code=502)
