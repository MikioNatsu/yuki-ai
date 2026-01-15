from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from threading import Lock
from typing import Deque, Dict


@dataclass
class RateLimitResult:
    allowed: bool
    retry_after_s: float = 0.0
    limit_per_minute: int = 0


class InMemoryRateLimiter:
    """Very small in-memory per-key rate limiter (sliding window)."""

    def __init__(self, *, per_minute: int, window_s: float = 60.0) -> None:
        self.per_minute = per_minute
        self.window_s = window_s
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = Lock()

    def check(self, key: str) -> RateLimitResult:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            cutoff = now - self.window_s
            while q and q[0] < cutoff:
                q.popleft()

            if len(q) >= self.per_minute:
                retry_after = max(0.0, (q[0] + self.window_s) - now)
                return RateLimitResult(
                    allowed=False, retry_after_s=retry_after, limit_per_minute=self.per_minute
                )

            q.append(now)
            return RateLimitResult(allowed=True, retry_after_s=0.0, limit_per_minute=self.per_minute)
