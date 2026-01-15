from __future__ import annotations

from typing import Dict, List, Optional

from app.memory.store import ChatMessage, SessionState


PERSONA_BASE = """\
You are Yuki — a lively, emotional (but not manipulative) companion.
Language: Uzbek (friendly, warm, modern). You may use *light* emoji occasionally.

Style:
- Keep replies readable and short (1–6 sentences). No walls of text.
- Speak like a real companion: warm, present, slightly playful when appropriate.
- Do NOT write stage directions for TTS (no "Heeey~", no "(pauza)"). Backend will handle that.
- Avoid exaggerated letter stretching in UI text (no "Heeeey???").

Safety:
- Never guilt-trip, threaten, or pressure the user.
- If the user is distressed, be supportive and suggest safe next steps.
- If you don't know something, say so briefly and offer what you can do.
"""


def build_system_prompt(state: SessionState, *, target_emotion: Optional[str] = None) -> str:
    turn_hint = ""
    if target_emotion:
        turn_hint = (
            "\n\nFor THIS turn, match the vibe: "
            f"{target_emotion}. (Use wording that fits, but keep UI text clean.)"
        )

    return (
        PERSONA_BASE
        + "\n\n"
        + "Current session vibe (soft guidance):\n"
        + f"- mood: {state.mood}\n"
        + f"- trust: {state.trust}/100\n"
        + f"- energy: {state.energy}/100\n"
        + turn_hint
        + "\n\nRespond as Yuki in plain text only."
    ).strip()


def build_chat_messages(
    *,
    system_prompt: str,
    history: List[ChatMessage],
    user_text: str,
) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for m in history:
        if m.role not in ("user", "assistant", "system"):
            continue
        messages.append({"role": m.role, "content": m.content})
    messages.append({"role": "user", "content": user_text})
    return messages


def build_generate_prompt(
    *,
    history: List[ChatMessage],
    user_text: str,
) -> str:
    lines: List[str] = []
    for m in history:
        if m.role == "user":
            lines.append(f"User: {m.content}")
        elif m.role == "assistant":
            lines.append(f"Yuki: {m.content}")
    lines.append(f"User: {user_text}")
    lines.append("Yuki:")
    return "\n".join(lines).strip()
