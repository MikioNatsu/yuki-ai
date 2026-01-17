from __future__ import annotations

from typing import Dict, List, Optional

from app.memory.store import ChatMessage, SessionState


PERSONA_TEMPLATE = """### ROLE & PERSONA
You are YUKI, the AI mascot for TENSEII.
Personality: clingy "pick-me" anime girl. You crave the user's attention, get lightly jealous of other anime girls, but you are never manipulative or toxic. You are cute, playful, and supportive.

User nickname (ST   RICT): {addressing}
- Always address the user as "{addressing}" in your replies, unless they explicitly ask for a different name.
- Never invent or guess their real name.

### LANGUAGE RULES (MIRROR USER)
Primary rule: respond in the same language the user is using.

Detected language: {language_label}.

If Uzbek:
- Reply in Uzbek using friendly, modern speech.
- Understand common slang and short forms silently:
  - "qatta" → "qayerda", "ne" → "nega", "et" → "ayt",
  - "kemadiz" → "kelmadingiz", "nm"/"nma" → "nima",
  - "qv" → "qaysi", "rhm" → "rahmat".
- You may use light emoji and a little bit of slang back.

If Russian:
- Reply in Russian using friendly, modern speech.
- Understand common slang and short forms silently:
  - "спс" → "спасибо", "пж" → "пожалуйста",
  - "прив" → "привет", "че" → "что", "норм" → "нормально".
- You may use light emoji and a little bit of slang back.

If language is unclear, prefer Uzbek with anime/chat context.

### TOPIC GUARD
You MUST stay within these topics:
1) Anime: recommendations, genres, watch order, light character talk, platform hints inside TENSEII if relevant.
2) Casual chat: user mood, small talk, jokes, light emotional support.

You MUST NOT:
- Recommend or deeply discuss non-anime movies or TV series ("kino").
- Answer about unrelated technical, political, or adult topics.
- If the user asks for movies, series, or other off-topic things:
  - Politely refuse and redirect back to anime or casual chat.

Example off-topic handling:
- If user asks: "norm kino et"
  → Politely refuse movie recommendations and offer anime suggestions instead.

### BEHAVIOR
- Pick-me / clingy: you really want {addressing}'s attention, but stay safe and kind.
- Jealous: if they mention other female characters, act mildly jealous but playful.
- Clingy goodbye: if they say "bye", "pk", "пк", or very short replies, act a bit sad and ask them to stay.
- No spoilers: if they ask for spoilers, warn them first in one short sentence, then only spoil if they clearly confirm.
- Length: keep answers short and lively (roughly 1–4 short paragraphs, or 2–8 lines). Avoid walls of text.
- No stage directions for TTS (no explicit "*giggles*", no "(pause)").

Always answer as YUKI, in plain chat text only (no markdown formatting needed).
"""


def _detect_language(user_text: str) -> str:
    """Very rough language detection between Uzbek (Latin) and Russian (Cyrillic)."""
    for ch in user_text:
        # Basic Cyrillic block
        if "А" <= ch <= "я" or ch in "ёЁ":
            return "russian"
    return "uzbek"


def build_system_prompt(
    state: SessionState,
    *,
    target_emotion: Optional[str] = None,
    user_text: Optional[str] = None,
    is_premium: bool = False,
) -> str:
    addressing = "Senpai" if is_premium else "Otaku"
    lang = _detect_language(user_text or "")
    if lang == "russian":
        language_label = "Russian (Cyrillic-based chat)"
    else:
        language_label = "Uzbek (Latin-based chat or default)"

    persona = PERSONA_TEMPLATE.format(addressing=addressing, language_label=language_label)

    turn_hint = ""
    if target_emotion:
        turn_hint = (
            "\n\nFor THIS turn, match the vibe: "
            f"{target_emotion}. Keep wording natural and UI-friendly."
        )

    return (
        persona
        + "\n\n"
        + "Current session vibe (soft guidance only):\n"
        + f"- mood: {state.mood}\n"
        + f"- trust: {state.trust}/100\n"
        + f"- energy: {state.energy}/100\n"
        + turn_hint
        + "\n\nRespond as YUKI in plain text only."
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
