from __future__ import annotations

import re
from dataclasses import replace

from app.memory.store import SessionState
from app.schemas import EmotionLabel, EmotionOut


_VOWELS = "aeiouөүAEIOUӨҮ"


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def sanitize_display_text(text: str) -> str:
    """Make UI text clean and readable (no heavy TTS cues)."""
    t = (text or "").strip()
    t = re.sub(r"^(assistant|yuki)\s*:\s*", "", t, flags=re.IGNORECASE)
    t = re.sub(r"```[\s\S]*?```", "", t).strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(rf"([{_VOWELS}])\1{{2,}}", r"\1\1", t)
    t = re.sub(r"!{2,}", "!", t)
    t = re.sub(r"\?{2,}", "?", t)
    t = re.sub(r"~{2,}", "~", t)
    return t


def infer_emotion(user_text: str) -> EmotionOut:
    """Heuristic emotion inference from *user* text (fast, offline)."""
    t = (user_text or "").lower().strip()

    label: EmotionLabel = "calm"
    valence = 0.55
    arousal = 0.35

    question_words = {"nima", "qanday", "nega", "qachon", "qayerda", "kim", "qaysi", "how", "why", "what"}
    if "?" in t or any(t.startswith(w + " ") for w in question_words):
        label = "curious"
        arousal = max(arousal, 0.45)

    sad_words = {
        "xafa", "yomon", "yig'lay", "yiglay", "ko'nglim", "konglim", "tushkun",
        "depress", "depression", "yolg'iz", "yolgiz", "charchadim", "zerikdim",
        "og'riyapti", "ogriyapti", "azob", "alam",
    }
    help_words = {"yordam", "iltimos", "muammo", "xato", "error", "404", "500", "qiyin", "qo'rq", "qorq"}
    excited_words = {"zo'r", "zor", "super", "ajoyib", "gap yo'q", "rahmat", "😍", "🔥", "🎉", "🤩"}
    playful_words = {"haha", "lol", "kulgili", "😄", "😂", "😁", "xax"}
    flirty_words = {"jonim", "sevgilim", "bo'sa", "bosa", "😘", "❤️", "💋"}
    surprised_words = {"voy", "rostdan", "rostmi", "jiddiy", "😳", "😮", "😲"}

    def has_any(words) -> bool:
        return any(w in t for w in words)

    if has_any(sad_words):
        label = "empathetic"
        valence = 0.2
        arousal = 0.25

    if has_any(help_words):
        label = "comforting" if label == "empathetic" else "curious"
        valence = min(valence, 0.45)
        arousal = max(arousal, 0.45)

    if has_any(flirty_words):
        label = "flirty"
        valence = 0.85
        arousal = 0.6

    if has_any(playful_words):
        label = "playful"
        valence = 0.75
        arousal = 0.65

    if has_any(excited_words) and label not in ("empathetic", "comforting"):
        label = "excited"
        valence = 0.9
        arousal = 0.75

    if has_any(surprised_words) and label not in ("empathetic", "comforting"):
        label = "surprised"
        valence = max(valence, 0.6)
        arousal = max(arousal, 0.8)

    if "!" in t:
        arousal += 0.1
    if t.count("?") >= 2 and label not in ("empathetic", "comforting"):
        label = "surprised"
        arousal = max(arousal, 0.8)

    return EmotionOut(label=label, valence=_clamp01(valence), arousal=_clamp01(arousal))


def update_session_state(state: SessionState, emotion: EmotionOut) -> SessionState:
    """Tiny state machine: keep vibe consistent between turns."""
    trust = state.trust
    energy = state.energy
    mood = state.mood

    if emotion.label in ("excited", "playful", "flirty"):
        trust = min(100, trust + 2)
        energy = min(100, energy + 3)
        mood = "bright"
    elif emotion.label in ("comforting", "empathetic"):
        trust = min(100, trust + 1)
        energy = max(0, energy - 1)
        mood = "soft"
    elif emotion.label == "surprised":
        energy = min(100, energy + 2)
        mood = "spark"
    elif emotion.label == "calm":
        energy = max(0, energy - 1)
        mood = "calm"

    return replace(state, mood=mood, trust=trust, energy=energy, last_emotion=emotion.label)


def _ajoyib_variant(label: EmotionLabel) -> str:
    return {
        "excited": "Aaaajoyiiib!!",
        "playful": "Ajoyib-ku! hehe~",
        "comforting": "Ajoyib... hammasi asta-sekin joyiga tushadi.",
        "empathetic": "Ajoyib... men sen bilanman.",
        "flirty": "Ajoyiiib~ jonim~",
        "surprised": "Ajoyib?! Voooy!",
        "curious": "Ajoyib... endi batafsil ayt-chi?",
        "calm": "Ajoyib.",
        "sad": "Ajoyib... (oh)...",
    }.get(label, "Ajoyib.")


def direct_speech(display_text: str, emotion: EmotionOut) -> str:
    """Convert display_text -> speech_text (TTS actor direction)."""
    label = emotion.label
    base = (display_text or "").strip()

    base_for_speech = re.sub(r"\bAjoyib\b", _ajoyib_variant(label), base, flags=re.IGNORECASE)

    prefix = ""
    suffix = ""
    if label == "excited":
        prefix = "Heeey! "
        suffix = "!"
    elif label == "playful":
        prefix = "Hihi~ "
    elif label == "flirty":
        prefix = "Hmm~ "
        suffix = " ~"
    elif label == "comforting":
        prefix = "Hey... "
    elif label == "empathetic":
        prefix = "Mm... "
    elif label == "surprised":
        prefix = "Voooy?! "
    elif label == "calm":
        prefix = "Xo‘p... "
    else:
        prefix = "Hmm... "

    speech = (prefix + base_for_speech + suffix).strip()

    if label in ("comforting", "empathetic"):
        speech = speech.replace(". ", "... ")
    if label in ("excited", "surprised"):
        speech = speech.replace(", ", ", ... ")

    speech = re.sub(r"\s+", " ", speech).strip()
    return speech
