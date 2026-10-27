"""Persistence helpers for the currently selected TTS voice profile."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CURRENT_VOICE_FILE = Path("config/current_voice.json")
LEGACY_VOICE_ID_FILE = Path("config/custom_voice_id.txt")


@dataclass(frozen=True, slots=True)
class VoiceSelection:
    """Selected cloned voice plus the matching TTS model."""

    voice_id: str
    target_model: str | None = None


def load_selected_voice() -> VoiceSelection | None:
    """Read the currently selected voice, with legacy txt fallback."""
    if CURRENT_VOICE_FILE.exists():
        try:
            data = json.loads(CURRENT_VOICE_FILE.read_text(encoding="utf-8"))
            voice_id = str(data.get("voice_id", "")).strip()
            target_model = str(data.get("target_model", "")).strip() or None
            if voice_id:
                return VoiceSelection(voice_id=voice_id, target_model=target_model)
        except Exception:
            pass

    if LEGACY_VOICE_ID_FILE.exists():
        voice_id = LEGACY_VOICE_ID_FILE.read_text(encoding="utf-8").strip()
        if voice_id:
            return VoiceSelection(voice_id=voice_id, target_model=None)
    return None


def save_selected_voice(voice_id: str, target_model: str | None = None) -> VoiceSelection:
    """Persist the current voice selection in JSON and legacy txt formats."""
    voice_id = voice_id.strip()
    if not voice_id:
        raise ValueError("voice_id is required")

    selection = VoiceSelection(
        voice_id=voice_id,
        target_model=(target_model or "").strip() or None,
    )

    CURRENT_VOICE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CURRENT_VOICE_FILE.write_text(
        f"{json.dumps(asdict(selection), ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )

    LEGACY_VOICE_ID_FILE.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_VOICE_ID_FILE.write_text(selection.voice_id, encoding="utf-8")

    return selection


def clear_selected_voice(*, expected_voice_id: str | None = None) -> bool:
    """Delete persisted selection, optionally only when it matches the current voice."""
    current = load_selected_voice()
    if expected_voice_id and current and current.voice_id != expected_voice_id:
        return False

    if CURRENT_VOICE_FILE.exists():
        CURRENT_VOICE_FILE.unlink()
    if LEGACY_VOICE_ID_FILE.exists():
        LEGACY_VOICE_ID_FILE.unlink()
    return True


def resolve_current_tts_profile(default_model: str, default_voice: str) -> tuple[str, str]:
    """Return the active TTS model/voice pair, falling back to defaults."""
    selection = load_selected_voice()
    if not selection:
        return default_model, default_voice
    return selection.target_model or default_model, selection.voice_id
