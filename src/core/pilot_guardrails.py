"""Runtime guardrails for the internal single-room pilot.

This module intentionally reads a few deployment boundary flags directly from
``os.environ`` instead of expanding ``Settings`` yet again.  The current project
is still a local/staging LiveKit pipeline; these helpers make that boundary
explicit and prevent obvious production misuse.
"""

from __future__ import annotations

import os
import re
from typing import Literal

from src.core.config import settings

AppEnv = Literal["development", "internal_pilot", "production"]
DebugPayloadLevel = Literal["raw", "redacted", "metrics_only"]

_VALID_APP_ENVS = {"development", "internal_pilot", "production"}
_VALID_DEBUG_PAYLOAD_LEVELS = {"raw", "redacted", "metrics_only"}
_TRUE_VALUES = {"1", "true", "yes", "on"}
_ROOM_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")
# Permit Chinese names for internal CSR labels while blocking whitespace/control chars.
_IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.@:\-\u4e00-\u9fff]{1,64}$")


class PilotGuardrailError(ValueError):
    """Raised when an internal-pilot boundary is violated."""


def _env_value(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env_value(name)
    if not raw:
        return default
    return raw.lower() in _TRUE_VALUES


def app_env() -> AppEnv:
    """Return the active app environment.

    Defaults to ``development`` to preserve existing local behavior unless a
    deploy explicitly opts into ``internal_pilot`` or ``production``.
    """
    raw = _env_value("APP_ENV", "development").lower()
    if raw not in _VALID_APP_ENVS:
        raise PilotGuardrailError(
            f"APP_ENV must be one of {sorted(_VALID_APP_ENVS)}, got {raw!r}.",
        )
    if raw == "development" and _env_bool("PILOT_MODE", default=False):
        return "internal_pilot"
    return raw  # type: ignore[return-value]


def is_development() -> bool:
    return app_env() == "development"


def is_internal_pilot() -> bool:
    return app_env() == "internal_pilot"


def is_production() -> bool:
    return app_env() == "production"


def debug_payload_level() -> DebugPayloadLevel:
    raw = _env_value("DEBUG_PAYLOAD_LEVEL", "raw").lower()
    if raw not in _VALID_DEBUG_PAYLOAD_LEVELS:
        raise PilotGuardrailError(
            f"DEBUG_PAYLOAD_LEVEL must be one of {sorted(_VALID_DEBUG_PAYLOAD_LEVELS)}, got {raw!r}.",
        )
    return raw  # type: ignore[return-value]


def live_transfer_enabled() -> bool:
    """Whether the runtime may attempt a real live transfer.

    For the current internal pilot this should remain false.  The flag is kept
    explicit so future production work cannot accidentally inherit a no-op SIP
    path as if it were a working handoff integration.
    """
    return _env_bool("ENABLE_LIVE_TRANSFER", default=False)


def uses_callback_followup() -> bool:
    """Return true when human handoff should become callback-required follow-up."""
    return not live_transfer_enabled()


def knowledge_qa_review_enabled() -> bool:
    """Per spec rollout gate for the knowledge_qa review state machine.

    Default flipped to True at PR-merge time per user decision (rollout-gate
    prereqs — HIGH callback consumer SLA, Type 3 ack-prefix choice, eval
    corpus — handled by ops-side review in this PR).
    Set ``KNOWLEDGE_QA_REVIEW_ENABLED=false`` to roll back.
    """
    return _env_bool("KNOWLEDGE_QA_REVIEW_ENABLED", default=True)


def kb_divert_nonclosing_enabled() -> bool:
    """Rollout gate for routing knowledge questions to the KB OUTSIDE the closing node.

    Spec 2026-05-26 §3 Change 4. Default OFF: the global Judge divert rule
    (共享规则 D) ships, but mid-SOP diverts stay suppressed until shadow/eval
    gate passes, because divert is binary-commit with no deterministic guard.
    Closing-node divert is unaffected by this flag. Set
    ``KB_DIVERT_NONCLOSING_ENABLED=true`` to roll forward.
    """
    return _env_bool("KB_DIVERT_NONCLOSING_ENABLED", default=False)


def resolve_livekit_credentials(
    api_key: str | None = None,
    api_secret: str | None = None,
) -> tuple[str, str]:
    """Resolve LiveKit credentials with dev-only fallback.

    The previous implementation silently fell back to ``devkey``/``secret``.
    That remains acceptable only in local development.
    """
    key = (api_key or settings.livekit_api_key or "").strip()
    secret = (api_secret or settings.livekit_api_secret or "").strip()

    if is_development():
        return key or "devkey", secret or "secret"

    if not key or not secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required outside development.")
    if key == "devkey" or secret == "secret":
        raise RuntimeError("LiveKit dev credentials are not allowed outside development.")
    return key, secret


def validate_token_request(room_name: str, participant_name: str) -> tuple[str, str]:
    """Validate a token request and return normalized room/identity values."""
    room = room_name.strip()
    identity = participant_name.strip()

    if not _ROOM_RE.fullmatch(room):
        raise PilotGuardrailError("Invalid LiveKit room name.")
    if not _IDENTITY_RE.fullmatch(identity):
        raise PilotGuardrailError("Invalid LiveKit participant identity.")

    if not is_development():
        configured_room = settings.livekit_room_name.strip() or "debug-full"
        allowed_prefix = _env_value("LIVEKIT_ALLOWED_ROOM_PREFIX")
        allowed = room == configured_room or (bool(allowed_prefix) and room.startswith(allowed_prefix))
        if not allowed:
            raise PilotGuardrailError(
                "Room is not allowed for this internal pilot. "
                "Use LIVEKIT_ROOM_NAME or LIVEKIT_ALLOWED_ROOM_PREFIX explicitly.",
            )

    return room, identity


def validate_runtime_guardrails(component: str = "app") -> None:
    """Fail fast on obvious environment misuse.

    ``internal_pilot`` remains intentionally permissive enough for staging debug.
    ``production`` rejects local/debug defaults because the current repository is
    not a real-patient production deployment target.
    """
    env = app_env()
    level = debug_payload_level()

    if live_transfer_enabled() and not is_development():
        raise RuntimeError(
            "ENABLE_LIVE_TRANSFER=true is deferred for this repository state. "
            "Use callback_required / needs_human_followup during internal pilot.",
        )

    errors: list[str] = []
    livekit_url = settings.livekit_url.strip()
    room_name = settings.livekit_room_name.strip()
    api_key = settings.livekit_api_key.strip()
    api_secret = settings.livekit_api_secret.strip()

    if env != "development":
        if not api_key:
            errors.append("LIVEKIT_API_KEY is required outside development.")
        elif api_key == "devkey":
            errors.append("LIVEKIT_API_KEY=devkey is development-only.")
        if not api_secret:
            errors.append("LIVEKIT_API_SECRET is required outside development.")
        elif api_secret == "secret":
            errors.append("LIVEKIT_API_SECRET=secret is development-only.")

    if env != "production":
        if errors:
            joined = " ".join(errors)
            raise RuntimeError(f"{component} refused to start with unsafe {env} settings. {joined}")
        return

    if settings.dev_mode:
        errors.append("DEV_MODE must be false in production.")
    if settings.allow_stub_fallback:
        errors.append("ALLOW_STUB_FALLBACK must be false in production.")
    if room_name == "debug-full":
        errors.append("LIVEKIT_ROOM_NAME=debug-full is internal-staging only.")
    if livekit_url.startswith(("ws://localhost", "ws://127.0.0.1")):
        errors.append("LIVEKIT_URL points at localhost; production requires a real WSS endpoint.")
    if level == "raw":
        errors.append("DEBUG_PAYLOAD_LEVEL=raw is not allowed in production.")

    if errors:
        joined = " ".join(errors)
        raise RuntimeError(f"{component} refused to start with unsafe production settings. {joined}")
