"""Single-source-of-truth loader for runtime behavior flags.

Production (ECS) and every eval harness apply the SAME flag values from
``config/runtime_flags.env`` so the evaluation measures the shipped
configuration and production stays reproducible from the repo. See that file
for the rationale.

Eval harnesses call :func:`apply_runtime_flag_profile` once at startup and write
the returned ``effective`` dict into the run's ``metadata.json``. Production
sources the same file via systemd ``EnvironmentFile=`` (so the helper is not
needed there — pydantic reads the env vars directly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.core.config import Settings
from src.core.config import settings as _settings

# repo_root/config/runtime_flags.env  (this file lives at repo_root/src/core/)
RUNTIME_FLAGS_PATH = Path(__file__).resolve().parents[2] / "config" / "runtime_flags.env"


def parse_flag_file(path: Path) -> dict[str, str]:
    """Parse a simple KEY=value env file (ignoring blanks and ``#`` comments)."""
    out: dict[str, str] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip()
    return out


def _coerce(raw: str, annotation: Any) -> Any:
    """Coerce a raw string to the Settings field's declared scalar type."""
    if annotation is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    return raw


def apply_runtime_flag_profile(settings: Settings | None = None, *, path: Path | None = None) -> dict[str, Any]:
    """Apply the flag profile onto ``settings`` in place.

    Only keys that map to a real ``Settings`` field are applied; the env-var key
    is matched to the field name case-insensitively (pydantic's own convention).
    Values are coerced to the field's declared type.

    Returns the ``{field_name: value}`` dict actually applied, for recording
    into run metadata so an artifact alone proves which configuration it ran.
    """
    target = settings if settings is not None else _settings
    raw = parse_flag_file(path or RUNTIME_FLAGS_PATH)
    fields = Settings.model_fields
    effective: dict[str, Any] = {}
    for env_key, raw_val in raw.items():
        field = env_key.strip().lower()
        if field not in fields:
            continue
        value = _coerce(raw_val, fields[field].annotation)
        setattr(target, field, value)
        effective[field] = value
    return effective
