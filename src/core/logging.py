"""Structured logging configuration.

Sets up structlog with:
- Dev mode: colorful console output
- Prod mode: JSON output for log aggregation
- Automatic capture of stdlib logging
- PII redactor processor that scrubs mobile phones / national IDs /
  device SNs / context-anchored Chinese names from every string value
  before rendering.  See ``src.core.pii`` for the regex implementation.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from src.core.pii import redact_text


def _redact_pii_processor(
    _logger: Any,
    _method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    """structlog processor that redacts PII in-place on every string value.

    Scrubs the ``event`` message itself plus any string-valued extra.
    Non-string values (ints, floats, dicts, lists, etc.) are walked one
    level deep — good enough for typical structlog call-sites like
    ``log.info("asr.final", text=user_utterance, latency_ms=42)``.

    The redaction runs regex-only (no jieba) via
    ``src.core.pii.redact_text`` — designed for the logging hot path.
    """
    for key, value in list(event_dict.items()):
        event_dict[key] = _redact_any(value)
    return event_dict


def _redact_any(value: Any) -> Any:
    """Walk one level into common containers; redact strings.

    Keeps a small call depth so the processor stays fast.  Deeply
    nested structures beyond depth 3 are left as-is — if a caller is
    logging arbitrarily deep payloads with PII they should scrub at
    the source.
    """
    return _redact_walk(value, depth=0)


def _redact_walk(value: Any, *, depth: int) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if depth >= 3:
        return value
    if isinstance(value, dict):
        return {k: _redact_walk(v, depth=depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        scrubbed = [_redact_walk(v, depth=depth + 1) for v in value]
        return type(value)(scrubbed) if isinstance(value, tuple) else scrubbed
    return value


def setup_logging(*, dev_mode: bool = False) -> None:
    """Initialize structured logging for the application.

    Replaces logging.basicConfig. Call once at startup.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        # PII redactor runs last in the shared chain so every event_dict
        # value — including ones added by upstream processors — is
        # scrubbed before the renderer serialises them.
        _redact_pii_processor,
    ]

    if dev_mode:
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer()
    else:
        renderer = structlog.processors.JSONRenderer(ensure_ascii=False)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound with module name."""
    result: structlog.stdlib.BoundLogger = structlog.get_logger(name)  # type: ignore[assignment]
    return result
