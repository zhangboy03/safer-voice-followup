"""WS handshake trust boundary (spec §3.1e): defend against cross-site WebSocket
hijacking. A loopback bind alone is insufficient because WebSocket is exempt from
the same-origin policy."""

from __future__ import annotations

import secrets


def origin_allowed(origin: str | None, port: int) -> bool:
    if not origin:
        return False
    return origin in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}


def make_token() -> str:
    return secrets.token_urlsafe(32)


def token_ok(provided: str | None, expected: str) -> bool:
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)
