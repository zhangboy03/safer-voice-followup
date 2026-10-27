from __future__ import annotations

from livekit.api import AccessToken, VideoGrants

from src.core.config import settings
from src.core.pilot_guardrails import resolve_livekit_credentials


def mint_room_token(
    identity: str,
    room_name: str,
    *,
    api_key: str | None = None,
    api_secret: str | None = None,
    grants: VideoGrants | None = None,
) -> str:
    key, secret = resolve_livekit_credentials(
        api_key=api_key or settings.livekit_api_key,
        api_secret=api_secret or settings.livekit_api_secret,
    )
    token = AccessToken(api_key=key, api_secret=secret)
    token.with_identity(identity)
    token.with_grants(grants or VideoGrants(room_join=True, room=room_name))
    return token.to_jwt()
