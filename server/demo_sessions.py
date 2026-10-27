"""Solo-only presentation API. Public voice credentials are room-scoped.

Provider credentials never cross this boundary. An opaque control capability
authorizes ending exactly one room, including after browser media disconnects.
"""
from __future__ import annotations

import asyncio
import os
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit

from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field

from server.live_dispatcher import AdmitResult, LiveDispatcher
from src.core.config import settings
from src.core.lk_tokens import mint_room_token

CONTRACT_VERSION = "solo.v1"


class StartSession(BaseModel):
    model_config = {"extra": "forbid"}
    participant_name: str = Field(default="demo-visitor", pattern=r"^[A-Za-z0-9_-]{1,64}$")


class EndSession(BaseModel):
    control_token: str = Field(min_length=16, max_length=128)


@dataclass
class SessionLease:
    room: str
    control_token: str
    expires_at: float
    ended: bool = False
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


_sessions: dict[str, SessionLease] = {}


def public_livekit_url() -> str:
    origin = os.getenv("DEMO_PUBLIC_ORIGIN", "http://localhost:5173").rstrip("/")
    url = urlsplit(origin)
    if url.scheme not in {"https", "http"} or not url.netloc or url.username or url.password or url.path:
        raise RuntimeError("Invalid DEMO_PUBLIC_ORIGIN")
    return ("wss" if url.scheme == "https" else "ws") + "://" + url.netloc


def register_demo_session_routes(app: FastAPI, dispatcher: Callable[[], LiveDispatcher | None]) -> None:
    router = APIRouter(prefix="/api/demo")

    @router.get("/config")
    async def config():
        return {
            "contract_version": CONTRACT_VERSION,
            "mode": "solo", "audio_transport": "livekit",
            "speech_language": "en", "ui_languages": ["zh", "en"],
            "record_source": "solo_async_extraction",
            "latency_scope": "server_speech_end_to_audio_queued",
            "max_duration_seconds": settings.live_max_call_seconds,
        }

    @router.post("/sessions", status_code=201)
    async def start(body: StartSession):
        now = time.monotonic()
        for key in [key for key, lease in _sessions.items() if lease.expires_at <= now]:
            del _sessions[key]
        if len(_sessions) >= 128:
            raise HTTPException(503, "Session capacity reached. Please try again shortly.")
        service = dispatcher()
        if service is None:
            raise HTTPException(503, "Voice service is starting. Please retry shortly.")
        session_id = uuid.uuid4().hex
        room = "pinchi-live-demo-" + session_id
        # Validate configuration and mint credentials before admitting a child.
        try:
            token = mint_room_token(body.participant_name, room)
            url = public_livekit_url()
        except RuntimeError:
            raise HTTPException(503, "Voice service configuration is unavailable.") from None
        result = await service.admit(room, brain="solo")
        if result is AdmitResult.BUSY:
            raise HTTPException(503, "All Solo lines are busy. End an existing call and retry.")
        control_token = secrets.token_urlsafe(32)
        _sessions[session_id] = SessionLease(room, control_token, now + settings.live_max_call_seconds + 60)
        return {
            "contract_version": CONTRACT_VERSION, "mode": "solo",
            "session_id": session_id, "room_name": room,
            "participant_name": body.participant_name,
            "livekit_url": url, "token": token, "control_token": control_token,
            "speech_language": "en", "max_duration_seconds": settings.live_max_call_seconds,
        }

    @router.post("/sessions/{session_id}/end")
    async def end(session_id: str, body: EndSession):
        lease = _sessions.get(session_id)
        if lease is None:
            raise HTTPException(404, "Unknown or expired session.")
        if not secrets.compare_digest(lease.control_token, body.control_token):
            raise HTTPException(403, "Invalid session control token.")
        async with lease.lock:
            if not lease.ended:
                service = dispatcher()
                if service is None:
                    raise HTTPException(503, "Voice service is unavailable.")
                await service.end_room(lease.room)
                lease.ended = True
        return {"contract_version": CONTRACT_VERSION, "session_id": session_id, "status": "ended"}

    app.include_router(router)
