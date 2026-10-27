"""LiveKit token API for the ECS internal pilot."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from server.demo_sessions import _sessions, register_demo_session_routes
from server.feedback_store import FeedbackCreate, FeedbackRecord, create_feedback, get_feedback, list_feedback
from server.live_dispatcher import AdmitResult, LiveDispatcher
from server.notes_api import register_notes_routes
from src.core.config import settings
from src.core.lk_tokens import mint_room_token
from src.core.pilot_guardrails import (
    PilotGuardrailError,
    validate_runtime_guardrails,
    validate_token_request,
)
from src.eval.runner.process_supervisor import ProcessSupervisor


class TokenRequest(BaseModel):
    room_name: str
    participant_name: str = "user"
    # Compatibility for older pages. No public API may select another mode.
    brain: Literal["solo"] | None = None


class TokenResponse(BaseModel):
    token: str


class FeedbackListResponse(BaseModel):
    items: list[FeedbackRecord]


_dispatcher: LiveDispatcher | None = None
_reaper_task = None


@asynccontextmanager
async def _lifespan(app):
    global _dispatcher, _reaper_task
    validate_runtime_guardrails("pinchi-token")
    _dispatcher = LiveDispatcher(
        supervisor=ProcessSupervisor(),
        max_agents=settings.max_live_agents,
        python_exe=sys.executable,  # token server runs under .venv/bin/python → correct interpreter
        cwd=".",
        artifact_dir=settings.live_artifact_dir,
        per_child_rss_max_mb=settings.live_per_child_rss_max_mb,
        max_call_seconds=settings.live_max_call_seconds,
        participant_wait_s=60,
    )
    import asyncio

    _reaper_task = asyncio.create_task(_dispatcher.reaper_loop(settings.live_reaper_interval_s))
    try:
        yield
    finally:
        _sessions.clear()
        if _reaper_task:
            _reaper_task.cancel()
        if _dispatcher:
            await _dispatcher.shutdown()


app = FastAPI(title="LiveKit Token Server", lifespan=_lifespan)
register_demo_session_routes(app, lambda: _dispatcher)


@app.post("/api/token", response_model=TokenResponse)
async def create_token(req: TokenRequest) -> TokenResponse:
    try:
        room_name, participant_name = validate_token_request(req.room_name, req.participant_name)
    except PilotGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if _dispatcher is not None:
        result = await _dispatcher.admit(room_name, brain="solo")
        if result is AdmitResult.BUSY:
            raise HTTPException(status_code=503, detail="All lines busy — please try again shortly.")

    try:
        token = mint_room_token(participant_name, room_name)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return TokenResponse(token=token)


@app.post("/api/feedback", response_model=FeedbackRecord)
async def submit_feedback(req: FeedbackCreate) -> FeedbackRecord:
    return create_feedback(req)


@app.get("/api/feedback", response_model=FeedbackListResponse)
async def feedback_items(limit: int = 100) -> FeedbackListResponse:
    return FeedbackListResponse(items=list_feedback(limit=limit))


@app.get("/api/feedback/{feedback_id}", response_model=FeedbackRecord)
async def feedback_detail(feedback_id: str) -> FeedbackRecord:
    record = get_feedback(feedback_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return record


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


register_notes_routes(app)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.token_server:app", host="127.0.0.1", port=8000)
