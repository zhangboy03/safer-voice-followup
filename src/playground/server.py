"""Local FastAPI playground server. Binds 127.0.0.1 only; WS handshake gated by
Origin allowlist + per-process token before any brain/API work (spec §3.1e)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.core.config import settings
from src.playground.form_schema import field_catalog
from src.playground.security import make_token, origin_allowed, token_ok
from src.playground.session import PlaygroundSession, build_arm

_STATIC = Path(__file__).parent / "static"


def create_app(*, token: str, port: int, arch: str, model: str) -> FastAPI:
    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html = (_STATIC / "index.html").read_text(encoding="utf-8")
        catalog = [
            {"name": f.name, "group": f.group, "definition": f.definition, "required": f.required}
            for f in field_catalog()
        ]
        return (
            html.replace("__PG_TOKEN__", token)
            .replace("__PG_ARCH__", arch)
            .replace("__PG_MODEL__", model)
            .replace("__PG_CATALOG__", json.dumps(catalog, ensure_ascii=False))
        )

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        provided = websocket.query_params.get("token")
        if not origin_allowed(origin, port) or not token_ok(provided, token):
            await websocket.close(code=1008)  # policy violation, before any brain work
            return
        await websocket.accept()

        async def emit(ev: dict) -> None:
            await websocket.send_text(json.dumps(ev, ensure_ascii=False))

        state = {"arch": arch, "model": model}
        session = PlaygroundSession(emit=emit, arm=build_arm(arch=state["arch"], model=state["model"]))
        await session.start()
        try:
            while True:
                msg = json.loads(await websocket.receive_text())
                if msg.get("type") == "user_text":
                    await session.submit((msg.get("text") or "").strip())
                elif msg.get("type") == "control" and msg.get("action") == "end_call":
                    await session.end_call()
                elif msg.get("type") == "control" and msg.get("action") == "reset":
                    await session.aclose()
                    state["arch"] = msg.get("arch", state["arch"])
                    state["model"] = msg.get("model", state["model"])
                    session = PlaygroundSession(emit=emit, arm=build_arm(arch=state["arch"], model=state["model"]))
                    await session.start()
                    await emit({"type": "session_state", "arch": state["arch"], "model": state["model"]})
        except Exception:
            await session.aclose()

    return app


def main() -> None:
    parser = argparse.ArgumentParser(prog="src.playground.server", description="Local brain playground (text-only)")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (loopback only; default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8920, help="bind port (default 8920)")
    parser.add_argument("--arch", default="ours", choices=["ours", "spa", "solo"], help="initial brain arch")
    parser.add_argument("--model", default=settings.conversation_model, help="DashScope agent model")
    args = parser.parse_args()  # --help exits 0 here, before any bind

    token = make_token()
    app = create_app(token=token, port=args.port, arch=args.arch, model=args.model)
    print(f"\n  Brain playground → http://{args.host}:{args.port}/?token={token}\n")
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
