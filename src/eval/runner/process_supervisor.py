"""ProcessSupervisor — spawn and manage cold agent subprocesses."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path
from typing import Literal

from src.core.logging import get_logger
from src.eval.runner.agent_handle import AgentHandle

logger = get_logger(__name__)


class ProcessSupervisor:
    """Manages agent subprocess lifecycle for evaluation scenarios."""

    async def spawn_agent(
        self,
        *,
        mode: Literal["cold"] = "cold",
        env_overrides: dict[str, str],
        cwd: Path | None = None,
        stderr_log: Path | None = None,
        command: list[str] | None = None,
    ) -> AgentHandle:
        """Spawn a cold agent subprocess.

        Default command runs `uv run python -m server.main` (eval) with the
        given env overrides (LIVEKIT_ROOM_NAME, LIVEKIT_AGENT_IDENTITY). The
        live dispatcher passes an explicit venv-python command to match prod.
        """
        env = {**os.environ, **env_overrides}
        # Default to unreliable data channel for eval spawns — works around
        # Docker Desktop macOS SCTP/DTLS stalls that manifest as publish_data
        # timeouts ~20s into the session. Caller can override by setting
        # EVAL_DATA_CHANNEL_UNRELIABLE=0 in env_overrides.
        env.setdefault("EVAL_DATA_CHANNEL_UNRELIABLE", "1")
        room_name = env_overrides.get("LIVEKIT_ROOM_NAME", "unknown")

        argv = command or ["uv", "run", "python", "-m", "server.main"]

        stderr_fp = None
        if stderr_log:
            stderr_log.parent.mkdir(parents=True, exist_ok=True)
            stderr_fp = stderr_log.open("wb")

        # Merge stdout into the same log file so LiveKit-agents structured logs
        # (which go to stdout by default) are captured alongside stderr. Without
        # this, all pipeline / warmup / SOP transition signals were silently
        # dropped to DEVNULL, making VC failures impossible to diagnose.
        stdout_target = stderr_fp if stderr_fp is not None else asyncio.subprocess.DEVNULL
        stderr_target = stderr_fp if stderr_fp is not None else asyncio.subprocess.DEVNULL

        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdout=stdout_target,
                stderr=stderr_target,
                cwd=str(cwd) if cwd else None,
                env=env,
            )
        finally:
            if stderr_fp is not None:
                stderr_fp.close()
        logger.info("agent_spawned", pid=proc.pid, room=room_name, mode=mode)
        return AgentHandle(proc=proc, room_name=room_name, mode=mode)

    async def terminate(self, handle: AgentHandle, timeout: float = 10.0) -> None:
        """Terminate agent subprocess. SIGTERM first, SIGKILL after timeout."""
        proc = handle.proc
        if proc.returncode is not None:
            logger.info("agent_already_exited", pid=proc.pid, returncode=proc.returncode)
            return

        proc.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            logger.info("agent_terminated", pid=proc.pid, returncode=proc.returncode)
        except asyncio.TimeoutError:
            logger.warning("agent_kill", pid=proc.pid, msg="SIGTERM timeout, sending SIGKILL")
            proc.kill()
            await proc.wait()

    async def is_alive(self, handle: AgentHandle) -> bool:
        """Check if agent subprocess is still running."""
        return handle.proc.returncode is None
