"""LiveDispatcher — admission control + lifecycle for live agent subprocesses.

One isolated `server.main` subprocess per LiveKit room, capped at MAX_LIVE_AGENTS.
Per-room file isolation via injected env. Background reaper enforces a per-child
RSS budget and a max-call-duration backstop (no-swap OOM safety net).
"""

from __future__ import annotations

import asyncio
import enum
import json
import time
from pathlib import Path

from src.core.logging import get_logger

logger = get_logger(__name__)


class AdmitResult(enum.Enum):
    SPAWNED = "spawned"  # new subprocess started for this room
    ALREADY = "already"  # live subprocess already serving this room (idempotent)
    BUSY = "busy"  # at cap → caller should return HTTP 503


class _LiveHandle:
    __slots__ = ("agent", "spawn_ts", "brain")

    def __init__(self, agent, spawn_ts: float, brain: str | None = None) -> None:
        self.agent = agent  # AgentHandle (has .proc, .room_name)
        self.spawn_ts = spawn_ts
        self.brain = brain  # AGENT_BRAIN frozen at spawn; None = child falls back to .env


class LiveDispatcher:
    def __init__(
        self,
        *,
        supervisor,
        max_agents: int,
        python_exe: str,
        cwd: str,
        artifact_dir: str,
        per_child_rss_max_mb: int,
        max_call_seconds: int,
        participant_wait_s: int,
        now_fn=time.monotonic,
    ) -> None:
        self._sup = supervisor
        self._max = max_agents
        self._python = python_exe
        self._cwd = cwd
        self._artifact_dir = Path(artifact_dir)
        self._rss_max_mb = per_child_rss_max_mb
        self._max_call_s = max_call_seconds
        self._wait_s = participant_wait_s
        self._now = now_fn
        self._handles: dict[str, _LiveHandle] = {}
        self._lock = asyncio.Lock()

    # ── admission ──
    async def admit(self, room_name: str, *, brain: str | None = None) -> AdmitResult:
        async with self._lock:
            self._reap_locked()
            existing = self._handles.get(room_name)
            if existing is not None and existing.agent.proc.returncode is None:
                if brain is not None and brain != existing.brain:
                    # brain is frozen at spawn — a same-room rejoin can't switch it
                    logger.warning(
                        "live.admit_brain_mismatch",
                        room=room_name,
                        requested=brain,
                        serving=existing.brain or "default",
                    )
                return AdmitResult.ALREADY
            if len(self._handles) >= self._max:
                logger.info("live.admit_busy", room=room_name, active=len(self._handles))
                return AdmitResult.BUSY

            self._artifact_dir.mkdir(parents=True, exist_ok=True)
            env_overrides = {
                "LIVEKIT_ROOM_NAME": room_name,
                "POLARIS_MEMORY_DB_PATH": f"data/memory/{room_name}.db",
                "CALLBACK_QUEUE_PATH": f"data/callbacks/{room_name}.jsonl",
                "LIVE_PARTICIPANT_WAIT_S": str(self._wait_s),
                "EVAL_DATA_CHANNEL_UNRELIABLE": "0",
            }
            if brain is not None:
                env_overrides["AGENT_BRAIN"] = brain
                if brain == "solo":
                    env_overrides["SOLO_LANGUAGE"] = "en"
            agent = await self._sup.spawn_agent(
                env_overrides=env_overrides,
                cwd=Path(self._cwd),
                stderr_log=self._artifact_dir / f"{room_name}.log",
                command=[self._python, "-m", "server.main"],
            )
            spawn_ts = self._now()
            self._handles[room_name] = _LiveHandle(agent, spawn_ts, brain=brain)
            self._write_sidecar(
                room_name, spawn_ts=spawn_ts, concurrent_at_spawn=len(self._handles), brain=brain or "default"
            )
            logger.info(
                "live.spawned", room=room_name, pid=agent.proc.pid, active=len(self._handles), brain=brain or "default"
            )
            return AdmitResult.SPAWNED

    def _reap_locked(self) -> None:
        dead = [r for r, h in self._handles.items() if h.agent.proc.returncode is not None]
        for room in dead:
            self._write_sidecar(room, exit_ts=self._now())
            logger.info("live.reaped", room=room, rc=self._handles[room].agent.proc.returncode)
            del self._handles[room]

    def active_count(self) -> int:
        return len(self._handles)

    async def end_room(self, room_name: str) -> bool:
        """Terminate only this room; keep its slot occupied until it exits."""
        async with self._lock:
            self._reap_locked()
            handle = self._handles.get(room_name)
        if handle is None:
            return False
        await self._sup.terminate(handle.agent, timeout=3.0)
        async with self._lock:
            self._reap_locked()
        return True

    # ── sidecar for pollution analysis (spawn_ts + exit_ts → time-overlap) ──
    def _write_sidecar(self, room_name: str, **fields) -> None:
        path = self._artifact_dir / f"{room_name}.meta.jsonl"
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"room": room_name, **fields}, ensure_ascii=False) + "\n")

    # ── reaper / watchdog ──
    def _rss_mb(self, pid: int) -> float:
        """Resident set size in MiB from /proc; 0.0 if unreadable."""
        try:
            with open(f"/proc/{pid}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except OSError:
            return 0.0
        return 0.0

    async def tick_once(self) -> None:
        """One reaper pass: reap dead, kill over-budget RSS, kill over-duration."""
        async with self._lock:
            self._reap_locked()
            now = self._now()
            to_kill = []
            for room, h in self._handles.items():
                if h.agent.proc.returncode is not None:
                    continue
                rss = self._rss_mb(h.agent.proc.pid)
                if rss > self._rss_max_mb:
                    logger.warning("live.kill_rss", room=room, rss_mb=round(rss, 1), budget=self._rss_max_mb)
                    to_kill.append(self._handles[room].agent)
                elif (now - h.spawn_ts) > self._max_call_s:
                    logger.warning("live.kill_duration", room=room, age_s=round(now - h.spawn_ts, 1))
                    to_kill.append(self._handles[room].agent)
        # terminate OUTSIDE the lock so a SIGTERM-ignoring child can't stall admissions
        for agent in to_kill:
            await self._sup.terminate(agent)
        if to_kill:
            async with self._lock:
                self._reap_locked()

    async def reaper_loop(self, interval_s: float) -> None:
        while True:
            try:
                await self.tick_once()
            except Exception:  # never let the reaper die silently
                logger.exception("live.reaper_tick_failed")
            await asyncio.sleep(interval_s)

    async def shutdown(self) -> None:
        async with self._lock:
            agents = [h.agent for h in self._handles.values() if h.agent.proc.returncode is None]
            self._handles.clear()
        for agent in agents:
            await self._sup.terminate(agent)
