"""PipelineRunner: manages the full lifecycle of a Pipeline.

Lifecycle:
  1. Optionally inject observer into every processor in the graph.
  2. Append _TerminalSink at the tail to detect EndFrame.
  3. setup() → start() all processors.
  4. Inject StartFrame DOWNSTREAM.
  5. Await asyncio.Event set by _TerminalSink on EndFrame arrival.
  6. stop() + cleanup() in finally block.

_TerminalSink sits after the pipeline's egress and sets an asyncio.Event
when it receives an EndFrame.  Data frames that reach the terminal are
dropped silently (there is no downstream consumer).
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.pipeline.frames import EndFrame, Frame, FrameDirection, StartFrame
from src.pipeline.pipeline import Pipeline
from src.pipeline.processor import FrameProcessor

if TYPE_CHECKING:
    from src.pipeline.events import EventBus


class _TerminalSink(FrameProcessor):
    """Pipeline tail: sets an Event on EndFrame, drops everything else."""

    def __init__(self, ended_event: asyncio.Event) -> None:
        super().__init__(name="terminal")
        self._ended = ended_event

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        pass  # Data frames at terminal: drop silently

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if isinstance(frame, EndFrame):
            self._ended.set()
        # Other system frames at terminal: drop silently


class PipelineRunner:
    """Manages the full lifecycle of a Pipeline from start to end.

    Parameters
    ----------
    pipeline:
        The Pipeline (or any FrameProcessor chain) to run.
    observer:
        Optional PipelineObserver injected into every processor before
        the run starts.  Must implement the PipelineObserver protocol.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        observer: object = None,
        *,
        bus: "EventBus | None" = None,
        observers: list | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._observer = observer
        self._bus = bus
        self._observers = observers or []
        self._ended: asyncio.Event = asyncio.Event()
        # Subscribe observers at construction so any bus.emit() between
        # PipelineRunner(...) and runner.run() (e.g. PipelineStarted,
        # AgentReady) reaches them. EventBus.emit is synchronous with no
        # replay buffer — late subscription silently drops earlier events.
        if self._bus is not None:
            for obs in self._observers:
                self._bus.subscribe(obs)

    async def run(self) -> None:
        """Run the pipeline to completion (until EndFrame reaches the terminal)."""
        if self._bus:
            self._inject_bus(self._pipeline, self._bus)

        if self._observer is not None:
            self._inject_observer(self._pipeline, self._observer)

        self._inject_topology(self._pipeline)

        terminal = _TerminalSink(self._ended)
        self._pipeline.link(terminal)

        await self._pipeline.setup()
        await terminal.setup()

        self._pipeline.start()
        terminal.start()

        self._pipeline.submit_frame(StartFrame(), FrameDirection.DOWNSTREAM)

        try:
            await self._ended.wait()
        finally:
            await self._pipeline.stop()
            await terminal.stop()
            await self._pipeline.cleanup()
            await terminal.cleanup()

    # ── Observer injection ────────────────────────────────────────────────────

    def _inject_observer(self, proc: FrameProcessor, observer: object) -> None:
        """Recursively inject observer into all processors in the graph."""
        if isinstance(proc, Pipeline):
            proc._observer = observer  # type: ignore[assignment]
            for p in [proc._ingress, *proc._processors, proc._egress]:
                self._inject_observer(p, observer)
        else:
            proc._observer = observer  # type: ignore[assignment]

    def _inject_bus(self, proc: FrameProcessor, bus: "EventBus") -> None:
        """Recursively inject EventBus into all processors in the graph."""
        if isinstance(proc, Pipeline):
            proc._bus = bus
            for p in [proc._ingress, *proc._processors, proc._egress]:
                self._inject_bus(p, bus)
        else:
            proc._bus = bus

    # ── Topology injection ────────────────────────────────────────────────────

    def _inject_topology(self, proc: FrameProcessor) -> None:
        """Assign _topology_index and _topology_registry to each processor in a Pipeline."""
        if isinstance(proc, Pipeline):
            registry: dict[str, int] = {}
            for idx, p in enumerate(proc._processors):
                registry[p.name] = idx
                p._topology_index = idx
                p._topology_registry = registry
                if isinstance(p, Pipeline):
                    self._inject_topology(p)
