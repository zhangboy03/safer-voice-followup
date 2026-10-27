"""Pipeline: composite FrameProcessor wrapping an ordered processor chain.

Frame flow:
  DOWNSTREAM: external → Pipeline.submit_frame(DOWN) → ingress → head → ... → tail → egress → external._next
  UPSTREAM:   external → Pipeline.submit_frame(UP)   → egress  → tail → ... → head → ingress → external._prev

The _PipelineIngress and _PipelineEgress bridge processors handle direction
routing at the Pipeline boundaries without exposing internal processor linking.
"""

from __future__ import annotations

from src.pipeline.frames import Frame, FrameDirection
from src.pipeline.processor import FrameProcessor


class _PipelineIngress(FrameProcessor):
    """Entry bridge: routes DOWNSTREAM into the chain, UPSTREAM to external prev."""

    def __init__(self, pipeline: "Pipeline") -> None:
        super().__init__(f"{pipeline.name}:ingress")
        self._pipeline = pipeline

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)


class _PipelineEgress(FrameProcessor):
    """Exit bridge: routes DOWNSTREAM to external next, UPSTREAM back into chain."""

    def __init__(self, pipeline: "Pipeline") -> None:
        super().__init__(f"{pipeline.name}:egress")
        self._pipeline = pipeline

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await self.push_frame(frame, direction)


class Pipeline(FrameProcessor):
    """Composite FrameProcessor: wraps an ordered list of processors as one unit.

    Internal chain:
      ingress → p0 → p1 → ... → pN → egress

    Upstream chain (reverse):
      egress → pN → ... → p0 → ingress

    Pipeline._next / Pipeline._prev are the external neighbors.
    The ingress bridges Pipeline._prev to the internal chain.
    The egress bridges the internal chain to Pipeline._next.
    """

    def __init__(self, name: str, processors: list[FrameProcessor]) -> None:
        super().__init__(name)
        self._processors = list(processors)
        self._ingress = _PipelineIngress(self)
        self._egress = _PipelineEgress(self)
        self._link_internals()

    def _link_internals(self) -> None:
        """Wire ingress → p0 → p1 → ... → pN → egress (both DOWNSTREAM and UPSTREAM)."""
        chain: list[FrameProcessor] = [self._ingress, *self._processors, self._egress]
        for i in range(len(chain) - 1):
            chain[i]._next = chain[i + 1]
            chain[i + 1]._prev = chain[i]

    # ── External frame entry ───────────────────────────────────────────────────

    def submit_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Route frames to the correct boundary processor."""
        if direction == FrameDirection.DOWNSTREAM:
            self._ingress.submit_frame(frame, direction)
        else:
            self._egress.submit_frame(frame, direction)

    # ── External neighbor wiring ───────────────────────────────────────────────

    def link(self, next_proc: FrameProcessor) -> FrameProcessor:
        """Set the downstream external neighbor."""
        self._next = next_proc
        next_proc._prev = self
        # Egress needs to know the external next directly
        self._egress._next = next_proc
        return next_proc

    @property
    def _prev(self):  # type: ignore[override]
        return self.__prev  # type: ignore[attr-defined]

    @_prev.setter
    def _prev(self, proc: FrameProcessor | None) -> None:  # type: ignore[override]
        self.__prev = proc  # type: ignore[attr-defined]
        # Keep ingress in sync so UPSTREAM frames exit correctly
        self._ingress._prev = proc

    # ── Lifecycle delegation ───────────────────────────────────────────────────

    def _all_processors(self) -> list[FrameProcessor]:
        return [self._ingress, *self._processors, self._egress]

    async def setup(self) -> None:
        for p in self._all_processors():
            await p.setup()

    def start(self) -> None:
        for p in self._all_processors():
            p.start()

    async def stop(self) -> None:
        for p in self._all_processors():
            await p.stop()

    async def cleanup(self) -> None:
        for p in self._all_processors():
            await p.cleanup()

    # ── Passthrough process_frame (Pipeline itself is not in the data path) ───

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        # Pipeline does not sit in the mailbox path — submit_frame routes directly
        # to ingress or egress. This is here to satisfy the abstract interface.
        await self.push_frame(frame, direction)
