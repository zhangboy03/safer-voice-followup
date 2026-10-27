"""FrameProcessor: core execution engine of the voice pipeline.

Architecture (following Pipecat's proven dual-task model):

  input_queue (PriorityQueue: SystemFrame=0, DataFrame=1)
      │
      ├── _input_task (Task 1: dispatcher)
      │     ├── SystemFrame → handle_system_frame() immediately
      │     └── DataFrame  → put into _process_queue
      │
      └── _process_queue (asyncio.Queue: FIFO)
            └── _process_task (Task 2: worker)
                  └── process_frame() one at a time, sequentially

Key insight: SystemFrames and DataFrames run in SEPARATE tasks.
SystemFrames never cancel in-flight DataFrame processing unless
explicitly requested (InterruptionFrame / CancelFrame / EndFrame).
"""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass, field

from src.pipeline.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    FrameDirection,
    InterruptionFrame,
    InterruptionScope,
    SystemFrame,
    UninterruptibleMixin,
)


@dataclass(order=True)
class _QueueItem:
    priority: int  # 0=SystemFrame, 1=DataFrame/ControlFrame
    seq: int  # monotonic for FIFO within same priority
    frame: Frame = field(compare=False)
    direction: FrameDirection = field(compare=False)


class FrameProcessor:
    """Base class for all pipeline processors."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._observer = None  # PipelineObserver | None, injected by runner
        self._bus = None  # EventBus | None, injected by runner
        self.__prev = None  # upstream neighbor
        self._next = None  # downstream neighbor
        self._loop: asyncio.AbstractEventLoop | None = None  # set in start()

        # Input queue: all frames enter here (SystemFrame priority 0, others 1)
        self._input_queue: asyncio.PriorityQueue[_QueueItem] = asyncio.PriorityQueue()
        self._seq = 0

        # Process queue: only DataFrames/ControlFrames, FIFO
        self._process_queue: asyncio.Queue[tuple[Frame, FrameDirection]] = asyncio.Queue()

        # Current in-flight process_frame task (for cancel_inflight)
        self._current_task: asyncio.Task | None = None

        self._input_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None
        self._running = False
        self._draining = False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def _prev(self):
        return self.__prev

    @_prev.setter
    def _prev(self, proc):
        self.__prev = proc

    def _emit(self, method: str, *args) -> None:
        """Call observer method if observer is set. Never raises."""
        if self._observer is not None:
            fn = getattr(self._observer, method, None)
            if fn:
                try:
                    fn(*args)
                except Exception:
                    pass

    # ── Subclass overrides ────────────────────────────────────────────────────

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Override to handle DataFrames. Must call push_frame to forward."""
        await self.push_frame(frame, direction)

    async def handle_system_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """SystemFrame handler. Runs in the input task, parallel to process_frame."""
        if isinstance(frame, EndFrame):
            await self.cancel_inflight()
            self._draining = True
            self._flush_process_queue()
            await self.push_frame(frame, direction)
            self._running = False
            return

        if isinstance(frame, InterruptionFrame):
            if self.interruption_applies(frame):
                await self.cancel_inflight()
                await self.on_flush(frame)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, CancelFrame):
            await self.cancel_inflight()
            await self.on_flush(frame)
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def interruption_applies(self, frame: InterruptionFrame) -> bool:
        match frame.scope:
            case InterruptionScope.ALL_DOWNSTREAM:
                return True
            case InterruptionScope.TTS_ONLY:
                return self._is_tts_related()
            case InterruptionScope.AFTER_PROCESSOR:
                target_name = frame.after_processor
                if not target_name or not hasattr(self, "_topology_registry"):
                    return False
                target_idx = self._topology_registry.get(target_name, -1)
                if target_idx < 0:
                    return False
                return self._topology_index > target_idx
            case _:
                return True

    async def cancel_inflight(self) -> None:
        if self._process_task and not self._process_task.done():
            self._process_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._process_task
            self._current_task = None
            # Restart process task so it can handle the next frame
            if self._running:
                self._process_task = asyncio.create_task(self._process_task_handler())

    async def on_flush(self, frame: Frame) -> None:
        self._flush_process_queue()

    async def setup(self) -> None:
        pass

    async def cleanup(self) -> None:
        pass

    def _is_tts_related(self) -> bool:
        return False

    # ── Frame passing ─────────────────────────────────────────────────────────

    async def push_frame(self, frame: Frame, direction: FrameDirection) -> None:
        target = self._next if direction == FrameDirection.DOWNSTREAM else self._prev
        if target is not None:
            target.submit_frame(frame, direction)

    # ── Thread-safe entry ─────────────────────────────────────────────────────

    def submit_frame(self, frame: Frame, direction: FrameDirection) -> None:
        if self._loop is None:
            raise RuntimeError(f"{self.name}: not started")
        try:
            running_loop = asyncio.get_running_loop()
            is_same_loop = running_loop is self._loop
        except RuntimeError:
            is_same_loop = False
        if is_same_loop:
            self._enqueue(frame, direction)
        else:
            self._loop.call_soon_threadsafe(self._enqueue, frame, direction)

    def _enqueue(self, frame: Frame, direction: FrameDirection) -> None:
        if self._draining and not isinstance(frame, (SystemFrame, UninterruptibleMixin)):
            return
        priority = 0 if isinstance(frame, SystemFrame) else 1
        self._seq += 1
        self._input_queue.put_nowait(_QueueItem(priority, self._seq, frame, direction))

    # ── Dual-task main loop (Pipecat model) ────────────────────────────────────

    async def _input_task_handler(self) -> None:
        """Task 1: Dispatch frames from input queue.

        SystemFrames are processed immediately in this task.
        DataFrames are forwarded to _process_queue for Task 2.
        """
        try:
            while self._running:
                item = await self._input_queue.get()

                if isinstance(item.frame, SystemFrame):
                    await self.handle_system_frame(item.frame, item.direction)
                else:
                    await self._process_queue.put((item.frame, item.direction))
        except asyncio.CancelledError:
            pass

    async def _process_task_handler(self) -> None:
        """Task 2: Process DataFrames sequentially.

        Each frame is processed one at a time. The current processing
        can be cancelled by cancel_inflight() (called from handle_system_frame
        in Task 1 when InterruptionFrame/EndFrame arrives).
        """
        try:
            while self._running:
                frame, direction = await self._process_queue.get()

                self._current_task = asyncio.current_task()
                try:
                    await self.process_frame(frame, direction)
                except asyncio.CancelledError:
                    # Cancelled by cancel_inflight — restart the task
                    # so we can continue processing the next frame
                    return
                except Exception as exc:
                    if self._observer:
                        self._observer.emit_error(
                            self.name,
                            type(exc).__name__,
                            str(exc),
                            getattr(frame, "turn_id", ""),
                        )
                    if self._bus:
                        from src.pipeline.events import ErrorEvent

                        self._bus.emit(
                            ErrorEvent(
                                processor=self.name,
                                error_type=type(exc).__name__,
                                message=str(exc),
                                turn_id=getattr(frame, "turn_id", ""),
                            )
                        )
                    await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)
                    self._running = False
                    return
                finally:
                    self._current_task = None
        except asyncio.CancelledError:
            pass

    def _flush_process_queue(self) -> None:
        """Discard all queued DataFrames in the process queue."""
        while not self._process_queue.empty():
            try:
                self._process_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # Keep old name for backward compatibility with tests
    def _flush_mailbox(self) -> None:
        # Flush input queue (preserve UninterruptibleMixin)
        preserved: list[_QueueItem] = []
        while not self._input_queue.empty():
            try:
                item = self._input_queue.get_nowait()
                if isinstance(item.frame, UninterruptibleMixin):
                    preserved.append(item)
            except asyncio.QueueEmpty:
                break
        for item in preserved:
            self._input_queue.put_nowait(item)
        # Also flush process queue
        self._flush_process_queue()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def link(self, next_proc: "FrameProcessor") -> "FrameProcessor":
        self._next = next_proc
        next_proc._prev = self
        return next_proc

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._running = True
        self._input_task = asyncio.create_task(self._input_task_handler())
        self._process_task = asyncio.create_task(self._process_task_handler())

    async def stop(self) -> None:
        self._running = False
        await self.cancel_inflight()
        for task in (self._input_task, self._process_task):
            if task:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
