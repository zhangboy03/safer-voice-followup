"""Single-shot asyncio timer for silent-rescue (ADR 0006 §4.1).

LLMProcessor 在 playout 结束时启 timer；3s（或 silent_rescue_timeout_ms）
后调用 callback；ASR partial 到达时 cancel。重复 start 是 no-op；cancel
对从未启动的 timer 也安全。Callback 可以是同步或 async。
"""

import asyncio
from typing import Awaitable, Callable, Union


class SilentRescueTimer:
    def __init__(
        self,
        timeout_ms: int,
        callback: Union[Callable[[], None], Callable[[], Awaitable[None]]],
    ) -> None:
        self._timeout_s = timeout_ms / 1000.0
        self._callback = callback
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return  # already running, no-op
        self._task = asyncio.create_task(self._run())

    def cancel(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def _run(self) -> None:
        try:
            await asyncio.sleep(self._timeout_s)
            result = self._callback()
            if asyncio.iscoroutine(result):
                await result
        except asyncio.CancelledError:
            pass  # cancelled before fire = expected path
