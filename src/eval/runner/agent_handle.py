from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal


@dataclass
class AgentHandle:
    """Handle to a running agent subprocess."""

    proc: asyncio.subprocess.Process
    room_name: str
    mode: Literal["cold"] = "cold"
