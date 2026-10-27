from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class PatientAction:
    kind: Literal["speak", "backchannel", "silence", "hangup"]
    mode: Literal["serial", "overlap", "replace"] = "serial"
    text: str | None = None
    silence_ms: int | None = None
    backchannel_label: str | None = None
    voice: str | None = None
    # overlap/replace: insertion point relative to current turn audio start (ms)
    # serial: ignored
    relative_ms: int | None = None
