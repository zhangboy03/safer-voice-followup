"""ECS voice-pipeline systemd entrypoint."""

from __future__ import annotations

import asyncio

from server.debug_full_pipeline import main as run_local_pipeline
from src.core.pilot_guardrails import validate_runtime_guardrails


def main() -> None:
    validate_runtime_guardrails("pinchi-agent")
    asyncio.run(run_local_pipeline())


if __name__ == "__main__":
    main()
