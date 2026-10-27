"""TurnDecisionEngine — thin wrapper for SmartTurn backend.

Runs the primary backend (SmartTurnV3) and returns its prediction.
"""

from __future__ import annotations

import logging

from src.processors.turn_detector.backends import TurnContext, TurnDetectorBackend, TurnPrediction

logger = logging.getLogger(__name__)


class TurnDecisionEngine:
    """Runs the primary turn detection backend.

    Parameters
    ----------
    backends:
        List of TurnDetectorBackend instances. Index 0 is used.
    mode:
        Kept for backward compatibility. Only "smart_turn" is supported.
    """

    def __init__(
        self,
        backends: list[TurnDetectorBackend],
        mode: str = "smart_turn",
        uncertainty_threshold: float = 0.5,
    ) -> None:
        if not backends:
            raise ValueError("TurnDecisionEngine requires at least one backend")
        self._mode = mode
        self._uncertainty_threshold = uncertainty_threshold
        self._backends = backends

    def get_last_shadow_results(self) -> list[dict] | None:
        """No shadow mode — always returns None."""
        return None

    async def decide(self, context: TurnContext) -> TurnPrediction:
        """Return a TurnPrediction from the primary backend."""
        return await self._backends[0].predict(context)
