"""RoomManager — create and delete LiveKit rooms for evaluation scenarios."""

from __future__ import annotations

from src.core.logging import get_logger

logger = get_logger(__name__)


class RoomManager:
    """Manages LiveKit room lifecycle for evaluation."""

    def __init__(self, livekit_api=None) -> None:
        self._api = livekit_api

    @staticmethod
    def room_name(item_id: str, scenario_id: str, repeat: int, ts: str) -> str:
        """Generate deterministic room name."""
        return f"eval-ch4-{item_id}-{scenario_id}-r{repeat:03d}-{ts}"

    async def new_room(self, name: str) -> None:
        """Create a new LiveKit room."""
        if self._api:
            try:
                from livekit.api import CreateRoomRequest

                await self._api.room.create_room(CreateRoomRequest(name=name))
                logger.info("room_created", room=name)
            except Exception as e:
                logger.warning("room_create_failed", room=name, error=str(e))
        else:
            logger.debug("room_create_skipped", room=name, msg="no API client")

    async def delete_room(self, name: str) -> None:
        """Delete a LiveKit room (best-effort)."""
        if self._api:
            try:
                from livekit.api import DeleteRoomRequest

                await self._api.room.delete_room(DeleteRoomRequest(room=name))
                logger.info("room_deleted", room=name)
            except Exception as e:
                logger.warning("room_delete_failed", room=name, error=str(e))
