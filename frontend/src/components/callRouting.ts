import type { Lang } from "../i18n/strings.js";

/**
 * `?room=` wins; otherwise VITE_ROOM_NAME, made unique per page load when
 * VITE_UNIQUE_ROOM_PER_LOAD=true so concurrent tabs never share an agent.
 */
export function resolveRoomName(): string {
  const fromQuery = new URLSearchParams(window.location.search).get("room");
  if (fromQuery) return fromQuery;
  const base = import.meta.env.VITE_ROOM_NAME || "debug-full";
  if (import.meta.env.VITE_UNIQUE_ROOM_PER_LOAD !== "true") return base;
  const csr = (window.localStorage.getItem("csrId") ?? "user").replace(
    /[^A-Za-z0-9_-]/g,
    "",
  );
  const rand = Math.random().toString(36).slice(2, 7);
  return `${base}-${csr}-${Date.now().toString(36)}-${rand}`;
}

export function tokenRequestBody(
  roomName: string,
  participantName: string,
  lang?: Lang,
): string {
  void lang;
  return JSON.stringify({
    room_name: roomName,
    participant_name: participantName,
    brain: "solo",
  });
}
