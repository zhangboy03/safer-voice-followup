const DEFAULT_VOICE_CLONE_URL = "/voice-clone";

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

const VOICE_CLONE_BASE_URL = trimTrailingSlash(
  import.meta.env.VITE_VOICE_CLONE_URL?.trim() || DEFAULT_VOICE_CLONE_URL,
);

export const VOICE_CLONE_UI_URL = `${VOICE_CLONE_BASE_URL}/`;

export function voiceCloneApiUrl(path: string): string {
  return `${VOICE_CLONE_BASE_URL}/${path.replace(/^\/+/, "")}`;
}
