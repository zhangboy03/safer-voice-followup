export type NoteCategory =
  | "caller_identity"
  | "resolved_topic"
  | "future_node_hint"
  | "correction_hint"
  | "salient_fact";

export type NoteItem = {
  key: string;
  text: string;
  category: NoteCategory;
  source_turn_ids: string[];
  confidence: number;
  expires_after_turn_index: number | null;
  expired: boolean;
  edit_history: Array<{
    edit_id: string;
    after_text: string;
    edited_at: string;
  }>;
  latest_event_id: string;
};

export type NotesResponse = {
  call_id: string;
  events: unknown[];
  current_items: NoteItem[];
  last_event_id: string | null;
};

export async function fetchNotes(callId: string): Promise<NotesResponse> {
  const response = await fetch(`/api/notes/${encodeURIComponent(callId)}`);
  if (!response.ok) {
    throw new Error(`fetchNotes failed (${response.status})`);
  }
  return (await response.json()) as NotesResponse;
}

export async function listCallIdsByRoom(roomName: string): Promise<string[]> {
  const response = await fetch(
    `/api/notes/by-room/${encodeURIComponent(roomName)}`,
  );
  if (!response.ok) {
    throw new Error(`listCallIdsByRoom failed (${response.status})`);
  }
  const body = (await response.json()) as { call_ids?: string[] };
  return body.call_ids ?? [];
}

export function readCsrId(): string {
  return window.localStorage.getItem("csrId") ?? "user";
}

export async function patchNote(args: {
  callId: string;
  itemKey: string;
  beforeText: string;
  afterText: string;
  basedOnEventId: string;
  csrId: string;
}): Promise<{
  ok: boolean;
  conflict?: { current_text: string; current_event_id: string };
}> {
  const resp = await fetch(
    `/api/notes/${encodeURIComponent(args.callId)}/items/${encodeURIComponent(args.itemKey)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        before_text: args.beforeText,
        after_text: args.afterText,
        based_on_event_id: args.basedOnEventId,
        csr_id: args.csrId,
      }),
    },
  );
  if (resp.status === 409) {
    const body = await resp.json();
    return {
      ok: false,
      conflict: {
        current_text: body.detail.current_text,
        current_event_id: body.detail.current_event_id,
      },
    };
  }
  if (!resp.ok) {
    throw new Error(`patch_failed_${resp.status}`);
  }
  return { ok: true };
}
