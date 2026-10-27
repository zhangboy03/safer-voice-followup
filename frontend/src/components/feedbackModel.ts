export type FeedbackSource = "csr_simple" | "debug";

export interface TraceEvent {
  topic: string;
  payload: Record<string, unknown>;
}

export interface FeedbackTurn {
  turn_id?: string;
  user_text?: string;
  assistant_text?: string;
  node_id?: string;
}

interface FeedbackTurnOption extends FeedbackTurn {
  turn_id: string;
  label: string;
}

export interface FeedbackEvidence {
  turn_metrics?: Record<string, unknown>;
  debug?: Record<string, unknown>;
  recent_events?: Record<string, unknown>[];
}

export interface FeedbackPayload {
  csr_id: string;
  room_name: string;
  source: FeedbackSource;
  note: string;
  turn: FeedbackTurn;
  evidence: FeedbackEvidence;
}

export interface FeedbackRecord extends FeedbackPayload {
  id: string;
  created_at: string;
}

interface FeedbackListItem {
  id: string;
  title: string;
  subtitle: string;
  note: string;
  assistantPreview: string;
  record: FeedbackRecord;
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function latestByTopic(
  events: TraceEvent[],
  topic: string,
  turnId = "",
): Record<string, unknown> | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (turnId && readString(event.payload.turn_id) !== turnId) continue;
    if (event.topic === topic) return event.payload;
  }
  return undefined;
}

function latestDebug(
  events: TraceEvent[],
  turnId = "",
): Record<string, unknown> | undefined {
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const event = events[i];
    if (event.topic !== "dual_llm_debug") continue;
    if (turnId && readString(event.payload.turn_id) !== turnId) continue;
    const name = readString(event.payload.event);
    if (
      name === "fast_judge_done" ||
      name === "judge_done" ||
      name === "speech_done" ||
      name === "fast_judge_start"
    ) {
      return event.payload;
    }
  }
  return latestByTopic(events, "dual_llm_debug", turnId);
}

function buildAssistantText(events: TraceEvent[], turnId = ""): string {
  const assistantPieces = events
    .filter(
      (event) =>
        event.topic === "dual_llm_debug" &&
        (!turnId || readString(event.payload.turn_id) === turnId) &&
        (event.payload.event === "speech_done" ||
          event.payload.event === "stage2_done") &&
        readString(event.payload.response_text),
    )
    .map((event) => ({
      turnId: readString(event.payload.turn_id),
      text: readString(event.payload.response_text),
    }));
  const latestTurnId =
    turnId ||
    [...assistantPieces].reverse().find((piece) => piece.turnId)?.turnId;
  if (latestTurnId) {
    return assistantPieces
      .filter((piece) => piece.turnId === latestTurnId)
      .map((piece) => piece.text)
      .join("");
  }
  const latestAssistant = assistantPieces[assistantPieces.length - 1];
  if (latestAssistant) return latestAssistant.text;

  const judgeStart = [...events]
    .reverse()
    .find(
      (event) =>
        event.topic === "dual_llm_debug" &&
        (!turnId || readString(event.payload.turn_id) === turnId) &&
        readString(event.payload.assistant_text),
    );
  if (judgeStart) return readString(judgeStart.payload.assistant_text);

  if (turnId) return "";

  return events
    .filter((event) => event.topic === "agent_text")
    .map((event) => readString(event.payload.delta))
    .join("");
}

function buildUserText(
  events: TraceEvent[],
  turnMetrics?: Record<string, unknown>,
  turnId = "",
): string {
  const fromMetrics =
    readString(turnMetrics?.asr_final_text) ||
    readString(turnMetrics?.asr_text_at_emit) ||
    readString(turnMetrics?.llm_actual_text);
  if (fromMetrics) return fromMetrics;

  const speechStart = [...events]
    .reverse()
    .find(
      (event) =>
        event.topic === "dual_llm_debug" &&
        (!turnId || readString(event.payload.turn_id) === turnId) &&
        readString(event.payload.asr_text),
    );
  if (speechStart) return readString(speechStart.payload.asr_text);

  const transcription = latestByTopic(events, "transcription", turnId);
  return readString(transcription?.text);
}

function collectTurnIds(events: TraceEvent[]): string[] {
  const seen = new Set<string>();
  const turnIds: string[] = [];
  for (const event of events) {
    const turnId = readString(event.payload.turn_id);
    if (!turnId || seen.has(turnId)) continue;
    seen.add(turnId);
    turnIds.push(turnId);
  }
  return turnIds;
}

function shortPreview(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= 24) return trimmed;
  return `${trimmed.slice(0, 24)}...`;
}

export function buildFeedbackTurnOptions(
  events: TraceEvent[],
): FeedbackTurnOption[] {
  return collectTurnIds(events).map((turnId, index) => {
    const turnMetrics = latestByTopic(events, "turn_metrics", turnId);
    const debug = latestDebug(events, turnId);
    const userText = buildUserText(events, turnMetrics, turnId);
    const assistantText = buildAssistantText(events, turnId);
    const nodeId = readString(debug?.node) || readString(debug?.node_id);
    const preview = shortPreview(userText || assistantText || turnId);
    const nodeLabel = nodeId || turnId;
    return {
      turn_id: turnId,
      label: `第 ${index + 1} 轮 · ${nodeLabel} · ${preview}`,
      user_text: userText,
      assistant_text: assistantText,
      node_id: nodeId,
    };
  });
}

export function buildFeedbackDraft(input: {
  csrId: string;
  roomName: string;
  source: FeedbackSource;
  events: TraceEvent[];
  selectedTurnId?: string;
}): FeedbackPayload {
  const selectedTurnId = input.selectedTurnId?.trim() ?? "";
  const turnMetrics = latestByTopic(
    input.events,
    "turn_metrics",
    selectedTurnId,
  );
  const debug = latestDebug(input.events, selectedTurnId);
  const assistantText = buildAssistantText(input.events, selectedTurnId);
  const userText = buildUserText(input.events, turnMetrics, selectedTurnId);
  const turnId =
    selectedTurnId ||
    readString(turnMetrics?.turn_id) ||
    readString(debug?.turn_id);
  const nodeId = readString(debug?.node) || readString(debug?.node_id);

  return {
    csr_id: input.csrId,
    room_name: input.roomName,
    source: input.source,
    note: "",
    turn: {
      turn_id: turnId,
      user_text: userText,
      assistant_text: assistantText,
      node_id: nodeId,
    },
    evidence: {
      turn_metrics: turnMetrics,
      debug,
      recent_events: input.events.map((event) => ({
        topic: event.topic,
        ...event.payload,
      })),
    },
  };
}

export function buildFeedbackListItems(
  records: FeedbackRecord[],
): FeedbackListItem[] {
  return [...records]
    .sort((a, b) => b.created_at.localeCompare(a.created_at))
    .map((record) => ({
      id: record.id,
      title: `${record.csr_id} · ${record.source === "debug" ? "调试台" : "客服页"}`,
      subtitle: `${record.room_name} · ${record.created_at}`,
      note: record.note,
      assistantPreview: record.turn.assistant_text || "未捕获 AI 回复文本",
      record,
    }));
}
