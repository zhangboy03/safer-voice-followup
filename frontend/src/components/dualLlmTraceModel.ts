export type JsonMap = Record<string, unknown>;

export interface PromptMessage extends JsonMap {
  role: string;
  content: unknown;
}

export type ReasonSource = "model" | "fallback" | "absent";

export interface FastJudgeTraceSummary {
  schemaVersion: string;
  rawResult: JsonMap | null;
  parseError: string;
  fieldDelta: JsonMap[];
  lookaheadFieldDelta: JsonMap[];
  missingFields: string[];
  stageComplete: boolean | null;
  nodeAction: string;
  nodeAfterCandidate: string;
  nextNodeGoal: string;
  stage2ShouldStart: boolean | null;
  stage2TargetNode: string;
  stage2Reason: string;
  answerAssessment: JsonMap | null;
  reasonSource: ReasonSource;
  handoff: JsonMap | null;
  knowledgeQuestion: JsonMap | null;
  speechDirective: JsonMap | null;
  cacheUsage: JsonMap | null;
}

const REASON_SOURCE_VALUES: readonly ReasonSource[] = [
  "model",
  "fallback",
  "absent",
];

function isObject(value: unknown): value is JsonMap {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(map: JsonMap, key: string): string {
  const value = map[key];
  return typeof value === "string" ? value : "";
}

function readBooleanOrNull(map: JsonMap, key: string): boolean | null {
  const value = map[key];
  return typeof value === "boolean" ? value : null;
}

function readObject(map: JsonMap, key: string): JsonMap | null {
  const value = map[key];
  return isObject(value) ? value : null;
}

function readObjectArray(map: JsonMap, key: string): JsonMap[] {
  const value = map[key];
  return Array.isArray(value) ? value.filter(isObject) : [];
}

function readStringArray(map: JsonMap, key: string): string[] {
  const value = map[key];
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      if (typeof item === "string") return item;
      if (isObject(item)) {
        return readString(item, "field") || readString(item, "reason");
      }
      return "";
    })
    .filter(Boolean);
}

function parseJsonMap(value: string): { data: JsonMap | null; error: string } {
  if (!value.trim()) return { data: null, error: "" };
  try {
    const parsed = JSON.parse(value) as unknown;
    return isObject(parsed)
      ? { data: parsed, error: "" }
      : { data: null, error: "parsed value is not an object" };
  } catch (error) {
    return {
      data: null,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

export function formatPromptContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function buildPromptTranscript(messages: PromptMessage[]): string {
  return messages
    .map((message) => formatPromptContent(message.content).trimEnd())
    .filter(Boolean)
    .join("\n\n");
}

export function emptyFastJudgeTraceSummary(): FastJudgeTraceSummary {
  return {
    schemaVersion: "",
    rawResult: null,
    parseError: "",
    fieldDelta: [],
    lookaheadFieldDelta: [],
    missingFields: [],
    stageComplete: null,
    nodeAction: "",
    nodeAfterCandidate: "",
    nextNodeGoal: "",
    stage2ShouldStart: null,
    stage2TargetNode: "",
    stage2Reason: "",
    answerAssessment: null,
    reasonSource: "absent",
    handoff: null,
    knowledgeQuestion: null,
    speechDirective: null,
    cacheUsage: null,
  };
}

export function summarizeFastJudgePayload(
  payload: JsonMap,
): FastJudgeTraceSummary {
  const rawText = readString(payload, "response_text");
  const parsed = parseJsonMap(rawText);
  const result = parsed.data;
  const assessment = result ? readObject(result, "answer_assessment") : null;
  // reason_source is set by the backend _prepare_fast_judge_payload step
  // (v2 maps top-level `reason` → answer_assessment.followup_reason and
  // tags reason_source). It lives on the top-level event payload —
  // llm_processor emits it as a separate kwarg — not inside response_text,
  // which is the raw pre-mapping LLM output. Read from the event first,
  // then fall back to judge_result.answer_assessment for completeness.
  const topLevelReasonSource = readString(payload, "reason_source");
  const judgeResult = readObject(payload, "judge_result");
  const judgeAssessment = judgeResult
    ? readObject(judgeResult, "answer_assessment")
    : null;
  const judgeReasonSource = judgeAssessment
    ? readString(judgeAssessment, "reason_source")
    : "";
  const rawReasonSource = topLevelReasonSource || judgeReasonSource;
  const reasonSource: ReasonSource = (
    REASON_SOURCE_VALUES as readonly string[]
  ).includes(rawReasonSource)
    ? (rawReasonSource as ReasonSource)
    : "absent";

  return {
    schemaVersion: result ? readString(result, "schema_version") : "",
    rawResult: result,
    parseError: parsed.error,
    fieldDelta: result ? readObjectArray(result, "field_delta") : [],
    lookaheadFieldDelta: result
      ? readObjectArray(result, "lookahead_field_delta")
      : [],
    missingFields:
      (result ? readStringArray(result, "missing_fields") : []).length > 0
        ? readStringArray(result ?? {}, "missing_fields")
        : assessment
          ? readStringArray(assessment, "missing_fields")
          : [],
    stageComplete: result
      ? readBooleanOrNull(result, "stage_complete")
      : readBooleanOrNull(payload, "stage_complete"),
    nodeAction:
      (result ? readString(result, "node_action") : "") ||
      readString(payload, "node_action"),
    nodeAfterCandidate: result
      ? readString(result, "node_after_candidate")
      : "",
    nextNodeGoal: result ? readString(result, "next_node_goal") : "",
    stage2ShouldStart:
      (result ? readBooleanOrNull(result, "stage2_should_start") : null) ??
      readBooleanOrNull(payload, "stage2_should_start"),
    stage2TargetNode:
      (result ? readString(result, "stage2_target_node") : "") ||
      readString(payload, "stage2_target_node"),
    stage2Reason:
      (result ? readString(result, "stage2_reason") : "") ||
      readString(payload, "stage2_reason"),
    answerAssessment: assessment,
    reasonSource,
    handoff: result ? readObject(result, "handoff") : null,
    knowledgeQuestion: result ? readObject(result, "knowledge_question") : null,
    speechDirective: result ? readObject(result, "speech_directive") : null,
    cacheUsage: readObject(payload, "cache_usage"),
  };
}

export function extractFastJudgeContext(
  messages: PromptMessage[],
): JsonMap | null {
  for (const message of messages) {
    if (typeof message.content !== "string") continue;
    const match = message.content.match(/<context>\s*([\s\S]*?)\s*<\/context>/);
    if (!match) continue;
    const parsed = parseJsonMap(match[1]);
    if (parsed.data) return parsed.data;
  }
  return null;
}
