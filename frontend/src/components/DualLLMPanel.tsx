import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import type { DataHandler } from "../hooks/useLiveKit";
import type { FastJudgeTraceSummary, PromptMessage } from "./dualLlmTraceModel";
import {
  buildPromptTranscript,
  emptyFastJudgeTraceSummary,
  extractFastJudgeContext,
  formatPromptContent,
  summarizeFastJudgePayload,
} from "./dualLlmTraceModel";

type JsonMap = Record<string, unknown>;
type LaneStatus = "idle" | "waiting" | "running" | "done" | "error";
type Tone = "good" | "warn" | "bad" | "muted" | "info";

interface DebugEvent extends JsonMap {
  event: string;
  turn_id: string;
  ts: number;
  trace_seq?: number;
}

interface LaneTrace {
  status: LaneStatus;
  prompt: PromptMessage[];
  response: string;
  latencyMs: number;
}

interface JudgeTrace extends LaneTrace {
  label: string;
  asrText: string;
  assistantText: string;
  fastJudge: FastJudgeTraceSummary;
  fastJudgeContext: JsonMap | null;
  extractedFields: JsonMap;
  stageComplete: boolean | null;
  transferToHuman: boolean;
  knowledgeQuery: boolean;
  reason: string;
  error: string;
  structuredResult: JsonMap | null;
  commitDecision: JsonMap | null;
  turnAssessment: JsonMap | null;
  fieldCandidates: JsonMap[];
  missingFields: JsonMap[];
  pendingFields: JsonMap[];
  futureObservations: JsonMap[];
  contradictions: string[];
  speechRisks: JsonMap[];
  handoff: JsonMap | null;
  knowledgeQueryDetail: JsonMap | null;
  rejectedReason: string;
}

interface TurnMetrics {
  e2e_ms?: number;
  speech_end_to_audio_ms?: number;
  llm_ttft_ms?: number;
  tts_ttfa_ms?: number;
  asr_text_type?: string;
  asr_text_at_emit?: string;
  asr_final_text?: string;
  llm_actual_text?: string;
  unexplained_gap_ms?: number;
  first_audio_mono_ms?: number;
  [key: string]: unknown;
}

// ADR 0006: silent-rescue trace. Backend emits 5 `silent_rescue_*` events
// during an AI playout → 3s silence → rerun Stage 1 cycle.
type SilentRescueState =
  | "idle"
  | "armed"
  | "fired"
  | "cancelled"
  | "capReached"
  | "empty";

interface SilentRescueTrace {
  state: SilentRescueState;
  total: number; // total fires this turn
  chainLength: number; // current consecutive chain
  timeoutMs: number; // configured timer value, from arm event
  lastReason: string; // cap_reached reason / empty_output reason / skip reason
}

function fullTurnLatency(
  metrics: TurnMetrics | null | undefined,
): number | undefined {
  if (!metrics) return undefined;
  return typeof metrics.speech_end_to_audio_ms === "number" &&
    Number.isFinite(metrics.speech_end_to_audio_ms) &&
    metrics.speech_end_to_audio_ms >= 0
    ? metrics.speech_end_to_audio_ms
    : metrics.e2e_ms;
}

interface DecisionTrace {
  received: boolean;
  judgeRan: boolean;
  judgeStageComplete: boolean;
  policySatisfied: boolean;
  policyMissing: string[];
  willAdvance: boolean;
  collectedFields: JsonMap;
}

interface SystemAction {
  label: string;
  detail: string;
  tone: Tone;
}

interface TurnTrace {
  turnId: string;
  node: string;
  userText: string;
  isSpeculative: boolean;
  speech: LaneTrace;
  judge: JudgeTrace;
  stage2: LaneTrace & { node: string };
  decision: DecisionTrace;
  advanced: boolean;
  oldNode: string;
  newNode: string;
  advanceReason: string;
  appendSkipReason: string;
  metrics: TurnMetrics | null;
  actions: SystemAction[];
  events: DebugEvent[];
  silentRescue: SilentRescueTrace;
  updatedAt: number;
}

function isObject(value: unknown): value is JsonMap {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(map: JsonMap, key: string): string {
  const value = map[key];
  return typeof value === "string" ? value : "";
}

function readNumber(map: JsonMap, key: string): number {
  const value = map[key];
  return typeof value === "number" && Number.isFinite(value) ? value : -1;
}

function readBoolean(map: JsonMap, key: string): boolean {
  return map[key] === true;
}

function readStringArray(map: JsonMap, key: string): string[] {
  const value = map[key];
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function readObjectArray(map: JsonMap, key: string): JsonMap[] {
  const value = map[key];
  return Array.isArray(value) ? value.filter(isObject) : [];
}

function readPromptMessages(value: unknown): PromptMessage[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isObject).map((msg) => ({
    ...msg,
    role: readString(msg, "role") || "unknown",
    content: msg.content,
  }));
}

function formatJson(value: unknown): string {
  if (value == null || value === "") return "";
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function ms(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? `${Math.round(value)}ms`
    : "--";
}

function shortId(value: string): string {
  if (!value) return "--";
  return value.length > 12 ? `${value.slice(0, 10)}…` : value;
}

function boolTone(value: boolean): Tone {
  return value ? "good" : "warn";
}

function emptyLane(status: LaneStatus = "idle"): LaneTrace {
  return {
    status,
    prompt: [],
    response: "",
    latencyMs: -1,
  };
}

function emptyJudge(): JudgeTrace {
  return {
    ...emptyLane(),
    label: "Judge",
    asrText: "",
    assistantText: "",
    fastJudge: emptyFastJudgeTraceSummary(),
    fastJudgeContext: null,
    extractedFields: {},
    stageComplete: null,
    transferToHuman: false,
    knowledgeQuery: false,
    reason: "",
    error: "",
    structuredResult: null,
    commitDecision: null,
    turnAssessment: null,
    fieldCandidates: [],
    missingFields: [],
    pendingFields: [],
    futureObservations: [],
    contradictions: [],
    speechRisks: [],
    handoff: null,
    knowledgeQueryDetail: null,
    rejectedReason: "",
  };
}

function emptySilentRescue(): SilentRescueTrace {
  return {
    state: "idle",
    total: 0,
    chainLength: 0,
    timeoutMs: 0,
    lastReason: "",
  };
}

function emptyDecision(): DecisionTrace {
  return {
    received: false,
    judgeRan: false,
    judgeStageComplete: false,
    policySatisfied: false,
    policyMissing: [],
    willAdvance: false,
    collectedFields: {},
  };
}

function emptyTurn(turnId: string): TurnTrace {
  return {
    turnId,
    node: "",
    userText: "",
    isSpeculative: false,
    speech: emptyLane(),
    judge: emptyJudge(),
    stage2: { ...emptyLane(), node: "" },
    decision: emptyDecision(),
    advanced: false,
    oldNode: "",
    newNode: "",
    advanceReason: "",
    appendSkipReason: "",
    metrics: null,
    actions: [],
    events: [],
    silentRescue: emptySilentRescue(),
    updatedAt: Date.now(),
  };
}

function action(
  label: string,
  detail: string,
  tone: Tone = "info",
): SystemAction {
  return { label, detail, tone };
}

function addAction(turn: TurnTrace, next: SystemAction): SystemAction[] {
  const deduped = turn.actions.filter(
    (item) => item.label !== next.label || item.detail !== next.detail,
  );
  return [...deduped.slice(-5), next];
}

function debugEventIdentity(event: DebugEvent): string {
  return [
    event.event,
    event.turn_id,
    event.ts,
    readString(event, "node"),
    readString(event, "old_node"),
    readString(event, "new_node"),
    readString(event, "reason"),
    readString(event, "error"),
  ].join("|");
}

function applyDebugEvent(turn: TurnTrace, event: DebugEvent): TurnTrace {
  const identity = debugEventIdentity(event);
  if (
    turn.events.some((existing) => debugEventIdentity(existing) === identity)
  ) {
    return turn;
  }

  const eventWithSeq: DebugEvent = {
    ...event,
    trace_seq:
      typeof event.trace_seq === "number"
        ? event.trace_seq
        : turn.events.length + 1,
  };
  const next: TurnTrace = {
    ...turn,
    speech: { ...turn.speech },
    judge: { ...turn.judge },
    stage2: { ...turn.stage2 },
    decision: { ...turn.decision },
    actions: [...turn.actions],
    events: [...turn.events.slice(-39), eventWithSeq],
    silentRescue: { ...turn.silentRescue },
    updatedAt: Date.now(),
  };

  if (event.turn_id) next.turnId = event.turn_id;

  switch (event.event) {
    case "speech_start":
      next.userText = readString(event, "asr_text") || next.userText;
      next.node = readString(event, "node") || next.node;
      next.isSpeculative = readBoolean(event, "is_speculative");
      next.speech = {
        status: "running",
        prompt: readPromptMessages(event.prompt_messages),
        response: "",
        latencyMs: -1,
      };
      if (next.judge.status === "idle") next.judge.status = "waiting";
      break;
    case "speech_done":
      next.node = readString(event, "node") || next.node;
      next.speech.status = "done";
      next.speech.response = readString(event, "response_text");
      next.speech.latencyMs = readNumber(event, "latency_ms");
      break;
    case "judge_start":
    case "judge_shadow_start":
    case "fast_judge_start":
      next.node = readString(event, "node") || next.node;
      next.judge.label =
        event.event === "fast_judge_start"
          ? "Fast Judge"
          : event.event === "judge_shadow_start"
            ? "Shadow Judge"
            : "Judge";
      next.judge.status = "running";
      next.judge.prompt = readPromptMessages(event.prompt_messages);
      next.judge.response = "";
      next.judge.error = "";
      next.judge.asrText = readString(event, "asr_text") || next.userText;
      next.judge.assistantText = readString(event, "assistant_text");
      next.judge.fastJudgeContext =
        event.event === "fast_judge_start"
          ? extractFastJudgeContext(next.judge.prompt)
          : null;
      break;
    case "judge_done":
    case "judge_shadow_done":
    case "fast_judge_done":
      next.node = readString(event, "node") || next.node;
      next.judge.label =
        event.event === "fast_judge_done"
          ? "Fast Judge"
          : event.event === "judge_shadow_done"
            ? "Shadow Judge"
            : next.judge.label;
      next.judge.status = readString(event, "error") ? "error" : "done";
      next.judge.response = readString(event, "response_text");
      next.judge.latencyMs = readNumber(event, "latency_ms");
      next.judge.error = readString(event, "error");
      next.judge.extractedFields = isObject(event.extracted_fields)
        ? event.extracted_fields
        : {};
      next.judge.stageComplete =
        typeof event.stage_complete === "boolean" ? event.stage_complete : null;
      next.judge.transferToHuman = readBoolean(event, "transfer_to_human");
      next.judge.knowledgeQuery = readBoolean(event, "knowledge_query");
      next.judge.reason =
        readString(event, "reason") ||
        (event.event === "fast_judge_done"
          ? ""
          : readString(event, "node_action"));
      next.judge.structuredResult = isObject(event.judge_result)
        ? event.judge_result
        : null;
      next.judge.commitDecision = isObject(event.commit_decision)
        ? event.commit_decision
        : null;
      next.judge.turnAssessment = isObject(event.turn_assessment)
        ? event.turn_assessment
        : null;
      next.judge.fieldCandidates = readObjectArray(event, "field_candidates");
      next.judge.missingFields = readObjectArray(event, "missing_fields");
      next.judge.pendingFields = readObjectArray(event, "pending_fields");
      next.judge.futureObservations = readObjectArray(
        event,
        "future_observations",
      );
      next.judge.contradictions = readStringArray(event, "contradictions");
      next.judge.speechRisks = readObjectArray(event, "speech_risks");
      next.judge.handoff = isObject(event.handoff) ? event.handoff : null;
      next.judge.knowledgeQueryDetail = isObject(event.knowledge_query_detail)
        ? event.knowledge_query_detail
        : null;
      if (event.event === "fast_judge_done") {
        const fastJudge = summarizeFastJudgePayload(event);
        next.judge.fastJudge = fastJudge;
        next.judge.stageComplete =
          fastJudge.stageComplete ?? next.judge.stageComplete;
        next.judge.reason =
          readString(fastJudge.answerAssessment ?? {}, "followup_reason") ||
          readString(event, "reason");
        next.judge.transferToHuman =
          readBoolean(fastJudge.handoff ?? {}, "transfer_to_human") ||
          next.judge.transferToHuman;
        next.judge.knowledgeQuery =
          readBoolean(fastJudge.knowledgeQuestion ?? {}, "detected") ||
          next.judge.knowledgeQuery;
      }
      break;
    case "advance_decision":
      next.node = readString(event, "node") || next.node;
      next.decision = {
        received: true,
        judgeRan: readBoolean(event, "judge_ran"),
        judgeStageComplete: readBoolean(event, "judge_stage_complete"),
        policySatisfied: readBoolean(event, "policy_satisfied"),
        policyMissing: readStringArray(event, "policy_missing"),
        willAdvance: readBoolean(event, "will_advance"),
        collectedFields: isObject(event.collected_fields)
          ? event.collected_fields
          : {},
      };
      break;
    case "judge_rejected":
      next.judge.status = "error";
      next.judge.rejectedReason = readString(event, "reason");
      next.actions = addAction(
        next,
        action("Judge rejected", next.judge.rejectedReason || "unknown", "bad"),
      );
      break;
    case "stage_advance":
      next.advanced = true;
      next.oldNode = readString(event, "old_node");
      next.newNode = readString(event, "new_node");
      next.advanceReason = readString(event, "reason");
      next.actions = addAction(
        next,
        action(
          "Node advanced",
          `${next.oldNode || next.node} -> ${next.newNode || "--"}`,
          "good",
        ),
      );
      break;
    case "stage2_start":
      next.stage2.status = "running";
      next.stage2.node = readString(event, "node");
      next.stage2.prompt = readPromptMessages(event.prompt_messages);
      next.stage2.response = "";
      break;
    case "stage2_done":
      next.stage2.status = "done";
      next.stage2.node = readString(event, "node") || next.stage2.node;
      next.stage2.response = readString(event, "response_text");
      next.stage2.latencyMs = readNumber(event, "latency_ms");
      break;
    case "same_playout_append_skipped":
      next.appendSkipReason = readString(event, "reason") || "unknown";
      next.actions = addAction(
        next,
        action("Same-playout append skipped", next.appendSkipReason, "warn"),
      );
      break;
    case "transfer_to_human":
      next.actions = addAction(
        next,
        action("Transfer", readString(event, "reason") || "requested", "bad"),
      );
      break;
    case "terminal_hook":
      next.actions = addAction(
        next,
        action("Terminal payload", "form submit + call complete", "good"),
      );
      break;
    case "goodbye_detected":
      next.actions = addAction(
        next,
        action("Goodbye detected", readString(event, "text"), "good"),
      );
      break;
    // ADR 0006 — silent-rescue lifecycle. Backend emits 5 silent_rescue_* events
    // around the AI playout → 3s silence → rerun Stage 1 cycle.
    case "silent_rescue_timer_start":
      next.silentRescue.state = "armed";
      next.silentRescue.timeoutMs = readNumber(event, "timeout_ms");
      break;
    case "silent_rescue_timer_cancel":
      // Cancellations are noise unless previously armed; rendering hides idle.
      if (next.silentRescue.state === "armed") {
        next.silentRescue.state = "cancelled";
      }
      next.silentRescue.lastReason =
        readString(event, "reason") || next.silentRescue.lastReason;
      break;
    case "silent_rescue_fire": {
      const total = readNumber(event, "total");
      const chain = readNumber(event, "chain_length");
      next.silentRescue.state = "fired";
      next.silentRescue.total =
        total >= 0 ? total : next.silentRescue.total + 1;
      next.silentRescue.chainLength =
        chain >= 0 ? chain : next.silentRescue.chainLength + 1;
      next.actions = addAction(
        next,
        action(
          "Silent rescue fired",
          `total=${next.silentRescue.total} chain=${next.silentRescue.chainLength}`,
          "warn",
        ),
      );
      break;
    }
    case "silent_rescue_cap_reached":
      next.silentRescue.state = "capReached";
      next.silentRescue.lastReason = `cap=${readNumber(event, "cap")} total=${readNumber(event, "total")}`;
      next.actions = addAction(
        next,
        action(
          "Silent rescue cap reached",
          next.silentRescue.lastReason,
          "bad",
        ),
      );
      break;
    case "silent_rescue_empty_output":
      next.silentRescue.state = "empty";
      next.silentRescue.lastReason = "model returned empty";
      next.actions = addAction(
        next,
        action("Silent rescue empty output", "no text emitted", "bad"),
      );
      break;
    case "silent_rescue_skipped":
      next.silentRescue.lastReason = readString(event, "reason") || "unknown";
      break;
  }

  return next;
}

function upsertTurn(
  turns: TurnTrace[],
  turnId: string,
  updater: (turn: TurnTrace) => TurnTrace,
): TurnTrace[] {
  const safeTurnId = turnId || "session";
  const index = turns.findIndex((turn) => turn.turnId === safeTurnId);
  if (index === -1) return [...turns, updater(emptyTurn(safeTurnId))];
  const next = [...turns];
  next[index] = updater(next[index]);
  return next;
}

function eventFromPayload(payload: Record<string, unknown>): DebugEvent {
  return {
    ...payload,
    event: readString(payload, "event"),
    turn_id: readString(payload, "turn_id"),
    ts:
      typeof payload.ts === "number" && Number.isFinite(payload.ts)
        ? payload.ts
        : Date.now() / 1000,
  };
}

function normalizeMetrics(payload: Record<string, unknown>): TurnMetrics {
  return { ...payload };
}

function extractPromptContext(messages: PromptMessage[]): string {
  const fastJudgeContext = extractFastJudgeContext(messages);
  if (fastJudgeContext) return formatJson(fastJudgeContext);

  const combined = messages
    .map((message) => formatPromptContent(message.content))
    .join("\n\n");
  const weighted = combined.indexOf("[WeightedContext]");
  if (weighted >= 0) return combined.slice(weighted).trim();
  const snapshot = combined.indexOf("[ContextSnapshot]");
  if (snapshot >= 0) return combined.slice(snapshot).trim();
  const working = combined.indexOf("[工作记忆]");
  if (working >= 0) return combined.slice(working).trim();
  return "";
}

function Badge({
  children,
  tone = "muted",
}: {
  children: ReactNode;
  tone?: Tone;
}) {
  return <span className={`trace-badge trace-badge--${tone}`}>{children}</span>;
}

function StatusPill({ status }: { status: LaneStatus }) {
  const tone: Tone =
    status === "done"
      ? "good"
      : status === "running"
        ? "info"
        : status === "error"
          ? "bad"
          : status === "waiting"
            ? "warn"
            : "muted";
  return <Badge tone={tone}>{status}</Badge>;
}

function EmptyState({ label }: { label: string }) {
  return <div className="debug-empty">{label}</div>;
}

function PromptMessages({ messages }: { messages: PromptMessage[] }) {
  if (messages.length === 0) {
    return <EmptyState label="尚未收到 model-visible prompt" />;
  }
  const transcript = buildPromptTranscript(messages);
  const roles = messages.map((message) => message.role).join(" / ");

  return (
    <article className="prompt-transcript">
      <div className="prompt-transcript__head">
        <span>Model-visible transcript</span>
        <span>
          {messages.length} messages · {roles}
        </span>
      </div>
      <pre>{transcript}</pre>
    </article>
  );
}

function PromptInspector({ turn }: { turn: TurnTrace }) {
  const available = [
    { id: "speech", label: "Speech", messages: turn.speech.prompt },
    { id: "judge", label: turn.judge.label, messages: turn.judge.prompt },
    { id: "stage2", label: "Stage 2", messages: turn.stage2.prompt },
  ].filter((item) => item.messages.length > 0);
  const [selected, setSelected] = useState("judge");
  const active =
    available.find((item) => item.id === selected) ?? available[0] ?? null;
  const context = active ? extractPromptContext(active.messages) : "";
  const hasFastJudgeContext = active
    ? extractFastJudgeContext(active.messages) !== null
    : false;

  return (
    <section className="trace-panel trace-panel--prompt">
      <div className="trace-panel__head">
        <div>
          <p className="panel-kicker">Prompt / Context</p>
          <h3>模型实际看到的输入</h3>
        </div>
        {active && <Badge tone="info">{active.label}</Badge>}
      </div>

      {available.length > 1 && (
        <div className="segmented-control">
          {available.map((item) => (
            <button
              className={item.id === active?.id ? "is-active" : ""}
              key={item.id}
              type="button"
              onClick={() => setSelected(item.id)}
            >
              {item.label}
            </button>
          ))}
        </div>
      )}

      {active ? (
        <PromptMessages messages={active.messages} />
      ) : (
        <EmptyState label="等待 Speech/Judge prompt" />
      )}

      <div className="context-extract">
        <div className="context-extract__title">
          {hasFastJudgeContext
            ? "Fast Judge context"
            : "Weighted context extract"}
        </div>
        {context ? (
          <pre>{context}</pre>
        ) : (
          <EmptyState label="当前 prompt 内没有可抽取的 context 标记" />
        )}
      </div>
    </section>
  );
}

function FieldList({
  title,
  items,
  empty,
}: {
  title: string;
  items: JsonMap[];
  empty: string;
}) {
  return (
    <div className="field-list">
      <div className="field-list__title">{title}</div>
      {items.length === 0 ? (
        <span className="field-list__empty">{empty}</span>
      ) : (
        items.map((item) => {
          const field =
            readString(item, "field") ||
            readString(item, "type") ||
            readString(item, "reason") ||
            "item";
          const nodeId = readString(item, "node_id");
          const status = readString(item, "status");
          const flowValue = isObject(item.flow_value)
            ? item.flow_value.value
            : undefined;
          const memoryValue = isObject(item.memory_value)
            ? item.memory_value.value
            : undefined;
          const value = item.value ?? flowValue ?? memoryValue;
          const confidence = item.confidence;
          return (
            <div
              className="field-row"
              key={`${field}-${formatJson(item).slice(0, 160)}`}
            >
              <span>{nodeId ? `${nodeId}.${field}` : field}</span>
              <code>
                {formatJson(value) || readString(item, "reason") || "--"}
              </code>
              {status && (
                <Badge tone={status === "confirmed" ? "good" : "warn"}>
                  {status}
                </Badge>
              )}
              {typeof confidence === "number" && (
                <span className="field-row__confidence">
                  {Math.round(confidence * 100)}%
                </span>
              )}
            </div>
          );
        })
      )}
    </div>
  );
}

function StringList({
  title,
  items,
  empty,
  tone = "warn",
}: {
  title: string;
  items: string[];
  empty: string;
  tone?: Tone;
}) {
  return (
    <div className="field-list">
      <div className="field-list__title">{title}</div>
      {items.length === 0 ? (
        <span className="field-list__empty">{empty}</span>
      ) : (
        <div className="string-chip-row">
          {items.map((item) => (
            <Badge key={item} tone={tone}>
              {item}
            </Badge>
          ))}
        </div>
      )}
    </div>
  );
}

function JudgeDecisionPanel({ turn }: { turn: TurnTrace }) {
  const fastJudge = turn.judge.fastJudge;
  const isFastJudge =
    turn.judge.label === "Fast Judge" ||
    fastJudge.schemaVersion === "fast_judge.v1";
  const assessment = fastJudge.answerAssessment ?? turn.judge.turnAssessment;
  const recommendedAction = assessment
    ? readString(assessment, "recommended_action") ||
      readString(assessment, "current_question_answered")
    : "";
  const needsFollowup = readBoolean(assessment ?? {}, "needs_followup");
  const directiveText = readString(fastJudge.speechDirective ?? {}, "text");
  const directiveCode = readString(fastJudge.speechDirective ?? {}, "code");
  const commitAccepted =
    turn.judge.commitDecision?.accepted === true ||
    turn.judge.commitDecision?.accepted === false
      ? turn.judge.commitDecision.accepted === true
      : null;
  const commitAction = turn.judge.commitDecision
    ? readString(turn.judge.commitDecision, "node_action")
    : "";
  const committedFields = readObjectArray(
    turn.judge.commitDecision ?? {},
    "field_commits",
  );
  const stage2Status =
    fastJudge.stage2ShouldStart === null
      ? "--"
      : fastJudge.stage2ShouldStart
        ? "requested"
        : "not requested";
  const explicitReason =
    readString(assessment ?? {}, "followup_reason") ||
    readString(assessment ?? {}, "reason") ||
    turn.judge.reason ||
    turn.judge.error ||
    turn.judge.rejectedReason;
  const reasonText =
    explicitReason ||
    (isFastJudge && fastJudge.nodeAction
      ? `无显式 reason；node_action=${fastJudge.nodeAction}`
      : "尚未输出");

  return (
    <section className="trace-panel">
      <div className="trace-panel__head">
        <div>
          <p className="panel-kicker">{turn.judge.label} / System</p>
          <h3>判定和系统响应</h3>
        </div>
        <StatusPill status={turn.judge.status} />
      </div>

      <div className="decision-grid">
        <div>
          <span>stage_complete</span>
          <strong
            className={`tone-${boolTone(turn.judge.stageComplete === true)}`}
          >
            {turn.judge.stageComplete === null
              ? "--"
              : String(turn.judge.stageComplete)}
          </strong>
        </div>
        <div>
          <span>{isFastJudge ? "answer" : "recommended"}</span>
          <strong className={needsFollowup ? "tone-warn" : ""}>
            {recommendedAction || "--"}
          </strong>
        </div>
        <div>
          <span>commit</span>
          <strong className={commitAccepted === false ? "tone-bad" : ""}>
            {commitAccepted === null
              ? "--"
              : commitAccepted
                ? "accepted"
                : "rejected"}
          </strong>
        </div>
        <div>
          <span>node action</span>
          <strong>
            {fastJudge.nodeAction ||
              commitAction ||
              (turn.decision.willAdvance ? "advance" : "stay")}
          </strong>
        </div>
      </div>

      <div className="reason-box">
        <span>
          {turn.judge.label} reason
          {fastJudge.reasonSource === "fallback" && (
            <span
              className="reason-source-badge reason-source-badge--fallback"
              title="LLM 没自己写 reason，从 speech_directive.text 兜底"
            >
              ⚠ 推理为兜底
            </span>
          )}
        </span>
        <p>{reasonText}</p>
      </div>

      {directiveText && (
        <div className="reason-box reason-box--directive">
          <span>
            speech_directive{directiveCode ? ` · ${directiveCode}` : ""}
          </span>
          <p>{directiveText}</p>
        </div>
      )}

      {isFastJudge && (
        <div className="gate-line">
          <Badge tone="info">stage2={stage2Status}</Badge>
          {fastJudge.stage2TargetNode && (
            <Badge tone="info">target={fastJudge.stage2TargetNode}</Badge>
          )}
          {fastJudge.stage2Reason && (
            <Badge tone="muted">reason={fastJudge.stage2Reason}</Badge>
          )}
          {fastJudge.nodeAfterCandidate && (
            <Badge tone="muted">
              node_after_candidate={fastJudge.nodeAfterCandidate}
            </Badge>
          )}
          {fastJudge.nextNodeGoal && (
            <Badge tone="muted">goal={fastJudge.nextNodeGoal}</Badge>
          )}
        </div>
      )}

      {turn.decision.received && (
        <div className="gate-line">
          <Badge tone={turn.decision.judgeRan ? "good" : "muted"}>
            judge_ran={String(turn.decision.judgeRan)}
          </Badge>
          <Badge tone={boolTone(turn.decision.policySatisfied)}>
            policy={String(turn.decision.policySatisfied)}
          </Badge>
          <Badge tone={boolTone(turn.decision.willAdvance)}>
            will_advance={String(turn.decision.willAdvance)}
          </Badge>
          {turn.decision.policyMissing.length > 0 && (
            <Badge tone="warn">
              missing={turn.decision.policyMissing.join(",")}
            </Badge>
          )}
        </div>
      )}

      <FieldList
        title={isFastJudge ? "field_delta" : "field_candidates"}
        items={isFastJudge ? fastJudge.fieldDelta : turn.judge.fieldCandidates}
        empty={isFastJudge ? "无本节点字段增量" : "无候选字段"}
      />
      {isFastJudge && (
        <FieldList
          title="lookahead_field_delta"
          items={fastJudge.lookaheadFieldDelta}
          empty="无前瞻字段增量"
        />
      )}
      {isFastJudge && (
        <StringList
          title="missing_fields"
          items={fastJudge.missingFields}
          empty="无缺失字段"
        />
      )}
      <FieldList
        title="field_commits"
        items={committedFields}
        empty="无写入字段"
      />
      {!isFastJudge && (
        <FieldList
          title="missing_fields"
          items={turn.judge.missingFields}
          empty="无缺失字段"
        />
      )}

      {(turn.judge.speechRisks.length > 0 ||
        turn.judge.contradictions.length > 0 ||
        turn.judge.transferToHuman ||
        turn.judge.knowledgeQuery) && (
        <div className="risk-strip">
          {turn.judge.transferToHuman && <Badge tone="bad">handoff</Badge>}
          {turn.judge.knowledgeQuery && <Badge tone="info">knowledge</Badge>}
          {turn.judge.speechRisks.map((risk) => (
            <Badge
              key={formatJson(risk)}
              tone={
                readString(risk, "severity") === "critical" ? "bad" : "warn"
              }
            >
              {readString(risk, "type") || "risk"}
            </Badge>
          ))}
          {turn.judge.contradictions.map((item) => (
            <Badge key={item} tone="bad">
              contradiction
            </Badge>
          ))}
        </div>
      )}

      {turn.actions.length > 0 && (
        <div className="system-actions">
          {turn.actions.map((item) => (
            <div
              className={`system-action system-action--${item.tone}`}
              key={`${item.label}-${item.detail}-${item.tone}`}
            >
              <span>{item.label}</span>
              <p>{item.detail || "--"}</p>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function ConversationPanel({ turn }: { turn: TurnTrace }) {
  return (
    <section className="trace-panel trace-panel--conversation">
      <div className="trace-panel__head">
        <div>
          <p className="panel-kicker">Conversation</p>
          <h3>用户说了什么，AI 回了什么</h3>
        </div>
        <Badge tone={turn.isSpeculative ? "warn" : "good"}>
          {turn.isSpeculative ? "speculative" : "formal"}
        </Badge>
      </div>

      <div className="utterance-stack">
        <article className="utterance utterance--user">
          <div>
            <span>用户</span>
            <code>{turn.metrics?.asr_text_type || "asr"}</code>
          </div>
          <p>{turn.userText || turn.judge.asrText || "等待用户本轮文本"}</p>
        </article>
        <article className="utterance utterance--assistant">
          <div>
            <span>AI Speech</span>
            <StatusPill status={turn.speech.status} />
          </div>
          <p>
            {turn.speech.response ||
              turn.judge.assistantText ||
              (turn.speech.status === "running" ? "生成中..." : "等待回复")}
          </p>
        </article>
        {turn.stage2.status !== "idle" && (
          <article className="utterance utterance--assistant utterance--stage2">
            <div>
              <span>Stage 2 · {turn.stage2.node || "--"}</span>
              <StatusPill status={turn.stage2.status} />
            </div>
            <p>{turn.stage2.response || "生成中..."}</p>
          </article>
        )}
        {turn.stage2.status === "idle" && turn.advanced && (
          <article className="utterance utterance--assistant utterance--stage2-missing">
            <div>
              <span>Stage 2 · {turn.newNode || "--"}</span>
              <Badge tone="warn">missing</Badge>
            </div>
            <p>
              节点已推进但 Stage 2 未发声
              {turn.appendSkipReason
                ? ` · reason=${turn.appendSkipReason}`
                : "（未捕获到 same_playout_append_skipped；可能 next_plan 为 None 或被 barge-in 打断）"}
            </p>
          </article>
        )}
      </div>

      <div className="truth-grid">
        <div>
          <span>ASR at emit</span>
          <code>{turn.metrics?.asr_text_at_emit || "--"}</code>
        </div>
        <div>
          <span>ASR final</span>
          <code>{turn.metrics?.asr_final_text || "--"}</code>
        </div>
        <div>
          <span>LLM actual text</span>
          <code>{turn.metrics?.llm_actual_text || "--"}</code>
        </div>
      </div>
    </section>
  );
}

// ADR 0006: silent-rescue chip — compact summary of the AI playout → 3s silence
// → rerun Stage 1 cycle. Renders only when state ≠ "idle".
function SilentRescuePanel({ turn }: { turn: TurnTrace }) {
  const r = turn.silentRescue;
  if (r.state === "idle" && r.total === 0) return null;

  const stateTone: Tone =
    r.state === "fired"
      ? "warn"
      : r.state === "capReached" || r.state === "empty"
        ? "bad"
        : r.state === "armed"
          ? "info"
          : "muted";

  return (
    <section className="trace-panel trace-panel--conversation">
      <div className="trace-panel__head">
        <div>
          <p className="panel-kicker">Silent Rescue (ADR 0006)</p>
          <h3>沉默兜底重跑 Stage 1</h3>
        </div>
        <Badge tone={stateTone}>
          {r.state} · total={r.total} chain={r.chainLength}
        </Badge>
      </div>

      <div className="decision-grid">
        <div>
          <span>state</span>
          <strong className={`tone-${stateTone}`}>{r.state}</strong>
        </div>
        <div>
          <span>total fires</span>
          <strong>{r.total || "--"}</strong>
        </div>
        <div>
          <span>chain length</span>
          <strong>{r.chainLength || "--"}</strong>
        </div>
        <div>
          <span>timeout</span>
          <strong>{r.timeoutMs > 0 ? `${r.timeoutMs}ms` : "--"}</strong>
        </div>
      </div>

      {r.lastReason && (
        <div className="risk-strip">
          <Badge tone="muted">{r.lastReason}</Badge>
        </div>
      )}
    </section>
  );
}

function RawOutputPanel({ turn }: { turn: TurnTrace }) {
  return (
    <section className="trace-panel trace-panel--raw">
      <div className="trace-panel__head">
        <div>
          <p className="panel-kicker">Raw Output</p>
          <h3>{turn.judge.label} 输出和事件流</h3>
        </div>
        <Badge tone="muted">{turn.events.length} events</Badge>
      </div>

      <div className="raw-output-grid">
        <div>
          <span>{turn.judge.label} raw response</span>
          {turn.judge.response ? (
            <pre>{turn.judge.response}</pre>
          ) : (
            <EmptyState label={`尚未收到 ${turn.judge.label} raw response`} />
          )}
        </div>
        <div>
          <span>Commit decision</span>
          {turn.judge.commitDecision ? (
            <pre>{formatJson(turn.judge.commitDecision)}</pre>
          ) : (
            <EmptyState label="尚未进入 commit path" />
          )}
        </div>
        <div>
          <span>Backend usage / parse</span>
          {turn.judge.fastJudge.cacheUsage ||
          turn.judge.fastJudge.parseError ? (
            <pre>
              {formatJson({
                cache_usage: turn.judge.fastJudge.cacheUsage,
                parse_error: turn.judge.fastJudge.parseError || undefined,
              })}
            </pre>
          ) : (
            <EmptyState label="暂无 cache usage 或解析错误" />
          )}
        </div>
      </div>

      <div className="event-feed">
        {turn.events.slice(-12).map((event) => (
          <div
            className="event-feed__row"
            key={`${readNumber(event, "trace_seq")}-${event.event}-${event.ts}-${readString(event, "node")}-${readString(event, "reason")}-${readString(event, "error")}`}
          >
            <span>{new Date(event.ts * 1000).toLocaleTimeString()}</span>
            <strong>{event.event}</strong>
            <code>{readString(event, "node") || turn.node || "--"}</code>
          </div>
        ))}
      </div>
    </section>
  );
}

function TurnRail({
  turns,
  selectedTurnId,
  onSelect,
}: {
  turns: TurnTrace[];
  selectedTurnId: string;
  onSelect: (turnId: string) => void;
}) {
  return (
    <aside className="turn-rail" aria-label="对话轮次">
      <div className="turn-rail__head">
        <span>Turns</span>
        <Badge tone="muted">{String(turns.length)}</Badge>
      </div>
      {turns.length === 0 ? (
        <EmptyState label="暂无轮次" />
      ) : (
        turns
          .slice()
          .reverse()
          .map((turn) => {
            const isSelected = turn.turnId === selectedTurnId;
            const tone =
              turn.judge.status === "error"
                ? "bad"
                : turn.silentRescue.total > 0
                  ? "warn"
                  : turn.advanced
                    ? "good"
                    : turn.speech.status === "running" ||
                        turn.judge.status === "running"
                      ? "info"
                      : "muted";
            return (
              <button
                className={`turn-chip ${isSelected ? "is-selected" : ""}`}
                key={turn.turnId}
                type="button"
                onClick={() => onSelect(turn.turnId)}
              >
                <span className={`turn-chip__dot tone-${tone}`} />
                <span>
                  <strong>{shortId(turn.turnId)}</strong>
                  <small>{turn.node || "--"}</small>
                </span>
                <code>{ms(fullTurnLatency(turn.metrics))}</code>
              </button>
            );
          })
      )}
    </aside>
  );
}

function SessionEventStrip({ events }: { events: JsonMap[] }) {
  if (events.length === 0) return null;
  return (
    <div className="session-event-strip">
      {events.slice(-8).map((event) => (
        <span
          key={`${readString(event, "event_id") || readString(event, "event_type") || readString(event, "topic")}-${readNumber(event, "mono_ms")}-${readNumber(event, "wall_ts")}`}
        >
          {readString(event, "event_type") ||
            readString(event, "topic") ||
            "event"}
        </span>
      ))}
    </div>
  );
}

export function DualLLMPanel({
  registerDataHandler,
}: {
  registerDataHandler: (handler: DataHandler) => () => void;
}) {
  const [turns, setTurns] = useState<TurnTrace[]>([]);
  const [selectedTurnId, setSelectedTurnId] = useState("");
  const [sessionEvents, setSessionEvents] = useState<JsonMap[]>([]);

  useEffect(() => {
    const unsub = registerDataHandler((topic, payload) => {
      if (topic === "dual_llm_debug") {
        const event = eventFromPayload(payload);
        const turnId = event.turn_id || "session";
        setTurns((prev) =>
          upsertTurn(prev, turnId, (turn) => applyDebugEvent(turn, event)),
        );
        setSelectedTurnId((current) => current || turnId);
        return;
      }

      if (topic === "turn_metrics") {
        const metrics = normalizeMetrics(payload);
        const turnId = readString(payload, "turn_id");
        if (!turnId) return;
        setTurns((prev) =>
          upsertTurn(prev, turnId, (turn) => ({
            ...turn,
            userText:
              turn.userText ||
              readString(payload, "llm_actual_text") ||
              readString(payload, "asr_final_text") ||
              readString(payload, "asr_text_at_emit"),
            metrics,
            updatedAt: Date.now(),
          })),
        );
        setSelectedTurnId((current) => current || turnId);
        return;
      }

      setSessionEvents((prev) => [...prev.slice(-31), { topic, ...payload }]);
    });
    return unsub;
  }, [registerDataHandler]);

  const selectedTurn = useMemo(() => {
    if (turns.length === 0) return null;
    return (
      turns.find((turn) => turn.turnId === selectedTurnId) ??
      turns[turns.length - 1]
    );
  }, [turns, selectedTurnId]);

  if (!selectedTurn) {
    return (
      <section className="debug-console debug-console--empty">
        <div>
          <p className="panel-kicker">Realtime Trace</p>
          <h2>等待 Speech / Judge 调试事件</h2>
          <p>
            第一轮对话开始后，这里会展示真实 prompt、context、Judge
            输出和系统响应。
          </p>
        </div>
        <SessionEventStrip events={sessionEvents} />
      </section>
    );
  }

  return (
    <section className="debug-console">
      <div className="debug-console__head">
        <div>
          <p className="panel-kicker">Realtime Trace</p>
          <h2>
            {selectedTurn.node || "active node"} · turn{" "}
            <code>{shortId(selectedTurn.turnId)}</code>
          </h2>
          <p>
            Speech {ms(selectedTurn.speech.latencyMs)} ·{" "}
            {selectedTurn.judge.label} {ms(selectedTurn.judge.latencyMs)} · E2E{" "}
            {ms(fullTurnLatency(selectedTurn.metrics))}
          </p>
        </div>
        <div className="debug-console__badges">
          <StatusPill status={selectedTurn.speech.status} />
          <StatusPill status={selectedTurn.judge.status} />
          {selectedTurn.advanced && <Badge tone="good">advanced</Badge>}
          {selectedTurn.silentRescue.total > 0 && (
            <Badge tone="warn">
              silent-rescue={selectedTurn.silentRescue.total} (chain=
              {selectedTurn.silentRescue.chainLength})
            </Badge>
          )}
          {selectedTurn.silentRescue.state === "armed" && (
            <Badge tone="info">silent-timer armed</Badge>
          )}
        </div>
      </div>

      <SessionEventStrip events={sessionEvents} />

      <div className="debug-console__body">
        <TurnRail
          turns={turns}
          selectedTurnId={selectedTurn.turnId}
          onSelect={setSelectedTurnId}
        />
        <div className="trace-main">
          <ConversationPanel turn={selectedTurn} />
          <SilentRescuePanel turn={selectedTurn} />
          <JudgeDecisionPanel turn={selectedTurn} />
          <RawOutputPanel turn={selectedTurn} />
        </div>
        <PromptInspector turn={selectedTurn} />
      </div>
    </section>
  );
}
