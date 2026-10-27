import {
  buildPromptTranscript,
  extractFastJudgeContext,
  summarizeFastJudgePayload,
} from "./dualLlmTraceModel.js";
import { buildFastJudgeTraceFixtureEvents } from "./traceFixtureData.js";

function assert(condition: unknown, message: string): asserts condition {
  if (!condition) {
    throw new Error(message);
  }
}

function assertEqual<T>(actual: T, expected: T, message: string) {
  if (actual !== expected) {
    throw new Error(
      `${message}: expected ${String(expected)}, got ${String(actual)}`,
    );
  }
}

const fastJudgeContext = extractFastJudgeContext([
  {
    role: "system",
    content: "static rules",
  },
  {
    role: "system",
    content:
      '<context>{"speech_text":"好的，我们先确认身份。","pending_fields_before_turn":["call_accepted"],"known_fields_before_turn":{"patient_name":{"value":"张三","status":"confirmed"}},"successor_path":["identity"]}</context>',
  },
  {
    role: "user",
    content: "可以，我是张三。",
  },
]);

assert(fastJudgeContext, "Fast Judge context should be parsed from <context>");
assertEqual(
  fastJudgeContext.speech_text,
  "好的，我们先确认身份。",
  "Fast Judge context should expose speech_text",
);
assertEqual(
  Array.isArray(fastJudgeContext.pending_fields_before_turn),
  true,
  "Fast Judge context should expose pending fields",
);

const summary = summarizeFastJudgePayload({
  response_text: JSON.stringify({
    schema_version: "fast_judge.v1",
    turn_id: "turn-1",
    node_id: "greeting",
    speech_result_id: "turn-1:stage1",
    evaluated_until: "speech_text_done",
    field_delta: [
      {
        field: "call_accepted",
        value: "confirmed",
        status: "confirmed",
        evidence_span: "可以",
        confidence: 0.98,
      },
    ],
    lookahead_field_delta: [
      {
        node_id: "identity",
        field: "patient_name",
        value: "张三",
        status: "confirmed",
        evidence_span: "我是张三",
        confidence: 0.96,
      },
    ],
    missing_fields: ["is_patient"],
    stage_complete: true,
    node_action: "advance",
    node_after_candidate: "identity",
    next_node_goal: "确认患者身份",
    stage2_should_start: true,
    stage2_target_node: "identity",
    stage2_reason: "greeting_resolved",
    answer_assessment: {
      current_question_answered: "answered",
      needs_followup: false,
      followup_reason: "",
      missing_fields: ["is_patient"],
    },
    handoff: {
      transfer_to_human: false,
      reason: "none",
      risk_level: "normal",
    },
    knowledge_question: {
      detected: false,
      query: "",
    },
    speech_directive: {
      code: "advance_to_identity",
      text: "用户已同意通话并说出姓名，下一句请简短承接并确认是否患者本人。",
    },
  }),
  stage2_should_start: true,
  stage2_target_node: "identity",
  stage2_reason: "greeting_resolved",
  cache_usage: {
    input_tokens: 401,
    output_tokens: 91,
  },
});

assertEqual(
  summary.schemaVersion,
  "fast_judge.v1",
  "schema version should be surfaced",
);
assertEqual(summary.fieldDelta.length, 1, "field_delta should be surfaced");
assertEqual(
  summary.lookaheadFieldDelta.length,
  1,
  "lookahead_field_delta should be surfaced",
);
assertEqual(
  summary.missingFields[0],
  "is_patient",
  "string missing_fields should be surfaced",
);
assertEqual(
  summary.nodeAction,
  "advance",
  "node action should come from raw Fast Judge result",
);
assertEqual(
  summary.stage2ShouldStart,
  true,
  "stage2_should_start should be surfaced",
);
assertEqual(
  summary.stage2TargetNode,
  "identity",
  "stage2 target should be surfaced",
);
assertEqual(
  summary.answerAssessment?.current_question_answered,
  "answered",
  "answer_assessment should be surfaced",
);
assertEqual(
  summary.speechDirective?.code,
  "advance_to_identity",
  "speech_directive should be surfaced",
);
assertEqual(
  summary.cacheUsage?.input_tokens,
  401,
  "cache usage should be surfaced",
);

const fixtureEvents = buildFastJudgeTraceFixtureEvents();
const fixtureTopics = fixtureEvents.map((event) => event.topic);
assert(
  fixtureTopics.includes("turn_metrics"),
  "fixture should include real turn_metrics topic",
);
assert(
  fixtureTopics.includes("dual_llm_debug"),
  "fixture should include real dual_llm_debug topic",
);

const fixtureJudgeDone = fixtureEvents.find(
  (event) =>
    event.topic === "dual_llm_debug" &&
    event.payload.event === "fast_judge_done",
);
assert(fixtureJudgeDone, "fixture should include fast_judge_done");

const fixtureSummary = summarizeFastJudgePayload(fixtureJudgeDone.payload);
assertEqual(
  fixtureSummary.schemaVersion,
  "fast_judge.v1",
  "fixture raw response should use current Fast Judge schema",
);
assert(
  fixtureSummary.speechDirective?.text,
  "fixture should show the backend speech_directive surface",
);

const promptTranscript = buildPromptTranscript([
  {
    role: "system",
    content: "规则 A",
  },
  {
    role: "user",
    content: "用户说法",
  },
  {
    role: "assistant",
    content: "上一句回复",
  },
]);

assertEqual(
  promptTranscript,
  "规则 A\n\n用户说法\n\n上一句回复",
  "prompt transcript should flatten model-visible messages without role numbering",
);

// ── fast judge prompt v2: reason_source surface ──────────────────────────────
// llm_processor._emit_debug("fast_judge_done", ...) ships reason_source as a
// top-level kwarg AND a nested judge_result.answer_assessment.reason_source.
// response_text holds the raw pre-mapping LLM output and does NOT carry
// reason_source — Codex caught this gap; mirror the real payload shape here.

function buildPayloadWithTopLevelReasonSource(value: unknown) {
  const payload: Record<string, unknown> = {
    response_text: JSON.stringify({ schema_version: "fast_judge.v1" }),
    judge_result: {
      schema_version: "fast_judge.v1",
      answer_assessment: { followup_reason: "..." },
    },
  };
  if (value !== undefined) {
    payload.reason_source = value;
  }
  return payload;
}

assertEqual(
  summarizeFastJudgePayload(buildPayloadWithTopLevelReasonSource("model"))
    .reasonSource,
  "model",
  "reasonSource=model should be read from top-level event kwarg",
);
assertEqual(
  summarizeFastJudgePayload(buildPayloadWithTopLevelReasonSource("fallback"))
    .reasonSource,
  "fallback",
  "reasonSource=fallback should be surfaced for observability",
);
assertEqual(
  summarizeFastJudgePayload(buildPayloadWithTopLevelReasonSource(undefined))
    .reasonSource,
  "absent",
  "reasonSource defaults to absent when missing (v1 legacy / pre-mapping)",
);
assertEqual(
  summarizeFastJudgePayload(buildPayloadWithTopLevelReasonSource("bogus-value"))
    .reasonSource,
  "absent",
  "reasonSource enum guard: unknown value coerces to absent, not propagated",
);

// Fallback path: judge_result.answer_assessment.reason_source when top-level
// kwarg is absent (older logs / partial replay scenarios).
assertEqual(
  summarizeFastJudgePayload({
    response_text: JSON.stringify({ schema_version: "fast_judge.v1" }),
    judge_result: {
      schema_version: "fast_judge.v1",
      answer_assessment: { reason_source: "fallback" },
    },
  }).reasonSource,
  "fallback",
  "reasonSource should also surface via judge_result.answer_assessment when top-level kwarg is missing",
);
