import {
  buildFeedbackDraft,
  buildFeedbackListItems,
  buildFeedbackTurnOptions,
} from "./feedbackModel.js";

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

const draft = buildFeedbackDraft({
  csrId: "csr-01",
  roomName: "debug-full",
  source: "csr_simple",
  events: [
    {
      topic: "transcription",
      payload: { topic: "transcription", text: "我姓张" },
    },
    {
      topic: "agent_text",
      payload: { topic: "agent_text", delta: "好的，", is_final: false },
    },
    {
      topic: "agent_text",
      payload: {
        topic: "agent_text",
        delta: "我再确认一下您的姓名。",
        is_final: false,
      },
    },
    {
      topic: "turn_metrics",
      payload: {
        topic: "turn_metrics",
        turn_id: "turn-7",
        asr_final_text: "我姓张",
        llm_actual_text: "我姓张",
        e2e_ms: 1820,
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event: "fast_judge_done",
        node_id: "identity",
        response_text: '{"node_action":"advance"}',
      },
    },
  ],
});

assertEqual(draft.csr_id, "csr-01", "csr id should be attached");
assertEqual(draft.room_name, "debug-full", "room name should be attached");
assertEqual(
  draft.turn.user_text,
  "我姓张",
  "latest user text should be attached",
);
assertEqual(
  draft.turn.assistant_text,
  "好的，我再确认一下您的姓名。",
  "assistant text should accumulate streamed deltas",
);
assertEqual(draft.turn.turn_id, "turn-7", "turn id should come from metrics");
assertEqual(
  draft.turn.node_id,
  "identity",
  "node id should come from debug payload",
);
assertEqual(
  draft.evidence.turn_metrics?.e2e_ms,
  1820,
  "latest turn metrics should be preserved",
);
assertEqual(
  draft.evidence.debug?.event,
  "fast_judge_done",
  "latest debug payload should be preserved",
);

const items = buildFeedbackListItems([
  {
    id: "fb-2",
    created_at: "2026-05-12T10:02:00Z",
    csr_id: "csr-02",
    room_name: "debug-full",
    source: "debug",
    note: "第二条",
    turn: { assistant_text: "回复二" },
    evidence: {},
  },
  {
    id: "fb-1",
    created_at: "2026-05-12T10:01:00Z",
    csr_id: "csr-01",
    room_name: "debug-full",
    source: "csr_simple",
    note: "第一条",
    turn: { assistant_text: "回复一" },
    evidence: {},
  },
]);

assert(items[0].title.includes("csr-02"), "newest item should be first");
assert(
  items[0].subtitle.includes("debug-full"),
  "room name should be visible in list item",
);

const manyEventsDraft = buildFeedbackDraft({
  csrId: "csr-01",
  roomName: "debug-full",
  source: "csr_simple",
  events: Array.from({ length: 45 }, (_, index) => ({
    topic: "dual_llm_debug",
    payload: {
      event: "speech_start",
      turn_id: `turn-${index}`,
      asr_text: `用户第 ${index} 句`,
    },
  })),
});

assertEqual(
  manyEventsDraft.evidence.recent_events?.length,
  45,
  "feedback evidence should keep the full available event context",
);

const stage2Draft = buildFeedbackDraft({
  csrId: "csr-01",
  roomName: "debug-full",
  source: "csr_simple",
  events: [
    {
      topic: "dual_llm_debug",
      payload: {
        event: "speech_done",
        turn_id: "turn-9",
        response_text: "好的张先生，",
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        event: "stage2_done",
        turn_id: "turn-9",
        response_text: "我再确认一下您是本人吗？",
      },
    },
  ],
});

assertEqual(
  stage2Draft.turn.assistant_text,
  "好的张先生，我再确认一下您是本人吗？",
  "feedback turn text should include same-turn stage 2 continuation",
);

const multiTurnEvents = [
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_start",
      turn_id: "turn-1",
      node: "identity",
      asr_text: "我是他女儿",
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_done",
      turn_id: "turn-1",
      node: "identity",
      response_text: "好的，我确认一下您和患者的关系。",
    },
  },
  {
    topic: "turn_metrics",
    payload: {
      topic: "turn_metrics",
      turn_id: "turn-1",
      asr_final_text: "我是他女儿",
      e2e_ms: 1000,
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "fast_judge_done",
      turn_id: "turn-1",
      node_id: "identity",
      response_text: '{"node_action":"advance"}',
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_start",
      turn_id: "turn-2",
      node: "symptom_rating",
      asr_text: "恢复好多了",
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_done",
      turn_id: "turn-2",
      node: "symptom_rating",
      response_text: "那您现在症状大概能打几分？",
    },
  },
  {
    topic: "turn_metrics",
    payload: {
      topic: "turn_metrics",
      turn_id: "turn-2",
      asr_final_text: "恢复好多了",
      e2e_ms: 1200,
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "fast_judge_done",
      turn_id: "turn-2",
      node_id: "symptom_rating",
      response_text: '{"node_action":"followup"}',
    },
  },
];

const turnOptions = buildFeedbackTurnOptions(multiTurnEvents);

assertEqual(turnOptions.length, 2, "feedback should expose both turn choices");
assertEqual(turnOptions[0].turn_id, "turn-1", "first choice should be turn 1");
assert(
  turnOptions[0].label.includes("第 1 轮"),
  "turn choice should show a human round index",
);

const selectedFirstTurnDraft = buildFeedbackDraft({
  csrId: "csr-01",
  roomName: "debug-full",
  source: "debug",
  events: multiTurnEvents,
  selectedTurnId: "turn-1",
});

assertEqual(
  selectedFirstTurnDraft.turn.turn_id,
  "turn-1",
  "selected draft should keep the chosen turn id",
);
assertEqual(
  selectedFirstTurnDraft.turn.user_text,
  "我是他女儿",
  "selected draft should use the chosen turn user text",
);
assertEqual(
  selectedFirstTurnDraft.turn.assistant_text,
  "好的，我确认一下您和患者的关系。",
  "selected draft should use the chosen turn assistant text",
);
assertEqual(
  selectedFirstTurnDraft.turn.node_id,
  "identity",
  "selected draft should use the chosen turn node",
);
assertEqual(
  selectedFirstTurnDraft.evidence.turn_metrics?.e2e_ms,
  1000,
  "selected draft should preserve the chosen turn metrics",
);
assertEqual(
  selectedFirstTurnDraft.evidence.debug?.node_id,
  "identity",
  "selected draft should preserve the chosen turn debug payload",
);
