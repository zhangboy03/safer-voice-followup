import type { JsonMap } from "./dualLlmTraceModel.js";

interface TraceFixtureEvent {
  topic: string;
  payload: JsonMap;
}

const turnId = "fixture-turn-001";
const nowSeconds = 1_789_000_000;

export function buildFastJudgeTraceFixtureEvents(): TraceFixtureEvent[] {
  const fastJudgeResponse = {
    schema_version: "fast_judge.v1",
    turn_id: turnId,
    node_id: "greeting",
    speech_result_id: `${turnId}:stage1`,
    evaluated_until: "speech_text_done",
    field_delta: [
      {
        field: "call_accepted",
        value: "confirmed",
        status: "confirmed",
        evidence_span: "可以，您说",
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
        confidence: 0.93,
      },
    ],
    missing_fields: ["is_patient"],
    stage_complete: true,
    node_action: "advance",
    node_after_candidate: "identity",
    next_node_goal: "确认接听人是否患者本人",
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
      text: "用户已同意通话并说出姓名。下一句请简短承接，并确认是否患者本人。",
    },
  };

  const promptContext = {
    turn_id: turnId,
    turn_index: 1,
    node_id: "greeting",
    speech_result_id: `${turnId}:stage1`,
    speech_text:
      "您好，这里是品驰医疗，想做一个术后随访问卷，方便占用您一分钟吗？",
    user_text: "可以，您说，我是张三。",
    speech_asked_this_turn: "确认是否方便接听",
    intended_ask_fields: ["call_accepted"],
    required_fields: ["call_accepted"],
    eligible_optional_fields: [],
    successor_path: ["identity"],
    pending_fields_before_turn: ["call_accepted"],
    known_fields_before_turn: {},
    recent_relevant_context: "",
  };

  return [
    {
      topic: "turn_metrics",
      payload: {
        topic: "turn_metrics",
        event_type: "TurnMetricsEvent",
        turn_id: turnId,
        smart_turn_ms: 15,
        vad_tail_wait_ms: 200,
        user_stop_to_emit_ms: 117,
        gate_wait_ms: 102,
        asr_text_at_emit: "可以，您说，我是张三。",
        asr_text_type: "final",
        asr_final_text: "可以，您说，我是张三。",
        asr_final_latency_ms: 2,
        asr_diff: "",
        emit_to_llm_ms: 7,
        llm_ttft_ms: 186,
        llm_actual_text: "可以，您说，我是张三。",
        tts_session_create_ms: 0,
        tts_pool_hit: true,
        tts_ttfa_ms: 243,
        shadow_results: null,
        emit_to_audio_ms: 446,
        output_queue_wait_ms: 5,
        e2e_ms: 568,
        speech_end_to_audio_ms: 768,
        first_audio_source: "speech",
        first_audio_is_bridge: false,
        unexplained_gap_ms: 10,
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "speech_start",
        turn_id: turnId,
        ts: nowSeconds,
        node: "greeting",
        asr_text: "可以，您说，我是张三。",
        is_speculative: false,
        prompt_messages: [
          {
            role: "system",
            content: "Speech prompt static rules",
          },
          {
            role: "user",
            content: "可以，您说，我是张三。",
          },
        ],
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "speech_done",
        turn_id: turnId,
        ts: nowSeconds + 0.2,
        node: "greeting",
        response_text: "好的张三，我们先确认一下，您是患者本人吗？",
        latency_ms: 219,
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "fast_judge_start",
        turn_id: turnId,
        ts: nowSeconds + 0.25,
        node: "greeting",
        asr_text: "可以，您说，我是张三。",
        assistant_text: "好的张三，我们先确认一下，您是患者本人吗？",
        speech_result_id: `${turnId}:stage1`,
        prompt_messages: [
          {
            role: "system",
            content:
              "You are the Fast SOP Judge. Output only fast_judge.v1 JSON.",
          },
          {
            role: "system",
            content: `<context>${JSON.stringify(promptContext)}</context>`,
          },
          {
            role: "user",
            content: "可以，您说，我是张三。",
          },
        ],
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "fast_judge_done",
        turn_id: turnId,
        ts: nowSeconds + 0.51,
        node: "greeting",
        response_text: JSON.stringify(fastJudgeResponse),
        stage_complete: true,
        transfer_to_human: false,
        knowledge_query: false,
        node_action: "advance",
        reason: "",
        stage2_should_start: true,
        stage2_target_node: "identity",
        stage2_reason: "greeting_resolved",
        cache_usage: {
          input_tokens: 622,
          output_tokens: 126,
          cache_read_input_tokens: 480,
        },
        commit_decision: {
          decision_id: `fast_judge:${turnId}:${turnId}:stage1:greeting`,
          accepted: true,
          node_before: "greeting",
          node_after: "greeting",
          node_action: "advance",
          idempotent: false,
          reject_reason: "",
          append_allowed_candidate: true,
          continuation_prefix: "",
          field_commits: [
            {
              field: "call_accepted",
              flow_value: {
                value: "confirmed",
                status: "confirmed",
              },
              memory_value: {
                value: "confirmed",
                status: "confirmed",
                evidence: "可以，您说",
              },
            },
          ],
          next_speech_strategy: {
            strategy_id: `${turnId}:fast_strategy`,
            node_action: "advance",
          },
          followup_missing_fields: ["is_patient"],
          stage2_should_start: true,
          stage2_target_node: "identity",
          stage2_reason: "greeting_resolved",
          speech_risks: [],
        },
        turn_assessment: {
          recommended_action: "advance",
          stage_complete: true,
          reason: "",
        },
        latency_ms: 256,
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "advance_decision",
        turn_id: turnId,
        ts: nowSeconds + 0.55,
        node: "greeting",
        judge_stage_complete: true,
        policy_satisfied: true,
        policy_missing: [],
        judge_ran: true,
        will_advance: true,
        collected_fields: {
          call_accepted: {
            value: "confirmed",
            status: "confirmed",
          },
        },
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "stage_advance",
        turn_id: turnId,
        ts: nowSeconds + 0.58,
        old_node: "greeting",
        new_node: "identity",
        reason: "advance",
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "stage2_start",
        turn_id: turnId,
        ts: nowSeconds + 0.62,
        node: "identity",
        separate_playout: false,
        prompt_messages: [
          {
            role: "system",
            content:
              "Judge给Speech的下一句指令（advance_to_identity）：用户已同意通话并说出姓名。下一句请简短承接，并确认是否患者本人。",
          },
        ],
      },
    },
    {
      topic: "dual_llm_debug",
      payload: {
        topic: "dual_llm_debug",
        event_type: "JudgeDecisionEvent",
        event: "stage2_done",
        turn_id: turnId,
        ts: nowSeconds + 0.78,
        node: "identity",
        response_text: "好的张三，确认一下您是患者本人吗？",
        latency_ms: 162,
      },
    },
  ];
}
