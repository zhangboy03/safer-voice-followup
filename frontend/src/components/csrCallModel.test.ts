import {
  buildCallRecordForm,
  buildConversationMessages,
  getNextPatientScenario,
  PATIENT_SCENARIOS,
} from "./csrCallModel.js";

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

const events = [
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_start",
      turn_id: "turn-1",
      ts: 1_789_000_000,
      asr_text: "我是患者女儿，他最近好多了。",
      node: "identity",
    },
  },
  {
    topic: "turn_metrics",
    payload: {
      turn_id: "turn-1",
      asr_final_text: "我是患者女儿，他最近好多了。",
      llm_actual_text: "我是患者女儿，他最近好多了。",
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "speech_done",
      turn_id: "turn-1",
      ts: 1_789_000_001,
      response_text: "好的，那我简单确认一下您和患者的关系。",
      node: "identity",
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "fast_judge_done",
      turn_id: "turn-1",
      ts: 1_789_000_002,
      node: "identity",
      response_text: JSON.stringify({
        field_delta: [
          {
            field: "is_patient",
            value: false,
            status: "confirmed",
            evidence_span: "患者女儿",
            confidence: 0.96,
          },
          {
            field: "relationship",
            value: "女儿",
            status: "confirmed",
            evidence_span: "女儿",
            confidence: 0.98,
          },
        ],
        missing_fields: ["control_score"],
      }),
      commit_decision: {
        field_commits: [
          {
            field: "is_patient",
            memory_value: {
              value: false,
              status: "confirmed",
              evidence: "患者女儿",
            },
          },
          {
            field: "relationship",
            flow_value: { value: "女儿", status: "confirmed" },
          },
        ],
      },
    },
  },
  {
    topic: "dual_llm_debug",
    payload: {
      event: "advance_decision",
      turn_id: "turn-1",
      node: "identity",
      collected_fields: {
        is_patient: { value: false, status: "confirmed" },
        relationship: { value: "女儿", status: "confirmed" },
      },
      policy_missing: ["control_score"],
    },
  },
] as const;

const messages = buildConversationMessages(events);

assertEqual(messages.length, 2, "deduplicated conversation should have 2 rows");
assertEqual(messages[0].role, "patient", "patient turn should be first");
assertEqual(
  messages[0].text,
  "我是患者女儿，他最近好多了。",
  "patient text should prefer final ASR",
);
assertEqual(messages[1].role, "assistant", "assistant turn should be second");

const callRecordForm = buildCallRecordForm(events);
assertEqual(
  callRecordForm.totalCount,
  18,
  "call record form should cover every handoff field",
);
assertEqual(
  callRecordForm.filledCount,
  2,
  "call record form should count committed fields",
);
assertEqual(
  callRecordForm.currentNode,
  "身份确认",
  "call record form should expose the localized latest node",
);
assert(
  callRecordForm.pendingText.includes("症状控制评分"),
  "call record form should show missing fields using friendly labels",
);
const identitySection = callRecordForm.sections.find(
  (section) => section.id === "identity",
);
assert(identitySection, "identity section should exist");
const isPatient = identitySection.fields.find(
  (field) => field.key === "is_patient",
);
const relationship = identitySection.fields.find(
  (field) => field.key === "relationship",
);
assert(isPatient, "is_patient field should exist");
assert(relationship, "relationship field should exist");
assertEqual(
  isPatient.value,
  "否",
  "form should render committed boolean field values",
);
assertEqual(
  relationship.value,
  "女儿",
  "form should render committed relationship values",
);
assert(isPatient.isFilled, "committed boolean field should be marked filled");
assert(
  relationship.isFilled,
  "committed relationship field should be marked filled",
);

const englishCallRecordForm = buildCallRecordForm(events, "en");
assertEqual(
  englishCallRecordForm.currentNode,
  "Identity",
  "English form should localize the current node",
);
assert(
  englishCallRecordForm.pendingText.includes("Control score"),
  "English form should show missing fields using localized labels",
);
const englishIdentitySection = englishCallRecordForm.sections.find(
  (section) => section.id === "identity",
);
assert(englishIdentitySection, "English identity section should exist");
assertEqual(
  englishIdentitySection.title,
  "Identity",
  "English form should localize section titles",
);
const englishIsPatient = englishIdentitySection.fields.find(
  (field) => field.key === "is_patient",
);
assert(englishIsPatient, "English is_patient field should exist");
assertEqual(
  englishIsPatient.label,
  "Patient answered",
  "English form should localize field labels",
);
assertEqual(
  englishIsPatient.value,
  "No",
  "English form should localize boolean values",
);
assertEqual(
  englishIsPatient.statusLabel,
  "Confirmed",
  "English form should localize status labels",
);

assert(
  PATIENT_SCENARIOS.some((scenario) => scenario.tags.includes("corner-case")),
  "scenario set should include corner-case coverage",
);
assertEqual(
  getNextPatientScenario(PATIENT_SCENARIOS[0].id).id,
  PATIENT_SCENARIOS[1].id,
  "next scenario should rotate deterministically",
);
