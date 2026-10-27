import type { Lang } from "../i18n/strings.js";

interface CsrTraceEvent {
  topic: string;
  payload: Record<string, unknown>;
}

type ConversationRole = "patient" | "assistant";

interface ConversationMessage {
  id: string;
  role: ConversationRole;
  text: string;
}

interface CallRecordField {
  key: string;
  label: string;
  value: string;
  statusLabel: string;
  evidence: string;
  isFilled: boolean;
}

interface CallRecordSection {
  id: string;
  title: string;
  fields: CallRecordField[];
}

interface CallRecordForm {
  currentNode: string;
  pendingText: string;
  filledCount: number;
  totalCount: number;
  sections: CallRecordSection[];
}

interface PatientScenario {
  id: string;
  title: string;
  profile: string;
  clinicalContext: string;
  behavior: string;
  replyGuide: string[];
  tags: string[];
}

interface FieldDefinition {
  key: string;
  label: Record<Lang, string>;
}

interface SectionDefinition {
  id: string;
  title: Record<Lang, string>;
  fields: FieldDefinition[];
}

const SECTION_DEFINITIONS: SectionDefinition[] = [
  {
    id: "greeting",
    title: { zh: "开场确认", en: "Opening Consent" },
    fields: [
      {
        key: "call_accepted",
        label: { zh: "是否同意通话", en: "Call accepted" },
      },
    ],
  },
  {
    id: "identity",
    title: { zh: "身份确认", en: "Identity" },
    fields: [
      {
        key: "is_patient",
        label: { zh: "患者本人接听", en: "Patient answered" },
      },
      {
        key: "relationship",
        label: { zh: "接听人与患者关系", en: "Relationship" },
      },
    ],
  },
  {
    id: "symptom_rating",
    title: { zh: "症状与评分", en: "Symptoms" },
    fields: [
      {
        key: "symptom_change",
        label: { zh: "症状变化", en: "Symptom change" },
      },
      {
        key: "control_score",
        label: { zh: "症状控制评分", en: "Control score" },
      },
    ],
  },
  {
    id: "programming",
    title: { zh: "程控信息", en: "Programming" },
    fields: [
      {
        key: "programming_count",
        label: { zh: "程控次数", en: "Programming sessions" },
      },
      {
        key: "programming_satisfaction",
        label: { zh: "程控满意度", en: "Programming satisfaction" },
      },
    ],
  },
  {
    id: "education",
    title: { zh: "宣教落实", en: "Education" },
    fields: [
      {
        key: "pre_op_informed",
        label: { zh: "术前告知", en: "Pre-op briefing" },
      },
      {
        key: "device_explained",
        label: { zh: "设备说明", en: "Device explained" },
      },
      {
        key: "id_card_explained",
        label: { zh: "植入卡说明", en: "Implant card explained" },
      },
    ],
  },
  {
    id: "service",
    title: { zh: "医院服务", en: "Hospital Service" },
    fields: [
      {
        key: "response_speed",
        label: { zh: "响应速度", en: "Response speed" },
      },
      {
        key: "special_praise",
        label: { zh: "特别表扬", en: "Specific praise" },
      },
      {
        key: "special_complaint",
        label: { zh: "特别投诉", en: "Specific complaint" },
      },
    ],
  },
  {
    id: "insurance",
    title: { zh: "费用与医保", en: "Costs & Insurance" },
    fields: [
      {
        key: "insurance_type",
        label: { zh: "医保类型", en: "Insurance type" },
      },
      { key: "insurance_city", label: { zh: "参保城市", en: "Insured city" } },
      {
        key: "total_cost",
        label: { zh: "手术总费用", en: "Total surgery cost" },
      },
      { key: "self_pay", label: { zh: "自付金额", en: "Out-of-pocket cost" } },
      {
        key: "huimin_bao",
        label: { zh: "惠民保", en: "Supplemental local insurance" },
      },
    ],
  },
];

const NODE_LABELS: Record<string, Record<Lang, string>> = {
  greeting: { zh: "开场确认", en: "Opening consent" },
  identity: { zh: "身份确认", en: "Identity" },
  symptom_rating: { zh: "症状与评分", en: "Symptoms" },
  programming: { zh: "程控信息", en: "Programming" },
  education: { zh: "宣教落实", en: "Education" },
  service: { zh: "医院服务", en: "Hospital service" },
  insurance: { zh: "费用与医保", en: "Costs & insurance" },
  knowledge_qa: { zh: "知识问答", en: "Knowledge Q&A" },
  closing: { zh: "结束通话", en: "Closing" },
};

export const PATIENT_SCENARIOS: PatientScenario[] = [
  {
    id: "proxy-family",
    title: "家属代接",
    profile: "患者术后 3 个月，女儿代接电话，能说明大概恢复情况。",
    clinicalContext: "症状明显改善，但女儿不确定具体评分；做过程控 2 次。",
    behavior: "先说自己不是患者本人，回答时经常用「我爸说」转述。",
    replyGuide: [
      "身份环节明确说「我是他女儿」。",
      "症状环节只说「好多了」，等 AI 追问评分再给 8 分。",
      "费用环节表示不太清楚，让 AI 接受不确定信息。",
    ],
    tags: ["corner-case", "proxy", "sop"],
  },
  {
    id: "vague-score",
    title: "评分模糊",
    profile: "患者本人接听，愿意配合，但不习惯用数字评分。",
    clinicalContext: "术后震颤有所改善，最近一次程控后效果满意。",
    behavior: "连续使用「还行」「比以前强点」这类模糊回答。",
    replyGuide: [
      "先同意通话，再确认「我是本人」。",
      "被问症状时不要直接给分，只说「还行」。",
      "AI 解释 0 到 10 分后再说「那大概 7 分吧」。",
    ],
    tags: ["corner-case", "followup", "score"],
  },
  {
    id: "fee-sensitive",
    title: "费用敏感",
    profile: "患者本人接听，恢复不错，但对费用问题比较敏感。",
    clinicalContext: "有职工医保，自付金额记得不精确。",
    behavior: "前面配合，问到费用时反问为什么要问，随后只给大概范围。",
    replyGuide: [
      "服务体验说响应挺快。",
      "医保类型回答「职工医保」。",
      "自付金额先拒绝一次，AI 降低压力后说「大概两三万」。",
    ],
    tags: ["corner-case", "insurance", "sensitive"],
  },
  {
    id: "kb-detour",
    title: "中途问知识库",
    profile: "患者本人接听，对设备充电和核磁检查有疑问。",
    clinicalContext: "症状有所改善，但担心后续检查限制。",
    behavior: "在宣教或程控环节突然问「这个设备能不能做核磁」。",
    replyGuide: [
      "身份和症状正常回答。",
      "程控次数说 1 次且满意。",
      "中途主动问设备问题，观察 AI 是否先回答再回到随访。",
    ],
    tags: ["corner-case", "knowledge", "divert"],
  },
];

const FIELD_DEFINITION_BY_KEY = new Map(
  SECTION_DEFINITIONS.flatMap((section) =>
    section.fields.map((field) => [field.key, field] as const),
  ),
);

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function readString(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function readStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((item): item is string => typeof item === "string");
}

function parseJsonRecord(value: unknown): Record<string, unknown> | null {
  if (isRecord(value)) return value;
  const text = readString(value);
  if (!text.trim()) return null;
  try {
    const parsed = JSON.parse(text) as unknown;
    return isRecord(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function formatValue(value: unknown, status: string, lang: Lang): string {
  const normalizedStatus = normalizeStatus(status);
  if (normalizedStatus === "refused")
    return lang === "en" ? "Refused" : "已拒答";
  if (normalizedStatus === "not_applicable")
    return lang === "en" ? "N/A" : "不适用";
  if (typeof value === "boolean") {
    return value ? (lang === "en" ? "Yes" : "是") : lang === "en" ? "No" : "否";
  }
  if (typeof value === "number") return String(value);
  if (typeof value === "string" && value.trim()) return value.trim();
  if (value == null || value === "")
    return lang === "en" ? "Pending" : "待补充";
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function normalizeStatus(value: unknown): string {
  return readString(value).toLowerCase();
}

function statusLabel(status: string, isFilled: boolean, lang: Lang): string {
  if (!isFilled) return lang === "en" ? "Pending" : "待补充";
  switch (normalizeStatus(status)) {
    case "confirmed":
      return lang === "en" ? "Confirmed" : "已确认";
    case "inferred":
      return lang === "en" ? "Inferred" : "已推断";
    case "refused":
      return lang === "en" ? "Refused" : "已拒答";
    case "not_applicable":
      return lang === "en" ? "N/A" : "不适用";
    default:
      return lang === "en" ? "Recorded" : "已记录";
  }
}

function isResolvedValue(value: unknown, status: string): boolean {
  const normalizedStatus = normalizeStatus(status);
  if (normalizedStatus === "unknown") return false;
  if (normalizedStatus === "refused" || normalizedStatus === "not_applicable") {
    return true;
  }
  if (value === null || value === undefined || value === "") return false;
  return true;
}

function upsertMessage(
  messages: ConversationMessage[],
  next: ConversationMessage,
): ConversationMessage[] {
  const index = messages.findIndex((message) => message.id === next.id);
  if (index === -1) return [...messages, next];
  return messages.map((message, idx) => (idx === index ? next : message));
}

export function buildConversationMessages(
  events: readonly CsrTraceEvent[],
): ConversationMessage[] {
  let messages: ConversationMessage[] = [];

  for (const event of events) {
    const payload = event.payload;
    const turnId = readString(payload.turn_id) || `event-${messages.length}`;
    const eventName = readString(payload.event);

    if (event.topic === "dual_llm_debug" && eventName === "speech_start") {
      const text = readString(payload.asr_text);
      if (text) {
        messages = upsertMessage(messages, {
          id: `${turnId}:patient`,
          role: "patient",
          text,
        });
      }
      continue;
    }

    if (event.topic === "turn_metrics") {
      const text =
        readString(payload.asr_final_text) ||
        readString(payload.asr_text_at_emit) ||
        readString(payload.llm_actual_text);
      if (text) {
        messages = upsertMessage(messages, {
          id: `${turnId}:patient`,
          role: "patient",
          text,
        });
      }
      continue;
    }

    if (
      event.topic === "dual_llm_debug" &&
      (eventName === "speech_done" || eventName === "stage2_done")
    ) {
      const text = readString(payload.response_text);
      if (text) {
        messages = upsertMessage(messages, {
          id: `${turnId}:assistant:${eventName}`,
          role: "assistant",
          text,
        });
      }
    }
  }

  return messages.slice(-16);
}

interface FieldSnapshot {
  value: unknown;
  status: string;
  evidence: string;
}

function readFieldSnapshot(value: unknown): FieldSnapshot | null {
  if (!isRecord(value)) return null;
  return {
    value: value.value,
    status: readString(value.status),
    evidence: readString(value.evidence) || readString(value.evidence_span),
  };
}

function applyFieldSnapshot(
  fields: Map<string, FieldSnapshot>,
  field: string,
  snapshot: FieldSnapshot | null,
): void {
  if (!field || !snapshot) return;
  const previous = fields.get(field);
  fields.set(field, {
    ...snapshot,
    evidence: snapshot.evidence || previous?.evidence || "",
  });
}

function mergeFieldSnapshot(
  primary: FieldSnapshot | null,
  fallback: FieldSnapshot | null,
): FieldSnapshot | null {
  if (!primary) return fallback;
  return {
    ...primary,
    evidence: primary.evidence || fallback?.evidence || "",
  };
}

function latestNode(payload: Record<string, unknown>): string {
  return (
    readString(payload.node) ||
    readString(payload.node_id) ||
    readString(payload.to_node) ||
    ""
  );
}

function pendingFieldsFromPayload(payload: Record<string, unknown>): string[] {
  const policyMissing = readStringArray(payload.policy_missing);
  if (policyMissing.length > 0) return policyMissing;
  const missingFields = readStringArray(payload.missing_fields);
  if (missingFields.length > 0) return missingFields;
  const result = parseJsonRecord(payload.response_text);
  return result ? readStringArray(result.missing_fields) : [];
}

function fieldLabel(key: string, lang: Lang): string {
  return FIELD_DEFINITION_BY_KEY.get(key)?.label[lang] ?? key;
}

function buildPendingText(fields: string[], lang: Lang): string {
  if (fields.length === 0)
    return lang === "en" ? "Continue workflow" : "按流程继续";
  return fields.map((field) => fieldLabel(field, lang)).join(" / ");
}

function nodeLabel(node: string, lang: Lang): string {
  return NODE_LABELS[node]?.[lang] ?? node;
}

export function buildCallRecordForm(
  events: readonly CsrTraceEvent[],
  lang: Lang = "zh",
): CallRecordForm {
  const fieldSnapshots = new Map<string, FieldSnapshot>();
  let currentNode = "";
  let pendingFields: string[] = [];

  for (const event of events) {
    const payload = event.payload;
    const eventName = readString(payload.event);
    const node = latestNode(payload);
    if (node) currentNode = node;

    if (event.topic !== "dual_llm_debug") {
      continue;
    }

    const nextPendingFields = pendingFieldsFromPayload(payload);
    if (nextPendingFields.length > 0 || eventName === "advance_decision") {
      pendingFields = nextPendingFields;
    }

    if (eventName === "fast_judge_done" && isRecord(payload.commit_decision)) {
      const commits = payload.commit_decision.field_commits;
      if (Array.isArray(commits)) {
        for (const commit of commits) {
          if (!isRecord(commit)) continue;
          const field = readString(commit.field);
          const flowValue = readFieldSnapshot(commit.flow_value);
          const memoryValue = readFieldSnapshot(commit.memory_value);
          applyFieldSnapshot(
            fieldSnapshots,
            field,
            mergeFieldSnapshot(flowValue, memoryValue),
          );
        }
      }
      continue;
    }

    if (
      eventName === "advance_decision" &&
      isRecord(payload.collected_fields)
    ) {
      for (const [field, value] of Object.entries(payload.collected_fields)) {
        applyFieldSnapshot(fieldSnapshots, field, readFieldSnapshot(value));
      }
    }
  }

  const sections = SECTION_DEFINITIONS.map((section) => ({
    id: section.id,
    title: section.title[lang],
    fields: section.fields.map((field) => {
      const snapshot = fieldSnapshots.get(field.key);
      const status = snapshot?.status ?? "";
      const isFilled = snapshot
        ? isResolvedValue(snapshot.value, snapshot.status)
        : false;
      return {
        key: field.key,
        label: field.label[lang],
        value: snapshot
          ? formatValue(snapshot.value, status, lang)
          : lang === "en"
            ? "Pending"
            : "待补充",
        statusLabel: statusLabel(status, isFilled, lang),
        evidence: snapshot?.evidence ?? "",
        isFilled,
      };
    }),
  }));
  const fields = sections.flatMap((section) => section.fields);

  return {
    currentNode: nodeLabel(currentNode || "greeting", lang),
    pendingText: buildPendingText(pendingFields, lang),
    filledCount: fields.filter((field) => field.isFilled).length,
    totalCount: fields.length,
    sections,
  };
}

export function getNextPatientScenario(currentId: string): PatientScenario {
  const index = PATIENT_SCENARIOS.findIndex(
    (scenario) => scenario.id === currentId,
  );
  return PATIENT_SCENARIOS[(index + 1) % PATIENT_SCENARIOS.length];
}
