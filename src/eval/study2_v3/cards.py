"""Study 2 v3 patient card schema, milestone derivation, and validator.

Cards move from v2's per-turn fixed utterances to a persona + per-node fact
sheet + per-field disclosure policy. The expected SOP "milestones" (coverage
gold) are derived deterministically from the disclosure policy so the LLM only
ever verbalizes — it never decides the gold trace.

See spec: docs/superpowers/specs/2026-06-01-study2-simulated-patient-design.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Policy = Literal["volunteer", "on_ask", "on_followup", "refuse"]

SOP_ORDER = [
    "greeting",
    "identity",
    "symptom_rating",
    "programming",
    "education",
    "service",
    "insurance",
    "closing",
]


@dataclass
class FieldSpec:
    """One collectable field plus how/when the patient is allowed to reveal it."""

    name: str
    policy: Policy
    gold_value: Any
    node: str
    status: str = "confirmed"
    scoring: str = "required"  # required | optional


@dataclass
class Card:
    case_id: str
    case_type: str
    persona: dict[str, Any]
    node_facts: dict[str, str]
    fields: list[FieldSpec]

    def fields_for(self, node: str) -> list[FieldSpec]:
        return [f for f in self.fields if f.node == node]


@dataclass
class Milestone:
    node: str
    required_fields: list[str]  # required and non-refused
    requires_followup: bool
    refused_fields: list[str]
    terminal: bool = False


def derive_milestones(card: Card) -> list[Milestone]:
    """Deterministically derive the coverage gold from the disclosure policy.

    Invariant: policy only governs WHEN a field is revealed, not whether it is
    required. A node's advance milestone = all its required non-refused fields
    collected. on_followup fields force requires_followup because the first
    utterance only yields a vague value.
    """
    out: list[Milestone] = []
    for node in SOP_ORDER:
        fs = card.fields_for(node)
        # greeting / closing always appear as control milestones even without fields.
        if not fs and node not in ("greeting", "closing"):
            continue
        req = [f for f in fs if f.scoring == "required" and f.policy != "refuse"]
        refused = [f for f in fs if f.policy == "refuse"]
        out.append(
            Milestone(
                node=node,
                required_fields=[f.name for f in req],
                requires_followup=any(f.policy == "on_followup" for f in req),
                refused_fields=[f.name for f in refused],
                terminal=(node == "closing"),
            )
        )
    return out


def validate_card(card: Card) -> list[str]:
    """Return a list of human-readable errors; empty means the card is valid."""
    errors: list[str] = []

    for f in card.fields:
        facts = card.node_facts.get(f.node, "")
        if f.scoring == "required" and f.policy != "refuse" and not facts.strip():
            errors.append(f"{f.name}: required 字段所属节点 {f.node} 无 node_facts 证据")

    milestones = derive_milestones(card)
    nodes = [m.node for m in milestones]
    idx = [SOP_ORDER.index(n) for n in nodes]
    if idx != sorted(idx):
        errors.append("milestone 节点序列不符合 SOP 顺序")

    for m in milestones:
        fs = card.fields_for(m.node)
        req_like = [f for f in fs if f.scoring == "required"]
        if req_like and all(f.policy == "refuse" for f in req_like):
            errors.append(f"{m.node}: required 字段全为 refused，无可推进字段")

    return errors


# --------------------------------------------------------------------------- #
# Card skeletons (4 case types, reusing v2 field values, expressed in v3 schema)
# --------------------------------------------------------------------------- #


def _proxy_card() -> Card:
    return Card(
        case_id="s2_v3_proxy_missing_followups_001",
        case_type="proxy_mainline_missing_parameter",
        persona={
            "who": "患者女儿代接",
            "temperament": "爱絮叨，先扯一段家长里短和自己的辛苦再说正事，很多节点先只答一半，被追问后补齐",
            "chatter": [
                "自己上班忙、请假难，两头跑医院",
                "妈妈年纪大不肯麻烦人、特别犟",
                "家里还有孩子要管、自己也顾不过来",
                "扯到做饭、接送、最近没睡好",
            ],
            "emotional_color": "操心、有点累但孝顺，爱跟人念叨自己不容易",
            "sensitivities": [],
        },
        node_facts={
            "greeting": "现在方便接听。",
            "identity": "接电话的是患者女儿，今天替母亲接听。",
            "symptom_rating": "母亲术后走路比以前稳一点，整体有所改善；满分十分自评 8 分。",
            "programming": "还没做过程控，一次都没做过。",
            "education": "设备具体怎么用母亲没太听明白。",
            "service": "护士和医生都挺负责；有问题打电话基本当天能回，响应挺快。",
            "insurance": "母亲是北京城镇职工医保；报销后自付大概五万左右。",
            "closing": "没有其他问题了。",
        },
        fields=[
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "volunteer", False, node="identity"),
            FieldSpec("relationship", "volunteer", "女儿", node="identity"),
            FieldSpec("symptom_change", "volunteer", "有所改善", node="symptom_rating"),
            FieldSpec("control_score", "on_followup", 8, node="symptom_rating"),
            FieldSpec("programming_count", "volunteer", 0, node="programming"),
            FieldSpec("device_explained", "on_followup", False, node="education"),
            FieldSpec("special_praise", "on_ask", "护士和医生都挺负责", node="service", scoring="optional"),
            FieldSpec("response_speed", "on_followup", "快速响应", node="service"),
            FieldSpec("insurance_type", "volunteer", "城镇职工医保", node="insurance"),
            FieldSpec("insurance_city", "on_ask", "北京", node="insurance", scoring="optional"),
            FieldSpec("self_pay", "on_followup", "五万左右", node="insurance"),
        ],
    )


def _complete_card() -> Card:
    return Card(
        case_id="s2_v3_patient_complete_context_002",
        case_type="self_patient_complete_context",
        persona={
            "who": "患者本人接听",
            "temperament": "开朗健谈，配合度高，爱拉家常，但每个问题最终都一次性给出完整信息",
            "chatter": [
                "退休生活、跳广场舞、在家养花种菜",
                "老伴儿、孙子、邻居家长里短",
                "感慨现在医疗条件比以前好太多",
                "扯到天气、最近吃得好不好睡得香",
            ],
            "emotional_color": "知足、热情、把客服当人唠嗑",
            "sensitivities": [],
        },
        node_facts={
            "greeting": "现在方便接听。",
            "identity": "接电话的就是患者本人。",
            "symptom_rating": "比手术前明显好多了，自评九分。",
            "programming": "做过两次程控，效果挺满意。",
            "education": "设备怎么用教过。",
            "service": "医院回复挺快，没什么投诉。",
            "insurance": "北京职工医保，报销后自己花了五万多。",
            "closing": "没有其他问题了。",
        },
        fields=[
            # Cooperative patient persona: volunteers complete info at each node.
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "volunteer", True, node="identity"),
            FieldSpec("symptom_change", "volunteer", "明显改善", node="symptom_rating"),
            FieldSpec("control_score", "volunteer", 9, node="symptom_rating"),
            FieldSpec("programming_count", "volunteer", 2, node="programming"),
            FieldSpec("programming_satisfaction", "volunteer", "满意", node="programming"),
            FieldSpec("device_explained", "volunteer", True, node="education"),
            FieldSpec("response_speed", "volunteer", "快速响应", node="service"),
            FieldSpec("insurance_type", "volunteer", "职工医保", node="insurance"),
            FieldSpec("insurance_city", "volunteer", "北京", node="insurance", scoring="optional"),
            FieldSpec("self_pay", "volunteer", "五万多", node="insurance"),
        ],
    )


def _ambiguous_card() -> Card:
    return Card(
        case_id="s2_v3_ambiguous_repair_003",
        case_type="ambiguous_then_repair",
        persona={
            "who": "患者本人接听",
            "temperament": "上了年纪、记性差、特别墨迹，半天说不到点子上，先给模糊回答，被追问后才给清楚信息",
            "chatter": [
                "总说自己年纪大了、脑子转得慢、记不清",
                "看病流程绕、搞不太明白、到处问人",
                "扯到买菜、天气、晚上睡不好、腿脚",
                "一件事来回重复念叨好几遍",
            ],
            "emotional_color": "絮叨、容易跑题、需要客服很有耐心地带",
            "sensitivities": [],
        },
        node_facts={
            "greeting": "现在可以接听。",
            "identity": "接电话的是患者本人。",
            "symptom_rating": "比以前强一点，算有所改善；满分十分大概七分。",
            "programming": "做过一次程控，效果还可以，算满意。",
            "education": "设备具体怎么用没太听明白。",
            "service": "找医院的时候回复还挺快。",
            "insurance": "居民医保；自己花了三万左右。",
            "closing": "没有别的事了。",
        },
        fields=[
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "volunteer", True, node="identity"),
            FieldSpec("symptom_change", "volunteer", "有所改善", node="symptom_rating"),
            FieldSpec("control_score", "on_followup", 7, node="symptom_rating"),
            FieldSpec("programming_count", "on_followup", 1, node="programming"),
            FieldSpec("programming_satisfaction", "on_followup", "满意", node="programming"),
            FieldSpec("device_explained", "volunteer", False, node="education"),
            FieldSpec("response_speed", "volunteer", "快速响应", node="service"),
            FieldSpec("insurance_type", "volunteer", "居民医保", node="insurance"),
            FieldSpec("self_pay", "on_followup", "三万左右", node="insurance"),
        ],
    )


def _refusal_card() -> Card:
    return Card(
        case_id="s2_v3_refusal_sensitive_cost_004",
        case_type="sensitive_field_refusal",
        persona={
            "who": "患者本人接听",
            "temperament": "话多爱哭穷，愿意回答大部分问题，但一碰到具体费用金额就岔开话题、绝不透露自付",
            "chatter": [
                "物价贵、退休金不经花、什么都涨价",
                "儿女负担重、不想给孩子添麻烦",
                "扯到邻居谁谁看病花了多少、感慨看病难",
                "一旦话到自己花了多少钱就赶紧扯别的",
            ],
            "emotional_color": "节俭、爱抱怨开销，但对自己的具体数目讳莫如深",
            "sensitivities": ["self_pay"],
        },
        node_facts={
            "greeting": "现在方便接听。",
            "identity": "接电话的就是患者本人。",
            "symptom_rating": "没什么变化，自评最多四分。",
            "programming": "还没做过程控。",
            "education": "设备怎么用讲过。",
            "service": "有时候回复有点慢。",
            "insurance": "职工医保；但自付金额不方便说。",
            "closing": "没什么问题了。",
        },
        fields=[
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "volunteer", True, node="identity"),
            FieldSpec("symptom_change", "volunteer", "无变化", node="symptom_rating"),
            FieldSpec("control_score", "volunteer", 4, node="symptom_rating"),
            FieldSpec("programming_count", "on_ask", 0, node="programming"),
            FieldSpec("device_explained", "volunteer", True, node="education"),
            FieldSpec("response_speed", "volunteer", "响应慢", node="service"),
            FieldSpec("special_complaint", "volunteer", "回复有点慢", node="service", scoring="optional"),
            FieldSpec("insurance_type", "volunteer", "职工医保", node="insurance"),
            FieldSpec("self_pay", "refuse", "不方便说", node="insurance", status="refused"),
        ],
    )


# --------------------------------------------------------------------------- #
# Hard discrimination cards (区分度优先). Same field set as the baseline cards —
# only the disclosure policy / persona is turned up, so they stay inside the
# existing scoring (no SOP / judge changes) and reuse the real SOP node fields.
# Each saturates ONE lever so a static-prompt free-conversation baseline (spa)
# drops fields while the orchestrated arm (ours), which has explicit per-node
# milestones + followup logic, holds.
# --------------------------------------------------------------------------- #


def _hard_minimal_passive_card() -> Card:
    """H2: passive patient — volunteers nothing, every field is on_ask. spa in
    free conversation only follows the patient's lead → skips whole nodes the
    patient never raises; ours walks every SOP node → asks each."""
    return Card(
        case_id="s2_v3_hard_minimal_passive_006",
        case_type="hard_minimal_passive",
        persona={
            "who": "患者本人接听",
            "temperament": "极度被动惜字如金，从不主动提起任何事，只回答被直接问到的那一点，答完即止，绝不展开",
            "sensitivities": [],
        },
        node_facts={
            "greeting": "现在方便接听。",
            "identity": "接电话的就是患者本人。",
            "symptom_rating": "比以前好一些，算有所改善；满分十分给七分。",
            "programming": "做过一次程控，还算满意。",
            "education": "设备怎么用医生教过。",
            "service": "医院回复挺快。",
            "insurance": "职工医保；报销后自付四万左右。",
            "closing": "没有其他问题了。",
        },
        fields=[
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "on_ask", True, node="identity"),
            FieldSpec("symptom_change", "on_ask", "有所改善", node="symptom_rating"),
            FieldSpec("control_score", "on_ask", 7, node="symptom_rating"),
            FieldSpec("programming_count", "on_ask", 1, node="programming"),
            FieldSpec("programming_satisfaction", "on_ask", "满意", node="programming"),
            FieldSpec("device_explained", "on_ask", True, node="education"),
            FieldSpec("response_speed", "on_ask", "快速响应", node="service"),
            FieldSpec("insurance_type", "on_ask", "职工医保", node="insurance"),
            FieldSpec("self_pay", "on_ask", "四万左右", node="insurance"),
        ],
    )


def _hard_multi_node_refusal_card() -> Card:
    """H3: refusals at three distinct nodes (symptom/programming/insurance), each
    keeping a non-refused required sibling so the node is still advanceable. spa
    either keeps pushing on refused fields (compliance drop) or abandons the whole
    node (sibling completeness drop); ours respects the refusal and collects the
    sibling."""
    return Card(
        case_id="s2_v3_hard_multi_refusal_007",
        case_type="hard_multi_node_refusal",
        persona={
            "who": "患者本人接听",
            "temperament": "平时话挺多爱闲扯，整体配合，但对三件事敏感不愿说：不愿给症状打分、不愿评价程控效果、不愿透露自付金额；其余照常回答",
            "chatter": [
                "扯到家里、天气、最近忙什么",
                "感慨年纪大了毛病多、跑医院累",
            ],
            "emotional_color": "随和爱聊，唯独碰到那三件敏感事就闭口",
            "sensitivities": ["control_score", "programming_satisfaction", "self_pay"],
        },
        node_facts={
            "greeting": "现在方便接听。",
            "identity": "接电话的就是患者本人。",
            "symptom_rating": "比术前好一些，算有所改善；但不愿给具体打分。",
            "programming": "做过两次程控；但不愿评价满不满意。",
            "education": "设备怎么用医生教过。",
            "service": "医院回复挺快。",
            "insurance": "职工医保；但自付金额不方便说。",
            "closing": "没有其他问题了。",
        },
        fields=[
            FieldSpec("call_accepted", "on_ask", "confirmed", node="greeting"),
            FieldSpec("is_patient", "volunteer", True, node="identity"),
            FieldSpec("symptom_change", "volunteer", "有所改善", node="symptom_rating"),
            FieldSpec("control_score", "refuse", "不愿评分", node="symptom_rating", status="refused"),
            FieldSpec("programming_count", "volunteer", 2, node="programming"),
            FieldSpec("programming_satisfaction", "refuse", "不便评价", node="programming", status="refused"),
            FieldSpec("device_explained", "volunteer", True, node="education"),
            FieldSpec("response_speed", "volunteer", "快速响应", node="service"),
            FieldSpec("insurance_type", "volunteer", "职工医保", node="insurance"),
            FieldSpec("self_pay", "refuse", "不方便说", node="insurance", status="refused"),
        ],
    )


def all_cards() -> list[Card]:
    return [
        # original baseline set (cooperative / lightly perturbed)
        _proxy_card(),
        _complete_card(),
        _ambiguous_card(),
        _refusal_card(),
        # retained diversity cards (ours holds ~1.0; passive + multi-refusal personas).
        # The all-vague (005) and compound (008) cards were dropped because a
        # maximally-vague patient anti-discriminates on completeness — see
        # tests/unit/eval/study2_v3/test_cards.py::test_six_cards_present.
        _hard_minimal_passive_card(),
        _hard_multi_node_refusal_card(),
    ]
