"""品驰随访 Agent 流程配置。

双路 LLM 架构：
- Speech LLM: 生成对话回复（流式 → TTS）
- Judge LLM:  字段提取 + 阶段完成判定（并行运行，per-node prompt override）

8 主线节点 + 1 条件触发节点 + 1 知识问答节点 = 10 总计。
三条收尾路径：正常主线→closing / [GOODBYE]→terminal hook / greeting refused→terminal hook。
"""

from __future__ import annotations

import json as _json
from pathlib import Path as _Path

from src.flows.ablation_switches import is_verify_disabled
from src.flows.models import (
    TERMINAL_ROUTE,
    AgentConfig,
    CompletionPolicy,
    ContextStrategy,
    ContextStrategyConfig,
    FlowState,
    NodeConfig,
)


def _load_knowledge_base() -> tuple[str, int]:
    """Load knowledge base and format for full-context injection."""
    kb_path = _Path(__file__).resolve().parents[2] / "config" / "knowledge_base.json"
    with open(kb_path, encoding="utf-8") as f:
        entries = _json.load(f)
    lines = []
    for i, entry in enumerate(entries, 1):
        lines.append(f"[{i}] 问：{entry['question']}\n答：{entry['answer']}")
    return "\n\n".join(lines), len(entries)


_kb_text, _kb_count = _load_knowledge_base()

# ── Field → Node 映射 ──

NODE_ALLOWED_FIELDS: dict[str, set[str]] = {
    "greeting": {"call_accepted"},
    "identity": {"is_patient", "relationship"},
    "symptom_rating": {"symptom_change", "control_score"},
    "programming": {"programming_count", "programming_satisfaction"},
    "education": {"device_explained"},
    "service": {"response_speed", "special_praise", "special_complaint"},
    "insurance": {
        "insurance_type",
        "insurance_city",
        "total_cost",
        "self_pay",
        "huimin_bao",
    },
}

NODE_TO_GROUP: dict[str, str] = {
    "greeting": "开场",
    "identity": "A-身份确认",
    "symptom_rating": "B-症状评分",
    "programming": "C-程控信息",
    "education": "D-宣教落实",
    "service": "E-医院服务",
    "insurance": "F-保费信息",
    "closing": "G-收尾",
    "knowledge_qa": "知识问答",
    "human_transfer": "转人工",
}

RAG_ENABLED_NODES: set[str] = set()  # v1: RAG not yet integrated


# ── Greeting route_fn ──


def _greeting_route(state: FlowState) -> str:
    """Route greeting: confirmed → identity, refused → __terminal__."""
    fv = state.collected_fields.get("call_accepted")
    if fv and fv.value == "refused":
        return TERMINAL_ROUTE
    return "identity"


_KNOWLEDGE_QA_SYSTEM_PROMPT_TEMPLATE = (
    "你是品驰随访客服小品，用知识库回答患者问题。\n\n"
    "# 规则\n"
    "知识库是唯一信息来源。回答前三步检查：\n"
    "1.定位：知识库中有直接对应条目？无→拒绝\n"
    "2.核实：条目的「答」包含患者要的具体信息？无→拒绝\n"
    "3.单源：只需一条条目？需组合多条→拒绝\n\n"
    '拒绝→"这类问题建议您咨询主治医生或随访医生，我这边只负责产品使用相关的标准答复。'
    '您需要我帮您转接人工客服吗？"\n'
    "通过→用该条目口语化回答（2-3句），"
    "不给额外建议，不解释原理，不推测原因，结尾问「还有其他想了解的吗」\n\n"
    "必须拒绝（以下任一类型一律转咨询医生，即使知识库里出现相关字样）：\n"
    "- 个体化医学判断：「我该怎么办」「我能不能吃/做 XX」「我的情况需不需要 XX」\n"
    "- 药物剂量、用法、调整、能否与某药同服\n"
    "- 症状原因解释、发病机理、预后判断\n"
    "- 具体治疗方案、是否需要复查或手术\n"
    "- 手术费用报销、其他厂商对比、医院医生推荐\n"
    "- 知识库未提及的任何内容\n\n"
    "绝对禁止：编造、推测、拼接不同条目、给医疗建议、解释医学原理、"
    "告知用药剂量、评估患者个体情况、告知可能的病因。\n\n"
    "回答只能客观陈述知识库条目的口语化改写，不得加入任何知识库外的医学判断。\n\n"
    "# 知识库（{count}条）\n\n"
    "{knowledge}"
)

# Stripped-down variant used when ABLATION_DISABLE_VERIFY=1 is set.  Keeps only
# the base persona + KB injection; the "三步检查" / "必须拒绝" / "绝对禁止"
# blocks are intentionally removed so the §4.6 A3 experiment can measure the
# marginal value of the verification discipline.
# ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
_KNOWLEDGE_QA_SYSTEM_PROMPT_ABLATED_TEMPLATE = (
    "你是品驰随访客服小品，用知识库回答患者问题。\n\n# 知识库（{count}条）\n\n{knowledge}"
)


def build_knowledge_qa_system_prompt(count: int, knowledge: str) -> str:
    """Render the knowledge_qa system prompt honouring the verify ablation flag.

    Returns the full three-step-verification prompt by default.  When
    ``ABLATION_DISABLE_VERIFY=1`` is set, returns the stripped variant with
    only persona + knowledge base (no 三步检查 / 必须拒绝 / 绝对禁止 blocks).
    """
    # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
    template = (
        _KNOWLEDGE_QA_SYSTEM_PROMPT_ABLATED_TEMPLATE if is_verify_disabled() else _KNOWLEDGE_QA_SYSTEM_PROMPT_TEMPLATE
    )
    return template.format(count=count, knowledge=knowledge)


# ── Agent Config Builder ──


def build_agent_config() -> AgentConfig:
    return AgentConfig(
        name="pinchi_followup_agent",
        persona_messages=[
            {
                "role": "system",
                "content": (
                    "# 角色\n"
                    "你是品驰医疗随访客服「小品」，按当前节点完成电话随访。\n\n"
                    "# 优先级\n"
                    "1.情绪安抚/安全/转人工/挂断 2.完成当前节点 3.简短自然\n\n"
                    "# 说话方式\n"
                    "口语化中文，像真人打电话一样自然。称呼「您」，每次1-2句话优先1句。"
                    "不用表情符号、括号、舞台说明。不说「嗯嗯」。不连续两轮同一开头词。\n"
                    "★ 绝对不要说「已确认」「已收到」「感谢您的配合」这类机器人话术。"
                    "用真人会说的话回应：「好的」「行」「了解了」「没问题」。\n\n"
                    "# 对话规则\n"
                    "你是主导方，一次只问一个问题，只处理当前节点，不预告下一节点。"
                    "回答空泛、没信息量（嗯/好/还行）→ 换个说法把本节点的问题再问一遍。"
                    "只有这句转录读不通、残缺、明显是语音识别出错，才自然说一句「这边信号好像有点不清楚，您方便再说一遍吗？」。\n\n"
                    "# 特殊标记\n"
                    "- 用户要转人工 → [TRANSFER]+一句话转接\n"
                    "- 用户要挂电话 → [GOODBYE]+一句话告别\n"
                    "- 用户情绪激动 → 先安抚再确认是否继续\n"
                    "- 用户主动提新的产品/医疗问题（非回答你的提问）→ 绝不自己作答（哪怕你知道答案），"
                    '也不要承诺"我帮您查一下"这类动作；只简短自然地接住一句，要不要查交给系统判断。\n'
                    "- 用户在回答你上一轮的问题 → 正常处理。"
                ),
            }
        ],
        global_tools=[],
        initial_node="greeting",
        nodes={
            # ── 1. 开场确认 ──
            "greeting": NodeConfig(
                name="greeting",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：确认是否方便接听\n"
                            "★ 第一次回复必须自我介绍（含 AI 身份/录音告知/转人工提示三项）：「您好，这里是品驰医疗，我是 AI 随访助手小品，本次通话将会录音，您随时可以要求转接人工客服。想了解一下您术后恢复情况，请问现在方便接电话吗？」\n"
                            "不管用户先说了什么（喂/你好/谁啊），你的第一次回复都要包含自我介绍+问是否方便。\n"
                            "反问身份来意→简短说明来意并重新确认。\n"
                            "用户确认方便→简短回应如「好的，那我简单问您几个问题」。\n"
                            "不方便→「没关系，那改天再联系您」。\n"
                            "禁止：「喂/你好」不算同意；确认方便后不说「已确认」「感谢配合」。"
                        ),
                    }
                ],
                tools=[],
                successors=["identity"],
                completion_policy=CompletionPolicy(required_fields=["call_accepted"]),
                route_fn=_greeting_route,
                field_definitions={
                    "call_accepted": "用户是否同意继续通话（confirmed/refused）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["greeting"],
            ),
            # ── 2. 身份确认 ──
            "identity": NodeConfig(
                name="identity",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：确认接听人是患者本人还是家属代接\n"
                            "自然过渡到身份确认，不要重新打招呼。\n"
                            "例：「那我先确认一下，请问您是患者本人吗？」\n"
                            "模糊回答→只围绕接听人身份重新确认，不扩展到手术、症状或后续问题。"
                            "确认后→自然回应如「好的」「嗯好的」。不愿说→「没关系」。\n"
                            "禁止：不重复「您好」；说「患者本人」不说「手术本人」；不提后续环节。"
                        ),
                    }
                ],
                tools=[],
                successors=["symptom_rating"],
                completion_policy=CompletionPolicy(required_fields=["is_patient"]),
                field_definitions={
                    "is_patient": "是否患者本人接听（bool）",
                    "relationship": "来电人与患者关系（仅非本人时，str）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["identity"],
            ),
            # ── 3. 症状变化与评分 ──
            "symptom_rating": NodeConfig(
                name="symptom_rating",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：了解症状变化 + 请患者打分（先症状后打分）\n"
                            "自然地问症状：「那您术后感觉怎么样，跟手术前比有改善吗？」\n"
                            "描述了症状没提分数→引导打分：「如果给恢复情况打个分，0分是特别难受，10分跟正常人一样，您觉得能打几分？」\n"
                            "不理解打分→白话解释。拒绝打分→「没关系」。\n"
                            "注意：先让患者描述感受再引导评分；"
                            "「好多了」是症状描述不是打分；「还行」→追问跟术前比如何。"
                        ),
                    }
                ],
                tools=[],
                successors=["programming"],
                completion_policy=CompletionPolicy(required_fields=["symptom_change", "control_score"]),
                field_definitions={
                    "symptom_change": "症状变化（枚举：明显改善/有所改善/无变化/有所加重）",
                    "control_score": "症状控制评分（整数0-10）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["symptom_rating"],
            ),
            # ── 4. 程控信息 ──
            "programming": NodeConfig(
                name="programming",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：了解程控次数和满意度\n"
                            "自然过渡：「对了，您术后到现在做过几次程控？」\n"
                            "说了次数→追问满意度「效果您还满意吗」。"
                            "没做过→「嗯好的，了解了」。"
                            "只说「做过」没次数→「大概做了几次？」\n"
                            "禁止：不解释什么是程控；不提后续环节。"
                        ),
                    }
                ],
                tools=[],
                successors=["education"],
                completion_policy=CompletionPolicy(required_fields=["programming_count", "programming_satisfaction"]),
                field_definitions={
                    "programming_count": "程控次数（整数）",
                    "programming_satisfaction": "程控满意度（枚举：满意/不满意）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["programming"],
            ),
            # ── 5. 宣教落实 ──
            "education": NodeConfig(
                name="education",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：了解设备说明情况\n"
                            "自然过渡：「另外想了解一下，设备的使用方法，当时医生有没有跟您讲清楚？」\n"
                            "讲过/没讲清→记录即可；记不清→「没关系，时间久了记不清很正常」。\n"
                            "注意：「设备说明」指手术时医生是否讲过设备怎么用，不是你现在的讲解；"
                            "用户说「好的知道了」回应你的讲解不算回答。"
                        ),
                    }
                ],
                tools=[],
                successors=["service"],
                completion_policy=CompletionPolicy(required_fields=["device_explained"]),
                field_definitions={
                    "device_explained": "设备使用方法是否已说明（bool）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["education"],
            ),
            # ── 6. 医院服务 ──
            "service": NodeConfig(
                name="service",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：了解医院服务体验\n"
                            "自然过渡：「那您觉得咱们医院的服务怎么样？有问题找医院的时候响应快不快？」\n"
                            "表扬→「那我帮您转达，谢谢您」。"
                            "投诉→「您说的我记下来了，会帮您反馈」。"
                            "没问题→「好的」。\n"
                            "禁止：不为医院辩解；聊症状/费用时的「还行」不是评价服务。"
                        ),
                    }
                ],
                tools=[],
                successors=["insurance"],
                completion_policy=CompletionPolicy(required_fields=["response_speed"]),
                field_definitions={
                    "response_speed": "医院响应速度（枚举：快速响应/响应慢/无问题）",
                    "special_praise": "特别表扬（str）",
                    "special_complaint": "特别投诉（str）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["service"],
            ),
            # ── 7. 保费信息 ──
            "insurance": NodeConfig(
                name="insurance",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：了解医保类型和费用（敏感话题，不想说随时停）\n"
                            "自然过渡：「最后想简单了解一下，您这次手术用的是什么医保？」\n"
                            "说了类型→追问自付「那报销下来自己大概花了多少？」\n"
                            "只说「有医保」→追问一次具体类型。"
                            "不想聊→「没关系，不方便说就不说了」。\n"
                            "禁止：不追问超过一次；不评价费用高低。"
                        ),
                    }
                ],
                tools=[],
                successors=["closing"],
                completion_policy=CompletionPolicy(required_fields=["insurance_type", "self_pay"]),
                field_definitions={
                    "insurance_type": "医保类型（str）",
                    "insurance_city": "参保城市（str）",
                    "total_cost": "手术总费用（str）",
                    "self_pay": "自付金额（str）",
                    "huimin_bao": "是否有惠民保（bool）",
                },
                allowed_fields=NODE_ALLOWED_FIELDS["insurance"],
            ),
            # ── 知识问答 ──
            "knowledge_qa": NodeConfig(
                name="knowledge_qa",
                task_messages=[
                    {
                        "role": "system",
                        # ABLATION: see docs/superpowers/specs/2026-04-15-thesis-ch4-eval-roadmap.md §4.6
                        # Render through the helper so ABLATION_DISABLE_VERIFY=1 can
                        # strip the 三步检查 / 必须拒绝 blocks at agent-build time.
                        "content": build_knowledge_qa_system_prompt(
                            count=_kb_count,
                            knowledge=_kb_text,
                        ),
                    }
                ],
                tools=[],
                successors=[],
                completion_policy=None,
                field_definitions={},
                allowed_fields=None,
                context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
                respond_immediately=True,
                followup_buffer="请稍等，",
            ),
            # ── 8. 收尾 ──
            "closing": NodeConfig(
                name="closing",
                task_messages=[
                    {
                        "role": "system",
                        "content": (
                            "# 当前任务：结束通话\n"
                            "自然收尾：「好的，今天就了解到这儿，感谢您抽时间，祝您身体健康，再见！」\n"
                            "禁止：不总结聊过的内容、不提醒注意事项、不问还有什么问题。"
                        ),
                    }
                ],
                tools=[],
                successors=[],
            ),
            # ── 9. 转人工 ──
            "human_transfer": NodeConfig(
                name="human_transfer",
                task_messages=[{"role": "system", "content": "正在转接人工客服。"}],
                tools=[],
                context_strategy=ContextStrategyConfig(strategy=ContextStrategy.RESET),
                successors=[],
            ),
        },
    )
