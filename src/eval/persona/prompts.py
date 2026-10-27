"""Patient system prompt template for LLMPatientDriver."""

from __future__ import annotations

from src.eval.persona.persona import PatientPersona

PATIENT_SYSTEM_PROMPT_TEMPLATE = """你正在扮演一位品驰医疗的起搏器用户，正在拨打品驰健康关爱中心的客服电话。

## 你的角色设定

- 身份：{basic_desc}
- 医疗背景：{medical_desc}
- 性格特点：{personality_desc}
- 交互风格：{interaction_desc}

## 你的主要诉求

{primary_concern}

## 说话风格参考

{style_examples}

## 严格约束

1. 你只能扮演患者，不能冒充客服或医生
2. 保持角色一致性——不要突然改变性格或记忆
3. 不要主动结束通话，除非客服明确说"再见"或你已经得到满意回答
4. 回复要简短自然（1-3句话），像真实患者一样
5. 可以犹豫、反问、跑题——这是自然的
{forbidden_rules}

## 输出格式

你必须输出 JSON：{{"kind": "speak"|"silence"|"hangup", "text": "你的回复", "silence_ms": null}}
- speak: 正常回复
- silence: 不知道说什么/在想（设置 silence_ms 毫秒数）
- hangup: 通话结束
"""


def render_patient_prompt(persona: PatientPersona) -> str:
    """Render a full system prompt from a PatientPersona."""
    basic_desc = f"{persona.basic.age_group}岁{persona.basic.gender}"
    if persona.basic.dialect_hint:
        basic_desc += f"，{persona.basic.dialect_hint}口音"

    medical_desc = persona.medical.device_summary
    if persona.medical.history:
        medical_desc += f"，病史：{'、'.join(persona.medical.history)}"

    personality_desc = (
        f"耐心程度：{persona.personality.patience}，"
        f"话多程度：{persona.personality.verbosity}，"
        f"记忆力：{persona.personality.memory_quality}，"
        f"情绪状态：{persona.personality.emotion}"
    )

    interaction_desc = (
        f"跑题倾向：{persona.interaction.tendency_to_digress:.0%}，"
        f"需要重复解释：{persona.interaction.needs_clarification:.0%}，"
        f"决策风格：{persona.interaction.decision_style}"
    )

    style_examples = (
        "\n".join(f"- 「{ex}」" for ex in persona.style_examples) if persona.style_examples else "（无参考）"
    )

    forbidden_rules = ""
    for idx, rule in enumerate(persona.forbidden, start=6):
        forbidden_rules += f"\n{idx}. {rule}"

    return PATIENT_SYSTEM_PROMPT_TEMPLATE.format(
        basic_desc=basic_desc,
        medical_desc=medical_desc,
        personality_desc=personality_desc,
        interaction_desc=interaction_desc,
        primary_concern=persona.medical.primary_concern_paraphrase,
        style_examples=style_examples,
        forbidden_rules=forbidden_rules,
    )
