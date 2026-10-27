"""Production solo-driver system prompt: the playground whole-SOP prompt +
unified-tool usage rules. KB content is spoken by the system, not the flash."""

from __future__ import annotations

from src.playground.solo_prompt import _SOLO_SYSTEM_PROMPT

# Strip the playground's 【知识库工具】 two-step block (it told the flash to restate
# the KB result) and append the production tool rules.
_BASE = _SOLO_SYSTEM_PROMPT.split("【知识库工具】")[0].rstrip()

_TOOL_RULES = """
【可用工具】你有三个工具,需要时直接调用,不要自己硬来:
1. query_knowledge_base —— 患者主动提问、尤其涉及病情/主治医生/品驰产品或设备使用/医院服务、或任何你没法只凭常识稳妥回答的问题,就调用它查。调用前先口头安抚一句(例如「好的,您这个情况我帮您查一下,稍等我一下哈」),把这句说完,在同一轮里紧接着发起调用。**查到的答复会由系统直接、准确地播报给患者——你不要再重复、改写或补充知识库的内容**,系统播报完你就接着按流程往下聊。
2. request_callback —— 需要安排人工/医生回电时调用(调用前先安抚一句)。登记后你继续往下聊。
3. end_call —— 该问的都问完、患者也没别的问题时调用来结束通话。系统会先确认必填项是否齐全,没齐我会告诉你还差什么,你就继续问。

调用工具时,你自己【绝不】编造医疗答复;知识库的事交给 query_knowledge_base。
"""


def build_solo_production_prompt(language: str = "zh") -> str:
    if language == "en":
        from src.prompts.solo_english import SYSTEM

        return SYSTEM
    return _BASE + "\n" + _TOOL_RULES
