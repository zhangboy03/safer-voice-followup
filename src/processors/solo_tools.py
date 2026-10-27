from __future__ import annotations

from enum import Enum, auto

from src.playground.kb_tool import KB_TOOL_SPEC


class ToolClass(Enum):
    KB = auto()  # driver speaks the grounded result; no flash re-stream
    CONTINUE = auto()  # side-effect then flash may continue
    END = auto()  # terminate (subject to backstop)


REQUEST_CALLBACK_SPEC = {
    "type": "function",
    "function": {
        "name": "request_callback",
        "description": (
            "当你判断需要安排人工/医生稍后回电时调用(例如患者强烈要求人工、情况复杂超出你能处理、"
            "或涉及需要人工跟进的投诉)。调用前先口头安抚一句(例如「这个我帮您记下来,安排医生稍后给您回电」)。"
            "调用后我会把回电需求登记好,你继续按随访流程往下聊即可。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "为什么需要回电,一句话。"},
                "priority": {
                    "type": "string",
                    "enum": ["normal", "high"],
                    "description": "涉及安全风险用 high,其余 normal。",
                },
            },
            "required": ["reason"],
        },
    },
}

END_CALL_SPEC = {
    "type": "function",
    "function": {
        "name": "end_call",
        "description": (
            "当随访流程该问的都问完、患者也没有别的问题时调用,用来结束本次通话。"
            "调用后系统会先确认必填信息是否齐全:如果还差,我会告诉你还差什么,你就继续问;齐全了通话才会真正结束。"
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": [],
        },
    },
}

TOOL_SPECS = [KB_TOOL_SPEC, REQUEST_CALLBACK_SPEC, END_CALL_SPEC]
TOOL_NAMES = {s["function"]["name"] for s in TOOL_SPECS}

_CLASS = {
    "query_knowledge_base": ToolClass.KB,
    "request_callback": ToolClass.CONTINUE,
    "end_call": ToolClass.END,
}


def classify(name: str) -> ToolClass:
    return _CLASS[name]
