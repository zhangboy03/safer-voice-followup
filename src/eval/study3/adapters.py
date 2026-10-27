# src/eval/study3/adapters.py
"""System adapters for Study 3 baselines. Each produces one patient-facing text.

complete_fn(messages)->str and retrieve_fn(question)->list[kb_dict] are injected
so unit tests run without network. The runner wires real Qwen + RAG."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Awaitable, Callable

CompleteFn = Callable[[list[dict]], Awaitable[str]]
RetrieveFn = Callable[[str], Awaitable[list[dict]]]


@dataclass
class SystemOutput:
    system: str
    patient_text: str
    kb_refs: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | retrieval_empty | adapter_error
    whitebox: dict = field(default_factory=dict)  # Pins only


def _kb_block(items: list[dict]) -> str:
    lines = []
    for it in items:
        lines.append(f"- (id:{it['id']}) Q:{it.get('question', '')} A:{it.get('answer', '')}")
    return "\n".join(lines)


# The full-injection baseline = "single prompt, single generation": dump all of
# Pins's policy (classification + answer norms + safety) into one prompt and let
# the LLM answer in one shot. But the raw fast_judge review prompt opens with
# "你只输出 JSON 路由事实，不直接回答患者" — operating instructions for Pins's
# classifier *node*, not policy content. Injected verbatim and first, they dominate
# the model's attention and sabotage the answer task we are measuring. We keep all
# three policy blocks injected (faithful to 全量注入) but lead with a controlling
# single-shot-answerer frame that explicitly neutralizes the classifier framing, so
# the baseline gets its full answering capability.
_FULL_INJECTION_HEADER = (
    "你是品驰随访客服 AI，采用单段式作答：对患者这一问，你要在一次生成里直接产出"
    "「患者最终听到的那一段中文回复」。\n\n"
    "下面依次给你三块策略材料（① 分类/路由策略 ② 回答规范 ③ 安全边界）和知识库检索结果，"
    "请把它们当作判断依据，而不是输出格式要求：\n"
    "- ① 中「只输出 JSON 路由事实 / 不直接回答患者 / 由下游代码决定说什么」是写给 Pins "
    "两段式管线里分类节点的指令，在本单段式系统里不适用——这里没有下游节点，必须由你把话说完。\n"
    "- 先用 ① 的分类逻辑在内部判断本问属于「可答 / 需追问澄清 / 应拒答转介」中的哪一类，"
    "再据此作答：可答就按 ② 的规范、严格基于知识库回答；需澄清就只追问缺失的关键信息；"
    "应拒答就按 ③ 说明情况并安排医生回电或建议尽快就医。\n"
    "- 严禁编造知识库以外的内容；不得给个体化诊断/病因判断、用药剂量增减停、"
    "放行 MRI/安检/放疗等相容性结论、或建议推迟就医。"
)
_FULL_INJECTION_OUTPUT_CONTRACT = (
    "只输出患者可以听到的最终回复（纯中文文本）。"
    "不要输出 JSON、字段名、内部分类、scope、reason、confidence 或任何调试字段。"
)


async def _retrieve_or_empty(question: str, retrieve_fn: RetrieveFn):
    try:
        items = await retrieve_fn(question)
    except Exception:
        return None, "adapter_error"
    if not items:
        return [], "retrieval_empty"
    return items, "ok"


async def run_rag_llm(
    question: str,
    *,
    complete_fn: CompleteFn,
    retrieve_fn: RetrieveFn,
    safety_prompt: str,
) -> SystemOutput:
    items, status = await _retrieve_or_empty(question, retrieve_fn)
    if status != "ok":
        return SystemOutput(system="rag_llm", patient_text="", kb_refs=[], status=status)
    system = f"{safety_prompt}\n\n[知识库检索结果]\n{_kb_block(items)}"
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    try:
        text = await complete_fn(messages)
    except Exception:
        return SystemOutput(system="rag_llm", patient_text="", status="adapter_error")
    return SystemOutput(system="rag_llm", patient_text=text, kb_refs=[it["id"] for it in items])


async def run_full_injection_llm(
    question: str,
    *,
    complete_fn: CompleteFn,
    retrieve_fn: RetrieveFn,
    review_prompt: str,
    speech_prompt: str,
    safety_prompt: str,
) -> SystemOutput:
    items, status = await _retrieve_or_empty(question, retrieve_fn)
    if status != "ok":
        return SystemOutput(system="full_prompt_injection_llm", patient_text="", kb_refs=[], status=status)
    # The "kitchen-sink" single prompt: controlling single-shot-answerer frame first,
    # then full classification policy + answer norms + safety + KB, output contract last.
    system = (
        f"{_FULL_INJECTION_HEADER}\n\n"
        f"===== ① 分类/路由策略 =====\n{review_prompt}\n\n"
        f"===== ② 回答规范 =====\n{speech_prompt}\n\n"
        f"===== ③ 安全边界 =====\n{safety_prompt}\n\n"
        f"===== 知识库检索结果 =====\n{_kb_block(items)}\n\n"
        f"===== 输出要求 =====\n{_FULL_INJECTION_OUTPUT_CONTRACT}"
    )
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    try:
        text = await complete_fn(messages)
    except Exception:
        return SystemOutput(system="full_prompt_injection_llm", patient_text="", status="adapter_error")
    return SystemOutput(
        system="full_prompt_injection_llm",
        patient_text=text,
        kb_refs=[it["id"] for it in items],
    )
