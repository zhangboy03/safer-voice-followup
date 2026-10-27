"""KB lookup exposed as a native-function-calling tool. The flash decides WHEN
to call; the tool runs study3's full-injection answerer (route → answer /
refuse-with-callback) over the full 245-entry KB and returns patient-facing text."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Awaitable, Callable

from src.backends.llm_base import StreamEventType
from src.eval.study3.adapters import run_full_injection_llm

_REPO = Path(__file__).resolve().parents[2]
_PACK = _REPO / "config" / "prompts" / "packs" / "sop-v1"
_KB_PATH = _REPO / "config" / "knowledge_base.json"

# Inline, callback-correct safety block — copied verbatim from study3's MVP runner
# (docs/superpowers/plans/2026-06-02-study3-mvp-eval.md). Note "安排医生回电".
_SAFETY_PROMPT = (
    "你是品驰医疗设备客服。只回答知识库覆盖的问题；个体化诊断/用药剂量/红旗症状/自伤/超范围"
    "都不要直接回答，应说明并安排医生回电或建议就医。"
)

KB_TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": "query_knowledge_base",
        "description": (
            "查询品驰医疗知识库并返回可向患者播报的答复。只要患者主动提问——尤其涉及自身病情、"
            "主治医生、品驰产品/设备使用或医院服务,或任何你无法仅凭常识稳妥回答的问题——就调用本工具。"
            "工具会判断该问能否依据知识库回答:能则给出标准答复,不能则给出婉拒并安排人工回电。"
            "你拿到返回内容后口语化转述即可。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "用患者原话关键词组成的问题,例如「体内产品能不能做核磁」。",
                }
            },
            "required": ["query"],
        },
    },
}


def _pack_text(rel: str) -> str:
    return (_PACK / rel).read_text(encoding="utf-8")


def _load_full_kb() -> list[dict]:
    return json.loads(_KB_PATH.read_text(encoding="utf-8"))


def _make_complete_fn(backend) -> Callable[[list[dict]], Awaitable[str]]:
    async def _complete(messages: list[dict]) -> str:
        parts: list[str] = []
        async for ev in backend.stream_with_tools(messages):
            if ev.type == StreamEventType.TEXT and ev.text:
                parts.append(ev.text)
        return "".join(parts)

    return _complete


async def run_kb_tool(query: str, *, backend, review_prompt: str, speech_prompt: str, safety_prompt: str) -> str:
    kb = _load_full_kb()

    async def _retrieve(_q: str):  # full injection: ignore the query, return all entries
        return kb

    out = await run_full_injection_llm(
        query,
        complete_fn=_make_complete_fn(backend),
        retrieve_fn=_retrieve,
        review_prompt=review_prompt,
        speech_prompt=speech_prompt,
        safety_prompt=safety_prompt,
    )
    return out.patient_text or "这个问题我这边先记一下,会安排人工稍后给您回电。"


def make_kb_tool(backend) -> Callable[[str], Awaitable[str]]:
    """Bind a real backend + the production policy prompts into an async (query)->text tool."""
    review = _pack_text("fast_judge/knowledge_qa_review.md")
    speech = _pack_text("speech/knowledge_qa.md")

    async def _tool(query: str) -> str:
        return await run_kb_tool(
            query, backend=backend, review_prompt=review, speech_prompt=speech, safety_prompt=_SAFETY_PROMPT
        )

    return _tool
