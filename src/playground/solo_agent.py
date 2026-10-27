"""solo arch journey: one flash drives the whole call (with one KB tool); a
silent off-path extractor fills the form after each turn. Mirrors the shape of
run_spa_journey but drives backend.stream_with_tools directly (tool loop) and
never routes through make_speech_runner. Does NOT import driver/spa_agent."""

from __future__ import annotations

import json
import time
from typing import Awaitable, Callable

from src.backends.llm_base import StreamEventType
from src.playground.human_patient import HumanPatient
from src.playground.kb_tool import KB_TOOL_SPEC

MAX_TOOL_CALLS = 2
_OPENING_TRIGGER = "（系统提示：电话已接通，请按要求开场——做自我介绍并询问现在是否方便。）"

KbTool = Callable[[str], Awaitable[str]]
Extract = Callable[[str], Awaitable[list]]


def _render_transcript(history: list[dict]) -> str:
    """Render the full conversation. The opening is already the first assistant
    entry in `history`, so it is not added separately (avoids a double opening)."""
    lines = []
    for m in history:
        who = "患者" if m["role"] == "user" else "客服"
        lines.append(f"{who}：{m['content']}")
    return "\n".join(lines)


async def _run_segmenting_turn(backend, messages, *, run_id, turn_id, kb_tool, on_pre_tool=None):
    """Stream; on a tool call, run it and re-stream (cap MAX_TOOL_CALLS). Returns
    (final_text, tool_calls, ttft_ms, total_ms). ttft/total measured from turn start
    to the first/last token of the FINAL answer (includes tool-detour latency).

    If the model speaks before calling the tool (text + tool_call in one response),
    that acknowledgment is surfaced via on_pre_tool BEFORE the slow KB round-trip, so
    the patient hears "稍等我帮您查一下" immediately instead of dead air."""
    turn_started = time.monotonic()
    tool_calls: list[dict] = []
    msgs = list(messages)
    while True:
        text_parts: list[str] = []
        pending = None
        seg_first = None
        async for ev in backend.stream_with_tools(msgs, tools=[KB_TOOL_SPEC], session_id=run_id, turn_id=turn_id):
            if ev.type == StreamEventType.TEXT and ev.text:
                if seg_first is None:
                    seg_first = time.monotonic()
                text_parts.append(ev.text)
            elif ev.type == StreamEventType.TOOL_CALL and ev.tool_call is not None:
                pending = ev.tool_call
            elif ev.type == StreamEventType.DONE:
                break
        text = "".join(text_parts)
        if pending is not None and len(tool_calls) < MAX_TOOL_CALLS:
            if text and on_pre_tool is not None:
                await on_pre_tool(
                    {
                        "text": text,
                        "speech_ttft_ms": (seg_first - turn_started) * 1000.0 if seg_first else None,
                        "speech_total_ms": (time.monotonic() - turn_started) * 1000.0,
                    }
                )
            query = pending.arguments.get("query", "")
            result = await kb_tool(query)
            tool_calls.append({"name": pending.name, "query": query, "result": result})
            msgs = msgs + [
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [
                        {
                            "id": pending.id,
                            "type": "function",
                            "function": {
                                "name": pending.name,
                                "arguments": json.dumps(pending.arguments, ensure_ascii=False),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": pending.id, "content": result},
            ]
            continue
        ttft = (seg_first - turn_started) * 1000.0 if seg_first else None
        total = (time.monotonic() - turn_started) * 1000.0
        return text, tool_calls, ttft, total


async def run_solo_journey(
    *,
    system_prompt: str,
    backend,
    kb_tool: KbTool,
    extract_form: Extract,
    catalog,
    patient: HumanPatient,
    run_id: str,
    on_opening,
    on_turn,
    on_form,
    max_turns: int = 30,
) -> dict:
    history: list[dict] = []  # [{role: user|assistant, content}]

    # --- opening (agent speaks first) ---
    opening, _tc, _ttft, _total = await _run_segmenting_turn(
        backend,
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": _OPENING_TRIGGER}],
        run_id=run_id,
        turn_id="solo-open",
        kb_tool=kb_tool,
    )
    await on_opening(opening)
    history.append({"role": "assistant", "content": opening})

    incomplete = True
    for idx in range(1, max_turns + 1):
        turn_id = f"solo-turn-{idx}"
        patient_turn = await patient.respond(
            None,
            history=_render_transcript(history),
            current_assistant_text=history[-1]["content"] if history else opening,
            disclosed_fields=[],
        )
        user_text = patient_turn.patient_utterance or "嗯。"
        history.append({"role": "user", "content": user_text})

        messages = [{"role": "system", "content": system_prompt}] + history

        async def _emit_ack(seg, _turn_id=turn_id):
            # The pre-tool "稍等我帮您查一下" surfaces as its own assistant utterance,
            # shown immediately (before the KB round-trip), sharing this turn's id.
            await on_turn(
                {
                    "run_id": run_id,
                    "turn_id": _turn_id,
                    "kind": "pretool_ack",
                    "user_text": "",
                    "stage1_text": seg["text"],
                    "tool_calls": [],
                    "speech_ttft_ms": seg["speech_ttft_ms"],
                    "speech_total_ms": seg["speech_total_ms"],
                }
            )

        text, tool_calls, ttft, total = await _run_segmenting_turn(
            backend, messages, run_id=run_id, turn_id=turn_id, kb_tool=kb_tool, on_pre_tool=_emit_ack
        )
        history.append({"role": "assistant", "content": text})

        await on_turn(
            {
                "run_id": run_id,
                "turn_id": turn_id,
                "user_text": user_text,
                "stage1_text": text,
                "tool_calls": tool_calls,
                "speech_ttft_ms": ttft,
                "speech_total_ms": total,
            }
        )

        # --- silent off-path extraction (after the utterance is shown) ---
        ex_started = time.monotonic()
        fields = await extract_form(_render_transcript(history))
        await on_form({"fields": fields, "extractor_ms": (time.monotonic() - ex_started) * 1000.0})

        if patient_turn.call_should_end:
            incomplete = False
            break

    return {"run_id": run_id, "arm": "solo", "incomplete_journey": incomplete, "turn_count": len(history) // 2}
