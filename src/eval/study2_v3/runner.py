"""Grid runner for the Study 2 SPA-baseline comparison.

Iterates {agent_model} × {arm} × {card} × {trial}; runs the arm, flattens to a
transcript, scores with the coverage judge; emits one row per trial. Arm and
judge are injectable so the grid logic is unit-tested without network.
"""

from __future__ import annotations

import statistics
import sys
from typing import Any, Awaitable, Callable

from src.core.config import settings
from src.eval.study2_v3.cards import Card
from src.eval.study2_v3.our_arm import journey_to_transcript

# Patient simulator (spec §5): settings.patient_qwen_model speaking the study2 JSON protocol
# (patient_utterance / revealed_fields / call_should_end). NOT the virtual_caller
# PatientAction protocol — that would never populate revealed_fields, breaking
# the closed-loop disclosed-field tracking.
PATIENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "patient_utterance": {"type": "string"},
        "revealed_fields": {"type": "array", "items": {"type": "string"}},
        "call_should_end": {"type": "boolean"},
    },
    "required": ["patient_utterance", "revealed_fields", "call_should_end"],
    "additionalProperties": False,
}

PATIENT_SIMULATOR_TEMPERATURE = 1.99

PATIENT_SYSTEM_PROMPT = """你在扮演一位接到术后电话随访的患者（或其家属代接）。下面的 JSON 给你这通电话里
你这个角色的全部设定：persona（人物与语气）、node_facts（每个话题你知道的真实情况）、
field_disclosure（每个字段的真实值 value 和披露策略 policy）。

像真人打电话一样：你并不知道客服接下来会问什么，只是凭手里的真实情况临场反应，每一轮只回应客服
刚说的话。persona 里的 who / temperament / sensitivities 是你的角色核心，必须让不同 card 听起来像
不同的人：配合型可以一次说完整，含糊型先绕一下，代接家属可以带一点家属视角，费用敏感型要明显回避，
极度被动型才惜字如金。不要为了简短把不同 card 说成同一种人。

你有临场发挥的自由——先答多少、用什么措辞和顺序、把几件事打包一起说、停顿犹豫、口头语、轻微抱怨或
感谢、需要客服追问几次——都可以自然变化，不必每次一样。长短由 temperament 和当前问题决定，不是每
一轮都短答；但不能变成念表格，也不能抢客服流程。

如果 persona 里有 chatter（你爱扯的生活闲话）或 emotional_color（情绪基调），那就是你这个人的底色：
你话多、爱跑题、爱诉苦，几乎每一轮都先绕一大段闲话、抱怨或寒暄，再顺着把客服的问题答了，能多说就多
说，不必急着切正题。chatter 列的话题你可以反复念叨、自由展开、临场添一些无关紧要的生活细节来填充——
这些闲话不计真假，怎么啰嗦都行，目的就是让对话里塞满和随访无关的内容。没给 chatter 的 card 不要硬扯，
按它的 temperament 来（比如惜字如金型就继续短答）。

但闲话只是“包装”，绝不能改写“事实层”：① 与计分有关的事实只能来自 node_facts 和 field_disclosure，
card 没给出的城市、医院、医生、药名、时间、金额、复查安排、参保地等一律不要编，客服问到 card 没有的
就自然说不清楚；② 字段的 policy 照旧——闲扯归闲扯，只要客服把某个 volunteer/on_ask/on_followup 的点
问到位，你最终仍要在啰嗦里给出真实值，不能用闲聊把该说的事实永远盖住；③ 不替客服推进流程，也不要因为
自己扯太久就主动说“没别的事了”而提前结束通话。

但有一条底线不能破：每个字段“最终该不该说得出来”由它的 policy 决定，你不能自行改变：
- policy=volunteer：相关话题被触及时你会说出来（措辞、顺序、是否和别的信息一起说都自由）。
- policy=on_ask：客服问到了你才说；可以答得自然、绕一点，但只要这个点被问清楚了，你就要给真实值，
  不能被直接问到还瞒着。
- policy=on_followup：第一次只给模糊回答（如『还行』『差不多』），客服追问后再给明确值；追问一两次
  都自然，但客服真的追问到了就必须给准值，不能始终不给。
- policy=refuse：明确表示这一项不方便透露，借口和语气可以自由，但始终不说出真实值。
一句话：volunteer / on_ask / on_followup 的字段，只要客服问到位，你最终都会给出真实值；只有 refuse
的字段你才始终不说。不要说设定里没有的信息，不要替客服推进流程。

只输出 JSON：
- patient_utterance：你这一轮说的话。
- revealed_fields：你这一轮真正披露了真实值的字段名列表（没披露就空列表）。
- call_should_end：客服已经道别/结束通话，或你确实没有别的事时为 true，否则 false。"""


def build_patient():
    """Patient simulator fixed to settings.patient_qwen_model (spec §5), study2 JSON protocol."""
    from src.backends.llm_dashscope import QwenDashScopeBackend
    from src.backends.structured_capabilities import build_response_format
    from src.eval.study2_v3.simulated_patient import SimulatedPatient

    backend = QwenDashScopeBackend(
        model=settings.patient_qwen_model,
        temperature=PATIENT_SIMULATOR_TEMPERATURE,
        enable_thinking=False,
    )

    async def _respond(messages: list[dict]) -> dict:
        completion = await backend.complete_structured(
            messages,
            response_format=build_response_format(
                model=settings.patient_qwen_model,
                schema_name="study2_patient",
                schema=PATIENT_SCHEMA,
            ),
            max_tokens=800,  # chatter/墨迹 personas ramble; 400 truncated the JSON tail
            temperature=PATIENT_SIMULATOR_TEMPERATURE,
        )
        return completion.parsed_json or {}

    return SimulatedPatient(_respond, system_prompt=PATIENT_SYSTEM_PROMPT)


ArmFactory = Callable[..., Awaitable[Callable[..., Awaitable[dict]]]]
JudgeFn = Callable[[Card, str], Awaitable[dict]]


async def run_grid(
    *,
    cards: list[Card],
    n_trials: int,
    arm_factory: ArmFactory,
    judge_fn: JudgeFn,
    agent_models: list[str] | None = None,
    arms: list[str] | None = None,
    cells: list[tuple[str, str]] | None = None,
    max_turns: int = 18,
) -> list[dict[str, Any]]:
    """Run each (agent_model, arm) cell over cards × trials.

    Pass explicit ``cells`` for the 3-cell final eval {flash·spa, plus·spa,
    flash·ours}; the strong-model plus·ours pair is deliberately excluded as
    token waste. Falls back to the ``agent_models × arms`` cross product when
    ``cells`` is omitted.
    """
    if cells is None:
        if agent_models is None or arms is None:
            raise ValueError("run_grid requires either cells=... or both agent_models=... and arms=...")
        cells = [(model, arm) for model in agent_models for arm in arms]

    rows: list[dict[str, Any]] = []
    for agent_model, arm in cells:
        run = await arm_factory(agent_model=agent_model, arm=arm)
        for card in cards:
            for trial in range(1, n_trials + 1):
                run_id = f"{agent_model}:{arm}:{card.case_id}:{trial}"
                result = await run(card, run_id=run_id, max_turns=max_turns)
                transcript = journey_to_transcript(result)
                scored = await judge_fn(card, transcript)
                ttfts = [
                    it["ttft_ms"]
                    for it in result.get("actual_items", [])
                    if isinstance(it.get("ttft_ms"), (int, float))
                ]
                rows.append(
                    {
                        "agent_model": agent_model,
                        "arm": arm,
                        "case_id": card.case_id,
                        "case_type": card.case_type,
                        "trial": trial,
                        "completeness": scored["completeness"],
                        "accuracy": scored["accuracy"],
                        "refusal_compliance": scored["refusal_compliance"],
                        "verdicts": scored.get("verdicts"),
                        "ttft_ms_median": statistics.median(ttfts) if ttfts else None,
                        "turn_count": result.get("turn_count"),
                        "incomplete_journey": result.get("incomplete_journey"),
                        # Diagnostic capture: flat transcript handed to the judge,
                        # plus the full per-turn trace (node_before/stage1/stage2/
                        # accepted_fields/fast_judge_action for ours; user/assistant
                        # for spa) so hard-case stalls are auditable turn by turn.
                        "transcript": transcript,
                        "trace": result.get("actual_items"),
                    }
                )
                print(
                    f"[grid] {run_id} comp={scored['completeness']:.3f} "
                    f"turns={result.get('turn_count')} incomplete={result.get('incomplete_journey')}",
                    file=sys.stderr,
                    flush=True,
                )
    return rows


async def default_arm_factory(*, agent_model: str, arm: str):
    """Production arm factory: real our-arm or real SPA arm on agent_model."""
    from src.backends.llm_dashscope import QwenDashScopeBackend
    from src.eval.study2_v3.our_arm import build_our_arm
    from src.eval.study2_v3.spa_agent import run_spa_journey
    from src.eval.study2_v3.speech_runner import make_speech_runner

    if arm == "ours":
        return build_our_arm(agent_model=agent_model)
    if arm == "spa":
        speech_runner = make_speech_runner(QwenDashScopeBackend(model=agent_model, enable_thinking=False))

        async def _run(card, *, run_id, max_turns=18):
            return await run_spa_journey(
                card, speech_runner=speech_runner, patient=build_patient(), run_id=run_id, max_turns=max_turns
            )

        return _run
    raise ValueError(f"unknown arm: {arm}")
