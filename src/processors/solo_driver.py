"""SoloDriverProcessor: single flash drives the whole call from a whole-SOP
prompt + native tools. Replaces LLMProcessor at the pipeline brain seam.
Honors the frame contract; spawns per-turn tasks (never awaits inline)."""

from __future__ import annotations

import asyncio
import contextlib
import enum
import re
import time
import uuid

from src.backends.llm_base import StreamEventType
from src.core.logging import get_logger
from src.pipeline.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    FrameDirection,
    LLMResponseEndFrame,
    LLMTextFrame,
    StartFrame,
    UserStartedSpeakingFrame,
    UserTurnCompleteFrame,
)
from src.pipeline.processor import FrameProcessor
from src.processors.solo_tools import TOOL_SPECS

logger = get_logger(__name__)

_OPENING_TRIGGER = "（系统提示：电话已接通，请按要求开场——做自我介绍并询问现在是否方便。）"

# B3 farewell backstop tuning: silent grace after a spoken farewell segment
# (no end_call tool) before we terminate the call ourselves. 8s is long enough
# that a patient who is merely pausing/thinking will speak (any sound clears the
# suspect via UserStartedSpeaking), but short enough not to leave a finished call
# hanging. Tests monkeypatch this down to ~0.15s.
_FAREWELL_GRACE_S = 8.0

# B3 farewell predicate. Each alternative is a phrase that, in practice, ENDS a
# call — chosen conservatively to avoid false positives that would hang up on a
# patient mid-conversation. Notably we do NOT match bare "好的"/"谢谢"/"挂" or
# "再说"(=talk again later? ambiguous) on their own, because those routinely
# appear mid-call. Every alternative must be a genuine CALL-ENDING frame, not a
# courtesy or advice substring — see the anchored 不打扰/先挂 branches below.
# The predicate uses re.search (substring match), so it relies on these
# call-ending phrases not appearing mid-utterance in practice; the grace+silence
# conjunction (a spoken farewell must be followed by _FAREWELL_GRACE_S of silence
# with no user activity) is the second safety net against a false positive.
# Alternatives:
#   再见            — the canonical sign-off.
#   拜拜            — colloquial sign-off.
#   (就|那就)?不(打扰|耽误)(您|你)?了 — "(then) I won't bother/take up more of your
#                     time"; the trailing 了 makes this a closing frame, NOT the
#                     opening courtesy "不打扰您太久" or the transition "不打扰您
#                     休息". Matches "不打扰您了"/"就不打扰了"/"那就不耽误您了".
#                     Optional 就/那就 prefix + optional pronoun.
#   祝(您|你)?(早日)?康复 — "wishing you a (speedy) recovery"; the medical
#                     domain's standard farewell blessing. Optional pronoun +
#                     optional 早日.
#   先挂(了|断|电话|机) — "I'll hang up now"; tightened to call-ending objects
#                     only so medical advice "先挂水"/"先挂个号" no longer trips.
#                     Matches "先挂了"/"先挂断"/"先挂电话"/"先挂机".
#   先(不耽误您了|不打扰了) — "I won't take up more of your time now"; the 先… +
#                     trailing 了 frame is an unambiguous farewell.
#   挂断            — explicit "hang up".
#   下次(再)?(聊|说|联系) — "talk/contact again next time"; a parting line.
_FAREWELL_RE = re.compile(
    r"\b(?:goodbye|bye[- ]bye|talk to you (?:again|soon)|have a (?:good|lovely|nice) (?:day|evening))\b"
    r"|再见"
    r"|拜拜"
    r"|(就|那就)?不(打扰|耽误)(您|你)?了"
    r"|祝(您|你)?(早日)?康复"
    r"|先挂(了|断|电话|机)"
    r"|先(不耽误您了|不打扰了)"
    r"|挂断"
    r"|下次(再)?(聊|说|联系)",
    re.IGNORECASE,
)


def _looks_like_farewell(text: str) -> bool:
    """True when a segment's spoken text reads as a call-ending goodbye. Used by
    the B3 backstop to flag a segment that finished WITHOUT an end_call tool."""
    return bool(text) and _FAREWELL_RE.search(text) is not None


# B4 escape valve: the end_call backstop may push back at most twice per call;
# the 3rd genuine end_call attempt is honored even with missing ★, so a patient
# who can't/won't answer a required field can still end the call. Without this
# bound a patient who keeps giving vague answers to a required ★ never satisfies
# all_required_covered() → every end_call across every turn rejects → the call
# can never end (G03). The per-turn C2 cap only bounds dispatches WITHIN one
# turn; this is the orthogonal cross-turn bound.
_MAX_ENDCALL_BACKSTOP_REJECTIONS = 2


# B1 turn-level error fallback: spoken on the FIRST consecutive turn failure.
# Legacy-brain parity (llm_processor inference_flow_error): never let an
# inference error end in silence — the patient must hear something actionable.
_TURN_ERROR_APOLOGY = "不好意思，刚才线路有点不稳，没听清您说的。麻烦您再说一遍好吗？"
# Spoken on the SECOND consecutive failure: callback model wording (someone
# calls BACK — never "transfer", per the callback-model decision), then the
# call ends boundedly instead of looping apologies forever.
_TURN_ERROR_GOODBYE = "不好意思，我们这边线路出了点问题，稍后会有同事给您回电话，今天先不耽误您了，再见。"


class _EndCallOutcome(enum.Enum):
    """What _dispatch_end_call did, so the segmenting loop knows how to exit."""

    ARMED = "armed"  # terminal armed → turn is over
    # Speculative/untrusted turn refused: NO tool result was appended for the
    # dangling assistant tool_call, so the caller must END the turn — a
    # re-stream would send a tool_call with no tool result and 400.
    REFUSED_SPECULATIVE = "refused_speculative"
    # Capture backstop appended its own tool message → caller re-enters so the
    # flash re-asks the missing required fields.
    BACKSTOP_INCOMPLETE = "backstop_incomplete"


class SoloDriverProcessor(FrameProcessor):
    def __init__(
        self,
        name,
        *,
        llm_backend,
        system_prompt,
        kb_three_stage,
        kb_answer_backend,
        capture,
        callback_writer,
        call_id,
        patient_info,
        turn_metrics=None,
        lifecycle=None,
        call_tracer=None,
        playout_gate=None,
        language="zh",
    ):
        super().__init__(name)
        self._backend = llm_backend
        self._system_prompt = system_prompt
        self._language = language
        self._kb = kb_three_stage
        self._kb_answer_backend = kb_answer_backend
        self._capture = capture
        self._callback_writer = callback_writer
        self._call_id = call_id
        self._patient_info = patient_info
        self._turn_metrics = turn_metrics
        self._lifecycle = lifecycle
        self._call_tracer = call_tracer
        self._playout_gate = playout_gate

        self._history: list[dict] = []  # [{role, content}], driver-local convo
        self._turn_task: asyncio.Task | None = None
        self._epoch = 0
        self._current_turn_id = ""
        self._abandon_pending = False  # set on UserStartedSpeaking (soft-abandon, H7)
        # B1: reset on any successful turn. A CANCELLED turn neither increments
        # nor resets (CancelledError re-raises before both), so failure →
        # cancelled turn(s) → failure still counts as consecutive.
        self._consecutive_turn_errors = 0
        self._pending_terminal = False
        self._terminal_watchdog: asyncio.Task | None = None
        # B4: cross-call count of backstop rejections; bounds the re-ask livelock
        # — orthogonal to the per-turn end_calls cap (C2).
        self._endcall_backstop_rejections: int = 0
        # B3 backstop for flash forgetting end_call: a segment that ENDED with no
        # tool call but whose text looks like a goodbye. Cleared on any user
        # activity (UserStarted / new turn / flush) and on _disarm_terminal.
        self._farewell_suspect = False
        self._farewell_timer: asyncio.Task | None = None
        self._bootstrapped = False
        self._opening_waiting_for_audio = True
        from src.processors.solo_episode import KbEpisodeManager

        self._episodes = KbEpisodeManager(patient_info=patient_info)

    # ---- frame contract ---------------------------------------------------
    async def process_frame(self, frame, direction):
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)  # forward StartFrame downstream
            if not self._bootstrapped:
                self._bootstrapped = True
                self._spawn_turn(self._open_turn())
            return
        if isinstance(frame, UserTurnCompleteFrame):
            await self._handle_user_turn(frame)  # spawn-and-return; consume (no forward)
            return
        await self.push_frame(frame, direction)

    async def _handle_user_turn(self, frame: UserTurnCompleteFrame):
        # A real transcript can always replace the pending greeting.
        self._opening_waiting_for_audio = False
        # M3: guard happens at the irreversible-action sites, not here.
        await self._cancel_turn()
        self._disarm_terminal()  # C1: the patient spoke again → abort any armed termination
        self._current_turn_id = frame.turn_id
        self._abandon_pending = False
        self._history.append({"role": "user", "content": frame.text})
        # frames.py contract: only "final" is ASR-confirmed text; "" (unlabeled
        # producer) counts as unconfirmed. Untrusted text still gets a spoken
        # reply and KB, but must not trigger irreversible actions.
        text_untrusted = frame.text_source != "final"
        self._spawn_turn(self._run_turn(frame.turn_id, self._epoch, frame.is_preemptive, text_untrusted))

    def _spawn_turn(self, coro):
        self._epoch += 1
        task = asyncio.create_task(coro)
        self._turn_task = task
        task.add_done_callback(self._on_turn_done)

    def _on_turn_done(self, task):
        # I1: the soft-abandon gate suppresses only the output in flight when the user
        # started speaking. Once THIS turn's task ends, clear it so a false barge-in
        # (UserStarted with no following UserTurnComplete, e.g. a cough) can't mute the
        # bot forever. Identity-guarded: a newer turn must not be un-gated by an older
        # task's completion.
        if task is self._turn_task:
            self._abandon_pending = False

    # ---- the turn loops ---------------------------------------------------
    async def _open_turn(self):
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": self._opening_trigger()},
            ]
            pid = self._new_pid()
            text, _ = await self._stream_segment(messages, pid)
            if text:
                self._history.append({"role": "assistant", "content": text})
            self._consecutive_turn_errors = 0
        except asyncio.CancelledError:
            raise  # turn cancellation (barge-in / flush) must keep working
        except Exception:
            logger.exception("solo.turn_error", turn_id="<open>")
            await self._handle_turn_error("")

    async def _run_turn(self, turn_id, epoch, is_preemptive, text_untrusted):
        # B1: a turn task is spawned fire-and-forget, so an uncaught exception
        # here dies as an unretrieved asyncio exception — the patient hears the
        # bot stop mid-sentence and nothing follows. Degrade audibly instead.
        try:
            messages = [{"role": "system", "content": self._system_prompt}, *self._history]
            await self._segmenting_loop(messages, turn_id, epoch, is_preemptive, text_untrusted)
            self._consecutive_turn_errors = 0
        except asyncio.CancelledError:
            raise  # turn cancellation (barge-in / flush) must keep working
        except Exception:
            logger.exception("solo.turn_error", turn_id=turn_id)
            await self._handle_turn_error(turn_id)
        # schedule() sits OUTSIDE the try on purpose: it must run on the error
        # path too (the apology is part of the transcript capture sees), and a
        # raise inside the try would route a capture bug into the apology path.
        # It is sync fire-and-forget; guard it separately so an unlikely sync
        # raise cannot re-create the silent-death mode this wrapper removes.
        if self._capture is not None:
            try:
                self._capture.schedule(self._render_transcript())
            except Exception:
                logger.exception("solo.capture_schedule_error", turn_id=turn_id)

    def _opening_trigger(self):
        if self._language == "en":
            from src.prompts.solo_english import OPENING

            return OPENING
        return _OPENING_TRIGGER

    async def _handle_turn_error(self, turn_id):
        """Spoken degradation for a turn that died mid-flight (DashScope error,
        network blip, tool-args decode...). First failure: apology + silent
        NORMAL callback. Second consecutive failure: callback-model goodbye +
        HIGH callback, then arm the terminal exactly like _dispatch_end_call's
        ARMED path so the call ends after the goodbye plays (BotStopped fires
        the hook; the 5s watchdog is the backstop). Must NEVER raise — a raise
        here would kill the very task it is rescuing (CancelledError still
        propagates: the except clauses are scoped to Exception only)."""
        self._consecutive_turn_errors += 1
        repeated = self._consecutive_turn_errors >= 2
        fallback = _TURN_ERROR_GOODBYE if repeated else _TURN_ERROR_APOLOGY
        if self._language == "en":
            from src.prompts.solo_english import ERROR_APOLOGY, ERROR_GOODBYE

            fallback = ERROR_GOODBYE if repeated else ERROR_APOLOGY
        try:
            await self._speak_phrase(fallback, turn_id, role="error_fallback")
        except Exception:
            logger.exception("solo.turn_error_fallback_failed", turn_id=turn_id, stage="speak")
        try:
            from src.flows.knowledge_review_models import CallbackPriority, EscalationKind

            episode = self._episodes.open_or_continue(turn_id=turn_id)
            try:
                self._enqueue_callback(
                    episode,
                    kind=EscalationKind.EXIT_ESCALATE,
                    priority=CallbackPriority.HIGH if repeated else CallbackPriority.NORMAL,
                    out_of_scope_class=None,
                    user_text="",
                    reason_text="turn_error_repeated" if repeated else "turn_error",
                )
            finally:
                # An enqueue raise must not leave the episode dangling open.
                # close() is a plain attribute reset (KbEpisodeManager: set
                # session.closed, drop the ref) and cannot realistically raise;
                # if it ever did, the outer except still upholds never-raise.
                self._episodes.close()
        except Exception:
            logger.exception("solo.turn_error_fallback_failed", turn_id=turn_id, stage="enqueue")
        if repeated:
            self._arm_terminal()

    async def _segmenting_loop(self, messages, turn_id, epoch, is_preemptive, text_untrusted):
        msgs = list(messages)
        tool_calls = 0
        kb_calls = 0
        end_calls = 0
        while True:
            pid = self._new_pid()
            text, pending = await self._stream_segment(msgs, pid, turn_id=turn_id)
            if text:
                self._history.append({"role": "assistant", "content": text})
            if pending is None:
                # B3: segment ended with NO tool call. If it sounds like a
                # goodbye, the flash likely forgot end_call — flag it so the
                # next BotStopped arms the silent-grace backstop. Set ONLY here
                # (real flash segments); KB answers and phrase-bank lines don't
                # close calls, and B1's goodbye arms the terminal explicitly.
                if _looks_like_farewell(text):
                    self._farewell_suspect = True
                return
            if is_preemptive:
                return  # M3: never run a tool on an unconfirmed speculative turn
            from src.processors.solo_tools import ToolClass, classify

            klass = classify(pending.name)
            # append the assistant tool-call message (OpenAI protocol)
            import json

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
                }
            ]
            if klass is ToolClass.KB:
                if kb_calls >= 2:
                    return
                kb_calls += 1
                await self._dispatch_kb(pending.arguments.get("query", ""), turn_id)
                return  # KB result is driver-spoken; no flash re-stream this turn
            if klass is ToolClass.END:
                if end_calls >= 2:
                    return  # C2: bound re-entry; the flash has had 2 chances this turn
                end_calls += 1
                # Irreversible action needs final-confirmed text (text_source "" counts
                # as unconfirmed per the frames.py contract).
                outcome = await self._dispatch_end_call(msgs, is_preemptive or text_untrusted)
                if outcome is _EndCallOutcome.ARMED:
                    return
                if outcome is _EndCallOutcome.REFUSED_SPECULATIVE:
                    # A4×B3 gap: A4 gated the end_call on unconfirmed text, but a
                    # spoken farewell + silence should still let B3 end the call
                    # (tool-independent backstop) — else the call hangs after the
                    # goodbye until the dispatcher reaper hard-kills it. The
                    # pending-is-None farewell branch was skipped because this
                    # segment DID carry an end_call. NON-farewell text must NOT
                    # arm (a gated end_call mid-conversation must not hang up).
                    if _looks_like_farewell(text):
                        self._farewell_suspect = True
                    return  # no tool result for the dangling tool_call → must NOT re-stream
                if outcome is _EndCallOutcome.BACKSTOP_INCOMPLETE:
                    continue  # backstop said incomplete (tool msg appended) → re-enter so flash re-asks
                # a future enum member must fail loudly here, not silently take the
                # re-stream path (dangling tool_call → 400 from the API)
                raise AssertionError(outcome)
            if klass is ToolClass.CONTINUE:
                if tool_calls >= 2:
                    return
                tool_calls += 1
                # Irreversible action needs final-confirmed text; the "ok" result is
                # still appended below, so the loop can safely continue (no dangling tool_call).
                result = await self._dispatch_callback(pending.arguments, turn_id, is_preemptive or text_untrusted)
                msgs = msgs + [{"role": "tool", "tool_call_id": pending.id, "content": result}]
                continue

    def _publish_caption(self, pid, turn_id, text, *, final=False):
        if self._bus is not None and text:
            from src.pipeline.events import AssistantTextEvent

            self._bus.emit(AssistantTextEvent(turn_id=turn_id, context_id=pid, text=text[-3000:], is_final=final))

    async def _stream_segment(self, messages, pid, *, turn_id=""):
        """Stream TEXT events to TTS as LLMTextFrames; flush with LLMResponseEndFrame.
        Returns (accumulated_text, pending_tool_call | None). Output is gated by _abandon_pending (H7)."""
        parts: list[str] = []
        pending = None
        requested_at = time.monotonic()
        first_text = True
        caption_at = 0.0
        if self._turn_metrics is not None and turn_id:
            self._turn_metrics.record_llm_request_start(turn_id)
        async for ev in self._backend.stream_with_tools(messages, tools=TOOL_SPECS, session_id="", turn_id=turn_id):
            if ev.type == StreamEventType.TEXT and ev.text:
                if first_text:
                    first_text = False
                    if self._turn_metrics is not None and turn_id:
                        self._turn_metrics.record_llm_ttft(turn_id, (time.monotonic() - requested_at) * 1000)
                if self._abandon_pending:  # soft-abandon: drop pending output
                    continue
                parts.append(ev.text)
                now = time.monotonic()
                if now - caption_at >= 0.1:
                    self._publish_caption(pid, turn_id, "".join(parts))
                    caption_at = now
                await self.push_frame(
                    LLMTextFrame(text=ev.text, turn_id=turn_id, metadata={"context_id": pid, "latency_role": "speech"}),
                    FrameDirection.DOWNSTREAM,
                )
            elif ev.type == StreamEventType.TOOL_CALL and ev.tool_call is not None:
                pending = ev.tool_call
            elif ev.type == StreamEventType.DONE:
                break
        self._publish_caption(pid, turn_id, "".join(parts), final=True)
        await self.push_frame(LLMResponseEndFrame(context_id=pid, turn_id=turn_id), FrameDirection.DOWNSTREAM)
        return "".join(parts), pending

    # ---- tool dispatch methods -------------------------------------------
    async def _dispatch_kb(self, query, turn_id):
        from src.flows.knowledge_review_models import CallbackPriority, RefuseKind
        from src.processors.solo_kb import KbDecisionKind
        from src.prompts.phrase_bank import PHRASE_TYPE_2, load_knowledge_phrases

        episode = self._episodes.open_or_continue(turn_id=turn_id)
        try:
            decision = await self._kb.decide(
                query=query, recent_dialogue=self._history, session=episode.session, turn_id=turn_id
            )
        except Exception:  # review RuntimeError / router unroutable / backend error → safe INFRA refuse
            if not self._abandon_pending:
                await self._speak_phrase(load_knowledge_phrases(self._language)[PHRASE_TYPE_2], turn_id, role="refuse")
                self._enqueue_callback(
                    episode,
                    kind=RefuseKind.INFRA_FALLBACK,
                    priority=CallbackPriority.NORMAL,
                    out_of_scope_class=None,
                    user_text=query,
                    reason_text="kb_infra_fallback",
                )
            self._episodes.close()
            return
        if self._abandon_pending:
            return
        if decision.kind is KbDecisionKind.ANSWER:
            await self._speak_kb_answer(query, turn_id)
            self._episodes.close()
        elif decision.kind is KbDecisionKind.CLARIFY:
            await self._speak_phrase(decision.clarify_text, turn_id, role="clarify")
            self._episodes.mark_followup()  # increment at audio-start (H10)
        else:  # REFUSE
            await self._speak_phrase(decision.verbatim_text, turn_id, role="refuse")
            self._enqueue_callback(
                episode,
                kind=decision.refuse_kind,
                priority=decision.priority,
                out_of_scope_class=decision.out_of_scope_class,
                user_text=query,
                reason_text=decision.phrase_key or "kb_refuse",
            )
            self._episodes.close()

    async def _speak_kb_answer(self, query, turn_id):
        # [D] real answer: history + patient_info injected, qwen3.7-plus, streamed to TTS by the driver.
        messages = self._build_kb_answer_messages(query)
        pid = self._new_pid()
        parts = []
        caption_at = 0.0
        async for ev in self._kb_answer_backend.stream_with_tools(
            messages, tools=None, session_id="", turn_id=turn_id, model="qwen3.7-plus"
        ):
            if ev.type == StreamEventType.TEXT and ev.text and not self._abandon_pending:
                parts.append(ev.text)
                now = time.monotonic()
                if now - caption_at >= 0.1:
                    self._publish_caption(pid, turn_id, "".join(parts))
                    caption_at = now
                await self.push_frame(
                    LLMTextFrame(
                        text=ev.text, turn_id=turn_id, metadata={"context_id": pid, "latency_role": "knowledge_answer"}
                    ),
                    FrameDirection.DOWNSTREAM,
                )
            elif ev.type == StreamEventType.DONE:
                break
        self._publish_caption(pid, turn_id, "".join(parts), final=True)
        await self.push_frame(LLMResponseEndFrame(context_id=pid, turn_id=turn_id), FrameDirection.DOWNSTREAM)
        if parts:
            self._history.append({"role": "assistant", "content": "".join(parts)})

    async def _speak_phrase(self, text, turn_id, *, role):
        pid = self._new_pid()
        parts = []
        for chunk in _chunk(text or "", 20):
            if self._abandon_pending:
                break
            parts.append(chunk)
            await self.push_frame(
                LLMTextFrame(text=chunk, turn_id=turn_id, metadata={"context_id": pid, "latency_role": role}),
                FrameDirection.DOWNSTREAM,
            )
        self._publish_caption(pid, turn_id, "".join(parts), final=True)
        await self.push_frame(LLMResponseEndFrame(context_id=pid, turn_id=turn_id), FrameDirection.DOWNSTREAM)
        if text:
            self._history.append({"role": "assistant", "content": text})

    def _enqueue_callback(self, episode, *, kind, priority, out_of_scope_class, user_text, reason_text):
        if self._callback_writer is None:
            return
        self._callback_writer.enqueue(
            flow_state=episode.flow_state,
            call_id=self._call_id,
            kind=kind,
            out_of_scope_class=out_of_scope_class,
            priority=priority,
            user_text=user_text,
            reason_text=reason_text,
        )

    async def _dispatch_callback(self, args, turn_id, untrusted_turn):
        if untrusted_turn:
            return "ok"  # M3/A4: never enqueue on speculative or unconfirmed text
        from src.flows.knowledge_review_models import CallbackPriority, EscalationKind

        episode = self._episodes.open_or_continue(turn_id=turn_id)  # ensures current_session for enqueue
        priority = CallbackPriority.HIGH if args.get("priority") == "high" else CallbackPriority.NORMAL
        self._enqueue_callback(
            episode,
            kind=EscalationKind.EXIT_ESCALATE,
            priority=priority,
            out_of_scope_class=None,
            user_text="",
            reason_text=args.get("reason", "callback"),
        )
        self._episodes.close()
        if self._language == "en":
            return "A callback request has been registered. Continue the follow-up in English."
        return "已为患者登记回电需求,请继续按流程往下聊。"

    async def _dispatch_end_call(self, msgs, untrusted_turn) -> _EndCallOutcome:
        if untrusted_turn:
            return _EndCallOutcome.REFUSED_SPECULATIVE  # M3/A4: never end on unconfirmed text
        # No capture wired → no backstop, arm immediately (pre-B2 behavior).
        # B2 optimistic accept: a snapshot that already covers all ★ fields is
        # trusted as-is — we assume the extractor never downgrades a confirmed
        # field given a longer transcript (not enforced; an extractor that
        # retracts fields would slip a stale "covered" through here). The
        # accept path adds zero latency.
        if self._capture is not None and not self._capture.all_required_covered():
            # The per-turn schedule() is fire-and-forget, so at end_call time
            # the snapshot structurally lags the current turn — the patient may
            # have answered the final ★ field in the very utterance that
            # triggered end_call. Refresh ONCE before rejecting: history here
            # already includes this turn's user answer and the streamed
            # assistant text. Latency trade-off: the awaited extraction is paid
            # only on the would-reject path, and the C2 cap (≤2 end_calls/turn,
            # enforced at the _segmenting_loop call site) bounds it.
            try:
                await self._capture.extract_now(self._render_transcript())
            except Exception:
                # An extraction-backend failure must not kill the turn (the
                # assistant tool_call is already in msgs — dying here leaves it
                # dangling and the bot mute). Fall through to the backstop with
                # the STALE missing list: re-asking an already-answered field is
                # annoying, but the call stays alive.
                logger.exception("solo.end_call_refresh_error")
            if not self._capture.all_required_covered():
                # B4 escape valve: this is a genuine would-reject (refreshed
                # snapshot still missing ★). If the backstop has already pushed
                # back the max number of times this CALL, honor the end_call now
                # despite the missing fields — a patient who can't/won't answer a
                # required field can still end the call (breaks the G03 livelock).
                # UX/latency trade-off: at most 2 re-asks per call, then the call
                # can always end. The counter increments ONLY here, on an actual
                # rejection that is NOT escaped — ARMED-because-covered and
                # REFUSED_SPECULATIVE never reach this point and never touch it.
                missing_fields = self._capture.missing_required()
                if self._endcall_backstop_rejections >= _MAX_ENDCALL_BACKSTOP_REJECTIONS:
                    # warning, not info: a call ending with a required ★ still
                    # unfilled is a degraded completion ops should see above the
                    # info firehose.
                    logger.warning(
                        "solo.endcall_backstop_exhausted",
                        call_id=self._call_id,
                        missing=missing_fields,
                    )
                    self._arm_terminal()
                    return _EndCallOutcome.ARMED
                self._endcall_backstop_rejections += 1
                # Missing list from the freshest snapshot we have: refreshed,
                # or stale if the refresh above failed.
                missing = "、".join(missing_fields)
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": _last_tool_id(msgs),
                        "content": f"还差这些必填没问到:{missing}。请先继续问清楚,不要结束通话。",
                    }
                )
                return _EndCallOutcome.BACKSTOP_INCOMPLETE
        self._arm_terminal()
        return _EndCallOutcome.ARMED

    async def _execute_terminal_hook(self):
        from src.pipeline.frames import EndFrame

        if not self._pending_terminal:
            return
        self._pending_terminal = False
        if self._capture is not None:
            try:
                await self._capture.extract_now(self._render_transcript())  # H11: input includes last turn
            except Exception:
                # The terminal must NEVER strand: this runs inside the system-
                # frame input task (BotStopped) or the watchdog — a raise here
                # kills that task, the EndFrame never goes out, and the call
                # hangs forever. Proceed with the stale snapshot instead.
                logger.exception("solo.terminal_extract_error")
            payload = self._capture.terminal_payload(patient_info=self._patient_info, turn_count=len(self._history))
        else:
            payload = {
                "collected_fields": {},
                "patient_info": self._patient_info,
                "conversation_turns": len(self._history),
                "nodes_completed": [],
            }
        if self._call_tracer is not None:
            fn = getattr(self._call_tracer, "trace_form_submitted", None)
            if callable(fn):
                fn(payload)
        if self._lifecycle is not None:
            self._lifecycle.schedule_post_call_once("completed")
        await self.push_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    async def _terminal_watchdog_run(self):
        await asyncio.sleep(5.0)
        if self._pending_terminal:
            await self._execute_terminal_hook()  # force-fire (H2: empty-closing-text hang)

    def _cancel_watchdog(self):
        wd = self._terminal_watchdog
        self._terminal_watchdog = None
        if wd is not None and not wd.done():
            wd.cancel()

    def _arm_terminal(self):
        self._pending_terminal = True
        self._terminal_watchdog = asyncio.create_task(self._terminal_watchdog_run())

    def _disarm_terminal(self):
        self._pending_terminal = False
        self._cancel_watchdog()
        # B3: fold farewell clearing in here so every existing _disarm_terminal
        # caller (_handle_user_turn, on_flush) clears the suspect+timer for free.
        # SAFE because the only callers are user-activity sites (a new turn, or a
        # flush/barge-in) — both legitimately abort a pending farewell. The
        # farewell-ARM path in handle_system_frame must NOT call _disarm_terminal
        # (it would cancel the timer it just started); it sets the flag and
        # starts the timer directly. Verified: no caller disarms a just-armed
        # farewell timer.
        self._clear_farewell()

    def _clear_farewell(self):
        self._farewell_suspect = False
        self._cancel_farewell_timer()

    def _cancel_farewell_timer(self):
        t = self._farewell_timer
        self._farewell_timer = None
        if t is not None and not t.done():
            t.cancel()

    def _build_kb_answer_messages(self, query):
        from src.playground.kb_tool import _PACK, _SAFETY_PROMPT, _load_full_kb

        speech = (_PACK / "speech/knowledge_qa.md").read_text(encoding="utf-8")
        kb = _load_full_kb()
        kb_block = "\n".join(f"- (id:{it['id']}) Q:{it.get('question', '')} A:{it.get('answer', '')}" for it in kb)
        system = f"{speech}\n\n{_SAFETY_PROMPT}\n\n[知识库]\n{kb_block}\n\n[已知患者信息]\n{self._patient_info}"
        if self._language == "en":
            from src.prompts.solo_english import KB_INSTRUCTION

            system += "\n\n" + KB_INSTRUCTION
        return [{"role": "system", "content": system}, *self._history, {"role": "user", "content": query}]

    # ---- cancellation (Task 10 expands) ----------------------------------
    async def _cancel_turn(self):
        if self._turn_task is not None and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._turn_task = None

    async def handle_system_frame(self, frame, direction):
        if isinstance(frame, BotStartedSpeakingFrame):
            self._opening_waiting_for_audio = False
        if isinstance(frame, UserStartedSpeakingFrame):
            # Initial microphone noise is not an interruption of audio that
            # has not started yet. Confirmed user turns still preempt above.
            if not self._opening_waiting_for_audio:
                self._abandon_pending = True  # H7 soft-abandon output gate
            # B3: ANY sound clears a pending farewell — conservative, we never
            # hang up on a patient who is making noise (even a cough). A real
            # turn that follows will re-confirm via _handle_user_turn.
            self._clear_farewell()
        if isinstance(frame, BotStoppedSpeakingFrame):
            if self._pending_terminal:
                # Existing terminal path (flash properly called end_call). This
                # takes priority and, via the NOT-_pending_terminal guard below,
                # ensures the farewell timer is NEVER started in the same breath
                # → no double EndFrame even when the closing text also looks like
                # a goodbye. Do NOT _disarm here (that would cancel the watchdog
                # we still want as backstop until the hook runs); _cancel_watchdog
                # is the targeted call.
                self._cancel_watchdog()  # M1: closing arrived; the backstop is no longer needed
                await self._execute_terminal_hook()
            elif self._farewell_suspect:
                # B3: flash spoke a goodbye but forgot end_call. Start the silent
                # grace timer (cancel any prior). Do NOT call _disarm_terminal in
                # this path — it would clear the suspect/timer we just set up.
                self._cancel_farewell_timer()
                self._farewell_timer = asyncio.create_task(self._farewell_grace_run())
        await super().handle_system_frame(frame, direction)

    async def _farewell_grace_run(self):
        await asyncio.sleep(_FAREWELL_GRACE_S)
        # Belt-and-braces against cancellation races: re-check the flag on wake.
        # Also require no user speech since arming (suspect still set — any user
        # activity clears it) and no in-flight turn task.
        if not self._farewell_suspect:
            return
        if self._turn_task is not None and not self._turn_task.done():
            return  # a turn is running (e.g. a late KB answer) — not over yet
        logger.info("solo.farewell_backstop_fired", call_id=self._call_id)
        self._farewell_suspect = False
        # Execute the hook DIRECTLY (no _arm_terminal): we are already past the
        # grace and terminating now, so the 5s watchdog _arm_terminal would add
        # is pointless — there is no BotStopped to wait for. Set _pending_terminal
        # so _execute_terminal_hook's own guard treats this as a real terminal
        # (and so a stray later fire finds it already reset → no double exec).
        self._pending_terminal = True
        await self._execute_terminal_hook()

    async def cancel_inflight(self):
        await super().cancel_inflight()
        await self._cancel_turn()

    async def on_flush(self, frame):
        await self._cancel_turn()
        self._current_turn_id = ""
        self._disarm_terminal()  # C1: a real barge-in interrupting the closing aborts the pending end
        # A4: continuation-cancel means the detector cancelled this turn because the
        # user kept talking — it will RE-EMIT the accumulated text (which contains
        # this turn's user message) as a new UserTurnCompleteFrame. Pop the stale
        # trailing user entry (only if that turn produced no assistant output yet)
        # so the same content isn't injected into flash context twice and doesn't
        # duplicate in solo_capture's transcript. Other flush sources (real
        # barge-in etc.) must NOT pop: the patient's question being answered
        # would be deleted.
        if (
            getattr(frame, "source", "") == "continuation_cancel"
            and self._history
            and self._history[-1]["role"] == "user"
        ):
            self._history.pop()
        await super().on_flush(frame)

    # ---- helpers ----------------------------------------------------------
    def _new_pid(self) -> str:
        return str(uuid.uuid4())

    def _render_transcript(self) -> str:
        who = {"user": "患者", "assistant": "客服"}
        return "\n".join(f"{who.get(m['role'], m['role'])}：{m['content']}" for m in self._history)


def _chunk(text: str, n: int):
    for i in range(0, len(text), n):
        yield text[i : i + n]


def _last_tool_id(msgs) -> str:
    for m in reversed(msgs):
        for tc in m.get("tool_calls", []) or []:
            return tc["id"]
    return "unknown"
