"""ECS full voice pipeline implementation.

The filename is historical. The active entrypoint is ``server.main``, launched
by ``pinchi-agent.service`` on ECS.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from src.core.config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("debug")

# Suppress noisy libs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _build_vad_processor(settings):
    import numpy as np

    from src.backends import EnergyVAD, PVADBackend, get_ecapa_singleton
    from src.backends.vad_pvad import load_onnx_session
    from src.core.logging import get_logger
    from src.processors.vad_processor import VADProcessor

    log = get_logger(__name__)

    primary = EnergyVAD(energy_threshold=settings.noise_min_threshold)
    shadows = []
    if "pvad" in settings.vad_shadow_backends:
        session = load_onnx_session(settings.vad_pvad_model_path)
        if session is None:
            log.warning("pvad.shadow.skipped", reason="onnx_load_failed")
        else:
            try:
                ecapa = get_ecapa_singleton()
                if not ecapa.available:
                    log.warning("pvad.shadow.skipped", reason="ecapa_unavailable")
                else:
                    preloaded_voiceprint = None
                    voiceprint_path = settings.vad_pvad_preloaded_voiceprint_path.strip()
                    if voiceprint_path:
                        path = Path(voiceprint_path)
                        if path.exists():
                            preloaded_voiceprint = np.load(path).astype(np.float32)
                            log.info("pvad.shadow.preloaded_voiceprint", path=str(path))
                        else:
                            log.warning("pvad.shadow.voiceprint_not_found", path=str(path))
                    pvad = PVADBackend(
                        ecapa=ecapa,
                        onnx_session=session,
                        threshold=settings.vad_pvad_threshold,
                        smoothing_alpha=settings.vad_pvad_smoothing_alpha,
                        enroll_target_ms=settings.vad_pvad_enroll_target_ms,
                        enroll_timeout_ms=settings.vad_pvad_enroll_timeout_ms,
                        encode_timeout_ms=settings.vad_pvad_encode_timeout_ms,
                        max_consecutive_errors=settings.vad_pvad_max_consecutive_errors,
                        min_embedding_magnitude=settings.vad_pvad_min_embedding_magnitude,
                        preloaded_voiceprint=preloaded_voiceprint,
                    )
                    shadows.append(pvad)
            except Exception as exc:
                log.warning("pvad.shadow.init_failed", error=str(exc))

    return VADProcessor(
        "vad",
        primary_backend=primary,
        shadow_backends=shadows,
        speech_start_threshold_ms=40,
        speech_end_threshold_ms=settings.silence_threshold_ms,
        enroll_pre_roll_ms=settings.vad_pvad_enroll_pre_roll_ms,
        pvad_gates_primary=settings.vad_pvad_gates_primary,
    )


async def _prewarm_local_agent(
    *,
    speech_backend,
    fast_judge_backend,
    analyzer_backend,
    tts_backend,
) -> None:
    """Warm the heavy backends before a user speaks the first turn."""
    started_at = time.monotonic()
    logger.info("Starting agent warmup...")

    warmers: list[tuple[str, object]] = [
        ("speech_llm", speech_backend.warmup()),
        ("fast_judge_llm", fast_judge_backend.warmup()),
        ("analyzer_llm", analyzer_backend.warmup()),
        ("tts_pool", tts_backend.setup()),
    ]
    # ECAPA is a sync PyTorch load (~80MB ckpt); only pay the cost when the
    # pVAD shadow lane is on. Offloaded to a thread so it runs alongside the
    # LLM/TTS warmers without blocking the event loop.
    if "pvad" in settings.vad_shadow_backends:
        from src.backends import get_ecapa_singleton

        warmers.append(("ecapa", asyncio.to_thread(get_ecapa_singleton)))

    results = await asyncio.gather(*(coro for _, coro in warmers), return_exceptions=True)
    for (name, _), result in zip(warmers, results, strict=False):
        if isinstance(result, Exception):
            logger.warning("warmup.failed backend=%s error=%s", name, result)

    elapsed_ms = (time.monotonic() - started_at) * 1000
    logger.info("Agent warmup complete in %.0fms", elapsed_ms)


def build_brain_processor(
    *,
    agent_brain,
    speech_backend,
    fast_judge_backend,
    history,
    flow_runtime,
    turn_metrics,
    debug_publish,
    sip_transfer,
    lifecycle,
    call_tracer,
    playout_gate,
    memory_scheduler,
    short_term_memory_scheduler,
    sop_progress_scheduler,
    kb_three_stage=None,
    kb_answer_backend=None,
    capture=None,
    callback_writer=None,
    call_id="",
    patient_info=None,
    system_prompt="",
    language="zh",
):
    """Select the conversational brain at the pipeline seam.
    AGENT_BRAIN=solo → SoloDriverProcessor; otherwise the dual-LLM LLMProcessor."""
    if agent_brain == "solo":
        from src.processors.solo_driver import SoloDriverProcessor

        return SoloDriverProcessor(
            "llm-solo",
            llm_backend=speech_backend,
            system_prompt=system_prompt,
            language=language,
            kb_three_stage=kb_three_stage,
            kb_answer_backend=kb_answer_backend,
            capture=capture,
            callback_writer=callback_writer,
            call_id=call_id,
            patient_info=patient_info or {},
            turn_metrics=turn_metrics,
            lifecycle=lifecycle,
            call_tracer=call_tracer,
            playout_gate=playout_gate,
        )
    from src.processors.llm_processor import LLMProcessor

    return LLMProcessor(
        "llm",
        llm_backend=speech_backend,
        fast_judge_backend=fast_judge_backend,
        history=history,
        flow_runtime=flow_runtime,
        turn_metrics=turn_metrics,
        debug_publish=debug_publish,
        sip_transfer=sip_transfer,
        lifecycle=lifecycle,
        call_tracer=call_tracer,
        playout_gate=playout_gate,
        memory_scheduler=memory_scheduler,
        short_term_memory_scheduler=short_term_memory_scheduler,
        sop_progress_scheduler=sop_progress_scheduler,
    )


async def main() -> None:
    from src.backends.llm_dashscope import QwenDashScopeBackend
    from src.backends.llm_factory import get_speech_llm_backend
    from src.backends.stt_aliyun import AliyunSTTBackend
    from src.backends.tts_cosyvoice_duplex import CosyVoiceDuplexBackend
    from src.pipeline.pipeline import Pipeline
    from src.pipeline.runner import PipelineRunner
    from src.processors.stt_processor import STTProcessor
    from src.processors.tts_duplex_processor import TTSDuplexProcessor
    from src.processors.turn_detector.backends import SmartTurnV3Backend
    from src.processors.turn_detector.engine import TurnDecisionEngine
    from src.processors.turn_detector.processor import TurnDetectorProcessor
    from src.session.conversation_history import ConversationHistory
    from src.session.playback_ledger import PlaybackLedger
    from src.session.playout_gate import PlayoutGate
    from src.transport.livekit_input import LiveKitInputTransport
    from src.transport.livekit_output import LiveKitOutputTransport

    livekit_url = settings.livekit_url.strip()
    api_key = settings.livekit_api_key.strip()
    api_secret = settings.livekit_api_secret.strip()
    room_name = settings.livekit_room_name.strip() or "debug-full"
    agent_identity = settings.livekit_agent_identity.strip() or "agent"

    if not livekit_url:
        raise RuntimeError("LIVEKIT_URL is required")
    if not api_key or not api_secret:
        raise RuntimeError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET are required")

    # ── Connect to room ──
    token = AccessToken(api_key, api_secret)
    token.with_identity(agent_identity)
    token.with_grants(VideoGrants(room_join=True, room=room_name))

    room = rtc.Room()
    await room.connect(livekit_url, token.to_jwt())
    logger.info("Connected to room: %s", room.name)

    # ── C-stage flow layer ──
    from src.flows.ablation_switches import is_short_term_memory_analyzer_enabled, is_sop_progress_analyzer_enabled
    from src.flows.context_builder import ContextBuilder
    from src.flows.memory_extractor import CallMemoryExtractor, MemoryExtractorScheduler
    from src.flows.pinchi_flow_config import build_agent_config
    from src.flows.runtime import FlowRuntime
    from src.flows.short_term_memory import ShortTermMemoryAnalyzer, ShortTermMemoryScheduler
    from src.flows.sop_progress import SopProgressAnalyzer, SopProgressScheduler
    from src.flows.tool_executor import ToolExecutor
    from src.flows.tools.global_tools import transfer_to_human
    from src.prompts.prompt_pack import PromptPackLoader, PromptRegistry

    agent_config = build_agent_config()
    context_builder = ContextBuilder()
    tool_executor = ToolExecutor()
    tool_executor.register_many({"transfer_to_human": transfer_to_human})
    prompt_registry = PromptRegistry(PromptPackLoader().load_active())

    # ── Shared session objects ──
    history = ConversationHistory(system_prompt=settings.system_prompt)

    flow_runtime = FlowRuntime(
        agent_config=agent_config,
        context_builder=context_builder,
        tool_executor=tool_executor,
        conversation_history=history,
        prompt_registry=prompt_registry,
    )
    flow_runtime.set_call_id(
        room_name=room_name,
        call_start_iso=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )

    # Speech LLM — selected by SPEECH_LLM_PROVIDER (default "qwen" → DashScope).
    # Judge LLM stays pinned to DashScope qwen below as the §4.3.1 control.
    speech_backend = get_speech_llm_backend()

    fast_judge_backend = QwenDashScopeBackend(
        model=settings.fast_judge_model,
        max_tokens=settings.fast_judge_max_tokens,
        temperature=settings.fast_judge_temperature,
        enable_thinking=False,
    )

    # Analyzer backend (background memory extraction / short-term memory / sop_progress)
    analyzer_backend = QwenDashScopeBackend(
        model=settings.analyzer_model,
        max_tokens=settings.analyzer_max_tokens,
        temperature=settings.analyzer_temperature,
        enable_thinking=False,
    )

    memory_extractor = CallMemoryExtractor(
        backend=analyzer_backend,
        model=settings.analyzer_model,
        max_tokens=settings.analyzer_max_tokens,
        temperature=settings.analyzer_temperature,
    )
    memory_scheduler = MemoryExtractorScheduler(flow=flow_runtime, extractor=memory_extractor)
    short_term_memory_scheduler = (
        ShortTermMemoryScheduler(
            flow=flow_runtime,
            analyzer=ShortTermMemoryAnalyzer(
                backend=analyzer_backend,
                model=settings.analyzer_model,
                max_tokens=settings.analyzer_max_tokens,
                temperature=settings.analyzer_temperature,
            ),
        )
        if is_short_term_memory_analyzer_enabled()
        else None
    )
    sop_progress_scheduler = (
        SopProgressScheduler(
            flow=flow_runtime,
            analyzer=SopProgressAnalyzer(
                backend=analyzer_backend,
                model=settings.analyzer_model,
                max_tokens=settings.analyzer_max_tokens,
                temperature=settings.analyzer_temperature,
            ),
        )
        if is_sop_progress_analyzer_enabled()
        else None
    )

    # Build turn detection backend (SmartTurn only)
    smart_turn = SmartTurnV3Backend(model_path=settings.smart_turn_model_path)
    engine = TurnDecisionEngine(backends=[smart_turn])
    tts_backend = CosyVoiceDuplexBackend()
    _prewarm_start = time.monotonic()
    prewarm_task = asyncio.create_task(
        _prewarm_local_agent(
            speech_backend=speech_backend,
            fast_judge_backend=fast_judge_backend,
            analyzer_backend=analyzer_backend,
            tts_backend=tts_backend,
        )
    )

    # ── Wait for user ──
    participant_event = asyncio.Event()
    primary_holder: list[rtc.RemoteParticipant | None] = [None]

    @room.on("participant_connected")
    def on_participant(p: rtc.RemoteParticipant) -> None:
        # Skip other agents
        if p.identity.startswith("agent"):
            logger.info("Skipping agent participant: %s", p.identity)
            return
        logger.info("User joined: %s", p.identity)
        primary_holder[0] = p
        participant_event.set()

    # Check already-joined participants
    for p in room.remote_participants.values():
        if not p.identity.startswith("agent"):
            logger.info("User already in room: %s", p.identity)
            primary_holder[0] = p
            participant_event.set()

    if not participant_event.is_set():
        logger.info("Waiting for user to join room '%s'...", room_name)
        logger.info("Open your frontend console and connect to the room")

    await asyncio.wait_for(participant_event.wait(), timeout=settings.live_participant_wait_s)
    primary = primary_holder[0]
    assert primary is not None
    await prewarm_task
    prewarm_elapsed_ms = (time.monotonic() - _prewarm_start) * 1000
    logger.info("Building pipeline for user: %s", primary.identity)

    playout_gate = PlayoutGate()
    ledger = PlaybackLedger(
        history=history,
        node_id_resolver=flow_runtime.resolve_context_node,
        playout_gate=playout_gate,
    )

    # ── Turn metrics collector (publishes to frontend via LiveKit data channel) ──
    import json
    import os as _os

    from src.pipeline.turn_metrics import TurnMetricsCollector

    _metrics_reliable = _os.environ.get("EVAL_DATA_CHANNEL_UNRELIABLE", "").lower() in ("", "0", "false")

    # Map wire topic → event_type so AssertOn.last("...Event") matches on the
    # VC side. The _publish callback is shared between the TurnMetricsCollector
    # and LLMProcessor._emit_debug (which sends dual_llm_debug payloads), so
    # we must not blanket-stamp everything as "TurnMetricsEvent".
    _TOPIC_TO_EVENT_TYPE = {
        "turn_metrics": "TurnMetricsEvent",
        "dual_llm_debug": "JudgeDecisionEvent",
    }

    def _publish_metrics(metrics: dict) -> None:
        """Send metrics / debug JSON to all participants via LiveKit data channel.

        Stamps event_type so AssertOn.last('TurnMetricsEvent') matches on the
        VC side (the raw _serialize() / _emit_debug dicts only carry `topic`).
        """
        try:
            topic = metrics.get("topic", "")
            event_type = _TOPIC_TO_EVENT_TYPE.get(topic)
            wire = {**metrics}
            if event_type and "event_type" not in wire:
                wire["event_type"] = event_type
            data = json.dumps(wire, ensure_ascii=False).encode("utf-8")
            asyncio.get_event_loop().create_task(room.local_participant.publish_data(data, reliable=_metrics_reliable))
            logger.info("[metrics] %s", " ".join(f"{k}={v}" for k, v in metrics.items() if k != "topic"))
        except Exception as e:
            logger.warning("[metrics] publish failed: %s", e)

    turn_metrics = TurnMetricsCollector(on_complete=_publish_metrics)
    lk_out = LiveKitOutputTransport(
        "lk-out",
        room=room,
        playback_ledger=ledger,
        playout_gate=playout_gate,
        turn_metrics=turn_metrics,
    )

    # ── Transfer dependencies ──
    from src.core.call_tracer import CallTracer
    from src.session.lifecycle import SessionLifecycleHelper

    call_tracer = CallTracer(session_id=room.name, patient_id="dev-patient")
    call_tracer.trace_call_started()

    # ── Solo-arch collaborators (built only when AGENT_BRAIN=solo) ──
    solo_capture = None
    solo_kb = None
    solo_callback_writer = None
    solo_prompt = ""
    if settings.agent_brain == "solo":
        import hashlib
        from pathlib import Path as _SoloPath

        from src.flows.knowledge_review_runner import KnowledgeReviewRunner
        from src.handoff.callback_queue import get_callback_queue_writer
        from src.processors.solo_capture import CaptureStore
        from src.processors.solo_kb import KbThreeStage
        from src.prompts.solo_system import build_solo_production_prompt

        _kb_bytes = (_SoloPath(__file__).resolve().parents[1] / "config" / "knowledge_base.json").read_bytes()
        _kb_hash = "sha256:" + hashlib.sha256(_kb_bytes).hexdigest()
        solo_kb = KbThreeStage(
            review_runner=KnowledgeReviewRunner.from_manifest(backend=speech_backend),
            kb_hash=_kb_hash,
            language=settings.solo_language,
        )
        solo_capture = CaptureStore(backend=analyzer_backend, on_update=_publish_metrics)
        solo_callback_writer = get_callback_queue_writer()
        solo_prompt = build_solo_production_prompt(settings.solo_language)

    lifecycle = SessionLifecycleHelper(
        session_id=room.name,
        patient_id="dev-patient",
        history=history,
        call_tracer=call_tracer,
        flow_state_fn=(
            (lambda: solo_capture.flow_state_view(patient_info={"patient_id": "dev-patient"}, turn_count=0))
            if settings.agent_brain == "solo"
            else (lambda: flow_runtime._state)
        ),
        turn_log=None,
    )

    async def _sip_transfer_dev(destination: str) -> None:
        logger.info("DEV: sip_transfer called (no-op)", destination=destination)

    from src.pipeline.events import AgentReadyEvent, EventBus, PipelineStartedEvent

    bus = EventBus()
    flow_runtime._bus = bus

    if settings.vad_shadow_trace_enabled and settings.vad_shadow_backends:
        from pathlib import Path

        from src.observability.pvad_trace import PVADTraceWriter

        trace_path = Path(settings.vad_shadow_trace_dir) / f"pvad_trace_{room.name}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        bus.subscribe(PVADTraceWriter(trace_path))

    # ── Build pipeline ──
    pipeline = Pipeline(
        "main",
        [
            LiveKitInputTransport("lk-in", room=room, primary_participant=primary),
            _build_vad_processor(settings),
            STTProcessor("stt", stt_backend=AliyunSTTBackend()),
            TurnDetectorProcessor(
                "turn-detector",
                engine=engine,
                smart_turn_backend=smart_turn,
                history=history,
                turn_metrics=turn_metrics,
                final_timeout_ms=settings.final_timeout_ms,
                asr_final_grace_ms=settings.asr_final_grace_ms,
                aec_warmup_s=settings.aec_warmup_s,
                min_words_during_bot=settings.min_words_during_bot,
            ),
            build_brain_processor(
                agent_brain=settings.agent_brain,
                speech_backend=speech_backend,
                fast_judge_backend=fast_judge_backend,
                history=history,
                flow_runtime=flow_runtime,
                turn_metrics=turn_metrics,
                debug_publish=_publish_metrics,
                sip_transfer=_sip_transfer_dev,
                lifecycle=lifecycle,
                call_tracer=call_tracer,
                playout_gate=playout_gate,
                memory_scheduler=memory_scheduler,
                short_term_memory_scheduler=short_term_memory_scheduler,
                sop_progress_scheduler=sop_progress_scheduler,
                kb_three_stage=solo_kb,
                kb_answer_backend=speech_backend,
                capture=solo_capture,
                callback_writer=solo_callback_writer,
                call_id=room.name,
                patient_info={"patient_id": "dev-patient"},
                system_prompt=solo_prompt,
                language=settings.solo_language,
            ),
            TTSDuplexProcessor(
                "tts",
                tts_backend=tts_backend,
                playback_ledger=ledger,
                turn_metrics=turn_metrics,
            ),
            lk_out,
        ],
    )

    from src.pipeline.observer import LoggingObserver
    from src.transport.livekit_events import DataChannelObserver

    logging_observer = LoggingObserver()
    data_channel_observer = DataChannelObserver(room)

    # PipelineRunner.__init__ subscribes observers immediately so the
    # PipelineStarted/AgentReady emits below are delivered.
    # TurnMetricsAggregator is driven by direct record_* calls from processors;
    # bus-subscribing it would double-record every checkpoint (later overwrite
    # biases timestamps).
    runner = PipelineRunner(pipeline, bus=bus, observers=[logging_observer, data_channel_observer])

    bus.emit(PipelineStartedEvent(pipeline_id="main", session_id=room.name))
    bus.emit(
        AgentReadyEvent(
            prewarm_duration_ms=prewarm_elapsed_ms,
            session_id=room.name,
        )
    )

    logger.info("Starting pipeline (VAD → STT → TurnDetector → LLM[FlowRuntime] → TTSDuplex)...")
    logger.info(
        "Flow: %s", " → ".join(["greeting", "identity", "symptom", "programming", "daily_life", "insurance", "closing"])
    )
    logger.info("Speak into your microphone!")

    try:
        await runner.run()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception:
        logger.exception("Pipeline crashed")
    finally:
        await room.disconnect()
        logger.info("Disconnected from room")


if __name__ == "__main__":
    asyncio.run(main())
