# src/backends/vad_pvad.py
"""Personalized VAD backend (FireRedChat pVAD).

State machine for live enrollment:

    WAITING_FOR_GREETING_END
       ↓ BotStoppedSpeaking (first one)
    READY_FOR_ENROLL
       ↓ on_speech_started
    COLLECTING (cross-turn accumulation)
       ↓ buffer ≥ enroll_target_ms
    ENCODING (async, via thread pool)
       ↓ ECAPA returns valid embedding
    READY → 10ms-window pVAD inference
       ↓ any-stage failure / timeout
    FAILED → permanent unavailable

ONNX wiring lands in Task 7. In Task 6 (skeleton) _run_inference is a
stub that returns is_speech=False so the state machine can be tested
without the real model.
"""

from __future__ import annotations

import enum
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from src.core.logging import get_logger
from src.processors.vad_processor import VADResult

logger = get_logger(__name__)


class PVADState(str, enum.Enum):
    WAITING_FOR_GREETING_END = "waiting_greeting"
    READY_FOR_ENROLL = "ready_for_enroll"
    COLLECTING = "collecting"
    ENCODING = "encoding"
    READY = "ready"
    FAILED = "pvad_enroll_failed"
    TIMEOUT = "pvad_enroll_timeout"
    DEAD = "pvad_dead"


@dataclass
class _RuntimeStats:
    speech_duration_ms: float = 0.0
    silence_duration_ms: float = 0.0
    consecutive_errors: int = 0


_DEFAULT_THRESHOLD = 0.5
_DEFAULT_SMOOTHING_ALPHA = 0.8
_SAMPLE_RATE = 16000
_BYTES_PER_SAMPLE = 2  # int16
_WINDOW_SAMPLES = 160  # 10ms @ 16k
_WINDOW_BYTES = _WINDOW_SAMPLES * _BYTES_PER_SAMPLE  # 320
_MEL_SHAPE = (1, 80, 15)
_GRU_SHAPE = (2, 1, 256)


class PVADBackend:
    backend_name = "pvad"

    def __init__(
        self,
        ecapa,
        onnx_session=None,
        threshold: float = _DEFAULT_THRESHOLD,
        smoothing_alpha: float = _DEFAULT_SMOOTHING_ALPHA,
        enroll_target_ms: int = 1000,
        enroll_timeout_ms: int = 30000,
        encode_timeout_ms: int = 5000,
        max_consecutive_errors: int = 50,
        min_embedding_magnitude: float = 1e-3,
        preloaded_voiceprint: np.ndarray | None = None,
        _test_mode: bool = False,
    ) -> None:
        self._ecapa = ecapa
        self._onnx_session = onnx_session
        self._threshold = threshold
        self._smoothing_alpha = smoothing_alpha
        self._enroll_target_bytes = int(enroll_target_ms / 1000 * _SAMPLE_RATE) * _BYTES_PER_SAMPLE
        self._enroll_timeout_s = enroll_timeout_ms / 1000.0
        self._encode_timeout_s = encode_timeout_ms / 1000.0
        self._max_consecutive_errors = max_consecutive_errors
        self._min_embedding_magnitude = min_embedding_magnitude
        self._test_mode = _test_mode

        self._lock = threading.Lock()
        self._enroll_state = PVADState.WAITING_FOR_GREETING_END
        self._enroll_buffer = bytearray()
        self._enroll_timeout_at_mono: float | None = None
        self._collecting_patient_audio = False
        self._voiceprint: np.ndarray | None = None
        self._encode_future: Future | None = None
        self._encode_started_mono: float | None = None
        self._encode_generation: int = 0  # invalidates stale workers (codex finding 2)
        self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvad-encode")

        # Task-7: sliding inference window, EMA probability smoothing,
        # recurrent model state, and runtime error budget.
        self._audio_buffer = bytearray()
        self._smoothed_prob: float | None = None
        self._mel_state = np.zeros(_MEL_SHAPE, dtype=np.float32)
        self._gru_state = np.zeros(_GRU_SHAPE, dtype=np.float32)
        self._stats = _RuntimeStats()
        # Init-failure flag distinct from enroll-failure (spec §6.2):
        # init failures report source="pvad_unavailable"; enroll failures keep
        # source="pvad_enroll_failed" so JSONL traces stay diagnostic.
        self._init_failed = False

        # Init-time gates (codex finding 3): if ECAPA or ONNX session isn't
        # available, fail-fast to FAILED — never let pVAD reach READY in a
        # degraded mode that produces misleading "runtime errors" later.
        # _test_mode bypasses the ONNX gate so state-machine tests can run
        # without supplying a real session.
        if not getattr(ecapa, "available", True):
            self._enroll_state = PVADState.FAILED
            self._init_failed = True
            logger.warning("pvad.init.ecapa_unavailable")
        elif onnx_session is None and not _test_mode:
            self._enroll_state = PVADState.FAILED
            self._init_failed = True
            logger.warning("pvad.init.onnx_unavailable")
        elif preloaded_voiceprint is not None:
            self.preload_voiceprint(preloaded_voiceprint)

    # ── Public state introspection (for trace events / tests) ───────────────

    @property
    def state(self) -> PVADState:
        return self._enroll_state

    @property
    def enroll_progress_ms(self) -> int | None:
        # Locked snapshot: worker thread can flip COLLECTING→ENCODING and
        # clear the buffer between the two reads otherwise.
        with self._lock:
            if self._enroll_state != PVADState.COLLECTING:
                return None
            return int(len(self._enroll_buffer) / _BYTES_PER_SAMPLE / _SAMPLE_RATE * 1000)

    def preload_voiceprint(self, embedding: np.ndarray) -> None:
        """Load a fixed target-speaker embedding and skip live enrollment.

        Study 1 pVAD evaluations need a reproducible target speaker identity:
        the target voiceprint is generated once from enrollment audio and then
        reused across target, non-target, and noise cases.
        """

        if not isinstance(embedding, np.ndarray) or embedding.shape != (192,):
            raise ValueError(f"bad embedding shape: {getattr(embedding, 'shape', None)}")
        norm = float(np.linalg.norm(embedding))
        if norm < self._min_embedding_magnitude:
            raise ValueError("embedding magnitude below threshold")
        with self._lock:
            self._voiceprint = (embedding.astype(np.float32) / norm).astype(np.float32)
            self._enroll_state = PVADState.READY
            self._enroll_buffer.clear()
            self._collecting_patient_audio = False
            self._encode_future = None
            self._encode_started_mono = None
            self._mel_state = np.zeros(_MEL_SHAPE, dtype=np.float32)
            self._gru_state = np.zeros(_GRU_SHAPE, dtype=np.float32)
            self._smoothed_prob = None

    # ── Lifecycle hooks invoked by VADProcessor ─────────────────────────────

    def on_greeting_finished(self) -> None:
        with self._lock:
            if self._enroll_state != PVADState.WAITING_FOR_GREETING_END:
                return
            self._enroll_state = PVADState.READY_FOR_ENROLL
            self._enroll_timeout_at_mono = time.monotonic() + self._enroll_timeout_s
            logger.info("pvad.enroll.ready", timeout_s=self._enroll_timeout_s)

    def on_speech_started(self, pre_roll: bytes = b"") -> None:
        with self._lock:
            if self._enroll_state == PVADState.READY_FOR_ENROLL:
                self._enroll_state = PVADState.COLLECTING
            if self._enroll_state == PVADState.COLLECTING:
                self._collecting_patient_audio = True
                if pre_roll:
                    self._enroll_buffer.extend(pre_roll)

    def on_speech_stopped(self) -> None:
        with self._lock:
            if self._enroll_state == PVADState.COLLECTING:
                self._collecting_patient_audio = False

    # ── VADBackend protocol ─────────────────────────────────────────────────

    def process(self, audio: bytes, sample_rate: int) -> VADResult:
        if sample_rate != _SAMPLE_RATE:
            return self._unavailable_result(audio, sample_rate, source="pvad_unavailable")

        # Init-failed session: spec §6.2 mandates source="pvad_unavailable"
        # rather than the FAILED enum's enroll-failed value.
        if self._init_failed:
            return self._unavailable_result(audio, sample_rate, source="pvad_unavailable")

        self._check_enroll_timeout()
        self._check_encode_timeout()
        self._maybe_accumulate_enroll(audio)
        self._maybe_kick_off_encoding()

        if self._enroll_state == PVADState.READY:
            return self._run_inference(audio, sample_rate)

        return self._status_result(audio, sample_rate)

    def reset(self) -> None:
        with self._lock:
            self._audio_buffer.clear()
            self._smoothed_prob = None
            self._stats.speech_duration_ms = 0.0
            self._stats.silence_duration_ms = 0.0
            # Spec §4.2: zero recurrent GRU state when READY so stale state from
            # the previous turn does not leak into the next utterance.
            if self._enroll_state == PVADState.READY:
                self._mel_state = np.zeros(_MEL_SHAPE, dtype=np.float32)
                self._gru_state = np.zeros(_GRU_SHAPE, dtype=np.float32)
            # Do NOT touch _enroll_state or _voiceprint — preserve across InterruptionFrame

    def shutdown(self) -> None:
        try:
            self._thread_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            logger.debug("pvad.shutdown.error", exc_info=True)

    # ── Internal: enrollment ────────────────────────────────────────────────

    def _check_enroll_timeout(self) -> None:
        with self._lock:
            if self._enroll_state in (PVADState.READY_FOR_ENROLL, PVADState.COLLECTING):
                if self._enroll_timeout_at_mono and time.monotonic() > self._enroll_timeout_at_mono:
                    logger.warning("pvad.enroll.timeout")
                    self._enroll_state = PVADState.TIMEOUT

    def _check_encode_timeout(self) -> None:
        """Spec §6.3 row 3: a hung ECAPA encode must not leave the machine
        permanently stuck in ENCODING. Bumping _encode_generation invalidates
        the stale worker so its eventual completion is silently dropped."""
        with self._lock:
            if self._enroll_state != PVADState.ENCODING:
                return
            if self._encode_started_mono is None:
                return
            if time.monotonic() - self._encode_started_mono <= self._encode_timeout_s:
                return
            logger.warning(
                "pvad.enroll.encode_timeout",
                waited_s=time.monotonic() - self._encode_started_mono,
            )
            self._enroll_state = PVADState.FAILED
            self._encode_generation += 1
            # Restart the thread pool to free the stuck worker for next session
            old_pool = self._thread_pool
            self._thread_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pvad-encode")
        try:
            old_pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def _maybe_accumulate_enroll(self, audio: bytes) -> None:
        with self._lock:
            if self._enroll_state == PVADState.COLLECTING and self._collecting_patient_audio:
                self._enroll_buffer.extend(audio)

    def _maybe_kick_off_encoding(self) -> None:
        with self._lock:
            if self._enroll_state != PVADState.COLLECTING:
                return
            if len(self._enroll_buffer) < self._enroll_target_bytes:
                return
            audio_copy = bytes(self._enroll_buffer)
            self._enroll_state = PVADState.ENCODING
            self._encode_started_mono = time.monotonic()
            self._encode_generation += 1
            generation = self._encode_generation
        # Submit OUTSIDE the lock (encode_and_set takes its own lock)
        self._encode_future = self._thread_pool.submit(self._encode_and_set, audio_copy, generation)

    def _encode_and_set(self, audio_bytes: bytes, generation: int) -> None:
        """Run ECAPA encode. Drop the result if the generation token is stale
        (i.e. timeout already invalidated this attempt). codex finding 2."""
        try:
            emb = self._ecapa.encode(audio_bytes, _SAMPLE_RATE)
            if not isinstance(emb, np.ndarray) or emb.shape != (192,):
                raise ValueError(f"bad embedding shape: {getattr(emb, 'shape', None)}")
            if float(np.linalg.norm(emb)) < self._min_embedding_magnitude:
                raise ValueError("embedding magnitude below threshold")
        except Exception as exc:
            logger.warning("pvad.enroll.encode_failed", error=str(exc))
            with self._lock:
                # Only flip state if this attempt is still current AND still ENCODING.
                if generation == self._encode_generation and self._enroll_state == PVADState.ENCODING:
                    self._enroll_state = PVADState.FAILED
            return
        with self._lock:
            if generation != self._encode_generation:
                logger.info("pvad.enroll.stale_worker_dropped", generation=generation)
                return
            if self._enroll_state != PVADState.ENCODING:
                logger.info(
                    "pvad.enroll.late_worker_dropped",
                    final_state=self._enroll_state.value,
                )
                return
            self._voiceprint = emb.astype(np.float32)
            self._enroll_state = PVADState.READY
            logger.info("pvad.enroll.success", embedding_norm=float(np.linalg.norm(emb)))

    # ── Internal: result construction ───────────────────────────────────────

    def _status_result(self, audio: bytes, sample_rate: int) -> VADResult:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        energy = float(np.sqrt(np.mean(pcm**2)) / 32768.0) if pcm.size else 0.0
        return VADResult(
            is_speech=False,
            energy=energy,
            speech_duration_ms=0.0,
            silence_duration_ms=0.0,
            source=self._enroll_state.value,
            speech_prob=None,
            threshold=None,
        )

    def _unavailable_result(self, audio: bytes, sample_rate: int, *, source: str) -> VADResult:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        energy = float(np.sqrt(np.mean(pcm**2)) / 32768.0) if pcm.size else 0.0
        return VADResult(
            is_speech=False,
            energy=energy,
            speech_duration_ms=0.0,
            silence_duration_ms=0.0,
            source=source,
            speech_prob=None,
            threshold=None,
        )

    def _run_inference(self, audio: bytes, sample_rate: int) -> VADResult:
        # 1. Accumulate to 10ms windows
        self._audio_buffer.extend(audio)
        latest_prob: float | None = None
        try:
            while len(self._audio_buffer) >= _WINDOW_BYTES:
                window = bytes(self._audio_buffer[:_WINDOW_BYTES])
                del self._audio_buffer[:_WINDOW_BYTES]
                latest_prob = self._infer_one_window(window)
            self._stats.consecutive_errors = 0
        except Exception as exc:
            self._stats.consecutive_errors += 1
            logger.warning(
                "pvad.runtime.frame_error",
                error=str(exc),
                consecutive=self._stats.consecutive_errors,
            )
            if self._stats.consecutive_errors >= self._max_consecutive_errors:
                with self._lock:
                    self._enroll_state = PVADState.DEAD
                logger.error("pvad.runtime.dead")
            return self._unavailable_result(audio, sample_rate, source="pvad_runtime_error")

        # 2. Smooth + threshold
        if latest_prob is None:
            # Window not full yet — return silent placeholder
            return self._unavailable_result(audio, sample_rate, source="ready")
        if self._smoothed_prob is None:
            self._smoothed_prob = latest_prob
        else:
            a = self._smoothing_alpha
            self._smoothed_prob = a * latest_prob + (1.0 - a) * self._smoothed_prob

        is_speech = self._smoothed_prob > self._threshold
        frame_ms = len(audio) / _BYTES_PER_SAMPLE / sample_rate * 1000.0
        if is_speech:
            self._stats.speech_duration_ms += frame_ms
            self._stats.silence_duration_ms = 0.0
        else:
            self._stats.silence_duration_ms += frame_ms
            self._stats.speech_duration_ms = 0.0
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
        energy = float(np.sqrt(np.mean(pcm**2)) / 32768.0) if pcm.size else 0.0
        return VADResult(
            is_speech=is_speech,
            energy=energy,
            speech_duration_ms=self._stats.speech_duration_ms,
            silence_duration_ms=self._stats.silence_duration_ms,
            source="ready",
            speech_prob=self._smoothed_prob,
            threshold=self._threshold,
        )

    def _infer_one_window(self, window_bytes: bytes) -> float:
        """Run one 10ms inference. Returns sigmoid_out probability.

        ONNX IO (verified against FireRedTeam/FireRedChat-pvad pvad.onnx):
          inputs:  input_audio[B,160], spkemb[B,192], mel_buffer[B,80,15],
                   gru_buffer[2,B,256]
          outputs: linear_out (logit, ignored), sigmoid_out (prob),
                   mel_buffer_out, gru_buffer_out
        """
        if self._onnx_session is None or self._voiceprint is None:
            raise RuntimeError("inference attempted without onnx_session or voiceprint")
        pcm = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_input = pcm.reshape(1, _WINDOW_SAMPLES).astype(np.float32)
        feed = {
            "input_audio": audio_input,
            "spkemb": self._voiceprint.reshape(1, 192).astype(np.float32),
            "mel_buffer": self._mel_state,
            "gru_buffer": self._gru_state,
        }
        outputs = self._onnx_session.run(None, feed)
        # outputs[0] = linear_out (logit, ignored)
        # outputs[1] = sigmoid_out (prob)  ← what we want
        # outputs[2] = mel_buffer_out
        # outputs[3] = gru_buffer_out
        prob = float(outputs[1].reshape(-1)[0])
        if len(outputs) >= 3:
            self._mel_state = outputs[2]
        if len(outputs) >= 4:
            self._gru_state = outputs[3]
        return prob


def load_onnx_session(model_path: str | None = None):
    """Load the FireRedChat pVAD ONNX session. Returns None if loading fails."""
    import os

    import onnxruntime as ort

    if model_path and os.path.exists(model_path):
        path = model_path
    else:
        if model_path:
            logger.warning("pvad.onnx.path_not_found", path=model_path)
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id="FireRedTeam/FireRedChat-pvad",
                filename="pvad.onnx",
            )
        except Exception as exc:
            logger.warning("pvad.onnx.download_failed", error=str(exc))
            return None
    try:
        return ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as exc:
        logger.warning("pvad.onnx.load_failed", error=str(exc), path=path)
        return None
