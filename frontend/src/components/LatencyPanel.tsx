import { useEffect, useMemo, useState } from "react";
import type { DataHandler } from "../hooks/useLiveKit";
import { getDisplayLatencyMs } from "./latencyPanelModel";

interface TurnMetrics {
  sample_key?: string;
  turn_id: string;
  smart_turn_ms: number;
  vad_tail_wait_ms: number;
  user_stop_to_emit_ms: number;
  gate_wait_ms: number;
  asr_text_at_emit: string;
  asr_text_type: string;
  asr_final_text: string;
  asr_final_latency_ms: number;
  asr_diff: string;
  emit_to_llm_ms: number;
  llm_ttft_ms: number;
  llm_actual_text: string;
  tts_session_create_ms: number;
  tts_pool_hit: boolean | null;
  tts_ttfa_ms: number;
  emit_to_audio_ms: number;
  output_queue_wait_ms: number;
  e2e_ms: number;
  speech_end_to_audio_ms: number;
  first_audio_source: string;
  first_audio_is_bridge: boolean;
  unexplained_gap_ms: number;
  shadow_results?: unknown[] | null;
  [key: string]: unknown;
}

function ms(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "--";
  return `${Math.round(value)}ms`;
}

function latencyTone(value: number, good: number, warn: number): string {
  if (!Number.isFinite(value) || value < 0) return "muted";
  if (value <= good) return "good";
  if (value <= warn) return "warn";
  return "bad";
}

function textOrEmpty(value: unknown): string {
  return typeof value === "string" ? value : "";
}

function normalizeMetrics(payload: Record<string, unknown>): TurnMetrics {
  const n = (key: string): number => {
    const value = payload[key];
    return typeof value === "number" && Number.isFinite(value) ? value : -1;
  };
  return {
    ...payload,
    turn_id: textOrEmpty(payload.turn_id),
    smart_turn_ms: n("smart_turn_ms"),
    vad_tail_wait_ms: n("vad_tail_wait_ms"),
    user_stop_to_emit_ms: n("user_stop_to_emit_ms"),
    gate_wait_ms: n("gate_wait_ms"),
    asr_text_at_emit: textOrEmpty(payload.asr_text_at_emit),
    asr_text_type: textOrEmpty(payload.asr_text_type),
    asr_final_text: textOrEmpty(payload.asr_final_text),
    asr_final_latency_ms: n("asr_final_latency_ms"),
    asr_diff: textOrEmpty(payload.asr_diff),
    emit_to_llm_ms: n("emit_to_llm_ms"),
    llm_ttft_ms: n("llm_ttft_ms"),
    llm_actual_text: textOrEmpty(payload.llm_actual_text),
    tts_session_create_ms: n("tts_session_create_ms"),
    tts_pool_hit:
      typeof payload.tts_pool_hit === "boolean" ? payload.tts_pool_hit : null,
    tts_ttfa_ms: n("tts_ttfa_ms"),
    emit_to_audio_ms: n("emit_to_audio_ms"),
    output_queue_wait_ms: n("output_queue_wait_ms"),
    e2e_ms: n("e2e_ms"),
    speech_end_to_audio_ms: n("speech_end_to_audio_ms"),
    first_audio_source: textOrEmpty(payload.first_audio_source),
    first_audio_is_bridge: payload.first_audio_is_bridge === true,
    unexplained_gap_ms: n("unexplained_gap_ms"),
    shadow_results: Array.isArray(payload.shadow_results)
      ? payload.shadow_results
      : null,
  };
}

function MetricTile({
  label,
  value,
  detail,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  tone: string;
}) {
  return (
    <div className={`metric-tile metric-tile--${tone}`}>
      <div className="metric-tile__label">{label}</div>
      <div className="metric-tile__value">{value}</div>
      <div className="metric-tile__detail">{detail}</div>
    </div>
  );
}

function TextProbe({
  label,
  value,
  empty,
}: {
  label: string;
  value: string;
  empty: string;
}) {
  return (
    <div className="latency-text-probe">
      <span>{label}</span>
      <code>{value.trim() || empty}</code>
    </div>
  );
}

export function LatencyPanel({
  registerDataHandler,
}: {
  registerDataHandler: (handler: DataHandler) => () => void;
}) {
  const [metrics, setMetrics] = useState<TurnMetrics | null>(null);
  const [history, setHistory] = useState<TurnMetrics[]>([]);

  useEffect(() => {
    const unsub = registerDataHandler((topic, payload) => {
      if (topic !== "turn_metrics") return;
      const next = normalizeMetrics(payload);
      setMetrics(next);
      setHistory((prev) => [
        ...prev.slice(-17),
        {
          ...next,
          sample_key: `${next.turn_id || "turn"}-${Date.now()}-${prev.length}`,
        },
      ]);
    });
    return unsub;
  }, [registerDataHandler]);

  const timeline = useMemo(() => {
    if (!metrics) return [];
    return [
      {
        key: "vad-tail",
        label: "VAD tail",
        value: metrics.vad_tail_wait_ms,
        good: 220,
        warn: 350,
        detail: "speech end -> stop",
      },
      {
        key: "smart",
        label: "SmartTurn",
        value: metrics.smart_turn_ms,
        good: 20,
        warn: 50,
        detail: "ONNX inference",
      },
      {
        key: "gate",
        label: "Gate wait",
        value: metrics.gate_wait_ms,
        good: 120,
        warn: 250,
        detail: "user stop -> emit",
      },
      {
        key: "asr-final",
        label: "ASR final",
        value: metrics.asr_final_latency_ms,
        good: 120,
        warn: 300,
        detail: "emit -> final text",
      },
      {
        key: "llm-ready",
        label: "LLM ready",
        value: metrics.emit_to_llm_ms,
        good: 40,
        warn: 120,
        detail: "emit -> request",
      },
      {
        key: "ttft",
        label: "LLM TTFT",
        value: metrics.llm_ttft_ms,
        good: 300,
        warn: 700,
        detail: "request -> first token",
      },
      {
        key: "tts-session",
        label: "TTS session",
        value: metrics.tts_session_create_ms,
        good: 80,
        warn: 180,
        detail:
          metrics.tts_pool_hit === true
            ? "pool hit"
            : metrics.tts_pool_hit === false
              ? "pool miss"
              : "acquire",
      },
      {
        key: "ttfa",
        label: "TTS TTFA",
        value: metrics.tts_ttfa_ms,
        good: 200,
        warn: 500,
        detail: "request -> first audio",
      },
      {
        key: "queue",
        label: "Queue",
        value: metrics.output_queue_wait_ms,
        good: 40,
        warn: 120,
        detail: "first audio -> output",
      },
    ];
  }, [metrics]);

  if (!metrics) {
    return (
      <section className="latency-console latency-console--empty">
        <div>
          <p className="panel-kicker">Latency</p>
          <h2>等待首个 TurnMetricsEvent</h2>
        </div>
        <p>连接后说第一句话，这里会显示从断句到首音的完整分解。</p>
      </section>
    );
  }

  const fullE2e = getDisplayLatencyMs(metrics);
  const gapTone = latencyTone(metrics.unexplained_gap_ms, 40, 120);
  const firstAudioLabel =
    metrics.first_audio_source === "bridge"
      ? "bridge"
      : metrics.first_audio_source || "speech";

  return (
    <section className="latency-console">
      <div className="latency-console__summary">
        <div>
          <p className="panel-kicker">Latency</p>
          <h2>完整端到端 {ms(fullE2e)}</h2>
          <p>
            turn <code>{metrics.turn_id || "--"}</code> · first audio{" "}
            <strong>{firstAudioLabel}</strong> · unexplained{" "}
            <strong className={`tone-${gapTone}`}>
              {ms(metrics.unexplained_gap_ms)}
            </strong>
          </p>
        </div>
      </div>

      <div className="metric-strip">
        {timeline.map((item) => (
          <MetricTile
            key={item.key}
            label={item.label}
            value={ms(item.value)}
            detail={item.detail}
            tone={latencyTone(item.value, item.good, item.warn)}
          />
        ))}
      </div>

      <div className="latency-text-grid">
        <TextProbe
          label={`ASR emit (${metrics.asr_text_type || "unknown"})`}
          value={metrics.asr_text_at_emit}
          empty="尚未捕捉"
        />
        <TextProbe
          label="ASR final"
          value={metrics.asr_final_text}
          empty="尚未到达"
        />
        <TextProbe
          label="LLM actual text"
          value={metrics.llm_actual_text}
          empty="尚未请求"
        />
        <TextProbe label="ASR diff" value={metrics.asr_diff} empty="无差异" />
      </div>

      {history.length > 1 && (
        <div className="latency-sparkline">
          {history.map((entry) => {
            const entryFullE2e = getDisplayLatencyMs(entry);
            const height =
              entryFullE2e >= 0
                ? Math.max(8, Math.min(58, (entryFullE2e / 2000) * 58))
                : 8;
            return (
              <span
                key={entry.sample_key ?? entry.turn_id}
                className={`latency-sparkline__bar tone-${latencyTone(
                  entryFullE2e,
                  1000,
                  1450,
                )}`}
                title={`${entry.turn_id || "turn"}: ${ms(entryFullE2e)}`}
                style={{ height }}
              />
            );
          })}
        </div>
      )}
    </section>
  );
}
