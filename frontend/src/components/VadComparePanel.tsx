// frontend/src/components/VadComparePanel.tsx
import { useEffect, useRef, useState } from "react";
import type { DataHandler } from "../hooks/useLiveKit";

type ShadowState =
  | "waiting_greeting"
  | "ready_for_enroll"
  | "collecting"
  | "encoding"
  | "ready"
  | "pvad_unavailable"
  | "pvad_enroll_timeout"
  | "pvad_enroll_failed"
  | "pvad_runtime_error"
  | "pvad_dead";

interface FrameEvent {
  ts_mono_ms: number;
  backend: string;
  state: ShadowState | string;
  is_speech: boolean;
  speech_prob: number | null;
  threshold: number | null;
  primary_is_speech: boolean;
  enroll_progress_ms: number | null;
}

interface SpeechEvent {
  ts_mono_ms: number;
  backend: string;
  event_type_: string;
  primary_event_type: string | null;
  speech_duration_ms: number | null;
}

interface TimelinePoint {
  ts: number;
  energy: boolean;
  pvad: boolean;
  pvad_ready: boolean;
  pvad_prob: number | null;
  pvad_threshold: number | null;
}

const WINDOW_MS = 30_000;
const STRIP_HEIGHT = 28;
const STRIP_PAD_Y = 6;
const COLOR_SPEECH = "#1f8c2e";
const COLOR_SILENCE = "#e5e7eb";
const COLOR_NOT_READY = "#f4e3bf";
const COLOR_GRID = "#d1d5db";
const COLOR_PROB_LINE = "#1d4ed8";
const COLOR_THRESHOLD_LINE = "#dc2626";

function stateZh(state: string, enrollMs: number | null): string {
  switch (state) {
    case "waiting_greeting":
      return "等 agent 说完开场白";
    case "ready_for_enroll":
      return "等用户开口说第一句";
    case "collecting": {
      const cur = enrollMs ?? 0;
      return `正在采集声纹 ${(cur / 1000).toFixed(1)}s / 5.0s`;
    }
    case "encoding":
      return "声纹编码中…";
    case "ready":
      return "✓ 已就绪（声纹已采，正在和 energy 并行判定）";
    case "pvad_unavailable":
      return "✕ 模型不可用";
    case "pvad_enroll_timeout":
      return "✕ 声纹采集超时（30s 内没听到用户开口）";
    case "pvad_enroll_failed":
      return "✕ 声纹采集失败";
    case "pvad_runtime_error":
      return "✕ 运行时错误";
    case "pvad_dead":
      return "✕ 已停用";
    default:
      return state || "—";
  }
}

function isReady(state: string): boolean {
  return state === "ready";
}

// Map server ts_mono_ms to a "client-now-aligned" timeline.
// On first event, anchor (Date.now(), ts_mono_ms); thereafter assume
// client wall clock ≈ server mono clock (drift over 30s is negligible).
function makeServerNow(
  clockRef: React.MutableRefObject<{
    serverMono: number;
    clientMs: number;
  } | null>,
): () => number {
  return () => {
    const c = clockRef.current;
    if (!c) return 0;
    return c.serverMono + (Date.now() - c.clientMs);
  };
}

function pushTimelinePoint(buf: TimelinePoint[], point: TimelinePoint): void {
  if (buf.length > 0 && point.ts <= buf[buf.length - 1].ts) {
    point.ts = buf[buf.length - 1].ts + 0.001;
  }
  buf.push(point);
  const cutoff = point.ts - WINDOW_MS;
  let trim = 0;
  while (trim < buf.length - 1 && buf[trim + 1].ts < cutoff) trim++;
  if (trim > 0) buf.splice(0, trim);
}

export function VadComparePanel({
  registerDataHandler,
}: {
  registerDataHandler: (handler: DataHandler) => () => void;
}) {
  const [latest, setLatest] = useState<FrameEvent | null>(null);
  const [edgeDelta, setEdgeDelta] = useState<{
    ms: number;
    ageMs: number;
  } | null>(null);
  const [, force] = useState(0);

  const timelineRef = useRef<TimelinePoint[]>([]);
  const lastEdgeRef = useRef<{ energy: number | null; pvad: number | null }>({
    energy: null,
    pvad: null,
  });
  const clockRef = useRef<{ serverMono: number; clientMs: number } | null>(
    null,
  );
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const serverNowRef = useRef(makeServerNow(clockRef));

  useEffect(() => {
    const unsub = registerDataHandler((topic, payload) => {
      if (topic === "vad_shadow_frame") {
        const ev = payload as unknown as FrameEvent;
        if (!clockRef.current) {
          clockRef.current = {
            serverMono: ev.ts_mono_ms,
            clientMs: Date.now(),
          };
        }
        setLatest(ev);
        const prev = timelineRef.current[timelineRef.current.length - 1];
        const pvadEffective = isReady(ev.state) ? ev.is_speech : false;
        pushTimelinePoint(timelineRef.current, {
          ts: ev.ts_mono_ms,
          energy: ev.primary_is_speech,
          pvad: pvadEffective,
          pvad_ready: isReady(ev.state),
          pvad_prob: ev.speech_prob,
          pvad_threshold: ev.threshold,
        });
        if (
          !prev ||
          prev.energy !== ev.primary_is_speech ||
          prev.pvad !== pvadEffective ||
          prev.pvad_ready !== isReady(ev.state)
        ) {
          force((n) => n + 1);
        }
      } else if (topic === "vad_shadow_speech") {
        const ev = payload as unknown as SpeechEvent;
        if (!clockRef.current) {
          clockRef.current = {
            serverMono: ev.ts_mono_ms,
            clientMs: Date.now(),
          };
        }
        if (ev.event_type_ === "started") {
          lastEdgeRef.current.pvad = ev.ts_mono_ms;
        }
        if (ev.primary_event_type === "started") {
          lastEdgeRef.current.energy = ev.ts_mono_ms;
        }
        const { energy: eEdge, pvad: pEdge } = lastEdgeRef.current;
        if (eEdge != null && pEdge != null && Math.abs(eEdge - pEdge) < 5_000) {
          setEdgeDelta({
            ms: pEdge - eEdge,
            ageMs: serverNowRef.current() - Math.max(eEdge, pEdge),
          });
        }
      }
    });
    return unsub;
  }, [registerDataHandler]);

  // RAF loop: slide strip + extend last segment to "now".
  useEffect(() => {
    let raf = 0;
    const tick = (): void => {
      drawStrip(canvasRef.current, timelineRef.current, serverNowRef.current());
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Divergence stats (time-weighted, only over segments where pvad was ready).
  const stats = computeStats(timelineRef.current, serverNowRef.current());
  const stateLabel = stateZh(
    latest?.state ?? "waiting_greeting",
    latest?.enroll_progress_ms ?? null,
  );
  const empty = timelineRef.current.length === 0;

  return (
    <section
      className="vad-compare-panel"
      style={{
        border: "1px solid #d1d5db",
        padding: 14,
        borderRadius: 8,
        background: "#fff",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
        }}
      >
        <h3 style={{ margin: 0, fontSize: 15 }}>VAD 对比</h3>
        <div style={{ fontSize: 11, color: "#6b7280" }}>
          agent 实际听 <b style={{ color: "#1f2937" }}>energy</b>；pvad
          影子并行跑，验证后才会切主路径
        </div>
      </header>

      {/* Strip + axis labels */}
      <div style={{ marginTop: 12, position: "relative" }}>
        <div style={{ display: "flex", marginBottom: 4 }}>
          <div style={{ width: 54, fontSize: 11, color: "#6b7280" }}>
            energy
          </div>
          <div
            style={{
              flex: 1,
              fontSize: 11,
              color: "#9ca3af",
              textAlign: "right",
            }}
          >
            ← 30 秒前 ··· 现在 →
          </div>
        </div>
        <canvas
          ref={canvasRef}
          width={1000}
          height={STRIP_HEIGHT * 2 + STRIP_PAD_Y}
          style={{
            width: "100%",
            height: STRIP_HEIGHT * 2 + STRIP_PAD_Y,
            background: "#fafafa",
            borderRadius: 4,
            border: "1px solid #e5e7eb",
            display: "block",
          }}
        />
        <div style={{ display: "flex", marginTop: 4 }}>
          <div style={{ width: 54, fontSize: 11, color: "#6b7280" }}>pvad</div>
          <div style={{ flex: 1, fontSize: 11, color: "#9ca3af" }}>
            <Legend />
          </div>
        </div>
        {empty && (
          <div
            style={{
              position: "absolute",
              inset: "18px 0 18px 60px",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#9ca3af",
              fontSize: 12,
              pointerEvents: "none",
            }}
          >
            等用户开口——说一句话后这里会出现两条同步的判定轨迹
          </div>
        )}
      </div>

      {/* pVAD 当前状态 */}
      <div style={{ marginTop: 14, fontSize: 12, color: "#374151" }}>
        <span style={{ color: "#6b7280" }}>pvad 状态：</span>
        <span>{stateLabel}</span>
      </div>

      {/* 分歧统计 */}
      <div
        style={{
          marginTop: 8,
          display: "flex",
          gap: 18,
          fontSize: 12,
          color: stats.ready_ms > 0 ? "#374151" : "#9ca3af",
        }}
      >
        <span>
          <span style={{ color: "#6b7280" }}>过去 30s 分歧：</span>
          {stats.ready_ms > 0
            ? `${stats.diverge_pct.toFixed(1)}% (${(stats.diverge_ms / 1000).toFixed(1)}s / ${(stats.ready_ms / 1000).toFixed(1)}s)`
            : "—（pvad 还没就绪）"}
        </span>
        <span>
          <span style={{ color: "#6b7280" }}>首边沿差：</span>
          {edgeDelta && edgeDelta.ageMs < WINDOW_MS
            ? `${edgeDelta.ms >= 0 ? "+" : ""}${Math.round(edgeDelta.ms)}ms（pvad ${edgeDelta.ms >= 0 ? "晚" : "早"}）`
            : "—"}
        </span>
      </div>
    </section>
  );
}

function Legend(): JSX.Element {
  const dot = (color: string): JSX.Element => (
    <span
      style={{
        display: "inline-block",
        width: 10,
        height: 10,
        borderRadius: 2,
        background: color,
        verticalAlign: "middle",
        marginRight: 4,
      }}
    />
  );
  const line = (color: string, dashed = false): JSX.Element => (
    <span
      style={{
        display: "inline-block",
        width: 16,
        height: 0,
        borderTop: `1.5px ${dashed ? "dashed" : "solid"} ${color}`,
        verticalAlign: "middle",
        marginRight: 4,
      }}
    />
  );
  return (
    <span>
      {dot(COLOR_SPEECH)}人声
      <span style={{ marginLeft: 10 }}>{dot(COLOR_SILENCE)}静音</span>
      <span style={{ marginLeft: 10 }}>{dot(COLOR_NOT_READY)}pvad 未就绪</span>
      <span style={{ marginLeft: 10 }}>{line(COLOR_PROB_LINE)}原始概率</span>
      <span style={{ marginLeft: 10 }}>
        {line(COLOR_THRESHOLD_LINE, true)}阈值
      </span>
    </span>
  );
}

function drawStrip(
  canvas: HTMLCanvasElement | null,
  timeline: TimelinePoint[],
  nowTs: number,
): void {
  if (!canvas) return;
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  if (cssW <= 0 || cssH <= 0) return;
  const pxW = Math.floor(cssW * dpr);
  const pxH = Math.floor(cssH * dpr);
  if (canvas.width !== pxW) canvas.width = pxW;
  if (canvas.height !== pxH) canvas.height = pxH;
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const energyY = 0;
  const pvadY = STRIP_HEIGHT + STRIP_PAD_Y;

  // Background fills (silence default)
  ctx.fillStyle = COLOR_SILENCE;
  ctx.fillRect(0, energyY, cssW, STRIP_HEIGHT);
  ctx.fillRect(0, pvadY, cssW, STRIP_HEIGHT);

  if (timeline.length === 0) return;

  const startTs = nowTs - WINDOW_MS;
  const xOf = (ts: number): number => ((ts - startTs) / WINDOW_MS) * cssW;

  // Paint segments. For each [point i, point i+1) draw a colored band per row.
  for (let i = 0; i < timeline.length; i++) {
    const p = timeline[i];
    const nextTs = i + 1 < timeline.length ? timeline[i + 1].ts : nowTs;
    if (nextTs < startTs) continue;
    const x0 = Math.max(0, xOf(p.ts));
    const x1 = Math.min(cssW, xOf(nextTs));
    const w = x1 - x0;
    if (w <= 0) continue;

    // energy row
    if (p.energy) {
      ctx.fillStyle = COLOR_SPEECH;
      ctx.fillRect(x0, energyY, w, STRIP_HEIGHT);
    }

    // pvad row
    if (!p.pvad_ready) {
      ctx.fillStyle = COLOR_NOT_READY;
      ctx.fillRect(x0, pvadY, w, STRIP_HEIGHT);
    } else if (p.pvad) {
      ctx.fillStyle = COLOR_SPEECH;
      ctx.fillRect(x0, pvadY, w, STRIP_HEIGHT);
    }
  }

  // Threshold reference (dashed horizontal) on pvad strip
  let latestThreshold: number | null = null;
  for (let i = timeline.length - 1; i >= 0; i--) {
    if (timeline[i].pvad_threshold != null) {
      latestThreshold = timeline[i].pvad_threshold;
      break;
    }
  }
  if (latestThreshold != null) {
    const thrY = pvadY + STRIP_HEIGHT * (1 - latestThreshold);
    ctx.strokeStyle = COLOR_THRESHOLD_LINE;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(0, thrY);
    ctx.lineTo(cssW, thrY);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Raw probability polyline on pvad strip (only where pvad is ready)
  ctx.strokeStyle = COLOR_PROB_LINE;
  ctx.lineWidth = 1.4;
  ctx.beginPath();
  let probStarted = false;
  for (let i = 0; i < timeline.length; i++) {
    const p = timeline[i];
    if (!p.pvad_ready || p.pvad_prob == null) {
      probStarted = false;
      continue;
    }
    const x = xOf(p.ts);
    if (x < 0 || x > cssW) {
      probStarted = false;
      continue;
    }
    const y = pvadY + STRIP_HEIGHT * (1 - p.pvad_prob);
    if (!probStarted) {
      ctx.moveTo(x, y);
      probStarted = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();
  ctx.lineWidth = 1;

  // "now" tick on right edge
  ctx.fillStyle = COLOR_GRID;
  ctx.fillRect(cssW - 1, 0, 1, cssH);
}

interface Stats {
  diverge_ms: number;
  ready_ms: number;
  diverge_pct: number;
}

function computeStats(timeline: TimelinePoint[], nowTs: number): Stats {
  if (timeline.length === 0)
    return { diverge_ms: 0, ready_ms: 0, diverge_pct: 0 };
  const startTs = nowTs - WINDOW_MS;
  let readyMs = 0;
  let divergeMs = 0;
  for (let i = 0; i < timeline.length; i++) {
    const p = timeline[i];
    const nextTs = i + 1 < timeline.length ? timeline[i + 1].ts : nowTs;
    const segStart = Math.max(startTs, p.ts);
    const segEnd = Math.min(nowTs, nextTs);
    const dur = segEnd - segStart;
    if (dur <= 0) continue;
    if (!p.pvad_ready) continue;
    readyMs += dur;
    if (p.pvad !== p.energy) divergeMs += dur;
  }
  return {
    diverge_ms: divergeMs,
    ready_ms: readyMs,
    diverge_pct: readyMs > 0 ? (divergeMs / readyMs) * 100 : 0,
  };
}
