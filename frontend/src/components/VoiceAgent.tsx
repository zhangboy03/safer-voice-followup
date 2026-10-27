import { useCallback, useEffect, useRef, useState } from "react";
import { type DataHandler, useLiveKit } from "../hooks/useLiveKit";
import { clearCsrId } from "./CsrIdGate";
import { resolveRoomName, tokenRequestBody } from "./callRouting";
import { DualLLMPanel } from "./DualLLMPanel";
import { FeedbackButton } from "./FeedbackButton";
import type { TraceEvent } from "./feedbackModel";
import { LatencyPanel } from "./LatencyPanel";
import { buildFastJudgeTraceFixtureEvents } from "./traceFixtureData";
import { VadComparePanel } from "./VadComparePanel";
import { VoiceCloneDrawer } from "./VoiceCloneDrawer";
import { VoiceSelector } from "./VoiceSelector";

const DEFAULT_LIVEKIT_URL = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
const LIVEKIT_URL =
  import.meta.env.VITE_LIVEKIT_URL?.trim() || DEFAULT_LIVEKIT_URL;
const TOKEN_ENDPOINT = import.meta.env.VITE_TOKEN_ENDPOINT ?? "/api/token";

export function VoiceAgent() {
  const {
    isConnected,
    isConnecting,
    connect,
    disconnect,
    transcript,
    agentText,
    registerDataHandler,
  } = useLiveKit();
  const [roomName] = useState(resolveRoomName);
  const traceFixture =
    new URLSearchParams(window.location.search).get("traceFixture") ===
    "fast-judge";
  const fixtureHandlersRef = useRef<DataHandler[]>([]);
  const [showCloneDrawer, setShowCloneDrawer] = useState(false);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [lineBusy, setLineBusy] = useState(false);
  const csrId = window.localStorage.getItem("csrId") ?? "user";

  // Suppress unused variable warnings (used via LiveKit data channel)
  void transcript;
  void agentText;

  const registerFixtureDataHandler = useCallback((handler: DataHandler) => {
    fixtureHandlersRef.current.push(handler);
    return () => {
      fixtureHandlersRef.current = fixtureHandlersRef.current.filter(
        (item) => item !== handler,
      );
    };
  }, []);

  useEffect(() => {
    if (!traceFixture) return;
    const timers: number[] = [];
    const events = buildFastJudgeTraceFixtureEvents();
    const replay = () => {
      if (fixtureHandlersRef.current.length === 0) {
        timers.push(window.setTimeout(replay, 50));
        return;
      }
      events.forEach((event, index) => {
        timers.push(
          window.setTimeout(() => {
            for (const handler of fixtureHandlersRef.current) {
              handler(event.topic, event.payload);
            }
          }, index * 90),
        );
      });
    };
    timers.push(window.setTimeout(replay, 30));
    return () => {
      for (const timer of timers) {
        window.clearTimeout(timer);
      }
    };
  }, [traceFixture]);

  const activeConnected = traceFixture || isConnected;
  const activeRegisterDataHandler = traceFixture
    ? registerFixtureDataHandler
    : registerDataHandler;

  useEffect(
    () =>
      activeRegisterDataHandler((topic, payload) => {
        if (topic === "session_reset") {
          setTraceEvents([]);
          return;
        }
        setTraceEvents((prev) => [...prev.slice(-79), { topic, payload }]);
      }),
    [activeRegisterDataHandler],
  );

  const handleConnect = async () => {
    setLineBusy(false);
    try {
      const res = await fetch(TOKEN_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: tokenRequestBody(
          roomName,
          window.localStorage.getItem("csrId") ?? "user",
        ),
      });
      if (res.status === 503) {
        setLineBusy(true);
        return;
      }
      if (!res.ok) {
        throw new Error(`token request failed (${res.status})`);
      }
      const { token } = (await res.json()) as { token: string };
      await connect(LIVEKIT_URL, token);
    } catch (error) {
      console.error("Connection failed:", error);
    }
  };

  return (
    <div
      style={{
        minHeight: "100%",
        padding: "24px 28px 56px",
        maxWidth: 1600,
        margin: "0 auto",
      }}
    >
      {lineBusy && (
        <div
          role="alert"
          style={{
            padding: "10px 14px",
            background: "#fde2e2",
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          当前并发已满（最多 3 路），请稍后再试。
        </div>
      )}
      {isConnecting && !isConnected && (
        <div
          style={{
            padding: "10px 14px",
            background: "#e6f0ff",
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          正在接通，AI 正在启动（约 5–10 秒）…
        </div>
      )}
      {/* ── Header ── */}
      <header
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 12,
          padding: "16px 24px",
          background: "var(--surface)",
          borderWidth: "3px 1px 1px 1px",
          borderStyle: "solid",
          borderColor:
            "var(--primary) var(--border) var(--border) var(--border)",
          borderRadius: "var(--radius-lg)",
          boxShadow: "var(--shadow-sm)",
          marginBottom: 24,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            flexWrap: "wrap",
            minWidth: 0,
          }}
        >
          {/* Status dot + Brand */}
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 9,
                height: 9,
                borderRadius: "50%",
                flexShrink: 0,
                background: activeConnected
                  ? "var(--success)"
                  : "var(--text-muted)",
                boxShadow: activeConnected
                  ? "0 0 10px rgba(22, 163, 74, 0.4)"
                  : "none",
                animation: activeConnected
                  ? "pulse-dot 2s ease-in-out infinite"
                  : "none",
                transition: "background 0.3s ease, box-shadow 0.3s ease",
              }}
              role="status"
              aria-label={activeConnected ? "已连接" : "未连接"}
            />
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 19,
                fontWeight: 700,
                letterSpacing: "-0.01em",
                color: "var(--primary)",
                whiteSpace: "nowrap",
              }}
            >
              品驰 AI
            </span>
          </div>
          <div
            style={{
              width: 1,
              height: 18,
              background: "var(--border)",
              flexShrink: 0,
            }}
          />
          <span
            style={{
              fontSize: 13,
              fontWeight: 500,
              color: "var(--text-secondary)",
              whiteSpace: "nowrap",
            }}
          >
            SOP 实时调试台
          </span>

          {/* Room badge */}
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              padding: "4px 12px",
              borderRadius: 20,
              background: "var(--primary-dim)",
              color: "var(--primary)",
              border: "1px solid rgba(12, 123, 111, 0.12)",
              fontWeight: 500,
              whiteSpace: "nowrap",
            }}
          >
            {roomName}
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span
            style={{
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            {csrId}
          </span>
          <a className="csr-link" href="/">
            客服页
          </a>
          <a className="csr-link" href="/feedback">
            反馈列表
          </a>
          <button
            type="button"
            onClick={clearCsrId}
            disabled={activeConnected}
            aria-label="退出登录"
            title={activeConnected ? "请先断开再退出" : "退出登录"}
            style={{
              padding: "5px 14px",
              fontSize: 12,
              fontWeight: 500,
              borderRadius: 20,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              color: activeConnected
                ? "var(--text-muted)"
                : "var(--text-secondary)",
              cursor: activeConnected ? "not-allowed" : "pointer",
              opacity: activeConnected ? 0.5 : 1,
              whiteSpace: "nowrap",
            }}
          >
            退出
          </button>
          <VoiceSelector />
          <FeedbackButton
            csrId={csrId}
            roomName={roomName}
            source="debug"
            events={traceEvents}
            variant="subtle"
          />
          <button
            type="button"
            onClick={() => setShowCloneDrawer(true)}
            aria-label="打开声音克隆工作台"
            style={{
              display: "flex",
              alignItems: "center",
              gap: 5,
              padding: "5px 14px",
              fontSize: 12,
              fontWeight: 500,
              fontFamily: "var(--font-sans)",
              borderRadius: 20,
              border: "1px solid var(--border)",
              background: "var(--surface)",
              color: "var(--text-secondary)",
              cursor: "pointer",
              transition: "all 0.15s ease",
              whiteSpace: "nowrap",
            }}
            onMouseEnter={(e) => {
              const t = e.currentTarget as HTMLElement;
              t.style.borderColor = "var(--accent)";
              t.style.color = "var(--accent)";
            }}
            onMouseLeave={(e) => {
              const t = e.currentTarget as HTMLElement;
              t.style.borderColor = "var(--border)";
              t.style.color = "var(--text-secondary)";
            }}
          >
            <svg
              aria-hidden="true"
              focusable="false"
              width="12"
              height="12"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="16" />
              <line x1="8" y1="12" x2="16" y2="12" />
            </svg>
            声音克隆
          </button>
          <div
            style={{
              width: 1,
              height: 18,
              background: "var(--border)",
              flexShrink: 0,
            }}
          />
          {traceFixture ? (
            <span
              style={{
                padding: "13px 18px",
                fontSize: 12,
                fontWeight: 700,
                fontFamily: "var(--font-mono)",
                borderRadius: "var(--radius-md)",
                border: "1px solid var(--border)",
                background: "var(--surface-alt)",
                color: "var(--text-secondary)",
              }}
            >
              fixture
            </span>
          ) : !isConnected ? (
            <button
              type="button"
              onClick={handleConnect}
              disabled={isConnecting}
              aria-label={isConnecting ? "正在连接" : "连接到 AI 客服房间"}
              style={{
                padding: "13px 28px",
                fontSize: 13,
                fontWeight: 600,
                fontFamily: "var(--font-sans)",
                borderRadius: "var(--radius-md)",
                border: "none",
                background: isConnecting ? "var(--border)" : "var(--primary)",
                color: "white",
                cursor: isConnecting ? "wait" : "pointer",
                transition:
                  "background 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease",
                boxShadow: isConnecting
                  ? "none"
                  : "0 2px 8px rgba(12, 123, 111, 0.25)",
              }}
            >
              {isConnecting ? "连接中..." : "连接"}
            </button>
          ) : (
            <button
              type="button"
              onClick={disconnect}
              aria-label="断开 AI 客服连接"
              style={{
                padding: "13px 28px",
                fontSize: 13,
                fontWeight: 500,
                fontFamily: "var(--font-sans)",
                borderRadius: "var(--radius-md)",
                border: "1px solid rgba(220, 38, 38, 0.2)",
                background: "var(--error-dim)",
                color: "var(--error)",
                cursor: "pointer",
                transition:
                  "background 0.2s ease, border-color 0.2s ease, color 0.2s ease",
              }}
            >
              断开
            </button>
          )}
        </div>
      </header>

      {/* ── Connected: Panels ── */}
      {activeConnected && (
        <main
          className="evaluation-shell"
          style={{ animation: "fade-in 0.4s ease-out" }}
        >
          <LatencyPanel registerDataHandler={activeRegisterDataHandler} />
          <VadComparePanel registerDataHandler={activeRegisterDataHandler} />
          <DualLLMPanel registerDataHandler={activeRegisterDataHandler} />
        </main>
      )}

      {/* ── Disconnected: Welcome ── */}
      {/* ── Voice Clone Drawer ── */}
      <VoiceCloneDrawer
        open={showCloneDrawer}
        onClose={() => setShowCloneDrawer(false)}
      />

      {!activeConnected && (
        <main
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minHeight: "65vh",
          }}
        >
          <div
            style={{
              textAlign: "center",
              padding: "64px 80px",
              background: "var(--surface)",
              borderWidth: "3px 1px 1px 1px",
              borderStyle: "solid",
              borderColor:
                "var(--primary) var(--border) var(--border) var(--border)",
              borderRadius: "var(--radius-xl)",
              boxShadow: "var(--shadow-lg)",
              animation: "fade-in 0.6s ease-out",
              maxWidth: "90vw",
            }}
          >
            {/* Sonar animation */}
            <div
              style={{
                position: "relative",
                width: 200,
                height: 200,
                margin: "0 auto 32px",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              {/* Rings */}
              {[0, 0.8, 1.6].map((delay) => (
                <div
                  key={delay}
                  style={{
                    position: "absolute",
                    width: 200,
                    height: 200,
                    borderRadius: "50%",
                    border: "2px solid var(--primary)",
                    animation: `sonar-ring 2.4s ease-out ${delay}s infinite`,
                  }}
                />
              ))}
              {/* Center dot */}
              <div
                style={{
                  width: 48,
                  height: 48,
                  borderRadius: "50%",
                  background:
                    "linear-gradient(135deg, var(--primary), var(--primary-hover))",
                  boxShadow: "0 4px 16px rgba(12, 123, 111, 0.3)",
                  animation: "breathe 3s ease-in-out infinite",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <svg
                  width="22"
                  height="22"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="white"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                  <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                  <line x1="12" x2="12" y1="19" y2="22" />
                </svg>
              </div>
            </div>

            <h1
              style={{
                margin: 0,
                fontFamily: "var(--font-display)",
                fontSize: 32,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                color: "var(--primary)",
                marginBottom: 4,
              }}
            >
              品驰 AI
            </h1>
            <h2
              style={{
                margin: 0,
                fontSize: 15,
                fontWeight: 500,
                color: "var(--text-secondary)",
                marginBottom: 40,
              }}
            >
              SOP 实时调试台
            </h2>
            <button
              type="button"
              onClick={handleConnect}
              disabled={isConnecting}
              aria-label={isConnecting ? "正在连接" : "连接到 AI 客服房间"}
              style={{
                padding: "14px 48px",
                fontSize: 15,
                fontWeight: 600,
                fontFamily: "var(--font-sans)",
                borderRadius: "var(--radius-md)",
                border: "none",
                background: isConnecting ? "var(--border)" : "var(--primary)",
                color: "white",
                cursor: isConnecting ? "wait" : "pointer",
                transition:
                  "background 0.2s ease, box-shadow 0.2s ease, opacity 0.2s ease",
                boxShadow: isConnecting
                  ? "none"
                  : "0 4px 16px rgba(12, 123, 111, 0.3)",
              }}
            >
              {isConnecting ? "连接中..." : "连接房间"}
            </button>
            <div
              style={{
                marginTop: 20,
                fontFamily: "var(--font-mono)",
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            >
              {LIVEKIT_URL} / {roomName}
            </div>
          </div>
        </main>
      )}
    </div>
  );
}
