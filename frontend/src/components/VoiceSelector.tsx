import { useCallback, useEffect, useRef, useState } from "react";
import { voiceCloneApiUrl } from "../services/voiceClone";

interface VoiceInfo {
  voice_id: string;
  status: string;
  target_model: string;
  kind?: "system" | "clone";
  display_name?: string;
  summary?: string;
  tags?: string[];
}

export function VoiceSelector() {
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const load = useCallback(async () => {
    try {
      // Probe with a short timeout to avoid console errors when server is down
      const probe = await fetch(voiceCloneApiUrl("/api/current"), {
        signal: AbortSignal.timeout(1500),
      });
      if (!probe.ok) return;
      const cr = await probe.json();
      setCurrentId(cr.voice_id ?? "");

      const lr = await fetch(voiceCloneApiUrl("/api/selectable-voices")).then(
        (r) => r.json(),
      );
      setVoices(lr.voices ?? []);
    } catch {
      /* voice clone server not running — silent */
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // Refresh when voice-clone iframe sends voice-changed message
  useEffect(() => {
    const handler = (e: MessageEvent) => {
      if (e.data?.type === "voice-changed") load();
    };
    window.addEventListener("message", handler);
    return () => window.removeEventListener("message", handler);
  }, [load]);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node))
        setOpen(false);
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [open]);

  const applyVoice = async (voice: VoiceInfo) => {
    await fetch(voiceCloneApiUrl("/api/use"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice_id: voice.voice_id,
        target_model: voice.target_model,
      }),
    });
    setCurrentId(voice.voice_id);
    setOpen(false);
  };

  const readyVoices = voices.filter(
    (v) => v.kind === "system" || v.status === "OK",
  );
  if (readyVoices.length === 0) return null;

  const currentVoice = readyVoices.find((v) => v.voice_id === currentId);
  const displayName = currentVoice
    ? currentVoice.display_name || abbreviateId(currentVoice.voice_id)
    : currentId
      ? abbreviateId(currentId)
      : "默认音色";
  const systemVoices = readyVoices.filter((v) => v.kind === "system");
  const cloneVoices = readyVoices.filter((v) => v.kind !== "system");

  return (
    <div ref={ref} style={{ position: "relative" }}>
      <button
        type="button"
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          padding: "5px 12px",
          fontFamily: "var(--font-mono)",
          fontSize: 11,
          fontWeight: 500,
          color: "var(--accent)",
          background: "var(--accent-dim)",
          border: "1px solid rgba(193, 126, 74, 0.15)",
          borderRadius: 20,
          cursor: "pointer",
          transition: "all 0.15s ease",
          whiteSpace: "nowrap",
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
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
          <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
        </svg>
        {displayName}
        <svg
          aria-hidden="true"
          focusable="false"
          width="10"
          height="10"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2.5"
          style={{
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.15s",
          }}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {open && (
        <div
          style={{
            position: "absolute",
            top: "calc(100% + 6px)",
            right: 0,
            minWidth: 320,
            maxHeight: 340,
            overflowY: "auto",
            background: "var(--surface)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            boxShadow: "var(--shadow-lg)",
            zIndex: 100,
            animation: "fade-in 0.15s ease-out",
          }}
        >
          <div
            style={{
              padding: "10px 14px 6px",
              fontSize: 11,
              fontWeight: 600,
              fontFamily: "var(--font-mono)",
              color: "var(--text-muted)",
              textTransform: "uppercase",
              letterSpacing: "0.8px",
              borderBottom: "1px solid var(--border)",
            }}
          >
            可用音色 ({readyVoices.length})
          </div>
          {systemVoices.length > 0 && (
            <VoiceSection
              title="官方客服音色"
              voices={systemVoices}
              currentId={currentId}
              onSelect={applyVoice}
            />
          )}
          {cloneVoices.length > 0 && (
            <VoiceSection
              title="我的克隆音色"
              voices={cloneVoices}
              currentId={currentId}
              onSelect={applyVoice}
            />
          )}
        </div>
      )}
    </div>
  );
}

function VoiceSection({
  title,
  voices,
  currentId,
  onSelect,
}: {
  title: string;
  voices: VoiceInfo[];
  currentId: string;
  onSelect: (voice: VoiceInfo) => void | Promise<void>;
}) {
  return (
    <div>
      <div
        style={{
          padding: "8px 14px 6px",
          fontSize: 10,
          fontWeight: 600,
          fontFamily: "var(--font-mono)",
          color: "var(--text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.8px",
          background: "var(--surface-alt)",
          borderBottom: "1px solid var(--border)",
        }}
      >
        {title}
      </div>
      {voices.map((v) => {
        const isActive = v.voice_id === currentId;
        return (
          <button
            type="button"
            disabled={isActive}
            key={`${v.kind ?? "voice"}-${v.voice_id}`}
            style={{
              width: "100%",
              padding: "10px 14px",
              display: "flex",
              alignItems: "center",
              gap: 10,
              font: "inherit",
              color: "inherit",
              textAlign: "left",
              borderBottom: "1px solid var(--border)",
              borderTop: "none",
              borderRight: "none",
              borderLeft: "none",
              cursor: isActive ? "default" : "pointer",
              background: isActive ? "var(--primary-dim)" : "transparent",
              opacity: 1,
              transition: "background 0.1s",
            }}
            onClick={() => onSelect(v)}
            onMouseEnter={(e) => {
              if (!isActive)
                (e.currentTarget as HTMLElement).style.background =
                  "var(--surface-hover)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.background = isActive
                ? "var(--primary-dim)"
                : "transparent";
            }}
          >
            <div style={{ flex: 1, minWidth: 0 }}>
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  flexWrap: "wrap",
                }}
              >
                <div
                  style={{
                    fontFamily: "var(--font-mono)",
                    fontSize: 11,
                    color: isActive ? "var(--primary)" : "var(--text-primary)",
                    wordBreak: "break-all",
                    fontWeight: isActive ? 600 : 400,
                  }}
                >
                  {v.display_name || abbreviateId(v.voice_id)}
                </div>
                {v.tags?.slice(0, 2).map((tag) => (
                  <span
                    key={`${v.voice_id}-${tag}`}
                    style={{
                      fontSize: 10,
                      padding: "1px 6px",
                      borderRadius: 999,
                      background: "var(--accent-dim)",
                      color: "var(--accent)",
                    }}
                  >
                    {tag}
                  </span>
                ))}
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: "var(--text-muted)",
                  marginTop: 2,
                }}
              >
                {v.summary || v.target_model}
              </div>
              <div
                style={{
                  fontSize: 10,
                  color: "var(--text-muted)",
                  marginTop: 2,
                  fontFamily: "var(--font-mono)",
                }}
              >
                {v.voice_id} / {v.target_model}
              </div>
            </div>
            {isActive && (
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 600,
                  padding: "2px 8px",
                  borderRadius: 10,
                  background: "var(--primary)",
                  color: "white",
                  flexShrink: 0,
                }}
              >
                当前
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

function abbreviateId(vid: string): string {
  // cosyvoice-v3.5-flash-vd-pinchics-8eb54202... → pinchics-8eb5
  const parts = vid.split("-vd-");
  if (parts.length > 1) {
    const tail = parts[1];
    return tail.length > 14 ? `${tail.slice(0, 14)}...` : tail;
  }
  return vid.length > 20 ? `${vid.slice(0, 20)}...` : vid;
}
