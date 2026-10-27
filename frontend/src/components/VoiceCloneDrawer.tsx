import { useEffect } from "react";
import { VOICE_CLONE_UI_URL } from "../services/voiceClone";

interface Props {
  open: boolean;
  onClose: () => void;
}

export function VoiceCloneDrawer({ open, onClose }: Props) {
  // Lock body scroll when open
  useEffect(() => {
    if (open) {
      document.body.style.overflow = "hidden";
      return () => {
        document.body.style.overflow = "";
      };
    }
  }, [open]);

  // Close on Escape
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 999,
        display: "flex",
        justifyContent: "flex-end",
      }}
    >
      {/* Backdrop */}
      <button
        type="button"
        aria-label="关闭声音克隆抽屉"
        onClick={onClose}
        onKeyDown={(e) => {
          if (e.key === "Escape") onClose();
        }}
        style={{
          position: "absolute",
          inset: 0,
          padding: 0,
          border: "none",
          background: "rgba(28, 32, 36, 0.35)",
          backdropFilter: "blur(2px)",
          animation: "fade-in 0.2s ease-out",
        }}
      />

      {/* Drawer panel */}
      <div
        style={{
          position: "relative",
          width: "min(560px, 92vw)",
          height: "100%",
          background: "var(--bg)",
          borderLeft: "1px solid var(--border)",
          boxShadow: "-8px 0 32px rgba(0, 0, 0, 0.08)",
          display: "flex",
          flexDirection: "column",
          animation: "slide-in-right 0.25s ease-out",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "14px 20px",
            background: "var(--surface)",
            borderBottom: "1px solid var(--border)",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div
              style={{
                width: 4,
                height: 16,
                borderRadius: 2,
                background: "var(--accent)",
              }}
            />
            <span
              style={{
                fontFamily: "var(--font-display)",
                fontSize: 15,
                fontWeight: 600,
                color: "var(--text-primary)",
                letterSpacing: "-0.2px",
              }}
            >
              声音克隆工作台
            </span>
            <span
              style={{
                fontFamily: "var(--font-mono)",
                fontSize: 10,
                padding: "2px 8px",
                borderRadius: 10,
                background: "var(--accent-dim)",
                color: "var(--accent)",
                border: "1px solid rgba(193, 126, 74, 0.12)",
              }}
            >
              CosyVoice
            </span>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="关闭"
            style={{
              width: 30,
              height: 30,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              background: "transparent",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: 16,
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor =
                "var(--border-hover)";
              (e.currentTarget as HTMLElement).style.color =
                "var(--text-primary)";
            }}
            onMouseLeave={(e) => {
              (e.currentTarget as HTMLElement).style.borderColor =
                "var(--border)";
              (e.currentTarget as HTMLElement).style.color =
                "var(--text-muted)";
            }}
          >
            <svg
              aria-hidden="true"
              focusable="false"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* iframe */}
        <iframe
          src={`${VOICE_CLONE_UI_URL}?embed`}
          title="声音克隆"
          allow="microphone"
          style={{
            flex: 1,
            width: "100%",
            border: "none",
            background: "var(--bg)",
          }}
        />
      </div>
    </div>
  );
}
