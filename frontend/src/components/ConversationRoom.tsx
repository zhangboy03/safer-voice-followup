import { useEffect, useRef, useState } from "react";
import { useSoloDemo } from "../hooks/useSoloDemo";
import { LiveKitService } from "../services/LiveKitService";
import type { SoloField } from "../services/soloDemoClient";
import "./ConversationRoom.css";

const FIELD_LABELS: Record<string, string> = {
  call_accepted: "Call readiness",
  is_patient: "Caller role",
  relationship: "Family relationship",
  symptom_change: "Symptom change",
  control_score: "Control score",
  programming_count: "Programming sessions",
  programming_satisfaction: "Programming satisfaction",
  device_explained: "Device use explained",
  response_speed: "Hospital responsiveness",
  insurance_type: "Insurance type",
  self_pay: "Out-of-pocket cost",
};
const ENGLISH_VALUES: Record<string, string> = {
  明显改善: "Much improved",
  有所改善: "Improved",
  无变化: "No change",
  有所加重: "Worse",
  满意: "Satisfied",
  不满意: "Not satisfied",
  快速响应: "Quick response",
  响应慢: "Slow response",
  无问题: "No issue",
  confirmed: "Ready",
  refused: "Declined",
  职工医保: "Employee health insurance",
  城乡居民医保: "Resident health insurance",
  商业保险: "Private insurance",
  自费: "Self-pay",
};
function fieldValue(key: string, field: SoloField): string {
  if (field.status === "refused") return "Declined";
  if (key === "is_patient" && typeof field.value === "boolean")
    return field.value ? "Patient" : "Family member";
  if (typeof field.value === "boolean") return field.value ? "Yes" : "No";
  const value =
    typeof field.value === "object"
      ? JSON.stringify(field.value)
      : String(field.value ?? "");
  if (key === "control_score" && value) return `${value} / 10`;
  return ENGLISH_VALUES[value] ?? value;
}
const PHASE_LABELS = {
  idle: "Ready when you are",
  connecting: "Connecting… Allow microphone access if prompted.",
  starting: "Preparing the conversation…",
  listening: "Listening",
  speaking: "SAFER is speaking",
  reconnecting: "Reconnecting…",
  ended: "Conversation ended",
  failed: "Unable to connect",
};
function Level({ label, level }: { label: string; level: number }) {
  const value = Math.round(Math.max(0, Math.min(1, level)) * 100);
  return (
    <div className="room-level">
      <span>{label}</span>
      <meter
        aria-label={`${label} audio level`}
        min={0}
        max={100}
        value={value}
      />
    </div>
  );
}

export function ConversationRoom() {
  const { state, start, stop, reset } = useSoloDemo();
  const [browserError] = useState(LiveKitService.getSupportError);
  const [starting, setStarting] = useState(false);
  const [closing, setClosing] = useState(false);
  const [localError, setLocalError] = useState("");
  const operation = useRef(0);
  const logEnd = useRef<HTMLDivElement | null>(null);
  const active = [
    "connecting",
    "starting",
    "listening",
    "speaking",
    "reconnecting",
  ].includes(state.phase);
  const lastMessage = state.messages[state.messages.length - 1];
  useEffect(() => {
    if (lastMessage) logEnd.current?.scrollIntoView({ block: "nearest" });
  }, [lastMessage]);
  useEffect(
    () => () => {
      operation.current += 1;
    },
    [],
  );
  const perform = async (action: "start" | "stop" | "reset") => {
    const id = ++operation.current;
    setLocalError("");
    if (action === "start") setStarting(true);
    else {
      setClosing(true);
      setStarting(false);
    }
    try {
      await (action === "start"
        ? start()
        : action === "stop"
          ? stop()
          : reset());
      return true;
    } catch (error) {
      if (id === operation.current)
        setLocalError(error instanceof Error ? error.message : String(error));
      return false;
    } finally {
      if (id === operation.current) {
        setStarting(false);
        setClosing(false);
      }
    }
  };
  const back = async () => {
    if ((!active && !starting && !state.sessionId) || (await perform("stop")))
      window.location.assign("/#9");
  };
  const fields = Object.entries(state.record.fields).filter(
    ([key]) => FIELD_LABELS[key],
  );
  const error = browserError || state.error || localError;
  return (
    <main className="conversation-room" data-phase={state.phase}>
      <header className="room-header">
        <button
          type="button"
          disabled={closing}
          onClick={() => void back()}
          className="room-brand"
        >
          SAFER
        </button>
        <button
          type="button"
          className="room-back"
          disabled={closing}
          onClick={() => void back()}
        >
          Back to the presentation
        </button>
      </header>
      <div className="room-heading">
        <h1>Talk with SAFER.</h1>
      </div>
      {error && (
        <div className="room-error" role="alert">
          <p>{error}</p>
          {!browserError && (
            <span>End the conversation or clear it, then try again.</span>
          )}
        </div>
      )}
      <div className="room-workspace">
        <section className="room-conversation" aria-label="Conversation">
          <div className="room-status">
            <span className={active ? "room-dot active" : "room-dot"} />
            <p role="status">{PHASE_LABELS[state.phase]}</p>
          </div>
          <div className="room-audio">
            <Level label="You" level={state.levels.patient} />
            <Level label="SAFER" level={state.levels.assistant} />
          </div>
          <section
            className="room-transcript"
            aria-label="Conversation transcript"
          >
            {state.messages.length === 0 ? (
              <p className="room-empty">
                Answer naturally, pause, or ask a device question.
              </p>
            ) : (
              state.messages.slice(-24).map((message) => (
                <article
                  className={`room-message ${message.role}${message.interrupted ? " interrupted" : ""}`}
                  key={message.id}
                >
                  <span>
                    {message.role === "patient" ? "You" : "SAFER"}
                    {message.interrupted ? " · Interrupted" : ""}
                  </span>
                  <p>{message.text}</p>
                </article>
              ))
            )}
            <div ref={logEnd} />
          </section>
          <div className="room-actions">
            <button
              type="button"
              className="room-primary"
              disabled={!!browserError || active || starting || closing}
              onClick={() => void perform("start")}
            >
              {starting
                ? "Connecting…"
                : state.phase === "ended"
                  ? "Start another conversation"
                  : "Start conversation"}
            </button>
            <button
              type="button"
              disabled={(!active && !starting && !state.sessionId) || closing}
              onClick={() => void perform("stop")}
            >
              {closing ? "Ending…" : "End conversation"}
            </button>
            <button
              type="button"
              disabled={active || starting || closing}
              onClick={() => void perform("reset")}
            >
              Clear
            </button>
          </div>
        </section>
        <section
          className="room-record"
          aria-label="Captured fields"
          data-revision={state.record.revision}
        >
          <h2>Captured fields</h2>
          {fields.length === 0 ? (
            <p className="room-empty">
              Information appears as the conversation progresses.
            </p>
          ) : (
            <dl>
              {fields.map(([key, field]) => (
                <div className="room-field" key={key}>
                  <dt>{FIELD_LABELS[key]}</dt>
                  <dd>{fieldValue(key, field)}</dd>
                  {field.evidence && <p>“{field.evidence}”</p>}
                </div>
              ))}
            </dl>
          )}
        </section>
      </div>
    </main>
  );
}
