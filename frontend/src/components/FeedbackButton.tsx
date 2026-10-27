import { useMemo, useState } from "react";
import {
  buildFeedbackDraft,
  buildFeedbackTurnOptions,
  type FeedbackPayload,
  type FeedbackSource,
  type TraceEvent,
} from "./feedbackModel";

const FEEDBACK_ENDPOINT = "/api/feedback";

export function FeedbackButton({
  csrId,
  roomName,
  source,
  events,
  variant = "primary",
}: {
  csrId: string;
  roomName: string;
  source: FeedbackSource;
  events: TraceEvent[];
  variant?: "primary" | "subtle";
}) {
  const [open, setOpen] = useState(false);
  const [note, setNote] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [selectedTurnId, setSelectedTurnId] = useState("");

  const turnOptions = useMemo(() => buildFeedbackTurnOptions(events), [events]);
  const latestTurnId =
    turnOptions.length > 0 ? turnOptions[turnOptions.length - 1].turn_id : "";
  const selectedTurnExists = turnOptions.some(
    (option) => option.turn_id === selectedTurnId,
  );
  const effectiveSelectedTurnId = selectedTurnExists
    ? selectedTurnId
    : latestTurnId;
  const draft = useMemo(
    () =>
      buildFeedbackDraft({
        csrId,
        roomName,
        source,
        events,
        selectedTurnId: effectiveSelectedTurnId,
      }),
    [csrId, roomName, source, events, effectiveSelectedTurnId],
  );
  const hasContext = Boolean(
    draft.turn.user_text || draft.turn.assistant_text || events.length > 0,
  );

  const submit = async () => {
    const trimmed = note.trim();
    if (!trimmed) return;
    setSubmitting(true);
    setError("");
    setResult("");
    const payload: FeedbackPayload = { ...draft, note: trimmed };
    try {
      const response = await fetch(FEEDBACK_ENDPOINT, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(`提交失败 (${response.status})`);
      }
      const body = (await response.json()) as { id?: string };
      setResult(body.id ? `已提交：${body.id}` : "已提交");
      setNote("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "提交失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <>
      <button
        type="button"
        className={
          variant === "primary"
            ? "feedback-button"
            : "feedback-button feedback-button--subtle"
        }
        onClick={() => {
          setSelectedTurnId(latestTurnId);
          setOpen(true);
          setResult("");
          setError("");
        }}
        disabled={!hasContext}
        title={hasContext ? "选择回合并反馈" : "等待一次对话后再反馈"}
      >
        反馈
      </button>
      {open && (
        <div className="feedback-modal" role="presentation">
          <div
            className="feedback-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="feedback-title"
          >
            <div className="feedback-dialog__head">
              <div>
                <p className="panel-kicker">BUG FEEDBACK</p>
                <h2 id="feedback-title">反馈 AI 回复</h2>
              </div>
              <button
                type="button"
                className="icon-button"
                onClick={() => setOpen(false)}
                aria-label="关闭反馈弹窗"
              >
                ×
              </button>
            </div>

            {turnOptions.length > 0 && (
              <label className="feedback-turn-select" htmlFor="feedback-turn">
                <span>反馈回合</span>
                <select
                  id="feedback-turn"
                  value={effectiveSelectedTurnId}
                  onChange={(event) => setSelectedTurnId(event.target.value)}
                >
                  {turnOptions.map((option) => (
                    <option key={option.turn_id} value={option.turn_id}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
            )}

            <div className="feedback-context">
              <div>
                <span>用户</span>
                <p>{draft.turn.user_text || "还没有捕获到用户话术"}</p>
              </div>
              <div>
                <span>AI</span>
                <p>{draft.turn.assistant_text || "还没有捕获到 AI 回复"}</p>
              </div>
            </div>

            <label className="feedback-note" htmlFor="feedback-note">
              <span>哪里不自然或卡住了</span>
              <textarea
                id="feedback-note"
                value={note}
                onChange={(e) => setNote(e.target.value)}
                placeholder="例如：AI 重复了上一句话；停顿太久；没有正面回答用户问题。"
                rows={5}
              />
            </label>

            {(result || error) && (
              <p
                className={
                  result ? "feedback-message" : "feedback-message is-error"
                }
              >
                {result || error}
              </p>
            )}

            <div className="feedback-dialog__actions">
              <button
                type="button"
                className="feedback-cancel"
                onClick={() => setOpen(false)}
              >
                取消
              </button>
              <button
                type="button"
                className="feedback-submit"
                onClick={submit}
                disabled={!note.trim() || submitting}
              >
                {submitting ? "提交中..." : "提交反馈"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
