import { useCallback, useEffect, useState } from "react";
import { CallNotesPanel } from "./CallNotesPanel";
import { buildFeedbackListItems, type FeedbackRecord } from "./feedbackModel";

export function FeedbackReviewPage() {
  const [items, setItems] = useState<FeedbackRecord[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [error, setError] = useState<string>("");

  const load = useCallback(async () => {
    setError("");
    try {
      const response = await fetch("/api/feedback");
      if (!response.ok) throw new Error(`加载失败 (${response.status})`);
      const body = (await response.json()) as { items?: FeedbackRecord[] };
      setItems(body.items ?? []);
      setSelectedId((current) => current || body.items?.[0]?.id || "");
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const listItems = buildFeedbackListItems(items);
  const selected =
    items.find((item) => item.id === selectedId) ?? items[0] ?? null;

  return (
    <div className="feedback-page">
      <header className="feedback-page__header">
        <div>
          <p className="panel-kicker">FEEDBACK INBOX</p>
          <h1>AI 回复反馈</h1>
          <p>按时间倒序查看客服和调试台提交的问题。</p>
        </div>
        <div className="csr-header__actions">
          <a className="csr-link" href="/">
            客服页
          </a>
          <a className="csr-link" href="/debug">
            调试台
          </a>
          <button type="button" className="csr-connect" onClick={load}>
            刷新
          </button>
        </div>
      </header>

      {error && <p className="feedback-message is-error">{error}</p>}

      <main className="feedback-inbox">
        <section className="feedback-list">
          {listItems.length === 0 ? (
            <div className="conversation-empty">还没有反馈。</div>
          ) : (
            listItems.map((item) => (
              <button
                type="button"
                key={item.id}
                className={
                  item.id === selected?.id
                    ? "feedback-list__item is-active"
                    : "feedback-list__item"
                }
                onClick={() => setSelectedId(item.id)}
              >
                <strong>{item.title}</strong>
                <span>{item.subtitle}</span>
                <p>{item.note}</p>
              </button>
            ))
          )}
        </section>

        <section className="feedback-detail">
          {selected ? (
            <>
              <div className="feedback-detail__head">
                <div>
                  <p className="panel-kicker">{selected.id}</p>
                  <h2>{selected.note}</h2>
                </div>
                <span className="csr-badge">{selected.source}</span>
              </div>
              <div className="feedback-context">
                <div>
                  <span>用户</span>
                  <p>{selected.turn.user_text || "未捕获"}</p>
                </div>
                <div>
                  <span>AI</span>
                  <p>{selected.turn.assistant_text || "未捕获"}</p>
                </div>
              </div>
              <pre className="feedback-json">
                {JSON.stringify(selected, null, 2)}
              </pre>
              {selected?.room_name && (
                <CallNotesPanel roomName={selected.room_name} />
              )}
            </>
          ) : (
            <div className="conversation-empty">选择一条反馈查看详情。</div>
          )}
        </section>
      </main>
    </div>
  );
}
