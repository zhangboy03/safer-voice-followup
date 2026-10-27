import { useEffect, useState } from "react";
import {
  fetchNotes,
  listCallIdsByRoom,
  type NoteItem,
  type NotesResponse,
  patchNote,
  readCsrId,
} from "./notesModel";

interface Props {
  roomName: string;
}

export function CallNotesPanel({ roomName }: Props) {
  const [callIds, setCallIds] = useState<string[] | null>(null);
  const [selectedCallId, setSelectedCallId] = useState<string | null>(null);
  const [data, setData] = useState<NotesResponse | null>(null);
  const [error, setError] = useState<string>("");
  const [conflictMsg, setConflictMsg] = useState<string | null>(null);

  useEffect(() => {
    setCallIds(null);
    setSelectedCallId(null);
    setData(null);
    setError("");
    listCallIdsByRoom(roomName)
      .then((ids) => {
        setCallIds(ids);
        setSelectedCallId(ids[0] ?? null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "加载通话列表失败");
      });
  }, [roomName]);

  useEffect(() => {
    if (!selectedCallId) return;
    setData(null);
    fetchNotes(selectedCallId)
      .then(setData)
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : "加载笔记失败");
      });
  }, [selectedCallId]);

  const refresh = async () => {
    setConflictMsg(null);
    if (!selectedCallId) return;
    const fresh = await fetchNotes(selectedCallId);
    setData(fresh);
  };

  if (error) {
    return (
      <div className="notes-panel notes-panel--error">
        无法加载 AI 提炼要点：{error}
      </div>
    );
  }

  if (callIds === null) {
    return <div className="notes-panel notes-panel--loading">加载中…</div>;
  }

  if (callIds.length === 0) {
    return (
      <div className="notes-panel notes-panel--empty">
        本次房间无 AI 提炼记录。
      </div>
    );
  }

  if (data === null) {
    return <div className="notes-panel notes-panel--loading">加载中…</div>;
  }

  return (
    <section className="notes-panel">
      <h3 className="notes-panel__title">AI 提炼要点</h3>
      {conflictMsg && (
        <div className="notes-panel__conflict-banner">
          {conflictMsg}
          <button type="button" onClick={refresh}>
            刷新
          </button>
        </div>
      )}
      {callIds.length > 1 && (
        <select
          value={selectedCallId ?? ""}
          onChange={(e) => setSelectedCallId(e.target.value)}
        >
          {callIds.map((cid) => (
            <option key={cid} value={cid}>
              {cid}
            </option>
          ))}
        </select>
      )}
      <ul className="notes-panel__list">
        {data.current_items.map((item: NoteItem) => (
          <NoteRow
            key={item.key}
            item={item}
            callId={selectedCallId ?? ""}
            onSaved={refresh}
            onConflict={(currentText, _eventId) =>
              setConflictMsg(
                `此条已被 AI 或他人更新，当前文本：「${currentText}」`,
              )
            }
          />
        ))}
      </ul>
    </section>
  );
}

function NoteRow({
  item,
  callId,
  onSaved,
  onConflict,
}: {
  item: NoteItem;
  callId: string;
  onSaved: () => void;
  onConflict: (currentText: string, currentEventId: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(item.text);
  const [saving, setSaving] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleSave = async () => {
    setSaving(true);
    setErrorMsg(null);
    try {
      const result = await patchNote({
        callId,
        itemKey: item.key,
        beforeText: item.text,
        afterText: draft,
        basedOnEventId: item.latest_event_id,
        csrId: readCsrId(),
      });
      if (!result.ok && result.conflict) {
        onConflict(
          result.conflict.current_text,
          result.conflict.current_event_id,
        );
      } else {
        setEditing(false);
        onSaved();
      }
    } catch (e) {
      setErrorMsg(String(e));
    } finally {
      setSaving(false);
    }
  };

  return (
    <li className={item.expired ? "is-expired" : ""}>
      <span className="notes-category">[{item.category}]</span>
      {editing ? (
        <>
          <input
            className="notes-edit-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            disabled={saving}
          />
          <button
            type="button"
            onClick={handleSave}
            disabled={saving || draft === item.text}
          >
            保存
          </button>
          <button
            type="button"
            onClick={() => {
              setEditing(false);
              setDraft(item.text);
            }}
          >
            取消
          </button>
          {errorMsg && <span className="notes-error">{errorMsg}</span>}
        </>
      ) : (
        <>
          <span className="notes-text">{item.text}</span>
          <button type="button" onClick={() => setEditing(true)}>
            编辑
          </button>
          {item.edit_history.length > 0 && (
            <span
              className="notes-edited"
              title={`已编辑 ${item.edit_history.length} 次`}
            >
              ✎
            </span>
          )}
        </>
      )}
    </li>
  );
}
