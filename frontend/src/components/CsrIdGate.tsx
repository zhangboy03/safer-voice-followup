import { type ReactNode, useEffect, useRef, useState } from "react";

const STORAGE_KEY = "csrId";

export function CsrIdGate({ children }: { children: ReactNode }) {
  const [csrId, setCsrId] = useState<string | null>(() =>
    window.localStorage.getItem(STORAGE_KEY),
  );
  const [inputValue, setInputValue] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!csrId) {
      inputRef.current?.focus();
    }
  }, [csrId]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const trimmed = inputValue.trim();
    if (!trimmed) return;
    window.localStorage.setItem(STORAGE_KEY, trimmed);
    setCsrId(trimmed);
  };

  if (csrId) {
    return <>{children}</>;
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
      }}
    >
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 16,
          padding: "48px 56px",
          background: "var(--surface)",
          borderWidth: "3px 1px 1px 1px",
          borderStyle: "solid",
          borderColor:
            "var(--primary) var(--border) var(--border) var(--border)",
          borderRadius: "var(--radius-xl)",
          boxShadow: "var(--shadow-lg)",
          minWidth: 360,
        }}
      >
        <h1
          style={{
            margin: 0,
            fontFamily: "var(--font-display)",
            fontSize: 22,
            fontWeight: 700,
            color: "var(--primary)",
          }}
        >
          品驰 AI SOP 调试台
        </h1>
        <label
          htmlFor="csr-id-input"
          style={{
            fontSize: 13,
            color: "var(--text-secondary)",
          }}
        >
          请输入你的工号 / 姓名
        </label>
        <input
          ref={inputRef}
          id="csr-id-input"
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          placeholder="如 csr-01 或 张三"
          style={{
            padding: "12px 14px",
            fontSize: 15,
            borderRadius: "var(--radius-md)",
            border: "1px solid var(--border)",
            background: "var(--bg)",
            color: "var(--text)",
            outline: "none",
          }}
        />
        <button
          type="submit"
          disabled={!inputValue.trim()}
          style={{
            padding: "12px 24px",
            fontSize: 14,
            fontWeight: 600,
            borderRadius: "var(--radius-md)",
            border: "none",
            background: inputValue.trim() ? "var(--primary)" : "var(--border)",
            color: "white",
            cursor: inputValue.trim() ? "pointer" : "not-allowed",
          }}
        >
          进入
        </button>
      </form>
    </div>
  );
}

export function clearCsrId() {
  window.localStorage.removeItem(STORAGE_KEY);
  window.location.reload();
}
