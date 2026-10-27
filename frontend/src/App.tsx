import { lazy, Suspense } from "react";

const ConversationRoom = lazy(() =>
  import("./components/ConversationRoom").then((module) => ({
    default: module.ConversationRoom,
  })),
);
const AdminPages = lazy(() =>
  import("./components/AdminPages").then((module) => ({
    default: module.AdminPages,
  })),
);

export default function App() {
  const path = window.location.pathname;
  if (["/room", "/lab", "/expo"].includes(path)) {
    if (path !== "/room") window.history.replaceState(null, "", "/room");
    return (
      <Suspense
        fallback={
          <p className="page-loading" role="status">
            Opening the conversation room…
          </p>
        }
      >
        <ConversationRoom />
      </Suspense>
    );
  }
  if (path === "/debug" || path === "/feedback") {
    return (
      <Suspense
        fallback={
          <p className="page-loading" role="status">
            Opening workspace…
          </p>
        }
      >
        <AdminPages />
      </Suspense>
    );
  }
  return (
    <iframe
      className="presentation-frame"
      title="SAFER motion poster"
      src={`/presentation/index.html?v=motion-20261026-1${window.location.hash}`}
      onLoad={(event) => event.currentTarget.contentWindow?.focus()}
      allow="fullscreen"
      allowFullScreen
    />
  );
}
