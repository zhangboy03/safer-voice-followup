import { CsrIdGate } from "./CsrIdGate";
import { FeedbackReviewPage } from "./FeedbackReviewPage";
import { VoiceAgent } from "./VoiceAgent";
import "../admin.css";

export function AdminPages() {
  if (window.location.pathname === "/feedback") return <FeedbackReviewPage />;
  return (
    <CsrIdGate>
      <VoiceAgent />
    </CsrIdGate>
  );
}
