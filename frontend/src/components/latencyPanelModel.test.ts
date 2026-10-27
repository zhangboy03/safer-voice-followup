import { getDisplayLatencyMs } from "./latencyPanelModel.js";

function assertEqual<T>(actual: T, expected: T, message: string) {
  if (actual !== expected) {
    throw new Error(
      `${message}: expected ${String(expected)}, got ${String(actual)}`,
    );
  }
}

assertEqual(
  getDisplayLatencyMs({ speech_end_to_audio_ms: 1004, e2e_ms: 804 }),
  1004,
  "display latency should prefer the full speech-end-to-audio measurement",
);

assertEqual(
  getDisplayLatencyMs({ speech_end_to_audio_ms: -1, e2e_ms: 804 }),
  804,
  "display latency should fall back to server e2e when full latency is missing",
);
