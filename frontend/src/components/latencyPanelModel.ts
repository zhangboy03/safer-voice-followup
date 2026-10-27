interface LatencyDisplaySource {
  e2e_ms: number;
  speech_end_to_audio_ms: number;
}

export function getDisplayLatencyMs(metrics: LatencyDisplaySource): number {
  return metrics.speech_end_to_audio_ms >= 0
    ? metrics.speech_end_to_audio_ms
    : metrics.e2e_ms;
}
