import {
  type DataHandler,
  LiveKitService,
  type LiveKitStatus,
} from "./LiveKitService";

export const SOLO_CONTRACT = "solo.v1" as const;
export type SoloPhase =
  | "idle"
  | "connecting"
  | "starting"
  | "listening"
  | "speaking"
  | "reconnecting"
  | "ended"
  | "failed";
export interface SoloMessage {
  id: string;
  role: "patient" | "assistant";
  text: string;
  final: boolean;
  source: "asr" | "generated";
  interrupted?: boolean;
}
export interface SoloField {
  value: unknown;
  status: "captured" | "refused";
  evidence: string;
  group: string;
}
export interface SoloSnapshot {
  contractVersion: typeof SOLO_CONTRACT;
  speechLanguage: "en";
  sessionId: string | null;
  phase: SoloPhase;
  ready: boolean;
  levels: { patient: number; assistant: number };
  messages: SoloMessage[];
  record: {
    revision: number;
    source: "solo_async_extraction";
    fields: Record<string, SoloField>;
    missingFields: string[];
  };
  metrics: {
    turnId: string | null;
    scope: "server_speech_end_to_audio_queued";
    speechEndToQueuedMs: number | null;
    llmTtftMs: number | null;
    ttsTtfaMs: number | null;
  };
  error: string | null;
}
export type SoloEvent =
  | { type: "transcript"; message: SoloMessage }
  | { type: "record"; record: SoloSnapshot["record"] }
  | { type: "metrics"; metrics: SoloSnapshot["metrics"] }
  | { type: "interrupted"; contextId: string }
  | { type: "ready" }
  | { type: "ended"; reason: string }
  | { type: "error"; message: string };
interface SoloTransport {
  prepareMicrophone?(): Promise<void>;
  connect(url: string, token: string): Promise<void>;
  disconnect(): Promise<void>;
  onData(handler: DataHandler): () => void;
  onStatus(handler: (status: LiveKitStatus) => void): () => void;
}
interface Lease {
  session_id: string;
  token: string;
  control_token: string;
  livekit_url: string;
  contract_version: string;
  mode: string;
}
interface Options {
  apiBaseUrl?: string;
  fetch?: typeof fetch;
  createTransport?: () => SoloTransport;
  getSupportError?: () => string | null;
}
const text = (v: unknown) => (typeof v === "string" ? v : "");
const number = (v: unknown) =>
  typeof v === "number" && Number.isFinite(v) && v >= 0 ? v : null;
export function initialSoloSnapshot(): SoloSnapshot {
  return {
    contractVersion: SOLO_CONTRACT,
    speechLanguage: "en",
    sessionId: null,
    phase: "idle",
    ready: false,
    levels: { patient: 0, assistant: 0 },
    messages: [],
    record: {
      revision: 0,
      source: "solo_async_extraction",
      fields: {},
      missingFields: [],
    },
    metrics: {
      turnId: null,
      scope: "server_speech_end_to_audio_queued",
      speechEndToQueuedMs: null,
      llmTtftMs: null,
      ttsTtfaMs: null,
    },
    error: null,
  };
}

/** Single owner of one Solo call. Rendering code never handles tokens or Room. */
export class SoloDemoClient {
  private state = initialSoloSnapshot();
  private listeners = new Set<(state: SoloSnapshot) => void>();
  private events = new Set<(event: SoloEvent) => void>();
  private generation = 0;
  private service: SoloTransport | null = null;
  private lease: Lease | null = null;
  private cleanupLeases = new Map<string, Lease>();
  private startTask: Promise<void> | null = null;
  private stopTask: Promise<void> | null = null;
  private readyTimer: ReturnType<typeof setTimeout> | null = null;
  private backendReady = false;
  private completedPatientTurns = new Set<string>();
  private connected = false;
  private audioSubscribed = false;
  private lastAgentAudio = -Infinity;
  private request: typeof fetch;
  private apiBase: string;
  private makeTransport: () => SoloTransport;
  private getSupportError: () => string | null;

  constructor(options: Options = {}) {
    this.getSupportError =
      options.getSupportError ?? LiveKitService.getSupportError;
    this.request = options.fetch ?? globalThis.fetch.bind(globalThis);
    this.apiBase = (options.apiBaseUrl ?? "").replace(/\/$/, "");
    this.makeTransport =
      options.createTransport ?? (() => new LiveKitService());
  }
  getSnapshot = (): SoloSnapshot => this.state;
  subscribe = (listener: (state: SoloSnapshot) => void): (() => void) => {
    this.listeners.add(listener);
    listener(this.state);
    return () => {
      this.listeners.delete(listener);
    };
  };
  onEvent(listener: (event: SoloEvent) => void): () => void {
    this.events.add(listener);
    return () => {
      this.events.delete(listener);
    };
  }
  private publish(patch: Partial<SoloSnapshot>) {
    this.state = { ...this.state, ...patch };
    for (const listener of this.listeners) {
      try {
        listener(this.state);
      } catch {
        /* Consumer failures never stop media. */
      }
    }
  }
  private emit(event: SoloEvent) {
    for (const listener of this.events) {
      try {
        listener(event);
      } catch {
        /* Isolate UI animations. */
      }
    }
  }
  private updatePhase() {
    const ready = this.backendReady && this.connected && this.audioSubscribed;
    const wasReady = this.state.ready;
    const speaking = performance.now() - this.lastAgentAudio < 250;
    this.publish({
      ready,
      phase: ready
        ? speaking
          ? "speaking"
          : "listening"
        : this.connected
          ? "starting"
          : this.state.phase,
    });
    if (ready && !wasReady) {
      if (this.readyTimer) clearTimeout(this.readyTimer);
      this.readyTimer = null;
      this.emit({ type: "ready" });
    }
  }
  private status(status: LiveKitStatus) {
    if (status.error) {
      this.fail(status.error);
      return;
    }
    this.connected = status.state === "connected";
    this.audioSubscribed = status.audioSubscribed;
    if (status.agentLevel > 0.01) this.lastAgentAudio = performance.now();
    this.publish({
      levels: {
        patient: Math.min(1, number(status.microphoneLevel) ?? 0),
        assistant: Math.min(1, number(status.agentLevel) ?? 0),
      },
    });
    if (status.state === "reconnecting")
      this.publish({ phase: "reconnecting", ready: false });
    else if (status.state === "failed")
      this.fail("The voice connection failed. Please retry.");
    else if (status.state === "disconnected")
      void this.stop().catch((error) => this.fail(String(error)));
    else this.updatePhase();
  }
  private data(topic: string, p: Record<string, unknown>) {
    if (topic === "agent_ready") {
      this.backendReady = true;
      this.updatePhase();
      return;
    }
    if (
      [
        "asr_interim",
        "asr_final",
        "transcription",
        "smart_turn_emit",
        "assistant_text",
      ].includes(topic)
    ) {
      const assistant = topic === "assistant_text";
      const patientTurn = text(p.turn_id);
      if (
        !assistant &&
        topic !== "smart_turn_emit" &&
        this.completedPatientTurns.has(patientTurn)
      )
        return;
      if (topic === "smart_turn_emit") {
        this.completedPatientTurns.add(patientTurn);
        if (this.completedPatientTurns.size > 128) {
          const oldest = this.completedPatientTurns.values().next().value;
          if (oldest !== undefined) this.completedPatientTurns.delete(oldest);
        }
      }
      const value = text(p.text);
      if (!value) return;
      const id = `${assistant ? "assistant" : "patient"}-${text(assistant ? p.context_id : p.turn_id) || "opening"}`;
      const message: SoloMessage = {
        id,
        role: assistant ? "assistant" : "patient",
        text: value,
        final: assistant ? p.is_final === true : topic !== "asr_interim",
        source: assistant ? "generated" : "asr",
      };
      const found = this.state.messages.some((m) => m.id === id);
      this.publish({
        messages: found
          ? this.state.messages.map((m) => (m.id === id ? message : m))
          : [...this.state.messages.slice(-99), message],
      });
      this.emit({ type: "transcript", message });
    } else if (
      topic === "dual_llm_debug" &&
      p.event === "advance_decision" &&
      p.record_source === "solo_async_extraction"
    ) {
      const revision = number(p.capture_revision);
      if (revision === null || revision <= this.state.record.revision) return;
      const fields: Record<string, SoloField> = Object.create(null);
      if (
        p.collected_fields &&
        typeof p.collected_fields === "object" &&
        !Array.isArray(p.collected_fields)
      ) {
        for (const [key, raw] of Object.entries(p.collected_fields)) {
          if (!raw || typeof raw !== "object" || Array.isArray(raw)) continue;
          const v = raw as Record<string, unknown>;
          if (v.status !== "confirmed" && v.status !== "refused") continue;
          fields[key] = {
            value: v.value,
            status: v.status === "confirmed" ? "captured" : "refused",
            evidence: text(v.evidence),
            group: text(v.source_node),
          };
        }
      }
      const record: SoloSnapshot["record"] = {
        revision,
        source: "solo_async_extraction",
        fields,
        missingFields: Array.isArray(p.policy_missing)
          ? p.policy_missing.filter((v): v is string => typeof v === "string")
          : [],
      };
      this.publish({ record });
      this.emit({ type: "record", record });
    } else if (topic === "turn_metrics") {
      const turnId = text(p.turn_id) || null;
      const previous =
        turnId === this.state.metrics.turnId
          ? this.state.metrics
          : initialSoloSnapshot().metrics;
      const metrics: SoloSnapshot["metrics"] = {
        turnId,
        scope: "server_speech_end_to_audio_queued",
        speechEndToQueuedMs:
          number(p.output_queue_wait_ms) !== null
            ? (number(p.speech_end_to_audio_ms) ?? previous.speechEndToQueuedMs)
            : previous.speechEndToQueuedMs,
        llmTtftMs: number(p.llm_ttft_ms) ?? previous.llmTtftMs,
        ttsTtfaMs: number(p.tts_ttfa_ms) ?? previous.ttsTtfaMs,
      };
      this.publish({ metrics });
      this.emit({ type: "metrics", metrics });
    } else if (topic === "tts_interrupted") {
      const contextId = text(p.context_id);
      this.publish({
        messages: this.state.messages.map((m) =>
          m.id === `assistant-${contextId}` ? { ...m, interrupted: true } : m,
        ),
      });
      this.emit({ type: "interrupted", contextId });
    } else if (topic === "error")
      this.fail(text(p.message) || "The Solo voice pipeline stopped.");
    else if (topic === "session_closed") {
      this.emit({ type: "ended", reason: text(p.reason) });
      void this.stop().catch((error) => this.fail(String(error)));
    }
  }
  private fail(message: string) {
    // stop() invalidates stale callbacks immediately; the error survives cleanup.
    void this.stop().catch(() => {});
    this.publish({ phase: "failed", ready: false, error: message });
    this.emit({ type: "error", message });
  }
  start(): Promise<void> {
    if (this.stopTask) return this.stopTask.then(() => this.start());
    if (this.startTask) return this.startTask;
    if (this.service && !this.stopTask) return Promise.resolve();
    const task = this.begin();
    this.startTask = task;
    void task
      .finally(() => {
        if (this.startTask === task) this.startTask = null;
      })
      .catch(() => {});
    return task;
  }
  private async begin() {
    if (this.stopTask) await this.stopTask;
    const generation = ++this.generation;
    this.backendReady = false;
    this.connected = false;
    this.audioSubscribed = false;
    this.lastAgentAudio = -Infinity;
    this.state = initialSoloSnapshot();
    this.completedPatientTurns.clear();
    this.publish({ phase: "connecting" });
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout> | null = null;
    try {
      const supportError = this.getSupportError();
      if (supportError) throw new Error(supportError);
      const service = this.makeTransport();
      this.service = service;
      service.onData((topic, p) => {
        if (generation === this.generation) this.data(topic, p);
      });
      service.onStatus((status) => {
        if (generation === this.generation) this.status(status);
      });
      // Browser/OS permission prompts are human-paced, before server admission.
      await service.prepareMicrophone?.();
      if (generation !== this.generation) return;
      timeout = setTimeout(() => controller.abort(), 15000);
      const response = await this.request(`${this.apiBase}/api/demo/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ participant_name: "demo-visitor" }),
        signal: controller.signal,
      });
      if (!response.ok)
        throw new Error(
          response.status === 503
            ? "All Solo lines are busy or starting. Please retry shortly."
            : `Could not start the call (${response.status}).`,
        );
      const lease = (await response.json()) as Lease;
      clearTimeout(timeout);
      timeout = null;
      if (
        lease.contract_version !== SOLO_CONTRACT ||
        lease.mode !== "solo" ||
        !lease.token ||
        !lease.control_token ||
        !lease.session_id ||
        !lease.livekit_url
      )
        throw new Error("The server returned an incompatible Solo session.");
      if (generation !== this.generation) {
        await this.endLease(lease);
        return;
      }
      this.lease = lease;
      this.publish({ sessionId: lease.session_id, phase: "starting" });
      await service.connect(lease.livekit_url, lease.token);
      if (generation !== this.generation) await service.disconnect();
      else if (!this.state.ready) {
        this.readyTimer = setTimeout(() => {
          if (generation === this.generation)
            this.fail(
              this.backendReady
                ? "The voice audio track did not arrive within 35 seconds. Please retry."
                : "The voice assistant did not finish starting within 35 seconds. Please retry.",
            );
        }, 35000);
      }
    } catch (error) {
      if (generation === this.generation) {
        // Do not await stop() here: it can be waiting for this startup promise.
        this.fail(
          error instanceof Error ? error.message : "Could not start Solo.",
        );
        throw error;
      }
    } finally {
      if (timeout !== null) clearTimeout(timeout);
    }
  }
  private async endLease(lease: Lease) {
    this.cleanupLeases.set(lease.session_id, lease);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 8000);
    try {
      const response = await this.request(
        this.apiBase +
          `/api/demo/sessions/${encodeURIComponent(lease.session_id)}/end`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ control_token: lease.control_token }),
          signal: controller.signal,
          keepalive: true,
        },
      );
      if (!response.ok && response.status !== 404)
        throw new Error(
          `Server did not confirm call cleanup (${response.status}).`,
        );
      this.cleanupLeases.delete(lease.session_id);
    } finally {
      clearTimeout(timeout);
    }
  }
  stop(): Promise<void> {
    if (this.stopTask) return this.stopTask;
    ++this.generation;
    if (this.readyTimer) clearTimeout(this.readyTimer);
    this.readyTimer = null;
    const service = this.service,
      lease = this.lease,
      starting = this.startTask;
    this.service = null;
    this.lease = null;
    if (lease) this.cleanupLeases.set(lease.session_id, lease);
    this.backendReady = false;
    this.connected = false;
    this.audioSubscribed = false;
    this.publish({
      phase: this.state.phase === "failed" ? "failed" : "ended",
      ready: false,
      levels: { patient: 0, assistant: 0 },
    });
    const task = (async () => {
      // Release the microphone immediately, even while HTTP startup is pending.
      let mediaError: unknown;
      try {
        await service?.disconnect();
      } catch (error) {
        mediaError = error;
      }
      await starting?.catch(() => {});
      await Promise.all(
        [...this.cleanupLeases.values()].map((old) => this.endLease(old)),
      );
      if (mediaError) throw mediaError;
    })();
    this.stopTask = task;
    void task
      .finally(() => {
        if (this.stopTask === task) this.stopTask = null;
      })
      .catch(() => {});
    return task;
  }
  async reset(): Promise<void> {
    await this.stop();
    this.state = initialSoloSnapshot();
    this.publish({});
  }
  async dispose(): Promise<void> {
    try {
      await this.stop();
    } finally {
      this.listeners.clear();
      this.events.clear();
    }
  }
}
