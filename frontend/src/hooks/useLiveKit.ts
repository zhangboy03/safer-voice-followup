import { useCallback, useEffect, useRef, useState } from "react";
import {
  type DataHandler,
  LiveKitService,
  type LiveKitStatus,
} from "../services/LiveKitService";

export type { DataHandler } from "../services/LiveKitService";

const INITIAL_STATUS: LiveKitStatus = {
  state: "idle",
  error: "",
  microphoneLevel: 0,
  agentLevel: 0,
  audioSubscribed: false,
};

export function useLiveKit() {
  const serviceRef = useRef<LiveKitService | null>(null);
  // Subscriptions belong to the hook, not a Room. They survive hangup/retry.
  const handlersRef = useRef(new Set<DataHandler>());
  const generationRef = useRef(0);
  const [status, setStatus] = useState<LiveKitStatus>(INITIAL_STATUS);
  const [agentReady, setAgentReady] = useState(false);
  const [transcript, setTranscript] = useState("");
  const [agentText, setAgentText] = useState("");

  const connect = useCallback(async (url: string, token: string) => {
    const generation = ++generationRef.current;
    const previous = serviceRef.current;
    serviceRef.current = null;
    await previous?.disconnect();
    if (generation !== generationRef.current) return;
    setTranscript("");
    setAgentText("");
    setAgentReady(false);
    setStatus({ ...INITIAL_STATUS, state: "connecting" });
    for (const handler of handlersRef.current) handler("session_reset", {});

    const service = new LiveKitService();
    serviceRef.current = service;
    service.onStatus((next) => {
      if (serviceRef.current !== service) return;
      setStatus(next);
      if (next.state === "disconnected" || next.state === "failed")
        setAgentReady(false);
    });
    service.onData((topic, payload) => {
      if (serviceRef.current !== service) return;
      if (topic === "agent_ready") setAgentReady(true);
      if (topic === "session_closed") setAgentReady(false);
      if (
        topic === "transcription" ||
        topic === "asr_interim" ||
        topic === "asr_final"
      ) {
        setTranscript(typeof payload.text === "string" ? payload.text : "");
      }
      if (topic === "user_started") {
        setTranscript("");
        setAgentText("");
      }
      if (topic === "assistant_text" && typeof payload.text === "string")
        setAgentText(payload.text);
      if (topic === "agent_text") {
        const delta = typeof payload.delta === "string" ? payload.delta : "";
        setAgentText((previousText) =>
          payload.is_final ? delta : previousText + delta,
        );
      }
      if (
        topic === "dual_llm_debug" &&
        (payload.event === "speech_done" || payload.event === "stage2_done")
      ) {
        if (typeof payload.response_text === "string")
          setAgentText(payload.response_text);
      }
      for (const handler of handlersRef.current) handler(topic, payload);
    });
    try {
      await service.connect(url, token);
    } catch (error) {
      if (serviceRef.current === service) {
        setStatus((current) => ({
          ...current,
          state: "failed",
          error:
            error instanceof Error ? error.message : "Voice connection failed",
        }));
      }
      throw error;
    }
  }, []);

  const disconnect = useCallback(async () => {
    generationRef.current += 1;
    const service = serviceRef.current;
    serviceRef.current = null;
    setStatus(INITIAL_STATUS);
    setAgentReady(false);
    setTranscript("");
    setAgentText("");
    await service?.disconnect();
  }, []);

  const registerDataHandler = useCallback(
    (handler: DataHandler): (() => void) => {
      handlersRef.current.add(handler);
      return () => {
        handlersRef.current.delete(handler);
      };
    },
    [],
  );

  useEffect(
    () => () => {
      generationRef.current += 1;
      const service = serviceRef.current;
      serviceRef.current = null;
      service?.disconnect().catch(() => {});
    },
    [],
  );

  return {
    isConnected:
      status.state === "connected" || status.state === "reconnecting",
    isConnecting: status.state === "connecting",
    connectionState: status.state,
    error: status.error,
    microphoneLevel: status.microphoneLevel,
    agentLevel: status.agentLevel,
    audioSubscribed: status.audioSubscribed,
    agentReady,
    connect,
    disconnect,
    transcript,
    agentText,
    registerDataHandler,
  };
}
