import {
  createLocalAudioTrack,
  isBrowserSupported,
  type LocalAudioTrack,
  type RemoteTrack,
  Room,
  RoomEvent,
  Track,
} from "livekit-client";

export type DataHandler = (
  topic: string,
  payload: Record<string, unknown>,
) => void;

export type LiveKitConnectionState =
  | "idle"
  | "connecting"
  | "connected"
  | "reconnecting"
  | "disconnected"
  | "failed";

export interface LiveKitStatus {
  state: LiveKitConnectionState;
  error: string;
  microphoneLevel: number;
  agentLevel: number;
  audioSubscribed: boolean;
}

type StatusHandler = (status: LiveKitStatus) => void;

export class LiveKitService {
  static getSupportError(): string | null {
    if (!isBrowserSupported()) {
      if (/\b(?:Chrome|Chromium)\//.test(navigator.userAgent))
        return "Live audio is disabled in Chrome. Allow WebRTC for this site in your browser or privacy extension, then reload this page.";
      return "This browser cannot run live audio calls. Open this page in Chrome to start a conversation.";
    }
    if (!navigator.mediaDevices?.getUserMedia)
      return "Microphone access is unavailable. Open this HTTPS page in Chrome to start a conversation.";
    return null;
  }

  private room: Room;
  private dataHandlers = new Set<DataHandler>();
  private statusHandlers = new Set<StatusHandler>();
  private audioElements = new Map<RemoteTrack, HTMLMediaElement>();
  private levelsTimer: ReturnType<typeof setInterval> | null = null;
  private closed = false;
  private microphone: LocalAudioTrack | null = null;
  private cancelMicrophonePreparation: (() => void) | null = null;
  private status: LiveKitStatus = {
    state: "idle",
    error: "",
    microphoneLevel: 0,
    agentLevel: 0,
    audioSubscribed: false,
  };

  constructor() {
    this.room = new Room({
      audioCaptureDefaults: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1,
      },
    });
    // A service owns one Room and installs each listener exactly once.
    this.room.on(RoomEvent.DataReceived, (data: Uint8Array) => {
      let decoded: Record<string, unknown>;
      try {
        const value: unknown = JSON.parse(new TextDecoder().decode(data));
        if (!value || typeof value !== "object" || Array.isArray(value)) return;
        decoded = value as Record<string, unknown>;
      } catch {
        return;
      }
      const topic =
        typeof decoded.topic === "string" ? decoded.topic : "unknown";
      if (topic === "error" || decoded.event_type === "ErrorEvent") {
        this.updateStatus({
          error:
            typeof decoded.message === "string"
              ? decoded.message
              : "Voice pipeline error",
        });
      }
      for (const handler of this.dataHandlers) {
        try {
          handler(topic, decoded);
        } catch (error) {
          console.error("LiveKit data subscriber failed", error);
        }
      }
    });
    this.room.on(RoomEvent.TrackSubscribed, (track: RemoteTrack) => {
      this.attachTrack(track);
    });
    this.room.on(RoomEvent.TrackUnsubscribed, (track: RemoteTrack) => {
      track.detach().forEach((element) => {
        element.remove();
      });
      this.audioElements.delete(track);
      this.updateStatus({ audioSubscribed: this.audioElements.size > 0 });
    });
    this.room.on(RoomEvent.Reconnecting, () => {
      if (!this.closed) this.updateStatus({ state: "reconnecting" });
    });
    this.room.on(RoomEvent.Reconnected, () => {
      if (!this.closed) this.updateStatus({ state: "connected", error: "" });
    });
    this.room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
      if (!this.closed && !this.room.canPlaybackAudio) {
        this.updateStatus({
          error:
            "Audio playback is blocked. Allow sound in this browser, then retry the call.",
        });
      }
    });
    this.room.on(RoomEvent.Disconnected, () => {
      this.clearPlayback();
      this.updateStatus({
        state: this.status.state === "failed" ? "failed" : "disconnected",
        microphoneLevel: 0,
        agentLevel: 0,
        audioSubscribed: false,
      });
    });
  }

  private updateStatus(patch: Partial<LiveKitStatus>): void {
    this.status = { ...this.status, ...patch };
    for (const handler of this.statusHandlers) handler(this.status);
  }

  private attachTrack(track: RemoteTrack): void {
    if (
      this.closed ||
      track.kind !== Track.Kind.Audio ||
      this.audioElements.has(track)
    )
      return;
    // Venue measurements showed 130–350ms of adaptive receive buffering.
    // A 40ms target keeps interactive speech responsive; the browser may still
    // increase its actual buffer when the network needs it.
    const receiver:
      | { jitterBufferTarget?: number | null; playoutDelayHint?: number }
      | undefined = track.receiver;
    if (receiver) {
      try {
        if ("jitterBufferTarget" in receiver) receiver.jitterBufferTarget = 40;
        else if ("playoutDelayHint" in receiver)
          receiver.playoutDelayHint = 0.04;
      } catch {
        /* Keep native buffering on unsupported browser implementations. */
      }
    }
    const element = track.attach();
    element.id = `audio-${track.sid ?? this.audioElements.size}`;
    element.autoplay = true;
    document.body.appendChild(element);
    this.audioElements.set(track, element);
    this.updateStatus({ audioSubscribed: true });
  }

  private clearPlayback(): void {
    if (this.levelsTimer !== null) {
      clearInterval(this.levelsTimer);
      this.levelsTimer = null;
    }
    for (const element of this.audioElements.values()) {
      element.pause();
      element.srcObject = null;
      element.remove();
    }
    this.audioElements.clear();
  }

  async prepareMicrophone(): Promise<void> {
    if (this.closed || this.microphone) return;
    const capture = createLocalAudioTrack({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      channelCount: 1,
    }).then((track) => {
      if (this.closed) track.stop();
      else this.microphone = track;
    });
    let cancel = () => {};
    const cancelled = new Promise<void>((resolve) => {
      cancel = resolve;
    });
    this.cancelMicrophonePreparation = cancel;
    try {
      await Promise.race([capture, cancelled]);
    } finally {
      if (this.cancelMicrophonePreparation === cancel)
        this.cancelMicrophonePreparation = null;
    }
  }

  async connect(url: string, token: string): Promise<void> {
    this.updateStatus({ state: "connecting", error: "" });
    try {
      await this.room.connect(url, token, { autoSubscribe: true });
      if (this.closed) {
        await this.room.disconnect();
        return;
      }
      await this.room.startAudio();
      for (const participant of this.room.remoteParticipants.values()) {
        for (const publication of participant.trackPublications.values()) {
          if (publication.track && publication.isSubscribed)
            this.attachTrack(publication.track);
        }
      }
      if (this.microphone)
        await this.room.localParticipant.publishTrack(this.microphone, {
          source: Track.Source.Microphone,
        });
      else await this.room.localParticipant.setMicrophoneEnabled(true);
      if (this.closed) {
        await this.room.disconnect();
        return;
      }
      this.updateStatus({ state: "connected" });
      // LiveKit's measured participant audio levels; no synthetic animation data.
      this.levelsTimer = setInterval(() => {
        let agentLevel = 0;
        for (const participant of this.room.remoteParticipants.values()) {
          agentLevel = Math.max(agentLevel, participant.audioLevel);
        }
        this.updateStatus({
          microphoneLevel: this.room.localParticipant.audioLevel,
          agentLevel,
        });
      }, 100);
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Voice connection failed";
      this.updateStatus({ state: "failed", error: message });
      this.clearPlayback();
      await this.room.disconnect().catch(() => {});
      throw error;
    }
  }

  onData(handler: DataHandler): () => void {
    this.dataHandlers.add(handler);
    return () => {
      this.dataHandlers.delete(handler);
    };
  }

  onStatus(handler: StatusHandler): () => void {
    this.statusHandlers.add(handler);
    handler(this.status);
    return () => {
      this.statusHandlers.delete(handler);
    };
  }

  async disconnect(): Promise<void> {
    this.closed = true;
    this.cancelMicrophonePreparation?.();
    this.microphone?.stop();
    this.microphone = null;
    this.clearPlayback();
    await this.room.disconnect();
    this.updateStatus({
      state: "disconnected",
      microphoneLevel: 0,
      agentLevel: 0,
      audioSubscribed: false,
    });
    this.dataHandlers.clear();
    this.statusHandlers.clear();
    this.room.removeAllListeners();
  }
}
