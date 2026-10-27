import { useCallback, useEffect, useRef, useState } from "react";
import {
  initialSoloSnapshot,
  SoloDemoClient,
  type SoloEvent,
  type SoloSnapshot,
} from "../services/soloDemoClient";

/** A hook owns its call. StrictMode teardown cannot dispose a newer instance. */
export function useSoloDemo() {
  const clientRef = useRef<SoloDemoClient | null>(null);
  const eventHandlers = useRef(new Set<(event: SoloEvent) => void>());
  const [state, setState] = useState<SoloSnapshot>(initialSoloSnapshot);
  useEffect(() => {
    const client = new SoloDemoClient();
    clientRef.current = client;
    const unsubscribe = client.subscribe(setState);
    const offEvents = client.onEvent((event) => {
      for (const handler of eventHandlers.current) {
        try {
          handler(event);
        } catch {
          /* Isolate visual consumers. */
        }
      }
    });
    return () => {
      unsubscribe();
      offEvents();
      if (clientRef.current === client) clientRef.current = null;
      void client.dispose().catch(() => {});
    };
  }, []);
  const start = useCallback(
    () => clientRef.current?.start() ?? Promise.resolve(),
    [],
  );
  const stop = useCallback(
    () => clientRef.current?.stop() ?? Promise.resolve(),
    [],
  );
  const reset = useCallback(
    () => clientRef.current?.reset() ?? Promise.resolve(),
    [],
  );
  const onEvent = useCallback((handler: (event: SoloEvent) => void) => {
    eventHandlers.current.add(handler);
    return () => {
      eventHandlers.current.delete(handler);
    };
  }, []);
  return { state, start, stop, reset, onEvent };
}
