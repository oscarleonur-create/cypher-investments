import { useEffect, useRef, useState } from "react";
import type { Quote } from "./types";

export interface QuotesState {
  quotes: Record<string, Quote>;
  connected: boolean;
  live: boolean; // true once a real tick (not just snapshot) arrives
}

/** Connect to /ws/quotes, keep a {symbol -> quote} map, auto-reconnect. */
export function useQuotes(): QuotesState {
  const [quotes, setQuotes] = useState<Record<string, Quote>>({});
  const [connected, setConnected] = useState(false);
  const [live, setLive] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let closed = false;

    const connect = () => {
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const ws = new WebSocket(`${proto}://${window.location.host}/ws/quotes`);
      wsRef.current = ws;

      ws.onopen = () => setConnected(true);
      ws.onclose = () => {
        setConnected(false);
        if (!closed) retryRef.current = setTimeout(connect, 2500);
      };
      ws.onerror = () => ws.close();
      ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === "snapshot") {
          const seed: Record<string, Quote> = {};
          for (const [sym, q] of Object.entries(msg.quotes || {})) {
            const qq = q as { mark: number; close: number };
            seed[sym] = {
              symbol: sym,
              bid: 0,
              ask: 0,
              mid: qq.mark || qq.close || 0,
              ts: "",
            };
          }
          setQuotes((prev) => ({ ...seed, ...prev }));
        } else if (msg.type === "quote") {
          setLive(true);
          setQuotes((prev) => ({ ...prev, [msg.symbol]: msg as Quote }));
        }
      };
    };

    connect();
    return () => {
      closed = true;
      if (retryRef.current) clearTimeout(retryRef.current);
      wsRef.current?.close();
    };
  }, []);

  return { quotes, connected, live };
}
