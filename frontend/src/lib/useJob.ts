import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Job } from "./types";

/** Poll a background job until it finishes. Call start(jobId) to begin. */
export function useJob(onDone?: () => void) {
  const [job, setJob] = useState<Job | null>(null);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);
  const doneRef = useRef(onDone);
  doneRef.current = onDone;

  const stop = () => {
    if (timer.current) clearInterval(timer.current);
    timer.current = null;
  };

  const start = (jobId: string) => {
    stop();
    timer.current = setInterval(async () => {
      try {
        const j = await api.job(jobId);
        setJob(j);
        if (j.status !== "running") {
          stop();
          if (j.status === "done") doneRef.current?.();
        }
      } catch {
        stop();
      }
    }, 1500);
  };

  useEffect(() => stop, []);
  return { job, start, running: job?.status === "running" };
}
