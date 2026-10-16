"""I/O-compute overlap scheduler (M4).

Two modes:

- ``simulate_overlap`` (pure function, deterministic): discrete-event model of
  double-buffered prefetch. Given per-token fetch and compute times it returns
  sequential vs overlapped totals, tok/s, p99 inter-token latency, and
  bytes/token. Used by tests and by simulator/.
- ``OverlapScheduler`` (live): thread-based orchestration around callables.
  While ``compute_fn`` runs token *t*, the pool prefetches the oracle's union
  mask pages for tokens *t+1..t+K*. On an oracle miss the needed page is faulted
  on demand and counted (the oracle can retrain on misses later).

The live scheduler never invents timings — it measures wall-clock time of the
callables it is given.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np


def simulate_overlap(
    fetch_times: list[float],
    compute_times: list[float],
    bytes_per_token: list[float],
    prefetch_hit_rate: float = 1.0,
) -> dict:
    """Discrete-event double-buffer simulation.

    Sequential baseline: sum(fetch + compute).
    Overlapped: pay fetch[0] up front, then per step max(compute[i], fetch[i+1]).
    A prefetch miss (1 - hit_rate) adds a full extra fetch for that token.
    """
    fetch_times = list(fetch_times)
    compute_times = list(compute_times)
    n = len(fetch_times)
    assert n == len(compute_times) and n == len(bytes_per_token) and n > 0

    sequential = sum(f + c for f, c in zip(fetch_times, compute_times))

    # overlapped timeline; latencies = completion time of each token's compute
    latencies = []
    t = fetch_times[0]
    for i in range(n):
        t += compute_times[i]
        latencies.append(t)
        if i + 1 < n:
            # next fetch overlaps this compute; only the uncovered part extends t
            t += max(0.0, fetch_times[i + 1] - compute_times[i])

    miss_penalty = (1.0 - prefetch_hit_rate) * sum(fetch_times)
    overlapped = t + miss_penalty
    total_bytes = sum(bytes_per_token)
    return {
        "tokens": n,
        "sequential_s": sequential,
        "overlapped_s": overlapped,
        "speedup": sequential / overlapped if overlapped else 0.0,
        "tok_per_s": n / overlapped if overlapped else 0.0,
        "p99_latency_s": float(np.percentile(latencies, 99)),
        "mean_latency_s": float(np.mean(latencies)),
        "bytes_per_token": total_bytes / n,
        "total_bytes": total_bytes,
    }


class OverlapScheduler:
    """Live double-buffered orchestration around user callables.

    - ``fetch_fn(page_ids)``: ensure pages are resident (prefetch).
    - ``compute_fn(token_ctx)``: run one decode step; returns anything.
    - ``pages_for_step(step)``: oracle union-mask page ids needed for a step.
    - ``on_miss(page_id)``: called when a needed page was not resident.
    """

    def __init__(self, lookahead_k: int = 4, workers: int = 2):
        self.k = int(lookahead_k)
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self.stats = {
            "steps": 0,
            "misses": 0,
            "prefetch_calls": 0,
            "total_compute_s": 0.0,
            "total_fetch_s": 0.0,
        }
        self._lock = threading.Lock()

    def run(
        self,
        n_steps: int,
        pages_for_step,
        fetch_fn,
        compute_fn,
        is_resident=None,
    ) -> list:
        """Run n_steps with prefetch of the next K steps overlapped with compute."""
        results = []
        pending: dict[int, object] = {}

        def _fetch(step):
            t0 = time.perf_counter()
            try:
                fetch_fn(pages_for_step(step))
            finally:
                with self._lock:
                    self.stats["total_fetch_s"] += time.perf_counter() - t0
                    self.stats["prefetch_calls"] += 1

        # prime the pipeline: prefetch steps 0..K-1 up front
        for s in range(min(self.k, n_steps)):
            pending[s] = self._pool.submit(_fetch, s)

        for step in range(n_steps):
            fut = pending.pop(step, None)
            if fut is not None:
                fut.result()  # wait for this step's pages (usually already done)
            # schedule prefetch for the step falling off the lookahead edge
            edge = step + self.k
            if edge < n_steps:
                pending[edge] = self._pool.submit(_fetch, edge)

            # oracle-miss check before compute
            if is_resident is not None:
                for pid in pages_for_step(step):
                    if not is_resident(pid):
                        with self._lock:
                            self.stats["misses"] += 1
                        fetch_fn([pid])  # on-demand fault (counted)

            t0 = time.perf_counter()
            results.append(compute_fn(step))
            with self._lock:
                self.stats["total_compute_s"] += time.perf_counter() - t0
                self.stats["steps"] += 1
        return results

    def tok_per_s(self) -> float:
        s = self.stats
        denom = s["total_compute_s"] + s["total_fetch_s"]
        return s["steps"] / denom if denom > 0 else 0.0

    def close(self):
        self._pool.shutdown(wait=True)
