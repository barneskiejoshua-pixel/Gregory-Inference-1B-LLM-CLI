"""Decode-latency benchmark: per-token latency percentiles + peak RSS.

Mean tok/s hides the tail: an occasional slow token (e.g. an allocator
hiccup) barely moves the average but spikes p99. The decode hot-path pooling
work targets exactly that tail, so this harness measures the distribution
(p50/p90/p99) instead of a single mean, plus process peak resident memory.

The shape is borrowed from the Nemotron-3.5-ASR on-device harness (which
reports p50/p90/p99 latency, RTF, peak memory). RTF -- real-time factor,
compute time over audio duration -- is ASR-specific and has no text-generation
analog, so it is intentionally omitted in favour of tokens/sec.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .model import Gregory
from .sampler import sample


def _peak_rss_mb() -> float | None:
    """Return process peak resident set size in MiB from /proc/self/status
    (the VmHWM high-water mark), or None when unavailable (non-Linux)."""
    try:
        with open("/proc/self/status", encoding="ascii") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


@dataclass
class BenchResult:
    """Outcome of one decode benchmark: timing distribution + memory.

    `per_token_ms` holds the wall-clock cost of each timed decode step
    (forward + sample); warmup steps are excluded before this is built."""

    prompt_tokens: int
    prefill_ms: float
    per_token_ms: list[float] = field(default_factory=list)
    peak_rss_mb: float | None = None

    def _pct(self, q: float) -> float:
        """Percentile `q` (0-100) of the per-token latencies, in ms."""
        return float(np.percentile(self.per_token_ms, q))

    def stats(self) -> dict:
        """Return the summary metrics as a flat dict of floats/ints."""
        ms = self.per_token_ms
        total = float(sum(ms))
        tok_s = 1000.0 * len(ms) / total if total > 0.0 else 0.0
        return {
            "prompt_tokens": self.prompt_tokens,
            "decode_tokens": len(ms),
            "prefill_ms": self.prefill_ms,
            "p50_ms": self._pct(50),
            "p90_ms": self._pct(90),
            "p99_ms": self._pct(99),
            "mean_ms": float(np.mean(ms)) if ms else 0.0,
            "tok_per_s": tok_s,
            "peak_rss_mb": self.peak_rss_mb,
        }


def bench_decode(
    model: Gregory,
    prompt_ids: list[int],
    *,
    decode_tokens: int = 64,
    warmup: int = 3,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int = 0,
) -> BenchResult:
    """Prefill `prompt_ids`, then time `decode_tokens` decode steps one by one.

    Each timed step is one `model.forward([id], kv)` plus one `sample(...)` --
    the true per-token cost a user feels. `warmup` leading steps run untimed so
    first-token allocation and cache warm-up do not pollute the tail. A fixed
    `seed` makes the token path reproducible run to run."""
    kv = model.init_kv_cache()
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    logits = model.forward(prompt_ids, kv, last_only=True)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    recent: list[int] = list(prompt_ids)
    next_id = sample(logits[-1], temperature, top_p,
                     recent_ids=recent, rng=rng)

    per_token_ms: list[float] = []
    for step in range(warmup + decode_tokens):
        t = time.perf_counter()
        logits = model.forward([next_id], kv)
        next_id = sample(logits[0], temperature, top_p,
                         recent_ids=recent, rng=rng)
        dt = (time.perf_counter() - t) * 1000.0
        recent.append(next_id)
        if step >= warmup:
            per_token_ms.append(dt)

    return BenchResult(
        prompt_tokens=len(prompt_ids),
        prefill_ms=prefill_ms,
        per_token_ms=per_token_ms,
        peak_rss_mb=_peak_rss_mb(),
    )


# Below this many timed tokens, the 99th percentile is essentially the single
# worst sample and swings with OS jitter -- not a stable tail estimate.
P99_STABLE_MIN_TOKENS = 100


def format_report(result: BenchResult, warmup: int) -> str:
    """Render a BenchResult as a human-readable multi-line report.

    When fewer than `P99_STABLE_MIN_TOKENS` were timed, a caveat is appended:
    p99 from a small sample is dominated by a single outlier, not the tail."""
    s = result.stats()
    rss = ("n/a" if s["peak_rss_mb"] is None
           else f"{s['peak_rss_mb']:.0f} MiB")
    lines = [
        f"decode benchmark  ({s['decode_tokens']} tokens timed, "
        f"{warmup} warmup discarded)",
        f"  prompt tokens : {s['prompt_tokens']}",
        f"  prefill       : {s['prefill_ms']:.1f} ms",
        f"  per-token p50 : {s['p50_ms']:.2f} ms",
        f"           p90  : {s['p90_ms']:.2f} ms",
        f"           p99  : {s['p99_ms']:.2f} ms",
        f"           mean : {s['mean_ms']:.2f} ms"
        f"   ({s['tok_per_s']:.2f} tok/s)",
        f"  peak RSS      : {rss}",
    ]
    if s["decode_tokens"] < P99_STABLE_MIN_TOKENS:
        lines.append(
            f"  note: p99 is noisy below {P99_STABLE_MIN_TOKENS} timed "
            f"tokens (it tracks the single worst sample); pass "
            f"--decode-tokens {P99_STABLE_MIN_TOKENS}+ for a stable tail.")
    return "\n".join(lines)
