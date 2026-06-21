"""Tests for the decode-latency benchmark harness (gregory.bench).

A stub model stands in for the real 2B so the harness logic is exercised
without loading a GGUF: the percentile math, warmup exclusion, token-count
accounting, and graceful peak-RSS handling."""

from __future__ import annotations

import numpy as np

from gregory import bench


class _StubModel:
    """Minimal Gregory-shaped model: forward returns (n_positions, vocab)
    logits so sample() can index [-1] (prefill) and [0] (decode)."""

    def __init__(self, vocab: int = 32) -> None:
        """Create a stub with a small `vocab` and a deterministic RNG."""
        self.vocab = vocab
        self._rng = np.random.default_rng(0)

    def init_kv_cache(self) -> dict:
        """Return a fresh (empty) cache object; the stub ignores its state."""
        return {}

    def forward(self, ids, kv, last_only: bool = False) -> np.ndarray:
        """Return random fp32 logits shaped (len(ids), vocab)."""
        return self._rng.standard_normal(
            (len(ids), self.vocab)).astype(np.float32)


def test_bench_token_accounting() -> None:
    """Timed-token count equals decode_tokens; warmup steps are excluded."""
    res = bench.bench_decode(_StubModel(), [1, 2, 3],
                             decode_tokens=20, warmup=4)
    assert res.prompt_tokens == 3
    assert len(res.per_token_ms) == 20
    assert res.prefill_ms >= 0.0


def test_bench_stats_shape_and_ordering() -> None:
    """stats() exposes the documented keys and percentiles are monotone."""
    res = bench.bench_decode(_StubModel(), [1, 2], decode_tokens=50, warmup=2)
    s = res.stats()
    for key in ("p50_ms", "p90_ms", "p99_ms", "mean_ms", "tok_per_s",
                "decode_tokens", "prompt_tokens"):
        assert key in s
    assert s["p50_ms"] <= s["p90_ms"] <= s["p99_ms"]
    assert s["decode_tokens"] == 50
    assert s["tok_per_s"] > 0.0


def test_peak_rss_is_float_or_none() -> None:
    """Peak RSS is a positive MiB float on Linux, or None elsewhere."""
    rss = bench._peak_rss_mb()
    assert rss is None or (isinstance(rss, float) and rss > 0.0)


def test_format_report_contains_percentiles() -> None:
    """The rendered report names each percentile and the warmup count."""
    res = bench.bench_decode(_StubModel(), [1], decode_tokens=8, warmup=1)
    text = bench.format_report(res, warmup=1)
    for token in ("p50", "p90", "p99", "tok/s", "peak RSS", "warmup"):
        assert token in text


def test_small_sample_p99_caveat() -> None:
    """The p99-noise caveat appears below threshold and not at/above it."""
    small = bench.bench_decode(_StubModel(), [1], decode_tokens=8, warmup=0)
    assert "p99 is noisy" in bench.format_report(small, warmup=0)
    big = bench.bench_decode(_StubModel(), [1],
                             decode_tokens=bench.P99_STABLE_MIN_TOKENS,
                             warmup=0)
    assert "p99 is noisy" not in bench.format_report(big, warmup=0)
