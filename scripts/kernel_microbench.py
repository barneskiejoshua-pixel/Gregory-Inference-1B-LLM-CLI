"""Single-thread microbenchmark of the ternary matvec kernel.

Isolates the kernel from model load / OMP contention / OS jitter: fixed shape,
many reps, reports median + p10/p90 ms so a real change is separable from
machine noise. Run with OMP_NUM_THREADS=1 for a clean per-core comparison."""

from __future__ import annotations

import statistics
import sys
import time

import numpy as np

from gregory import kernels


def run(m: int, k: int, reps: int) -> None:
    """Time `reps` matvec calls on an (m, k) ternary weight; print stats."""
    rng = np.random.default_rng(0)
    w = rng.choice([-1, 0, 1], size=(m, k)).astype(np.float32) * 0.02
    x = rng.standard_normal(k).astype(np.float32)
    packed, s = kernels.pack_ternary(w)
    out = np.empty(m, dtype=np.float32)
    for _ in range(20):                       # warmup
        kernels.matvec(packed, s, x, k, out=out)
    samples = []
    for _ in range(reps):
        t = time.perf_counter()
        kernels.matvec(packed, s, x, k, out=out)
        samples.append((time.perf_counter() - t) * 1e3)
    samples.sort()
    p10 = samples[len(samples) // 10]
    p90 = samples[len(samples) * 9 // 10]
    print(f"M={m} K={k} reps={reps}: median {statistics.median(samples):.4f} "
          f"ms  p10 {p10:.4f}  p90 {p90:.4f}  min {samples[0]:.4f}")


if __name__ == "__main__":
    n_reps = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    for shape in ((2560, 2560), (6912, 2560), (2560, 6912)):
        run(shape[0], shape[1], n_reps)
