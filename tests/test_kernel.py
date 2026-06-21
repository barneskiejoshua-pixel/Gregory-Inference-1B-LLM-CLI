"""Kernel parity tests (model-free; skipped if the C kernel won't build).

Verify the packed 2-bit ternary kernel agrees with the fp32 reference matvec.
The only difference is int8 activation quantization, so agreement is near-exact
(high cosine, small relative error), not bit-identical.
"""

from __future__ import annotations

import numpy as np
import pytest

kernels = pytest.importorskip("gregory.kernels")

if not kernels.available():
    pytest.skip("C kernel unavailable (no compiler/AVX2)",
                allow_module_level=True)


def test_pack_roundtrip_scale():
    """pack_ternary recovers the per-tensor scale exactly."""
    rng = np.random.default_rng(0)
    w = rng.choice([-1, 0, 1], size=(64, 128)).astype(np.float32) * 0.047
    _, scale = kernels.pack_ternary(w)
    assert abs(scale - 0.047) < 1e-6


def test_matvec_matches_fp32_reference():
    """Kernel matvec tracks scale * (W @ x) within int8-quant tolerance."""
    rng = np.random.default_rng(1)
    n, k = 512, 2560
    scale = 0.031
    w = rng.choice([-1, 0, 1], size=(n, k)).astype(np.float32) * scale
    x = rng.standard_normal(k).astype(np.float32)
    packed, s = kernels.pack_ternary(w)
    y_fast = kernels.matvec(packed, s, x, k)
    y_ref = w @ x
    cos = float(y_fast @ y_ref
                / (np.linalg.norm(y_fast) * np.linalg.norm(y_ref)))
    assert cos > 0.999


def test_zero_weight_is_zero():
    """An all-zero weight yields a zero result (no divide-by-zero)."""
    k = 128
    w = np.zeros((8, k), dtype=np.float32)
    packed, s = kernels.pack_ternary(w)
    y = kernels.matvec(packed, s, np.ones(k, dtype=np.float32), k)
    assert np.all(y == 0.0)


def test_avx2_bit_identical_to_scalar():
    """The AVX2 matvec must equal the scalar reference bit-for-bit, across
    aligned and remainder shapes. M values straddle the 8-row block boundary
    (the unroll-8 + prefetch path) so leftover rows exercise the backstop;
    K values straddle the 32-wide vector boundary so the scalar K-tail runs."""
    rng = np.random.default_rng(7)
    for n in (8, 16, 512, 13, 7, 25):          # multiples of 8 and <8 / odd
        for k in (640, 2560, 128, 132):        # mult of 32 and non-mult
            w = rng.choice([-1, 0, 1], size=(n, k)).astype(np.float32) * 0.02
            x = rng.standard_normal(k).astype(np.float32)
            packed, s = kernels.pack_ternary(w)
            y_fast = kernels.matvec(packed, s, x, k)
            y_scalar = kernels.matvec_scalar(packed, s, x, k)
            assert np.array_equal(y_fast, y_scalar), f"mismatch at n={n} k={k}"


def test_int8_head_matches_fp32_reference():
    """The int8 row-quantized head tracks the fp32 head within tolerance."""
    rng = np.random.default_rng(2)
    n, k = 4096, 2560
    emb = (rng.standard_normal((n, k)) * 0.02).astype(np.float32)
    x = rng.standard_normal(k).astype(np.float32)
    q, scale = kernels.pack_head(emb)
    y_fast = kernels.head_matvec(q, scale, x, k)
    y_ref = emb @ x
    cos = float(y_fast @ y_ref
                / (np.linalg.norm(y_fast) * np.linalg.norm(y_ref)))
    assert cos > 0.999
