"""I2_S ternary dequantization to fp32 -- clean-room NumPy.

Layout (BitNet's ggml fork: ggml_nbytes special-cases I2_S to
`nbytes = n_elems/4 + 32`, and the dequant consumes a single tensor-wide
scale):

    [ n_elems/4 packed bytes ][ 4-byte f32 scale ][ 28-byte pad ]

Packed bytes hold 2-bit codes. Within each 128-element window, 32 bytes carry
4 codes each at bit shifts 6, 4, 2, 0. Code map (ggml-quants.c::map2bit):
    0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0 (3 should never appear; mapped to 0).
Dequantized value = ternary_code * tensor_scale.
"""

from __future__ import annotations

import numpy as np

GROUP = 32          # packed bytes per 128-element window
WINDOW = 128        # elements per window

# Lookup for codes 0..3. Code 3 is invalid; map to 0 as a fail-safe.
_CODE_TO_TERNARY = np.array([-1, 0, 1, 0], dtype=np.int8)


def _nbytes_i2s(n_elems: int) -> int:
    """Total I2_S blob length (packed bytes + 32-byte scale/pad trailer)."""
    return n_elems // 4 + 32


def dequantize_i2_s(blob: bytes, n_elems: int) -> np.ndarray:
    """Dequantize an I2_S `blob` of `n_elems` elements to a flat fp32 array."""
    if n_elems % 4 != 0:
        raise ValueError(f"I2_S n_elems {n_elems} must be divisible by 4")
    if n_elems % WINDOW != 0:
        raise ValueError(
            f"I2_S n_elems {n_elems} not divisible by window {WINDOW}")
    packed_bytes = n_elems // 4
    expected = _nbytes_i2s(n_elems)
    if len(blob) < expected:
        raise ValueError(
            f"I2_S blob {len(blob)} < expected {expected} for {n_elems} elems")

    packed = np.frombuffer(blob, dtype=np.uint8, count=packed_bytes)
    scale = float(
        np.frombuffer(blob, dtype=np.float32, count=1, offset=packed_bytes)[0])
    if not np.isfinite(scale):
        scale = 0.0    # fail-safe

    n_windows = n_elems // WINDOW
    win = packed.reshape(n_windows, GROUP)               # (n_windows, 32)
    codes = np.empty((n_windows, 4, GROUP), dtype=np.uint8)
    for g in range(4):
        codes[:, g, :] = (win >> (6 - 2 * g)) & 0b11
    ternary = _CODE_TO_TERNARY[codes.reshape(n_windows, WINDOW)]
    return (ternary.astype(np.float32) * scale).ravel()


def dequantize_tensor(blob: bytes, shape: tuple[int, ...]) -> np.ndarray:
    """Dequantize and reshape into PyTorch (out, in) convention.

    GGUF stores dims fast-first, so we reshape with REVERSED `shape` to get
    C-order indexing that matches the byte layout (a 2-D weight then reads as
    (out, in), like nn.Linear.weight)."""
    n = int(np.prod(shape))
    flat = dequantize_i2_s(blob, n)
    return flat.reshape(tuple(reversed(shape)))
