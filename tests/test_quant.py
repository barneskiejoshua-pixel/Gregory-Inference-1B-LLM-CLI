"""Unit tests for I2_S dequantization (model-free, synthetic blobs)."""

from __future__ import annotations

import struct

import numpy as np

from gregory import quant


def _make_blob(codes: list[int], scale: float) -> bytes:
    """Pack a list of 2-bit codes (len multiple of 128) into an I2_S blob."""
    n = len(codes)
    assert n % 128 == 0
    packed = bytearray(n // 4)
    n_windows = n // 128
    for w in range(n_windows):
        for gp in range(32):
            byte = 0
            for g in range(4):
                code = codes[w * 128 + g * 32 + gp]
                byte |= (code & 0b11) << (6 - 2 * g)
            packed[w * 32 + gp] = byte
    return bytes(packed) + struct.pack("<f", scale) + b"\x00" * 28


def test_roundtrip_codes_map_to_ternary():
    """Codes 0/1/2/3 dequantize to -1/0/+1/0 times the scale."""
    codes = ([0, 1, 2, 3] * 32)            # 128 elements, one window
    scale = 2.5
    out = quant.dequantize_i2_s(_make_blob(codes, scale), len(codes))
    expected = np.array([-1, 0, 1, 0], dtype=np.float32) * scale
    assert np.allclose(out.reshape(-1, 4), expected)


def test_shape_reversal():
    """dequantize_tensor reshapes with reversed dims (GGUF fast-first)."""
    codes = [2] * 256                       # all +1
    blob = _make_blob(codes, 1.0)
    arr = quant.dequantize_tensor(blob, (4, 64))
    assert arr.shape == (64, 4)
    assert np.allclose(arr, 1.0)


def test_rejects_bad_length():
    """A blob shorter than the I2_S layout requires raises ValueError."""
    try:
        quant.dequantize_i2_s(b"\x00" * 4, 128)
    except ValueError:
        return
    raise AssertionError("expected ValueError on short blob")
