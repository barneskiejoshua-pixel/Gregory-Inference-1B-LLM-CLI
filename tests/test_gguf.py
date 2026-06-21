"""Unit tests for the GGUF reader (synthetic in-memory file, model-free)."""

from __future__ import annotations

import struct

from gregory import gguf
from gregory.gguf import GGMLType, TensorInfo, bytes_for_tensor


def _str(s: str) -> bytes:
    """Encode a u64-length-prefixed GGUF string."""
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _build_minimal_gguf(tmp_path):
    """Write a tiny valid GGUF (one u32 KV, one F32 tensor) and return path."""
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", 1)            # tensor_count
    body += struct.pack("<Q", 1)            # kv_count
    # one KV: "answer" = u32 42
    body += _str("answer") + struct.pack("<I", int(gguf.GGUFValueType.UINT32))
    body += struct.pack("<I", 42)
    # one tensor info: name, n_dims=1, dims=[4], type=F32, offset=0
    body += _str("t") + struct.pack("<I", 1) + struct.pack("<Q", 4)
    body += struct.pack("<I", int(GGMLType.F32)) + struct.pack("<Q", 0)
    pad = (32 - (len(body) % 32)) % 32
    body += b"\x00" * pad
    body += struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    path = tmp_path / "mini.gguf"
    path.write_bytes(body)
    return path


def test_load_parses_kv_and_tensor(tmp_path):
    """A minimal GGUF round-trips its KV and tensor table."""
    g = gguf.load(_build_minimal_gguf(tmp_path))
    assert g.version == 3
    assert g.get("answer") == 42
    assert "t" in g.tensors
    assert g.tensor_bytes("t") == struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
    g.close()


def test_bad_magic_raises(tmp_path):
    """A non-GGUF file raises ValueError."""
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOPE" + b"\x00" * 32)
    try:
        gguf.load(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError on bad magic")


def test_bytes_for_i2s():
    """I2_S sizing is n_elems/4 + 32-byte trailer."""
    t = TensorInfo("w", 2, (256, 4), GGMLType.I2_S, 0)
    assert bytes_for_tensor(t) == (256 * 4) // 4 + 32
