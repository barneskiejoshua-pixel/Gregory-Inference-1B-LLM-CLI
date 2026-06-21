"""GGUF v3 binary-format reader (read-only, lazy, clean-room).

Format spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

Layout:
    magic 'GGUF' | u32 version | u64 tensor_count | u64 kv_count
    | KV entries   (key:string, type:u32, value:typed)
    | tensor info  (name:string, n_dims:u32, dims[u64], type:u32, offset:u64)
    | alignment padding
    | raw tensor blob region

Tensor data is left in the mmap; dequantization happens in gregory.quant only
when a tensor is actually accessed.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path


class GGMLType(IntEnum):
    """ggml tensor element types. Only the subset Gregory reads is dequantized;
    unknown codes are allowed through and fail only on a blob read."""

    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q8_0 = 8
    Q4_K = 12
    Q6_K = 14
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30
    # BitNet b1.58 ternary matmul tensors are vendored as type 36 in the
    # b1.58-2B-4T i2_s release (see BitNet/include/ggml-bitnet.h).
    I2_S = 36

    @classmethod
    def _missing_(cls, value):
        """Allow unknown type codes; fail loudly only on dequantization."""
        obj = int.__new__(cls, value)
        obj._name_ = f"UNKNOWN_{value}"
        obj._value_ = value
        return obj


class GGUFValueType(IntEnum):
    """Typed-value tags used in the GGUF key/value metadata block."""

    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


@dataclass
class TensorInfo:
    """One entry from the GGUF tensor table."""

    name: str
    n_dims: int
    shape: tuple[int, ...]
    dtype: GGMLType
    offset: int        # byte offset within the data region
    nbytes: int = 0    # filled in once the layout is known


@dataclass
class GGUF:
    """A parsed GGUF file: metadata, tensor table, and the live mmap."""

    path: Path
    version: int
    kv: dict[str, object] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    data_start: int = 0
    mm: mmap.mmap | None = None

    def get(self, key: str, default=None):
        """Return metadata value `key`, or `default` if absent."""
        return self.kv.get(key, default)

    def tensor_bytes(self, name: str) -> bytes:
        """Return the raw (still-quantized) blob for tensor `name`."""
        t = self.tensors[name]
        start = self.data_start + t.offset
        return bytes(self.mm[start:start + t.nbytes])

    def close(self) -> None:
        """Release the mmap. Safe to call more than once."""
        if self.mm is not None:
            self.mm.close()
            self.mm = None


class _Reader:
    """Little-endian cursor over an mmap, one primitive per method."""

    def __init__(self, mm: mmap.mmap) -> None:
        self.mm = mm
        self.pos = 0

    def read(self, n: int) -> bytes:
        """Consume and return the next `n` bytes."""
        b = self.mm[self.pos:self.pos + n]
        self.pos += n
        return b

    def u32(self) -> int:
        """Read an unsigned 32-bit int."""
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        """Read a signed 32-bit int."""
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        """Read an unsigned 64-bit int."""
        return struct.unpack("<Q", self.read(8))[0]

    def i64(self) -> int:
        """Read a signed 64-bit int."""
        return struct.unpack("<q", self.read(8))[0]

    def u16(self) -> int:
        """Read an unsigned 16-bit int."""
        return struct.unpack("<H", self.read(2))[0]

    def i16(self) -> int:
        """Read a signed 16-bit int."""
        return struct.unpack("<h", self.read(2))[0]

    def u8(self) -> int:
        """Read an unsigned byte."""
        return self.read(1)[0]

    def i8(self) -> int:
        """Read a signed byte."""
        return struct.unpack("<b", self.read(1))[0]

    def f32(self) -> float:
        """Read a 32-bit float."""
        return struct.unpack("<f", self.read(4))[0]

    def f64(self) -> float:
        """Read a 64-bit float."""
        return struct.unpack("<d", self.read(8))[0]

    def boolean(self) -> bool:
        """Read a one-byte boolean."""
        return bool(self.u8())

    def string(self) -> str:
        """Read a u64-length-prefixed UTF-8 string."""
        n = self.u64()
        return self.read(n).decode("utf-8", errors="replace")


_MAX_VALUE_DEPTH = 8


def _read_value(r: _Reader, t: int, depth: int = 0):
    """Read one typed metadata value. Depth-capped: GGUF is untrusted input, so
    nested arrays raise past the cap instead of overflowing the stack."""
    if depth > _MAX_VALUE_DEPTH:
        raise ValueError(f"GGUF value nesting exceeds {_MAX_VALUE_DEPTH}")
    simple = {
        GGUFValueType.UINT8: r.u8,
        GGUFValueType.INT8: r.i8,
        GGUFValueType.UINT16: r.u16,
        GGUFValueType.INT16: r.i16,
        GGUFValueType.UINT32: r.u32,
        GGUFValueType.INT32: r.i32,
        GGUFValueType.FLOAT32: r.f32,
        GGUFValueType.BOOL: r.boolean,
        GGUFValueType.STRING: r.string,
        GGUFValueType.UINT64: r.u64,
        GGUFValueType.INT64: r.i64,
        GGUFValueType.FLOAT64: r.f64,
    }
    fn = simple.get(t)
    if fn is not None:
        return fn()
    if t == GGUFValueType.ARRAY:
        elem_t = r.u32()
        n = r.u64()
        return [_read_value(r, elem_t, depth + 1) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type: {t}")


# Bytes-per-element for fixed-width types (exact).
_BPE = {
    GGMLType.F32: 4, GGMLType.F16: 2, GGMLType.F64: 8, GGMLType.BF16: 2,
    GGMLType.I8: 1, GGMLType.I16: 2, GGMLType.I32: 4, GGMLType.I64: 8,
}


def bytes_for_tensor(t: TensorInfo) -> int:
    """Byte length of tensor `t`'s raw blob.

    Exact for fixed-width and the block-quantized types Gregory recognizes;
    returns 0 for unmodeled types so the table still lists them (a blob read
    is what fails, not the parse)."""
    n_elems = 1
    for d in t.shape:
        n_elems *= d
    if t.dtype in _BPE:
        return n_elems * _BPE[t.dtype]
    if t.dtype == GGMLType.Q8_0:
        return ((n_elems + 31) // 32) * (32 + 2)
    if t.dtype == GGMLType.Q4_0:
        return ((n_elems + 31) // 32) * (16 + 2)
    if t.dtype == GGMLType.Q4_K:
        return ((n_elems + 255) // 256) * 144
    if t.dtype == GGMLType.Q6_K:
        return ((n_elems + 255) // 256) * 210
    if t.dtype == GGMLType.I2_S:
        # I2_S: n_elems/4 packed bytes + 32-byte trailer (f32 scale + pad).
        return n_elems // 4 + 32
    return 0


def load(path: str | Path) -> GGUF:
    """Parse the GGUF at `path` and return a `GGUF` with a live read-only mmap.

    Raises ValueError on a bad magic or unsupported container version."""
    path = Path(path)
    fh = open(path, "rb")
    mm = mmap.mmap(fh.fileno(), 0, prot=mmap.PROT_READ)
    r = _Reader(mm)
    magic = r.read(4)
    if magic != b"GGUF":
        mm.close()
        fh.close()
        raise ValueError(f"not a GGUF file: magic={magic!r}")
    version = r.u32()
    if version not in (1, 2, 3):
        raise ValueError(f"unsupported GGUF version: {version}")
    tensor_count = r.u64()
    kv_count = r.u64()

    kv: dict[str, object] = {}
    for _ in range(kv_count):
        key = r.string()
        v_type = r.u32()
        kv[key] = _read_value(r, v_type)

    tensors: dict[str, TensorInfo] = {}
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        shape = tuple(r.u64() for _ in range(n_dims))
        dtype = GGMLType(r.u32())
        offset = r.u64()
        tensors[name] = TensorInfo(name=name, n_dims=n_dims, shape=shape,
                                   dtype=dtype, offset=offset)

    alignment = int(kv.get("general.alignment", 32))
    pad = (alignment - (r.pos % alignment)) % alignment
    data_start = r.pos + pad

    for t in tensors.values():
        t.nbytes = bytes_for_tensor(t)

    return GGUF(path=path, version=version, kv=kv, tensors=tensors,
                data_start=data_start, mm=mm)
