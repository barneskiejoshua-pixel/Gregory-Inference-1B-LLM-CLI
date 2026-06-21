"""Compile-on-first-use ctypes binding for the ternary matvec kernel.

Wraps gregory/kernels/ternary_matmul.c (int8 activation x 2-bit ternary weight)
and provides the weight packer it expects. The kernel is OPTIONAL: if no C
compiler (or no AVX2) is available, the model falls back to the pure-NumPy fp32
reference path. fp32 stays the correctness oracle; this is a verified-equivalent
acceleration (int8 activation quantization is the regime BitNet was trained in).

Public API:
    available() -> bool                 # kernel compiled and loadable?
    pack_ternary(W) -> (packed, scale)  # fp32 ternary (N,K) -> uint8 (N,K//4)
    matvec(packed, scale, x, K) -> Y    # Y = scale * (unpacked_W @ x), fp32
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "ternary_matmul.c"
_SO = _HERE / "ternary_matmul.so"
_lib: ctypes.CDLL | None = None


def _compile() -> Path:
    """Build the .so if missing or older than the source; return its path."""
    if _SO.exists() and _SO.stat().st_mtime >= _SRC.stat().st_mtime:
        return _SO
    # NB: NOT -ffast-math. The kernel's invariant is bit-identical AVX2==scalar
    # output (see AGENTS.md); -ffast-math licenses FP reassociation/FTZ that can
    # break that across toolchains. -fno-math-errno keeps the quantize loop's
    # copysignf auto-vectorizable without touching FP results.
    cmd = ["gcc", "-O3", "-mavx2", "-mfma", "-fopenmp", "-fno-math-errno",
           "-fPIC", "-shared", str(_SRC), "-o", str(_SO)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"kernel compile failed:\n{proc.stderr}")
    return _SO


def _get_lib() -> ctypes.CDLL:
    """Load (compiling if needed) the kernel and bind its signatures."""
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(str(_compile()))
        sig = [ctypes.c_int, ctypes.c_int,
               ctypes.POINTER(ctypes.c_uint8), ctypes.c_float,
               ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.gk_matvec.argtypes = sig
        lib.gk_matvec.restype = None
        lib.gk_matvec_scalar_pub.argtypes = sig
        lib.gk_matvec_scalar_pub.restype = None
        lib.gk_has_avx2.argtypes = []
        lib.gk_has_avx2.restype = ctypes.c_int
        lib.gk_has_omp.argtypes = []
        lib.gk_has_omp.restype = ctypes.c_int
        lib.gk_head_matvec.argtypes = [
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int8), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
        lib.gk_head_matvec.restype = None
        _lib = lib
    return _lib


def available() -> bool:
    """True if the kernel compiles and loads on this machine."""
    try:
        _get_lib()
        return True
    except (OSError, RuntimeError):
        return False


def has_avx2() -> bool:
    """True if the loaded kernel was built with the AVX2 path."""
    return bool(_get_lib().gk_has_avx2())


def pack_ternary(weight: np.ndarray) -> tuple[np.ndarray, float]:
    """Pack an fp32 ternary weight (N, K) into the kernel's 2-bit format.

    Returns (packed uint8 of shape (N, K//4), scale). The weight's nonzero
    magnitude is the per-tensor scale; values map to codes c = round(w/s)+1 in
    {0,1,2}, four per byte at bit shifts 6,4,2,0. Requires K divisible by 4."""
    w = np.ascontiguousarray(weight, dtype=np.float32)
    n_rows, k = w.shape
    if k % 4 != 0:
        raise ValueError(f"K={k} must be divisible by 4 to pack")
    scale = float(np.abs(w).max())
    div = scale if scale > 0.0 else 1.0
    codes = (np.rint(w / div).astype(np.int8) + 1).astype(np.uint8)
    codes = codes.reshape(n_rows, k // 4, 4)
    packed = ((codes[:, :, 0] << 6) | (codes[:, :, 1] << 4)
              | (codes[:, :, 2] << 2) | codes[:, :, 3]).astype(np.uint8)
    return np.ascontiguousarray(packed), scale


def matvec(packed: np.ndarray, scale: float, x: np.ndarray,
           k: int, out: np.ndarray | None = None) -> np.ndarray:
    """Return Y = scale * (unpacked_W @ x) as fp32 (length N).

    `packed` is (N, k//4) uint8 from pack_ternary; `x` is a length-k vector.
    Pass a reusable `out` buffer (C-contiguous float32, length N) to avoid a
    fresh allocation on every decode call -- the hot-path pool pattern."""
    lib = _get_lib()
    if packed.dtype != np.uint8 or not packed.flags["C_CONTIGUOUS"]:
        packed = np.ascontiguousarray(packed, dtype=np.uint8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_rows = packed.shape[0]
    if out is None:
        y = np.empty(n_rows, dtype=np.float32)
    elif (out.shape != (n_rows,) or out.dtype != np.float32
          or not out.flags["C_CONTIGUOUS"]):
        raise ValueError("out must be a C-contiguous float32 array of "
                         f"shape ({n_rows},)")
    else:
        y = out
    lib.gk_matvec(
        ctypes.c_int(n_rows), ctypes.c_int(k),
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_float(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return y


def matvec_scalar(packed: np.ndarray, scale: float, x: np.ndarray,
                  k: int) -> np.ndarray:
    """Same contract as matvec(), but forces the pure-scalar reference path.

    Exposed for the AVX2==scalar bit-identity test; not used on the hot path."""
    lib = _get_lib()
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_rows = packed.shape[0]
    y = np.empty(n_rows, dtype=np.float32)
    lib.gk_matvec_scalar_pub(
        ctypes.c_int(n_rows), ctypes.c_int(k),
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_float(scale),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return y


def pack_head(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row-quantize the LM head (vocab, embed) fp32 -> int8 + per-row scale.

    Returns (int8 of shape (N, K), row_scale fp32 of shape (N,)). Each row's
    absmax is its scale; a zero row gets scale 1.0 and quantizes to zeros."""
    w = np.ascontiguousarray(emb, dtype=np.float32)
    absmax = np.abs(w).max(axis=1)
    scale = np.where(absmax > 0.0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.rint(w / scale[:, None]).clip(-127, 127).astype(np.int8)
    return np.ascontiguousarray(q), np.ascontiguousarray(scale)


def head_matvec(int8_w: np.ndarray, row_scale: np.ndarray, x: np.ndarray,
                k: int, out: np.ndarray | None = None) -> np.ndarray:
    """Return logits = (int8_w @ x) * row_scale as fp32 (length N).

    `int8_w` is (N, k) from pack_head; `row_scale` is (N,); `x` is length k.
    Pass a reusable `out` buffer (C-contiguous float32, length N) to skip the
    per-token vocab-sized allocation. `int8_w`/`row_scale` are guaranteed
    contiguous and typed at pack time, so these wraps are no-op views."""
    lib = _get_lib()
    int8_w = np.ascontiguousarray(int8_w, dtype=np.int8)
    row_scale = np.ascontiguousarray(row_scale, dtype=np.float32)
    x = np.ascontiguousarray(x, dtype=np.float32)
    n_rows = int8_w.shape[0]
    if out is None:
        y = np.empty(n_rows, dtype=np.float32)
    elif (out.shape != (n_rows,) or out.dtype != np.float32
          or not out.flags["C_CONTIGUOUS"]):
        raise ValueError("out must be a C-contiguous float32 array of "
                         f"shape ({n_rows},)")
    else:
        y = out
    lib.gk_head_matvec(
        ctypes.c_int(n_rows), ctypes.c_int(k),
        int8_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        row_scale.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
    return y
