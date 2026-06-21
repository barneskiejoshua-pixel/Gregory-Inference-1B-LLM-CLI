"""Gregory transformer forward pass -- pure-NumPy reference path.

Architecture (ternary-weight decoder-only transformer, BitNet b1.58 family):

  per block:
    x -> attn_norm -> Q/K/V -> RoPE -> causal GQA softmax(QK^T / sqrt(d)) V
       -> attn_sub_norm -> attn_output                          (+ residual)
    x -> ffn_norm -> gate, up -> relu(gate)^2 * up -> ffn_sub_norm
       -> ffn_down                                              (+ residual)
  output: rmsnorm(output_norm) -> tied LM head (token_embd) -> logits

This is the readable REFERENCE path: ternary weights are dequantized to fp32
once (cached as .npy) and every matmul is a plain NumPy fp32 matmul. There is
no packed/int8 fast kernel here -- speed is a non-goal for the skeleton (see
README). Correctness and clarity come first.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from . import cache, gguf, quant

# The seven ternary (I2_S) matmul weights packed for the fast kernel. The norm
# weights are F32 (used in rmsnorm, not projected) and token_embd is the head.
_FAST_KEYS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd")
# Maps the short handle to the GGUF tensor suffix (for the packed cache).
_KEY_TENSOR = {
    "wq": "attn_q", "wk": "attn_k", "wv": "attn_v", "wo": "attn_output",
    "wg": "ffn_gate", "wu": "ffn_up", "wd": "ffn_down",
}


class Gregory:
    """A loaded ternary-weight transformer with a single-sequence forward."""

    def __init__(self, g: gguf.GGUF, verbose: bool = False) -> None:
        """Read architecture metadata from `g`, load/cache fp32 weights, and
        precompute the RoPE cos/sin tables."""
        arch = g.get("general.architecture")
        p = arch + "."
        self.n_layers = int(g.get(p + "block_count"))
        self.n_heads = int(g.get(p + "attention.head_count"))
        self.n_kv_heads = int(g.get(p + "attention.head_count_kv"))
        self.embed_dim = int(g.get(p + "embedding_length"))
        self.ffn_dim = int(g.get(p + "feed_forward_length"))
        self.rope_dim = int(g.get(p + "rope.dimension_count"))
        self.rope_base = float(g.get(p + "rope.freq_base"))
        self.norm_eps = float(g.get(p + "attention.layer_norm_rms_epsilon"))
        self.vocab_size = int(g.get(p + "vocab_size"))
        self.head_dim = self.embed_dim // self.n_heads
        # GQA requires n_heads to be a multiple of n_kv_heads; otherwise the
        # grouping below would silently truncate query heads. Make it loud.
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads {self.n_heads} not divisible by n_kv_heads "
                f"{self.n_kv_heads} -- GQA grouping would truncate heads")
        self.kv_group = self.n_heads // self.n_kv_heads
        if self.head_dim != self.rope_dim:
            raise ValueError(
                f"head_dim {self.head_dim} != rope_dim {self.rope_dim}")
        self.max_seq = 4096

        self.W: dict[str, np.ndarray] = self._load_weights(g, verbose)
        self._bind_layer_handles()
        self._build_rope_tables()

        # Optional fast path: packed 2-bit ternary kernel for decode (seq==1).
        # Default ON when buildable; opt out with GREGORY_FAST=0. fp32 stays the
        # reference/oracle and is always used for prefill (seq>1).
        self._packed: dict[tuple[int, str], tuple] = {}
        self._head_int8: np.ndarray | None = None
        self._head_scale: np.ndarray | None = None
        # Reusable decode output buffers (the hot-path pool). One per
        # (layer, key) so simultaneously-live projections never alias -- e.g.
        # the FFN holds `gate` and `up` at once. Filled in _init_fast.
        self._proj_out: dict[tuple[int, str], np.ndarray] = {}
        self._logit_buf: np.ndarray | None = None
        self._fast = os.environ.get("GREGORY_FAST", "1") not in ("0", "false")
        if self._fast:
            self._init_fast(g, verbose)

    # ---------------- load ----------------

    def _load_weights(self, g: gguf.GGUF,
                      verbose: bool) -> dict[str, np.ndarray]:
        """Return name -> fp32 array, from the .npy cache or by dequantizing."""
        cdir = cache.cache_dir_for(g.path)
        n_total = len(g.tensors)
        if cache.is_valid(cdir, n_total):
            if verbose:
                print(f"  loaded {n_total} tensors from cache: {cdir}",
                      flush=True)
            return cache.load_mmap(cdir)

        if verbose:
            print(f"  building weight cache at {cdir}", flush=True)
        weights: dict[str, np.ndarray] = {}
        for i, (name, t) in enumerate(g.tensors.items(), 1):
            blob = g.tensor_bytes(name)
            shape_np = tuple(reversed(t.shape))
            if t.dtype == gguf.GGMLType.I2_S:
                arr = quant.dequantize_tensor(blob, tuple(t.shape))
            elif t.dtype == gguf.GGMLType.F32:
                arr = np.frombuffer(blob, np.float32).reshape(shape_np).copy()
            elif t.dtype == gguf.GGMLType.F16:
                arr = (np.frombuffer(blob, np.float16)
                       .reshape(shape_np).astype(np.float32))
            else:
                raise NotImplementedError(
                    f"unsupported dtype {t.dtype.name} for {name}")
            weights[name] = arr
            if verbose and (i % 50 == 0 or i == n_total):
                print(f"  dequant {i}/{n_total}", flush=True)
        cache.save(cdir, weights)
        return weights

    def _bind_layer_handles(self) -> None:
        """Cache per-layer weight references to skip dict lookups in forward."""
        self._L = []
        for il in range(self.n_layers):
            p = f"blk.{il}."
            self._L.append({
                "an": self.W[p + "attn_norm.weight"],
                "wq": self.W[p + "attn_q.weight"],
                "wk": self.W[p + "attn_k.weight"],
                "wv": self.W[p + "attn_v.weight"],
                "asn": self.W[p + "attn_sub_norm.weight"],
                "wo": self.W[p + "attn_output.weight"],
                "fn": self.W[p + "ffn_norm.weight"],
                "wg": self.W[p + "ffn_gate.weight"],
                "wu": self.W[p + "ffn_up.weight"],
                "fsn": self.W[p + "ffn_sub_norm.weight"],
                "wd": self.W[p + "ffn_down.weight"],
            })
        self._out_norm = self.W["output_norm.weight"]
        self._tok_embd = self.W["token_embd.weight"]

    def _build_rope_tables(self) -> None:
        """Precompute RoPE cos/sin tables of shape (max_seq, rope_dim/2)."""
        inv = 1.0 / (self.rope_base ** (
            np.arange(0, self.rope_dim, 2, dtype=np.float32) / self.rope_dim))
        pos = np.arange(self.max_seq, dtype=np.float32)
        ang = np.outer(pos, inv)
        self.cos = np.cos(ang).astype(np.float32)
        self.sin = np.sin(ang).astype(np.float32)

    # ---------------- fast packed kernel ----------------

    def _init_fast(self, g: gguf.GGUF, verbose: bool) -> None:
        """Build (or load from cache) packed 2-bit weights for the kernel.

        Degrades gracefully: if the kernel will not compile/load, the fast flag
        is cleared and the model runs the pure-NumPy fp32 path."""
        try:
            from . import kernels
        except ImportError:
            self._fast = False
            return
        if not kernels.available():
            if verbose:
                print("  fast kernel unavailable; using fp32 path", flush=True)
            self._fast = False
            return

        packed_dir = cache.cache_dir_for(g.path) / "_packed_ternary"
        manifest = packed_dir / "manifest.json"
        if not self._load_packed_cache(packed_dir, manifest, verbose):
            self._build_packed(kernels, packed_dir, manifest, verbose)
        # One persistent output buffer per projection (arr.shape[0] == out dim),
        # reused across every decode token instead of np.empty per call.
        self._proj_out = {
            key: np.empty(int(arr.shape[0]), dtype=np.float32)
            for key, (arr, _scale, _k) in self._packed.items()}
        # The tied LM head (token_embd) is the single largest tensor; an int8
        # row-quantized copy streams 4x fewer bytes per decode token.
        if os.environ.get("GREGORY_FAST_HEAD", "1") not in ("0", "false"):
            self._init_head(kernels, packed_dir, verbose)
            if self._head_int8 is not None:
                self._logit_buf = np.empty(int(self._head_int8.shape[0]),
                                           dtype=np.float32)

    def _build_packed(self, kernels, packed_dir: Path, manifest: Path,
                      verbose: bool) -> None:
        """Pack the seven ternary weights of every layer and cache them."""
        if verbose:
            print(f"  packing ternary weights -> {packed_dir}", flush=True)
        packed_dir.mkdir(parents=True, exist_ok=True)
        scales: dict[str, float] = {}
        for layer in range(self.n_layers):
            for key in _FAST_KEYS:
                w = self._L[layer][key]
                arr, scale = kernels.pack_ternary(np.asarray(w))
                k = int(w.shape[1])
                np.save(packed_dir / f"l{layer}_{key}.npy", arr)
                scales[f"l{layer}_{key}"] = scale
                self._packed[(layer, key)] = (arr, scale, k)
        manifest.write_text(json.dumps(
            {"n_layers": self.n_layers, "scales": scales}))
        if verbose:
            print("  packing done", flush=True)

    def _init_head(self, kernels, packed_dir: Path, verbose: bool) -> None:
        """Load or build the int8 row-quantized LM head into _head_int8."""
        hi = packed_dir / "head_int8.npy"
        hs = packed_dir / "head_scale.npy"
        try:
            if hi.is_file() and hs.is_file():
                self._head_int8 = np.load(hi, mmap_mode="r")
                self._head_scale = np.load(hs, mmap_mode="r")
                if verbose:
                    print(f"  int8 head loaded from cache: {hi}", flush=True)
                return
        except (OSError, ValueError):
            self._head_int8 = self._head_scale = None
        if verbose:
            print("  building int8 head", flush=True)
        packed_dir.mkdir(parents=True, exist_ok=True)
        q, scale = kernels.pack_head(np.asarray(self._tok_embd))
        np.save(hi, q)
        np.save(hs, scale)
        self._head_int8 = q
        self._head_scale = scale

    def _load_packed_cache(self, packed_dir: Path, manifest: Path,
                           verbose: bool) -> bool:
        """Load packed weights from `packed_dir`; return True on success."""
        if not manifest.is_file():
            return False
        try:
            meta = json.loads(manifest.read_text())
            if meta.get("n_layers") != self.n_layers:
                return False
            scales = meta["scales"]
            for layer in range(self.n_layers):
                for key in _FAST_KEYS:
                    arr = np.load(packed_dir / f"l{layer}_{key}.npy",
                                  mmap_mode="r")
                    k = int(self._L[layer][key].shape[1])
                    self._packed[(layer, key)] = (
                        arr, float(scales[f"l{layer}_{key}"]), k)
        except (OSError, ValueError, KeyError):
            self._packed.clear()
            return False
        if verbose:
            print(f"  packed weights loaded from cache: {packed_dir}",
                  flush=True)
        return True

    # ---------------- kv cache ----------------

    def init_kv_cache(self, max_seq: int | None = None) -> dict:
        """Pre-allocate fp32 K/V slots for `max_seq` tokens (default: ctx).

        Each forward step writes into the next slice, not a fresh array."""
        m = max_seq or self.max_seq
        shape = (self.n_layers, m, self.n_kv_heads, self.head_dim)
        return {"_k": np.empty(shape, dtype=np.float32),
                "_v": np.empty(shape, dtype=np.float32),
                "_len": 0, "_max": m}

    # ---------------- elemental ops ----------------

    def rmsnorm(self, x: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """Root-mean-square layer norm, scaled by `weight`."""
        var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
        normed = x / np.sqrt(var + self.norm_eps)
        return (normed * weight).astype(np.float32)

    @staticmethod
    def relu_sqr(x: np.ndarray) -> np.ndarray:
        """Squared-ReLU max(0, x)^2 -- BitNet b1.58's FFN activation
        (llama.cpp LLM_FFN_RELU_SQR), not SiLU."""
        clipped = np.maximum(x, 0.0)
        return clipped * clipped

    @staticmethod
    def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically-stable softmax along `axis`."""
        x = x - x.max(axis=axis, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=axis, keepdims=True)

    def apply_rope(self, x: np.ndarray, cos: np.ndarray,
                   sin: np.ndarray) -> np.ndarray:
        """NeoX-style RoPE (rope_type=2, as LLaMA-3 / BitNet use): split the
        head dim into halves and rotate them as a pair.

        x: (seq, n_heads, head_dim). `cos`/`sin` are the gathered tables of
        shape (seq, 1, rope_dim/2) -- built once per forward in `forward` and
        shared across all layers (the positions are layer-independent)."""
        h = self.rope_dim // 2
        x1 = x[..., :h]
        x2 = x[..., h:]
        out = np.empty_like(x)
        out[..., :h] = x1 * cos - x2 * sin
        out[..., h:] = x1 * sin + x2 * cos
        return out

    def _proj(self, x: np.ndarray, layer: int, key: str) -> np.ndarray:
        """Project x through layer weight `key` (a (out, in) matrix).

        Decode (a single row) routes through the packed 2-bit kernel when the
        fast path is active; prefill (multiple rows) and the fp32 fallback both
        use BLAS x @ W.T -- the correctness oracle."""
        if self._fast and x.shape[0] == 1:
            pk = self._packed.get((layer, key))
            if pk is not None:
                from . import kernels
                arr, scale, k = pk
                out = self._proj_out.get((layer, key))
                y = kernels.matvec(arr, scale, x[0], k, out=out)
                return y[None, :]
        return x @ self._L[layer][key].T

    # ---------------- layer ops ----------------

    def attention(self, x: np.ndarray, layer: int, rope: tuple,
                  mask: np.ndarray | None, kv_cache: dict | None) -> np.ndarray:
        """Causal grouped-query attention for one block.

        `rope` is the (cos, sin) pair and `mask` the additive causal mask, both
        precomputed once per forward and shared across layers."""
        seq = x.shape[0]
        lw = self._L[layer]
        x_norm = self.rmsnorm(x, lw["an"])

        q = self._proj(x_norm, layer, "wq")
        k = self._proj(x_norm, layer, "wk")
        v = self._proj(x_norm, layer, "wv")

        q = q.reshape(seq, self.n_heads, self.head_dim)
        k = k.reshape(seq, self.n_kv_heads, self.head_dim)
        v = v.reshape(seq, self.n_kv_heads, self.head_dim)

        cos, sin = rope
        q = self.apply_rope(q, cos, sin)
        k = self.apply_rope(k, cos, sin)

        if kv_cache is not None:
            start = kv_cache["_step_start"]
            total = start + seq
            kv_cache["_k"][layer, start:total] = k
            kv_cache["_v"][layer, start:total] = v
            k = kv_cache["_k"][layer, :total]
            v = kv_cache["_v"][layer, :total]
        else:
            total = k.shape[0]

        g, nkv, hd = self.kv_group, self.n_kv_heads, self.head_dim
        # GQA by broadcast: each KV head feeds a contiguous block of `g` query
        # heads. Reshape the query axis to (nkv, g) and broadcast K/V over the
        # group axis -- equivalent to np.repeat(k, g) but without the copy.
        q_g = q.reshape(seq, nkv, g, hd).transpose(1, 2, 0, 3)
        k_g = k.transpose(1, 2, 0)
        v_g = v.transpose(1, 0, 2)
        scores = q_g @ k_g[:, None] / np.sqrt(hd)

        if mask is not None:
            scores = scores + mask

        attn = self.softmax(scores)
        out = attn @ v_g[:, None]
        out = out.transpose(2, 0, 1, 3).reshape(seq, self.n_heads * hd)

        out = self.rmsnorm(out, lw["asn"])     # BitNet sub-norm before output
        return self._proj(out, layer, "wo")

    def ffn(self, x: np.ndarray, layer: int) -> np.ndarray:
        """Gated feed-forward block with squared-ReLU activation."""
        lw = self._L[layer]
        x_norm = self.rmsnorm(x, lw["fn"])
        gate = self._proj(x_norm, layer, "wg")
        up = self._proj(x_norm, layer, "wu")
        hidden = self.relu_sqr(gate) * up
        hidden = self.rmsnorm(hidden, lw["fsn"])
        return self._proj(hidden, layer, "wd")

    def block(self, x: np.ndarray, layer: int, rope: tuple,
              mask: np.ndarray | None, kv_cache: dict | None) -> np.ndarray:
        """One transformer block: attention + FFN, each with a residual."""
        x = x + self.attention(x, layer, rope, mask, kv_cache)
        x = x + self.ffn(x, layer)
        return x

    # ---------------- forward ----------------

    def forward(self, token_ids: list[int], kv_cache: dict | None = None,
                last_only: bool = False) -> np.ndarray:
        """Run the full stack over `token_ids`; return logits (rows, vocab).

        With a `kv_cache`, positions resume after the cached prefix. With
        `last_only=True`, only the final position is projected through the LM
        head (rows == 1) -- the right choice for generation prefill."""
        ids = np.asarray(token_ids, dtype=np.int64)
        seq = len(ids)
        emb = self._tok_embd                       # (vocab, embed)
        x = emb[ids].astype(np.float32)            # (seq, embed)

        if kv_cache is None:
            offset = 0
        else:
            offset = kv_cache["_len"]
            kv_cache["_step_start"] = offset
            if offset + seq > kv_cache["_max"]:
                raise RuntimeError(
                    f"KV cache full: {offset}+{seq} > {kv_cache['_max']}")
        positions = np.arange(offset, offset + seq, dtype=np.int64)

        # Gather the RoPE tables and build the causal mask ONCE per forward:
        # both are layer-independent (same positions, same shape), so computing
        # them inside each block would repeat identical work n_layers times.
        cos = self.cos[positions][:, None, :]
        sin = self.sin[positions][:, None, :]
        rope = (cos, sin)
        mask = None
        if seq > 1:
            total = offset + seq
            i = np.arange(seq, dtype=np.int64)[:, None]
            j = np.arange(total, dtype=np.int64)[None, :]
            mask = np.where(j <= (total - seq) + i,
                            np.float32(0.0), np.float32(-np.inf))

        for layer in range(self.n_layers):
            x = self.block(x, layer, rope, mask, kv_cache)

        if kv_cache is not None:
            kv_cache["_len"] = offset + seq

        x = self.rmsnorm(x, self._out_norm)
        if last_only and seq > 1:
            x = x[-1:]
        # Decode (a single row) streams the head at int8 (4x less bandwidth on
        # the largest tensor); prefill/scoring with multiple rows stay fp32.
        if self._fast and self._head_int8 is not None and x.shape[0] == 1:
            from . import kernels
            logits = kernels.head_matvec(self._head_int8, self._head_scale,
                                         x[0], self.embed_dim,
                                         out=self._logit_buf)
            return logits[None, :]
        return x @ emb.T                           # tied LM head (fp32 oracle)
