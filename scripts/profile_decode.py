"""Profile a real decode loop: split C-kernel time from NumPy-glue time.

Wraps kernels.matvec / kernels.head_matvec with a wall-clock accumulator, runs
a warmed decode loop, and reports kernel vs glue vs sampler. Also runs cProfile
over the glue so the elementwise ops (rmsnorm/rope/softmax/attention) are ranked.
"""

from __future__ import annotations

import cProfile
import io
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gregory import kernels  # noqa: E402
from gregory import gguf as gguf_mod  # noqa: E402
from gregory.model import Gregory  # noqa: E402

MODEL = "/home/joshua/BitNet/models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf"
N_WARMUP = 8
N_DECODE = 48

# --- wrap the kernel entry points to accumulate their wall time ----------
_acc = {"matvec_s": 0.0, "matvec_n": 0, "head_s": 0.0, "head_n": 0}
_real_matvec = kernels.matvec
_real_head = kernels.head_matvec


def _timed_matvec(*a, **k):
    t = time.perf_counter()
    r = _real_matvec(*a, **k)
    _acc["matvec_s"] += time.perf_counter() - t
    _acc["matvec_n"] += 1
    return r


def _timed_head(*a, **k):
    t = time.perf_counter()
    r = _real_head(*a, **k)
    _acc["head_s"] += time.perf_counter() - t
    _acc["head_n"] += 1
    return r


kernels.matvec = _timed_matvec
kernels.head_matvec = _timed_head


def main() -> None:
    print(f"loading {MODEL} ...", flush=True)
    g = gguf_mod.load(MODEL)
    m = Gregory(g, verbose=True)
    print(f"fast path: {m._fast}  int8 head: {m._head_int8 is not None}",
          flush=True)
    print(f"n_layers={m.n_layers} embed={m.embed_dim} ffn={m.ffn_dim} "
          f"vocab={m.vocab_size}", flush=True)

    kv = m.init_kv_cache()
    # Prefill a short prompt (fp32 path) so decode runs at seq==1.
    prompt = list(range(1, 17))
    m.forward(prompt, kv, last_only=True)

    tok = 100  # arbitrary in-vocab token to feed the decode loop

    # Warmup (touch the packed weights / fill caches; not timed).
    for _ in range(N_WARMUP):
        m.forward([tok], kv)

    _acc["matvec_s"] = _acc["head_s"] = 0.0
    _acc["matvec_n"] = _acc["head_n"] = 0

    t0 = time.perf_counter()
    for _ in range(N_DECODE):
        m.forward([tok], kv)
    total = time.perf_counter() - t0

    per_tok = total / N_DECODE * 1e3
    kern = _acc["matvec_s"] + _acc["head_s"]
    glue = total - kern
    print("\n================ DECODE PROFILE ================")
    print(f"tokens timed     : {N_DECODE}")
    print(f"throughput       : {N_DECODE / total:6.2f} tok/s "
          f"({per_tok:6.2f} ms/tok)")
    print(f"total decode     : {total * 1e3:8.1f} ms")
    print(f"  C kernel (mv+hd): {kern * 1e3:8.1f} ms  "
          f"({kern / total * 100:5.1f}%)")
    print(f"    proj matvec   : {_acc['matvec_s'] * 1e3:8.1f} ms  "
          f"({_acc['matvec_n']} calls, "
          f"{_acc['matvec_s'] / _acc['matvec_n'] * 1e6:.1f} us/call)")
    print(f"    head matvec   : {_acc['head_s'] * 1e3:8.1f} ms  "
          f"({_acc['head_n']} calls)")
    print(f"  NumPy glue+py   : {glue * 1e3:8.1f} ms  "
          f"({glue / total * 100:5.1f}%)")
    print("================================================\n")

    # --- cProfile the glue to rank the elementwise ops ------------------
    print("Top glue functions by cumulative self-time (cProfile, "
          f"{N_DECODE} tokens):\n")
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(N_DECODE):
        m.forward([tok], kv)
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
    ps.print_stats(18)
    # Trim the cProfile header noise to the table.
    out = s.getvalue()
    for line in out.splitlines():
        if "ncalls" in line or "/" in line.split()[:1] or "function" in line:
            pass
    print(out)


if __name__ == "__main__":
    main()
