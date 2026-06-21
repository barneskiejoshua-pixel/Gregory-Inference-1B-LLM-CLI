"""Cap the BLAS / OpenMP thread pool at the physical core count.

Gregory's decode is memory-bandwidth bound: it streams ~10 GB of fp32 weights
per token. On an SMT (hyperthreaded) CPU, letting OpenBLAS spin up one thread
per *logical* core adds fork/join and cache contention that *slows* a
bandwidth-bound matvec. Measured on a 4-core / 8-thread i7: 8 threads ran at
1.43 tok/s versus 1.80 tok/s at 4 threads. So we pin the pool to physical cores.

These env vars are read by the BLAS / OpenMP runtime when it first loads (the
first `import numpy`), so `configure()` must run BEFORE numpy is imported --
hence it is called from gregory/__init__.py, which imports no numpy itself. An
explicit `GREGORY_THREADS` overrides the auto choice; a thread var the user has
already set is never clobbered.
"""

from __future__ import annotations

import os

_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def _cpuinfo_physical() -> int:
    """Count unique (physical id, core id) pairs in /proc/cpuinfo, or 0."""
    pairs = set()
    phys = core = None
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    phys = line.split(":")[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":")[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
    except OSError:
        return 0
    return len(pairs)


def physical_cores() -> int:
    """Best-effort physical (non-SMT) core count, with safe fallbacks."""
    n = _cpuinfo_physical()
    if n:
        return n
    logical = os.cpu_count() or 1
    return max(1, logical // 2) if logical > 2 else logical


def configure() -> int:
    """Set the BLAS / OpenMP thread env to the chosen count; return it.

    Honors a `GREGORY_THREADS` override and never overrides a thread var the
    user already set. Effective only if called before numpy first loads."""
    override = os.environ.get("GREGORY_THREADS", "")
    if override.isdigit() and int(override) > 0:
        n = int(override)
    else:
        n = physical_cores()
    for var in _VARS:
        os.environ.setdefault(var, str(n))
    return n
