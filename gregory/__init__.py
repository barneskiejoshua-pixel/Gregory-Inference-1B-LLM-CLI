"""Gregory -- a clean-room ternary-weight LLM inference engine in NumPy.

Named for James Gregory (1638-1675), the Scottish mathematician and astronomer
who built the first practical reflecting telescope and discovered the *Gregory
series* -- the infinite series that expresses a function as a convergent sum of
simple terms (arctan / pi). Gregory the engine works in the same spirit: it
turns a prompt into a convergent sequence of tokens, one simple term at a time,
and it owns every line of the path from GGUF bytes to logits.

Skeleton scope: the readable reference inference path (GGUF parse -> I2_S
dequant -> BPE tokenize -> fp32 transformer forward -> sampling -> generate).
Speed is a non-goal; clarity, correctness, and NPARC code-standard alignment
are the goals. See README.md and docs/NPARC_GUIDELINES.md.
"""

__version__ = "0.0.1"

# Cap the BLAS/OpenMP pool at physical cores BEFORE numpy loads anywhere (this
# module imports no numpy). On SMT CPUs the all-logical-cores default slows the
# bandwidth-bound decode -- see gregory/threads.py.
from . import threads as _threads   # noqa: E402

_threads.configure()
