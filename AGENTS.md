# Working in Gregory

Gregory is a clean-room NumPy reference inference engine for ternary-weight
(BitNet b1.58) LLMs, with an optional in-repo C kernel for fast decode. Read
`README.md` first.

## Invariants

- **Dep-light runtime.** numpy only, plus `regex` as an optional,
  fallback-guarded dep. Do not add a runtime import without (a) a graceful
  stdlib fallback and (b) adding it to `ALLOWED_IMPORTS` in
  `scripts/nparc_compliance.py`. The NPARC gate fails otherwise.
- **NPARC standards.** Module + public-function docstrings, no tabs, ≤80-col
  lines, one statement per line, no keyword/builtin shadowing. The project is
  strict-clean (ratchet baseline 0) — keep it there. See
  `docs/NPARC_GUIDELINES.md`.
- **fp32 is the oracle.** The pure-NumPy fp32 forward is the correctness
  reference. The optional packed kernel (`gregory/kernels/`, `GREGORY_FAST`,
  decode-only) must stay numerically equivalent to it (kernel parity test +
  logit-cosine check); prefill always runs fp32. The kernel degrades gracefully
  to fp32 when `gcc`/AVX2 are absent.
- **Tests are model-free by default.** Anything needing the GGUF is marked
  `@pytest.mark.model` / `slow` and skips when the file is absent.

## The gate

```bash
make ci     # nparc strict + model-free tests — run before every commit
```
