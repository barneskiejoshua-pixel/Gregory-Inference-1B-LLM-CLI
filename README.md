# Gregory

A clean-room, pure-Python + NumPy inference engine for ternary-weight
(BitNet b1.58 family) LLMs.

James Gregory is an American inventor who patented a mechanical motor on April 26, 1887. His device (Patent No. 361,937) 

> **Readable first, with an optional fast path.** The forward pass is a clean
> fp32 reference (the correctness oracle). An opt-in C kernel accelerates decode
> ~4× while staying numerically equivalent — see *Performance* below. Clarity,
> correctness, and code-standard discipline come first; the kernel is a verified
> add-on, not the spec.

## What it does

```
GGUF bytes ──▶ parse ──▶ I2_S dequant ──▶ BPE tokenize ──▶ transformer
   gguf.py      ▲          quant.py         tokenizer.py     forward
               cache.py                                       model.py
                                                                 │
                                              sampler.py ◀── logits
                                                  │
                                            generate.py ──▶ tokens
```

| module | responsibility |
|---|---|
| `gregory/gguf.py` | GGUF v3 parser (header, KV metadata, tensor table, mmap) |
| `gregory/quant.py` | I2_S ternary → fp32 dequantization |
| `gregory/cache.py` | on-disk `.npy` cache of dequantized weights |
| `gregory/tokenizer.py` | GPT-2-style byte-level BPE (LLaMA-3 vocab) |
| `gregory/model.py` | `Gregory` transformer forward (RMSNorm, GQA, RoPE, FFN) |
| `gregory/sampler.py` | greedy / temperature / top-p / min-p + repeat penalty |
| `gregory/generate.py` | prefill + autoregressive decode with KV cache |
| `gregory/threads.py` | cap BLAS/OpenMP pool at physical cores (auto) |
| `gregory/kernels/` | optional C kernel: int8 × 2-bit ternary matvec |
| `gregory/cli.py` | `show` / `tokenize` / `dequant` / `generate` / `chat` |

## Performance

Decode is **memory-bandwidth bound** — it streams the weights once per token.
Two levers, both shipped:

1. **BLAS thread cap** (`gregory/threads.py`, automatic). On SMT CPUs the
   all-logical-cores default *slows* a bandwidth-bound matvec; Gregory pins the
   pool to physical cores before numpy loads. (~+29% on a 4C/8T i7.)
2. **Packed 2-bit ternary kernel** (`gregory/kernels/`, opt-in, default on when
   a compiler is present). Decode (single-token) projections run as int8
   activations × 2-bit ternary weights (`_mm256_maddubs_epi16`), streaming ~16×
   fewer weight bytes than fp32. fp32 is kept as the oracle and used for prefill.
3. **int8 LM head** (same kernel, `GREGORY_FAST_HEAD`, default on). The tied head
   (token_embd) is the largest tensor (1.3 GB fp32); a row-quantized int8 copy
   (int8 × int8 matvec) cuts its per-token cost ~2.5× (≈58 → ≈23 ms).

Measured on a 4-core / 8-thread i7 (BitNet b1.58-2B-4T):

| path | decode |
|---|---|
| original default (8 threads, fp32) | 1.43 tok/s |
| thread-capped fp32 reference | 1.84 tok/s |
| + packed ternary kernel | 5.4–7.2 tok/s |
| + int8 head (full fast path, default) | **~8.3 tok/s** |

Parity: the fast path picks the same tokens as fp32 (logit cosine ≈ 0.9996+);
the only difference is int8 quantization — the regime BitNet was trained in.
First launch packs weights + head once (~20s) and caches them.

Controls: `GREGORY_FAST=0` forces the pure-NumPy fp32 path; `GREGORY_FAST_HEAD=0`
keeps the fp32 head only; `GREGORY_THREADS=N` overrides the thread count. The
kernel needs `gcc` + AVX2; without them Gregory falls back to fp32 automatically.

## Architecture

Decoder-only transformer, BitNet b1.58 family:

- ternary (`{-1, 0, +1}`) weights, dequantized to fp32 for the reference matmul
- RMSNorm with BitNet's pre-projection **sub-norms** (attn + ffn)
- **GQA** (grouped-query attention) via broadcast, NeoX-style **RoPE**
- FFN with **squared-ReLU** activation (`max(0, x)²`), not SiLU
- tied LM head (output projection shares `token_embd`)

All architecture constants (layer count, dims, RoPE base, eps, vocab) are read
from the GGUF metadata at load time.

## Automotive engineering mode

Gregory ships specialized for **automotive engineering** (mechanics + engineers).
Because the BitNet weights are frozen and Gregory has no trainer, specialization
is done the way that works with frozen weights: a domain **system-prompt persona**
plus **retrieval grounding (RAG)** over a curated knowledge base
(`gregory/data/*.md`), structured on the Awesome-Automotive taxonomy plus the
standards the field uses (ISO 26262/ASIL, AUTOSAR, CAN/LIN/FlexRay, OBD-II/UDS,
SAE J3016, WLTP, ...).

```bash
GREGORY                              # automotive assistant (RAG on) by default
gregory kb                           # list knowledge-base topics
gregory kb "CAN bus arbitration"     # show what retrieval injects for a query
GREGORY --general                    # plain assistant, no domain
GREGORY --no-rag                     # domain persona, retrieval off
```

**Extend the corpus — the realistic "training" lever.** Feed documents through
the ingestion pipeline; it chunks them into KB entries Gregory retrieves over at
answer time (no code change). This is how you grow a frozen-weight RAG model.

```bash
# text / markdown (manuals, SOPs, notes) -> chunked + auto-tagged
python3 scripts/ingest_docs.py manual.md --out brakes --topic "Brake System"

# CSV code tables (DTC/OBD, parts catalogs) -> one entry per row
python3 scripts/ingest_docs.py dtc.csv --out obd_codes \
    --csv-topic-col code --csv-body-col description

# a whole folder of mixed .md/.txt/.csv
python3 scripts/ingest_docs.py ./docs --out manuals
# or: make ingest IN=./docs OUT=manuals

python3 scripts/ingest_docs.py manual.md --stdout   # preview, don't write
```

Supported inputs are stdlib-only: `.txt`, `.md`, `.csv` (PDF/DOCX → convert to
text first). Output lands in `gregory/data/<out>.md`; **review the draft** (the
chunking and auto-tags are heuristic), then `make kb` to confirm and chat. See
*Roadmap & honest limits* below for what this can and cannot do.

## Usage

```bash
# talk to the model interactively (multi-turn, keeps context across turns)
./GREGORY
./GREGORY --temperature 0.3 --max-tokens 200

# inspect a model
./gregory-cli show
./gregory-cli tokenize "Hello, Gregory."
./gregory-cli dequant blk.0.attn_q.weight

# one-shot generate (first run dequantizes + caches weights; later runs mmap)
./gregory-cli generate "The capital of France is" --max-tokens 30 --seed 0

# chat is also a subcommand of the main CLI
./gregory-cli chat --system "You are a terse assistant."
```

`./GREGORY` is a thin wrapper for `gregory chat`: an interactive REPL that holds
one KV cache across turns (so "what language do they speak *there*?" resolves
against the previous answer). Type `exit` or Ctrl-D to quit.

Default model path:
`~/BitNet/models/BitNet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf`
(override with `--model`).

## Development

```bash
make ci              # NPARC standards gate + model-free tests
make test            # tests only (model-free)
make nparc           # code-standard compliance gate
pytest -m model      # run the slow end-to-end smoke test (needs the GGUF)
```

## Code standards — NPARC

Gregory's Python is held to a Python-adapted subset of NASA Glenn's
[*Programming Guidelines for NPARC Alliance Software Development*][nparc]
(Towne, GRC, v2.0, 2004). The translatable rules — module/function docstrings,
no tabs, ≤80-column lines, one statement per line, no keyword/builtin
shadowing, a vetted dependency allowlist — are enforced by
`scripts/nparc_compliance.py` and a meta-test in the default gate. The
Fortran-only rules (common blocks, `implicit none`, kind params, statement
labels, column-7 format, Hollerith, Fortran I/O) are documented as N/A. See
[`docs/NPARC_GUIDELINES.md`](docs/NPARC_GUIDELINES.md).

The project starts **strict-clean** (zero violations, ratchet baseline at 0):
new debt fails CI.

[nparc]: https://www.grc.nasa.gov/www/winddocs/guidelines/pgmstds.pdf

## Roadmap & honest limits

Gregory runs a **frozen** BitNet b1.58 2B text model and has **no training
stack**. "Specialized for automotive engineering" therefore means *grounding a
frozen model*, not retraining it. What that does and does not cover:

**Works now (frozen weights + RAG over text):**
- Domain persona and terminology.
- Grounded Q&A over any text you add to the KB: OEM manual prose, SOP
  checklists, OBD/DTC code descriptions, standards summaries, parts notes.

**Not possible without a different stack / base model:**
- *True fine-tuning* (gradient training on the 2B) — needs a trainer and a
  labelled corpus; out of scope for this inference engine.
- *Multimodal* inputs — CAD/3D, schematics, photos, audio, time-series
  telemetry — needs a multimodal model and encoders; this is a text LLM.
- *Predictive maintenance* on sensor streams — a time-series ML problem, not an
  LLM task.

The KB ships with automotive engineering plus **PHY 1/2 physics** (mechanics,
energy, thermodynamics, fluids, electricity & magnetism) so concepts are
grounded with their formulas. Note: the 2B reliably recalls the right *formula
and concept* but is weak at *arithmetic* (it can fumble unit conversions) — see
the program-aided item below.

**Realistic improvement path (in priority order):**
1. Grow the text KB (`gregory/data/*.md`) — biggest quality gain per effort.
2. Program-aided execution (PAL): have the model write and run a Python
   snippet for quantitative questions, so numbers are computed, not guessed.
   This is the fix for the arithmetic weakness above.
3. Better retrieval (semantic embeddings using the model's own hidden states)
   instead of keyword overlap.
4. Verified few-shot exemplars for common diagnostic/spec tasks.
5. For real fine-tuning or multimodal: a separate training pipeline and a base
   model that supports it — a different project, not this engine.

## Relation to Vosne

The inference architecture and module boundaries follow the sibling project
[Vosne](../Vosne) (clean-room BitNet b1.58). Gregory is a fresh, trimmed
skeleton written to the NPARC standard from the first commit, focused on the
core inference path rather than the full assistant stack.
