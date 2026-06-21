"""Gregory command-line interface.

Subcommands:
    show                  model spec + tensor inventory from the GGUF
    tokenize "<text>"     encode text -> ids -> round-trip decode
    dequant <tensor>      dequantize one I2_S tensor; show stats + a sample
    generate "<prompt>"   load the model and stream a continuation
    chat                  interactive REPL (speak to the model across turns)
    bench ["<prompt>"]    decode-latency percentiles (p50/p90/p99) + peak RSS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import automotive, bench, gguf, model as model_mod, quant
from .generate import generate as run_generate
from .generate import generate_from_ids as gen_from_ids
from .tokenizer import Tokenizer

DEFAULT_MODEL = str(Path.home()
                    / "BitNet/models/BitNet-b1.58-2B-4T-gguf"
                    / "ggml-model-i2_s.gguf")


def cmd_show(args: argparse.Namespace) -> int:
    """Print model metadata and a sample of the tensor inventory."""
    g = gguf.load(args.model)
    arch = g.get("general.architecture")
    print(f"file       : {g.path}")
    print(f"version    : GGUF v{g.version}")
    print(f"arch       : {arch}")
    print(f"tensors    : {len(g.tensors)}")
    keys = ("block_count", "embedding_length", "feed_forward_length",
            "attention.head_count", "attention.head_count_kv", "vocab_size")
    for k in keys:
        print(f"  {k:<28}: {g.get(arch + '.' + k)}")
    print("\nfirst tensors:")
    for name in list(g.tensors)[:8]:
        t = g.tensors[name]
        print(f"  {name:<28} {t.dtype.name:<8} {t.shape}")
    g.close()
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    """Encode `args.text`, print the ids, and verify the decode round-trip."""
    g = gguf.load(args.model)
    tok = Tokenizer.from_gguf(g)
    ids = tok.encode(args.text)
    print(f"ids   : {ids}")
    print(f"count : {len(ids)}")
    print(f"decode: {tok.decode(ids)!r}")
    g.close()
    return 0


def cmd_dequant(args: argparse.Namespace) -> int:
    """Dequantize one tensor to fp32 and print summary statistics."""
    g = gguf.load(args.model)
    if args.tensor not in g.tensors:
        print(f"no such tensor: {args.tensor}", file=sys.stderr)
        g.close()
        return 1
    t = g.tensors[args.tensor]
    blob = g.tensor_bytes(args.tensor)
    if t.dtype == gguf.GGMLType.I2_S:
        arr = quant.dequantize_tensor(blob, tuple(t.shape))
    else:
        print(f"tensor dtype {t.dtype.name} is not I2_S", file=sys.stderr)
        g.close()
        return 1
    print(f"shape : {arr.shape}")
    print(f"min   : {arr.min():.5f}")
    print(f"max   : {arr.max():.5f}")
    print(f"mean  : {arr.mean():.5f}")
    print(f"sample: {arr.ravel()[:8]}")
    g.close()
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Load the model and stream a continuation of `args.prompt`."""
    g = gguf.load(args.model)
    tok = Tokenizer.from_gguf(g)
    print("loading model (first run dequantizes + caches weights)...",
          flush=True)
    net = model_mod.Gregory(g, verbose=args.verbose)
    print(f"\n{args.prompt}", end="", flush=True)
    for tid in run_generate(net, tok, args.prompt, max_tokens=args.max_tokens,
                            temperature=args.temperature, seed=args.seed,
                            stream=True):
        print(tok.decode([tid]), end="", flush=True)
    print()
    g.close()
    return 0


def _special_ids(tok: Tokenizer) -> tuple[int, int, int]:
    """Return (start_header, end_header, eot) ids for the LLaMA-3 chat
    template, falling back to the canonical ids if a name is absent."""
    sh = tok.vocab.get("<|start_header_id|>", 128006)
    eh = tok.vocab.get("<|end_header_id|>", 128007)
    eot = tok.vocab.get("<|eot_id|>", 128009)
    return sh, eh, eot


def _chat_header(tok: Tokenizer, role: str, sh: int, eh: int) -> list[int]:
    """Token ids for one turn header: <|start|> role <|end|> newline."""
    ids = [sh]
    ids += tok.encode(role, add_bos=False)
    ids.append(eh)
    ids += tok.encode("\n\n", add_bos=False)
    return ids


def _stream_reply(net, tok, segment, kv, args) -> None:
    """Decode the assistant turn from `segment`, streaming text as it lands."""
    stops = tok.eog_ids
    acc: list[int] = []
    printed = ""
    for tid in gen_from_ids(net, segment, kv_cache=kv,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            stop_ids=stops, stream=True):
        acc.append(tid)
        text = tok.decode(acc)        # re-decode so multibyte UTF-8 is intact
        sys.stdout.write(text[len(printed):])
        sys.stdout.flush()
        printed = text


def _load_studio():
    """Lazily import the CAD CREATOR + CODER sub-layers for in-chat routing.

    Returns (cad, coder) where each is a (router_module, pipeline_module) pair
    or None if that sibling package is not importable. Imported here at REPL
    start -- never at module load -- so the frozen inference core has no hard
    dependency on the sub-layers and Gregory still runs standalone if they are
    absent. Fail-open: any import problem just disables that layer."""
    root = Path(__file__).resolve().parents[1]
    # Prefer the canonical (normalized) package dirs; only fall back to a
    # spaced-name sibling if its canonical form is absent. This avoids importing
    # an unrelated project that happens to live under a similar spaced name.
    subs = ["gregory_cad_creator", "gregory_coder"]
    for spaced, canon in (("GREGORY CAD CREATOR", "gregory_cad_creator"),
                          ("GREOGORY CODER", "gregory_coder")):
        if not (root / canon).is_dir() and (root / spaced).is_dir():
            subs.append(spaced)
    for sub in subs:
        p = root / sub
        if p.is_dir() and str(p) not in sys.path:
            sys.path.insert(0, str(p))
    cad = coder = None
    try:
        from cadcreator import pipeline as cad_pipe, router as cad_router
        cad = (cad_router, cad_pipe)
    except Exception:
        cad = None
    try:
        from coder import pipeline as code_pipe, router as code_router
        coder = (code_router, code_pipe)
    except Exception:
        coder = None
    return cad, coder


def _print_cad_turn(res) -> None:
    """Print a CAD pipeline result as a chat turn."""
    if res.family is None:
        print(f"gregory[cad]> {res.message}")
        return
    print(f"gregory[cad:{res.family}]> {res.message}")
    if res.scad:
        print(res.scad)


def _print_code_turn(res) -> None:
    """Print a CODER pipeline result as a chat turn."""
    tag = res.language or "code"
    print(f"gregory[code:{tag}]> {res.message}")
    if res.code:
        print(res.code)


def _maybe_save_cad(user, res) -> None:
    """If the request named an output path, save the .scad (+ .stl) there.

    Fail-open: the save logic lives in cadcreator.output (the frozen core only
    calls it), and any problem just prints a note instead of breaking chat."""
    if not getattr(res, "scad", None):
        return
    try:
        from cadcreator import output as cad_output
        dest = cad_output.extract_path(user)
        if not dest:
            return
        saved = cad_output.save_part(res, dest)
        extra = f" + {saved['stl']}" if saved.get("stl") else ""
        print(f"gregory[cad]> saved {saved['scad']}{extra}")
    except Exception as exc:                       # noqa: BLE001
        print(f"gregory[cad]> (could not save: {exc})")


def _studio_route(user, cad, coder, net, tok) -> bool:
    """Dispatch a CAD/code turn to its validated pipeline; True if handled.

    CAD-intent goes to the curated CAD pipeline (model-free, valid by
    construction); code-intent goes to the CODER pipeline with the loaded model
    available for the free path. Anything else returns False to fall through to
    normal Gregory chat."""
    if cad is not None and cad[0].route(user).is_cad:
        res = cad[1].generate_part(user)
        _print_cad_turn(res)
        _maybe_save_cad(user, res)
        return True
    if coder is not None and coder[0].route(user).is_code:
        _print_code_turn(coder[1].generate_code(user, model=net, tok=tok))
        return True
    return False


def cmd_chat(args: argparse.Namespace) -> int:
    """Run an interactive chat REPL, holding one KV cache across turns."""
    g = gguf.load(args.model)
    tok = Tokenizer.from_gguf(g)
    print("loading model (first run dequantizes + caches weights)...",
          flush=True)
    net = model_mod.Gregory(g, verbose=args.verbose)
    sh, eh, eot = _special_ids(tok)
    kv = net.init_kv_cache()

    domain = not args.general
    rag = domain and not args.no_rag
    system = args.system
    if system is None:
        system = (automotive.SYSTEM_PROMPT if domain
                  else "You are Gregory, a helpful assistant.")

    # The first segment carries BOS plus the system turn; later turns append
    # only the new user turn (the KV cache already holds the history).
    segment = [tok.bos_id]
    segment += _chat_header(tok, "system", sh, eh)
    segment += tok.encode(system, add_bos=False)
    segment.append(eot)

    cad, coder = (None, None) if args.no_studio else _load_studio()
    studio_on = cad is not None or coder is not None

    if domain:
        n = len(automotive.kb())
        mode = f"automotive domain, {n} KB entries" + (", RAG on" if rag
                                                        else ", RAG off")
        print(f"\nGregory ({mode}). 'exit' or Ctrl-D to quit.")
    else:
        print("\nGregory (general). 'exit' or Ctrl-D to quit.")
    if studio_on:
        layers = [name for name, mod in (("CAD", cad), ("code", coder))
                  if mod is not None]
        print(f"studio routing on ({' + '.join(layers)}) -- design/code "
              f"requests build real, validated output.")
    print()
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if user in ("exit", "quit"):
            break
        if not user:
            continue
        if studio_on and _studio_route(user, cad, coder, net, tok):
            print()
            continue
        content = user
        if rag:
            ctx = automotive.build_context(user)
            if ctx:
                content = f"{ctx}\n\nQuestion: {user}"
        segment += _chat_header(tok, "user", sh, eh)
        segment += tok.encode(content, add_bos=False)
        segment.append(eot)
        segment += _chat_header(tok, "assistant", sh, eh)

        print("gregory> ", end="", flush=True)
        _stream_reply(net, tok, segment, kv, args)
        print("\n")
        net.forward([eot], kv)        # close the assistant turn in the cache
        segment = []                  # history now lives in kv; start fresh
    g.close()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    """Load the model and report decode-latency percentiles + peak RSS."""
    g = gguf.load(args.model)
    tok = Tokenizer.from_gguf(g)
    print("loading model (first run dequantizes + caches weights)...",
          flush=True)
    net = model_mod.Gregory(g, verbose=args.verbose)
    prompt_ids = tok.encode(args.prompt, add_bos=True)
    result = bench.bench_decode(
        net, prompt_ids, decode_tokens=args.decode_tokens,
        warmup=args.warmup, temperature=args.temperature, seed=args.seed)
    print(bench.format_report(result, args.warmup))
    g.close()
    return 0


def cmd_kb(args: argparse.Namespace) -> int:
    """Inspect the automotive knowledge base, or show retrieval for a query."""
    entries = automotive.kb()
    if args.query:
        hits = automotive.retrieve(args.query, args.k)
        if not hits:
            print("no relevant entries")
            return 0
        for entry in hits:
            print(f"## {entry.topic}  [{entry.source}]")
            print(entry.text)
            print()
        return 0
    sources = sorted({entry.source for entry in entries})
    print(f"knowledge base: {len(entries)} entries from "
          f"{len(sources)} file(s)")
    for src in sources:
        print(f"  {src}")
    print("\ntopics:")
    for topic in automotive.topics():
        print(f"  - {topic}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to a subcommand handler."""
    p = argparse.ArgumentParser(
        prog="gregory",
        description="Gregory -- clean-room ternary-weight LLM inference.")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"path to GGUF model (default: {DEFAULT_MODEL})")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("show", help="model spec + tensor inventory")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("tokenize", help="encode text and decode back")
    sp.add_argument("text", type=str)
    sp.set_defaults(func=cmd_tokenize)

    sp = sub.add_parser("dequant", help="dequantize an I2_S tensor")
    sp.add_argument("tensor", type=str, help="e.g. blk.0.attn_q.weight")
    sp.set_defaults(func=cmd_dequant)

    sp = sub.add_parser("generate", help="stream a continuation")
    sp.add_argument("prompt", type=str)
    sp.add_argument("--max-tokens", type=int, default=50, dest="max_tokens")
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--seed", type=int, default=None)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_generate)

    sp = sub.add_parser("chat", help="interactive chat REPL (automotive)")
    sp.add_argument("--system", type=str, default=None,
                    help="override the system prompt")
    sp.add_argument("--general", action="store_true",
                    help="general assistant instead of automotive domain")
    sp.add_argument("--no-rag", action="store_true", dest="no_rag",
                    help="domain persona but no knowledge-base retrieval")
    sp.add_argument("--no-studio", action="store_true", dest="no_studio",
                    help="disable CAD/code routing to the sub-layer pipelines")
    sp.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_chat)

    sp = sub.add_parser("bench", help="decode-latency percentiles + peak RSS")
    sp.add_argument("prompt", nargs="?", type=str,
                    default="The quick brown fox jumps over the lazy dog.")
    sp.add_argument("--decode-tokens", type=int, default=64,
                    dest="decode_tokens", help="timed decode steps")
    sp.add_argument("--warmup", type=int, default=3,
                    help="untimed leading steps to discard")
    sp.add_argument("--temperature", type=float, default=0.7)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--verbose", action="store_true")
    sp.set_defaults(func=cmd_bench)

    sp = sub.add_parser("kb", help="inspect/search the knowledge base")
    sp.add_argument("query", nargs="?", default=None,
                    help="optional query; omitted = list topics")
    sp.add_argument("-k", type=int, default=3, help="number of entries")
    sp.set_defaults(func=cmd_kb)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
