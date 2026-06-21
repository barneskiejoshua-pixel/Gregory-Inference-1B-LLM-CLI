#!/usr/bin/env python3
"""NPARC Alliance programming-guideline compliance -- adapted for Python.

Source: "Programming Guidelines for NPARC Alliance Software Development"
(Charles E. Towne, NASA Glenn Research Center, v2.0, 2004) --
https://www.grc.nasa.gov/www/winddocs/guidelines/pgmstds.pdf

It is a FORTRAN 90 *style* guide for the Wind-US CFD code, optimizing (in
order) maintainability, portability, efficiency. Most of it is Fortran-specific
and N/A to Python (common blocks, `implicit none`, kind params, statement
labels, column-7 formatting, continuation `&`, Hollerith, Fortran I/O). This
checker enforces the subset that translates to language-agnostic hygiene.

What maps (Fortran rule -> Python analogue), and how it is treated here:
  2.1  "Use the standard language" -> imports limited to stdlib + numpy + a
       vetted optional-deps allowlist; a NEW third-party import fails (STRICT).
  3.5  "Don't use tabs"            -> zero tabs (STRICT).
  3.11 "Standard header per unit"  -> every module has a docstring (STRICT);
       public functions have docstrings (RATCHET).
  3.2  "Keep lines below 80 chars" -> RATCHET.
  3.2  "One statement per line"    -> no ';'-joined statements (RATCHET).
  3.6  "Don't reuse a keyword/name"-> don't shadow a keyword/builtin (RATCHET).

RATCHET = the current count is the baseline (scripts/nparc_baseline.json); it
may not GROW. Working code is never mass-reflowed (that risks breaking a tested
system); new debt is blocked. Reducing a backlog is safe incremental work the
ratchet protects.

Usage:
  python3 scripts/nparc_compliance.py            # report
  python3 scripts/nparc_compliance.py --strict   # CI gate
  python3 scripts/nparc_compliance.py --set-baseline
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import keyword
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "gregory"
BASELINE = ROOT / "scripts" / "nparc_baseline.json"
MAX_LINE = 80

# stdlib + numpy + the project package, plus optional deps used WITH a graceful
# fallback. Adding a name here must be a conscious choice -- that is the point.
# `cadcreator`/`coder` are the first-party sibling sub-layers (CAD CREATOR /
# CODER); the chat REPL imports them lazily and fail-open, so they are optional
# project packages, not external third-party dependencies.
ALLOWED_IMPORTS = (set(sys.stdlib_module_names)
                   | {"gregory", "numpy", "regex", "cadcreator", "coder"})
_RESERVED = set(dir(builtins)) | set(keyword.kwlist)


@dataclass
class Findings:
    """Tallied rule violations across the package."""

    new_deps: list = field(default_factory=list)      # 2.1  STRICT
    tabs: list = field(default_factory=list)          # 3.5  STRICT
    no_mod_doc: list = field(default_factory=list)    # 3.11 STRICT
    long_lines: int = 0                               # 3.2  ratchet
    multi_stmt: int = 0                               # 3.2  ratchet
    shadowing: int = 0                                # 3.6  ratchet
    no_fn_doc: int = 0                                # 3.11 ratchet
    fn_total: int = 0


def _py_files() -> list[Path]:
    """All package source files, excluding __pycache__."""
    return sorted(p for p in PKG.rglob("*.py")
                  if "__pycache__" not in p.parts)


def _imports(tree: ast.AST) -> set[str]:
    """Top-level module names imported anywhere in `tree`."""
    mods: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            mods.add(n.module.split(".")[0])
    return mods


def _count_semicolons(src: str) -> int:
    """Count ';' statement separators via tokenize, so prose semicolons inside
    strings and comments are never miscounted (a per-line scan flags them)."""
    n = 0
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.OP and tok.string == ";":
                n += 1
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return n


def scan() -> Findings:
    """Walk the package and tally every tracked rule into a Findings."""
    f = Findings()
    for path in _py_files():
        rel = path.relative_to(ROOT)
        src = path.read_text(encoding="utf-8")
        f.multi_stmt += _count_semicolons(src)
        for i, ln in enumerate(src.splitlines(), 1):
            if len(ln) > MAX_LINE:
                f.long_lines += 1
            if "\t" in ln:
                f.tabs.append(f"{rel}:{i}")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        if not ast.get_docstring(tree):
            f.no_mod_doc.append(str(rel))
        for m in _imports(tree):
            if m not in ALLOWED_IMPORTS:
                f.new_deps.append(f"{rel}: import {m}")
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f.fn_total += 1
                if not n.name.startswith("_") and not ast.get_docstring(n):
                    f.no_fn_doc += 1
            elif isinstance(n, ast.arg) and n.arg in _RESERVED:
                f.shadowing += 1
            elif (isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                    and n.id in _RESERVED):
                f.shadowing += 1
    return f


def _baseline() -> dict:
    """Load the ratchet baseline, or an empty dict if absent/corrupt."""
    try:
        return json.loads(BASELINE.read_text())
    except (OSError, ValueError):
        return {}


def report(f: Findings) -> None:
    """Print a human-readable scorecard."""
    b = _baseline()
    print("=== NPARC Alliance guidelines (Python-adapted) ===\n")
    print("  STRICT (must stay clean):")
    print(f"    2.1  new third-party imports : {len(f.new_deps)}")
    for v in f.new_deps[:6]:
        print(f"           {v}")
    print(f"    3.5  tabs                    : {len(f.tabs)}")
    print(f"    3.11 modules w/o docstring   : {len(f.no_mod_doc)}")
    for v in f.no_mod_doc[:6]:
        print(f"           {v}")

    def _r(name, val, key):
        base = b.get(key)
        tag = f"  (baseline {base})" if base is not None else ""
        flag = "  REGRESSED" if base is not None and val > base else ""
        print(f"    {name:<28}: {val}{tag}{flag}")

    print("\n  RATCHET (may not grow):")
    _r("3.2  lines >80 chars", f.long_lines, "long_lines")
    _r("3.2  multi-statement lines", f.multi_stmt, "multi_stmt")
    _r("3.6  keyword/builtin shadow", f.shadowing, "shadowing")
    _r("3.11 public fns w/o docstring", f.no_fn_doc, "no_fn_doc")
    print(f"         (public functions total: {f.fn_total})")
    print("\n  N/A in Python: common blocks, implicit none, kind params, "
          "statement labels,\n  column/continuation format, Hollerith, "
          "Fortran I/O.")


def _strict_fails(f: Findings) -> list[str]:
    """Return the list of STRICT-rule failures (empty == pass)."""
    fails = []
    if f.new_deps:
        fails.append(f"{len(f.new_deps)} new third-party import(s)")
    if f.tabs:
        fails.append(f"{len(f.tabs)} tab(s)")
    if f.no_mod_doc:
        fails.append(f"{len(f.no_mod_doc)} module(s) w/o docstring")
    b = _baseline()
    for key, val in (("long_lines", f.long_lines),
                     ("multi_stmt", f.multi_stmt),
                     ("shadowing", f.shadowing), ("no_fn_doc", f.no_fn_doc)):
        base = b.get(key)
        if base is not None and val > base:
            fails.append(f"{key} regressed {base}->{val}")
    return fails


def main() -> int:
    """CLI entry: report, optionally gate (--strict) or set the baseline."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--set-baseline", action="store_true")
    args = ap.parse_args()
    f = scan()
    report(f)
    if args.set_baseline:
        BASELINE.write_text(json.dumps({
            "long_lines": f.long_lines, "multi_stmt": f.multi_stmt,
            "shadowing": f.shadowing, "no_fn_doc": f.no_fn_doc}, indent=2)
            + "\n")
        print(f"\nbaseline set: {BASELINE.name}")
        return 0
    if args.strict:
        fails = _strict_fails(f)
        if fails:
            print(f"\nSTRICT FAIL: {'; '.join(fails)}")
            return 1
        print("\nSTRICT PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
