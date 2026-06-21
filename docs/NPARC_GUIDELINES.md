# NPARC code standards in Gregory

Source: **"Programming Guidelines for NPARC Alliance Software Development"**,
Charles E. Towne, NASA Glenn Research Center, v2.0, 2004.
<https://www.grc.nasa.gov/www/winddocs/guidelines/pgmstds.pdf>

It is a **Fortran 90 style guide** for the Wind-US CFD code. Its stated
priority order is **maintainability > portability > efficiency**. It is a
*style/maintainability* standard, not a safety specification — most of it is
Fortran-specific. This document records, honestly, which rules translate to
Gregory's Python and how each is treated.

## Enforced — STRICT (must stay clean; any violation fails CI)

| NPARC rule | Python analogue | Mechanism |
|---|---|---|
| 2.1 Use the standard language | imports limited to stdlib + `numpy` + a vetted allowlist (`regex`, fallback-guarded); a **new** third-party import fails | `nparc_compliance.py` `new_deps` |
| 3.5 Do not use tabs | zero tab characters | `tabs` |
| 3.11 Standard header per unit | every module has a docstring | `no_mod_doc` |

## Enforced — RATCHET (count may not grow above the committed baseline)

| NPARC rule | Python analogue | Mechanism |
|---|---|---|
| 3.2 Keep lines below 80 chars | ≤ 80-column lines | `long_lines` |
| 3.2 One statement per line | no `;`-joined statements | `multi_stmt` |
| 3.6 Do not reuse a keyword/name | no shadowing of a Python keyword/builtin | `shadowing` |
| 3.11 Standard header per unit | public functions have docstrings | `no_fn_doc` |

The baseline lives in `scripts/nparc_baseline.json`. **Gregory starts
strict-clean: every ratchet count is 0.** A reflow that *reduces* a backlog is
welcome; a change that grows one is rejected. (Mass-reflowing working code for
a style rule is itself a risk — the ratchet captures the intent of the rule
without forcing that risk, the same call Vosne made.)

## Not applicable in Python (documented, not enforced)

These NPARC rules are Fortran-language constructs with no Python equivalent:

- common blocks / `include` of shared state
- `implicit none` (Python has no implicit typing of undeclared names)
- `kind` type parameters for portable precision
- statement labels and `GOTO`
- column-7 fixed-format source and `&` line continuation
- Hollerith constants
- alternate `RETURN`
- Fortran-specific I/O (`OPEN`/`READ`/`WRITE`/`FORMAT`)
- 1-based / explicit-bound array declarations

## Spirit-level guidance carried over (not mechanically checkable)

NPARC's prose also asks for: meaningful names, comments that explain *why*,
small single-purpose procedures, and explicit interfaces. Gregory follows these
by convention; they are reviewed, not linted.

## Running the checker

```bash
python3 scripts/nparc_compliance.py            # human-readable scorecard
python3 scripts/nparc_compliance.py --strict   # CI gate (exit 1 on any fail)
python3 scripts/nparc_compliance.py --set-baseline   # freeze ratchet counts
```

The default test gate also runs `tests/test_nparc_compliance.py`, so a
standards regression fails `pytest`, not only the separate CI step.
