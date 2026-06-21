"""Meta-test: the package must satisfy the NPARC STRICT rules and the ratchet.

This keeps the code-standard alignment honest -- a new tab, a missing module
docstring, an unvetted dependency, or a regression past the ratchet baseline
fails the default test gate, not just a separate CI step.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "nparc_compliance", ROOT / "scripts" / "nparc_compliance.py")
nparc = importlib.util.module_from_spec(_SPEC)
# Register before exec: dataclass introspection resolves cls.__module__ via
# sys.modules, which fails if the module is not yet registered.
sys.modules["nparc_compliance"] = nparc
_SPEC.loader.exec_module(nparc)


def test_strict_rules_pass():
    """No tabs, no missing module docstrings, no unvetted imports."""
    f = nparc.scan()
    assert f.tabs == [], f"tabs: {f.tabs[:5]}"
    assert f.no_mod_doc == [], f"modules w/o docstring: {f.no_mod_doc}"
    assert f.new_deps == [], f"unvetted imports: {f.new_deps}"


def test_ratchet_not_regressed():
    """Ratcheted counts do not exceed the committed baseline."""
    f = nparc.scan()
    assert nparc._strict_fails(f) == []
