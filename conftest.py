"""Root pytest config: make the repo root importable as `gregory`.

Belt-and-suspenders to `[tool.pytest.ini_options].pythonpath`, so tests import
the package regardless of how pytest is invoked (root, subdir, IDE runner).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
