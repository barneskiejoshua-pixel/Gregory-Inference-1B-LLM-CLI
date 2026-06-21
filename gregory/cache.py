"""On-disk cache of dequantized fp32 weights.

The first load dequantizes every I2_S tensor to fp32 (slow); each tensor is
then saved as its own `.npy`. Subsequent loads `np.load(..., mmap_mode='r')`
each tensor -- no dequant, no copy, near-instant.

Cache location: `<gregory_root>/.cache/<fingerprint>/`. The fingerprint is a
hash of (absolute path, file size, mtime), so re-quantizing the source GGUF
invalidates the cache automatically.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache"

# Bump when the dequant math or stored tensor shape changes, so a stale cache
# from older (possibly wrong) math is not silently reused.
DEQUANT_VERSION = 1


def fingerprint(model_path: Path) -> str:
    """Return a short content-fingerprint hash for `model_path`."""
    st = model_path.stat()
    key = f"v{DEQUANT_VERSION}|{model_path.resolve()}|{st.st_size}|" \
          f"{int(st.st_mtime)}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def cache_dir_for(model_path: str | Path) -> Path:
    """Return the cache directory for `model_path` (not created here)."""
    return CACHE_ROOT / fingerprint(Path(model_path))


def is_valid(cache_dir: Path, expected_tensor_count: int) -> bool:
    """True if `cache_dir` holds a complete manifest for the model."""
    meta = cache_dir / "manifest.json"
    if not meta.exists():
        return False
    try:
        m = json.loads(meta.read_text())
        return m.get("tensor_count") == expected_tensor_count
    except (json.JSONDecodeError, OSError):
        return False


def save(cache_dir: Path, weights: dict[str, np.ndarray]) -> None:
    """Write each tensor as its own `.npy` plus an index manifest."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    name_index: dict[str, str] = {}
    for i, (name, arr) in enumerate(weights.items()):
        safe = f"t{i:04d}.npy"
        np.save(cache_dir / safe, arr)
        name_index[name] = safe
    (cache_dir / "manifest.json").write_text(json.dumps({
        "tensor_count": len(weights),
        "index": name_index,
    }))


def load_mmap(cache_dir: Path) -> dict[str, np.ndarray]:
    """Memory-map every cached tensor and return name -> array."""
    meta = json.loads((cache_dir / "manifest.json").read_text())
    out: dict[str, np.ndarray] = {}
    for name, fname in meta["index"].items():
        out[name] = np.load(cache_dir / fname, mmap_mode="r")
    return out
