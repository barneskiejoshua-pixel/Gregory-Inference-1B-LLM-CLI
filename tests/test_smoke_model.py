"""End-to-end smoke test against the real GGUF (model-gated, slow).

Skipped automatically when the model file is absent, so the default CI gate
stays model-free. Run with `pytest -m model` once the GGUF is in place.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gregory import gguf, model as model_mod
from gregory.generate import generate
from gregory.tokenizer import Tokenizer

MODEL = (Path.home() / "BitNet/models/BitNet-b1.58-2B-4T-gguf"
         / "ggml-model-i2_s.gguf")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.model,
    pytest.mark.skipif(not MODEL.exists(), reason="GGUF model not present"),
]


def test_forward_and_generate():
    """Load the model, run a forward, and greedily decode a few tokens."""
    g = gguf.load(MODEL)
    tok = Tokenizer.from_gguf(g)
    net = model_mod.Gregory(g)
    logits = net.forward(tok.encode("The capital of France is"),
                         net.init_kv_cache(), last_only=True)
    assert logits.shape == (1, net.vocab_size)
    assert np.isfinite(logits).all()
    out = generate(net, tok, "The capital of France is", max_tokens=5,
                   temperature=0.0, seed=0)
    assert len(out) >= 1
    g.close()
