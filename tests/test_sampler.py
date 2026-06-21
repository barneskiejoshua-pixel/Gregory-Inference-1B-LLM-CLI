"""Unit tests for the sampler (deterministic, model-free)."""

from __future__ import annotations

import numpy as np

from gregory.sampler import sample


def test_greedy_picks_argmax():
    """temperature <= 0 returns the highest-logit index."""
    logits = np.array([0.1, 5.0, -2.0, 1.0], dtype=np.float32)
    assert sample(logits, temperature=0.0) == 1


def test_sampling_is_seed_deterministic():
    """A fixed RNG gives a reproducible draw."""
    logits = np.array([1.0, 2.0, 3.0, 0.5], dtype=np.float32)
    a = sample(logits, temperature=1.0, rng=np.random.default_rng(0))
    b = sample(logits, temperature=1.0, rng=np.random.default_rng(0))
    assert a == b


def test_repeat_penalty_demotes_recent_token():
    """A heavy repeat penalty can move the argmax off a recent token."""
    logits = np.array([10.0, 9.0, 0.0], dtype=np.float32)
    out = sample(logits, temperature=0.0, repeat_penalty=5.0,
                 recent_ids=[0])
    assert out == 1
