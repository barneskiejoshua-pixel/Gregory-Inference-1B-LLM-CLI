"""Logits -> next-token sampling: greedy / temperature / top-p / min-p.

Includes the llama.cpp repeat penalty: positive logits of recently-seen tokens
are divided by the penalty, negatives multiplied -- discouraging loops without
hard-banning repeats.
"""

from __future__ import annotations

import numpy as np


def sample(
    logits: np.ndarray,
    temperature: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repeat_penalty: float = 1.1,
    recent_ids: list[int] | None = None,
    repeat_window: int = 64,
    rng: np.random.Generator | None = None,
) -> int:
    """Sample one token id from `logits` (1-D float array, length vocab_size).

    `temperature <= 0` selects the argmax (greedy). `recent_ids`, when given,
    applies the repeat penalty over the last `repeat_window` ids."""
    x = logits.astype(np.float32).copy()

    if recent_ids and repeat_penalty > 1.0:
        window = recent_ids[-repeat_window:]
        ids = np.unique(np.asarray(window, dtype=np.int64))
        ids = ids[(ids >= 0) & (ids < x.size)]
        vals = x[ids]
        x[ids] = np.where(vals > 0, vals / repeat_penalty,
                          vals * repeat_penalty)

    if temperature <= 0.0:
        return int(np.argmax(x))
    x = x / temperature
    x -= x.max()
    probs = np.exp(x)
    probs /= probs.sum()

    if min_p > 0.0:
        cutoff = min_p * probs.max()
        probs = np.where(probs < cutoff, 0.0, probs)
        s = probs.sum()
        if s == 0:                      # min_p too strict; fall back to greedy
            return int(np.argmax(logits))
        probs /= s

    if 0.0 < top_p < 1.0:
        # top_p keeps only the largest tokens until the cumulative mass reaches
        # `top_p` -- a handful for a peaked distribution. Sorting the full vocab
        # (~128k) every token was the dominant sampler cost, so sort only a
        # candidate pool. Candidates come from the NONZERO probs: min_p has
        # usually already zeroed nearly the whole vocab, and restricting to
        # nonzeros also keeps argpartition off its duplicate-pivot pathology
        # (a vocab-long run of equal zeros makes it ~14x slower than on the
        # distinct softmax values). Exact: when the pool's mass falls short of
        # top_p (near-uniform, e.g. high temperature) we sort all candidates, so
        # the kept set is always identical to a full sort.
        pool = 1024
        nz = np.flatnonzero(probs)
        pv = probs[nz]
        if nz.size > pool:
            top = np.argpartition(pv, nz.size - pool)[nz.size - pool:]
            order = nz[top[np.argsort(-pv[top])]]
        else:
            order = nz[np.argsort(-pv)]
        cum = np.cumsum(probs[order])
        if cum[-1] < top_p:                     # pool short of mass: exact sort
            order = nz[np.argsort(-pv)]
            cum = np.cumsum(probs[order])
        keep_n = int(np.searchsorted(cum, top_p)) + 1
        keep = order[:keep_n]
        new = np.zeros_like(probs)
        new[keep] = probs[keep]
        s = new.sum()
        if s == 0:
            return int(np.argmax(logits))
        probs = new / s

    if rng is None:
        rng = np.random.default_rng()
    # Draw only over the surviving support: top_p/min_p have usually zeroed
    # almost the whole vocab, and rng.choice over the full ~128k p-vector is
    # otherwise ~40x costlier than over the few live tokens. flatnonzero returns
    # ascending token ids, so the single inverse-CDF draw lands on exactly the
    # token the full-vocab choice would (zeros don't shift the cumulative). When
    # nothing was truncated, the support is the whole vocab -- draw directly.
    support = np.flatnonzero(probs)
    if support.size == probs.size:
        return int(rng.choice(probs.size, p=probs))
    return int(rng.choice(support, p=probs[support] / probs[support].sum()))
