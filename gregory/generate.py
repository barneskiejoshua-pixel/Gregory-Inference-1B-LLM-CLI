"""Generation loop: encode -> prefill -> autoregressive decode with KV cache."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .model import Gregory
from .sampler import sample
from .tokenizer import Tokenizer


def generate(
    model: Gregory,
    tok: Tokenizer,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repeat_penalty: float = 1.1,
    seed: int | None = None,
    stream: bool = False,
    stop_ids: set[int] | None = None,
):
    """Generate up to `max_tokens` ids after `prompt`, stopping at EOS.

    `stream=True` returns a generator yielding ids; otherwise a list."""
    rng = np.random.default_rng(seed) if seed is not None else None
    prompt_ids = tok.encode(prompt, add_bos=True)
    stops = stop_ids if stop_ids is not None else tok.eog_ids
    return generate_from_ids(
        model, prompt_ids, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, min_p=min_p, repeat_penalty=repeat_penalty, rng=rng,
        stream=stream, stop_ids=stops)


def generate_from_ids(
    model: Gregory,
    prompt_ids: list[int],
    *,
    kv_cache: dict | None = None,
    max_tokens: int = 50,
    temperature: float = 0.7,
    top_p: float = 0.9,
    min_p: float = 0.05,
    repeat_penalty: float = 1.1,
    rng: np.random.Generator | None = None,
    stream: bool = False,
    stop_ids: set[int] | None = None,
):
    """Decode from raw token ids. A caller can thread one `kv_cache` across
    turns to keep prior context (append each turn's prompt_ids first)."""
    stops = stop_ids or set()
    if kv_cache is None:
        kv_cache = model.init_kv_cache()
    recent: list[int] = list(prompt_ids)

    logits = model.forward(prompt_ids, kv_cache, last_only=True)
    next_id = sample(logits[-1], temperature, top_p, min_p,
                     repeat_penalty=repeat_penalty, recent_ids=recent, rng=rng)

    def _loop() -> Iterator[int]:
        nonlocal next_id
        for _ in range(max_tokens):
            if next_id in stops:
                return
            yield next_id
            recent.append(next_id)
            logits = model.forward([next_id], kv_cache)
            next_id = sample(
                logits[0], temperature, top_p, min_p,
                repeat_penalty=repeat_penalty, recent_ids=recent, rng=rng)

    return _loop() if stream else list(_loop())
