"""Unit tests for the BPE tokenizer (synthetic vocab, model-free)."""

from __future__ import annotations

from gregory.tokenizer import Tokenizer, _bytes_to_unicode


class _FakeGGUF:
    """Minimal stand-in exposing the `kv` dict Tokenizer.from_gguf reads."""

    def __init__(self, kv):
        self.kv = kv


def _byte_vocab_tokenizer() -> Tokenizer:
    """Build a tokenizer whose vocab is exactly the 256 byte-unicode chars."""
    b2u = _bytes_to_unicode()
    tokens = [b2u[b] for b in range(256)]
    kv = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": [],
        "tokenizer.ggml.bos_token_id": 0,
        "tokenizer.ggml.eos_token_id": 1,
    }
    return Tokenizer.from_gguf(_FakeGGUF(kv))


def test_roundtrip_ascii():
    """With a full byte vocab and no merges, encode/decode round-trips."""
    tok = _byte_vocab_tokenizer()
    text = "hello, gregory!"
    ids = tok.encode(text, add_bos=False)
    assert tok.decode(ids) == text


def test_bos_prefix():
    """encode(add_bos=True) prepends the BOS id."""
    tok = _byte_vocab_tokenizer()
    ids = tok.encode("x", add_bos=True)
    assert ids[0] == tok.bos_id


def test_byte_unicode_is_bijective():
    """The GPT-2 byte<->unicode map covers all 256 bytes uniquely."""
    b2u = _bytes_to_unicode()
    assert len(b2u) == 256
    assert len(set(b2u.values())) == 256
