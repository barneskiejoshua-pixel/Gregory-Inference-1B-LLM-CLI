"""GPT-2-style BPE tokenizer built from the vocab + merges in the GGUF.

The BitNet b1.58 family ships the LLaMA-3 tokenizer: GPT-2-style byte-level BPE
with LLaMA-3's vocab, merges, and the standard GPT-2 byte<->unicode permutation.

The pre-tokenizer regex uses Unicode classes (\\p{L}, \\p{N}). Those need the
third-party `regex` module; when it is absent we fall back to a coarser stdlib
`re` pattern so the package stays importable with numpy alone (the fallback
splits slightly differently but still round-trips bytes correctly).

Public API:
    Tokenizer.from_gguf(g)        # build from a loaded gregory.gguf.GGUF
    tok.encode(text) -> list[int]
    tok.decode(ids)  -> str
    tok.bos_id, tok.eos_id, tok.eog_ids
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import regex as _re                       # PCRE-style: supports \p{L}/\p{N}
    _PRE_TOK = _re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\p{L}\p{N}]?\p{L}+"
        r"|\p{N}{1,3}"
        r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+",
    )
except ImportError:                           # numpy-only fallback
    import re as _re
    _PRE_TOK = _re.compile(
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
        r"|[^\r\n\w]?\w+"
        r"|\d{1,3}"
        r"| ?[^\s\w]+[\r\n]*"
        r"|\s*[\r\n]+"
        r"|\s+(?!\S)"
        r"|\s+",
        _re.UNICODE,
    )


def _bytes_to_unicode() -> dict[int, str]:
    """Map every byte 0..255 to a printable unicode char (the GPT-2 trick), so
    BPE can run in unicode space while staying byte-reversible."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(ord("¡"), ord("¬") + 1))
          + list(range(ord("®"), ord("ÿ") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


_BYTE_TO_UNI = _bytes_to_unicode()
_UNI_TO_BYTE = {v: k for k, v in _BYTE_TO_UNI.items()}


@dataclass
class Tokenizer:
    """Byte-level BPE tokenizer: vocab + merges + special-token ids."""

    vocab: dict[str, int]
    id_to_token: list[str]
    merges: dict[tuple[str, str], int]
    bos_id: int
    eos_id: int
    eot_id: int | None
    eog_ids: set[int]

    @classmethod
    def from_gguf(cls, g) -> "Tokenizer":
        """Build a Tokenizer from a loaded gregory.gguf.GGUF."""
        tokens = g.kv["tokenizer.ggml.tokens"]
        merges_lines = g.kv["tokenizer.ggml.merges"]
        bos = int(g.kv.get("tokenizer.ggml.bos_token_id", 128000))
        eos = int(g.kv.get("tokenizer.ggml.eos_token_id", 128001))
        vocab = {t: i for i, t in enumerate(tokens)}
        # LLaMA-3 chat models end turns with <|eot_id|>, not <|end_of_text|>.
        eot = vocab.get("<|eot_id|>")
        eog = {eos}
        if eot is not None:
            eog.add(eot)
        merges: dict[tuple[str, str], int] = {}
        for i, line in enumerate(merges_lines):
            try:
                a, b = line.split(" ", 1)
            except ValueError:
                continue
            merges[(a, b)] = i
        return cls(vocab=vocab, id_to_token=list(tokens), merges=merges,
                   bos_id=bos, eos_id=eos, eot_id=eot, eog_ids=eog)

    def _bpe_word(self, word: str) -> list[str]:
        """Apply BPE merges to one unicode-mapped word; return its pieces."""
        if not word:
            return []
        if len(word) == 1:
            return [word]
        toks = list(word)
        # Each iteration merges one adjacent pair, shrinking toks by one, so the
        # initial length is a hard upper bound on iterations (bounded loop).
        for _ in range(len(toks)):
            best_pair = None
            best_rank = None
            for i in range(len(toks) - 1):
                pair = (toks[i], toks[i + 1])
                rank = self.merges.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = (i, pair)
            if best_pair is None:
                break
            i, (a, b) = best_pair
            toks = toks[:i] + [a + b] + toks[i + 2:]
        return toks

    def encode(self, text: str, add_bos: bool = True) -> list[int]:
        """Encode `text` to token ids, optionally prefixing the BOS id."""
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in _PRE_TOK.findall(text):
            uni = "".join(_BYTE_TO_UNI[b] for b in chunk.encode("utf-8"))
            for piece in self._bpe_word(uni):
                idx = self.vocab.get(piece)
                if idx is None:
                    for ch in piece:
                        single = self.vocab.get(ch)
                        if single is not None:
                            ids.append(single)
                else:
                    ids.append(idx)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token ids back to text via the byte<->unicode permutation."""
        pieces = []
        for i in ids:
            if 0 <= i < len(self.id_to_token):
                pieces.append(self.id_to_token[i])
        text_uni = "".join(pieces)
        raw = bytes(_UNI_TO_BYTE[c] for c in text_uni if c in _UNI_TO_BYTE)
        return raw.decode("utf-8", errors="replace")
