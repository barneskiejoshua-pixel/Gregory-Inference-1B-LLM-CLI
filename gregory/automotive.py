"""Automotive-engineering domain layer: persona + retrieval grounding.

Gregory wraps a *frozen* BitNet b1.58 model -- the weights cannot be retrained
here. This module specializes Gregory for automotive engineering the way that is
actually possible with frozen weights: a domain system prompt plus lightweight
keyword retrieval over a curated knowledge base (RAG). Relevant snippets are
injected into the prompt so answers are grounded in domain facts rather than the
2B model's unaided recall.

The knowledge base is every `*.md` file under gregory/data/. Each file holds
entries in the form:

    ## <topic> | tags: <comma,separated,keywords>
    <body text until the next "## ">

Extend the corpus by dropping more `.md` files in (OEM manual excerpts, DTC code
libraries, parts catalogs, standards summaries) -- no code change needed.

The taxonomy follows the Awesome-Automotive list's core subjects plus the
standards the field uses in practice (ISO 26262, AUTOSAR, CAN/LIN/FlexRay,
OBD-II, SAE J3016, ...).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "for",
    "on", "with", "by", "as", "at", "it", "that", "this", "from", "be",
    "how", "what", "why", "do", "does", "can", "i", "you", "my", "me",
}

SYSTEM_PROMPT = (
    "You are Gregory, an automotive-engineering assistant and a mechanic's and "
    "engineer's best friend. You help with vehicle design, powertrain, EV and "
    "hybrid systems, electronics and in-vehicle networks, diagnostics and "
    "fault codes, functional safety, materials, and emissions -- grounded in "
    "the underlying physics (mechanics, energy, thermodynamics, fluids, "
    "electricity and magnetism). Show the relevant formula and plug in numbers "
    "when a question is quantitative. When reference material is provided "
    "below, ground your answer in it and say so; if you are unsure or it does "
    "not cover the question, say so plainly rather than guessing. Flag any "
    "safety-critical step (high voltage, fuel, brakes, airbags, torque specs)."
)


@dataclass
class Entry:
    """One knowledge-base snippet: a topic, its tags, and the body text."""

    topic: str
    tags: list[str] = field(default_factory=list)
    text: str = ""
    source: str = ""

    def haystack(self) -> str:
        """Lowercased searchable text (topic + tags weighted by repetition)."""
        tagstr = " ".join(self.tags)
        weighted = f"{self.topic} {self.topic} {tagstr} {tagstr} {self.text}"
        return weighted.lower()


def _parse_file(path: Path) -> list[Entry]:
    """Parse one knowledge-base markdown file into entries."""
    entries: list[Entry] = []
    topic = None
    tags: list[str] = []
    body: list[str] = []

    def flush() -> None:
        """Append the entry being accumulated, if any."""
        if topic is not None:
            entries.append(Entry(topic=topic, tags=list(tags),
                                 text="\n".join(body).strip(),
                                 source=path.name))

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            flush()
            header = line[3:].strip()
            if "| tags:" in header:
                topic, raw = header.split("| tags:", 1)
                topic = topic.strip()
                tags = [t.strip() for t in raw.split(",") if t.strip()]
            else:
                topic, tags = header.strip(), []
            body = []
        elif topic is not None:
            body.append(line)
    flush()
    return entries


def load_kb() -> list[Entry]:
    """Load every entry from all `*.md` files under gregory/data/."""
    out: list[Entry] = []
    if not _DATA_DIR.is_dir():
        return out
    for path in sorted(_DATA_DIR.glob("*.md")):
        out.extend(_parse_file(path))
    return out


_KB_CACHE: list[Entry] | None = None


def kb() -> list[Entry]:
    """Return the knowledge base, loading and caching it on first use."""
    global _KB_CACHE
    if _KB_CACHE is None:
        _KB_CACHE = load_kb()
    return _KB_CACHE


def _tokens(text: str) -> list[str]:
    """Content tokens of `text` (lowercased, stopwords removed)."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def keywords(text: str, n: int = 8) -> list[str]:
    """Return up to `n` salient content words of `text`, most frequent first.

    Used by the ingestion tool to auto-tag chunks; short tokens (len < 3) are
    skipped so tags are meaningful."""
    counts: dict[str, int] = {}
    for tok in _tokens(text):
        if len(tok) >= 3:
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts, key=lambda w: counts[w], reverse=True)
    return ranked[:n]


def retrieve(query: str, k: int = 3) -> list[Entry]:
    """Return the `k` knowledge-base entries most relevant to `query`.

    Scores each entry by how many query content-tokens it contains (topic and
    tags count double via haystack repetition). Entries with no overlap are
    dropped, so an off-topic query returns fewer than `k` (possibly none)."""
    q = set(_tokens(query))
    if not q:
        return []
    scored: list[tuple[int, Entry]] = []
    for entry in kb():
        hay = entry.haystack()
        score = sum(hay.count(tok) for tok in q)
        if score > 0:
            scored.append((score, entry))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [entry for _, entry in scored[:k]]


def build_context(query: str, k: int = 3) -> str:
    """Format retrieved grounding for `query` as a prompt block, or ''.

    Returns an empty string when nothing relevant is found, so the caller can
    skip injecting context entirely."""
    hits = retrieve(query, k)
    if not hits:
        return ""
    lines = ["Reference material (automotive engineering knowledge base):"]
    for entry in hits:
        lines.append(f"- [{entry.topic}] {entry.text}")
    return "\n".join(lines)


def topics() -> list[str]:
    """Return the list of knowledge-base topic names."""
    return [entry.topic for entry in kb()]
