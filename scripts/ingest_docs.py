#!/usr/bin/env python3
"""Ingest documents into Gregory's automotive knowledge base.

Converts plain-text, markdown, and CSV sources into KB entries
(`## <topic> | tags: ...` blocks) that Gregory retrieves over at answer time.
This is the realistic "training" step for a frozen-weight RAG model: grow the
grounding corpus. It produces a *draft* KB file -- review it before relying on
it (chunking and auto-tags are heuristic).

Supported inputs (dep-light, stdlib only):
  .txt / .md  -> chunked by blank-line paragraphs into ~--max-words blocks;
                 markdown headings (#, ##, ...) become entry topics.
  .csv        -> one entry per row; pick the topic/body/tags columns. Ideal for
                 DTC/OBD code tables and parts catalogs.

Examples:
  # a manual or notes file
  python3 scripts/ingest_docs.py manual.md --out brakes --topic "Brake System"

  # a DTC code table (CSV with columns: code, description)
  python3 scripts/ingest_docs.py dtc.csv --out obd_codes \\
      --csv-topic-col code --csv-body-col description

  # a whole folder
  python3 scripts/ingest_docs.py ./docs --out manuals

Output goes to gregory/data/<out>.md, which the KB loader picks up
automatically. Use --append to add to an existing file, --force to overwrite,
--stdout to preview without writing.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gregory import automotive  # noqa: E402  (after sys.path setup)

DATA_DIR = ROOT / "gregory" / "data"
TEXT_EXTS = {".txt", ".md", ".markdown", ".text"}


@dataclass
class Chunk:
    """A draft KB entry produced by ingestion."""

    topic: str
    tags: list[str]
    text: str


def _clean_body(text: str) -> str:
    """Collapse whitespace and neutralize lines that would break the parser."""
    out_lines = []
    for line in text.splitlines():
        line = line.rstrip()
        if line.startswith("## "):      # would be read as a new entry header
            line = " " + line
        out_lines.append(line)
    return "\n".join(out_lines).strip()


def _paragraphs(text: str):
    """Yield (heading, paragraph) pairs; heading carries the nearest #-title."""
    heading = ""
    buf: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if buf:
                yield heading, "\n".join(buf).strip()
                buf = []
            heading = stripped.lstrip("#").strip()
        elif not stripped:
            if buf:
                yield heading, "\n".join(buf).strip()
                buf = []
        else:
            buf.append(line)
    if buf:
        yield heading, "\n".join(buf).strip()


def chunks_from_text(text: str, base_topic: str, max_words: int,
                     extra_tags: list[str]) -> list[Chunk]:
    """Chunk free text into ~max_words entries, titled by heading or base."""
    chunks: list[Chunk] = []
    cur: list[str] = []
    cur_words = 0
    cur_head = ""
    idx = 1

    def emit() -> None:
        nonlocal cur, cur_words, idx
        if not cur:
            return
        body = _clean_body("\n\n".join(cur))
        if not body:
            cur, cur_words = [], 0
            return
        topic = cur_head or f"{base_topic} {idx}"
        tags = sorted(set(extra_tags) | set(automotive.keywords(body, 8)))
        chunks.append(Chunk(topic=topic, tags=tags, text=body))
        idx += 1
        cur, cur_words = [], 0

    for head, para in _paragraphs(text):
        if head and head != cur_head and cur:
            emit()
        cur_head = head or cur_head
        cur.append(para)
        cur_words += len(para.split())
        if cur_words >= max_words:
            emit()
    emit()
    return chunks


def chunks_from_csv(path: Path, topic_col: str, body_col: str,
                    tags_col: str | None,
                    extra_tags: list[str]) -> list[Chunk]:
    """One entry per CSV row using the named topic/body/tags columns."""
    chunks: list[Chunk] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            body = (row.get(body_col) or "").strip()
            topic = (row.get(topic_col) or "").strip()
            if not topic and not body:
                continue
            row_tags = list(extra_tags)
            if tags_col and row.get(tags_col):
                row_tags += [t.strip() for t in row[tags_col].split(",")
                             if t.strip()]
            if not row_tags:
                row_tags = automotive.keywords(f"{topic} {body}", 6)
            chunks.append(Chunk(topic=topic or "entry",
                                tags=sorted(set(row_tags)),
                                text=_clean_body(body)))
    return chunks


def render(chunks: list[Chunk], source: str) -> str:
    """Render chunks to the KB markdown format."""
    lines = [f"# Ingested from {source} -- DRAFT, review before relying on it",
             ""]
    for ch in chunks:
        lines.append(f"## {ch.topic} | tags: {', '.join(ch.tags)}")
        lines.append(ch.text)
        lines.append("")
    return "\n".join(lines)


def _gather_inputs(paths: list[str]) -> list[Path]:
    """Expand files and directories into a flat list of readable inputs."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += [q for q in sorted(p.rglob("*"))
                    if q.suffix.lower() in TEXT_EXTS or q.suffix.lower() == ".csv"]
        elif p.is_file():
            out.append(p)
        else:
            print(f"skip (not found): {p}", file=sys.stderr)
    return out


def ingest(inputs: list[Path], args: argparse.Namespace) -> list[Chunk]:
    """Run the right chunker per input and return all chunks."""
    chunks: list[Chunk] = []
    extra = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
    for path in inputs:
        ext = path.suffix.lower()
        if ext == ".csv":
            chunks += chunks_from_csv(path, args.csv_topic_col,
                                      args.csv_body_col, args.csv_tags_col,
                                      extra)
        elif ext in TEXT_EXTS:
            base = args.topic or path.stem.replace("_", " ").title()
            text = path.read_text(encoding="utf-8", errors="replace")
            chunks += chunks_from_text(text, base, args.max_words, extra)
        else:
            print(f"skip (unsupported {ext}): {path}", file=sys.stderr)
    return chunks


def main() -> int:
    """Parse args, ingest the inputs, and write/preview the KB file."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="files or directories to ingest")
    ap.add_argument("--out", help="output KB name (gregory/data/<out>.md)")
    ap.add_argument("--topic", help="base topic label for text inputs")
    ap.add_argument("--tags", help="comma-separated tags added to every entry")
    ap.add_argument("--max-words", type=int, default=120, dest="max_words",
                    help="approx words per text chunk (default 120)")
    ap.add_argument("--csv-topic-col", default="code", dest="csv_topic_col")
    ap.add_argument("--csv-body-col", default="description", dest="csv_body_col")
    ap.add_argument("--csv-tags-col", default=None, dest="csv_tags_col")
    ap.add_argument("--append", action="store_true", help="append to existing")
    ap.add_argument("--force", action="store_true", help="overwrite existing")
    ap.add_argument("--stdout", action="store_true", help="print, don't write")
    args = ap.parse_args()

    inputs = _gather_inputs(args.inputs)
    if not inputs:
        print("no readable inputs", file=sys.stderr)
        return 1
    chunks = ingest(inputs, args)
    if not chunks:
        print("no entries produced", file=sys.stderr)
        return 1

    source = ", ".join(p.name for p in inputs[:4])
    if len(inputs) > 4:
        source += f", +{len(inputs) - 4} more"
    body = render(chunks, source)

    if args.stdout:
        print(body)
        print(f"\n# {len(chunks)} entries (preview only)", file=sys.stderr)
        return 0

    if not args.out:
        print("--out is required (or use --stdout)", file=sys.stderr)
        return 1
    out_path = DATA_DIR / f"{args.out}.md"
    if out_path.exists() and not (args.append or args.force):
        print(f"{out_path} exists; use --append or --force", file=sys.stderr)
        return 1
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if args.append and out_path.exists():
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write("\n" + body)
    else:
        out_path.write_text(body, encoding="utf-8")
    print(f"wrote {len(chunks)} entries -> {out_path}")
    print("review it, then: gregory kb   (and ask Gregory)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
