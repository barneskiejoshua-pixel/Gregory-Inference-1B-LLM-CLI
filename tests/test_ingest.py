"""Tests for the doc ingestion pipeline (model-free).

Loads scripts/ingest_docs.py as a module and checks that its chunkers produce
entries which round-trip through the KB parser the runtime actually uses.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from gregory import automotive

ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "ingest_docs", ROOT / "scripts" / "ingest_docs.py")
ingest = importlib.util.module_from_spec(_SPEC)
sys.modules["ingest_docs"] = ingest
_SPEC.loader.exec_module(ingest)


def test_text_chunks_roundtrip(tmp_path):
    """Chunked text renders to KB markdown that the loader parses back."""
    text = ("# Brake Bleeding\n\n"
            "Bleed the brakes starting from the wheel farthest from the master "
            "cylinder. Keep the reservoir topped up.\n\n"
            "Torque the caliper bolts to the manufacturer spec.\n")
    chunks = ingest.chunks_from_text(text, "Brakes", max_words=10,
                                     extra_tags=["brakes"])
    assert chunks
    out = tmp_path / "brakes.md"
    out.write_text(ingest.render(chunks, "test"))
    entries = automotive._parse_file(out)
    assert len(entries) == len(chunks)
    assert any("brake" in e.text.lower() for e in entries)
    assert all("brakes" in e.tags for e in entries)


def test_csv_chunks_one_per_row(tmp_path):
    """CSV ingestion makes one entry per row with topic/body from columns."""
    csv_path = tmp_path / "dtc.csv"
    csv_path.write_text(
        "code,description\n"
        "P0301,Cylinder 1 misfire detected\n"
        "P0420,Catalyst system efficiency below threshold\n")
    chunks = ingest.chunks_from_csv(csv_path, "code", "description", None, [])
    assert len(chunks) == 2
    assert chunks[0].topic == "P0301"
    assert "misfire" in chunks[0].text.lower()


def test_body_cannot_inject_header(tmp_path):
    """A body line starting with '## ' is neutralized so it is not a new entry."""
    text = "Normal line.\n## Not A Real Header\nmore text.\n"
    chunks = ingest.chunks_from_text(text, "Doc", max_words=100, extra_tags=[])
    out = tmp_path / "x.md"
    out.write_text(ingest.render(chunks, "t"))
    entries = automotive._parse_file(out)
    assert len(entries) == len(chunks)   # the body '##' did not split entries
