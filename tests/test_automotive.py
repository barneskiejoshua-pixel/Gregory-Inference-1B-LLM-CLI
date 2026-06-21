"""Tests for the automotive domain layer (model-free)."""

from __future__ import annotations

from gregory import automotive


def test_kb_loads_entries():
    """The knowledge base parses into a non-trivial set of entries."""
    entries = automotive.kb()
    assert len(entries) >= 20
    assert all(e.topic and e.text for e in entries)


def test_retrieve_is_relevant():
    """A domain query surfaces the on-topic entry first."""
    hits = automotive.retrieve("What ASIL levels does ISO 26262 define?", k=3)
    assert hits
    joined = " ".join(h.topic.lower() for h in hits)
    assert "safety" in joined or "26262" in " ".join(
        " ".join(h.tags).lower() for h in hits)


def test_retrieve_can_match_can_bus():
    """A CAN-bus question retrieves the in-vehicle-networks entry."""
    hits = automotive.retrieve("how does CAN bus arbitration work", k=2)
    assert any("CAN" in h.topic or "can" in " ".join(h.tags).lower()
               for h in hits)


def test_off_topic_query_returns_nothing():
    """An unrelated query yields no grounding (so none is injected)."""
    assert automotive.build_context("photosynthesis in ferns") == ""


def test_build_context_format():
    """build_context returns a labelled block listing retrieved topics."""
    ctx = automotive.build_context("regenerative braking on an EV", k=2)
    assert ctx.startswith("Reference material")
    assert "[" in ctx and "]" in ctx


def test_physics_concepts_present():
    """PHY 1/2 concepts are in the KB and retrievable."""
    hits = automotive.retrieve("how do I compute torque from force and radius")
    assert any("torque" in h.topic.lower() or "torque" in " ".join(h.tags)
               for h in hits)
    ohm = automotive.retrieve("Ohm's law voltage current resistance")
    assert any("ohm" in " ".join(h.tags).lower() or "Ohm" in h.topic
               for h in ohm)
    energy = automotive.retrieve("kinetic energy a car must dissipate braking")
    assert energy
