"""Span localization for extracted fields.

Pairs each extracted value with character offsets in the source text,
distinguishing between explicit extractions (value is verbatim in text)
and implicit extractions (value is inferred, evidence passage is verbatim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Span:
    """A character-offset span in the source text."""

    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class EvidencedField:
    """An extracted field with its evidence and source span.

    Attributes:
        name: Field name from the schema.
        value: The extracted value (may be inferred, not verbatim).
        evidence: Verbatim quote from the source text supporting the value.
        is_explicit: True if the value itself appears verbatim in the source.
        span: Character-offset span of the evidence in the source text,
              or None if localization failed.
    """

    name: str
    value: Any
    evidence: str
    is_explicit: bool
    span: Span | None


@dataclass(frozen=True, slots=True)
class SpannedResult:
    """Complete extraction result with per-field spans."""

    fields: list[EvidencedField]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict of field name to value."""
        return {f.name: f.value for f in self.fields}

    def __getitem__(self, field_name: str) -> EvidencedField:
        for f in self.fields:
            if f.name == field_name:
                return f
        raise KeyError(field_name)


def locate_span(source: str, needle: str) -> Span | None:
    """Find the first exact occurrence of needle in source.

    Returns None if needle is empty or not found.
    """
    if not needle:
        return None
    idx = source.find(needle)
    if idx == -1:
        return None
    return Span(start=idx, end=idx + len(needle))


def locate_all_spans(source: str, needle: str) -> list[Span]:
    """Find all non-overlapping exact occurrences of needle in source."""
    if not needle:
        return []
    spans: list[Span] = []
    start = 0
    while True:
        idx = source.find(needle, start)
        if idx == -1:
            break
        spans.append(Span(start=idx, end=idx + len(needle)))
        start = idx + len(needle)
    return spans


def build_spanned_result(
    source: str,
    raw_extraction: dict[str, Any],
) -> SpannedResult:
    """Convert a raw evidenced extraction dict into a SpannedResult.

    Expects raw_extraction to have the evidenced schema shape:
    {"field_name": {"value": ..., "evidence": "...", "is_explicit": ...}, ...}
    """
    fields: list[EvidencedField] = []
    for name, entry in raw_extraction.items():
        if not isinstance(entry, dict) or "value" not in entry:
            # Non-evidenced field (shouldn't happen, but handle gracefully)
            fields.append(
                EvidencedField(name=name, value=entry, evidence="", is_explicit=False, span=None)
            )
            continue

        value = entry["value"]
        evidence = entry.get("evidence", "")
        is_explicit = entry.get("is_explicit", False)

        if is_explicit and isinstance(value, str) and value:
            span = locate_span(source, value)
        elif evidence:
            span = locate_span(source, evidence)
        else:
            span = None

        fields.append(
            EvidencedField(
                name=name,
                value=value,
                evidence=evidence,
                is_explicit=is_explicit,
                span=span,
            )
        )

    return SpannedResult(fields=fields)
