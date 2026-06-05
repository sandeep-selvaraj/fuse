"""Span localization for extracted fields.

Pairs each extracted value with character offsets in the source text,
distinguishing between explicit extractions (value is verbatim in text)
and implicit extractions (value is inferred, evidence passage is verbatim).
"""

from __future__ import annotations

import json
import math
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
    confidence: float | None = None
    """Geometric mean of the token probabilities for this field's value
    (0-1), or None if logprobs were unavailable. See `char_range_confidence`."""


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


@dataclass(frozen=True, slots=True)
class TokenScores:
    """Per-token decoding info for the generated JSON, used to score confidence.

    Attributes:
        text: The raw generated JSON string.
        tokens: The decoded tokens, in order; "".join(tokens) == text.
        logprobs: Natural-log probability of each chosen token (same length
            as tokens). Entries may be None if a logprob was unavailable.
    """

    text: str
    tokens: list[str]
    logprobs: list[float | None]


def _token_offsets(tokens: list[str]) -> list[int]:
    """Character start offset of each token within "".join(tokens)."""
    offsets: list[int] = []
    pos = 0
    for tok in tokens:
        offsets.append(pos)
        pos += len(tok)
    return offsets


def char_range_confidence(scores: TokenScores, start: int, end: int) -> float | None:
    """Confidence for the value occupying [start, end) in the generated text.

    Selects every token that overlaps the character range and returns the
    geometric mean of those token probabilities, i.e. exp(mean(logprobs)).
    Geometric mean is length-normalized, so longer values are not penalized.
    Returns None if no scored token overlaps the range.
    """
    offsets = _token_offsets(scores.tokens)
    selected = [
        lp
        for off, tok, lp in zip(offsets, scores.tokens, scores.logprobs, strict=False)
        if lp is not None and off < end and off + len(tok) > start
    ]
    if not selected:
        return None
    return math.exp(sum(selected) / len(selected))


def _locate_evidenced_value_range(
    json_text: str, field_name: str, value: Any
) -> tuple[int, int] | None:
    """Find the char range of a field's `value` literal in the evidenced JSON.

    Anchors on the field key, then the nested "value" key, then the JSON
    literal of the value — so a value that also appears inside the evidence
    string is not matched by mistake. Returns None if any anchor is missing.
    """
    ki = json_text.find(f'"{field_name}"')
    if ki == -1:
        return None
    vk = json_text.find('"value"', ki)
    if vk == -1:
        return None
    literal = json.dumps(value, ensure_ascii=False)
    vi = json_text.find(literal, vk)
    if vi == -1:
        return None
    return vi, vi + len(literal)


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
    scores: TokenScores | None = None,
) -> SpannedResult:
    """Convert a raw evidenced extraction dict into a SpannedResult.

    Expects raw_extraction to have the evidenced schema shape:
    {"field_name": {"value": ..., "evidence": "...", "is_explicit": ...}, ...}

    If `scores` is provided, each field is annotated with a `confidence`
    derived from the token logprobs of its value (see char_range_confidence).
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

        confidence: float | None = None
        if scores is not None:
            value_range = _locate_evidenced_value_range(scores.text, name, value)
            if value_range is not None:
                confidence = char_range_confidence(scores, *value_range)

        fields.append(
            EvidencedField(
                name=name,
                value=value,
                evidence=evidence,
                is_explicit=is_explicit,
                span=span,
                confidence=confidence,
            )
        )

    return SpannedResult(fields=fields)
