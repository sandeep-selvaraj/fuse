from typing import Any

import pytest
from pydantic import BaseModel

from fuse.extraction.extractor import Extractor, _to_evidenced_json_schema
from fuse.extraction.prompts import format_evidenced_extraction_prompt
from fuse.extraction.spans import (
    EvidencedField,
    Span,
    SpannedResult,
    build_spanned_result,
    locate_all_spans,
    locate_span,
)

# --- Span localization tests ---


class TestLocateSpan:
    def test_exact_match(self) -> None:
        span = locate_span("John Smith is 30 years old", "John Smith")
        assert span == Span(start=0, end=10)

    def test_mid_text_match(self) -> None:
        span = locate_span("Hello, John Smith is here", "John Smith")
        assert span == Span(start=7, end=17)

    def test_no_match(self) -> None:
        assert locate_span("Hello world", "xyz") is None

    def test_empty_needle(self) -> None:
        assert locate_span("Hello world", "") is None

    def test_returns_first_occurrence(self) -> None:
        span = locate_span("cat and cat", "cat")
        assert span == Span(start=0, end=3)


class TestLocateAllSpans:
    def test_multiple_occurrences(self) -> None:
        spans = locate_all_spans("cat and cat and cat", "cat")
        assert spans == [Span(0, 3), Span(8, 11), Span(16, 19)]

    def test_no_matches(self) -> None:
        assert locate_all_spans("hello", "xyz") == []

    def test_empty_needle(self) -> None:
        assert locate_all_spans("hello", "") == []

    def test_single_match(self) -> None:
        spans = locate_all_spans("hello world", "world")
        assert spans == [Span(6, 11)]


class TestSpan:
    def test_length(self) -> None:
        assert Span(5, 15).length == 10


# --- SpannedResult tests ---


class TestSpannedResult:
    def test_to_dict(self) -> None:
        result = SpannedResult(
            fields=[
                EvidencedField("name", "John", "John", True, Span(0, 4)),
                EvidencedField("sentiment", "positive", "loved it", False, Span(10, 18)),
            ]
        )
        assert result.to_dict() == {"name": "John", "sentiment": "positive"}

    def test_getitem(self) -> None:
        field = EvidencedField("name", "John", "John", True, Span(0, 4))
        result = SpannedResult(fields=[field])
        assert result["name"] is field

    def test_getitem_missing(self) -> None:
        result = SpannedResult(fields=[])
        with pytest.raises(KeyError):
            result["missing"]


# --- build_spanned_result tests ---


class TestBuildSpannedResult:
    def test_explicit_field(self) -> None:
        source = "John Smith is a software engineer"
        raw = {
            "name": {
                "value": "John Smith",
                "evidence": "John Smith",
                "is_explicit": True,
            }
        }
        result = build_spanned_result(source, raw)
        field = result["name"]
        assert field.value == "John Smith"
        assert field.is_explicit is True
        assert field.span == Span(0, 10)

    def test_implicit_field(self) -> None:
        source = "the product broke after two days and support never responded"
        raw = {
            "sentiment": {
                "value": "negative",
                "evidence": "broke after two days and support never responded",
                "is_explicit": False,
            }
        }
        result = build_spanned_result(source, raw)
        field = result["sentiment"]
        assert field.value == "negative"
        assert field.is_explicit is False
        assert field.span == Span(12, 60)
        assert source[field.span.start : field.span.end] == field.evidence

    def test_missing_evidence(self) -> None:
        raw = {"x": {"value": None, "evidence": "", "is_explicit": False}}
        result = build_spanned_result("some text", raw)
        assert result["x"].span is None

    def test_evidence_not_found(self) -> None:
        raw = {"x": {"value": "foo", "evidence": "not in source", "is_explicit": False}}
        result = build_spanned_result("some text", raw)
        assert result["x"].span is None

    def test_non_evidenced_entry_handled(self) -> None:
        raw = {"x": "plain_value"}
        result = build_spanned_result("some text", raw)
        assert result["x"].value == "plain_value"
        assert result["x"].span is None


# --- Evidenced JSON schema wrapping ---


class TestToEvidencedJsonSchema:
    def test_wraps_properties(self) -> None:
        original = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        evidenced = _to_evidenced_json_schema(original)

        assert "name" in evidenced["properties"]
        name_prop = evidenced["properties"]["name"]
        assert name_prop["type"] == "object"
        assert name_prop["properties"]["value"] == {"type": "string"}
        assert name_prop["properties"]["evidence"] == {"type": "string"}
        assert name_prop["properties"]["is_explicit"] == {"type": "boolean"}
        assert set(name_prop["required"]) == {"value", "evidence", "is_explicit"}

    def test_all_fields_required(self) -> None:
        original = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        }
        evidenced = _to_evidenced_json_schema(original)
        assert set(evidenced["required"]) == {"a", "b"}


# --- Evidenced prompt tests ---


class TestEvidencedPromptFormatting:
    def test_llama_format(self) -> None:
        prompt = format_evidenced_extraction_prompt("some text", "- name (string)", "llama")
        assert "<|start_header_id|>system<|end_header_id|>" in prompt
        assert "verbatim evidence" in prompt.lower()
        assert "some text" in prompt

    def test_chatml_format(self) -> None:
        prompt = format_evidenced_extraction_prompt("some text", "- name (string)", "chatml")
        assert "<|im_start|>" in prompt
        assert "verbatim" in prompt.lower()

    def test_generic_format(self) -> None:
        prompt = format_evidenced_extraction_prompt("some text", "- name (string)", "generic")
        assert "<|system|>" in prompt


# --- Extractor integration with mock backend ---


class _EvidencedMockBackend:
    """Mock backend that returns evidenced extraction results."""

    def __init__(self, source_text: str) -> None:
        self._source = source_text

    def load(self, model_path: str, **kwargs: Any) -> None:
        pass

    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs: Any) -> str:
        return ""

    def generate_structured(
        self, prompt: str, json_schema: dict[str, Any], *, max_tokens: int = 512, **kwargs: Any
    ) -> dict[str, Any]:
        props = json_schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, prop in props.items():
            if prop.get("type") == "object" and "evidence" in prop.get("properties", {}):
                # Evidenced field — return mock evidenced response
                value_type = prop["properties"]["value"].get("type", "string")
                if value_type == "string":
                    result[key] = {
                        "value": "John Smith",
                        "evidence": "John Smith",
                        "is_explicit": True,
                    }
                elif value_type == "integer":
                    result[key] = {
                        "value": 30,
                        "evidence": "30 years old",
                        "is_explicit": True,
                    }
                else:
                    result[key] = {
                        "value": None,
                        "evidence": "",
                        "is_explicit": False,
                    }
            else:
                result[key] = None
        return result


class TestExtractorWithSpans:
    def test_extract_with_spans(self) -> None:
        class Person(BaseModel):
            name: str
            age: int

        source = "John Smith is 30 years old"
        extractor = Extractor(_EvidencedMockBackend(source))
        result = extractor.extract_with_spans(source, Person)

        assert isinstance(result, SpannedResult)
        name_field = result["name"]
        assert name_field.value == "John Smith"
        assert name_field.is_explicit is True
        assert name_field.span == Span(0, 10)

        age_field = result["age"]
        assert age_field.value == 30
        assert age_field.evidence == "30 years old"
        assert age_field.span == Span(14, 26)

    def test_extract_from_fields_with_spans(self) -> None:
        source = "John Smith is 30 years old"
        extractor = Extractor(_EvidencedMockBackend(source))
        result = extractor.extract_from_fields_with_spans(source, {"name": str, "age": int})

        assert isinstance(result, SpannedResult)
        assert result.to_dict() == {"name": "John Smith", "age": 30}
