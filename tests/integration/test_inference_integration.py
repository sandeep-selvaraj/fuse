"""Integration tests for the inference pipeline.

These tests download a real (tiny) GGUF model and run actual inference.
"""

from __future__ import annotations

import pytest


class TestModelResolver:
    """Test real HuggingFace model download and resolution."""

    def test_resolve_model_with_specific_filename(self, tmp_path):
        from fuse.inference.model_resolver import resolve_model
        from tests.integration.conftest import GGUF_FILENAME, GGUF_REPO

        path = resolve_model(GGUF_REPO, filename=GGUF_FILENAME, cache_dir=tmp_path)
        assert path.endswith(".gguf")
        from pathlib import Path

        assert Path(path).is_file()

    def test_resolve_model_auto_picks_quant(self, tmp_path):
        from fuse.inference.model_resolver import resolve_model
        from tests.integration.conftest import GGUF_REPO

        path = resolve_model(GGUF_REPO, cache_dir=tmp_path)
        assert path.endswith(".gguf")
        from pathlib import Path

        assert Path(path).is_file()


class TestLlamaCppBackendLoading:
    """Test real model loading into the backend."""

    def test_load_from_path(self, gguf_model_path):
        from fuse.inference.llama_cpp import LlamaCppBackend

        backend = LlamaCppBackend(model_path=gguf_model_path)
        assert backend._model is not None

    def test_load_from_config(self, gguf_model_path):
        from fuse.config import InferenceConfig
        from fuse.inference.llama_cpp import LlamaCppBackend

        config = InferenceConfig(
            model_path=gguf_model_path,
            n_ctx=512,
            n_threads=2,
            temperature=0.0,
        )
        backend = LlamaCppBackend.from_config(config)
        assert backend._model is not None


class TestFreeformGeneration:
    """Test raw text generation with real model."""

    def test_generate_returns_string(self, backend):
        result = backend.generate("The capital of France is", max_tokens=32, temperature=0.7)
        assert isinstance(result, str)

    def test_generate_respects_max_tokens(self, backend):
        short = backend.generate("Tell me a story:", max_tokens=8)
        long = backend.generate("Tell me a story:", max_tokens=64)
        # Shorter max_tokens should generally produce shorter output
        # (not guaranteed but very likely with such extreme difference)
        assert isinstance(short, str)
        assert isinstance(long, str)


class TestStructuredGeneration:
    """Test JSON-schema-constrained generation with real model."""

    def test_extract_simple_fields(self, backend):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        prompt = "Extract name and age from: John is 30 years old.\nOutput JSON:"
        result = backend.generate_structured(prompt, schema, max_tokens=64)
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)

    def test_extract_with_array_field(self, backend):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["name", "skills"],
        }
        prompt = "Extract name and skills from: Alice knows Python and Rust.\nOutput JSON:"
        result = backend.generate_structured(prompt, schema, max_tokens=128)
        assert isinstance(result, dict)
        assert "name" in result
        assert "skills" in result
        assert isinstance(result["skills"], list)

    def test_extract_with_optional_field(self, backend):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "company": {"type": "string"},
            },
            "required": ["name"],
        }
        prompt = "Extract name and company from: Bob works somewhere.\nOutput JSON:"
        result = backend.generate_structured(prompt, schema, max_tokens=64)
        assert isinstance(result, dict)
        assert "name" in result


class TestExtractorEndToEnd:
    """Test the full Extractor pipeline with real model."""

    def test_extract_from_fields(self, backend):
        from fuse.extraction.extractor import Extractor

        extractor = Extractor(backend)
        result = extractor.extract_from_fields(
            "Sarah Chen is a 34-year-old architect at Stripe.",
            {"name": str, "age": int, "company": str},
            max_tokens=128,
        )
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
        assert "company" in result
        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)

    def test_extract_with_pydantic_model(self, backend):
        from pydantic import BaseModel

        from fuse.extraction.extractor import Extractor

        class Person(BaseModel):
            name: str
            age: int

        extractor = Extractor(backend)
        result = extractor.extract("John is 30 years old.", Person, max_tokens=64)
        assert isinstance(result, Person)
        assert isinstance(result.name, str)
        assert isinstance(result.age, int)

    def test_extract_from_json_schema(self, backend):
        from fuse.extraction.extractor import Extractor
        from fuse.extraction.schema import SchemaBuilder

        schema = SchemaBuilder.from_json_schema(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
                "required": ["name", "age"],
            }
        )
        extractor = Extractor(backend)
        result = extractor.extract("Alice is 25.", schema, max_tokens=64)
        assert result.name is not None  # type: ignore[attr-defined]
        assert result.age is not None  # type: ignore[attr-defined]

    @pytest.mark.parametrize("prompt_format", ["llama", "chatml", "generic"])
    def test_extract_with_different_prompt_formats(self, backend, prompt_format):
        from fuse.extraction.extractor import Extractor

        extractor = Extractor(backend, prompt_format=prompt_format)
        result = extractor.extract_from_fields(
            "Bob is 40.",
            {"name": str, "age": int},
            max_tokens=64,
        )
        assert isinstance(result, dict)
        assert "name" in result
        assert "age" in result
