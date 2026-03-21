from typing import Any

from pydantic import BaseModel

from fuse.extraction.extractor import Extractor, _schema_to_description
from fuse.extraction.prompts import format_extraction_prompt
from fuse.extraction.schema import SchemaBuilder

# --- Schema Builder Tests ---


class TestSchemaBuilderFromFields:
    def test_basic_types(self) -> None:
        model = SchemaBuilder.from_fields({"name": str, "age": int, "active": bool})
        instance = model(name="Alice", age=30, active=True)
        assert instance.name == "Alice"  # type: ignore[attr-defined]
        assert instance.age == 30  # type: ignore[attr-defined]

    def test_list_field(self) -> None:
        model = SchemaBuilder.from_fields({"tags": list[str]})
        instance = model(tags=["a", "b"])
        assert instance.tags == ["a", "b"]  # type: ignore[attr-defined]

    def test_with_defaults(self) -> None:
        model = SchemaBuilder.from_fields(
            {
                "name": str,
                "score": (float, 0.0),
            }
        )
        instance = model(name="test")
        assert instance.score == 0.0  # type: ignore[attr-defined]

    def test_custom_model_name(self) -> None:
        model = SchemaBuilder.from_fields({"x": int}, model_name="MyModel")
        assert model.__name__ == "MyModel"


class TestSchemaBuilderFromJsonSchema:
    def test_basic_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        model = SchemaBuilder.from_json_schema(schema)
        instance = model(name="Bob", age=25)
        assert instance.name == "Bob"  # type: ignore[attr-defined]
        assert instance.age == 25  # type: ignore[attr-defined]

    def test_optional_fields(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "bio": {"type": "string"},
            },
            "required": ["name"],
        }
        model = SchemaBuilder.from_json_schema(schema)
        instance = model(name="Bob")
        assert instance.bio is None  # type: ignore[attr-defined]

    def test_array_field(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["skills"],
        }
        model = SchemaBuilder.from_json_schema(schema)
        instance = model(skills=["python", "rust"])
        assert instance.skills == ["python", "rust"]  # type: ignore[attr-defined]

    def test_to_json_schema_roundtrip(self) -> None:
        model = SchemaBuilder.from_fields({"name": str, "age": int})
        json_schema = SchemaBuilder.to_json_schema(model)
        assert "properties" in json_schema
        assert "name" in json_schema["properties"]


# --- Prompt Tests ---


class TestPromptFormatting:
    def test_llama_format(self) -> None:
        prompt = format_extraction_prompt("some text", "- name (string)", "llama")
        assert "<|begin_of_text|>" in prompt
        assert "some text" in prompt

    def test_chatml_format(self) -> None:
        prompt = format_extraction_prompt("some text", "- name (string)", "chatml")
        assert "<|im_start|>" in prompt

    def test_generic_format(self) -> None:
        prompt = format_extraction_prompt("some text", "- name (string)", "generic")
        assert "<|system|>" in prompt


# --- Extractor Tests (with mock backend) ---


class _MockBackend:
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
            t = prop.get("type", "string")
            if t == "string":
                result[key] = "extracted"
            elif t == "integer":
                result[key] = 42
            elif t == "array":
                result[key] = ["item1"]
            else:
                result[key] = None
        return result


class TestExtractor:
    def test_extract_with_pydantic_model(self) -> None:
        class Person(BaseModel):
            name: str
            age: int

        extractor = Extractor(_MockBackend())
        result = extractor.extract("John is 30", Person)
        assert isinstance(result, Person)
        assert result.name == "extracted"
        assert result.age == 42

    def test_extract_from_fields(self) -> None:
        extractor = Extractor(_MockBackend())
        result = extractor.extract_from_fields(
            "John is 30",
            {"name": str, "age": int},
        )
        assert result["name"] == "extracted"
        assert result["age"] == 42

    def test_schema_to_description(self) -> None:
        schema = {
            "properties": {
                "name": {"type": "string", "description": "Full name"},
                "skills": {"type": "array", "items": {"type": "string"}},
            }
        }
        desc = _schema_to_description(schema)
        assert "name (string): Full name" in desc
        assert "skills (array of string)" in desc
