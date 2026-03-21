from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fuse.extraction.prompts import format_extraction_prompt
from fuse.extraction.schema import SchemaBuilder

if TYPE_CHECKING:
    from pydantic import BaseModel

    from fuse.inference.backend import InferenceBackend


class Extractor:
    """Structured data extraction from text using constrained LLM generation.

    Supports three modes:
    - extract(): with a pre-defined Pydantic model
    - extract_from_fields(): with a dict of field names to types (zero-shot)
    - extract_from_description(): with a natural language description (zero-shot)
    """

    def __init__(
        self,
        backend: InferenceBackend,
        prompt_format: str = "llama",
    ) -> None:
        self._backend = backend
        self._prompt_format = prompt_format

    def extract(
        self,
        text: str,
        schema: type[BaseModel],
        *,
        max_tokens: int = 512,
    ) -> BaseModel:
        """Extract structured data matching a Pydantic model.

        Args:
            text: Input text to extract from.
            schema: A Pydantic model class defining the expected output.
            max_tokens: Maximum tokens for generation.

        Returns:
            An instance of the provided Pydantic model.
        """
        json_schema = schema.model_json_schema()
        schema_desc = _schema_to_description(json_schema)
        prompt = format_extraction_prompt(text, schema_desc, self._prompt_format)

        result = self._backend.generate_structured(
            prompt, json_schema=json_schema, max_tokens=max_tokens
        )
        return schema.model_validate(result)

    def extract_from_fields(
        self,
        text: str,
        fields: dict[str, type | tuple[type, Any]],
        *,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Extract structured data using a dict of field names to types.

        No need to define a Pydantic model — fields are specified inline.

        Args:
            text: Input text to extract from.
            fields: Dict mapping field names to Python types.
                e.g. {"name": str, "age": int, "skills": list[str]}
            max_tokens: Maximum tokens for generation.

        Returns:
            A dict with the extracted fields.
        """
        model = SchemaBuilder.from_fields(fields)
        result = self.extract(text, model, max_tokens=max_tokens)
        return result.model_dump()

    def extract_from_description(
        self,
        text: str,
        description: str,
        *,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Extract structured data using a natural language description.

        The LLM first infers the schema from the description, then extracts.

        Args:
            text: Input text to extract from.
            description: Natural language description of what to extract.
                e.g. "Extract the person's name, age, and list of skills"
            max_tokens: Maximum tokens for generation.

        Returns:
            A dict with the extracted fields.
        """
        model = SchemaBuilder.from_description(description, backend=self._backend)
        result = self.extract(text, model, max_tokens=max_tokens)
        return result.model_dump()


def _schema_to_description(json_schema: dict[str, Any]) -> str:
    """Convert a JSON schema to a human-readable field description."""
    properties = json_schema.get("properties", {})
    lines = []
    for name, prop in properties.items():
        type_str = prop.get("type", "string")
        if type_str == "array":
            item_type = prop.get("items", {}).get("type", "string")
            type_str = f"array of {item_type}"
        desc = prop.get("description", "")
        line = f"- {name} ({type_str})"
        if desc:
            line += f": {desc}"
        lines.append(line)
    return "\n".join(lines)
