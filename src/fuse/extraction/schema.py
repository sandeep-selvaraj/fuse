from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model

# Mapping from JSON schema type strings to Python types
_JSON_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}


class SchemaBuilder:
    """Dynamically build Pydantic models for structured extraction.

    Supports three ways to define a schema:
    - from_fields: dict of field names to Python types
    - from_json_schema: a JSON schema dict
    - from_description: natural language (uses the LLM to infer fields)
    """

    @staticmethod
    def from_fields(
        fields: dict[str, type | tuple[type, Any]],
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """Create a Pydantic model from a dict of field names to types.

        Args:
            fields: Mapping of field name to type, or (type, default) tuple.
                Supports: str, int, float, bool, list[T], dict[str, T], T | None.
            model_name: Name for the generated model class.

        Example:
            schema = SchemaBuilder.from_fields({
                "name": str,
                "age": int,
                "skills": list[str],
            })
        """
        field_definitions: dict[str, Any] = {}
        for name, type_or_tuple in fields.items():
            if isinstance(type_or_tuple, tuple):
                field_type, default = type_or_tuple
                field_definitions[name] = (field_type, Field(default=default))
            else:
                field_definitions[name] = (type_or_tuple, ...)

        return create_model(model_name, **field_definitions)

    @staticmethod
    def from_json_schema(
        schema: dict[str, Any],
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """Create a Pydantic model from a JSON schema dict.

        Handles common JSON schema patterns: string, integer, number, boolean,
        arrays with typed items, and nullable fields.

        Args:
            schema: A JSON schema dict with "type": "object" and "properties".
            model_name: Name for the generated model class.
        """
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        field_definitions: dict[str, Any] = {}
        for name, prop in properties.items():
            field_type = _resolve_json_schema_type(prop)
            if name in required:
                field_definitions[name] = (field_type, Field(description=prop.get("description")))
            else:
                field_definitions[name] = (
                    field_type | None,
                    Field(default=None, description=prop.get("description")),
                )

        return create_model(model_name, **field_definitions)

    @staticmethod
    def from_description(
        description: str,
        backend: Any,
        model_name: str = "DynamicModel",
    ) -> type[BaseModel]:
        """Use the LLM itself to infer a schema from a natural language description.

        Args:
            description: Natural language description of what to extract.
            backend: An InferenceBackend instance to use for schema inference.
            model_name: Name for the generated model class.

        Example:
            schema = SchemaBuilder.from_description(
                "Extract the person's name, age, and skills",
                backend=my_backend,
            )
        """
        from fuse.extraction.prompts import SCHEMA_INFERENCE_PROMPT

        schema_spec = {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {
                                "type": "string",
                                "enum": [
                                    "string",
                                    "integer",
                                    "number",
                                    "boolean",
                                    "array",
                                ],
                            },
                            "description": {"type": "string"},
                            "items_type": {
                                "type": "string",
                                "enum": [
                                    "string",
                                    "integer",
                                    "number",
                                    "boolean",
                                ],
                            },
                        },
                        "required": ["name", "type"],
                    },
                }
            },
            "required": ["fields"],
        }

        prompt = SCHEMA_INFERENCE_PROMPT.format(description=description)
        result = backend.generate_structured(prompt, json_schema=schema_spec, max_tokens=1024)

        fields: dict[str, type | tuple[type, Any]] = {}
        for field in result["fields"]:
            field_type = _JSON_TYPE_MAP.get(field["type"], str)
            if field["type"] == "array":
                item_type = _JSON_TYPE_MAP.get(field.get("items_type", "string"), str)
                field_type = list[item_type]  # type: ignore[valid-type]
            fields[field["name"]] = field_type

        return SchemaBuilder.from_fields(fields, model_name=model_name)

    @staticmethod
    def to_json_schema(model: type[BaseModel]) -> dict[str, Any]:
        """Convert a Pydantic model to a JSON schema dict."""
        return model.model_json_schema()


def _resolve_json_schema_type(prop: dict[str, Any]) -> type:
    """Resolve a JSON schema property to a Python type."""
    prop_type = prop.get("type", "string")

    if prop_type == "array":
        items = prop.get("items", {})
        item_type = _JSON_TYPE_MAP.get(items.get("type", "string"), str)
        return list[item_type]  # type: ignore[valid-type]

    if prop_type == "object":
        # Nested objects become dict[str, Any]
        return dict[str, Any]

    return _JSON_TYPE_MAP.get(prop_type, str)
