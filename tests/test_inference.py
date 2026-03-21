from typing import Any

import pytest

from fuse.inference._rust import CandleBackend
from fuse.inference.backend import InferenceBackend


class MockBackend:
    """A mock backend that satisfies the InferenceBackend protocol."""

    def __init__(self) -> None:
        self.loaded_path: str | None = None

    def load(self, model_path: str, **kwargs: Any) -> None:
        self.loaded_path = model_path

    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs: Any) -> str:
        return f"mock response to: {prompt[:20]}"

    def generate_structured(
        self, prompt: str, json_schema: dict[str, Any], *, max_tokens: int = 512, **kwargs: Any
    ) -> dict[str, Any]:
        # Return a dict matching the schema keys
        properties = json_schema.get("properties", {})
        result: dict[str, Any] = {}
        for key, prop in properties.items():
            prop_type = prop.get("type", "string")
            if prop_type == "string":
                result[key] = "mock"
            elif prop_type == "integer":
                result[key] = 0
            elif prop_type == "number":
                result[key] = 0.0
            elif prop_type == "boolean":
                result[key] = False
            elif prop_type == "array":
                result[key] = []
            else:
                result[key] = None
        return result


class TestInferenceBackendProtocol:
    def test_mock_satisfies_protocol(self) -> None:
        backend = MockBackend()
        assert isinstance(backend, InferenceBackend)

    def test_mock_generate(self) -> None:
        backend = MockBackend()
        result = backend.generate("Hello world")
        assert "mock response" in result

    def test_mock_generate_structured(self) -> None:
        backend = MockBackend()
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = backend.generate_structured("test", json_schema=schema)
        assert "name" in result
        assert "age" in result
        assert isinstance(result["name"], str)
        assert isinstance(result["age"], int)

    def test_mock_load(self) -> None:
        backend = MockBackend()
        backend.load("/path/to/model.gguf")
        assert backend.loaded_path == "/path/to/model.gguf"


class TestCandleBackendStub:
    def test_raises_not_implemented(self) -> None:
        backend = CandleBackend()
        with pytest.raises(NotImplementedError):
            backend.load("model.gguf")
        with pytest.raises(NotImplementedError):
            backend.generate("hello")
        with pytest.raises(NotImplementedError):
            backend.generate_structured("hello", json_schema={})
