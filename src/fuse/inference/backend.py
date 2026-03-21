from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class InferenceBackend(Protocol):
    """Protocol that all inference backends must implement.

    This is the seam where the Rust/candle backend will plug in.
    """

    def load(self, model_path: str, **kwargs: Any) -> None:
        """Load a model from the given path."""
        ...

    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs: Any) -> str:
        """Generate free-form text from a prompt."""
        ...

    def generate_structured(
        self, prompt: str, json_schema: dict[str, Any], *, max_tokens: int = 512, **kwargs: Any
    ) -> dict[str, Any]:
        """Generate JSON output constrained to the given JSON schema."""
        ...
