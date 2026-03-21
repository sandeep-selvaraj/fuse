"""Future Rust/candle inference backend via PyO3.

This module will be replaced with a PyO3 binding to the Rust candle backend.
For now it serves as a placeholder documenting the expected interface.
"""

from __future__ import annotations

from typing import Any


class CandleBackend:
    """Placeholder for the future Rust/candle inference backend.

    Will implement the InferenceBackend protocol via PyO3 bindings to
    the candle ML framework written in Rust.
    """

    def load(self, model_path: str, **kwargs: Any) -> None:
        raise NotImplementedError("Rust/candle backend not yet implemented")

    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs: Any) -> str:
        raise NotImplementedError("Rust/candle backend not yet implemented")

    def generate_structured(
        self, prompt: str, json_schema: dict[str, Any], *, max_tokens: int = 512, **kwargs: Any
    ) -> dict[str, Any]:
        raise NotImplementedError("Rust/candle backend not yet implemented")
