from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fuse.config import InferenceConfig


class LlamaCppBackend:
    """Inference backend using llama-cpp-python for GGUF models on CPU."""

    def __init__(
        self,
        model_path: str | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self._model: Any = None
        self._config = config or InferenceConfig(model_path=Path(model_path or ""))
        if model_path:
            self.load(model_path)

    def load(self, model_path: str, **kwargs: Any) -> None:
        from llama_cpp import Llama

        self._config = self._config.model_copy(update={"model_path": model_path})
        self._model = Llama(
            model_path=model_path,
            n_ctx=self._config.n_ctx,
            n_threads=self._config.n_threads,
            n_gpu_layers=self._config.n_gpu_layers,
            seed=self._config.seed,
            verbose=False,
            **kwargs,
        )

    def _ensure_loaded(self) -> None:
        if self._model is None:
            msg = "No model loaded. Call load() first or pass model_path to constructor."
            raise RuntimeError(msg)

    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs: Any) -> str:
        self._ensure_loaded()
        response = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", self._config.temperature),
        )
        return response["choices"][0]["text"]

    def generate_structured(
        self, prompt: str, json_schema: dict[str, Any], *, max_tokens: int = 512, **kwargs: Any
    ) -> dict[str, Any]:
        self._ensure_loaded()

        from outlines.integrations.llamacpp import JSONLogitsProcessor

        logits_processor = JSONLogitsProcessor(json_schema, self._model)
        response = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", self._config.temperature),
            logits_processor=[logits_processor],
        )
        raw = response["choices"][0]["text"]
        return json.loads(raw)
