from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from fuse.config import InferenceConfig

if TYPE_CHECKING:
    from fuse.extraction.spans import TokenScores


class LlamaCppBackend:
    """Inference backend using llama-cpp-python for GGUF models on CPU.

    Accepts either a local GGUF path or a HuggingFace repo name.
    HF models are auto-downloaded and cached.
    """

    def __init__(
        self,
        model_path: str | None = None,
        model_name: str | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self._model: Any = None
        self._config = config or InferenceConfig()

        # Priority: explicit model_path > explicit model_name > config fields
        source = (
            model_path
            or model_name
            or (str(self._config.model_path) if self._config.model_path else None)
            or self._config.model_name
        )

        if source:
            self.load(source)

    @classmethod
    def from_config(cls, config: InferenceConfig) -> LlamaCppBackend:
        """Create a backend from an InferenceConfig."""
        return cls(config=config)

    def load(self, model_path: str, **kwargs: Any) -> None:
        from llama_cpp import Llama

        resolved = self._resolve(model_path)
        self._config = self._config.model_copy(update={"model_path": resolved})
        self._model = Llama(
            model_path=resolved,
            n_ctx=self._config.n_ctx,
            n_threads=self._config.n_threads,
            n_gpu_layers=self._config.n_gpu_layers,
            seed=self._config.seed,
            logits_all=self._config.logits_all,
            verbose=False,
            **kwargs,
        )

    def _resolve(self, model_path: str) -> str:
        """Resolve a model path or HF repo name to a local GGUF file."""
        if Path(model_path).is_file():
            return str(Path(model_path).resolve())

        from fuse.inference.model_resolver import resolve_model

        return resolve_model(
            model_path,
            filename=self._config.gguf_filename,
        )

    @property
    def supports_logprobs(self) -> bool:
        """Whether this backend can produce per-token logprobs for confidence.

        Requires a loaded model and logits_all=True (set in InferenceConfig).
        """
        return self._model is not None and self._config.logits_all

    def _ensure_loaded(self) -> None:
        if self._model is None:
            msg = (
                "No model loaded. Pass model_path, model_name, "
                "or a config with model_name/model_path."
            )
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
        self,
        prompt: str,
        json_schema: dict[str, Any],
        *,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._ensure_loaded()

        import outlines

        outlines_model = outlines.from_llamacpp(self._model, chat_mode=False)
        generator = outlines.Generator(outlines_model, outlines.json_schema(json_schema))
        # Set max_tokens on the underlying model for this generation
        self._model.n_tokens = 0
        raw = generator(prompt, max_tokens=max_tokens)
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                msg = (
                    f"Model returned invalid JSON (likely truncated). "
                    f"Try increasing max_tokens. Raw output: {raw!r}"
                )
                raise ValueError(msg) from None
        return cast("dict[str, Any]", raw)

    def generate_structured_with_logprobs(
        self,
        prompt: str,
        json_schema: dict[str, Any],
        *,
        max_tokens: int = 512,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], TokenScores]:
        """Like generate_structured, but also return per-token logprobs.

        Reuses outlines' constrained-decoding logits processor but calls
        llama.cpp directly (with logprobs=1) so the full completion — text
        and token logprobs — is returned in a single pass. The TokenScores
        let the extraction layer compute a per-field confidence.
        """
        self._ensure_loaded()

        if not self._config.logits_all:
            msg = (
                "Confidence scoring requires per-token logprobs, which need the "
                "model to be loaded with logits_all=True. Set logits_all=True in "
                "InferenceConfig (uses more memory)."
            )
            raise ValueError(msg)

        import outlines
        from llama_cpp import LogitsProcessorList

        from fuse.extraction.spans import TokenScores

        outlines_model = outlines.from_llamacpp(self._model, chat_mode=False)
        generator = outlines.Generator(outlines_model, outlines.json_schema(json_schema))
        # SteerableGenerator (the llama.cpp case) exposes the built logits processor.
        processor = cast("Any", generator).logits_processor

        self._model.n_tokens = 0
        completion = self._model(
            prompt,
            logits_processor=LogitsProcessorList([processor]) if processor else None,
            logprobs=1,
            max_tokens=max_tokens,
            temperature=kwargs.get("temperature", self._config.temperature),
        )
        self._model.reset()

        choice = completion["choices"][0]
        text = choice["text"]
        lp = choice.get("logprobs") or {}
        tokens = lp.get("tokens") or []
        token_logprobs = lp.get("token_logprobs") or []

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            msg = (
                f"Model returned invalid JSON (likely truncated). "
                f"Try increasing max_tokens. Raw output: {text!r}"
            )
            raise ValueError(msg) from None

        return data, TokenScores(text=text, tokens=list(tokens), logprobs=list(token_logprobs))
