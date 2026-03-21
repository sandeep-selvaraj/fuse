import tempfile
from pathlib import Path

from fuse.inference.model_resolver import _pick_best_quant, resolve_model


class TestPickBestQuant:
    def test_prefers_q4_k_m(self) -> None:
        files = [
            "model-Q8_0.gguf",
            "model-Q4_K_M.gguf",
            "model-Q5_K_M.gguf",
            "model-f16.gguf",
        ]
        assert _pick_best_quant(files) == "model-Q4_K_M.gguf"

    def test_falls_back_to_q4_0(self) -> None:
        files = ["model-Q4_0.gguf", "model-Q8_0.gguf"]
        assert _pick_best_quant(files) == "model-Q4_0.gguf"

    def test_falls_back_to_first(self) -> None:
        files = ["model-custom.gguf"]
        assert _pick_best_quant(files) == "model-custom.gguf"

    def test_lowercase_quant(self) -> None:
        files = ["model-q4_k_m.gguf", "model-q8_0.gguf"]
        assert _pick_best_quant(files) == "model-q4_k_m.gguf"


class TestResolveLocalPath:
    def test_local_file_returned_as_is(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".gguf") as f:
            result = resolve_model(f.name)
            assert result == str(Path(f.name).resolve())
