from pathlib import Path

from fuse.config import (
    ExportConfig,
    ExtractConfig,
    InferenceConfig,
    QuantMethod,
    TrainConfig,
)


class TestInferenceConfig:
    def test_defaults(self) -> None:
        config = InferenceConfig(model_path=Path("/tmp/model.gguf"))
        assert config.n_ctx == 2048
        assert config.n_threads == 4
        assert config.n_gpu_layers == 0
        assert config.temperature == 0.0
        assert config.max_tokens == 512

    def test_with_model_name(self) -> None:
        config = InferenceConfig(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF")
        assert config.model_name == "bartowski/Llama-3.2-1B-Instruct-GGUF"
        assert config.model_path is None

    def test_with_gguf_filename(self) -> None:
        config = InferenceConfig(
            model_name="bartowski/Llama-3.2-1B-Instruct-GGUF",
            gguf_filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        )
        assert config.gguf_filename == "Llama-3.2-1B-Instruct-Q4_K_M.gguf"

    def test_custom_values(self) -> None:
        config = InferenceConfig(
            model_path=Path("/tmp/model.gguf"),
            n_ctx=4096,
            n_threads=8,
            temperature=0.7,
        )
        assert config.n_ctx == 4096
        assert config.n_threads == 8
        assert config.temperature == 0.7


class TestExtractConfig:
    def test_with_fields(self) -> None:
        config = ExtractConfig(
            model=InferenceConfig(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF"),
            fields={"name": "str", "age": "int"},
        )
        assert config.fields == {"name": "str", "age": "int"}
        assert config.prompt_format == "llama"

    def test_with_schema_file(self) -> None:
        config = ExtractConfig(
            model=InferenceConfig(model_path=Path("/tmp/model.gguf")),
            schema_file=Path("schema.json"),
        )
        assert config.schema_file == Path("schema.json")

    def test_with_description(self) -> None:
        config = ExtractConfig(
            model=InferenceConfig(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF"),
            description="Extract name and age",
        )
        assert config.description == "Extract name and age"

    def test_from_dict(self) -> None:
        raw = {
            "model": {
                "model_name": "bartowski/Llama-3.2-1B-Instruct-GGUF",
                "n_ctx": 2048,
                "temperature": 0.0,
            },
            "fields": {"name": "str", "age": "int"},
            "prompt_format": "llama",
            "max_tokens": 256,
        }
        config = ExtractConfig(**raw)
        assert config.model.model_name == "bartowski/Llama-3.2-1B-Instruct-GGUF"
        assert config.fields == {"name": "str", "age": "int"}
        assert config.max_tokens == 256


class TestTrainConfig:
    def test_defaults(self) -> None:
        config = TrainConfig(model_name="meta-llama/Llama-3.2-1B")
        assert config.lora_r == 16
        assert config.lora_alpha == 32
        assert config.num_epochs == 3
        assert config.batch_size == 4
        assert config.use_unsloth is True
        assert config.quantize == QuantMethod.Q4_0

    def test_from_dict(self) -> None:
        raw = {
            "model_name": "meta-llama/Llama-3.2-1B",
            "output_dir": "/tmp/output",
            "num_epochs": 5,
            "learning_rate": 1e-4,
        }
        config = TrainConfig(**raw)
        assert config.num_epochs == 5
        assert config.learning_rate == 1e-4


class TestExportConfig:
    def test_defaults(self) -> None:
        config = ExportConfig(
            model_path=Path("/tmp/model"),
            output_path=Path("/tmp/model.gguf"),
        )
        assert config.quant_method == QuantMethod.Q4_0
