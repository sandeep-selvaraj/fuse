"""Integration tests for the training pipeline.

These tests run actual (minimal) fine-tuning with real models.
Requires the training extra: uv sync --extra training
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.integration.conftest import HF_MODEL


def _has_training_deps() -> bool:
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError:
        return False
    return True


requires_training = pytest.mark.skipif(
    not _has_training_deps(),
    reason="Training dependencies not installed (install with: uv sync --extra training)",
)


class TestDatasetLoading:
    """Test real dataset loading and formatting."""

    def test_load_jsonl(self, sample_training_data):
        from fuse.training.dataset import load_dataset_from_file

        data = load_dataset_from_file(sample_training_data)
        assert len(data) == 4
        assert "instruction" in data[0]
        assert "input" in data[0]
        assert "output" in data[0]

    def test_format_for_sft(self, sample_training_data):
        from fuse.training.dataset import format_for_sft, load_dataset_from_file

        data = load_dataset_from_file(sample_training_data)
        formatted = format_for_sft(data)
        assert len(formatted) == 4
        for item in formatted:
            assert "text" in item
            assert "### Instruction:" in item["text"]
            assert "### Input:" in item["text"]
            assert "### Response:" in item["text"]

    def test_load_json_file(self, tmp_path):
        import json

        from fuse.training.dataset import load_dataset_from_file

        data = [
            {"instruction": "Test", "input": "Input", "output": "Output"},
            {"instruction": "Test2", "input": "Input2", "output": "Output2"},
        ]
        path = tmp_path / "data.json"
        with open(path, "w") as f:
            json.dump(data, f)

        loaded = load_dataset_from_file(path)
        assert len(loaded) == 2

    def test_unsupported_format_raises(self, tmp_path):
        from fuse.training.dataset import load_dataset_from_file

        path = tmp_path / "data.csv"
        path.write_text("a,b\n1,2")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataset_from_file(path)


@requires_training
class TestTrainingPipeline:
    """Test actual model fine-tuning with a tiny model and small dataset."""

    def test_train_minimal_hf(self, sample_training_data, tmp_path):
        """Train for 1 step with HF Transformers + LoRA on a tiny model."""
        from fuse.config import TrainConfig
        from fuse.training.trainer import Trainer

        config = TrainConfig(
            model_name=HF_MODEL,
            output_dir=tmp_path / "output",
            dataset_path=sample_training_data,
            use_unsloth=False,
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-4,
            max_seq_length=128,
            lora_r=4,
            lora_alpha=8,
        )
        trainer = Trainer(config)
        output_dir = trainer.train()

        assert output_dir.exists()
        # Check that model files were saved
        assert (output_dir / "adapter_config.json").exists() or any(
            f.suffix == ".safetensors" for f in output_dir.iterdir()
        )

    def test_train_saves_tokenizer(self, sample_training_data, tmp_path):
        """Verify tokenizer is saved alongside the model."""
        from fuse.config import TrainConfig
        from fuse.training.trainer import Trainer

        config = TrainConfig(
            model_name=HF_MODEL,
            output_dir=tmp_path / "output",
            dataset_path=sample_training_data,
            use_unsloth=False,
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=128,
            lora_r=4,
            lora_alpha=8,
        )
        trainer = Trainer(config)
        output_dir = trainer.train()

        tokenizer_files = [
            "tokenizer_config.json",
            "tokenizer.json",
            "special_tokens_map.json",
        ]
        found = [f for f in tokenizer_files if (output_dir / f).exists()]
        assert len(found) >= 1, f"No tokenizer files found in {output_dir}"

    def test_config_validation_requires_dataset(self):
        """Trainer should fail if no dataset is provided."""
        from fuse.config import TrainConfig
        from fuse.training.trainer import Trainer

        config = TrainConfig(
            model_name=HF_MODEL,
            output_dir=Path("/tmp/fuse-test-no-data"),
        )
        trainer = Trainer(config)
        with pytest.raises(ValueError, match="dataset_path or dataset_name"):
            trainer.train()


@requires_training
class TestTrainingConfig:
    """Test that training configs produce valid training runs."""

    def test_lora_params_applied(self, sample_training_data, tmp_path):
        """Verify LoRA parameters are actually applied to the model."""
        from peft import PeftModel

        from fuse.config import TrainConfig
        from fuse.training.trainer import Trainer

        config = TrainConfig(
            model_name=HF_MODEL,
            output_dir=tmp_path / "output",
            dataset_path=sample_training_data,
            use_unsloth=False,
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            max_seq_length=128,
            lora_r=8,
            lora_alpha=16,
        )
        trainer = Trainer(config)
        trainer._load_data()
        trainer._load_model()

        assert isinstance(trainer._model, PeftModel)
        lora_config = trainer._model.peft_config["default"]
        assert lora_config.r == 8
        assert lora_config.lora_alpha == 16
