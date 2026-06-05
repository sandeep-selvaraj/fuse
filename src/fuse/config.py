from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class QuantMethod(StrEnum):
    Q4_0 = "q4_0"
    Q4_1 = "q4_1"
    Q5_0 = "q5_0"
    Q5_1 = "q5_1"
    Q8_0 = "q8_0"
    F16 = "f16"


class InferenceConfig(BaseModel):
    model_name: str | None = Field(
        default=None,
        description="HuggingFace GGUF repo name",
    )
    model_path: Path | None = Field(default=None, description="Local path to GGUF model file")
    gguf_filename: str | None = Field(
        default=None, description="Specific GGUF filename to download from the repo"
    )
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=4, description="Number of CPU threads for inference")
    n_gpu_layers: int = Field(default=0, description="Layers to offload to GPU (0 = CPU only)")
    max_tokens: int = Field(default=512, description="Default max tokens for generation")
    temperature: float = Field(default=0.0, description="Sampling temperature (0 = greedy)")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    logits_all: bool = Field(
        default=True,
        description="Compute logits for all tokens (required for confidence scoring). "
        "On by default; set False to save memory if you don't need confidence.",
    )


class ExtractConfig(BaseModel):
    model: InferenceConfig = Field(description="Model configuration")
    schema_file: Path | None = Field(default=None, description="Path to JSON schema file")
    fields: dict[str, str] | None = Field(
        default=None, description="Field name to type mapping (e.g. {'name': 'str', 'age': 'int'})"
    )
    description: str | None = Field(
        default=None, description="Natural language description of what to extract"
    )
    prompt_format: str = Field(
        default="llama", description="Prompt format: llama, chatml, generic"
    )
    max_tokens: int = Field(default=512, description="Max tokens for generation")


class TrainConfig(BaseModel):
    model_name: str = Field(description="HuggingFace model name or path")
    output_dir: Path = Field(
        default=Path("./output"), description="Directory to save trained model"
    )
    dataset_name: str | None = Field(default=None, description="HuggingFace dataset name")
    dataset_path: Path | None = Field(default=None, description="Local dataset path (JSONL/JSON)")

    # Validation data (optional). Resolution order: explicit eval dataset >
    # a predefined validation split in a Hub dataset > val_split_ratio auto-split.
    eval_dataset_name: str | None = Field(
        default=None, description="HuggingFace dataset name for validation"
    )
    eval_dataset_path: Path | None = Field(
        default=None, description="Local validation dataset path (JSONL/JSON)"
    )
    train_split: str = Field(
        default="train", description="Split name to load for training (Hub datasets)"
    )
    val_split: str = Field(
        default="validation",
        description="Validation split name to auto-detect/load (Hub datasets)",
    )
    val_split_ratio: float | None = Field(
        default=None,
        description="Fraction of training data to hold out for validation when no explicit "
        "validation set is provided (e.g. 0.1). Off by default.",
    )
    seed: int = Field(default=42, description="Random seed (used for the validation auto-split)")

    # LoRA parameters
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")

    # Training hyperparameters
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=4, description="Training batch size")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(default=4, description="Gradient accumulation steps")

    # Experiment reporting
    report_to: list[str] = Field(
        default_factory=lambda: ["none"],
        description='Reporting integrations: "tensorboard", "mlflow", "wandb", or "none"',
    )
    logging_steps: int = Field(default=10, description="Log metrics every N steps")
    logging_dir: Path | None = Field(
        default=None,
        description=(
            "Directory for TensorBoard logs (sets TENSORBOARD_LOGGING_DIR; "
            "defaults to output_dir/runs)"
        ),
    )
    run_name: str | None = Field(
        default=None, description="Run name for the reporting backend (e.g. MLflow/W&B run)"
    )

    # Export
    quantize: QuantMethod | None = Field(
        default=QuantMethod.Q4_0, description="GGUF quantization method"
    )
    use_unsloth: bool = Field(default=True, description="Use Unsloth for faster training")


class ExportConfig(BaseModel):
    model_path: Path = Field(description="Path to the trained model (HF format)")
    output_path: Path = Field(description="Output path for the GGUF file")
    quant_method: QuantMethod = Field(default=QuantMethod.Q4_0, description="Quantization method")
