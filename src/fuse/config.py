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
    model_path: Path
    n_ctx: int = Field(default=2048, description="Context window size")
    n_threads: int = Field(default=4, description="Number of CPU threads for inference")
    n_gpu_layers: int = Field(default=0, description="Layers to offload to GPU (0 = CPU only)")
    max_tokens: int = Field(default=512, description="Default max tokens for generation")
    temperature: float = Field(default=0.0, description="Sampling temperature (0 = greedy)")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class TrainConfig(BaseModel):
    model_name: str = Field(description="HuggingFace model name or path")
    output_dir: Path = Field(
        default=Path("./output"), description="Directory to save trained model"
    )
    dataset_name: str | None = Field(default=None, description="HuggingFace dataset name")
    dataset_path: Path | None = Field(default=None, description="Local dataset path (JSONL/CSV)")

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

    # Export
    quantize: QuantMethod | None = Field(
        default=QuantMethod.Q4_0, description="GGUF quantization method"
    )
    use_unsloth: bool = Field(default=True, description="Use Unsloth for faster training")


class ExportConfig(BaseModel):
    model_path: Path = Field(description="Path to the trained model (HF format)")
    output_path: Path = Field(description="Output path for the GGUF file")
    quant_method: QuantMethod = Field(default=QuantMethod.Q4_0, description="Quantization method")
