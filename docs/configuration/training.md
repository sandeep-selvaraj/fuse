---
icon: lucide/graduation-cap
---

# Training Configuration

## Overview

Fuse supports fine-tuning small LLMs with LoRA adapters. Training requires the optional `training` extra:

```bash
uv add "fusellm[training]"
```

Fuse tries [Unsloth](https://github.com/unslothai/unsloth) first for faster training, and falls back to HuggingFace Transformers + PEFT if Unsloth is not available.

---

## TrainConfig

### YAML format

```yaml
model_name: "unsloth/Llama-3.2-1B-Instruct"
output_dir: "./output/llama-extraction"
dataset_path: "./data/extraction_dataset.jsonl"

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj

training:
  epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  warmup_steps: 10
  max_seq_length: 2048
  fp16: true
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | HuggingFace model name (e.g., `unsloth/Llama-3.2-1B-Instruct`) |
| `output_dir` | `str` | `"./output"` | Directory for saving checkpoints and final model |
| `dataset_path` | `str \| None` | `None` | Path to local JSONL dataset |
| `dataset_name` | `str \| None` | `None` | HuggingFace dataset name |
| `dataset_split` | `str` | `"train"` | Dataset split to use |

### LoRA parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lora.r` | `int` | `16` | LoRA rank |
| `lora.alpha` | `int` | `32` | LoRA alpha (scaling factor) |
| `lora.dropout` | `float` | `0.05` | LoRA dropout |
| `lora.target_modules` | `list[str]` | `["q_proj", "k_proj", "v_proj", "o_proj"]` | Modules to apply LoRA to |

### Training parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `training.epochs` | `int` | `3` | Number of training epochs |
| `training.batch_size` | `int` | `4` | Per-device batch size |
| `training.gradient_accumulation_steps` | `int` | `4` | Gradient accumulation steps |
| `training.learning_rate` | `float` | `2e-4` | Learning rate |
| `training.warmup_steps` | `int` | `10` | Warmup steps |
| `training.max_seq_length` | `int` | `2048` | Maximum sequence length |
| `training.fp16` | `bool` | `true` | Use FP16 mixed precision |

---

## Dataset format

Training data should be in JSONL format with Alpaca-style fields:

```json
{"instruction": "Extract person details", "input": "John Smith is a 30-year-old engineer at Google.", "output": "{\"name\": \"John Smith\", \"age\": 30, \"job_title\": \"engineer\", \"company\": \"Google\"}"}
{"instruction": "Extract person details", "input": "Alice is 25 and works as a designer.", "output": "{\"name\": \"Alice\", \"age\": 25, \"job_title\": \"designer\", \"company\": null}"}
```

Each line has:

| Field | Description |
|---|---|
| `instruction` | The task description |
| `input` | The text to extract from |
| `output` | The expected JSON output |

You can also load datasets from HuggingFace Hub:

```yaml
dataset_name: "my-org/extraction-dataset"
dataset_split: "train"
```

---

## Running training

```bash
fuse train --config train_config.yaml
```

### Python API

```python
from fuse.training.trainer import Trainer
from fuse.config import TrainConfig

config = TrainConfig(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    output_dir="./output",
    dataset_path="./data/train.jsonl",
)
trainer = Trainer(config)
trainer.train()
```

---

## Exporting to GGUF

After training, convert the model to GGUF for CPU inference:

```bash
fuse quantize --model ./output --output model.gguf --method q4_0
```

Available quantization methods: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`.

Then use the exported model:

```python
backend = fuse.LlamaCppBackend(model_path="./model.gguf")
extractor = fuse.Extractor(backend)
```

---

## Example configs

### Extraction fine-tuning

```yaml
model_name: "unsloth/Llama-3.2-1B-Instruct"
output_dir: "./output/llama-extraction"
dataset_path: "./data/extraction_dataset.jsonl"

lora:
  r: 16
  alpha: 32
  dropout: 0.05

training:
  epochs: 3
  batch_size: 4
  learning_rate: 2.0e-4
  max_seq_length: 2048
```

### General SFT from HuggingFace dataset

```yaml
model_name: "unsloth/Llama-3.2-3B-Instruct"
output_dir: "./output/llama-sft"
dataset_name: "tatsu-lab/alpaca"
dataset_split: "train"

lora:
  r: 32
  alpha: 64

training:
  epochs: 1
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 1.0e-4
  max_seq_length: 4096
  fp16: true
```
