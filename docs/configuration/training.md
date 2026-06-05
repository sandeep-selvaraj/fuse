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

`TrainConfig` is a **flat** mapping — all fields sit at the top level (there are no
nested `lora:` or `training:` sections):

```yaml
model_name: "unsloth/Llama-3.2-1B-Instruct"
output_dir: "./output/llama-extraction"
dataset_path: "./data/extraction_dataset.jsonl"

# LoRA
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

# Training
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4
learning_rate: 2.0e-4
max_seq_length: 2048

use_unsloth: true
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str` | required | HuggingFace model name (e.g., `unsloth/Llama-3.2-1B-Instruct`) |
| `output_dir` | `str` | `"./output"` | Directory for saving checkpoints and final model |
| `dataset_path` | `str \| None` | `None` | Path to local JSONL/CSV dataset |
| `dataset_name` | `str \| None` | `None` | HuggingFace dataset name (used if `dataset_path` is unset) |
| `use_unsloth` | `bool` | `true` | Try Unsloth first; fall back to HuggingFace if unavailable |
| `quantize` | `str \| None` | `"q4_0"` | GGUF quantization method for export |

### LoRA parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `lora_r` | `int` | `16` | LoRA rank |
| `lora_alpha` | `int` | `32` | LoRA alpha (scaling factor) |
| `lora_dropout` | `float` | `0.05` | LoRA dropout |

### Training parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_epochs` | `int` | `3` | Number of training epochs |
| `batch_size` | `int` | `4` | Per-device batch size |
| `gradient_accumulation_steps` | `int` | `4` | Gradient accumulation steps |
| `learning_rate` | `float` | `2e-4` | Learning rate |
| `max_seq_length` | `int` | `2048` | Maximum sequence length |

### Validation data

Provide a validation set to have the trainer report `eval_loss` and
`eval_mean_token_accuracy` each epoch (visible in the [reporting](#experiment-reporting)
backends). The validation set is resolved in this priority order:

1. **Explicit validation set** — `eval_dataset_path` (local file) or `eval_dataset_name` (Hub).
2. **Predefined split** — if a Hub `dataset_name` already contains a `validation` split, it is detected and used automatically.
3. **Auto-split** — set `val_split_ratio` to carve a fraction off the training set.

If none apply, training runs without evaluation (unchanged default behavior).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `eval_dataset_path` | `str \| None` | `None` | Local validation dataset (JSONL/JSON) |
| `eval_dataset_name` | `str \| None` | `None` | HuggingFace validation dataset name |
| `train_split` | `str` | `"train"` | Split name to load for training (Hub datasets) |
| `val_split` | `str` | `"validation"` | Validation split name to auto-detect/load (Hub datasets) |
| `val_split_ratio` | `float \| None` | `None` | Fraction of training data held out for validation when no explicit set is given (e.g. `0.1`) |
| `seed` | `int` | `42` | Random seed for the validation auto-split |

```yaml
# A) explicit validation file
dataset_path: "./data/train.jsonl"
eval_dataset_path: "./data/val.jsonl"

# B) a Hub dataset that already has a validation split — detected automatically
dataset_name: "my-org/extraction-dataset"

# C) one dataset, auto-split 10% for validation
dataset_path: "./data/all.jsonl"
val_split_ratio: 0.1
```

### Experiment reporting

Training metrics can be logged to TensorBoard, MLflow, or Weights & Biases. The
TensorBoard and MLflow integrations are installed via the optional `reporting` extra:

```bash
uv add "fusellm[training,reporting]"
```

```yaml
report_to: ["tensorboard"]      # or ["mlflow"], ["wandb"], or several
logging_steps: 10
logging_dir: "./runs/llama-extraction"   # TensorBoard log dir
run_name: "llama-extraction-v1"          # run name for MLflow / W&B
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `report_to` | `list[str]` | `["none"]` | Reporting integrations: `"tensorboard"`, `"mlflow"`, `"wandb"`, or `"none"` |
| `logging_steps` | `int` | `10` | Log metrics every N steps |
| `logging_dir` | `str \| None` | `None` | TensorBoard log dir (sets `TENSORBOARD_LOGGING_DIR`; defaults to `output_dir/runs`) |
| `run_name` | `str \| None` | `None` | Run name for the reporting backend (MLflow / W&B) |

After training with `report_to: ["tensorboard"]`:

```bash
tensorboard --logdir ./runs/llama-extraction
```

For MLflow, point at a tracking server with `export MLFLOW_TRACKING_URI=...` (otherwise
runs are written to a local `./mlruns`). `wandb` is not bundled in the `reporting` extra —
install it separately if you use `report_to: ["wandb"]`.

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

lora_r: 16
lora_alpha: 32
lora_dropout: 0.05

num_epochs: 3
batch_size: 4
learning_rate: 2.0e-4
max_seq_length: 2048

report_to: ["tensorboard"]
logging_dir: "./runs/llama-extraction"
```

### General SFT from HuggingFace dataset

```yaml
model_name: "unsloth/Llama-3.2-3B-Instruct"
output_dir: "./output/llama-sft"
dataset_name: "tatsu-lab/alpaca"

lora_r: 32
lora_alpha: 64

num_epochs: 1
batch_size: 2
gradient_accumulation_steps: 8
learning_rate: 1.0e-4
max_seq_length: 4096
```
