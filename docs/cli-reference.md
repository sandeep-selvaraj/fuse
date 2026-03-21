---
icon: lucide/terminal
---

# CLI Reference

Fuse provides a command-line interface for extraction, training, and quantization.

```bash
fuse --help
```

---

## `fuse extract`

Run structured extraction on text input.

### With a config file

```bash
fuse extract "Sarah Chen is a 34-year-old architect at Stripe" \
  --config extract_person.yaml
```

### With inline flags

```bash
# HuggingFace model — auto-downloads
fuse extract "SpaceX was founded in 2002" \
  --model bartowski/Phi-4-mini-instruct-GGUF \
  --fields "company:str,year:int,industry:str"

# Local GGUF model
fuse extract "John is 30" \
  --model ./model.gguf \
  --fields "name:str,age:int"

# Using a JSON schema file
fuse extract "John is 30 and knows Python" \
  --model bartowski/Llama-3.2-1B-Instruct-GGUF \
  --schema schema.json
```

### With uvx (no install)

```bash
uvx fusellm extract "Sarah Chen is a 34-year-old architect" \
  --model bartowski/Llama-3.2-1B-Instruct-GGUF \
  --fields "name:str,age:int,job_title:str"
```

### Options

| Option | Description |
|---|---|
| `--config PATH` | YAML config file for extraction |
| `--model TEXT` | Model path (local GGUF) or HuggingFace repo name |
| `--fields TEXT` | Comma-separated field definitions (e.g., `name:str,age:int`) |
| `--schema PATH` | JSON schema file path |
| `--format TEXT` | Prompt format: `llama`, `chatml`, or `generic` (default: `llama`) |
| `--max-tokens INT` | Maximum tokens to generate (default: `512`) |

---

## `fuse train`

Fine-tune a model using LoRA.

```bash
fuse train --config train_extraction.yaml
```

### Options

| Option | Description |
|---|---|
| `--config PATH` | **Required.** YAML training config file |

See [Training Configuration](configuration/training.md) for config file format.

---

## `fuse quantize`

Convert a trained model to GGUF format.

```bash
fuse quantize --model ./output --output model.gguf --method q4_0
```

### Options

| Option | Description |
|---|---|
| `--model PATH` | **Required.** Path to the trained model directory |
| `--output PATH` | **Required.** Output GGUF file path |
| `--method TEXT` | Quantization method (default: `q4_0`) |

---

## Config file examples

### Extraction config

```yaml
model:
  model_name: "bartowski/Llama-3.2-1B-Instruct-GGUF"
  n_ctx: 2048
  temperature: 0.0

fields:
  name: str
  age: int
  job_title: str
  company: str

prompt_format: llama
max_tokens: 256
```

### Training config

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
