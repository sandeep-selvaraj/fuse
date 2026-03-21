# Fuse

Train small LLMs and deploy them for fast structured extraction on CPU.

Fuse lets you pull any GGUF model from HuggingFace, run zero-shot structured extraction with dynamic schemas, fine-tune with LoRA via Unsloth/HuggingFace, and export to GGUF for fast CPU inference. No predefined Pydantic models required.

## Install

### With uv (recommended)

```bash
uv add fusellm
```

With training support:

```bash
uv add "fusellm[training]"
```

### Run without installing

```bash
# One-shot extraction — no install needed
uvx fusellm extract "Sarah Chen is a 34-year-old architect at Stripe" \
  --model bartowski/Llama-3.2-1B-Instruct-GGUF \
  --fields "name:str,age:int,job_title:str"

# Or with a config file
uvx fusellm extract "SpaceX was founded in 2002" \
  --config extract_company.yaml
```

### With pip

```bash
pip install fusellm
pip install "fusellm[training]"
```

## Quick Start

### Pull a model from HuggingFace and extract

```python
import fuse

# Auto-downloads the best Q4 GGUF from HuggingFace Hub
backend = fuse.LlamaCppBackend(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF")
extractor = fuse.Extractor(backend)

# Zero-shot structured extraction — no Pydantic model needed
result = extractor.extract_from_fields(
    "Sarah Chen is a 34-year-old software architect at Stripe.",
    {"name": str, "age": int, "job_title": str, "company": str}
)
# {'name': 'Sarah Chen', 'age': 34, 'job_title': 'software architect', 'company': 'Stripe'}
```

### Use a local GGUF model

```python
backend = fuse.LlamaCppBackend(model_path="./models/llama-3.2-1b-q4.gguf")
extractor = fuse.Extractor(backend)

result = extractor.extract_from_fields(
    "John is 30 years old and knows Python and Rust",
    {"name": str, "age": int, "skills": list[str]}
)
# {'name': 'John', 'age': 30, 'skills': ['Python', 'Rust']}
```

### Config-driven extraction

```python
config = fuse.InferenceConfig(
    model_name="bartowski/Phi-4-mini-instruct-GGUF",
    n_ctx=4096,
    n_threads=8,
    temperature=0.0,
)
backend = fuse.LlamaCppBackend.from_config(config)
```

### Extract from a JSON schema

```python
schema = fuse.SchemaBuilder.from_json_schema({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age"],
})
result = extractor.extract("John is 30 and knows Rust", schema)
```

### Let the LLM infer the schema

```python
result = extractor.extract_from_description(
    "The Series A raised $15M from Sequoia, following a $2.5M seed from YC.",
    "Extract monetary amounts, funding round type, and investor names"
)
```

## CLI

### Extract with a config file

```bash
fuse extract "Sarah Chen is a 34-year-old architect at Stripe" \
  --config examples/extract_person.yaml
```

`extract_person.yaml`:

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

### Extract with inline flags

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

### Train

```bash
fuse train --config examples/train_extraction.yaml
```

### Quantize to GGUF

```bash
fuse quantize --model ./output --output model.gguf --method q4_0
```

## Supported Models

Any GGUF model on HuggingFace works. Some good small models for CPU extraction:

| Model | Size | HuggingFace Repo |
|---|---|---|
| Llama 3.2 1B Instruct | ~1GB Q4 | `bartowski/Llama-3.2-1B-Instruct-GGUF` |
| Llama 3.2 3B Instruct | ~2GB Q4 | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| Qwen 2.5 1.5B Instruct | ~1GB Q4 | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` |
| Phi-4 Mini Instruct | ~2.5GB Q4 | `bartowski/Phi-4-mini-instruct-GGUF` |

Models are auto-downloaded and cached to `~/.cache/fuse/models/`.

## Development

```bash
uv sync --extra dev
uv run nox                    # All CI checks
uv run nox -s lint            # Ruff lint + format
uv run nox -s typecheck       # ty type check
uv run nox -s tests           # Pytest across Python 3.11-3.13
```

## License

MIT
