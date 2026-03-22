---
icon: lucide/lightbulb
---

# Concepts

## Architecture

Fuse has three layers that are cleanly separated:

```
┌─────────────────────────────────┐
│  Extraction Layer               │
│  Extractor · SchemaBuilder      │
│  Prompts · JSON parsing         │
├─────────────────────────────────┤
│  Inference Backend (Protocol)   │
│  LlamaCppBackend (v0.1)        │
│  RustBackend (future)           │
├─────────────────────────────────┤
│  Model Resolution               │
│  Local GGUF · HuggingFace Hub   │
└─────────────────────────────────┘
```

The extraction layer never imports a concrete backend directly — it works against the `InferenceBackend` protocol. This means the backend can be swapped (e.g., from Python to Rust) without touching extraction code.

---

## Inference backends

All backends implement the `InferenceBackend` protocol:

```python
class InferenceBackend(Protocol):
    def load(self, model_path: str, **kwargs) -> None: ...
    def generate(self, prompt: str, *, max_tokens: int = 512, **kwargs) -> str: ...
    def generate_structured(
        self, prompt: str, json_schema: dict, *, max_tokens: int = 512, **kwargs
    ) -> dict: ...
```

### LlamaCppBackend

The v0.1 backend uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) for GGUF inference on CPU. Structured generation is powered by [outlines](https://github.com/dottxt-ai/outlines) for JSON schema constrained decoding.

```python
import fuse

# From HuggingFace (auto-downloads best Q4 GGUF)
backend = fuse.LlamaCppBackend(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF")

# From a local file
backend = fuse.LlamaCppBackend(model_path="./model.gguf")

# From config
config = fuse.InferenceConfig(model_name="bartowski/Phi-4-mini-instruct-GGUF", n_ctx=4096)
backend = fuse.LlamaCppBackend.from_config(config)
```

---

## Schema builder

Fuse does not require predefined Pydantic models. Schemas are built dynamically using `SchemaBuilder`:

### From a fields dict

The simplest approach — pass Python types directly:

```python
schema = fuse.SchemaBuilder.from_fields({
    "name": str,
    "age": int,
    "skills": list[str],
})
```

You can also provide defaults:

```python
schema = fuse.SchemaBuilder.from_fields({
    "name": str,
    "score": (float, 0.0),  # default value
})
```

### From a JSON schema

Load a standard JSON Schema object:

```python
schema = fuse.SchemaBuilder.from_json_schema({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
})
```

### From a description

Let the LLM infer the schema from natural language:

```python
result = extractor.extract_from_description(
    "The Series A raised $15M from Sequoia.",
    "Extract monetary amounts, round type, and investors"
)
```

---

## Extraction modes

The `Extractor` class supports three extraction modes:

| Mode | Method | Input | Output |
|---|---|---|---|
| Pydantic model | `extract()` | `type[BaseModel]` | `BaseModel` instance |
| Fields dict | `extract_from_fields()` | `dict[str, type]` | `dict` |
| Description | `extract_from_description()` | `str` | `dict` |

All modes use constrained generation under the hood — the model is forced to output valid JSON matching the schema.

### Extraction with spans

Every extraction mode has a `_with_spans` variant that returns source text localization:

| Mode | Method | Output |
|---|---|---|
| Pydantic model | `extract_with_spans()` | `SpannedResult` |
| Fields dict | `extract_from_fields_with_spans()` | `SpannedResult` |

Each field in a `SpannedResult` includes:

- **value** — the extracted value
- **evidence** — a verbatim quote from the source text supporting the value
- **is_explicit** — whether the value appears word-for-word in the source
- **span** — character-offset `(start, end)` in the source text

Fuse distinguishes between two types of extraction:

| Type | Example | Span points to |
|---|---|---|
| **Explicit** | Name: "Sarah Chen" (verbatim in text) | The value itself |
| **Implicit** | Sentiment: "negative" (inferred) | The evidence passage |

For explicit extractions, the value is a direct substring of the source — localization uses exact substring matching. For implicit extractions (e.g., sentiment, category), the model quotes the evidence passage verbatim, and the span points to that passage instead.

```python
result = extractor.extract_with_spans(text, PersonSchema)

for field in result.fields:
    print(f"{field.name}: {field.value}")
    print(f"  evidence: {field.evidence!r}")
    print(f"  type: {'explicit' if field.is_explicit else 'implicit'}")
    if field.span:
        print(f"  source[{field.span.start}:{field.span.end}]")
```

### HTML visualization

Generate an HTML page with color-coded highlighted spans:

```python
from fuse.extraction.visualize import render_html

html = render_html(source_text, result)
Path("result.html").write_text(html)
```

Or via CLI:

```bash
fuse extract "..." --model repo/name --fields "..." --html result.html
```

The visualization uses solid outlines for explicit extractions and dashed outlines for implicit ones, with a legend showing all fields, their values, and character offsets.

---

## Prompt formats

Fuse includes prompt templates for common model families:

| Format | Models | Template style |
|---|---|---|
| `llama` | Llama 3.x | `<\|start_header_id\|>...<\|eot_id\|>` |
| `chatml` | Qwen, Phi | `<\|im_start\|>...<\|im_end\|>` |
| `generic` | Fallback | `<\|system\|>...<\|end\|>` |

The prompt format is set via `prompt_format` in config files or defaults to `llama`.

!!! tip
    llama-cpp-python automatically adds BOS tokens. Fuse's templates do **not** include `<|begin_of_text|>` to avoid duplication.

---

## Model resolution

When you pass a HuggingFace repo name (e.g., `bartowski/Llama-3.2-1B-Instruct-GGUF`), Fuse:

1. Lists all GGUF files in the repo
2. Picks the best quantization (preference: Q4_K_M > Q4_K_S > Q4_0 > Q5_K_M > Q8_0)
3. Downloads to `~/.cache/fuse/models/`
4. Returns the local path

You can override the filename with `gguf_filename` in config.

---

## Training

Fuse supports fine-tuning with LoRA via two paths:

1. **Unsloth** (preferred) — faster training with optimized kernels
2. **HuggingFace Transformers + PEFT** — fallback when Unsloth is unavailable

Training produces a HuggingFace-format model that can be exported to GGUF for CPU deployment.

See [Training Configuration](configuration/training.md) for details.
