---
icon: lucide/file-json
---

# Extraction Configuration

## ExtractConfig

The `ExtractConfig` class bundles model settings with schema and prompt options for extraction.

### YAML format

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

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `InferenceConfig` | required | Model configuration (see [Inference](inference.md)) |
| `schema_file` | `Path \| None` | `None` | Path to a JSON schema file |
| `fields` | `dict \| None` | `None` | Field name to type mapping |
| `description` | `str \| None` | `None` | Natural language description of what to extract |
| `prompt_format` | `str` | `"llama"` | Prompt template format: `llama`, `chatml`, or `generic` |
| `max_tokens` | `int` | `512` | Maximum tokens to generate |

!!! tip
    You must provide exactly one of `fields`, `schema_file`, or `description`.

---

## Schema input methods

### Fields dict

Define fields inline with Python type names:

```yaml
fields:
  name: str
  age: int
  skills: list[str]
  active: bool
```

CLI equivalent:

```bash
fuse extract "some text" --model repo/name --fields "name:str,age:int,skills:list[str]"
```

### JSON schema file

Point to a standard JSON Schema:

```yaml
schema_file: "./schemas/person.json"
```

`person.json`:

```json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"},
    "skills": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["name", "age"]
}
```

CLI equivalent:

```bash
fuse extract "some text" --model repo/name --schema person.json
```

### Description

Let the LLM design the schema from a natural language description:

```yaml
description: "Extract person name, age, and list of programming skills"
```

---

## Prompt formats

Choose the format matching your model family:

| Format | Use with | Example models |
|---|---|---|
| `llama` (default) | Llama 3.x family | Llama 3.2 1B/3B |
| `chatml` | ChatML-based models | Qwen 2.5, Phi-4 |
| `generic` | Other instruction-tuned models | Fallback |

```yaml
prompt_format: chatml  # for Qwen or Phi models
```

---

## Evidence spans and visualization

Any extraction mode supports evidence spans via the CLI flags `--spans` and `--html`:

```bash
# Table output with evidence, type, and character offsets
fuse extract "..." --config extract_person.yaml --spans

# HTML file with color-coded highlighted spans
fuse extract "..." --config extract_person.yaml --html result.html
```

In Python, use the `_with_spans` extraction methods:

```python
result = extractor.extract_with_spans(text, schema)
result = extractor.extract_from_fields_with_spans(text, fields)
```

Each field in the result includes:

- `value` — the extracted value
- `evidence` — verbatim quote from the source text
- `is_explicit` — `true` if the value appears word-for-word in the source
- `span` — character-offset `(start, end)` in the source text

See [Concepts — Extraction with spans](../concepts.md#extraction-with-spans) for details on explicit vs. implicit extractions.

---

## Example configs

### Person extraction

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

### Company extraction with Phi-4

```yaml
model:
  model_name: "bartowski/Phi-4-mini-instruct-GGUF"
  n_ctx: 4096
  temperature: 0.0

fields:
  company_name: str
  founded_year: int
  industry: str
  headquarters: str

prompt_format: chatml
max_tokens: 256
```

### Schema-based extraction

```yaml
model:
  model_name: "bartowski/Llama-3.2-3B-Instruct-GGUF"
  n_ctx: 2048
  temperature: 0.0

schema_file: "./schemas/person.json"
prompt_format: llama
max_tokens: 512
```
