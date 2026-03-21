---
icon: lucide/rocket
---

# Get Started

## Installation

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
uvx fusellm extract "Sarah Chen is a 34-year-old architect at Stripe" \
  --model bartowski/Llama-3.2-1B-Instruct-GGUF \
  --fields "name:str,age:int,job_title:str"
```

### With pip

```bash
pip install fusellm
pip install "fusellm[training]"
```

---

## Your first extraction

### 1. Pull a model and extract

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
print(result)
# {'name': 'Sarah Chen', 'age': 34, 'job_title': 'software architect', 'company': 'Stripe'}
```

### 2. Use a local GGUF model

```python
backend = fuse.LlamaCppBackend(model_path="./models/llama-3.2-1b-q4.gguf")
extractor = fuse.Extractor(backend)

result = extractor.extract_from_fields(
    "John is 30 years old and knows Python and Rust",
    {"name": str, "age": int, "skills": list[str]}
)
# {'name': 'John', 'age': 30, 'skills': ['Python', 'Rust']}
```

### 3. Extract from a JSON schema

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

### 4. Let the LLM infer the schema

```python
result = extractor.extract_from_description(
    "The Series A raised $15M from Sequoia, following a $2.5M seed from YC.",
    "Extract monetary amounts, funding round type, and investor names"
)
```

---

## CLI quickstart

### Extract with inline flags

```bash
fuse extract "SpaceX was founded in 2002" \
  --model bartowski/Phi-4-mini-instruct-GGUF \
  --fields "company:str,year:int,industry:str"
```

### Extract with a config file

```bash
fuse extract "Sarah Chen is a 34-year-old architect at Stripe" \
  --config extract_person.yaml
```

See [CLI Reference](cli-reference.md) for all commands and options.

---

## Next steps

- [Concepts](concepts.md) — understand schemas, backends, and extraction modes
- [Configuration](configuration/inference.md) — configure models and extraction
- [CLI Reference](cli-reference.md) — full command reference
