# Fuse

Train small LLMs and deploy them for fast structured extraction on CPU.

## Install

```bash
pip install fusellm
```

## Quick Start

```python
import fuse

backend = fuse.LlamaCppBackend("model.gguf")
extractor = fuse.Extractor(backend)

# Zero-shot structured extraction — no Pydantic model needed
result = extractor.extract_from_fields(
    "John is 30 years old and knows Python and Rust",
    {"name": str, "age": int, "skills": list[str]}
)
# {'name': 'John', 'age': 30, 'skills': ['Python', 'Rust']}
```

## CLI

```bash
fuse extract "John is 30" --model model.gguf --fields "name:str,age:int"
fuse train --config train.yaml
fuse quantize --model ./output --output model.gguf
```
