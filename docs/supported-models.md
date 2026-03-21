---
icon: lucide/cpu
---

# Supported Models

Any GGUF model on HuggingFace works with Fuse. Below are some recommended small models for CPU extraction.

---

## Recommended models

| Model | Parameters | GGUF Size (Q4) | HuggingFace Repo | Best for |
|---|---|---|---|---|
| Llama 3.2 1B Instruct | 1.2B | ~1 GB | `bartowski/Llama-3.2-1B-Instruct-GGUF` | Fast extraction, low memory |
| Llama 3.2 3B Instruct | 3.2B | ~2 GB | `bartowski/Llama-3.2-3B-Instruct-GGUF` | Better accuracy, still fast |
| Qwen 2.5 1.5B Instruct | 1.5B | ~1 GB | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` | Multilingual extraction |
| Phi-4 Mini Instruct | 3.8B | ~2.5 GB | `bartowski/Phi-4-mini-instruct-GGUF` | Complex reasoning tasks |

---

## How model resolution works

When you pass a HuggingFace repo name, Fuse automatically:

1. Lists all GGUF files in the repository
2. Selects the best quantization based on this preference order:

    | Priority | Quantization | Notes |
    |---|---|---|
    | 1 | Q4_K_M | Best quality-to-size ratio |
    | 2 | Q4_K_S | Slightly smaller |
    | 3 | Q4_0 | Basic 4-bit |
    | 4 | Q5_K_M | Higher quality, larger |
    | 5 | Q8_0 | Near-original quality |

3. Downloads the file to `~/.cache/fuse/models/`
4. Caches for future use

### Override the filename

If you want a specific quantization:

```python
config = fuse.InferenceConfig(
    model_name="bartowski/Llama-3.2-1B-Instruct-GGUF",
    gguf_filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
)
backend = fuse.LlamaCppBackend.from_config(config)
```

Or via CLI config:

```yaml
model:
  model_name: "bartowski/Llama-3.2-1B-Instruct-GGUF"
  gguf_filename: "Llama-3.2-1B-Instruct-Q8_0.gguf"
```

---

## Using local models

You can also use any local GGUF file directly:

```python
backend = fuse.LlamaCppBackend(model_path="./models/my-model-q4.gguf")
```

```bash
fuse extract "some text" --model ./models/my-model-q4.gguf --fields "name:str"
```

---

## Choosing a model

- **Speed priority**: Llama 3.2 1B — fastest inference, fits in ~1GB RAM
- **Accuracy priority**: Phi-4 Mini or Llama 3.2 3B — better at complex extraction
- **Multilingual**: Qwen 2.5 1.5B — strong multilingual performance
- **Fine-tuning**: Start with Llama 3.2 1B or 3B — good LoRA targets with Unsloth support
