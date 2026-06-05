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
| Gemma 4 E2B Instruct | Effective 2B | ~3.5 GB | `bartowski/google_gemma-4-E2B-it-GGUF` | Latest Gemma, on-device |
| Gemma 4 E4B Instruct | Effective 4B (~8B) | ~5.4 GB | `bartowski/google_gemma-4-E4B-it-GGUF` | Latest Gemma, higher accuracy |
| Qwen 2.5 1.5B Instruct | 1.5B | ~1 GB | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` | Multilingual extraction |
| Phi-4 Mini Instruct | 3.8B | ~2.5 GB | `bartowski/Phi-4-mini-instruct-GGUF` | Complex reasoning tasks |

!!! note "Latest model families"
    **Gemma 4** (released April 2026) ships in `E2B`/`E4B` on-device sizes plus larger
    `26B` (MoE) and `31B` (dense) variants — for CPU extraction the `E2B`/`E4B` sizes are
    recommended. **Llama**: the smallest current instruct models remain Llama 3.2 1B/3B;
    Llama 4 (Scout/Maverick) are large mixture-of-experts models not suited to CPU
    extraction.

!!! warning "Gemma prompt format"
    Gemma models use their own chat template (`<start_of_turn>` / `<end_of_turn>`), which
    Fuse's `prompt_format` options (`llama`, `chatml`, `generic`) do not reproduce exactly.
    Until a dedicated `gemma` format is added, results may be suboptimal — `chatml` is the
    closest approximation. Constrained decoding still guarantees valid JSON output.

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
- **Accuracy priority**: Phi-4 Mini, Llama 3.2 3B, or Gemma 4 E4B — better at complex extraction
- **Latest Gemma**: Gemma 4 E2B/E4B — strong on-device performance and multilingual support (140+ languages)
- **Multilingual**: Qwen 2.5 1.5B or Gemma 4 E2B — strong multilingual performance
- **Fine-tuning**: Start with Llama 3.2 1B or 3B — good LoRA targets with Unsloth support
