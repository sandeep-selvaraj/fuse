---
icon: lucide/settings
---

# Inference Configuration

## InferenceConfig

The `InferenceConfig` class controls model loading and generation behavior.

```python
import fuse

config = fuse.InferenceConfig(
    model_name="bartowski/Llama-3.2-1B-Instruct-GGUF",
    n_ctx=4096,
    n_threads=8,
    temperature=0.0,
)
backend = fuse.LlamaCppBackend.from_config(config)
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_name` | `str \| None` | `None` | HuggingFace GGUF repo name (e.g., `bartowski/Llama-3.2-1B-Instruct-GGUF`) |
| `model_path` | `Path \| None` | `None` | Local path to a GGUF model file |
| `gguf_filename` | `str \| None` | `None` | Specific GGUF filename to download from the repo |
| `n_ctx` | `int` | `2048` | Context window size |
| `n_threads` | `int` | `4` | Number of CPU threads for inference |
| `n_gpu_layers` | `int` | `0` | Number of layers to offload to GPU (0 = CPU only) |
| `max_tokens` | `int` | `512` | Maximum tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature (0.0 = deterministic) |
| `seed` | `int` | `42` | Random seed for reproducibility |
| `logits_all` | `bool` | `True` | Compute logits for all tokens — required for per-field [confidence scoring](extraction.md#confidence-scoring). On by default; set `False` to save memory if you don't need confidence. |

!!! note "Confidence scoring and memory"
    Confidence scores require `logits_all=True` (the default). This keeps per-token
    logprobs around, which costs extra memory. If you don't need confidence, set
    `logits_all=False` to reduce memory usage — extraction still works, and
    confidence values are simply reported as `None`.

### Priority order

When creating a `LlamaCppBackend`, the model source is resolved in this order:

1. Explicit `model_path` argument
2. Explicit `model_name` argument
3. `config.model_path`
4. `config.model_name`

```python
# These are all equivalent:
backend = fuse.LlamaCppBackend(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF")
backend = fuse.LlamaCppBackend(config=fuse.InferenceConfig(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF"))
backend = fuse.LlamaCppBackend.from_config(fuse.InferenceConfig(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF"))
```

---

## YAML config

Inference settings can also be specified in YAML config files used with the CLI:

```yaml
model:
  model_name: "bartowski/Llama-3.2-1B-Instruct-GGUF"
  n_ctx: 4096
  n_threads: 8
  n_gpu_layers: 0
  temperature: 0.0
  seed: 42
  logits_all: true   # required for confidence scoring (default); set false to save memory
```

This is the `model` section of an [extraction config](extraction.md).
