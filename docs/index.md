---
icon: lucide/home
---

<div class="fuse-hero">
  <h1>Fuse</h1>
  <p class="hero-tagline">Pull any GGUF model from HuggingFace. Extract structured data on CPU. No Pydantic boilerplate.</p>
  <div class="hero-actions">
    <a href="get-started/" class="md-button md-button--primary">Get Started</a>
    <a href="https://github.com/sandeep-selvaraj/fuse" class="md-button">View on GitHub</a>
  </div>
  <div class="hero-terminal">
    <span class="term-comment"># one-shot extraction — no install needed</span><br>
    <span class="term-prompt">$</span> <span class="term-cmd">uvx fusellm extract "Sarah Chen, 34, architect at Stripe" \</span><br>
    <span class="term-cmd">&nbsp;&nbsp;--model bartowski/Llama-3.2-1B-Instruct-GGUF \</span><br>
    <span class="term-cmd">&nbsp;&nbsp;--fields "name:str,age:int,company:str"</span><br><br>
    <span class="term-output">{"name": "Sarah Chen", "age": 34, "company": "Stripe"}</span>
  </div>
</div>

<div class="badge-row">
  <span class="badge">Python 3.11+</span>
  <span class="badge">CPU inference</span>
  <span class="badge">GGUF models</span>
  <span class="badge">Zero-shot</span>
  <span class="badge">JSON Schema</span>
  <span class="badge">LoRA fine-tuning</span>
</div>

---

## What is Fuse?

Fuse lets you pull any GGUF model from HuggingFace, run zero-shot structured extraction with dynamic schemas, fine-tune with LoRA, and export to GGUF for fast CPU inference.

Define what to extract — as a Python dict, JSON schema, or natural language description — and Fuse handles prompt construction, constrained generation, and JSON parsing.

---

## How it works

``` mermaid
flowchart LR
    S[Schema\ndict · JSON · description] --> E[Extractor]
    M[GGUF Model\nlocal or HuggingFace] --> B[LlamaCppBackend]
    B --> E
    E -->|constrained generation\nvia outlines| R[Structured JSON]
```

---

## Key features

<div class="feature-grid">
  <div class="feature-card">
    <span class="feature-icon">&#9889;</span>
    <h3>Zero-shot extraction</h3>
    <p>No training data needed. Pass a dict of fields and get structured JSON back from any instruction-tuned model.</p>
  </div>
  <div class="feature-card">
    <span class="feature-icon">&#128300;</span>
    <h3>Dynamic schemas</h3>
    <p>Build schemas from Python dicts, JSON Schema, or natural language. No Pydantic boilerplate required.</p>
  </div>
  <div class="feature-card">
    <span class="feature-icon">&#128229;</span>
    <h3>HuggingFace auto-download</h3>
    <p>Pass a repo name and Fuse downloads the best Q4 GGUF automatically. Models are cached locally.</p>
  </div>
  <div class="feature-card">
    <span class="feature-icon">&#9881;</span>
    <h3>Fine-tune with LoRA</h3>
    <p>Train on your domain data with Unsloth or HuggingFace Transformers, then export to GGUF for deployment.</p>
  </div>
</div>

---

## Quick example

```python
import fuse

backend = fuse.LlamaCppBackend(model_name="bartowski/Llama-3.2-1B-Instruct-GGUF")
extractor = fuse.Extractor(backend)

result = extractor.extract_from_fields(
    "Sarah Chen is a 34-year-old software architect at Stripe.",
    {"name": str, "age": int, "job_title": str, "company": str}
)
# {'name': 'Sarah Chen', 'age': 34, 'job_title': 'software architect', 'company': 'Stripe'}
```

---

## Supported models

Any GGUF model on HuggingFace works. Some good small models for CPU extraction:

| Model | Size | HuggingFace Repo |
|---|---|---|
| Llama 3.2 1B Instruct | ~1GB Q4 | `bartowski/Llama-3.2-1B-Instruct-GGUF` |
| Llama 3.2 3B Instruct | ~2GB Q4 | `bartowski/Llama-3.2-3B-Instruct-GGUF` |
| Qwen 2.5 1.5B Instruct | ~1GB Q4 | `bartowski/Qwen2.5-1.5B-Instruct-GGUF` |
| Phi-4 Mini Instruct | ~2.5GB Q4 | `bartowski/Phi-4-mini-instruct-GGUF` |

[See all supported models](supported-models.md){ .md-button }
