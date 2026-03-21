# Fuse

Train small LLMs, deploy for fast structured extraction on CPU.

## Setup

```bash
uv sync --all-extras
```

## Development Commands

```bash
uv run nox                           # Run all CI sessions (lint, typecheck, tests)
uv run nox -s lint                   # Lint + format check
uv run nox -s typecheck              # Type check
uv run nox -s tests                  # Tests across Python 3.11-3.13
uv run nox -s tests-3.13             # Tests on specific Python version
uv run ruff check src/ tests/        # Lint only (no nox)
uv run ruff format src/ tests/       # Auto-format
uv run pytest tests/                 # Tests only (no nox)
uv run fuse --help                   # CLI
```

## Architecture

- `src/fuse/inference/backend.py` — `InferenceBackend` Protocol. All backends implement this.
- `src/fuse/inference/llama_cpp.py` — v0.1 Python backend (llama-cpp-python, GGUF).
- `src/fuse/inference/_rust.py` — Future Rust/candle backend stub (PyO3).
- `src/fuse/extraction/` — Structured extraction with dynamic schemas and zero-shot prompts.
- `src/fuse/training/` — Fine-tuning via Unsloth or HuggingFace Transformers + LoRA.
- `src/fuse/cli/app.py` — Typer CLI.

## Key Conventions

- Use `uv` for package management, `ruff` for linting, `ty` for type checking.
- All inference backends implement the `InferenceBackend` protocol — never import a concrete backend inside the extraction layer.
- Training deps are optional (`uv sync --extra training`).
- Lazy-import heavy dependencies (torch, transformers, llama_cpp, outlines) to keep CLI startup fast.
