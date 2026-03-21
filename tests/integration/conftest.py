"""Shared fixtures for integration tests.

These tests download real models and run actual inference/training.
They are slow and require network access.

Run with:
    uv run nox -s integration
    uv run pytest tests/integration/ -v
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Small GGUF model for inference tests (~70MB Q4)
GGUF_REPO = "bartowski/SmolLM2-135M-Instruct-GGUF"
GGUF_FILENAME = "SmolLM2-135M-Instruct-Q4_K_M.gguf"

# Small HF model for training tests
HF_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"


@pytest.fixture(scope="session")
def gguf_model_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Download a tiny GGUF model for inference tests. Cached per session."""
    from fuse.inference.model_resolver import resolve_model

    cache_dir = tmp_path_factory.mktemp("models")
    return resolve_model(GGUF_REPO, filename=GGUF_FILENAME, cache_dir=cache_dir)


@pytest.fixture(scope="session")
def backend(gguf_model_path: str):
    """Create a LlamaCppBackend loaded with the test model."""
    from fuse.inference.llama_cpp import LlamaCppBackend

    return LlamaCppBackend(model_path=gguf_model_path)


@pytest.fixture()
def sample_training_data(tmp_path: Path) -> Path:
    """Create a minimal JSONL training dataset."""
    import json

    data = [
        {
            "instruction": "Extract person details from the text.",
            "input": "John Smith is a 30-year-old engineer at Google.",
            "output": '{"name": "John Smith", "age": 30, "company": "Google"}',
        },
        {
            "instruction": "Extract person details from the text.",
            "input": "Alice Lee is a 25-year-old designer at Apple.",
            "output": '{"name": "Alice Lee", "age": 25, "company": "Apple"}',
        },
        {
            "instruction": "Extract person details from the text.",
            "input": "Bob Chen is a 40-year-old manager at Meta.",
            "output": '{"name": "Bob Chen", "age": 40, "company": "Meta"}',
        },
        {
            "instruction": "Extract person details from the text.",
            "input": "Carol Davis is a 35-year-old scientist at NASA.",
            "output": '{"name": "Carol Davis", "age": 35, "company": "NASA"}',
        },
    ]
    path = tmp_path / "train.jsonl"
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path
