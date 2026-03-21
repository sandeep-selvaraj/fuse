from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def load_dataset_from_file(path: Path) -> list[dict[str, Any]]:
    """Load a dataset from a local JSONL or JSON file.

    Args:
        path: Path to a .jsonl or .json file.

    Returns:
        List of dicts, each representing one training example.
    """
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _load_jsonl(path)
    if suffix == ".json":
        return _load_json(path)
    msg = f"Unsupported file format: {suffix}. Use .json or .jsonl."
    raise ValueError(msg)


def load_dataset_from_hub(name: str, split: str = "train") -> Any:
    """Load a dataset from the HuggingFace Hub.

    Args:
        name: HuggingFace dataset name (e.g. "tatsu-lab/alpaca").
        split: Dataset split to load.

    Returns:
        A HuggingFace Dataset object.
    """
    from datasets import load_dataset

    return load_dataset(name, split=split)


def format_for_sft(
    examples: list[dict[str, Any]],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
) -> list[dict[str, str]]:
    """Format examples into instruction/response pairs for SFT.

    Args:
        examples: Raw training examples.
        instruction_key: Key for the instruction field.
        input_key: Key for the optional input/context field.
        output_key: Key for the expected output field.

    Returns:
        List of dicts with "text" key formatted for training.
    """
    formatted = []
    for ex in examples:
        instruction = ex.get(instruction_key, "")
        context = ex.get(input_key, "")
        output = ex.get(output_key, "")

        if context:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{context}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        formatted.append({"text": text})

    return formatted


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def _load_json(path: Path) -> list[dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    msg = "JSON file must contain a top-level array of objects."
    raise ValueError(msg)
