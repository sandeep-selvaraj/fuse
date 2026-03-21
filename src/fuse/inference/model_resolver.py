from __future__ import annotations

from pathlib import Path

from rich.console import Console

console = Console()

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "fuse" / "models"

# Preferred quantization patterns in order of preference (small + good quality first)
GGUF_QUANT_PREFERENCE = [
    "*Q4_K_M*",
    "*Q4_K_S*",
    "*q4_k_m*",
    "*q4_k_s*",
    "*Q4_0*",
    "*q4_0*",
    "*Q5_K_M*",
    "*q5_k_m*",
    "*Q8_0*",
    "*q8_0*",
    "*.gguf",
]


def resolve_model(
    model_name: str,
    filename: str | None = None,
    cache_dir: Path | None = None,
) -> str:
    """Resolve a model name to a local GGUF file path.

    If model_name is a local path to an existing file, returns it as-is.
    Otherwise, treats it as a HuggingFace repo name and downloads the GGUF.

    Args:
        model_name: Local path or HuggingFace repo (e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF").
        filename: Specific GGUF filename to download. If None, picks the best Q4 quant.
        cache_dir: Local directory to cache downloaded models.

    Returns:
        Absolute path to the local GGUF file.
    """
    # If it's already a local file, use it directly
    if Path(model_name).is_file():
        return str(Path(model_name).resolve())

    cache = cache_dir or DEFAULT_CACHE_DIR

    # If filename given, download that specific file
    if filename:
        return _download_file(model_name, filename, cache)

    # Auto-detect best GGUF file from the repo
    return _download_best_gguf(model_name, cache)


def _download_file(repo_id: str, filename: str, cache_dir: Path) -> str:
    """Download a specific file from a HuggingFace repo."""
    from huggingface_hub import hf_hub_download

    console.print(f"Downloading [cyan]{filename}[/cyan] from [cyan]{repo_id}[/cyan]...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=str(cache_dir),
    )
    console.print(f"Model cached at [green]{path}[/green]")
    return path


def _download_best_gguf(repo_id: str, cache_dir: Path) -> str:
    """Find and download the best GGUF quant from a HuggingFace repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    console.print(f"Scanning [cyan]{repo_id}[/cyan] for GGUF files...")

    try:
        files = api.list_repo_files(repo_id)
    except Exception as e:
        msg = f"Could not list files in repo '{repo_id}': {e}"
        raise RuntimeError(msg) from e

    gguf_files = [f for f in files if f.endswith(".gguf")]
    if not gguf_files:
        msg = (
            f"No .gguf files found in '{repo_id}'. "
            "Make sure this is a GGUF model repo "
            "(e.g. 'bartowski/Llama-3.2-1B-Instruct-GGUF')."
        )
        raise FileNotFoundError(msg)

    # Pick the best quant based on preference order
    selected = _pick_best_quant(gguf_files)
    console.print(f"Selected [bold]{selected}[/bold]")
    return _download_file(repo_id, selected, cache_dir)


def _pick_best_quant(filenames: list[str]) -> str:
    """Pick the best GGUF quant from a list of filenames."""
    import fnmatch

    for pattern in GGUF_QUANT_PREFERENCE:
        for f in filenames:
            if fnmatch.fnmatch(f, pattern):
                return f
    # Fallback: just pick the first .gguf file
    return filenames[0]
