from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from pathlib import Path

    from fuse.config import ExportConfig

console = Console()


def export_to_gguf(config: ExportConfig) -> Path:
    """Export a trained HuggingFace model to GGUF format.

    Uses llama.cpp's convert scripts. Requires llama.cpp to be installed
    or available on PATH.

    Args:
        config: Export configuration with model path, output path, and quant method.

    Returns:
        Path to the exported GGUF file.
    """
    model_path = config.model_path
    output_path = config.output_path
    quant_method = config.quant_method.value

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert HF model to GGUF F16
    f16_path = output_path.with_suffix(".f16.gguf")
    console.print(f"Converting [cyan]{model_path}[/cyan] to GGUF F16...")

    _run_command(
        [
            "python",
            "-m",
            "llama_cpp.convert",
            str(model_path),
            "--outfile",
            str(f16_path),
            "--outtype",
            "f16",
        ]
    )

    # Step 2: Quantize if not F16
    if quant_method == "f16":
        if f16_path != output_path:
            f16_path.rename(output_path)
        console.print(f"[bold green]Exported to {output_path}[/bold green]")
        return output_path

    console.print(f"Quantizing to [cyan]{quant_method}[/cyan]...")
    _run_command(
        [
            "llama-quantize",
            str(f16_path),
            str(output_path),
            quant_method,
        ]
    )

    # Clean up intermediate F16 file
    if f16_path.exists():
        f16_path.unlink()

    console.print(f"[bold green]Exported to {output_path}[/bold green]")
    return output_path


def _run_command(cmd: list[str]) -> None:
    """Run a shell command, raising on failure."""
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        msg = f"Command not found: {cmd[0]}. Ensure llama.cpp tools are installed."
        raise RuntimeError(msg) from None
    except subprocess.CalledProcessError as e:
        msg = f"Command failed: {' '.join(cmd)}\n{e.stderr}"
        raise RuntimeError(msg) from e
