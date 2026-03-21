import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="fuse",
    help="Train small LLMs and deploy them for fast structured extraction on CPU.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def extract(
    text: Annotated[str, typer.Argument(help="Text to extract from")],
    model: Annotated[Path, typer.Option("--model", "-m", help="Path to GGUF model")],
    schema: Annotated[
        Path | None,
        typer.Option("--schema", "-s", help="Path to JSON schema file"),
    ] = None,
    fields: Annotated[
        str | None,
        typer.Option(
            "--fields",
            "-f",
            help='Comma-separated fields, e.g. "name:str,age:int"',
        ),
    ] = None,
    prompt_format: Annotated[
        str,
        typer.Option("--format", help="Prompt format: llama, chatml, generic"),
    ] = "llama",
    max_tokens: Annotated[int, typer.Option("--max-tokens", help="Max generation tokens")] = 512,
) -> None:
    """Extract structured data from text using a GGUF model."""
    from fuse.extraction.extractor import Extractor
    from fuse.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend(str(model))
    extractor = Extractor(backend, prompt_format=prompt_format)

    if schema:
        from fuse.extraction.schema import SchemaBuilder

        with open(schema) as f:
            json_schema = json.load(f)
        model_cls = SchemaBuilder.from_json_schema(json_schema)
        result = extractor.extract(text, model_cls, max_tokens=max_tokens)
        _print_result(result.model_dump())
    elif fields:
        parsed_fields = _parse_field_spec(fields)
        result_dict = extractor.extract_from_fields(text, parsed_fields, max_tokens=max_tokens)
        _print_result(result_dict)
    else:
        console.print("[red]Provide either --schema or --fields[/red]")
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to training config YAML/JSON"),
    ],
) -> None:
    """Fine-tune a model using Unsloth or HuggingFace."""
    import yaml

    from fuse.config import TrainConfig
    from fuse.training.trainer import Trainer

    with open(config) as f:
        raw = yaml.safe_load(f) if config.suffix in (".yaml", ".yml") else json.load(f)

    train_config = TrainConfig(**raw)
    trainer = Trainer(train_config)
    output_dir = trainer.train()
    console.print(f"[bold green]Model saved to {output_dir}[/bold green]")


@app.command()
def quantize(
    model_path: Annotated[Path, typer.Option("--model", "-m", help="Path to trained HF model")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output GGUF file path")],
    method: Annotated[str, typer.Option("--method", help="Quantization method")] = "q4_0",
) -> None:
    """Convert a trained model to GGUF format."""
    from fuse.config import ExportConfig, QuantMethod
    from fuse.training.export import export_to_gguf

    export_config = ExportConfig(
        model_path=model_path,
        output_path=output,
        quant_method=QuantMethod(method),
    )
    export_to_gguf(export_config)


def _parse_field_spec(spec: str) -> dict[str, type | tuple[type, Any]]:
    """Parse a CLI field spec like 'name:str,age:int,skills:list' into a dict."""
    type_map: dict[str, type | tuple[type, Any]] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list[str],
    }
    fields: dict[str, type | tuple[type, Any]] = {}
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            name, type_name = part.split(":", 1)
            fields[name.strip()] = type_map.get(type_name.strip(), str)
        else:
            fields[part] = str
    return fields


def _print_result(data: dict) -> None:
    """Pretty-print extraction results as a table."""
    table = Table(title="Extraction Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)
