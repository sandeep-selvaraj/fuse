import json
from pathlib import Path
from typing import Annotated, Any

import typer
import yaml
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
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to extraction config YAML/JSON",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            "-m",
            help="GGUF path or HuggingFace repo name",
        ),
    ] = None,
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
    spans: Annotated[
        bool,
        typer.Option("--spans", help="Include evidence spans with source text localization"),
    ] = False,
    html_out: Annotated[
        Path | None,
        typer.Option("--html", help="Write highlighted HTML visualization to file"),
    ] = None,
) -> None:
    """Extract structured data from text using a GGUF model.

    Use --config for config-driven extraction, or --model with --schema/--fields
    for ad-hoc extraction. The model can be a local GGUF path or a HuggingFace
    repo name (auto-downloads the best GGUF quant).
    """
    use_spans = spans or html_out is not None
    if config:
        _extract_from_config(text, config, spans=use_spans, html_out=html_out)
    elif model:
        _extract_from_flags(
            text,
            model,
            schema,
            fields,
            prompt_format,
            max_tokens,
            spans=use_spans,
            html_out=html_out,
        )
    else:
        console.print("[red]Provide --config or --model[/red]")
        raise typer.Exit(code=1)


def _extract_from_config(
    text: str,
    config_path: Path,
    *,
    spans: bool = False,
    html_out: Path | None = None,
) -> None:
    """Run extraction driven by a YAML/JSON config file."""
    from fuse.config import ExtractConfig
    from fuse.extraction.extractor import Extractor
    from fuse.inference.llama_cpp import LlamaCppBackend

    with open(config_path) as f:
        raw = yaml.safe_load(f) if config_path.suffix in (".yaml", ".yml") else json.load(f)

    cfg = ExtractConfig(**raw)
    backend = LlamaCppBackend.from_config(cfg.model)
    extractor = Extractor(backend, prompt_format=cfg.prompt_format)

    if cfg.schema_file:
        from fuse.extraction.schema import SchemaBuilder

        with open(cfg.schema_file) as f:
            json_schema = json.load(f)
        model_cls = SchemaBuilder.from_json_schema(json_schema)
        if spans:
            spanned = extractor.extract_with_spans(text, model_cls, max_tokens=cfg.max_tokens)
            _output_spanned(text, spanned, html_out)
        else:
            result = extractor.extract(text, model_cls, max_tokens=cfg.max_tokens)
            _print_result(result.model_dump())
    elif cfg.fields:
        parsed = _parse_config_fields(cfg.fields)
        if spans:
            spanned = extractor.extract_from_fields_with_spans(
                text, parsed, max_tokens=cfg.max_tokens
            )
            _output_spanned(text, spanned, html_out)
        else:
            result_dict = extractor.extract_from_fields(text, parsed, max_tokens=cfg.max_tokens)
            _print_result(result_dict)
    elif cfg.description:
        result_dict = extractor.extract_from_description(
            text, cfg.description, max_tokens=cfg.max_tokens
        )
        _print_result(result_dict)
    else:
        console.print("[red]Config must specify schema_file, fields, or description[/red]")
        raise typer.Exit(code=1)


def _extract_from_flags(
    text: str,
    model: str,
    schema: Path | None,
    fields: str | None,
    prompt_format: str,
    max_tokens: int,
    *,
    spans: bool = False,
    html_out: Path | None = None,
) -> None:
    """Run extraction from CLI flags."""
    from fuse.extraction.extractor import Extractor
    from fuse.inference.llama_cpp import LlamaCppBackend

    backend = LlamaCppBackend(model_path=model)
    extractor = Extractor(backend, prompt_format=prompt_format)

    if schema:
        from fuse.extraction.schema import SchemaBuilder

        with open(schema) as f:
            json_schema = json.load(f)
        model_cls = SchemaBuilder.from_json_schema(json_schema)
        if spans:
            spanned = extractor.extract_with_spans(text, model_cls, max_tokens=max_tokens)
            _output_spanned(text, spanned, html_out)
        else:
            result = extractor.extract(text, model_cls, max_tokens=max_tokens)
            _print_result(result.model_dump())
    elif fields:
        parsed_fields = _parse_field_spec(fields)
        if spans:
            spanned = extractor.extract_from_fields_with_spans(
                text, parsed_fields, max_tokens=max_tokens
            )
            _output_spanned(text, spanned, html_out)
        else:
            result_dict = extractor.extract_from_fields(text, parsed_fields, max_tokens=max_tokens)
            _print_result(result_dict)
    else:
        console.print("[red]Provide --schema, --fields, or use --config[/red]")
        raise typer.Exit(code=1)


@app.command()
def train(
    config: Annotated[
        Path,
        typer.Option("--config", "-c", help="Path to training config YAML/JSON"),
    ],
) -> None:
    """Fine-tune a model using Unsloth or HuggingFace."""
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
    """Parse a CLI field spec like 'name:str,age:int' into a dict."""
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


def _parse_config_fields(
    fields: dict[str, str],
) -> dict[str, type | tuple[type, Any]]:
    """Parse config field mapping (e.g. {'name': 'str'}) to Python types."""
    type_map: dict[str, type] = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list[str],
        "list[str]": list[str],
        "list[int]": list[int],
    }
    return {name: type_map.get(t, str) for name, t in fields.items()}


def _print_result(data: dict) -> None:
    """Pretty-print extraction results as a table."""
    table = Table(title="Extraction Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    for key, value in data.items():
        table.add_row(key, str(value))
    console.print(table)


def _output_spanned(text: str, result: Any, html_out: Path | None) -> None:
    """Print spanned result table and optionally write HTML."""
    _print_spanned_result(result)
    if html_out:
        from fuse.extraction.visualize import render_html

        html_content = render_html(text, result)
        html_out.write_text(html_content)
        console.print(f"[bold green]HTML written to {html_out}[/bold green]")


def _print_spanned_result(result: Any) -> None:
    """Pretty-print a SpannedResult with evidence and spans."""
    from fuse.extraction.spans import SpannedResult

    assert isinstance(result, SpannedResult)
    table = Table(title="Extraction Result (with spans)")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_column("Evidence", style="yellow")
    table.add_column("Type", style="magenta")
    table.add_column("Span", style="dim")
    for field in result.fields:
        span_str = f"{field.span.start}:{field.span.end}" if field.span else "-"
        table.add_row(
            field.name,
            str(field.value),
            field.evidence or "-",
            "explicit" if field.is_explicit else "implicit",
            span_str,
        )
    console.print(table)
