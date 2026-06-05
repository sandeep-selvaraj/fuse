from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from rich.console import Console

from fuse.training.dataset import (
    load_dataset_from_hub,
    load_file_as_dataset,
    try_load_hub_split,
)

if TYPE_CHECKING:
    from pathlib import Path

    from fuse.config import TrainConfig

console = Console()


class Trainer:
    """Orchestrates fine-tuning of small LLMs using Unsloth or HuggingFace."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self._model: Any = None
        self._tokenizer: Any = None
        self._hf_trainer: Any = None

    def train(self) -> Path:
        """Run the full training pipeline.

        Returns:
            Path to the saved model directory.
        """
        console.print(f"[bold]Training {self.config.model_name}[/bold]")

        train_dataset, eval_dataset = self._load_data()
        self._load_model()
        self._run_training(train_dataset, eval_dataset)
        return self._save_model()

    def _load_data(self) -> tuple[Any, Any | None]:
        """Load the training set and, when available, a validation set.

        Returns a ``(train, eval)`` tuple where ``eval`` may be None. The
        validation set is resolved in priority order:

        1. An explicit eval dataset (``eval_dataset_path`` / ``eval_dataset_name``).
        2. A predefined validation split inside a Hub ``dataset_name``.
        3. A fraction carved from the training set via ``val_split_ratio``.
        """
        cfg = self.config
        if cfg.dataset_path:
            console.print(f"Loading dataset from [cyan]{cfg.dataset_path}[/cyan]")
            train = load_file_as_dataset(cfg.dataset_path)
        elif cfg.dataset_name:
            console.print(f"Loading dataset [cyan]{cfg.dataset_name}[/cyan] from Hub")
            train = load_dataset_from_hub(cfg.dataset_name, split=cfg.train_split)
        else:
            msg = "Either dataset_path or dataset_name must be provided."
            raise ValueError(msg)

        eval_dataset = self._resolve_explicit_eval()
        if eval_dataset is None and cfg.val_split_ratio:
            console.print(
                f"Holding out [cyan]{cfg.val_split_ratio:.0%}[/cyan] of train for validation"
            )
            split = train.train_test_split(test_size=cfg.val_split_ratio, seed=cfg.seed)
            train, eval_dataset = split["train"], split["test"]

        return train, eval_dataset

    def _resolve_explicit_eval(self) -> Any | None:
        """Resolve an explicit / predefined validation set, or None."""
        cfg = self.config
        if cfg.eval_dataset_path:
            console.print(f"Loading validation set from [cyan]{cfg.eval_dataset_path}[/cyan]")
            return load_file_as_dataset(cfg.eval_dataset_path)
        if cfg.eval_dataset_name:
            console.print(
                f"Loading validation set [cyan]{cfg.eval_dataset_name}[/cyan] "
                f"(split [cyan]{cfg.val_split}[/cyan]) from Hub"
            )
            return load_dataset_from_hub(cfg.eval_dataset_name, split=cfg.val_split)
        if cfg.dataset_name:
            eval_dataset = try_load_hub_split(cfg.dataset_name, cfg.val_split)
            if eval_dataset is not None:
                console.print(
                    f"Using predefined [cyan]{cfg.val_split}[/cyan] split from "
                    f"[cyan]{cfg.dataset_name}[/cyan]"
                )
            return eval_dataset
        return None

    def _load_model(self) -> None:
        if self.config.use_unsloth:
            self._load_with_unsloth()
        else:
            self._load_with_hf()

    def _load_with_unsloth(self) -> None:
        console.print("Loading model with [green]Unsloth[/green]")
        try:
            from unsloth import FastLanguageModel
        except ImportError:
            console.print("[yellow]Unsloth not available, falling back to HuggingFace[/yellow]")
            self._load_with_hf()
            return

        self._model, self._tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config.model_name,
            max_seq_length=self.config.max_seq_length,
            load_in_4bit=True,
        )
        self._model = FastLanguageModel.get_peft_model(
            self._model,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def _load_with_hf(self) -> None:
        console.print("Loading model with [blue]HuggingFace Transformers[/blue]")
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype="auto",
        )

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        )
        self._model = get_peft_model(self._model, lora_config)

    def _run_training(self, train_dataset: Any, eval_dataset: Any | None = None) -> None:
        from trl import SFTConfig, SFTTrainer

        console.print("[bold green]Starting training...[/bold green]")

        output_dir = str(self.config.output_dir)
        report_to = self.config.report_to
        if report_to and report_to != ["none"]:
            console.print(f"Reporting metrics to [magenta]{', '.join(report_to)}[/magenta]")
        # transformers reads the TensorBoard log dir from this env var
        # (TrainingArguments.logging_dir is deprecated as of transformers 5.x).
        if self.config.logging_dir is not None:
            os.environ["TENSORBOARD_LOGGING_DIR"] = str(self.config.logging_dir)
        has_eval = eval_dataset is not None
        if has_eval:
            console.print("Validation set provided — reporting eval metrics each epoch")
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_seq_length,
            logging_steps=self.config.logging_steps,
            run_name=self.config.run_name,
            save_strategy="epoch",
            eval_strategy="epoch" if has_eval else "no",
            fp16=True,
            report_to=report_to,
        )

        trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=training_args,
        )
        trainer.train()
        # Exposed for inspection/testing (e.g. eval metrics in state.log_history).
        self._hf_trainer = trainer

    def _save_model(self) -> Path:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Always save the LoRA adapter + tokenizer (small; lets you resume or re-merge).
        console.print(f"Saving adapter to [cyan]{output_dir}[/cyan]")
        self._model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

        if self.config.export_after_train:
            self._export_for_inference(output_dir)

        console.print("[bold green]Training complete![/bold green]")
        return output_dir

    def _export_for_inference(self, output_dir: Path) -> None:
        """Export inference-ready artifacts after training.

        Produces:
        - ``<output_dir>/merged`` — a standalone (merged) HF model for GPU /
          transformers / vLLM inference.
        - ``<output_dir>/gguf`` — a quantized GGUF for CPU / llama.cpp, loadable
          by ``LlamaCppBackend``. Requires Unsloth; skipped with a note otherwise.
        """
        merged_dir = output_dir / "merged"
        console.print(f"Saving merged model for GPU/HF inference to [cyan]{merged_dir}[/cyan]")
        if hasattr(self._model, "save_pretrained_merged"):
            # Unsloth: merge LoRA into the base weights in one call.
            self._model.save_pretrained_merged(
                str(merged_dir), self._tokenizer, save_method="merged_16bit"
            )
        else:
            # HuggingFace/PEFT path: merge then save a full model.
            merged = self._model.merge_and_unload()
            merged.save_pretrained(str(merged_dir))
            self._tokenizer.save_pretrained(str(merged_dir))

        if self.config.quantize is None:
            return

        quant = self.config.quantize.value
        gguf_dir = output_dir / "gguf"
        if hasattr(self._model, "save_pretrained_gguf"):
            console.print(f"Exporting GGUF ([cyan]{quant}[/cyan]) for CPU/llama.cpp...")
            self._model.save_pretrained_gguf(
                str(gguf_dir), self._tokenizer, quantization_method=quant
            )
            console.print(
                f"GGUF written to [cyan]{gguf_dir}[/cyan] — load with "
                f"[green]LlamaCppBackend(model_path=...)[/green]"
            )
        else:
            console.print(
                "[yellow]GGUF export needs Unsloth; skipped. The merged HF model in "
                f"{merged_dir} can be used for GPU inference or converted to GGUF "
                "with llama.cpp's convert_hf_to_gguf.py.[/yellow]"
            )
