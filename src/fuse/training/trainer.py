from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.console import Console

from fuse.training.dataset import (
    format_for_sft,
    load_dataset_from_file,
    load_dataset_from_hub,
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

    def train(self) -> Path:
        """Run the full training pipeline.

        Returns:
            Path to the saved model directory.
        """
        console.print(f"[bold]Training {self.config.model_name}[/bold]")

        data = self._load_data()
        self._load_model()
        self._run_training(data)
        return self._save_model()

    def _load_data(self) -> Any:
        if self.config.dataset_path:
            console.print(f"Loading dataset from [cyan]{self.config.dataset_path}[/cyan]")
            raw = load_dataset_from_file(self.config.dataset_path)
            formatted = format_for_sft(raw)
            from datasets import Dataset

            return Dataset.from_list(formatted)

        if self.config.dataset_name:
            console.print(f"Loading dataset [cyan]{self.config.dataset_name}[/cyan] from Hub")
            return load_dataset_from_hub(self.config.dataset_name)

        msg = "Either dataset_path or dataset_name must be provided."
        raise ValueError(msg)

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

    def _run_training(self, dataset: Any) -> None:
        from trl import SFTConfig, SFTTrainer

        console.print("[bold green]Starting training...[/bold green]")

        output_dir = str(self.config.output_dir)
        training_args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            max_length=self.config.max_seq_length,
            logging_steps=10,
            save_strategy="epoch",
            fp16=True,
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=dataset,
            args=training_args,
        )
        trainer.train()

    def _save_model(self) -> Path:
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        console.print(f"Saving model to [cyan]{output_dir}[/cyan]")
        self._model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))

        console.print("[bold green]Training complete![/bold green]")
        return output_dir
