"""Command-line interface for BiDoRA fine-tuning."""

from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import BiDoRAConfig, DataConfig, FullConfig, ModelConfig, TrainingConfig
from .data import load_and_prepare_dataset, prepare_dataset_for_training
from .model import get_hardware_info, load_model_and_tokenizer, prepare_bidora_model
from .trainer import train_bidora

app = typer.Typer(
    name="bidora",
    help="BiDoRA/LoRA fine-tuning for 3D code generation and spatial intelligence",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    train_file: Annotated[
        Optional[Path],
        typer.Option("--train-file", "-t", help="Training data JSONL file"),
    ] = None,
    val_file: Annotated[
        Optional[Path],
        typer.Option("--val-file", "-v", help="Validation data JSONL file (required for BiDoRA)"),
    ] = None,
    dataset_name: Annotated[
        Optional[str],
        typer.Option("--dataset", "-d", help="HuggingFace dataset name"),
    ] = None,
    model_name: Annotated[
        str,
        typer.Option("--model", "-m", help="Model name or path"),
    ] = "Qwen/Qwen2.5-Coder-7B-Instruct",
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory"),
    ] = Path("./output"),
    rank: Annotated[
        int,
        typer.Option("--rank", "-r", help="LoRA rank (lower = fewer params)"),
    ] = 8,
    epochs: Annotated[
        int,
        typer.Option("--epochs", "-e", help="Number of training epochs"),
    ] = 3,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Training batch size"),
    ] = 4,
    learning_rate: Annotated[
        float,
        typer.Option("--lr", help="Learning rate for lower level"),
    ] = 2e-4,
    upper_lr_multiplier: Annotated[
        float,
        typer.Option("--upper-lr-mult", help="Upper level LR multiplier"),
    ] = 2.0,
    max_samples: Annotated[
        Optional[int],
        typer.Option("--max-samples", help="Maximum training samples"),
    ] = None,
    auto_hardware: Annotated[
        bool,
        typer.Option("--auto-hardware", help="Auto-adjust config for hardware"),
    ] = True,
) -> None:
    """Fine-tune a model with BiDoRA bi-level optimization."""
    # Print header
    console.print(
        Panel.fit(
            "[bold magenta]BiDoRA Training[/bold magenta]\n"
            "[dim]Bi-level optimization with magnitude-direction decomposition[/dim]",
            border_style="magenta",
        )
    )

    # Validate input
    if train_file is None and dataset_name is None:
        console.print("[red]Error: Either --train-file or --dataset must be provided[/red]")
        raise typer.Exit(1)

    if val_file is None and train_file is not None:
        console.print("[red]Error: --val-file is required for BiDoRA training[/red]")
        console.print("[yellow]BiDoRA needs separate train and validation splits[/yellow]")
        raise typer.Exit(1)

    # Create config
    config = FullConfig(
        model=ModelConfig(model_name=model_name),
        bidora=BiDoRAConfig(
            rank=rank,
            use_bidora=True,
            upper_lr_multiplier=upper_lr_multiplier,
        ),
        training=TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=epochs,
        ),
        data=DataConfig(
            train_file=train_file,
            val_file=val_file,
            dataset_name=dataset_name,
            max_samples=max_samples,
        ),
        output_dir=output_dir,
    )

    # Auto-adjust for hardware
    if auto_hardware:
        console.print("[yellow]Auto-adjusting config for available hardware...[/yellow]")
        config.auto_adjust_for_hardware()

    # Show hardware info
    hw_info = get_hardware_info()
    hw_table = Table(title="Hardware Information")
    hw_table.add_column("Property", style="cyan")
    hw_table.add_column("Value", style="green")

    hw_table.add_row("CUDA Available", str(hw_info["cuda_available"]))
    if hw_info["cuda_available"]:
        hw_table.add_row("GPU Name", hw_info["gpu_name"])
        hw_table.add_row("GPU Memory", f"{hw_info['gpu_memory_gb']:.1f} GB")

    console.print(hw_table)

    # Show training config
    config_table = Table(title="BiDoRA Configuration")
    config_table.add_column("Parameter", style="cyan")
    config_table.add_column("Value", style="green")

    config_table.add_row("Model", config.model.model_name)
    config_table.add_row("Quantization", config.model.quantization.value)
    config_table.add_row("LoRA Rank", str(config.bidora.rank))
    config_table.add_row("Batch Size", str(config.training.batch_size))
    config_table.add_row("Lower LR", f"{config.training.learning_rate:.2e}")
    config_table.add_row("Upper LR", f"{config.training.learning_rate * upper_lr_multiplier:.2e}")
    config_table.add_row("Epochs", str(config.training.num_epochs))

    console.print(config_table)

    # Load model and tokenizer
    console.print("\n[yellow]Loading model and tokenizer...[/yellow]")
    model, tokenizer = load_model_and_tokenizer(config.model)

    # Prepare model with BiDoRA layers
    console.print("[yellow]Preparing model with BiDoRA layers...[/yellow]")
    model = prepare_bidora_model(
        model,
        config.bidora,
        quantized=config.model.quantization.value != "none",
    )

    # Load and prepare dataset
    console.print("[yellow]Loading and preparing dataset...[/yellow]")
    dataset = load_and_prepare_dataset(config.data)
    tokenized_dataset = prepare_dataset_for_training(
        dataset,
        tokenizer,
        config.training.max_seq_length,
    )

    console.print(f"[green]Training samples: {len(tokenized_dataset['train'])}[/green]")
    console.print(f"[green]Validation samples: {len(tokenized_dataset['validation'])}[/green]")

    # Train with BiDoRA
    console.print("\n[yellow]Starting BiDoRA bi-level training...[/yellow]")
    trainer = train_bidora(model, tokenizer, tokenized_dataset, config)

    # Show results
    console.print(
        Panel.fit(
            f"[bold green]BiDoRA training completed![/bold green]\n"
            f"Model saved to: [cyan]{config.output_dir / 'final_model'}[/cyan]",
            border_style="green",
        )
    )



@app.command()
def info() -> None:
    """Show hardware and environment information."""
    hw_info = get_hardware_info()

    table = Table(title="System Information")
    table.add_column("Property", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("PyTorch Version", hw_info["pytorch_version"])
    table.add_row("CUDA Available", str(hw_info["cuda_available"]))

    if hw_info["cuda_available"]:
        table.add_row("CUDA Version", hw_info["cuda_version"])
        table.add_row("GPU Count", str(hw_info["gpu_count"]))
        table.add_row("GPU Name", hw_info["gpu_name"])
        table.add_row("GPU Memory", f"{hw_info['gpu_memory_gb']:.2f} GB")

    console.print(table)


@app.command()
def list_models() -> None:
    """List recommended models for different hardware."""
    table = Table(title="Recommended Models by Hardware")
    table.add_column("Hardware", style="cyan")
    table.add_column("Model", style="green")
    table.add_column("Size", style="yellow")
    table.add_column("VRAM", style="magenta")

    models = [
        ("Laptop CPU", "Qwen/Qwen2.5-Coder-1.5B-Instruct", "1.5B", "~6GB"),
        ("Laptop GPU (8GB)", "Qwen/Qwen2.5-Coder-7B-Instruct", "7B", "~4GB (4-bit)"),
        ("Desktop GPU (16GB)", "Qwen/Qwen2.5-Coder-14B-Instruct", "14B", "~8GB (4-bit)"),
        ("A100 (40GB)", "Qwen/Qwen2.5-Coder-32B-Instruct", "32B", "~16GB (4-bit)"),
        ("A100 (40GB)", "deepseek-ai/deepseek-coder-33b-instruct", "33B", "~18GB (4-bit)"),
    ]

    for hw, model, size, vram in models:
        table.add_row(hw, model, size, vram)

    console.print(table)


if __name__ == "__main__":
    app()
