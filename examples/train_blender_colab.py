"""Train Blender code generation model on Colab A100 with BiDoRA."""

from pathlib import Path
import json
from datasets import Dataset, DatasetDict

from bidora import (
    BiDoRAConfig,
    DataConfig,
    FullConfig,
    ModelConfig,
    TrainingConfig,
    load_model_and_tokenizer,
    prepare_bidora_model,
    train_bidora,
)


def load_blender_dataset(jsonl_path: Path, val_split: float = 0.1) -> DatasetDict:
    """Load JSONL dataset and split into train/val.

    Args:
        jsonl_path: Path to blender_dataset.jsonl
        val_split: Fraction of data for validation

    Returns:
        DatasetDict with train/validation splits
    """
    print(f"Loading dataset from {jsonl_path}...")

    # Load JSONL
    examples = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            example = json.loads(line)

            # Convert to text format
            if "instruction" in example and "output" in example:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
            elif "prompt" in example and "completion" in example:
                text = f"{example['prompt']}\n{example['completion']}"
            else:
                continue

            examples.append({"text": text})

    print(f"Loaded {len(examples)} examples")

    # Split into train/val
    split_idx = int(len(examples) * (1 - val_split))
    train_data = examples[:split_idx]
    val_data = examples[split_idx:]

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    return DatasetDict({
        "train": Dataset.from_dict({"text": [ex["text"] for ex in train_data]}),
        "validation": Dataset.from_dict({"text": [ex["text"] for ex in val_data]})
    })


def main():
    """Train Blender code generation model with BiDoRA on A100."""

    # Configuration optimized for Colab A100
    config = FullConfig(
        model=ModelConfig(
            model_name="Qwen/Qwen3-4B",  # 4B model for A100
            quantization="none",  # BiDoRA requires full precision
            use_flash_attention=True,  # A100 supports Flash Attention 2
        ),
        bidora=BiDoRAConfig(
            rank=16,  # Higher rank for better quality
            alpha=32,
            dropout=0.05,
            use_bidora=True,
            upper_lr_multiplier=2.0,
        ),
        training=TrainingConfig(
            batch_size=16,  # A100 can handle large batches
            gradient_accumulation_steps=2,
            learning_rate=2e-4,
            num_epochs=3,
            max_seq_length=2048,
            warmup_steps=100,
            save_steps=500,
            logging_steps=50,
        ),
        output_dir=Path("./output/blender_model_a100"),
        seed=42,
    )

    print("=" * 60)
    print("BiDoRA Training - Blender Code Generation")
    print("=" * 60)
    print(f"Model: {config.model.model_name}")
    print(f"Rank: {config.bidora.rank}")
    print(f"Batch Size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print("=" * 60)

    # Load Blender dataset
    dataset_path = Path("data/blender_dataset.jsonl")
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please upload blender_dataset.jsonl to data/ directory")
        return

    dataset = load_blender_dataset(dataset_path, val_split=0.1)

    # Load model
    print("\nLoading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(config.model)

    # Prepare BiDoRA
    print("Preparing BiDoRA layers...")
    model = prepare_bidora_model(model, config.bidora, quantized=False)

    # Train
    print("\nStarting BiDoRA training...")
    trainer = train_bidora(model, tokenizer, dataset, config)

    print("\n" + "=" * 60)
    print(f"Training complete!")
    print(f"Model saved to: {config.output_dir / 'final_model'}")
    print("=" * 60)

    # Save config for reference
    config_path = config.output_dir / "training_config.txt"
    with config_path.open("w") as f:
        f.write(f"Model: {config.model.model_name}\n")
        f.write(f"Rank: {config.bidora.rank}\n")
        f.write(f"Batch Size: {config.training.batch_size}\n")
        f.write(f"Epochs: {config.training.num_epochs}\n")
        f.write(f"Dataset: {len(dataset['train'])} train / {len(dataset['validation'])} val\n")

    print(f"\nConfig saved to: {config_path}")


if __name__ == "__main__":
    main()
