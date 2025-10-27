"""Example: Training with BiDoRA bi-level optimization."""

from pathlib import Path
from datasets import Dataset, DatasetDict
import torch

from bidora import (
    BiDoRAConfig,
    FullConfig,
    ModelConfig,
    TrainingConfig,
    load_model_and_tokenizer,
    prepare_bidora_model,
    train_bidora,
)


def create_example_dataset() -> DatasetDict:
    """Create small example dataset."""
    train_data = {
        "text": [
            "use bevy::prelude::*;\\n\\nfn main() {\\n    App::new().run();\\n}",
            "use three_d::*;\\n\\nfn render_cube() {\\n    let cube = Cube::new();\\n}",
        ]
        * 10
    }
    val_data = {"text": ["use bevy::prelude::*;\\n\\nfn setup() {}"] * 5}
    return DatasetDict(
        {"train": Dataset.from_dict(train_data), "validation": Dataset.from_dict(val_data)}
    )


def main():
    """Train with BiDoRA."""
    config = FullConfig(
        model=ModelConfig(model_name="Qwen/Qwen3-1.7B", quantization="none"),  # BiDoRA needs full precision
        bidora=BiDoRAConfig(rank=8, use_bidora=True, upper_lr_multiplier=2.0),
        training=TrainingConfig(batch_size=2, num_epochs=2),
        output_dir=Path("./output/bidora_example"),
    )
    config.auto_adjust_for_hardware()  # Will force quantization=none

    model, tokenizer = load_model_and_tokenizer(config.model)
    model = prepare_bidora_model(model, config.bidora, quantized=False)  # No quantization with BiDoRA
    dataset = create_example_dataset()

    trainer = train_bidora(model, tokenizer, dataset, config)
    print(f"Training done! Model: {config.output_dir / 'final_model'}")


if __name__ == "__main__":
    main()
