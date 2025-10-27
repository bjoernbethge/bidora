"""BiDoRA trainer with bi-level optimization."""

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

from .optimizer import BiLevelOptimizer, create_bilevel_optimizer
from .config import FullConfig


class BiDoRATrainer:
    """
    BiDoRA trainer with bi-level optimization.

    Implements the training loop:
    1. Lower level: Optimize direction on train split
    2. Upper level: Optimize magnitude on val split
    3. Final: Fine-tune direction on combined data
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
        config: FullConfig,
        output_dir: Path,
    ):
        """
        Initialize BiDoRA trainer.

        Args:
            model: Model with BiDoRALinear layers
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Full configuration
            output_dir: Output directory for checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.output_dir = output_dir

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)

        # Create bi-level optimizer
        self.bilevel_optimizer = create_bilevel_optimizer(
            model=model,
            lower_lr=config.training.learning_rate,
            upper_lr=config.training.learning_rate * 2.0,  # Higher LR for magnitude
            lower_steps=1,
            weight_decay=config.training.weight_decay,
        )

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

    def _create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create DataLoader for dataset."""
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            collate_fn=self._collate_fn,
            num_workers=0,  # Set to 0 to avoid pickling issues
            pin_memory=torch.cuda.is_available(),
        )

    def _collate_fn(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate function for DataLoader."""
        # Check if batch is already tokenized
        if "input_ids" in batch[0]:
            # Already tokenized - just pad and convert to tensors
            input_ids = [torch.tensor(item["input_ids"]) for item in batch]
            attention_mask = [torch.tensor(item["attention_mask"]) for item in batch]
            labels = [torch.tensor(item["labels"]) for item in batch]

            # Pad sequences
            from torch.nn.utils.rnn import pad_sequence
            input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

            return {
                "input_ids": input_ids_padded,
                "attention_mask": attention_mask_padded,
                "labels": labels_padded,
            }
        else:
            # Not tokenized yet - tokenize now
            texts = [item["text"] for item in batch]
            tokenized = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.config.training.max_seq_length,
                return_tensors="pt",
            )
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized

    def _compute_loss(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute loss for a batch.

        Args:
            model: Model
            batch: Batch dictionary

        Returns:
            Loss tensor
        """
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)

        return outputs.loss

    def train(self) -> dict[str, Any]:
        """
        Run training loop.

        Returns:
            Training statistics
        """
        print(f"Starting BiDoRA training for {self.config.training.num_epochs} epochs")
        print(f"Train samples: {len(self.train_dataset)}")
        print(f"Val samples: {len(self.val_dataset)}")

        # Training loop
        for epoch in range(self.config.training.num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{self.config.training.num_epochs}")

            # Bi-level training phase
            train_metrics = self._train_epoch()

            # Validation phase
            val_metrics = self._validate()

            # Log metrics
            print(f"Train loss: {train_metrics['train_loss']:.4f}")
            print(f"Val loss: {val_metrics['val_loss']:.4f}")

            # Save checkpoint if best
            if val_metrics["val_loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["val_loss"]
                self._save_checkpoint("best_model")
                print(f"Saved best model (val_loss: {self.best_val_loss:.4f})")

            # Save periodic checkpoint
            if (epoch + 1) % self.config.training.save_steps == 0:
                self._save_checkpoint(f"checkpoint_epoch_{epoch + 1}")

        # Final fine-tuning phase
        print("\nFinal fine-tuning on combined data...")
        self._final_finetune()

        # Save final model
        self._save_checkpoint("final_model")
        print("Training complete!")

        return {
            "best_val_loss": self.best_val_loss,
            "final_epoch": self.epoch,
            "global_step": self.global_step,
        }

    def _train_epoch(self) -> dict[str, float]:
        """
        Train for one epoch using bi-level optimization.

        Returns:
            Training metrics
        """
        self.model.train()
        total_train_loss = 0.0
        total_val_loss = 0.0
        num_batches = 0

        # Create iterators
        train_iter = iter(self.train_loader)
        val_iter = iter(self.val_loader)

        # Progress bar
        pbar = tqdm(
            total=len(self.train_loader),
            desc=f"Epoch {self.epoch + 1}",
        )

        while True:
            try:
                # Get train and val batches
                train_batch = next(train_iter)
                val_batch = next(val_iter)

            except StopIteration:
                # Restart val iterator if exhausted
                if num_batches < len(self.train_loader):
                    val_iter = iter(self.val_loader)
                    val_batch = next(val_iter)
                else:
                    break

            # Bi-level optimization step
            train_loss, val_loss = self.bilevel_optimizer.bilevel_step(
                loss_fn=self._compute_loss,
                train_batch=train_batch,
                val_batch=val_batch,
            )

            # Accumulate losses
            total_train_loss += train_loss.item()
            total_val_loss += val_loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.update(1)
            pbar.set_postfix(
                {
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item(),
                }
            )

            # Log periodically
            if self.global_step % self.config.training.logging_steps == 0:
                avg_train_loss = total_train_loss / num_batches
                avg_val_loss = total_val_loss / num_batches
                print(
                    f"\nStep {self.global_step}: "
                    f"train_loss={avg_train_loss:.4f}, "
                    f"val_loss={avg_val_loss:.4f}"
                )

        pbar.close()

        return {
            "train_loss": total_train_loss / num_batches,
            "val_loss": total_val_loss / num_batches,
        }

    def _validate(self) -> dict[str, float]:
        """
        Run validation.

        Returns:
            Validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                loss = self._compute_loss(self.model, batch)
                total_loss += loss.item()
                num_batches += 1

        return {"val_loss": total_loss / num_batches}

    def _final_finetune(self) -> None:
        """
        Final fine-tuning phase on combined train+val data.

        This phase fine-tunes the direction parameters with fixed magnitude
        on the combined dataset.
        """
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset(
            [self.train_dataset, self.val_dataset]
        )
        combined_loader = DataLoader(
            combined_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # Fine-tune for 1 epoch
        for batch in tqdm(combined_loader, desc="Final fine-tuning"):
            loss = self.bilevel_optimizer.final_finetune_step(
                loss_fn=self._compute_loss,
                combined_batch=batch,
            )

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Final fine-tuning loss: {avg_loss:.4f}")

    def _save_checkpoint(self, checkpoint_name: str) -> None:
        """
        Save model checkpoint.

        Args:
            checkpoint_name: Name of checkpoint
        """
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(
            self.model.state_dict(),
            checkpoint_dir / "pytorch_model.bin",
        )

        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)

        # Save config
        config_path = checkpoint_dir / "config.json"
        config_path.write_text(
            self.config.model_dump_json(indent=2),
            encoding="utf-8"
        )

        print(f"Saved checkpoint to {checkpoint_dir}")


def train_bidora(
    model: nn.Module,
    tokenizer: PreTrainedTokenizer,
    dataset: DatasetDict,
    config: FullConfig,
) -> BiDoRATrainer:
    """
    Train model with BiDoRA bi-level optimization.

    Args:
        model: Model with BiDoRALinear layers
        tokenizer: Tokenizer
        dataset: Dataset dict with 'train' and 'validation' splits
        config: Full configuration

    Returns:
        Trained BiDoRATrainer instance
    """
    # Create trainer
    trainer = BiDoRATrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        val_dataset=dataset["validation"],
        config=config,
        output_dir=config.output_dir,
    )

    # Train
    trainer.train()

    return trainer
