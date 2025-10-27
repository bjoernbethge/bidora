"""Dataset loading and preprocessing for code fine-tuning."""

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizer

from .config import DataConfig


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into list of dicts."""
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_dataset_from_files(data_config: DataConfig) -> DatasetDict:
    """Load dataset from JSONL files."""
    if data_config.train_file is None:
        raise ValueError("train_file must be provided")

    # Load training data
    train_data = load_jsonl(data_config.train_file)

    # Apply max_samples limit
    if data_config.max_samples is not None:
        train_data = train_data[: data_config.max_samples]

    # Load or split validation data
    if data_config.val_file is not None:
        val_data = load_jsonl(data_config.val_file)
        return DatasetDict(
            {
                "train": Dataset.from_list(train_data),
                "validation": Dataset.from_list(val_data),
            }
        )

    # Split training data for validation
    split_idx = int(len(train_data) * (1 - data_config.val_split_ratio))
    return DatasetDict(
        {
            "train": Dataset.from_list(train_data[:split_idx]),
            "validation": Dataset.from_list(train_data[split_idx:]),
        }
    )


def create_dataset_from_hub(data_config: DataConfig) -> DatasetDict:
    """Load dataset from HuggingFace Hub."""
    if data_config.dataset_name is None:
        raise ValueError("dataset_name must be provided")

    dataset = load_dataset(data_config.dataset_name, split=data_config.dataset_split)

    # Apply max_samples limit
    if data_config.max_samples is not None:
        dataset = dataset.select(range(min(data_config.max_samples, len(dataset))))

    # Split if only train split exists
    if isinstance(dataset, Dataset):
        split_idx = int(len(dataset) * (1 - data_config.val_split_ratio))
        return DatasetDict(
            {
                "train": dataset.select(range(split_idx)),
                "validation": dataset.select(range(split_idx, len(dataset))),
            }
        )

    return dataset


def load_and_prepare_dataset(data_config: DataConfig) -> DatasetDict:
    """Load dataset from files or hub."""
    if data_config.train_file is not None:
        return create_dataset_from_files(data_config)
    elif data_config.dataset_name is not None:
        return create_dataset_from_hub(data_config)
    else:
        raise ValueError("Either train_file or dataset_name must be provided")


def format_instruction_prompt(example: dict[str, Any]) -> str:
    """Format example into instruction prompt."""
    # Support various input formats
    if "instruction" in example and "output" in example:
        # Instruction-tuning format
        prompt = f"### Instruction:\n{example['instruction']}\n\n"
        if "input" in example and example["input"]:
            prompt += f"### Input:\n{example['input']}\n\n"
        prompt += f"### Response:\n{example['output']}"
        return prompt

    if "prompt" in example and "completion" in example:
        # Simple prompt-completion format
        return f"{example['prompt']}\n\n{example['completion']}"

    if "code" in example:
        # Code-only format (for pre-training style)
        return example["code"]

    if "text" in example:
        # Generic text format
        return example["text"]

    raise ValueError(f"Unknown data format. Keys: {example.keys()}")


def preprocess_function(
    examples: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> dict[str, list[Any]]:
    """Tokenize and prepare training examples."""
    # Format prompts
    texts = []
    for i in range(len(examples[next(iter(examples.keys()))])):
        example = {key: examples[key][i] for key in examples.keys()}
        texts.append(format_instruction_prompt(example))

    # Tokenize
    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Labels are the same as input_ids for causal LM
    model_inputs["labels"] = model_inputs["input_ids"].clone()

    return model_inputs


def prepare_dataset_for_training(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
) -> DatasetDict:
    """Tokenize and prepare dataset for training."""

    def tokenize_fn(examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return preprocess_function(examples, tokenizer, max_length)

    # Tokenize datasets
    tokenized_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing dataset",
    )

    return tokenized_dataset
