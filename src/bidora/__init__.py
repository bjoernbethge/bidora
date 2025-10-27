"""BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation."""

__version__ = "0.2.0"

from .optimizer import BiLevelOptimizer, create_bilevel_optimizer
from .trainer import BiDoRATrainer, train_bidora
from .config import (
    BiDoRAConfig,
    DataConfig,
    FullConfig,
    ModelConfig,
    ModelSize,
    QuantizationMode,
    TrainingConfig,
)
from .data import load_and_prepare_dataset, prepare_dataset_for_training
from .layers import BiDoRALinear, replace_linear_with_bidora
from .model import (
    create_bnb_config,
    get_hardware_info,
    load_model_and_tokenizer,
    prepare_bidora_model,
)

__all__ = [
    # Version
    "__version__",
    # Layers
    "BiDoRALinear",
    "replace_linear_with_bidora",
    # Optimization
    "BiLevelOptimizer",
    "create_bilevel_optimizer",
    # Training
    "BiDoRATrainer",
    "train_bidora",
    # Config
    "BiDoRAConfig",
    "DataConfig",
    "FullConfig",
    "ModelConfig",
    "ModelSize",
    "QuantizationMode",
    "TrainingConfig",
    # Data
    "load_and_prepare_dataset",
    "prepare_dataset_for_training",
    # Model
    "create_bnb_config",
    "get_hardware_info",
    "load_model_and_tokenizer",
    "prepare_bidora_model",
]
