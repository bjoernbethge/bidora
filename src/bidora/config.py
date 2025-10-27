"""Configuration models for BiDoRA fine-tuning."""

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelSize(str, Enum):
    """Supported model sizes with automatic hardware detection."""

    TINY = "1b"  # Laptop CPU/small GPU
    SMALL = "7b"  # Laptop with 8GB+ VRAM
    MEDIUM = "14b"  # Laptop with 16GB+ VRAM or A100
    LARGE = "20b"  # A100 40GB
    XLARGE = "34b"  # A100 80GB


class QuantizationMode(str, Enum):
    """Quantization options for memory efficiency."""

    NONE = "none"  # Full precision (fp16/bf16)
    INT8 = "8bit"  # 8-bit quantization
    NF4 = "4bit"  # 4-bit NormalFloat


class BiDoRAConfig(BaseModel):
    """BiDoRA/LoRA adapter configuration."""

    rank: int = Field(default=8, ge=1, le=128, description="LoRA rank (lower = fewer params)")
    alpha: int = Field(default=16, ge=1, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.05, ge=0.0, le=0.5, description="LoRA dropout rate")
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
        description="Transformer modules to apply LoRA",
    )
    use_rslora: bool = Field(
        default=True, description="Use Rank-Stabilized LoRA (DoRA-style)"
    )
    use_dora: bool = Field(default=False, description="Use full DoRA weight decomposition")
    use_bidora: bool = Field(
        default=True, description="Use true BiDoRA with bi-level optimization"
    )
    lower_lr_multiplier: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Learning rate multiplier for lower level"
    )
    upper_lr_multiplier: float = Field(
        default=2.0, ge=0.1, le=10.0, description="Learning rate multiplier for upper level (magnitude)"
    )
    lower_steps: int = Field(
        default=1, ge=1, le=10, description="Number of lower-level steps per upper-level step"
    )

    @field_validator("target_modules")
    @classmethod
    def validate_target_modules(cls, v: list[str]) -> list[str]:
        valid_modules = {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        }
        for module in v:
            if module not in valid_modules:
                raise ValueError(f"Invalid target module: {module}. Must be one of {valid_modules}")
        return v


class TrainingConfig(BaseModel):
    """Training hyperparameters."""

    batch_size: int = Field(default=4, ge=1, le=128, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(
        default=4, ge=1, description="Gradient accumulation steps"
    )
    learning_rate: float = Field(
        default=2e-4, ge=1e-6, le=1e-3, description="Peak learning rate"
    )
    num_epochs: int = Field(default=3, ge=1, le=100, description="Number of training epochs")
    warmup_steps: int = Field(default=100, ge=0, description="Linear warmup steps")
    max_seq_length: int = Field(default=2048, ge=128, le=8192, description="Maximum sequence length")
    save_steps: int = Field(default=500, ge=1, description="Save checkpoint every N steps")
    logging_steps: int = Field(default=10, ge=1, description="Log metrics every N steps")
    eval_steps: int | None = Field(
        default=None, description="Evaluate every N steps (None = per epoch)"
    )
    weight_decay: float = Field(default=0.01, ge=0.0, le=1.0, description="Weight decay")
    max_grad_norm: float = Field(default=1.0, ge=0.0, description="Gradient clipping")


class ModelConfig(BaseModel):
    """Model selection and quantization."""

    model_name: str = Field(
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        description="HuggingFace model name or path",
    )
    quantization: QuantizationMode = Field(
        default=QuantizationMode.NF4, description="Quantization mode"
    )
    trust_remote_code: bool = Field(
        default=True, description="Trust remote code for custom models"
    )
    use_flash_attention: bool = Field(
        default=False, description="Use Flash Attention 2 if available"
    )


class DataConfig(BaseModel):
    """Dataset configuration."""

    train_file: Path | None = Field(default=None, description="Training data (JSONL)")
    val_file: Path | None = Field(default=None, description="Validation data (JSONL)")
    dataset_name: str | None = Field(
        default=None, description="HuggingFace dataset name (alternative to files)"
    )
    dataset_split: str = Field(default="train", description="Dataset split to use")
    val_split_ratio: float = Field(
        default=0.1, ge=0.0, le=0.5, description="Validation split ratio if val_file not provided"
    )
    max_samples: int | None = Field(
        default=None, description="Max training samples (None = all)"
    )

    @field_validator("train_file", "val_file")
    @classmethod
    def validate_file_exists(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"File not found: {v}")
        return v


class FullConfig(BaseModel):
    """Complete configuration for BiDoRA fine-tuning."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    bidora: BiDoRAConfig = Field(default_factory=BiDoRAConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output_dir: Path = Field(default=Path("./output"), description="Output directory")
    seed: int = Field(default=42, description="Random seed")
    resume_from_checkpoint: Path | None = Field(
        default=None, description="Resume from checkpoint"
    )

    def auto_adjust_for_hardware(self) -> None:
        """Automatically adjust config based on available hardware."""
        import torch

        # BiDoRA requires full precision (no quantization)
        if self.bidora.use_bidora:
            self.model.quantization = QuantizationMode.NONE

        if not torch.cuda.is_available():
            # CPU-only mode
            self.model.quantization = QuantizationMode.NONE
            self.training.batch_size = 1
            self.training.gradient_accumulation_steps = 16
            self.model.use_flash_attention = False
            return

        # Get GPU memory in GB
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        # Adjust settings based on VRAM (skip quantization if BiDoRA is enabled)
        if gpu_memory_gb < 8:
            # Small GPU (6-8GB)
            if not self.bidora.use_bidora:
                self.model.quantization = QuantizationMode.NF4
            self.training.batch_size = 1
            self.training.gradient_accumulation_steps = 16
            self.training.max_seq_length = 1024
        elif gpu_memory_gb < 16:
            # Medium GPU (8-16GB)
            if not self.bidora.use_bidora:
                self.model.quantization = QuantizationMode.NF4
            self.training.batch_size = 2
            self.training.gradient_accumulation_steps = 8
            self.training.max_seq_length = 2048
        elif gpu_memory_gb < 40:
            # Large GPU (16-40GB)
            if not self.bidora.use_bidora:
                self.model.quantization = QuantizationMode.INT8
            self.training.batch_size = 4
            self.training.gradient_accumulation_steps = 4
            self.training.max_seq_length = 2048
        else:
            # A100 or better (40GB+)
            if not self.bidora.use_bidora:
                self.model.quantization = QuantizationMode.INT8
            self.training.batch_size = 8
            self.training.gradient_accumulation_steps = 2
            self.training.max_seq_length = 4096
