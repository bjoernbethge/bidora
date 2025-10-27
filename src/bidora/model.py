"""Model loading with automatic quantization and hardware optimization."""

from typing import Any

import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from .config import BiDoRAConfig, ModelConfig, QuantizationMode


def create_bnb_config(quantization: QuantizationMode) -> BitsAndBytesConfig | None:
    """Create bitsandbytes quantization config."""
    if quantization == QuantizationMode.NONE:
        return None

    if quantization == QuantizationMode.INT8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
        )

    # NF4 quantization (default for memory efficiency)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def load_model_and_tokenizer(
    model_config: ModelConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load base model and tokenizer with quantization."""
    bnb_config = create_bnb_config(model_config.quantization)

    # Model loading kwargs
    model_kwargs: dict[str, Any] = {
        "pretrained_model_name_or_path": model_config.model_name,
        "device_map": "auto",
        "trust_remote_code": model_config.trust_remote_code,
    }

    # Set dtype (use 'dtype' instead of deprecated 'torch_dtype')
    if bnb_config is None:
        model_kwargs["dtype"] = torch.bfloat16

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    # Try Flash Attention 2 if requested
    if model_config.use_flash_attention:
        try:
            import flash_attn
            model_kwargs["attn_implementation"] = "flash_attention_2"
        except ImportError:
            # Flash Attention not available, use default attention
            # Don't set attn_implementation to let model use its default
            pass

    # Load model
    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=model_config.trust_remote_code,
        padding_side="right",
    )

    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer


def get_hardware_info() -> dict[str, Any]:
    """Get hardware information for logging."""
    info: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "gpu_count": torch.cuda.device_count(),
            }
        )

    return info


def prepare_bidora_model(
    model: PreTrainedModel,
    bidora_config: BiDoRAConfig,
    quantized: bool = True,
    use_gradient_checkpointing: bool = False,
) -> PreTrainedModel:
    """
    Prepare model with custom BiDoRA layers for bi-level optimization.

    Args:
        model: Base model
        bidora_config: BiDoRA configuration
        quantized: Whether model is quantized
        use_gradient_checkpointing: Enable gradient checkpointing (slower but saves memory)

    Returns:
        Model with BiDoRALinear layers
    """
    from .layers import replace_linear_with_bidora

    # Prepare for k-bit training if quantized
    if quantized:
        model = prepare_model_for_kbit_training(model)

    # DO NOT use gradient checkpointing with BiDoRA!
    # BiDoRA requires computing hypergradients (second-order derivatives) via forward-mode differentiation
    # Gradient checkpointing would make this even slower without saving meaningful memory
    # since the bi-level optimization already has high memory overhead from storing gradients
    if use_gradient_checkpointing:
        print("WARNING: Gradient checkpointing is not recommended with BiDoRA bi-level optimization!")

    # Replace Linear layers with BiDoRALinear
    replace_linear_with_bidora(
        module=model,
        rank=bidora_config.rank,
        lora_alpha=float(bidora_config.alpha),
        lora_dropout=bidora_config.dropout,
        target_modules=bidora_config.target_modules,
    )

    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_pct = 100 * trainable_params / total_params

    print(f"Trainable params: {trainable_params:,} || "
          f"All params: {total_params:,} || "
          f"Trainable%: {trainable_pct:.2f}%")

    return model
