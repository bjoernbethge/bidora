# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BiDoRA is a Python package for parameter-efficient fine-tuning of Large Language Models using BiDoRA/LoRA adapters, optimized for 3D code generation (Rust, Blender, CAD) and spatial intelligence tasks.

**Core Technologies:**
- Python 3.11+ with uv package manager
- PyTorch + Transformers + PEFT (LoRA/DoRA)
- Bitsandbytes for quantization (4-bit NF4, 8-bit, full precision)
- Pydantic for type-safe configuration
- Typer + Rich for CLI interface

## Package Structure

```
src/bidora/
├── config.py    # Pydantic models for all configuration (Model, BiDoRA, Training, Data)
├── model.py     # Model loading, quantization, PEFT adapter setup, hardware detection
├── data.py      # JSONL/HuggingFace dataset loading and tokenization
├── trainer.py   # Training loop with Transformers Trainer
└── cli.py       # Typer CLI commands (train, info, list-models)
```

**Key Architecture Pattern:**
Configuration flows through typed Pydantic models (`FullConfig` containing `ModelConfig`, `BiDoRAConfig`, `TrainingConfig`, `DataConfig`) → auto-adjustment for hardware → model loading with quantization → PEFT adapter application → training.

## Development Commands

### Installation & Setup
```bash
# Install in editable mode with uv (required)
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"
```

### Testing & Quality
```bash
# Run linting (ruff)
ruff check src/

# Run type checking (mypy)
mypy src/

# Run tests (if available)
pytest
```

### Running the CLI
```bash
# Show hardware info and recommendations
bidora info

# List recommended models
bidora list-models

# Train with JSONL files
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output ./output \
  --rank 8 \
  --epochs 3

# Train with HuggingFace dataset
bidora train --dataset "code_search_net" --model Qwen/Qwen2.5-Coder-7B-Instruct

# Train with DoRA (full weight decomposition)
bidora train --train-file data/train.jsonl --dora --rank 8
```

## Configuration System

All configuration uses Pydantic models with validation:

- **ModelConfig**: Model selection, quantization mode, Flash Attention
- **BiDoRAConfig**: LoRA rank, alpha, dropout, target modules, DoRA/RSLoRA flags
- **TrainingConfig**: Batch size, learning rate, epochs, gradient accumulation, max sequence length
- **DataConfig**: Train/val files or HuggingFace dataset, max samples, validation split ratio
- **FullConfig**: Combines all configs + `auto_adjust_for_hardware()` method

**Hardware Auto-adjustment:**
The `FullConfig.auto_adjust_for_hardware()` method detects available VRAM and automatically adjusts:
- Quantization mode (4-bit for <16GB, 8-bit for <40GB, full precision otherwise)
- Batch size
- Gradient accumulation steps
- Max sequence length

## Data Format Requirements

BiDoRA supports three JSONL formats:

1. **Instruction-tuning:** `{"instruction": "...", "input": "...", "output": "..."}`
2. **Code completion:** `{"prompt": "...", "completion": "..."}`
3. **Code-only:** `{"code": "..."}`

Preprocessing in `data.py` converts all formats to `{"text": "..."}` for tokenization.

## Code Style Guidelines

- Use type hints everywhere (mypy strict mode enabled)
- Pydantic models for all configuration
- Direct imports (no try/except import blocks)
- Use uv for package management (never pip)
- Line length: 100 characters (ruff)
- Target Python 3.11+

## Hardware Optimization Notes

**Memory Requirements:**
- 7B model (4-bit): ~4GB VRAM, ~12GB training
- 14B model (4-bit): ~8GB VRAM, ~20GB training
- 32B model (4-bit): ~16GB VRAM, ~40GB training

**Quantization Modes:**
- `QuantizationMode.NF4`: 4-bit NormalFloat (default, most memory efficient)
- `QuantizationMode.INT8`: 8-bit quantization
- `QuantizationMode.NONE`: Full precision (bf16)

**Flash Attention 2:**
Automatically enabled if available. Gracefully falls back to standard attention if not installed.

## Key Implementation Details

**PEFT Adapter Setup:**
- Uses RSLoRA by default (`use_rslora=True`) for better convergence
- Optional full DoRA decomposition (`use_dora=True`)
- Default target modules: `["q_proj", "k_proj", "v_proj", "o_proj"]`
- LoRA rank default: 8 (adjust based on VRAM)

**Training Configuration:**
- BF16 precision for training
- Gradient checkpointing enabled (saves memory)
- Cosine learning rate schedule
- AdamW optimizer with weight decay 0.01

**Tokenizer Setup:**
- Padding side: right
- Falls back to EOS token for pad token if not present
- Max sequence length configurable (default: 2048)

## Common Workflows

**Adding a new model:**
Update the model recommendations in `cli.py:list_models()` command.

**Modifying quantization:**
Edit `model.py:create_bnb_config()` for BitsAndBytesConfig adjustments.

**Changing data formats:**
Update `data.py:format_training_sample()` to support new JSONL field combinations.

**Adjusting auto-hardware logic:**
Modify `config.py:FullConfig.auto_adjust_for_hardware()` for different VRAM thresholds.

## Troubleshooting

**CUDA Out of Memory:**
- Use `--batch-size 1` and lower `--rank`
- Ensure 4-bit quantization is enabled (auto with `--auto-hardware`)

**Flash Attention Error:**
- Automatically disabled if unavailable
- Can manually disable via `use_flash_attention=False` in ModelConfig

**Import Errors:**
```bash
uv pip install --force-reinstall transformers accelerate peft bitsandbytes
```
