# BiDoRA Project Overview

## Purpose
BiDoRA is a Python package for parameter-efficient fine-tuning of Large Language Models (LLMs) using BiDoRA/LoRA adapters. Specifically optimized for:
- 3D Code Generation (Rust, Blender, CAD)
- Spatial Intelligence Tasks
- Automatic Hardware Adaptation (Laptop to A100)
- Minimal Memory Footprint through Quantization

## Tech Stack
- **Language**: Python 3.11+ (currently using 3.13.2)
- **Package Manager**: uv (NOT pip - this is critical!)
- **Core Libraries**:
  - PyTorch for deep learning
  - Transformers (HuggingFace) for LLM loading
  - PEFT for LoRA/DoRA adapters
  - Bitsandbytes for quantization (4-bit NF4, 8-bit)
  - Pydantic for type-safe configuration
  - Typer + Rich for CLI interface
  
## Package Structure
```
src/bidora/
├── config.py    # Pydantic models (ModelConfig, BiDoRAConfig, TrainingConfig, DataConfig)
├── model.py     # Model loading, quantization, PEFT adapter setup
├── data.py      # JSONL/HuggingFace dataset loading
├── trainer.py   # Training loop with Transformers Trainer
└── cli.py       # Typer CLI (train, info, list-models commands)
```

## Key Architecture Pattern
Configuration flows through typed Pydantic models → auto-adjustment for hardware → model loading with quantization → PEFT adapter application → training

## Entry Point
CLI command: `bidora` (installed via pyproject.toml scripts)
