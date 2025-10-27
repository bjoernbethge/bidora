# Installation Guide

## Quick Install (Recommended)

### With uv (fast and modern)

```bash
# 1. Clone or download the package
cd bidora

# 2. Install with uv
uv pip install -e .

# 3. Verify installation
bidora info
```

### With pip (traditional)

```bash
# 1. Optional: Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# 2. Install
pip install -e .

# 3. Verify
bidora info
```

## System Requirements

### Minimum (CPU-only)
- Python 3.11+
- 16GB RAM
- 50GB Storage

### Recommended (GPU)
- Python 3.11+
- CUDA 12.1+
- 16GB+ RAM
- NVIDIA GPU with 8GB+ VRAM
- 50GB Storage

### Optimal (A100)
- Python 3.11+
- CUDA 12.1+
- 64GB+ RAM
- NVIDIA A100 (40GB or 80GB)
- 100GB Storage

## Dependency Installation

### CUDA Setup

**Linux:**
```bash
# Check CUDA version
nvcc --version

# If not installed:
# Follow: https://developer.nvidia.com/cuda-downloads
```

**Google Colab:**
```python
# CUDA is already installed
!nvidia-smi
```

### Flash Attention 2 (Optional but recommended)

```bash
pip install flash-attn --no-build-isolation
```

If error occurs:
```bash
# Flash Attention will be automatically disabled
# Training still works (just slightly slower)
```

## Verification

```bash
# 1. Check installation
bidora --help

# 2. Check hardware
bidora info

# 3. List models
bidora list-models

# 4. Run test training (optional)
bidora train \
  --train-file examples/rust_3d_train.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --max-samples 10 \
  --epochs 1
```

## Troubleshooting

### ImportError: No module named 'transformers'

```bash
# Reinstall dependencies
pip install --force-reinstall transformers accelerate peft bitsandbytes
```

### CUDA not available

```bash
# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, reinstall PyTorch with CUDA:
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### bitsandbytes error

```bash
# Linux:
pip install bitsandbytes

# Windows (experimental):
pip install bitsandbytes-windows
```

### Out of Memory during install

```bash
# Install without build isolation
pip install --no-build-isolation -e .
```

## Development Setup

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run type checking
mypy src/

# Run linting
ruff check src/
```

## Uninstall

```bash
pip uninstall bidora
```

---

For issues: See [USAGE.md](USAGE.md) for detailed troubleshooting steps.
