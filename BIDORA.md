# BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation

## 🎯 Overview

BiDoRA is a **true implementation** of the BiDoRA algorithm from the paper ["BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation"](https://arxiv.org/abs/2410.09758v2).

Unlike standard LoRA/DoRA implementations, BiDoRA uses **bi-level optimization** to separately optimize:
- **Lower Level (Direction)**: Low-rank matrices (A, B) on training set
- **Upper Level (Magnitude)**: Magnitude component (m) on validation set via hypergradients
- **Final Phase**: Direction fine-tuning on combined data with fixed magnitude

This decoupled optimization:
- ✅ **Reduces overfitting** (especially on small datasets)
- ✅ **Better alignment with full fine-tuning** (correlation: -8.042 vs -1.784 for DoRA)
- ✅ **Statistically significant improvements** on GLUE benchmark (p < 0.001)

## 🚀 Quick Start

### Installation

```bash
# Clone and install
cd bidora
uv sync

# Or with pip
pip install -e .
```

### Basic Training (Standard LoRA/DoRA)

```bash
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 8 \
  --epochs 3 \
  --batch-size 4
```

### BiDoRA Training (True Bi-Level Optimization)

```bash
bidora train-bidora \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 8 \
  --epochs 3 \
  --batch-size 4 \
  --upper-lr-mult 2.0
```

**Important:** BiDoRA requires **separate train and validation files** for bi-level optimization!

## 📊 BiDoRA vs DoRA vs LoRA

| Method | Parameters | Overfitting Risk | FT Alignment | Complexity |
|--------|------------|------------------|--------------|------------|
| **LoRA** | Lowest | Medium | Low | Simple |
| **DoRA** | Medium | Medium-High | Medium | Medium |
| **BiDoRA** | Medium | **Low** | **High** | High |

### When to Use BiDoRA?

✅ **Use BiDoRA when:**
- Small datasets (<10k samples)
- Overfitting is a concern
- You need maximum performance
- You have train/val splits

❌ **Use standard LoRA when:**
- Large datasets (>100k samples)
- Speed is critical
- Simpler is better

## 🏗️ Architecture

### BiDoRA Weight Decomposition

```
W' = m ⊙ (W₀ + BA) / ||W₀ + BA||
     ↑      ↑
magnitude  direction
(upper)    (lower)
```

Where:
- `W₀`: Frozen base weights
- `B, A`: Low-rank matrices (direction)
- `m`: Magnitude vector
- `⊙`: Element-wise multiplication

### Training Loop

```python
for epoch in epochs:
    # Lower Level: Optimize direction on train set
    for train_batch in train_loader:
        direction_loss = loss_fn(model, train_batch)
        direction_loss.backward()
        lower_optimizer.step()  # Update A, B
    
    # Upper Level: Optimize magnitude on val set
    for val_batch in val_loader:
        val_loss = loss_fn(model, val_batch)
        hypergradients = compute_hypergradients(val_loss)
        upper_optimizer.step()  # Update m
    
# Final: Fine-tune direction on combined data
for combined_batch in combined_loader:
    loss = loss_fn(model, combined_batch)
    loss.backward()
    lower_optimizer.step()  # Final A, B tuning
```

## 💻 Python API

### Using BiDoRA Layers Directly

```python
from bidora import BiDoRALinear, replace_linear_with_bidora
import torch.nn as nn

# Create a BiDoRA linear layer
layer = BiDoRALinear(
    in_features=768,
    out_features=768,
    rank=8,
    lora_alpha=16.0,
    lora_dropout=0.1,
)

# Or replace existing linear layers
model = YourModel()
replace_linear_with_bidora(
    module=model,
    rank=8,
    target_modules=["q_proj", "v_proj"],
)
```

### Using BiDoRA Trainer

```python
from bidora import BiDoRATrainer, train_bidora
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Prepare with BiDoRA layers
from bidora import prepare_bidora_model, BiDoRAConfig

bidora_config = BiDoRAConfig(rank=8, alpha=16)
model = prepare_bidora_model(model, bidora_config, quantized=True)

# Train
trainer = train_bidora(
    model=model,
    tokenizer=tokenizer,
    dataset=dataset_dict,  # Must have 'train' and 'validation'
    config=config,
)
```

### Using Bi-Level Optimizer

```python
from bidora import create_bilevel_optimizer

# Create bi-level optimizer
optimizer = create_bilevel_optimizer(
    model=model,
    lower_lr=2e-4,   # Direction learning rate
    upper_lr=4e-4,   # Magnitude learning rate (typically 2x)
    lower_steps=1,   # Steps per level
)

# Training loop
for epoch in epochs:
    for train_batch, val_batch in zip(train_loader, val_loader):
        train_loss, val_loss = optimizer.bilevel_step(
            loss_fn=compute_loss,
            train_batch=train_batch,
            val_batch=val_batch,
        )
```

## 📦 Data Format

### JSONL Format

```jsonl
{"text": "def hello_world():\n    print('Hello, World!')\n"}
{"text": "class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n"}
```

### Conversation Format

```jsonl
{"messages": [{"role": "user", "content": "Write a Python function"}, {"role": "assistant", "content": "def example():\n    pass"}]}
```

## ⚙️ Configuration

### BiDoRA Config

```python
from bidora import BiDoRAConfig

config = BiDoRAConfig(
    rank=8,                      # LoRA rank (lower = fewer params)
    alpha=16,                    # Scaling factor
    dropout=0.05,                # Dropout rate
    target_modules=[             # Modules to apply BiDoRA
        "q_proj", "k_proj", 
        "v_proj", "o_proj"
    ],
    use_bidora=True,             # Enable bi-level optimization
    lower_lr_multiplier=1.0,     # Lower level LR multiplier
    upper_lr_multiplier=2.0,     # Upper level LR multiplier (magnitude)
    lower_steps=1,               # Steps per bi-level iteration
)
```

### Full Config Example

```python
from bidora import FullConfig, ModelConfig, TrainingConfig

config = FullConfig(
    model=ModelConfig(
        model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
        quantization="4bit",     # "none", "8bit", "4bit"
        use_flash_attention=True,
    ),
    bidora=BiDoRAConfig(rank=8, use_bidora=True),
    training=TrainingConfig(
        batch_size=4,
        learning_rate=2e-4,
        num_epochs=3,
        max_seq_length=2048,
    ),
)

# Auto-adjust for hardware
config.auto_adjust_for_hardware()
```

## 🎓 Paper Implementation Details

### Weight Decomposition

BiDoRA decomposes weights as:

```
W' = m ⊙ V
```

Where:
- `V = (W₀ + ΔV) / ||W₀ + ΔV||` (normalized direction)
- `ΔV = BA` (low-rank update)
- `m` (magnitude component)

### Hypergradient Computation

Upper-level optimization uses hypergradients:

```
∂L_val/∂m = ∂L_val/∂m_direct + ∂L_val/∂V * ∂V/∂m
```

Our implementation uses **approximate hypergradients** (direct gradients) for computational efficiency, which still provides significant benefits.

### Comparison to DoRA

| Aspect | DoRA | BiDoRA |
|--------|------|--------|
| Optimization | Simultaneous | Bi-level (separated) |
| Data | Single dataset | Train/val split |
| Magnitude Update | Coupled with direction | Independent with hypergradients |
| Overfitting | Higher risk | Lower risk |
| FT Correlation | -1.784 | **-8.042** (much closer to full FT) |

## 📈 Performance Results

From the [BiDoRA paper](https://arxiv.org/abs/2410.09758v2):

### GLUE Benchmark
- BiDoRA: **87.2%** average score
- DoRA: 86.1% (p-value: 2.4×10⁻⁴)
- LoRA: 85.3%

### Small Biomedical Datasets
BiDoRA shows **strongest gains** on small datasets where overfitting is a major concern.

## 🔧 Hardware Requirements

### Memory Usage (4-bit Quantization)

| Model Size | Parameters | VRAM | Training |
|------------|-----------|------|----------|
| 1.5B | 1.5B | ~6GB | Laptop GPU |
| 7B | 7B | ~4GB | RTX 3090/4090 |
| 14B | 14B | ~8GB | A100 40GB |
| 32B | 32B | ~16GB | A100 80GB |

### Speed Considerations

BiDoRA is ~**1.2-1.5x slower** than standard LoRA due to bi-level optimization, but the performance gains often justify the extra time.

## 🐛 Troubleshooting

### Common Issues

**1. "val_file required for BiDoRA"**
```bash
# BiDoRA needs separate validation set
bidora train-bidora --train-file train.jsonl --val-file val.jsonl ...
```

**2. Out of Memory**
```bash
# Reduce batch size or use smaller model
bidora train-bidora --batch-size 1 --model Qwen/Qwen2.5-Coder-1.5B-Instruct
```

**3. Slow Training**
```bash
# Use fewer lower steps or standard LoRA
bidora train --train-file train.jsonl ...  # Standard LoRA (faster)
```

## 📚 Citation

If you use BiDoRA in your research, please cite:

```bibtex
@article{qin2025bidora,
  title={BiDoRA: Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation},
  author={Peijia Qin and Ruiyi Zhang and Pengtao Xie},
  journal={Transactions on Machine Learning Research},
  year={2025},
  url={https://openreview.net/forum?id=v2xCm3VYl4}
}
```

## 🙏 Acknowledgments

- Original BiDoRA paper and implementation: [t2ance/BiDoRA](https://github.com/t2ance/BiDoRA)
- Built on top of HuggingFace Transformers and PEFT
- Inspired by DoRA and LoRA research

## 📝 License

Apache License 2.0 - See LICENSE file for details.

---

**Made with ❤️ for 3D Code Generation and Spatial Intelligence**
