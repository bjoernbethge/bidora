# BiDoRA: Bi-Level Optimization for Parameter-Efficient Fine-Tuning

**BiDoRA** is a Python package implementing true BiDoRA (Bi-level Optimization-Based Weight-Decomposed Low-Rank Adaptation) for efficient fine-tuning of Large Language Models. Specifically optimized for:
- 3D Code Generation (Rust, Blender, CAD)
- Spatial Intelligence Tasks
- Small Datasets (<10k samples)
- Automatic Hardware Adaptation (Laptop to A100)

## 🔬 What is BiDoRA?

BiDoRA uses **bi-level optimization** to separately optimize magnitude and direction components of weight updates:

```
W' = m ⊙ (W₀ + BA) / ||W₀ + BA||
     ↑      ↑
  magnitude direction
  (upper)   (lower)
```

**Training Process:**
1. **Lower Level**: Optimize direction (A, B matrices) on training set
2. **Upper Level**: Optimize magnitude (m) on validation set via hypergradients
3. **Final Phase**: Direction fine-tuning on combined data with fixed magnitude

**Benefits:**
- ✅ Reduces overfitting on small datasets (<10k samples)
- ✅ Better alignment with full fine-tuning (correlation: -8.042 vs -1.784 for DoRA)
- ✅ Statistically significant improvements on GLUE (p < 0.001)

**Important Notes:**
- ⚠️ **Training Time**: 3-4x slower than standard LoRA due to bi-level optimization
- ⚠️ **No Quantization**: BiDoRA requires full precision (bfloat16) - quantization disabled automatically
- ⚠️ **Memory**: Uses 8-bit AdamW optimizer (75% memory reduction) to compensate
- ✅ **Best For**: Small specialized datasets where quality > speed

## 🚀 Features

- ✅ **BiDoRA Bi-Level Optimization**: True magnitude-direction decomposition
- ✅ **Auto Hardware Detection**: Automatically adapts config to available hardware
- ✅ **Multiple Quantization Modes**: 4-bit (NF4), 8-bit, Full Precision
- ✅ **Flexible Data Formats**: JSONL, HuggingFace Datasets
- ✅ **Type-Safe Config**: Pydantic-validated configuration
- ✅ **CLI Interface**: Simple command-line interface with Typer

## 📦 Installation

### With uv (recommended)

```bash
# Install package
uv pip install -e .

# Or directly via uv add
cd bidora
uv pip install -e .
```

### With pip

```bash
pip install -e .
```

## 🎯 Quick Start

### 1. Show hardware info

```bash
bidora info
```

Shows available hardware and recommended configuration.

### 2. Show recommended models

```bash
bidora list-models
```

### 3. Start BiDoRA training

**Important:** BiDoRA requires **separate train and validation files** for bi-level optimization.

#### Basic training

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-4B \
  --output ./output \
  --rank 8 \
  --epochs 3
```

#### With custom learning rates

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-4B \
  --lr 2e-4 \
  --upper-lr-mult 2.0 \
  --rank 8
```

#### With HuggingFace dataset

```bash
bidora train \
  --dataset "code_search_net" \
  --model Qwen/Qwen3-8B \
  --output ./output \
  --rank 8
```

## 📊 Data Format

### JSONL Format (Instruction-Tuning)

```json
{"instruction": "Generate a Rust function to create a 3D cube mesh", "output": "fn create_cube() -> Mesh { ... }"}
{"instruction": "Write Blender Python code to add a sphere", "input": "radius: 2.0", "output": "import bpy\nbpy.ops.mesh.primitive_uv_sphere_add(radius=2.0)"}
```

### JSONL Format (Code Completion)

```json
{"prompt": "// Generate 3D mesh\nfn create_mesh()", "completion": " -> Mesh {\n    let vertices = vec![...];\n    Mesh::new(vertices)\n}"}
```

### JSONL Format (Code-Only)

```json
{"code": "use bevy::prelude::*;\n\nfn setup_3d_scene(mut commands: Commands) { ... }"}
```

## ⚙️ Hardware-Specific Setups

### Laptop (8GB GPU)

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-4B \
  --rank 4 \
  --batch-size 1 \
  --auto-hardware  # Automatic adaptation
```

**Config automatically adjusted:**
- Quantization: 4-bit (NF4)
- Batch Size: 1-2
- Gradient Accumulation: 8-16
- Max Seq Length: 1024-2048

### Desktop (16GB GPU)

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-8B \
  --rank 16 \
  --batch-size 2 \
  --auto-hardware
```

**Auto-Config:**
- Quantization: 4-bit (NF4)
- Batch Size: 2-4
- Gradient Accumulation: 4-8
- Max Seq Length: 2048

### A100 (40GB)

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-32B \
  --rank 16 \
  --batch-size 8 \
  --auto-hardware
```

**Auto-Config:**
- Quantization: 8-bit
- Batch Size: 4-8
- Gradient Accumulation: 2-4
- Max Seq Length: 4096

## 🎛️ Advanced Options

### All CLI Parameters

```bash
bidora train --help
```

**Most Important Parameters:**

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model, -m` | Model name or path | `Qwen/Qwen3-4B` |
| `--train-file, -t` | Training JSONL | Required |
| `--val-file, -v` | Validation JSONL | **Required for BiDoRA** |
| `--dataset, -d` | HuggingFace Dataset | - |
| `--output, -o` | Output directory | `./output` |
| `--rank, -r` | LoRA Rank | `8` |
| `--epochs, -e` | Training Epochs | `3` |
| `--batch-size, -b` | Batch Size | `4` |
| `--lr` | Learning Rate (lower level) | `2e-4` |
| `--upper-lr-mult` | Upper level LR multiplier | `2.0` |
| `--max-samples` | Max Training Samples | All |
| `--auto-hardware` | Auto-adjustment | `True` |

### Manual Config (without Auto-Hardware)

```bash
bidora train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --model Qwen/Qwen3-8B \
  --rank 16 \
  --batch-size 8 \
  --lr 3e-4 \
  --epochs 5 \
  --no-auto-hardware  # Manual config
```

## 💾 Memory Requirements

### Qwen3 Model Sizes (BiDoRA - Full Precision)

⚠️ **Note**: BiDoRA requires full precision (bfloat16) - no quantization. Memory requirements higher than standard LoRA.

| Model | Parameter | VRAM (bf16) | Training VRAM | Recommended For |
|-------|-----------|-------------|---------------|-----------------|
| Qwen3-0.6B | 0.6B | ~2GB | ~6GB | Laptop GPU (6-8GB) |
| Qwen3-1.7B | 1.7B | ~4GB | ~10GB | **Laptop GPU (8GB+)** |
| Qwen3-4B | 4B | ~8GB | ~16GB | Desktop GPU (12-16GB) |
| Qwen3-8B | 8B | ~16GB | ~24GB | Desktop GPU (24GB+) / A100 |
| Qwen3-14B | 14B | ~28GB | ~40GB | A100 (40GB) |
| Qwen3-32B | 32B | ~64GB | ~80GB | A100 (80GB) |

💡 **Memory Optimization**: Uses 8-bit AdamW optimizer (75% memory reduction) to compensate for full precision requirement.

### Trainable Parameters (LoRA Rank=8)

| Base Model | LoRA Params | Reduction |
|------------|-------------|-----------|
| 7B | ~2M | **3500×** |
| 14B | ~4M | **3500×** |
| 32B | ~8M | **4000×** |

## 🧪 Example Workflow: 3D Rust Code Fine-Tuning

### 1. Prepare data

```bash
# data/rust_3d_train.jsonl
{"instruction": "Create a three-rs mesh for a cube", "output": "use three::*;\n\nfn create_cube(size: f32) -> Mesh {\n    let geometry = Geometry::cuboid(size, size, size);\n    Mesh::new(geometry, Material::default())\n}"}
{"instruction": "Generate Bevy 3D scene setup", "output": "use bevy::prelude::*;\n\nfn setup(mut commands: Commands) {\n    commands.spawn(Camera3dBundle::default());\n    commands.spawn(PbrBundle {\n        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),\n        ..default()\n    });\n}"}
```

### 2. Start training

```bash
bidora train \
  --train-file data/rust_3d_train.jsonl \
  --val-file data/rust_3d_val.jsonl \
  --model Qwen/Qwen3-4B \
  --output ./rust_3d_model \
  --rank 8 \
  --epochs 3 \
  --batch-size 2
```

### 3. Use model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load base model with BiDoRA adapters
model = AutoModelForCausalLM.from_pretrained(
    "./rust_3d_model/final_model",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")

# Generate
prompt = "### Instruction:\nCreate a three-rs function to render a sphere\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 🔧 Programmatic Usage

```python
from bidora import (
    FullConfig, ModelConfig, BiDoRAConfig, TrainingConfig, DataConfig,
    load_model_and_tokenizer, prepare_bidora_model,
    load_and_prepare_dataset, prepare_dataset_for_training,
    train_bidora
)
from pathlib import Path

# Create config
config = FullConfig(
    model=ModelConfig(
        model_name="Qwen/Qwen3-4B",
        quantization="4bit"
    ),
    bidora=BiDoRAConfig(
        rank=8,
        use_bidora=True,  # Enable BiDoRA bi-level optimization
        upper_lr_multiplier=2.0
    ),
    training=TrainingConfig(
        batch_size=2,
        learning_rate=2e-4,
        num_epochs=3
    ),
    data=DataConfig(
        train_file=Path("data/train.jsonl"),
        val_file=Path("data/val.jsonl")  # Required for BiDoRA
    ),
    output_dir=Path("./output")
)

# Auto-adjust for hardware
config.auto_adjust_for_hardware()

# Load model with BiDoRA layers
model, tokenizer = load_model_and_tokenizer(config.model)
model = prepare_bidora_model(model, config.bidora, quantized=True)

# Load data
dataset = load_and_prepare_dataset(config.data)
tokenized_dataset = prepare_dataset_for_training(
    dataset, tokenizer, config.training.max_seq_length
)

# Train with bi-level optimization
trainer = train_bidora(model, tokenizer, tokenized_dataset, config)
```

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
bidora train --batch-size 1 ...

# Or use smaller model
bidora train --model Qwen/Qwen3-1.7B ...

# Note: BiDoRA cannot use quantization (requires full precision)
```

### Flash Attention Error

If Flash Attention 2 is not available:
- Automatically disabled
- Or manually: Set `use_flash_attention=False` in ModelConfig

### Import Errors

```bash
# Reinstall dependencies
uv pip install --force-reinstall transformers accelerate peft bitsandbytes
```

## 📚 Further Resources

- [BiDoRA Paper](https://arxiv.org/abs/2410.09758) - Original bi-level optimization paper
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation
- [DoRA Paper](https://arxiv.org/abs/2402.09353) - Weight-Decomposed LoRA
- [Qwen3 Models](https://huggingface.co/collections/Qwen/qwen3-680edabfb790c8c34a242f95) - HuggingFace model collection

## 📝 License

MIT License - see LICENSE file.
