# BiDoRA Usage Guide

Detailed guide for various hardware setups and use cases.

## 🖥️ Hardware-Specific Setups

### Laptop with Integrated GPU (Intel/AMD)

**Hardware:**
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 16GB+
- GPU: Integrated (no dedicated NVIDIA)

**Setup:**
```bash
# Use CPU-only with small model
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --rank 4 \
  --batch-size 1 \
  --epochs 3 \
  --max-samples 100
```

**Expectations:**
- Training Speed: ~5-10 min/epoch (100 samples)
- Memory: ~8GB RAM
- Quality: Good for simple code completion tasks

---

### Laptop with NVIDIA GPU (8GB VRAM)

**Hardware:**
- GPU: RTX 3060 Laptop, RTX 4060, GTX 1070
- VRAM: 6-8GB

**Setup:**
```bash
# 7B Model with 4-bit Quantization
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 8 \
  --batch-size 2 \
  --epochs 3 \
  --auto-hardware  # Automatically enables 4-bit
```

**VRAM Usage:**
- Model: ~4GB (4-bit)
- Training: ~7GB total
- Free: ~1GB Buffer

**Expectations:**
- Training Speed: ~15 min/epoch (1000 samples)
- Quality: Very good for Rust/3D code

**Optimization Tips:**
```bash
# If OOM (Out of Memory):
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 4 \           # Lower rank
  --batch-size 1 \     # Smaller batch size
  --epochs 3
```

---

### Desktop with NVIDIA GPU (16GB VRAM)

**Hardware:**
- GPU: RTX 3080, RTX 3090, RTX 4080
- VRAM: 12-16GB

**Setup:**
```bash
# 14B Model with 4-bit or 7B with 8-bit
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --rank 16 \
  --batch-size 4 \
  --epochs 3 \
  --auto-hardware
```

**Alternative: 7B without Quantization**
```bash
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 16 \
  --batch-size 8 \
  --no-auto-hardware  # Uses full precision (bf16)
```

**VRAM Usage:**
- 14B (4-bit): ~8GB model + ~12GB training = ~20GB (needs gradient checkpointing)
- 7B (bf16): ~14GB model + ~16GB training = ~30GB (tight!)

---

### Google Colab with A100 (40GB)

**Setup for maximum utilization:**
```bash
# 32B Model with 8-bit Quantization
bidora train \
  --train-file data/train.jsonl \
  --model Qwen/Qwen2.5-Coder-32B-Instruct \
  --rank 16 \
  --batch-size 8 \
  --epochs 3 \
  --auto-hardware
```

**VRAM Usage:**
- Model: ~16GB (8-bit)
- Training: ~35GB total
- Free: ~5GB Buffer

**Maximum Setup (uses full A100):**
```bash
bidora train \
  --train-file data/train.jsonl \
  --model deepseek-ai/deepseek-coder-33b-instruct \
  --rank 32 \
  --batch-size 16 \
  --epochs 5
```

---

## 🎯 Use-Case Specific Setups

### 1. Rust 3D Code Generation (drei-rs, Bevy)

**Prepare data:**
```json
{"instruction": "Create a Bevy 3D scene with cube", "output": "use bevy::prelude::*;\n\nfn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {\n    commands.spawn(PbrBundle {\n        mesh: meshes.add(Mesh::from(shape::Cube { size: 1.0 })),\n        ..default()\n    });\n}"}
{"instruction": "Generate drei-rs mesh loader", "output": "use drei::*;\n\npub fn load_mesh(path: &str) -> Result<Mesh> {\n    let data = std::fs::read(path)?;\n    Mesh::from_gltf(&data)\n}"}
```

**Training:**
```bash
bidora train \
  --train-file data/rust_3d.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 8 \
  --epochs 5 \
  --batch-size 4
```

**After Training - Inference:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, "./output/final_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

prompt = "### Instruction:\nCreate a drei-rs function for sphere mesh\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

### 2. Blender Python API Code

**Data:**
```json
{"instruction": "Add material to object in Blender", "output": "import bpy\n\ndef add_material(obj, name, color):\n    mat = bpy.data.materials.new(name=name)\n    mat.diffuse_color = (*color, 1.0)\n    obj.data.materials.append(mat)\n    return mat"}
{"instruction": "Create Blender camera setup", "output": "import bpy\nfrom mathutils import Vector\n\ndef setup_camera(location, target):\n    bpy.ops.object.camera_add(location=location)\n    camera = bpy.context.object\n    direction = target - Vector(location)\n    camera.rotation_euler = direction.to_track_quat('Z', 'Y').to_euler()\n    return camera"}
```

**Training:**
```bash
bidora train \
  --train-file data/blender_python.jsonl \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --rank 16 \
  --epochs 3
```

---

### 3. WGPU/WebGPU Shader Code

**Data:**
```json
{"instruction": "Write WGPU vertex shader for textured mesh", "output": "@vertex\nfn vs_main(\n    @location(0) pos: vec3<f32>,\n    @location(1) uv: vec2<f32>,\n) -> VertexOutput {\n    var out: VertexOutput;\n    out.position = camera.proj * camera.view * vec4<f32>(pos, 1.0);\n    out.uv = uv;\n    return out;\n}"}
```

**Training:**
```bash
bidora train \
  --train-file data/wgpu_shaders.jsonl \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --rank 8 \
  --epochs 5 \
  --batch-size 4
```

---

### 4. Point Cloud Processing

**Data:**
```json
{"instruction": "Convert point cloud to voxel grid in Rust", "output": "use nalgebra::Point3;\nuse std::collections::HashMap;\n\npub fn voxelize(points: &[Point3<f32>], voxel_size: f32) -> HashMap<(i32, i32, i32), Vec<Point3<f32>>> {\n    let mut grid = HashMap::new();\n    for p in points {\n        let voxel = (\n            (p.x / voxel_size).floor() as i32,\n            (p.y / voxel_size).floor() as i32,\n            (p.z / voxel_size).floor() as i32,\n        );\n        grid.entry(voxel).or_insert_with(Vec::new).push(*p);\n    }\n    grid\n}"}
```

---

## ⚡ Performance Optimization

### Gradient Checkpointing

Automatically enabled, saves ~40% VRAM:
```python
# Automatically in config:
config = FullConfig(...)
# gradient_checkpointing is always on
```

### Mixed Precision Training

Automatic through bfloat16:
```python
config = ModelConfig(...)
# Automatically uses bf16 when available
```

### Gradient Accumulation

For larger effective batch size:
```bash
bidora train \
  --batch-size 2 \           # Per-device batch
  --gradient-accumulation-steps 8  # Effective: 2 * 8 = 16
```

### Flash Attention 2

Automatically enabled when available:
```python
config = ModelConfig(
    use_flash_attention=True  # Default
)
```

---

## 🔧 Troubleshooting

### Problem: CUDA Out of Memory

**Solution 1: Smaller Batch Size**
```bash
bidora train --batch-size 1 --gradient-accumulation-steps 16 ...
```

**Solution 2: Lower LoRA Rank**
```bash
bidora train --rank 4 ...  # Instead of rank 8
```

**Solution 3: Shorter Sequences**
```bash
# Manual config:
config = TrainingConfig(
    max_seq_length=1024  # Instead of 2048
)
```

**Solution 4: Smaller Model**
```bash
# 7B instead of 14B:
bidora train --model Qwen/Qwen2.5-Coder-7B-Instruct ...
```

---

### Problem: Training too slow

**Check 1: GPU Utilization**
```bash
watch -n 1 nvidia-smi
# GPU-Util should be >90%
```

**Solution: Larger Batch Size**
```bash
bidora train --batch-size 8 ...  # If VRAM allows
```

---

### Problem: Model generates poor outputs

**Solution 1: More Training Epochs**
```bash
bidora train --epochs 5 ...  # Instead of 3
```

**Solution 2: Lower Learning Rate**
```bash
bidora train --lr 1e-4 ...  # Instead of 2e-4
```

**Solution 3: Higher LoRA Rank**
```bash
bidora train --rank 16 ...  # Instead of 8
```

**Solution 4: More Training Data**
```bash
# Collect more diverse examples!
```

---

## 📊 Expected Results

### Training Times (Laptop 8GB GPU, 7B Model)

| Samples | Epochs | Time |
|---------|--------|------|
| 100 | 3 | ~15 min |
| 500 | 3 | ~1 hour |
| 1000 | 3 | ~2 hours |
| 5000 | 3 | ~10 hours |

### Training Times (A100 40GB, 32B Model)

| Samples | Epochs | Time |
|---------|--------|------|
| 100 | 3 | ~5 min |
| 500 | 3 | ~20 min |
| 1000 | 3 | ~40 min |
| 5000 | 3 | ~3 hours |

---

## 🎓 Best Practices

### 1. Start Small
```bash
# First test with few samples:
bidora train --max-samples 100 --epochs 1 ...
```

### 2. Use Validation Split
```bash
# Always use validation for overfitting detection:
bidora train --val-file data/val.jsonl ...
# Or auto-split:
# config.data.val_split_ratio = 0.1
```

### 3. Save Checkpoints
```bash
# Automatically via save_steps:
config = TrainingConfig(
    save_steps=500  # Saves every 500 steps
)
```

### 4. Monitor Training
```bash
# Logs are written automatically:
tail -f output/logs/train.log
```

### 5. Test Before Full Training
```python
# Quick test after training:
model.eval()
test_prompt = "your test prompt"
outputs = model.generate(...)
print(tokenizer.decode(outputs[0]))
```

---

## 📈 Scaling Guidelines

| Dataset Size | Recommended Epochs | LoRA Rank | Model Size |
|--------------|-------------------|-----------|------------|
| < 100 | 5-10 | 4-8 | 1.5B-7B |
| 100-1000 | 3-5 | 8-16 | 7B-14B |
| 1000-10000 | 2-3 | 16-32 | 14B-32B |
| > 10000 | 1-2 | 32-64 | 32B+ |

---

Good luck with BiDoRA! 🚀
