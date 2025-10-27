# Architecture Patterns

## Configuration Flow Pattern
BiDoRA uses a centralized configuration system with Pydantic models:

```
FullConfig
├── ModelConfig (model selection, quantization, flash attention)
├── BiDoRAConfig (LoRA rank, alpha, dropout, target modules)
├── TrainingConfig (batch size, learning rate, epochs)
└── DataConfig (train/val files, dataset name, max samples)
```

**Key Method**: `FullConfig.auto_adjust_for_hardware()`
- Detects available VRAM
- Adjusts quantization mode (4-bit <16GB, 8-bit <40GB, none otherwise)
- Adjusts batch size, gradient accumulation, max sequence length

## Data Processing Pattern
All JSONL formats are normalized to `{"text": "..."}`:

1. **Instruction-tuning**: `{"instruction": "...", "input": "...", "output": "..."}`
2. **Code completion**: `{"prompt": "...", "completion": "..."}`
3. **Code-only**: `{"code": "..."}`

Preprocessing in `data.py` handles all conversions.

## Model Loading Pattern
```python
# 1. Create quantization config based on mode
bnb_config = create_bnb_config(quantization_mode)

# 2. Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

# 3. Prepare for training (gradient checkpointing, etc.)
model = prepare_model_for_kbit_training(model)

# 4. Apply PEFT adapters
peft_config = create_peft_config(bidora_config)
model = get_peft_model(model, peft_config)
```

## PEFT Adapter Pattern
- **RSLoRA by default**: Better convergence than standard LoRA
- **Optional DoRA**: Full weight decomposition with `use_dora=True`
- **Target modules**: `["q_proj", "k_proj", "v_proj", "o_proj"]` by default
- **Trainable params**: Only ~2-8M for 7B-32B models (3500-4000x reduction)

## Training Pattern
```python
# 1. Create training arguments from config
training_args = create_training_arguments(config, output_dir)

# 2. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val
)

# 3. Train and save
trainer.train()
trainer.save_model(output_dir / "final_model")
```

## Hardware Optimization Features
- **Gradient Checkpointing**: Always enabled (saves ~40% VRAM)
- **Mixed Precision**: BF16 for training
- **Flash Attention 2**: Auto-enabled if available, graceful fallback
- **Gradient Accumulation**: Configurable for effective larger batches
- **Quantization**: 4-bit NF4 (default), 8-bit, or full precision

## CLI Pattern
Built with Typer:
- **Commands**: `train`, `info`, `list-models`
- **Rich formatting**: Tables, panels for output
- **Validation**: Pydantic models validate all inputs
- **Help**: Auto-generated from Typer annotations
