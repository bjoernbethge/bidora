# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-01-28

### Added
- Initial release of BiDoRA package
- BiDoRA bi-level optimization implementation for LLM fine-tuning
- Automatic hardware detection and configuration adjustment
- Support for multiple quantization modes:
  - 4-bit NormalFloat (NF4)
  - 8-bit quantization
  - Full precision (bfloat16)
- CLI interface with commands:
  - `bidora train` - Train models with BiDoRA/LoRA
  - `bidora info` - Show hardware information
  - `bidora list-models` - Display recommended models
- Data format support:
  - JSONL files (instruction-tuning, code completion, code-only)
  - HuggingFace datasets
- Type-safe configuration system with Pydantic
- DoRA (Weight-Decomposed LoRA) support
- RSLoRA for improved convergence
- Memory-efficient training with gradient checkpointing
- 8-bit AdamW optimizer for reduced memory usage
- Optimized for:
  - 3D code generation (Rust, Blender, CAD)
  - Spatial intelligence tasks
  - Small datasets (<10k samples)

### Documentation
- Comprehensive README with examples and hardware-specific setups
- Installation guide with PyPI, dependency, and development setup
- BibTeX citation for BiDoRA paper
- CLAUDE.md for project-specific AI assistance context

[0.1.0]: https://github.com/bjoernbethge/bidora/releases/tag/v0.1.0
