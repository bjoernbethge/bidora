# Suggested Commands for BiDoRA Development

## Package Management (CRITICAL: Use uv, NOT pip!)
```bash
# Install package in editable mode
uv pip install -e .

# Install with dev dependencies
uv pip install -e ".[dev]"

# Add a new dependency
uv add <package-name>

# Reinstall dependencies
uv pip install --force-reinstall transformers accelerate peft bitsandbytes
```

## Quality Checks (Run before committing)
```bash
# Lint code with ruff
ruff check src/

# Type check with mypy
mypy src/

# Run tests (if available)
pytest

# Run all checks together
ruff check src/ && mypy src/ && pytest
```

## CLI Usage
```bash
# Show hardware info
bidora info

# List recommended models
bidora list-models

# Train with JSONL
bidora train --train-file data/train.jsonl --model Qwen/Qwen2.5-Coder-7B-Instruct

# Train with auto-hardware adjustment
bidora train --train-file data/train.jsonl --auto-hardware

# Get help
bidora --help
bidora train --help
```

## Development Workflow
```bash
# 1. Make changes to src/bidora/*.py
# 2. Run linting
ruff check src/

# 3. Run type checking
mypy src/

# 4. Test CLI (if changes affect CLI)
bidora info

# 5. Commit changes
git add .
git commit -m "Description"
```

## Windows-Specific Notes
- Use forward slashes in paths when possible: `bidora train --train-file data/train.jsonl`
- PowerShell is the default shell
- Python must be in PATH (disable Windows Store aliases!)
- Git Bash available for Unix-like commands
