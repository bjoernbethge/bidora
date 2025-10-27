# Code Style and Conventions

## General Principles
- **Clean Slate Approach**: Use modern APIs directly, no try/except imports, no version checks
- **Type Safety**: Use type hints everywhere (mypy strict mode enabled)
- **Direct Imports**: No compatibility wrappers or conditional imports

## Python Standards
- **Python Version**: Target 3.11+ (currently 3.13.2)
- **Line Length**: 100 characters (ruff configured)
- **Type Hints**: Required everywhere
- **Package Manager**: ALWAYS use `uv`, NEVER use `pip`

## Code Organization
- **Configuration**: All config uses Pydantic models with validation
- **Imports**: Top-level, explicit, direct - no try/except blocks
- **Error Handling**: Direct modern APIs, no backward compatibility layers

## Linting & Formatting
- **Linter**: ruff (configured in pyproject.toml)
- **Type Checker**: mypy (strict mode)
- **Selected Rules**: E, F, I, N, W (E501 ignored for line length)

## Naming Conventions
- **Classes**: PascalCase (e.g., `ModelConfig`, `BiDoRAConfig`)
- **Functions**: snake_case (e.g., `load_model_and_tokenizer`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `QuantizationMode`)
- **Private**: Leading underscore for internal functions/vars

## Documentation
- Type hints serve as primary documentation
- Docstrings for public APIs and complex logic
- README.md, INSTALL.md, USAGE.md for user-facing docs
- CLAUDE.md for AI assistant guidance
