# Contributing to BiDoRA

Thank you for your interest in contributing to BiDoRA! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended)
- Git

### Getting Started

1. **Fork and clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/bidora.git
cd bidora
```

2. **Install dependencies**

```bash
# Using uv (recommended)
uv sync --dev

# This will:
# - Create a virtual environment
# - Install all dependencies
# - Install development dependencies (pytest, ruff, mypy)
# - Install bidora in editable mode
```

3. **Verify installation**

```bash
bidora --help
```

## Code Style

We use strict code quality tools to maintain consistency:

### Linting with Ruff

```bash
# Check for issues
ruff check src/

# Auto-fix issues
ruff check --fix src/
```

### Type Checking with mypy

```bash
mypy src/
```

**Code style requirements:**
- Line length: 100 characters
- Type hints required for all functions
- Python 3.11+ syntax
- Follow PEP 8 naming conventions

## Making Changes

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/` - New features (e.g., `feature/add-gpt-support`)
- `bugfix/` - Bug fixes (e.g., `bugfix/fix-memory-leak`)
- `refactor/` - Code refactoring (e.g., `refactor/simplify-config`)
- `docs/` - Documentation updates (e.g., `docs/update-readme`)

### Commit Messages

Write clear, descriptive commit messages:

```
Add support for GPT models

- Implement GPT tokenizer integration
- Add configuration options for GPT variants
- Update documentation with GPT examples
```

**Guidelines:**
- Use present tense ("Add feature" not "Added feature")
- First line should be a summary (50 chars or less)
- Add detailed description after blank line if needed
- Reference issues/PRs when relevant (#123)

### Code Standards

**Imports:**
- Use top-level imports only
- No try/except import blocks
- Group imports: stdlib, third-party, local

**Configuration:**
- Use Pydantic models for all config
- Validate inputs with type hints
- Provide sensible defaults

**Error Handling:**
- Use specific exception types
- Provide clear error messages
- Include context in error logs

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=bidora --cov-report=html

# Run specific test file
pytest tests/test_config.py
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use descriptive test names
- Include docstrings for complex tests

Example:

```python
def test_auto_hardware_adjustment_for_low_vram():
    """Test that config adjusts correctly for systems with <8GB VRAM."""
    config = FullConfig(...)
    config.auto_adjust_for_hardware()
    assert config.model.quantization == QuantizationMode.NF4
    assert config.training.batch_size <= 2
```

## Pull Request Process

1. **Update documentation**
   - Update README.md if adding features
   - Update CHANGELOG.md with your changes
   - Add/update docstrings

2. **Run quality checks**
   ```bash
   ruff check src/
   mypy src/
   pytest
   ```

3. **Create pull request**
   - Use a clear, descriptive title
   - Reference related issues
   - Describe what changes you made and why
   - Include examples if adding features

4. **PR template**
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update

   ## Testing
   Describe how you tested your changes

   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   - [ ] All tests pass
   ```

## Reporting Issues

### Bug Reports

Include:
- BiDoRA version (`pip show bidora`)
- Python version
- Operating system
- Hardware (GPU model, VRAM)
- Complete error message/traceback
- Minimal code to reproduce
- Expected vs actual behavior

### Feature Requests

Include:
- Clear description of the feature
- Use case / motivation
- Example API or usage
- Potential implementation approach (optional)

## Development Workflow

### Typical Workflow

1. Create feature branch from `master`
2. Make changes following guidelines
3. Run tests and quality checks
4. Commit with descriptive messages
5. Push to your fork
6. Create pull request
7. Address review feedback
8. Merge after approval

### Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions will build and publish to PyPI
6. Create GitHub Release with notes

## Questions?

- Open an issue for questions
- Check existing issues and discussions
- Review documentation in README.md

## Code of Conduct

Be respectful, inclusive, and professional. We aim to maintain a welcoming community for all contributors.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
