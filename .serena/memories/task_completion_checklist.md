# Task Completion Checklist

When a coding task is completed, perform these checks:

## 1. Code Quality
- [ ] Run `ruff check src/` - No linting errors
- [ ] Run `mypy src/` - No type errors
- [ ] Code follows clean slate principles (no try/except imports, no version checks)
- [ ] Type hints added to all new functions/methods
- [ ] Used `uv` for any package operations (NOT pip)

## 2. Testing
- [ ] Run `pytest` if tests exist
- [ ] Test CLI commands if CLI was modified:
  ```bash
  bidora info
  bidora list-models
  bidora train --help
  ```

## 3. Documentation
- [ ] Update CLAUDE.md if architecture/workflows changed
- [ ] Update README.md if user-facing features changed
- [ ] Update USAGE.md if new use cases added
- [ ] Update INSTALL.md if installation process changed

## 4. File Management
- [ ] Ask before creating new files
- [ ] Prefer editing existing files over creating new ones
- [ ] No unnecessary markdown/documentation files created

## 5. Commit (if requested)
- [ ] Run all checks above
- [ ] Use meaningful commit message
- [ ] Use uv for any package changes (committed in pyproject.toml)
- [ ] No secrets in commits (.env, credentials, etc.)

## Common Mistakes to Avoid
- ❌ Using `pip install` instead of `uv pip install` or `uv add`
- ❌ Adding try/except import blocks
- ❌ Missing type hints
- ❌ Creating files without asking
- ❌ Line length > 100 characters
- ❌ Committing without running ruff/mypy first
