<!-- 828809ce-b16f-4904-a9fc-6a5a539aa446 202858bd-5148-4f55-90af-4691e522163c -->
# Claude-Code+ Developer Agent (CLI + Editor) for Pydantic‑AI with Ollama Cloud

### Goals

- CLI REPL with custom slash-commands and streaming.
- Minimal editor bridge (VS Code/Cursor): palette → diff preview → safe apply.
- Provider: Ollama “-cloud” models via OpenAI‑compatible endpoint.

### Better than Claude Code

- Safe writes by default: 3‑way diff → confirm → apply → rollback queue.
- Frontmatter macros (YAML) chainable; tag/priority scoped; session manifests in `.agent/`.
- Deterministic apply pipeline with typed tool outputs (Pydantic) and audit log.
- Cloud/local fallback: Ollama cloud first, auto‑fallback to local if desired.

### Architecture

- prompts/: frontmatter Markdown (`system`, `command`, `note`).
- agent/: Pydantic‑AI agent + tools (read/list/grep/diff/apply).
- cli/: Typer+Rich REPL with slash registry, streaming, diff confirmation.
- bridge/: tiny HTTP server (FastAPI or stdlib) for editor calls.
- vscode-ext/: thin client calling bridge; uses built‑in diff/apply UI.

### Provider Config (Ollama Cloud)

- OLLAMA endpoint must end with `/v1`.
- PowerShell (Windows):
- `$env:OLLAMA_BASE_URL = "https://api.ollama.ai/v1"`
- `$env:OLLAMA_API_KEY = "<your-key>"`
- `$env:OPENAI_BASE_URL = $env:OLLAMA_BASE_URL`
- `$env:OPENAI_API_KEY = $env:OLLAMA_API_KEY`
- Use installed models like `qwen3-coder:480b-cloud`, `gpt-oss:20b-cloud`.

### Quickstart

- Install: `uv add pydantic-ai httpx rich typer python-frontmatter`
- Run CLI: `uv run python -m cli.app`
- Run bridge: `uv run python -m bridge.server`

### Implementation Steps

1) Frontmatter loader: parse YAML; assemble system prompt by priority+tags.
2) Agent core (Pydantic‑AI): tools `read_file`, `list_files`, `grep`, `diff_files`, `apply_patch` (guarded), optional `run_cmd` (denylist).
3) CLI REPL: slash registry from prompts/commands; streaming output; writes require diff confirmation and rollback queue.
4) Provider wiring: `openai:<model>` with `OPENAI_BASE_URL` and key (Ollama cloud compatible); httpx retries/timeouts.
5) Editor bridge: endpoints for slash list, run prompt, get diff, apply; shared session store under `.agent/`.
6) Minimal VS Code extension: palette → bridge; diff preview; apply/abort.
7) Session logging: `.agent/sessions/<timestamp>.jsonl` + md transcript export.

### Deliverables

- `prompts/` sample files: `system/style.md`, `command/plan.md`, `note/python.md`.
- `agent/agent.py` with tools + provider config.
- `cli/app.py` for streaming REPL.
- `bridge/server.py` and minimal `vscode-ext/` skeleton.
- README with env and run instructions.

### To-dos

- [ ] Implement frontmatter loader and system prompt assembler by tags/priority
- [ ] Create Pydantic‑AI agent with typed tools (read/list/grep/diff/apply)
- [ ] Build Typer+Rich streaming REPL with slash registry and confirmations
- [ ] Add Ollama local/-cloud provider config via env with /v1 base URL
- [ ] Expose minimal HTTP bridge for prompts, diffs, apply, sessions
- [ ] Implement minimal VS Code extension calling the bridge with diff UI
- [ ] Persist session events, prompts, diffs; export to markdown