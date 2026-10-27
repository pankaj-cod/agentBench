# AgentBench

AgentBench is an open-source developer tool for evaluating, debugging, and comparing
tool-using AI agents.

> **Status: early development.** The repository currently contains the project skeleton
> (packaging, linting, tests, CI). No evaluation features are implemented yet.

## Goals

AgentBench aims to help developers answer:

1. Does an agent complete a task correctly?
2. Which execution steps or tool calls cause failures?
3. How much latency and model usage does a run consume?
4. Does a new model, prompt, or agent implementation improve performance?
5. Can a failure be inspected and reproduced?
6. What are the trade-offs between task success, latency, and cost?

Core evaluation uses deterministic scoring rules and deterministic mock tools, so results
are reproducible and the test suite never depends on a live LLM API.

## Development setup

Requirements: [uv](https://docs.astral.sh/uv/). uv installs Python 3.12 automatically if
it is missing.

```bash
git clone https://github.com/pankaj-cod/agentBench.git
cd agentBench
uv sync
```

Run the checks (the same ones CI runs):

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

## Project layout

```
src/agentbench/   library code
tests/            pytest test suite
.github/          CI workflow
```

## Roadmap

- [x] Project skeleton: packaging, Ruff, pytest, CI
- [ ] Core data models: tasks, runs, steps, tool calls
- [ ] Deterministic mock tools with argument validation
- [ ] Minimal agent runner with step limits, timeouts, and bounded retries
- [ ] Trace recording
- [ ] Deterministic evaluator and scoring
- [ ] Persistence (SQLAlchemy + PostgreSQL)
- [ ] HTTP API (FastAPI)
- [ ] Dashboard (Streamlit)
- [ ] Docker Compose setup
- [ ] Real model adapter

## License

[MIT](LICENSE)
