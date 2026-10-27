# AgentBench

AgentBench is an open-source developer tool for evaluating, debugging, and comparing
tool-using AI agents.

> **Status: early development.** Working today: a deterministic agent runner with
> validated mock tools, ordered execution traces, a deterministic evaluator with
> documented metrics, benchmark comparisons between agent configurations, and PostgreSQL
> persistence. The HTTP API and the dashboard are not built yet. No real LLM is used.

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

## What works today

- **`AgentRunner`**: a controlled tool-calling loop.
  - Enforces a step limit, a run deadline, per-call timeouts, and bounded retries with
    exponential backoff.
  - Every run ends with exactly one termination reason.
  - A failed tool call is never reported as a success.
- **`ToolRegistry`**: rejects unknown tools, tools a task does not allow, malformed
  arguments, and unsafe tool definitions — all before anything runs.
- **Mock tools**: `search_records`, `lookup_record`, and a `calculator` that never uses
  `eval`.
- **Agents**: `RuleBasedAgent` follows a documented deterministic policy;
  `ScriptedAgent` replays fixed decisions for tests.
- **Trace recording**: every event has an event id, run id, gap-free sequence number,
  type, UTC timestamp, monotonic elapsed time, and a JSON payload.
- **`EvaluationEngine`**: scores each run against explicit per-case criteria:
  - expected answers, including numeric tolerance;
  - required tools with expected arguments;
  - forbidden tools;
  - expected retrieved records.

  No LLM judge. Each run gets one outcome — `PASSED`, `FAILED` (the agent's fault) or
  `INVALID` (an infrastructure fault) — plus a primary failure category and an
  explanation.
- **Metrics with stated numerators, denominators and exclusions**:
  - success rate;
  - tool selection correctness;
  - tool argument validity;
  - tool execution error rate;
  - timeout and step-limit rates;
  - latency p50/p95;
  - steps per run;
  - tokens and cost.

  Anything that can't be measured is reported as unavailable, never as 0.
- **Comparisons** between agent configurations: percentage-point changes for rates,
  absolute and relative changes for values, and per-case fixed/broken lists.
- **PostgreSQL persistence** of runs, traces and versioned evaluations, with Alembic
  migrations. The database enforces the same rules as the code.

How it works: [docs/runner.md](docs/runner.md), [docs/evaluation.md](docs/evaluation.md).
Design: [docs/architecture.md](docs/architecture.md), [docs/data-model.md](docs/data-model.md),
[docs/interfaces.md](docs/interfaces.md).

## Quick start (no database)

You need [uv](https://docs.astral.sh/uv/). It installs Python 3.12 automatically if it
is missing.

```bash
git clone https://github.com/pankaj-cod/agentBench.git
cd agentBench
uv sync
uv run agentbench run "What is 17% of the quantity of record A-12?"
```

```
status   COMPLETED (COMPLETED)
answer   20.4
steps    3, tool calls 2, 4.746 ms

seq  step        +ms  event                 detail
  0     0      0.067  RUN_STARTED           tools=['search_records', 'lookup_record', 'calculator'] max_steps=10
  1     1      4.181  AGENT_DECISION        call lookup_record {"record_id":"A-12"}
  2     1      4.245  TOOL_CALL_STARTED     call-1 attempt 1
  3     1      4.281  TOOL_CALL_SUCCEEDED   call-1 -> {"record":{"category":"hardware","id":"A-12",...
  4     2      4.353  AGENT_DECISION        call calculator {"expression":"120 * 17 / 100"}
  5     2      4.630  TOOL_CALL_STARTED     call-2 attempt 1
  6     2      4.685  TOOL_CALL_SUCCEEDED   call-2 -> {"expression":"120 * 17 / 100","result":20.4}
  7     3      4.722  AGENT_DECISION        final answer '20.4'
  8     3      4.731  FINAL_ANSWER          '20.4'
  9     3      4.745  RUN_FINISHED          COMPLETED
```

**Useful options:**
- `--max-steps 1` — watch a run stop with `MAX_STEPS_EXCEEDED`;
- `--tools calculator` — restrict which tools the agent may use;
- `--json` — print the full result as JSON.

The exit code is 0 only for `COMPLETED` runs.

## Saving runs to PostgreSQL

You also need Docker. Start a local PostgreSQL 17. It listens only on 127.0.0.1, and
its credentials are development defaults:

```bash
docker compose up -d --wait db
```

Point AgentBench at it:

```bash
export AGENTBENCH_DATABASE_URL=postgresql+psycopg://agentbench:agentbench@localhost:5432/agentbench
```

Create the tables:

```bash
uv run alembic upgrade head
```

Run a task and save it:

```bash
uv run agentbench run "What is the quantity of record A-12?" --save
```

Show a saved run again, using the run id printed above:

```bash
uv run agentbench show <run-id>
```

## Tests

Without a database, `uv run pytest` runs the unit tests and *skips* the PostgreSQL
tests, giving a reason. To run everything, start the database as above, then:

```bash
AGENTBENCH_TEST_DATABASE_URL=postgresql+psycopg://agentbench:agentbench@localhost:5432/agentbench uv run pytest
```

**How the database tests are isolated:**
- each test session creates a throwaway database, migrates it with Alembic, and drops it
  afterwards;
- each test runs inside a transaction that is rolled back.

**In CI**, PostgreSQL runs as a service container, and `AGENTBENCH_REQUIRE_POSTGRES=1`
makes the database tests fail rather than skip.

Lint and formatting checks (CI runs these too):

```bash
uv run ruff check .
uv run ruff format --check .
```

## Project layout

```
src/agentbench/
  domain/       models, enums, errors
  tools/        ToolRegistry and mock tools
  agents/       rule-based and scripted agents
  tracing/      TraceRecorder and clocks
  runner/       AgentRunner
  persistence/  SQLAlchemy tables, repository, unit of work
  service.py    run a task and save it
  cli.py        agentbench run / show
migrations/     Alembic migrations
tests/unit/         no database, no network
tests/integration/  PostgreSQL
docs/           architecture and design
```

## Limitations

- **No evaluation yet.** `COMPLETED` means the agent gave an answer, not that the answer
  is correct.
- **No real model.** The agents are deterministic test agents, so there is no token
  usage or cost yet.
- **Only single tasks.** Benchmarks, comparisons, the API, and the dashboard are planned.
- **Timeouts can't interrupt CPU-bound tool code** (see docs/runner.md §9).

## Roadmap

- [x] Project skeleton: packaging, Ruff, pytest, CI
- [x] Core data models: tasks, decisions, runs, trace events
- [x] Deterministic mock tools with argument validation
- [x] Agent runner with step limits, timeouts, and bounded retries
- [x] Trace recording
- [x] Persistence of runs and traces (SQLAlchemy + PostgreSQL + Alembic)
- [ ] Deterministic evaluator and scoring
- [ ] Benchmarks and comparisons
- [ ] HTTP API (FastAPI)
- [ ] Dashboard (Streamlit)
- [ ] Docker Compose for the full stack
- [ ] Real model adapter

## License

[MIT](LICENSE)
