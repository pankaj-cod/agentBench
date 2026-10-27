# Evaluation

How AgentBench scores runs, aggregates metrics, and compares agent configurations.

Everything here is **deterministic**: scores come from explicit, per-case criteria
checked against the recorded trace. No LLM judge is used. Code:
- `src/agentbench/evaluation/engine.py` — scoring;
- `src/agentbench/evaluation/metrics.py` — aggregate metrics;
- `src/agentbench/evaluation/comparison.py` — comparisons.

## 1. Test cases and criteria

A benchmark is a JSON file containing named, versioned test cases. See
`examples/benchmarks/inventory-v1.json`. Each case has a prompt, the tools it allows, and
`expectations`:

| Criterion | Meaning | Failure category |
|-----------|---------|------------------|
| `final_answer: exact` | After trimming whitespace, the answer equals `value`; case-insensitive unless `case_sensitive` is set | `WRONG_ANSWER` |
| `final_answer: numeric` | The **whole** answer is a finite number within `tolerance` of `value`. `"about 20.4"` fails. | `WRONG_ANSWER` |
| `final_answer: regex` | The whole answer matches `pattern` (`re.fullmatch`) | `WRONG_ANSWER` |
| `final_answer: one_of` | The answer equals one of `values` (case-insensitive) | `WRONG_ANSWER` |
| `required_tools` | Each listed tool was called **successfully** at least once. With `args_subset`, at least one successful call had those argument values. | `MISSING_REQUIRED_TOOL`, `WRONG_TOOL_ARGUMENTS` |
| `forbidden_tools` | None of these tools was requested | `FORBIDDEN_TOOL_USED` |
| `max_tool_calls` | No more tool-call decisions than this | `TOOL_CALL_LIMIT_EXCEEDED` |
| `expected_records` | These record ids appear in the outputs of successful `search_records` / `lookup_record` calls | `MISSING_RETRIEVED_RECORDS` |
| *(always, unless `allow_invalid_tool_calls`)* | Every tool call passed validation: a known tool, allowed for the case, with valid arguments | `INVALID_TOOL_CALL` |
| *(always)* | Execution: the run ended `COMPLETED` | see §2 |

A benchmark is identified by its name, its version, and a SHA-256 **content hash**, so
results are compared only between identical case sets.

## 2. Outcomes, failure categories, and sources

Each run gets one outcome:

| Outcome | Meaning |
|---------|---------|
| `PASSED` | Every check passed. |
| `FAILED` | The **agent** behaved incorrectly. |
| `INVALID` | **Infrastructure** failed, so the run says nothing about the agent. Excluded from agent-quality metrics. |

**When several checks fail, one category is primary**: the first failing category in
this precedence order. Execution problems come first, because they make the later checks
meaningless. The other failing checks are listed in the explanation.

| Precedence | Category | Source | Typical cause |
|------------|----------|--------|---------------|
| 1 | `HARNESS_ERROR` | infrastructure | A bug in AgentBench (termination `HARNESS_ERROR`) |
| 2 | `TOOL_FAILURE` | infrastructure | Retryable tool errors (timeouts, transient faults) on every attempt (`TOOL_FAILURE_EXHAUSTED`) |
| 3 | `AGENT_ERROR` | agent | The agent raised an exception or returned an invalid decision |
| 4 | `TIMEOUT` | agent | The run deadline passed |
| 5 | `STEP_LIMIT` | agent | `max_steps` reached without an answer |
| 6 | `FORBIDDEN_TOOL_USED` | agent | |
| 7 | `INVALID_TOOL_CALL` | agent | Unknown tool, not-allowed tool, or invalid arguments |
| 8 | `TOOL_CALL_LIMIT_EXCEEDED` | agent | |
| 9 | `MISSING_REQUIRED_TOOL` | agent | |
| 10 | `WRONG_TOOL_ARGUMENTS` | agent | A valid call, but not with the expected values |
| 11 | `MISSING_RETRIEVED_RECORDS` | agent | |
| 12 | `WRONG_ANSWER` | agent | |

**Judgement calls**, stated so they can be challenged:
- **An exhausted retryable tool error is infrastructure.** Only timeouts and transient
  failures are retried, and those come from the environment.
- **A permanent tool error is the agent's.** For example, looking up a record id that
  doesn't exist.
- **A run timeout is the agent's.** The agent spent the time.

**Each evaluation stores:**
- the outcome, the primary category and its source;
- a human-readable explanation;
- every check result;
- per-run metric values;
- the exact `expectations` that were applied;
- the `evaluator_version` and `pricing_version`.

## 3. Aggregate metrics

**Valid runs** are runs whose outcome is `PASSED` or `FAILED`.

**A metric that can't be measured has `value: null` and a reason.** It is never 0. For
example, "no valid runs", or "reported by 3 of 5 runs".

| Metric | Numerator | Denominator | Exclusions |
|--------|-----------|-------------|------------|
| Task success rate | `PASSED` runs | Valid runs | `INVALID` runs |
| Tool selection correctness | Valid runs where every required tool was requested, no forbidden tool was requested, and no unknown or not-allowed tool was requested | Valid runs whose case has `required_tools` or `forbidden_tools` | `INVALID` runs; cases with no tool expectations |
| Tool argument validity | Tool calls whose arguments passed validation | Tool calls naming a registered, allowed tool (counted per call) | `INVALID` runs; calls to unknown or not-allowed tools (selection errors, not argument errors) |
| Tool execution error rate | Failed tool executions | Tool executions started, per attempt (retries included) | Calls rejected by validation (they never execute). **Includes** `INVALID` runs, because infrastructure failures are what this rate shows. |
| Timeout rate | Runs ending `RUN_TIMEOUT` | All runs | none |
| Step-limit rate | Runs ending `MAX_STEPS_EXCEEDED` | All runs | none |
| Latency p50 / p95 | Nearest-rank percentile of end-to-end run duration (ms) | — | `INVALID` runs. p50 needs at least 1 sample; **p95 needs at least 20** (with fewer, the nearest-rank p95 is just the maximum). |
| Latency mean | Mean run duration (ms) | Valid runs | `INVALID` runs |
| Steps per run | Mean agent decisions per run | Valid runs | `INVALID` runs |
| Input / output tokens | Sum of reported tokens | All runs | **Unavailable unless every run reported usage.** A partial sum would understate usage. |
| Estimated cost (USD) | Sum of `tokens × price` from a versioned price table | All runs | Unavailable unless every decision of every run reported usage for a model that is in the table |

**Notes on usage and cost:**
- **Tokens and cost cover all runs**, including invalid ones, because those still
  consumed usage.
- **No real provider prices are shipped.** They change often, and a stale table would
  produce confidently wrong numbers. Cost appears only when a caller supplies a
  `PriceTable`.
- **The current deterministic agents report no usage**, so token and cost metrics are
  unavailable for them, by design.

## 4. Comparing configurations

`compare(baseline, candidate)` refuses to compare unless both summaries have:
- the same benchmark content hash;
- the same evaluator version;
- the same set of cases.

| Metric kind | Reported changes |
|-------------|------------------|
| Rates | Absolute change (candidate − baseline, as a fraction) and **percentage points** (pp). No relative change: going from 50% to 75% is +25 pp, and calling it "+50%" is easy to misread. |
| Values (latency, steps, tokens, cost) | Absolute change in the metric's unit, and relative change in % (undefined when the baseline is 0) |
| Unavailable on either side | No change is reported; a note says why |

**Per-case changes:**
- `FIXED` — FAILED → PASSED;
- `BROKEN` — PASSED → FAILED;
- `STILL_PASSING`;
- `STILL_FAILING`;
- `NOT_COMPARABLE` — INVALID on either side.

The failure distribution is reported for both configurations, with the change in each
category.

```bash
uv run agentbench bench examples/benchmarks/inventory-v1.json --label default --output default.json
uv run agentbench bench examples/benchmarks/inventory-v1.json --label two-steps --max-steps 2 --output two-steps.json
uv run agentbench compare default.json two-steps.json
```

## 5. Measurement validity, agent accuracy, and evaluator quality

These are three different questions, and a benchmark result depends on all three.

**Agent accuracy** — did *this agent* do the task correctly? This is what the task
success rate tries to estimate. It is a property of the agent.

**Evaluator quality** — does the scoring rule correctly decide whether a given run was
right? Ways it can go wrong:
- **Too lenient.** A loose regex accepts a wrong answer: a false pass.
- **Too strict.** `numeric` rejects `"20.4 units"` although it is correct: a false fail.

It is checked by tests that feed known-good and known-bad traces to the evaluator
(`tests/unit/test_evaluation_engine.py`), and by keeping the evaluator deterministic and
versioned, so the same trace always gets the same verdict.

**Measurement validity** — does the number measure what it claims to? Even with a
perfect evaluator, the result can be invalid:
- **The infrastructure failed.** A timed-out tool is not the agent's fault. AgentBench
  marks such runs `INVALID` and excludes them, rather than counting them as agent
  failures.
- **The cases are wrong or unrepresentative.** A wrong expected answer, or a benchmark of
  only easy prompts, makes 100% meaningless. The example benchmark deliberately includes
  two phrasings the rule-based agent can't handle.
- **The sample is too small.** 14 runs can't support a p95, or a claim that a 1-point
  difference is real. AgentBench reports p95 as unavailable below 20 runs, and reports
  absolute counts next to every rate.
- **Non-determinism.** With deterministic agents, one run per case is enough. Real models
  will need repeated runs before differences can be trusted.

In short: **a high score from an unvalidated evaluator on an unrepresentative benchmark
says nothing about agent accuracy.** Each layer has to be checked on its own.

## 6. Limitations

- **Answer checks are syntactic.** "120 units" or "one hundred twenty" fail a numeric
  check. That is deliberately strict, but it means a correct answer can be marked wrong.
- **`expected_records` relies on the mock tools' output format** (`record` / `records`
  with an `id`).
- **No repeated runs or confidence intervals yet.**
- **Benchmarks are files, not database tables.** Each evaluation stores a snapshot of
  the criteria it applied, but benchmark runs are not stored as a group yet; summaries
  are JSON files.
