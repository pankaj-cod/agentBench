from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from agentbench.agents.base import Agent
from agentbench.domain.benchmark import Benchmark, BenchmarkRef
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.evaluation import Evaluation
from agentbench.domain.models import RunLimits
from agentbench.evaluation.engine import EvaluationEngine
from agentbench.evaluation.metrics import AgentConfigRef, BenchmarkSummary, summarize
from agentbench.persistence.uow import UnitOfWork
from agentbench.runner.runner import AgentRunner


class BenchmarkLoadError(AgentBenchError):
    pass


def load_benchmark(path: Path) -> Benchmark:
    try:
        return Benchmark.model_validate_json(path.read_text())
    except OSError as exc:
        raise BenchmarkLoadError(f"cannot read {path}: {exc}") from exc
    except ValidationError as exc:
        raise BenchmarkLoadError(f"invalid benchmark {path}:\n{exc}") from exc


@dataclass(frozen=True)
class AgentConfig:
    """One agent configuration to evaluate: how to build the agent, and its limits."""

    label: str
    make_agent: Callable[[], Agent]
    limits: RunLimits = field(default_factory=RunLimits)


class BenchmarkRunner:
    """Runs every case of a benchmark with one agent configuration, evaluates each run,
    and summarises the results.

    If a unit-of-work factory is given, each case's run, trace, and evaluation are saved
    in one transaction, so a crash loses at most the case in progress. Each case gets a
    fresh agent instance so no state leaks between cases.
    """

    def __init__(
        self,
        runner: AgentRunner,
        engine: EvaluationEngine,
        uow_factory: Callable[[], UnitOfWork] | None = None,
    ) -> None:
        self._runner = runner
        self._engine = engine
        self._uow_factory = uow_factory

    async def run(self, benchmark: Benchmark, config: AgentConfig) -> BenchmarkSummary:
        evaluations: list[Evaluation] = []
        agent_ref: AgentConfigRef | None = None
        for case in benchmark.cases:
            agent = config.make_agent()
            agent_ref = agent_ref or AgentConfigRef(
                label=config.label,
                agent_name=agent.name,
                agent_version=agent.version,
                limits=config.limits,
            )
            result = await self._runner.run(case.to_task(), agent, config.limits)
            evaluation = self._engine.evaluate(case, result)
            if self._uow_factory is not None:
                with self._uow_factory() as uow:
                    uow.agent_runs.add(result)
                    uow.evaluations.save(evaluation)
                    uow.commit()
            evaluations.append(evaluation)
        assert agent_ref is not None  # a Benchmark always has at least one case
        return summarize(BenchmarkRef.of(benchmark), agent_ref, evaluations)
