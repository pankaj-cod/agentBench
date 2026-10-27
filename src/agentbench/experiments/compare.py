from collections.abc import Callable

from agentbench.agents.model_agent import ModelAgent
from agentbench.benchmarks.service import AgentConfig, BenchmarkRunner
from agentbench.domain.agent_config import ModelAgentConfig
from agentbench.domain.benchmark import Benchmark
from agentbench.domain.models import DomainModel, RunLimits
from agentbench.evaluation.comparison import BenchmarkComparison, compare
from agentbench.evaluation.metrics import BenchmarkSummary
from agentbench.llm.base import ModelAdapter
from agentbench.tracing.clock import Clock

LIVE_MODEL_NOTE = (
    "Each case was run once per configuration. Live model output is not deterministic, "
    "so a single run per case cannot show whether a difference would repeat."
)


class ConfiguredAgent(DomainModel):
    config: ModelAgentConfig
    limits: RunLimits


class ExperimentReport(DomainModel):
    baseline: ConfiguredAgent
    candidate: ConfiguredAgent
    changed_fields: tuple[str, ...]  # top-level config fields that differ
    baseline_summary: BenchmarkSummary
    candidate_summary: BenchmarkSummary
    comparison: BenchmarkComparison
    notes: tuple[str, ...]


def changed_fields(a: ModelAgentConfig, b: ModelAgentConfig) -> tuple[str, ...]:
    da, db = a.model_dump(exclude={"label"}), b.model_dump(exclude={"label"})
    return tuple(k for k in da if da[k] != db[k])


class ExperimentService:
    """Evaluates the same benchmark against two model-agent configurations and compares
    them. Both runs use the same benchmark content, tools, limits, and evaluator."""

    def __init__(self, runner: BenchmarkRunner, clock: Clock | None = None) -> None:
        self._runner = runner
        self._clock = clock

    async def compare(
        self,
        benchmark: Benchmark,
        baseline: ConfiguredAgent,
        candidate: ConfiguredAgent,
        adapter_for: Callable[[ModelAgentConfig], ModelAdapter],
    ) -> ExperimentReport:
        summaries = []
        for configured in (baseline, candidate):
            config = configured.config

            def make_agent(config: ModelAgentConfig = config) -> ModelAgent:
                return ModelAgent(config, adapter_for(config), clock=self._clock)

            agent_config = AgentConfig(config.label, make_agent, configured.limits)
            summaries.append(await self._runner.run(benchmark, agent_config))
        changed = changed_fields(baseline.config, candidate.config)
        providers = {baseline.config.provider, candidate.config.provider}
        notes = [LIVE_MODEL_NOTE] if providers - {"scripted"} else []
        if baseline.limits != candidate.limits:
            notes.append("limits differ between configurations")
        if not changed and baseline.limits == candidate.limits:
            notes.append("configurations are identical: any difference is run-to-run variation")
        return ExperimentReport(
            baseline=baseline,
            candidate=candidate,
            changed_fields=changed,
            baseline_summary=summaries[0],
            candidate_summary=summaries[1],
            comparison=compare(summaries[0], summaries[1]),
            notes=tuple(notes),
        )
