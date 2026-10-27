"""A deterministic agent that understands a small, fixed set of prompt patterns.

Policy (patterns are case-insensitive; a trailing "?" is optional):

1. "What is the <field> of record <ID>"                field: quantity | price | name | category
   -> lookup_record(record_id=ID) -> answer the field.
2. "What is <P>% of the <field> of record <ID>"        field: quantity | price
   -> lookup_record(ID) -> calculator("<value> * <P> / 100") -> answer the result.
3. "Find records matching '<text>'"
   -> search_records(query=text, limit=20) -> answer the ids, comma-separated, or "none".
4. "What is the total <field> of records matching '<text>'"   field: quantity | value
   -> search_records(text, limit=20) -> calculator(sum of quantity, or of quantity * price)
   -> answer the result ("0" with no matches, without calling the calculator).
5. "Compute <expression>" / "Calculate <expression>"
   -> calculator(expression) -> answer the result.

If any tool call fails, the agent stops and answers
"unable to answer: <error type>: <message>". It never invents a value.
A prompt that matches no pattern raises UnsupportedTaskError.
"""

import re
from collections.abc import Callable
from typing import Any

from agentbench.agents.base import AgentContext
from agentbench.domain.errors import AgentBenchError
from agentbench.domain.models import FinalAnswerDecision, HistoryStep, ToolCallDecision

type AgentDecision = ToolCallDecision | FinalAnswerDecision
type Plan = Callable[[re.Match[str], tuple[HistoryStep, ...]], AgentDecision]

RECORD_ID = r"(?P<id>[A-Za-z]{1,3}-\d{1,4})"
QUOTED = r"""['"](?P<text>[^'"]+)['"]"""
FIELD_KEYS = {"quantity": "quantity", "price": "unit_price", "name": "name", "category": "category"}


class UnsupportedTaskError(AgentBenchError):
    pass


def _call(tool: str, **arguments: Any) -> ToolCallDecision:
    return ToolCallDecision(tool=tool, arguments=arguments)


def _answer(value: Any) -> FinalAnswerDecision:
    # 32.0 is answered as "32"; other values use their plain string form.
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return FinalAnswerDecision(answer=str(value))


def _last_output(history: tuple[HistoryStep, ...]) -> Any:
    observation = history[-1].observation
    assert observation is not None and observation.ok, "failed calls are handled in decide()"
    return observation.output


def _lookup_field(m: re.Match[str], history: tuple[HistoryStep, ...]) -> AgentDecision:
    if not history:
        return _call("lookup_record", record_id=m["id"].upper())
    return _answer(_last_output(history)["record"][FIELD_KEYS[m["field"].lower()]])


def _percent_of_field(m: re.Match[str], history: tuple[HistoryStep, ...]) -> AgentDecision:
    if not history:
        return _call("lookup_record", record_id=m["id"].upper())
    if len(history) == 1:
        value = _last_output(history)["record"][FIELD_KEYS[m["field"].lower()]]
        return _call("calculator", expression=f"{value} * {m['pct']} / 100")
    return _answer(_last_output(history)["result"])


def _find(m: re.Match[str], history: tuple[HistoryStep, ...]) -> AgentDecision:
    if not history:
        return _call("search_records", query=m["text"], limit=20)
    ids = [r["id"] for r in _last_output(history)["records"]]
    return _answer(", ".join(ids) if ids else "none")


def _total(m: re.Match[str], history: tuple[HistoryStep, ...]) -> AgentDecision:
    if not history:
        return _call("search_records", query=m["text"], limit=20)
    if len(history) == 1:
        records = _last_output(history)["records"]
        if not records:
            return _answer(0)
        if m["field"].lower() == "quantity":
            terms = [str(r["quantity"]) for r in records]
        else:
            terms = [f"{r['quantity']} * {r['unit_price']}" for r in records]
        return _call("calculator", expression=" + ".join(terms))
    return _answer(_last_output(history)["result"])


def _compute(m: re.Match[str], history: tuple[HistoryStep, ...]) -> AgentDecision:
    if not history:
        return _call("calculator", expression=m["expr"].strip())
    return _answer(_last_output(history)["result"])


RULES: tuple[tuple[re.Pattern[str], Plan], ...] = tuple(
    (re.compile(pattern, re.IGNORECASE), plan)
    for pattern, plan in [
        (
            rf"what is the (?P<field>quantity|price|name|category) of record {RECORD_ID}\??",
            _lookup_field,
        ),
        (
            rf"what is (?P<pct>\d+(?:\.\d+)?)% of the (?P<field>quantity|price)"
            rf" of record {RECORD_ID}\??",
            _percent_of_field,
        ),
        (rf"find records matching {QUOTED}", _find),
        (rf"what is the total (?P<field>quantity|value) of records matching {QUOTED}\??", _total),
        (r"(?:compute|calculate) (?P<expr>.+)", _compute),
    ]
)


class RuleBasedAgent:
    name = "rule_based"
    version = "1"

    async def decide(self, context: AgentContext) -> AgentDecision:
        prompt = context.task.prompt.strip()
        for pattern, plan in RULES:
            match = pattern.fullmatch(prompt)
            if match is not None:
                return self._follow(plan, match, context.history)
        raise UnsupportedTaskError(f"no rule matches the prompt: {prompt!r}")

    @staticmethod
    def _follow(
        plan: Plan, match: re.Match[str], history: tuple[HistoryStep, ...]
    ) -> AgentDecision:
        if history:
            observation = history[-1].observation
            if observation is not None and observation.error is not None:
                error = observation.error
                return FinalAnswerDecision(
                    answer=f"unable to answer: {error.type}: {error.message}"
                )
        return plan(match, history)
