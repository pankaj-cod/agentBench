from collections.abc import Iterable, Mapping
from decimal import Decimal

from agentbench.domain.models import TokenUsage

MILLION = Decimal(1_000_000)
MICRO_USD = Decimal("0.000001")


class PriceTable:
    """USD prices per million tokens, by model name, under an explicit version.

    AgentBench ships no real provider prices: they change often and a stale table
    would produce confidently wrong costs. Callers supply a table; a model that is not
    in it makes the cost unavailable rather than zero.
    """

    def __init__(self, version: str, prices: Mapping[str, tuple[Decimal, Decimal]]) -> None:
        self.version = version
        self._prices = dict(prices)  # model -> (input per 1M, output per 1M)

    def cost(self, usages: Iterable[TokenUsage | None]) -> Decimal | None:
        """Total cost, or None if any decision lacks usage or uses an unpriced model."""
        total = Decimal(0)
        seen = False
        for usage in usages:
            if usage is None or usage.model not in self._prices:
                return None
            input_price, output_price = self._prices[usage.model]
            total += (
                usage.input_tokens * input_price + usage.output_tokens * output_price
            ) / MILLION
            seen = True
        # Rounded to the stored precision (NUMERIC(12, 6)), so a re-computed cost always
        # equals the stored one.
        return total.quantize(MICRO_USD) if seen else None


EMPTY_PRICE_TABLE = PriceTable("none", {})
