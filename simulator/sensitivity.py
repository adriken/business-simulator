"""Tornado analysis: which inputs move profit the most?

Method (one-at-a-time sensitivity):
    1. Set every input to its likely value and compute profit: the base case.
    2. For each uncertain input, move just that one to its low, then to its
       high, with everything else left at likely. Record both profits.
    3. Sort by swing = |profit at high - profit at low|. The biggest swing
       goes at the top, which gives the chart its tornado shape.

This is deterministic: no random scenarios, just 1 + 2k runs of the engine
with n = 1, for k uncertain inputs. Its limit is that it ignores
interactions: demand only matters above a certain level if capacity is
already full, and one-at-a-time swings can't show that.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .distributions import Sampler, high, is_uncertain, likely, low
from .engine import simulate
from .model import Business, Plan


@dataclass(frozen=True)
class Tornado:
    base_profit: float
    table: pd.DataFrame  # one row per uncertain input, biggest swing first


def _point_profit(business: Business, plan: Plan | None, overrides: dict[str, float]) -> tuple[float, Sampler]:
    sampler = Sampler(n=1, point=True, overrides=overrides)
    result = simulate(business, plan=plan, sampler=sampler)
    return float(result.total_profit[0]), sampler


def tornado(business: Business, plan: Plan | None = None) -> Tornado:
    base, sampler = _point_profit(business, plan, {})
    rows = []
    # The base run recorded every input it drew, so the list of inputs comes
    # from the engine itself and can't drift out of sync with it.
    for key, spec in sampler.specs.items():
        if not is_uncertain(spec):
            continue
        at_low, _ = _point_profit(business, plan, {key: low(spec)})
        at_high, _ = _point_profit(business, plan, {key: high(spec)})
        rows.append(
            {
                "input": key,
                "low": low(spec),
                "likely": likely(spec),
                "high": high(spec),
                "profit_at_low": at_low,
                "profit_at_high": at_high,
                "swing": abs(at_high - at_low),
            }
        )
    columns = ["input", "low", "likely", "high", "profit_at_low", "profit_at_high", "swing"]
    table = pd.DataFrame(rows, columns=columns)
    table = table.sort_values("swing", ascending=False, ignore_index=True)
    return Tornado(base_profit=base, table=table)
