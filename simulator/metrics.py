"""Turn the scenario arrays into the numbers a business owner reads.

Every metric is computed per scenario first, then summarised across
scenarios. That order matters: "the median of the profit" is not the same
as "the profit in a median-looking year".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .distributions import FloatArray
from .engine import SimulationResult


@dataclass(frozen=True)
class Percentiles:
    """P10: only 1 scenario in 10 is worse. P90: only 1 in 10 is better."""

    p10: float
    p50: float
    p90: float

    @classmethod
    def of(cls, values: FloatArray) -> Percentiles:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return cls(np.nan, np.nan, np.nan)
        p10, p50, p90 = np.percentile(finite, [10, 50, 90])
        return cls(float(p10), float(p50), float(p90))


@dataclass(frozen=True)
class Summary:
    business: str
    plan: str | None
    period: str
    currency: str
    n: int
    horizon: int
    total_profit: Percentiles
    ending_cash: Percentiles
    prob_loss: float
    prob_cash_runs_out: float
    cash_runs_out_period: float | None  # median period, among scenarios where it happens
    breakeven_ratio: Percentiles
    breakeven_units: dict[str, float]  # median units per period needed to break even
    prob_payback: float
    payback_period: float | None  # median period, among scenarios that pay back


def prob_loss(result: SimulationResult) -> float:
    """Share of scenarios that lose money over the whole horizon."""
    return float(np.mean(result.total_profit < 0))


def cash_runout_period(result: SimulationResult) -> FloatArray:
    """First period (1-based) where cash drops below zero; NaN if it never does."""
    below = result.cash[:, 1:] < 0
    # argmax on booleans returns the index of the first True. It returns 0
    # when there is no True at all, so rows that never go negative are masked.
    return np.where(below.any(axis=1), below.argmax(axis=1) + 1.0, np.nan)


def payback_period(result: SimulationResult) -> FloatArray:
    """First period where cumulative profit reaches zero or more; NaN if never."""
    reached = np.cumsum(result.profit, axis=1) >= 0
    return np.where(reached.any(axis=1), reached.argmax(axis=1) + 1.0, np.nan)


def breakeven_ratio(result: SimulationResult) -> FloatArray:
    """Share of each scenario's sales volume needed to cover its overheads.

    contribution = revenue - variable costs  (what sales add after their own costs)
    overheads    = fixed + one-off costs     (what must be paid regardless)
    ratio        = overheads / contribution

    A ratio of 0.6 means the business breaks even at 60% of the volume it
    actually sold in that scenario (a 40% margin of safety). Above 1 means a
    loss. Infinite if every sale loses money. This assumes costs scale in a
    straight line with volume and ignores capacity limits.
    """
    contribution = (result.revenue - result.variable_costs).sum(axis=1)
    overheads = (result.fixed_costs + result.one_off_costs).sum(axis=1)
    return np.divide(
        overheads, contribution, out=np.full_like(overheads, np.inf), where=contribution > 0
    )


def _median_or_none(values: FloatArray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else None


def summarize(result: SimulationResult) -> Summary:
    runout = cash_runout_period(result)
    payback = payback_period(result)
    ratio = breakeven_ratio(result)
    units = {
        name: float(np.nanmedian(np.where(np.isfinite(ratio), ratio, np.nan) * sold.mean(axis=1)))
        for name, sold in result.units_sold.items()
    }
    return Summary(
        business=result.business,
        plan=result.plan,
        period=result.period,
        currency=result.currency,
        n=result.n,
        horizon=result.horizon,
        total_profit=Percentiles.of(result.total_profit),
        ending_cash=Percentiles.of(result.cash[:, -1]),
        prob_loss=prob_loss(result),
        prob_cash_runs_out=float(np.mean(np.isfinite(runout))),
        cash_runs_out_period=_median_or_none(runout),
        breakeven_ratio=Percentiles.of(ratio),
        breakeven_units=units,
        prob_payback=float(np.mean(np.isfinite(payback))),
        payback_period=_median_or_none(payback),
    )


def period_table(result: SimulationResult) -> pd.DataFrame:
    """P10 / median / P90 of profit and closing cash for each period (for charts)."""
    pct = [10, 50, 90]
    profit = np.percentile(result.profit, pct, axis=0)  # shape (3, T)
    cash = np.percentile(result.cash[:, 1:], pct, axis=0)
    return pd.DataFrame(
        {
            "period": np.arange(1, result.horizon + 1),
            "profit_p10": profit[0],
            "profit_p50": profit[1],
            "profit_p90": profit[2],
            "cash_p10": cash[0],
            "cash_p50": cash[1],
            "cash_p90": cash[2],
        }
    )
