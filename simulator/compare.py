"""Compare two plans fairly, using common random numbers.

Both plans are simulated with the same seed. Because the Sampler gives each
named input its own random stream, scenario 17 has the same demand, price and
costs draws under both plans. So the per-scenario difference
    profit_B[i] - profit_A[i]
measures the effect of the plan alone, not luck. That makes the key question
answerable directly: in what share of possible futures does B beat A?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .engine import SimulationResult, simulate
from .metrics import Percentiles, Summary, summarize
from .model import Business, Plan


@dataclass(frozen=True)
class Comparison:
    a: Summary
    b: Summary
    prob_b_better: float  # share of scenarios where B's total profit beats A's
    profit_difference: Percentiles  # B minus A, per scenario
    result_a: SimulationResult
    result_b: SimulationResult

    def table(self) -> pd.DataFrame:
        rows = [
            ("Median profit", self.a.total_profit.p50, self.b.total_profit.p50),
            ("P10 profit (bad case)", self.a.total_profit.p10, self.b.total_profit.p10),
            ("P90 profit (good case)", self.a.total_profit.p90, self.b.total_profit.p90),
            ("Chance of a loss", self.a.prob_loss, self.b.prob_loss),
            ("Median ending cash", self.a.ending_cash.p50, self.b.ending_cash.p50),
            ("Chance cash runs out", self.a.prob_cash_runs_out, self.b.prob_cash_runs_out),
        ]
        df = pd.DataFrame(rows, columns=["measure", "plan_a", "plan_b"])
        df["change"] = df["plan_b"] - df["plan_a"]
        return df


def compare(
    business: Business,
    plan_b: Plan,
    plan_a: Plan | None = None,
    n: int = 5000,
    seed: int = 42,
) -> Comparison:
    """Plan A defaults to the business exactly as written."""
    result_a = simulate(business, n=n, seed=seed, plan=plan_a)
    result_b = simulate(business, n=n, seed=seed, plan=plan_b)
    diff = result_b.total_profit - result_a.total_profit
    return Comparison(
        a=summarize(result_a),
        b=summarize(result_b),
        prob_b_better=float(np.mean(diff > 0)),
        profit_difference=Percentiles.of(diff),
        result_a=result_a,
        result_b=result_b,
    )
