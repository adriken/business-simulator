"""The simulation engine: turns a Business into thousands of possible futures.

Every quantity is a NumPy array of shape (n, T):
    n = number of scenarios (rows), T = number of periods (columns).

Each uncertain input is sampled once per scenario, not once per period. A
scenario is one possible version of the business, for example "a year where
average demand turned out low". Drawing fresh values every period would let
good and bad periods cancel out and make the business look much safer than
it is. Variation within a scenario comes from seasonality, which is fixed.

The calculation for each period:
    1. demand      = base demand x seasonality       (or active customers, if recurring)
    2. produced    = demand, scaled down if a shared capacity pool is full
    3. sold        = produced x (1 - loss)
    4. revenue     = sold x price
    5. costs       = fixed + per-unit (on produced or sold) + one-off
    6. profit      = revenue - costs;  cash = starting cash + running total of profit
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .distributions import FloatArray, Sampler, Spec
from .model import Business, Plan, Stream, check_plan


@dataclass(frozen=True)
class SimulationResult:
    """All scenario-by-period arrays from one run. Shapes: (n, T) unless noted."""

    business: str
    plan: str | None
    period: str
    currency: str
    revenue: FloatArray
    variable_costs: FloatArray
    fixed_costs: FloatArray
    one_off_costs: FloatArray
    starting_cash: FloatArray  # shape (n,)
    units_sold: dict[str, FloatArray]
    units_produced: dict[str, FloatArray]

    @property
    def n(self) -> int:
        return self.revenue.shape[0]

    @property
    def horizon(self) -> int:
        return self.revenue.shape[1]

    @property
    def profit(self) -> FloatArray:
        return self.revenue - self.variable_costs - self.fixed_costs - self.one_off_costs

    @property
    def total_profit(self) -> FloatArray:
        """Profit over the whole horizon, one number per scenario. Shape (n,)."""
        return self.profit.sum(axis=1)

    @property
    def cash(self) -> FloatArray:
        """Cash balance at the start (column 0) and after each period. Shape (n, T + 1)."""
        start = self.starting_cash[:, None]
        return np.hstack([start, start + np.cumsum(self.profit, axis=1)])


class _Inputs:
    """Draws each input from the Sampler, then applies the plan's changes to it."""

    def __init__(self, sampler: Sampler, plan: Plan | None) -> None:
        self.sampler = sampler
        self.changes = plan.changes if plan is not None else ()

    def get(self, path: str, spec: Spec, lo: float | None = None, hi: float | None = None) -> FloatArray:
        mine = [(i, c) for i, c in enumerate(self.changes) if c.path == path]

        # "set" replaces the spec. It keeps the same random stream (same key),
        # so a low draw in Plan A is still a low draw in Plan B.
        for _, c in mine:
            if c.op == "set":
                spec = c.value
        values = self.sampler.draw(path, spec)

        # "multiply" and "add" get their own random stream: how guests react to
        # a price rise is a separate uncertainty from the base demand.
        for i, c in mine:
            if c.op == "multiply":
                values = values * self.sampler.draw(f"{path}:multiply[{i}]", c.value)
            elif c.op == "add":
                values = values + self.sampler.draw(f"{path}:add[{i}]", c.value)

        if (lo is not None and values.min() < lo) or (hi is not None and values.max() > hi):
            raise ValueError(
                f"after applying the plan, '{path}' goes outside the allowed range "
                f"{lo if lo is not None else '-inf'} to {hi if hi is not None else 'inf'}"
            )
        return values


def simulate(
    business: Business,
    n: int = 5000,
    seed: int = 42,
    plan: Plan | None = None,
    sampler: Sampler | None = None,
) -> SimulationResult:
    """Run n scenarios of the business, optionally with a plan applied.

    Two runs with the same seed share their random draws input by input,
    which is what makes plan comparisons fair.
    """
    if plan is not None:
        check_plan(business, plan)
        business = business.with_plan_costs(plan)
    sampler = sampler if sampler is not None else Sampler(n, seed)
    inputs = _Inputs(sampler, plan)
    n, T = sampler.n, business.horizon

    # 1. Demand: what customers want before any limits.
    wanted = {s.name: _wanted_units(business, s, inputs, n, T) for s in business.streams}

    # 2. Capacity: streams sharing a pool get scaled down by the same fraction.
    produced = dict(wanted)
    for pool, cap_spec in business.capacity.items():
        users = [s for s in business.streams if s.capacity == pool]
        if users:
            produced.update(_ration(pool, cap_spec, users, wanted, inputs))

    revenue = np.zeros((n, T))
    variable = np.zeros((n, T))
    fixed = np.zeros((n, T))
    one_off = np.zeros((n, T))
    sold: dict[str, FloatArray] = {}

    # 3-4. Losses and revenue.
    for s in business.streams:
        loss = inputs.get(f"streams.{s.name}.loss", s.loss, lo=0, hi=1)
        sold[s.name] = produced[s.name] * (1 - loss[:, None])
        price = inputs.get(f"streams.{s.name}.price", s.price, lo=0)
        revenue += sold[s.name] * price[:, None]

    # 5. Costs. [:, None] turns a (n,) column of per-scenario values into (n, 1),
    # which NumPy broadcasts across all T periods.
    for c in business.costs.per_unit:
        units = produced[c.stream] if c.basis == "produced" else sold[c.stream]
        variable += units * inputs.get(f"costs.per_unit.{c.name}", c.amount)[:, None]
    for c in business.costs.fixed:
        fixed += inputs.get(f"costs.fixed.{c.name}", c.amount)[:, None]
    for c in business.costs.one_off:
        one_off[:, c.period - 1] += inputs.get(f"costs.one_off.{c.name}", c.amount)

    return SimulationResult(
        business=business.name,
        plan=plan.name if plan is not None else None,
        period=business.period,
        currency=business.currency,
        revenue=revenue,
        variable_costs=variable,
        fixed_costs=fixed,
        one_off_costs=one_off,
        starting_cash=inputs.get("starting_cash", business.starting_cash),
        units_sold=sold,
        units_produced=produced,
    )


def _wanted_units(business: Business, s: Stream, inputs: _Inputs, n: int, T: int) -> FloatArray:
    season = business.seasonality_for(s)  # shape (T,)
    demand = inputs.get(f"streams.{s.name}.demand", s.demand, lo=0)  # shape (n,)
    new = demand[:, None] * season[None, :]  # (n, 1) x (1, T) -> (n, T)
    if not s.recurring:
        return new

    # Recurring (subscriptions): customers carry over, minus those who cancel.
    #     active[t] = active[t-1] x (1 - churn) + new[t]
    # Each period depends on the one before, so this loops over periods. It is
    # still vectorised across all n scenarios at once.
    assert s.churn is not None
    churn = inputs.get(f"streams.{s.name}.churn", s.churn, lo=0, hi=1)
    active = inputs.get(f"streams.{s.name}.start_units", s.start_units, lo=0)
    out = np.empty((n, T))
    for t in range(T):
        active = active * (1 - churn) + new[:, t]
        out[:, t] = active
    return out


def _ration(
    pool: str,
    cap_spec: Spec,
    users: list[Stream],
    wanted: dict[str, FloatArray],
    inputs: _Inputs,
) -> dict[str, FloatArray]:
    """Scale down every stream in a full capacity pool by the same fraction.

    Example: a salon has 300 stylist-hours. Cuts want 200 hours and colours
    want 200. Total 400 > 300, so each gets 300 / 400 = 75% of its demand.
    """
    capacity = inputs.get(f"capacity.{pool}", cap_spec, lo=0)[:, None]  # (n, 1)
    usage = {
        s.name: wanted[s.name]
        * inputs.get(f"streams.{s.name}.capacity_per_unit", s.capacity_per_unit, lo=0)[:, None]
        for s in users
    }
    total = sum(usage.values())
    assert isinstance(total, np.ndarray)
    # fill = capacity / total, but never above 1, and 1 where nothing is wanted
    # (avoids dividing by zero).
    ratio = np.divide(capacity, total, out=np.ones_like(total), where=total > 0)
    fill = np.minimum(1.0, ratio)
    return {s.name: wanted[s.name] * fill for s in users}
