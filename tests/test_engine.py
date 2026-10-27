"""Engine tests use fixed numbers so every expected value can be checked by hand.

Base test business (see conftest.py), per month:
    sold      = 100 x (1 - 0.1)                = 90
    revenue   = 90 x 1000                      = 90,000
    materials = 90 x 100                       = 9,000
    rent                                       = 50,000
    profit    = 31,000 (month 2: minus 10,000 repair = 21,000)
"""

import numpy as np
import pytest

from simulator.engine import simulate
from simulator.model import Business, Plan


def test_hand_calculated_profit(business):
    r = simulate(business, n=4)
    assert r.profit.shape == (4, 3)
    assert np.allclose(r.profit, [31_000, 21_000, 31_000])
    assert np.allclose(r.total_profit, 83_000)
    assert np.allclose(r.cash[0], [0, 31_000, 52_000, 83_000])


def test_capacity_caps_production(data):
    data["streams"][0]["demand"] = 400  # more than the 300 slots
    r = simulate(Business.model_validate(data), n=2)
    assert np.allclose(r.units_produced["main"], 300)
    assert np.allclose(r.units_sold["main"], 270)


def test_shared_capacity_is_rationed_proportionally(data):
    data["streams"].append({"name": "extra", "demand": 200, "price": 10, "capacity": "slots"})
    data["streams"][0]["demand"] = 200
    r = simulate(Business.model_validate(data), n=2)
    # 400 wanted, 300 available: each stream gets 75%
    assert np.allclose(r.units_produced["main"], 150)
    assert np.allclose(r.units_produced["extra"], 150)


def test_capacity_per_unit(data):
    data["streams"][0]["capacity_per_unit"] = 2  # each unit uses 2 slots
    data["streams"][0]["demand"] = 200  # 400 slots wanted, 300 available
    r = simulate(Business.model_validate(data), n=1)
    assert np.allclose(r.units_produced["main"], 150)


def test_cost_basis_produced_vs_sold(data):
    data["costs"]["per_unit"][0]["basis"] = "produced"
    r = simulate(Business.model_validate(data), n=1)
    assert np.allclose(r.variable_costs, 100 * 100)  # charged on all 100, not 90


def test_seasonality_scales_demand(data):
    data["streams"][0]["seasonality"] = [0.5, 1.0, 2.0]
    r = simulate(Business.model_validate(data), n=1)
    assert np.allclose(r.units_produced["main"], [[50, 100, 200]])


def test_recurring_stream_accumulates_customers(data):
    data["capacity"] = {}
    data["streams"] = [{"name": "subs", "demand": 10, "price": 1, "churn": 0.5}]
    data["costs"] = {}
    r = simulate(Business.model_validate(data), n=1)
    # active[t] = active[t-1] x 0.5 + 10  ->  10, 15, 17.5
    assert np.allclose(r.units_sold["subs"], [[10, 15, 17.5]])


def test_uncertain_inputs_vary_between_scenarios(data):
    data["streams"][0]["price"] = [800, 1000, 1200]
    r = simulate(Business.model_validate(data), n=1000, seed=1)
    assert r.total_profit.std() > 0
    assert r.revenue[:, 0].min() >= 90 * 800
    assert r.revenue[:, 0].max() <= 90 * 1200


def test_one_draw_per_scenario_not_per_period(data):
    # Price is random, but within a scenario it must be the same every month.
    data["streams"][0]["price"] = [800, 1200]
    r = simulate(Business.model_validate(data), n=50, seed=3)
    assert np.allclose(r.revenue, r.revenue[:, [0]])


def test_same_seed_is_reproducible(data):
    data["streams"][0]["demand"] = [50, 100, 150]
    b = Business.model_validate(data)
    assert np.array_equal(simulate(b, n=100, seed=9).profit, simulate(b, n=100, seed=9).profit)
    assert not np.array_equal(simulate(b, n=100, seed=9).profit, simulate(b, n=100, seed=10).profit)


def test_plan_multiply_add_and_extra_costs(business):
    plan = Plan(
        name="B",
        changes=[
            {"path": "streams.main.price", "op": "multiply", "value": 1.1},
            {"path": "costs.fixed.rent", "op": "add", "value": 5000},
        ],
        extra_costs={"one_off": [{"name": "setup", "period": 1, "amount": 20_000}]},
    )
    r = simulate(business, n=1, plan=plan)
    # revenue 90 x 1100 = 99,000; costs 9,000 + 55,000 = 64,000 -> 35,000 a month
    assert np.allclose(r.profit, [[35_000 - 20_000, 35_000 - 10_000, 35_000]])
    assert r.plan == "B"


def test_plan_that_breaks_a_limit_is_rejected(business):
    plan = Plan(name="bad", changes=[{"path": "streams.main.loss", "op": "add", "value": 2}])
    with pytest.raises(ValueError, match="outside the allowed range"):
        simulate(business, n=1, plan=plan)
