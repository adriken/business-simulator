import numpy as np
import pytest

from simulator.compare import compare
from simulator.engine import simulate
from simulator.metrics import (
    breakeven_ratio,
    cash_runout_period,
    payback_period,
    period_table,
    summarize,
)
from simulator.model import Business, Plan, get_plan, load_business
from simulator.sensitivity import tornado

from .conftest import TEMPLATES


# ---- metrics -------------------------------------------------------------

def test_breakeven_ratio_by_hand(business):
    # contribution = 3 x (90,000 - 9,000) = 243,000; overheads = 150,000 + 10,000
    r = simulate(business, n=1)
    assert breakeven_ratio(r)[0] == pytest.approx(160_000 / 243_000)
    s = summarize(r)
    assert s.breakeven_units["main"] == pytest.approx(90 * 160_000 / 243_000)


def test_cash_runout_and_payback(data):
    data["starting_cash"] = 40_000
    data["costs"]["fixed"][0]["amount"] = 90_000  # profit -9,000 a month, -19,000 in month 2
    r = simulate(Business.model_validate(data), n=1)
    # cash: 40,000 -> 31,000 -> 12,000 -> 3,000: never below zero
    assert np.isnan(cash_runout_period(r)[0])
    assert np.isnan(payback_period(r)[0])  # cumulative profit never reaches zero

    data["starting_cash"] = 20_000  # 20,000 -> 11,000 -> -8,000
    r = simulate(Business.model_validate(data), n=1)
    assert cash_runout_period(r)[0] == 2


def test_summary_probabilities(data):
    data["streams"][0]["demand"] = [0, 100]  # some scenarios lose money
    s = summarize(simulate(Business.model_validate(data), n=5000, seed=1))
    assert 0 < s.prob_loss < 1
    assert s.total_profit.p10 < s.total_profit.p50 < s.total_profit.p90


def test_period_table_shape(business):
    t = period_table(simulate(business, n=10))
    assert len(t) == 3
    assert list(t["cash_p50"]) == pytest.approx([31_000, 52_000, 83_000])


# ---- tornado -------------------------------------------------------------

def test_tornado_ranks_the_biggest_swing_first(data):
    data["streams"][0]["price"] = [900, 1000, 1100]  # swing 90 x 200 x 3 = 54,000
    data["costs"]["fixed"][0]["amount"] = [49_000, 50_000, 51_000]  # swing 6,000
    t = tornado(Business.model_validate(data))
    assert list(t.table["input"]) == ["streams.main.price", "costs.fixed.rent"]
    assert t.table["swing"].iloc[0] == pytest.approx(54_000)
    assert t.base_profit == pytest.approx(83_000)


def test_tornado_skips_fixed_inputs(business):
    assert tornado(business).table.empty


# ---- compare -------------------------------------------------------------

def test_identical_plans_never_differ(data):
    data["streams"][0]["demand"] = [50, 100, 150]
    b = Business.model_validate(data)
    c = compare(b, Plan(name="same"), n=500)
    assert c.prob_b_better == 0
    assert c.profit_difference.p50 == 0


def test_common_random_numbers(data):
    # B only changes rent, so per-scenario revenue must match exactly.
    data["streams"][0]["demand"] = [50, 100, 150]
    b = Business.model_validate(data)
    plan = Plan(name="cheaper rent", changes=[{"path": "costs.fixed.rent", "op": "add", "value": -1000}])
    c = compare(b, plan, n=500)
    assert np.array_equal(c.result_a.revenue, c.result_b.revenue)
    assert c.prob_b_better == 1
    assert c.profit_difference.p50 == pytest.approx(3000)


def test_guesthouse_end_to_end():
    b = load_business(TEMPLATES / "guesthouse.yaml")
    s = summarize(simulate(b, n=2000, seed=42))
    assert 0 < s.prob_loss < 0.5
    t = tornado(b)
    assert t.table["input"].iloc[0] == "streams.rooms.demand"
    c = compare(b, get_plan(b, "Raise price 10%"), n=2000)
    assert 0 < c.prob_b_better < 1
    assert "change" in c.table().columns
