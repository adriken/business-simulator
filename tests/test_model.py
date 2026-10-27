import pytest
from pydantic import ValidationError

from simulator.model import Business, Plan, check_plan, load_business

from .conftest import TEMPLATES


def test_guesthouse_template_loads():
    b = load_business(TEMPLATES / "guesthouse.yaml")
    assert b.name == "Small guesthouse"
    assert b.streams[0].price == (3500.0, 4500.0, 5500.0)
    assert len(b.plans) == 2


def test_unknown_key_is_an_error(data):
    data["streams"][0]["prise"] = 100  # typo
    with pytest.raises(ValidationError, match="prise"):
        Business.model_validate(data)


def test_unknown_capacity_pool(data):
    data["streams"][0]["capacity"] = "seats"
    with pytest.raises(ValidationError, match="seats"):
        Business.model_validate(data)


def test_per_unit_cost_must_point_at_a_stream(data):
    data["costs"]["per_unit"][0]["stream"] = "nothing"
    with pytest.raises(ValidationError, match="unknown stream"):
        Business.model_validate(data)


def test_one_off_after_horizon(data):
    data["costs"]["one_off"][0]["period"] = 4
    with pytest.raises(ValidationError, match="after the horizon"):
        Business.model_validate(data)


def test_loss_must_be_a_fraction(data):
    data["streams"][0]["loss"] = [0.1, 1.5]
    with pytest.raises(ValidationError, match="rate"):
        Business.model_validate(data)


def test_seasonality_must_divide_horizon(data):
    data["streams"][0]["seasonality"] = [1, 1]  # horizon is 3
    with pytest.raises(ValidationError, match="seasonality"):
        Business.model_validate(data)


def test_seasonality_repeats(data):
    data["horizon"] = 6
    data["streams"][0]["seasonality"] = [1, 2, 3]
    b = Business.model_validate(data)
    assert list(b.seasonality_for(b.streams[0])) == [1, 2, 3, 1, 2, 3]


def test_duplicate_names(data):
    data["costs"]["fixed"].append({"name": "rent", "amount": 1})
    with pytest.raises(ValidationError, match="both named 'rent'"):
        Business.model_validate(data)


def test_plan_with_unknown_path(business):
    plan = Plan(name="bad", changes=[{"path": "streams.main.prize", "op": "add", "value": 1}])
    with pytest.raises(ValueError, match="streams.main.prize"):
        check_plan(business, plan)


def test_plan_can_change_its_own_extra_costs(business):
    plan = Plan(
        name="ok",
        extra_costs={"fixed": [{"name": "marketing", "amount": 5000}]},
        changes=[{"path": "costs.fixed.marketing", "op": "multiply", "value": 2}],
    )
    check_plan(business, plan)  # no error


def test_business_is_read_only(business):
    with pytest.raises(ValidationError):
        business.horizon = 10


def test_parameters_lists_every_input(business):
    params = business.parameters()
    assert params["streams.main.price"] == (1000.0,)
    assert params["costs.fixed.rent"] == (50_000.0,)
    assert set(params) == business.parameter_paths()


def test_with_inputs_replaces_and_revalidates(business):
    edited = business.with_inputs({"streams.main.price": (900.0, 1100.0), "costs.fixed.rent": (60_000.0,)})
    assert edited.streams[0].price == (900.0, 1100.0)
    assert edited.costs.fixed[0].amount == (60_000.0,)
    assert business.streams[0].price == (1000.0,)  # the original is unchanged
    with pytest.raises(ValidationError):
        business.with_inputs({"streams.main.loss": (0.5, 1.5)})
    with pytest.raises(ValueError, match="isn't an input"):
        business.with_inputs({"streams.main.prize": (1.0,)})


def test_dump_round_trips(business):
    data = business.model_dump()
    assert data["streams"][0]["price"] == 1000.0  # fixed values dump as plain numbers
    assert Business.model_validate(data) == business
    assert Business.model_validate_json(business.model_dump_json()) == business
