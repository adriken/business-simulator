"""Shared test fixtures.

`make_business` builds a small business with fixed numbers, so expected
results can be worked out by hand.
"""

from pathlib import Path
from typing import Any

import pytest

from simulator.model import Business

TEMPLATES = Path(__file__).resolve().parent.parent / "templates"


def base_data() -> dict[str, Any]:
    return {
        "name": "Test shop",
        "period": "month",
        "horizon": 3,
        "starting_cash": 0,
        "capacity": {"slots": 300},
        "streams": [
            {"name": "main", "demand": 100, "price": 1000, "loss": 0.1, "capacity": "slots"}
        ],
        "costs": {
            "fixed": [{"name": "rent", "amount": 50_000}],
            "per_unit": [{"name": "materials", "stream": "main", "basis": "sold", "amount": 100}],
            "one_off": [{"name": "repair", "period": 2, "amount": 10_000}],
        },
    }


@pytest.fixture
def data() -> dict[str, Any]:
    return base_data()


@pytest.fixture
def business(data: dict[str, Any]) -> Business:
    return Business.model_validate(data)
