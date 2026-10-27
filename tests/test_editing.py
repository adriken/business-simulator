import numpy as np

from simulator.editing import inputs_frame, read_frame
from simulator.model import load_business

from .conftest import TEMPLATES


def guesthouse():
    return load_business(TEMPLATES / "guesthouse.yaml")


def test_frame_has_one_row_per_visible_input():
    df = inputs_frame(guesthouse())
    assert "streams.rooms.price" in df.index
    assert "streams.rooms.capacity_per_unit" not in df.index  # hidden default
    assert list(df.loc["streams.rooms.price", ["Low", "Likely", "High"]]) == [3500, 4500, 5500]
    assert list(df.loc["starting_cash", ["Low", "Likely", "High"]]) == [250_000] * 3


def test_unedited_frame_round_trips_to_the_same_business():
    b = guesthouse()
    updates, errors = read_frame(inputs_frame(b))
    assert errors == []
    assert b.with_inputs(updates) == b


def test_blank_likely_means_uniform_and_equal_means_fixed():
    df = inputs_frame(guesthouse())
    df.loc["streams.rooms.price", "Likely"] = np.nan
    df.loc["costs.fixed.staff", ["Low", "Likely", "High"]] = [140_000, 140_000, 140_000]
    updates, errors = read_frame(df)
    assert errors == []
    assert updates["streams.rooms.price"] == (3500.0, 5500.0)
    assert updates["costs.fixed.staff"] == (140_000.0,)


def test_bad_rows_give_readable_errors():
    df = inputs_frame(guesthouse())
    df.loc["streams.rooms.price", "Low"] = 9000  # above likely
    df.loc["costs.fixed.staff", "High"] = np.nan
    _, errors = read_frame(df)
    assert len(errors) == 2
    assert any("rooms: price" in e and "order" in e for e in errors)
    assert any("staff" in e and "Low and High" in e for e in errors)
