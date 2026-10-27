"""The editable "Your numbers" table: Business inputs <-> a pandas DataFrame.

Kept out of app.py so it can be tested without Streamlit.
One row per input, with columns Low / Likely / High:
    fixed value 4500            -> 4500 / 4500 / 4500
    uniform [3500, 5500]        -> 3500 / (blank) / 5500
    triangular [3500, 4500, 5500] -> 3500 / 4500 / 5500
"""

from __future__ import annotations

import math

import pandas as pd

from .distributions import parse_spec
from .formatting import input_label
from .model import Business

HIDDEN_DEFAULTS = {"capacity_per_unit": (1.0,), "start_units": (0.0,)}


def inputs_frame(business: Business) -> pd.DataFrame:
    """One editable row per input. A blank 'Likely' means a uniform range."""
    rows = []
    for path, spec in business.parameters().items():
        field = path.rsplit(".", 1)[-1]
        if HIDDEN_DEFAULTS.get(field) == spec:
            continue  # technical defaults most people never change
        if len(spec) == 1:
            lo, mid, hi = spec[0], spec[0], spec[0]
        elif len(spec) == 2:
            lo, mid, hi = spec[0], None, spec[1]
        else:
            lo, mid, hi = spec
        rows.append({"path": path, "Input": input_label(path), "Low": lo, "Likely": mid, "High": hi})
    return pd.DataFrame(rows).set_index("path")


def _missing(v: object) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def read_frame(edited: pd.DataFrame) -> tuple[dict[str, tuple[float, ...]], list[str]]:
    """Turn the edited table back into specs, collecting readable errors."""
    updates: dict[str, tuple[float, ...]] = {}
    errors: list[str] = []
    for path, row in edited.iterrows():
        lo, mid, hi = row["Low"], row["Likely"], row["High"]
        if _missing(lo) or _missing(hi):
            errors.append(f"**{row['Input']}**: fill in both Low and High.")
            continue
        if _missing(mid):
            raw = [lo, hi]
        elif lo == mid == hi:
            raw = [lo]
        else:
            raw = [lo, mid, hi]
        try:
            updates[str(path)] = parse_spec(raw[0] if len(raw) == 1 else raw)
        except ValueError as e:
            errors.append(f"**{row['Input']}**: {e}")
    return updates, errors
