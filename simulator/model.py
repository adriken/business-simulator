"""The business schema: what a YAML template may contain, and the checks on it.

A YAML file is untrusted input, so it is validated here before the engine
sees it. Pydantic turns each section into a typed, read-only object, and
every mistake (a typo in a key, a range in the wrong order, a cost pointing
at a stream that doesn't exist) is reported with its location in the file.

Every number in the file becomes a `Spec` (see distributions.py). Each one
also has a dotted "path" such as `streams.rooms.price` or `costs.fixed.staff`.
Plans use these paths to say what they change, and the tornado chart uses
them as row labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    field_validator,
    model_validator,
)

from .distributions import FloatArray, parse_spec
from .distributions import Spec as SpecTuple


def to_raw(spec: SpecTuple) -> float | list[float]:
    """The inverse of parse_spec: (4500.0,) -> 4500.0, (1.0, 2.0) -> [1.0, 2.0]."""
    return spec[0] if len(spec) == 1 else list(spec)


# A number or range from YAML, parsed into a tuple of 1-3 floats on the way in,
# and written back the way a person would type it on the way out (model_dump),
# so a dumped business can always be loaded again.
Spec = Annotated[tuple[float, ...], BeforeValidator(parse_spec), PlainSerializer(to_raw)]
# Names become parts of paths, so no dots or spaces.
Name = Annotated[str, Field(pattern=r"^[A-Za-z][A-Za-z0-9_]*$")]


def _within(spec: SpecTuple, what: str, lo: float | None = None, hi: float | None = None) -> SpecTuple:
    if lo is not None and spec[0] < lo:
        raise ValueError(f"{what} cannot be below {lo:g}, got {list(spec)}")
    if hi is not None and spec[-1] > hi:
        raise ValueError(f"{what} cannot be above {hi:g}, got {list(spec)}")
    return spec


class _Frozen(BaseModel):
    # extra="forbid": an unknown key (say, "prise") is an error instead of being ignored.
    # frozen=True: objects can't be changed after loading, so a run can't alter its input.
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stream(_Frozen):
    """One revenue stream: units x price, limited by capacity, reduced by losses.

    For a recurring stream (anything with `churn`), `demand` means new
    customers per period and units are the active customer base, which
    carries over from period to period.
    """

    name: Name
    demand: Spec
    price: Spec
    loss: Spec = (0.0,)
    capacity: Name | None = None
    capacity_per_unit: Spec = (1.0,)
    seasonality: tuple[float, ...] | None = None
    churn: Spec | None = None
    start_units: Spec = (0.0,)

    @field_validator("demand", "price", "start_units")
    @classmethod
    def _not_negative(cls, v: SpecTuple) -> SpecTuple:
        return _within(v, "value", lo=0)

    @field_validator("loss", "churn")
    @classmethod
    def _is_fraction(cls, v: SpecTuple | None) -> SpecTuple | None:
        return None if v is None else _within(v, "a rate (0 to 1)", lo=0, hi=1)

    @field_validator("capacity_per_unit")
    @classmethod
    def _positive(cls, v: SpecTuple) -> SpecTuple:
        if v[0] <= 0:
            raise ValueError(f"capacity used per unit must be above 0, got {list(v)}")
        return v

    @field_validator("seasonality")
    @classmethod
    def _season_ok(cls, v: tuple[float, ...] | None) -> tuple[float, ...] | None:
        if v is not None and (len(v) == 0 or min(v) < 0):
            raise ValueError("seasonality must be a non-empty list of numbers 0 or above")
        return v

    @property
    def recurring(self) -> bool:
        return self.churn is not None


class FixedCost(_Frozen):
    """Paid every period regardless of sales: rent, salaries."""

    name: Name
    amount: Spec

    @field_validator("amount")
    @classmethod
    def _not_negative(cls, v: SpecTuple) -> SpecTuple:
        return _within(v, "cost", lo=0)


class PerUnitCost(FixedCost):
    """Scales with volume. `basis` chooses which volume:

    * produced: paid on everything made, including what is later lost
      (flour for loaves that go stale, feed for birds that die)
    * sold: paid only on what is actually sold (laundry for occupied rooms)
    """

    stream: Name
    basis: Literal["produced", "sold"] = "sold"


class OneOffCost(FixedCost):
    """Paid once, in a given period (1 = the first period)."""

    period: int = Field(ge=1)


class Costs(_Frozen):
    fixed: tuple[FixedCost, ...] = ()
    per_unit: tuple[PerUnitCost, ...] = ()
    one_off: tuple[OneOffCost, ...] = ()

    def merged(self, other: Costs) -> Costs:
        return Costs(
            fixed=self.fixed + other.fixed,
            per_unit=self.per_unit + other.per_unit,
            one_off=self.one_off + other.one_off,
        )


class Change(_Frozen):
    """One edit a plan makes to an input, e.g. multiply streams.rooms.price by 1.1.

    `value` can be a range: "demand falls by 2-15%" is written as a
    multiply by [0.85, 0.92, 0.98].
    """

    path: str
    op: Literal["set", "multiply", "add"]
    value: Spec


class Plan(_Frozen):
    """An alternative to compare against the business as written."""

    name: str
    description: str = ""
    changes: tuple[Change, ...] = ()
    extra_costs: Costs = Costs()


class Business(_Frozen):
    name: str
    period: Literal["day", "week", "month"]
    horizon: int = Field(gt=0, le=520)
    currency: str = "KES"
    starting_cash: Spec = (0.0,)
    capacity: dict[Name, Spec] = {}
    streams: tuple[Stream, ...] = Field(min_length=1)
    costs: Costs = Costs()
    plans: tuple[Plan, ...] = ()

    @field_validator("capacity")
    @classmethod
    def _capacity_positive(cls, v: dict[str, SpecTuple]) -> dict[str, SpecTuple]:
        for key, spec in v.items():
            _within(spec, f"capacity '{key}'", lo=0)
        return v

    @model_validator(mode="after")
    def _check_references(self) -> Business:
        """Cross-field checks that a single field can't do on its own."""
        _unique([s.name for s in self.streams], "stream")
        _unique([c.name for c in self.costs.fixed], "fixed cost")
        _unique([c.name for c in self.costs.per_unit], "per-unit cost")
        _unique([c.name for c in self.costs.one_off], "one-off cost")
        _unique([p.name for p in self.plans], "plan")

        stream_names = {s.name for s in self.streams}
        for s in self.streams:
            if s.capacity is not None and s.capacity not in self.capacity:
                raise ValueError(
                    f"stream '{s.name}' uses capacity '{s.capacity}', which isn't "
                    f"defined under capacity: (defined: {sorted(self.capacity) or 'none'})"
                )
            if s.recurring and s.capacity is not None:
                raise ValueError(f"stream '{s.name}': recurring streams can't use capacity yet")
            if s.seasonality is not None and self.horizon % len(s.seasonality) != 0:
                raise ValueError(
                    f"stream '{s.name}': seasonality has {len(s.seasonality)} values, "
                    f"which doesn't divide the horizon of {self.horizon} periods"
                )
        for c in self.costs.per_unit:
            if c.stream not in stream_names:
                raise ValueError(f"per-unit cost '{c.name}' refers to unknown stream '{c.stream}'")
        for c in self.costs.one_off:
            if c.period > self.horizon:
                raise ValueError(
                    f"one-off cost '{c.name}' is in period {c.period}, "
                    f"after the horizon of {self.horizon}"
                )
        for plan in self.plans:
            check_plan(self, plan)
        return self

    def parameters(self) -> dict[str, SpecTuple]:
        """Every input, path -> spec, in the order a person would read them."""
        params: dict[str, SpecTuple] = {}
        for s in self.streams:
            params[f"streams.{s.name}.demand"] = s.demand
            params[f"streams.{s.name}.price"] = s.price
            params[f"streams.{s.name}.loss"] = s.loss
            if s.churn is not None:
                params[f"streams.{s.name}.churn"] = s.churn
            params[f"streams.{s.name}.start_units"] = s.start_units
            params[f"streams.{s.name}.capacity_per_unit"] = s.capacity_per_unit
        for key, spec in self.capacity.items():
            params[f"capacity.{key}"] = spec
        for kind in ("per_unit", "fixed", "one_off"):
            for c in getattr(self.costs, kind):
                params[f"costs.{kind}.{c.name}"] = c.amount
        params["starting_cash"] = self.starting_cash
        return params

    def parameter_paths(self) -> set[str]:
        """Every input a plan is allowed to change."""
        return set(self.parameters())

    def with_inputs(self, updates: dict[str, SpecTuple]) -> Business:
        """A copy with some inputs replaced, e.g. {"streams.rooms.price": (4000, 5000)}.

        The copy is fully re-validated, so a bad value raises ValidationError.
        """
        data = self.model_dump()
        known = self.parameter_paths()
        for path, new in updates.items():
            spec = to_raw(new)
            if path not in known:
                raise ValueError(f"'{path}' isn't an input of '{self.name}'")
            parts = path.split(".")
            if parts[0] == "starting_cash":
                data["starting_cash"] = spec
            elif parts[0] == "capacity":
                data["capacity"][parts[1]] = spec
            elif parts[0] == "streams":
                stream = next(s for s in data["streams"] if s["name"] == parts[1])
                stream[parts[2]] = spec
            else:  # costs.<kind>.<name>
                cost = next(c for c in data["costs"][parts[1]] if c["name"] == parts[2])
                cost["amount"] = spec
        return Business.model_validate(data)

    def seasonality_for(self, stream: Stream) -> FloatArray:
        """The per-period demand multiplier, repeated to fill the horizon."""
        if stream.seasonality is None:
            return np.ones(self.horizon)
        reps = self.horizon // len(stream.seasonality)
        return np.tile(np.asarray(stream.seasonality, dtype=float), reps)

    def with_plan_costs(self, plan: Plan) -> Business:
        """A copy of this business with the plan's extra costs added (and re-validated)."""
        data = self.model_dump()
        data["costs"] = self.costs.merged(plan.extra_costs).model_dump()
        data["plans"] = []
        return Business.model_validate(data)


def _unique(names: list[str], what: str) -> None:
    seen: set[str] = set()
    for n in names:
        if n in seen:
            raise ValueError(f"two {what}s are both named '{n}'")
        seen.add(n)


def check_plan(business: Business, plan: Plan) -> None:
    """Raise ValueError if the plan refers to inputs the business doesn't have."""
    merged = business.with_plan_costs(plan)
    known = merged.parameter_paths()
    for change in plan.changes:
        if change.path not in known:
            raise ValueError(
                f"plan '{plan.name}' changes '{change.path}', which isn't an input of "
                f"'{business.name}'. Known inputs: {', '.join(sorted(known))}"
            )


def load_business(path: str | Path) -> Business:
    """Read and validate a business template from a YAML or JSON file."""
    with open(path, encoding="utf-8") as f:
        # safe_load only builds plain data (dicts, lists, numbers, strings).
        # yaml.load can construct arbitrary Python objects, which is unsafe
        # for files you didn't write yourself.
        data = yaml.safe_load(f)  # JSON is valid YAML, so this reads both
    return Business.model_validate(data)


def get_plan(business: Business, name: str) -> Plan:
    for plan in business.plans:
        if plan.name == name:
            return plan
    raise KeyError(f"no plan named '{name}' in '{business.name}'")
