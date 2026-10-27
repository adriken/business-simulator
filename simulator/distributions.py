"""Uncertain numbers: parse them from YAML and turn them into random samples.

Any number in a business file can be written in one of three ways:

    4500                  fixed value
    [3500, 5500]          uniform: anywhere in the range, all values equally likely
    [3500, 4500, 5500]    triangular: low, most likely, high

Internally every number is normalised to a "spec": a tuple of 1, 2 or 3 floats.

Sampling uses the inverse-CDF method. We first draw a uniform number u in [0, 1),
then map it through the distribution's inverse cumulative distribution function.
Keeping u separate from the distribution is what lets two plans share the same
random "worlds" (see Sampler below).
"""

from __future__ import annotations

import math
import numbers
import zlib
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

Spec = tuple[float, ...]
FloatArray = NDArray[np.float64]


def parse_spec(raw: Any) -> Spec:
    """Validate a number or range from YAML and return it as a tuple of floats.

    Raises ValueError with a message a non-programmer can act on.
    """
    # bool is a subclass of int in Python, so `true` in YAML would pass as 1.
    # Reject it explicitly: it is almost always a typo.
    if isinstance(raw, bool):
        raise ValueError("expected a number or a list of numbers, got true/false")

    if isinstance(raw, numbers.Real):
        values: list[Any] = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, str):
        values = list(raw)
        if len(values) not in (2, 3):
            raise ValueError(
                "a range needs 2 values [low, high] or 3 values "
                f"[low, likely, high], got {len(values)}: {values}"
            )
        if any(isinstance(v, bool) or not isinstance(v, numbers.Real) for v in values):
            raise ValueError(f"every value in a range must be a number, got {values}")
    else:
        raise ValueError(f"expected a number or a list of numbers, got {raw!r}")

    spec = tuple(float(v) for v in values)
    if not all(math.isfinite(v) for v in spec):
        raise ValueError(f"values must be finite numbers, got {list(spec)}")
    # Wrong order is rejected, never silently swapped: a swap would hide a
    # mistake in the user's file.
    if any(a > b for a, b in zip(spec, spec[1:])):
        raise ValueError(
            f"range values must be in order low <= likely <= high, got {list(spec)}"
        )
    return spec


def low(spec: Spec) -> float:
    return spec[0]


def high(spec: Spec) -> float:
    return spec[-1]


def likely(spec: Spec) -> float:
    """The single most representative value: the mode, or the midpoint of a uniform."""
    if len(spec) == 2:
        return (spec[0] + spec[1]) / 2
    return spec[0] if len(spec) == 1 else spec[1]


def is_uncertain(spec: Spec) -> bool:
    return spec[0] != spec[-1]


def from_uniform(spec: Spec, u: FloatArray) -> FloatArray:
    """Map uniform numbers u in [0, 1) to samples of `spec` (inverse CDF).

    The same u always gives the same position in the distribution: u = 0.1
    is "a bad draw" (near the low end) whatever the range is.
    """
    a, b = spec[0], spec[-1]
    if a == b:  # fixed value, or a degenerate range like [5000, 5000, 5000]
        return np.full_like(u, a)
    if len(spec) == 2:  # uniform: stretch [0, 1) onto [a, b)
        return a + u * (b - a)

    # Triangular with low a, mode c, high b. The CDF is two quadratic pieces
    # that meet at the mode; `split` is the probability of landing left of it.
    c = spec[1]
    split = (c - a) / (b - a)
    left = a + np.sqrt(u * (b - a) * (c - a))
    right = b - np.sqrt((1 - u) * (b - a) * (b - c))
    return np.where(u < split, left, right)


def sample(spec: Spec, n: int, rng: np.random.Generator) -> FloatArray:
    """Draw n samples of `spec` from a NumPy generator.

    The generator is passed in rather than created here so callers control
    the seed: the same seed reproduces the same results, which makes tests
    deterministic and lets two plans be compared on identical random draws.
    """
    return from_uniform(spec, rng.random(n))


class Sampler:
    """Hands out samples for named inputs such as "streams.rooms.price".

    Each input name gets its own random stream, seeded from (seed, name).
    That gives two guarantees:

    * Reproducible: the same seed always produces the same scenarios.
    * Common random numbers: scenario 17's price draw is identical in Plan A
      and Plan B, even if Plan B adds or removes other inputs. Differences
      between plans therefore come from the plan, not from luck.

    Point mode (point=True) skips randomness and returns each input's likely
    value; `overrides` pins chosen inputs to exact values. The tornado chart
    uses both.
    """

    def __init__(
        self,
        n: int,
        seed: int = 42,
        *,
        point: bool = False,
        overrides: Mapping[str, float] | None = None,
    ) -> None:
        if n < 1:
            raise ValueError("number of scenarios must be at least 1")
        if seed < 0:
            raise ValueError("seed must be zero or positive")
        self.n = n
        self.seed = seed
        self.point = point
        self.overrides = dict(overrides or {})
        self.specs: dict[str, Spec] = {}  # every input drawn, for the tornado chart
        self.drawn: dict[str, FloatArray] = {}  # the sampled values, for inspection

    def uniforms(self, key: str) -> FloatArray:
        # crc32 turns the name into a stable integer. Python's built-in hash()
        # can't be used: it changes between runs for strings.
        stream = np.random.default_rng([self.seed, zlib.crc32(key.encode("utf-8"))])
        return stream.random(self.n)

    def draw(self, key: str, spec: Spec) -> FloatArray:
        self.specs[key] = spec
        if key in self.overrides:
            values = np.full(self.n, float(self.overrides[key]))
        elif self.point:
            values = np.full(self.n, likely(spec))
        elif not is_uncertain(spec):
            values = np.full(self.n, spec[0])
        else:
            values = from_uniform(spec, self.uniforms(key))
        self.drawn[key] = values
        return values
