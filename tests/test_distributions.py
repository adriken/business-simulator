import numpy as np
import pytest

from simulator.distributions import Sampler, from_uniform, likely, parse_spec, sample


class TestParseSpec:
    def test_number_becomes_one_item_tuple(self):
        assert parse_spec(4500) == (4500.0,)

    def test_ranges_keep_their_length(self):
        assert parse_spec([1, 2]) == (1.0, 2.0)
        assert parse_spec([1, 2, 3]) == (1.0, 2.0, 3.0)

    @pytest.mark.parametrize(
        "bad",
        [True, "100", [1], [1, 2, 3, 4], [3, 1], [1, 5, 3], [1, "2"], [1, True], float("nan"), None],
    )
    def test_rejects_bad_input(self, bad):
        with pytest.raises(ValueError):
            parse_spec(bad)

    def test_equal_values_are_allowed(self):
        assert parse_spec([5, 5, 5]) == (5.0, 5.0, 5.0)


class TestSampling:
    def test_fixed_value(self):
        x = sample((7.0,), 100, np.random.default_rng(0))
        assert x.shape == (100,)
        assert np.all(x == 7.0)

    def test_degenerate_triangle_is_fixed(self):
        # NumPy's own rng.triangular raises here; our inverse CDF must not.
        x = sample((5.0, 5.0, 5.0), 10, np.random.default_rng(0))
        assert np.all(x == 5.0)

    def test_uniform_bounds_and_mean(self):
        x = sample((10.0, 20.0), 100_000, np.random.default_rng(1))
        assert x.min() >= 10 and x.max() < 20
        assert x.mean() == pytest.approx(15, abs=0.05)

    def test_triangular_bounds_and_mean(self):
        x = sample((0.0, 3.0, 12.0), 100_000, np.random.default_rng(2))
        assert x.min() >= 0 and x.max() <= 12
        assert x.mean() == pytest.approx(5.0, abs=0.05)  # (0 + 3 + 12) / 3

    def test_triangular_mode_split(self):
        # P(X < mode) = (c - a) / (b - a) = 0.25 for this triangle
        x = sample((0.0, 3.0, 12.0), 100_000, np.random.default_rng(3))
        assert np.mean(x < 3) == pytest.approx(0.25, abs=0.01)

    def test_inverse_cdf_is_monotonic(self):
        u = np.linspace(0, 0.999, 50)
        x = from_uniform((1.0, 2.0, 10.0), u)
        assert np.all(np.diff(x) > 0)

    def test_same_seed_same_numbers(self):
        a = sample((1.0, 2.0, 3.0), 50, np.random.default_rng(42))
        b = sample((1.0, 2.0, 3.0), 50, np.random.default_rng(42))
        assert np.array_equal(a, b)

    def test_likely(self):
        assert likely((4.0,)) == 4
        assert likely((2.0, 8.0)) == 5
        assert likely((1.0, 2.0, 9.0)) == 2


class TestSampler:
    def test_same_key_same_draws_regardless_of_order(self):
        s1, s2 = Sampler(100, seed=7), Sampler(100, seed=7)
        s1.draw("other", (0.0, 1.0))
        a = s1.draw("price", (1.0, 2.0))
        b = s2.draw("price", (1.0, 2.0))  # no "other" drawn first
        assert np.array_equal(a, b)

    def test_different_keys_are_independent(self):
        s = Sampler(20_000, seed=7)
        a = s.draw("a", (0.0, 1.0))
        b = s.draw("b", (0.0, 1.0))
        assert abs(np.corrcoef(a, b)[0, 1]) < 0.03

    def test_point_mode_and_overrides(self):
        s = Sampler(3, point=True, overrides={"x": 9.0})
        assert np.all(s.draw("x", (1.0, 5.0, 10.0)) == 9.0)
        assert np.all(s.draw("y", (1.0, 5.0, 10.0)) == 5.0)

    def test_rejects_bad_settings(self):
        with pytest.raises(ValueError):
            Sampler(0)
        with pytest.raises(ValueError):
            Sampler(10, seed=-1)
