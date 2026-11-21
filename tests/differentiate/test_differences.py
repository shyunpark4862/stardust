"""The accurate 5-point differences (docs/design/integrate.md §6.1; C3, C15, C25): one Differences
in this area, called by sd.optimize's approx_fprime and check_grad and by sd.regression.nls, and the
evaluation counts of its step search (R176: documented in docs/api/optimize.md §2, not changed, so
that approx_fprime's and nls's values stay the same)."""

import math

import pytest

import stardust as sd
from stardust.differentiate import _differences
from stardust.differentiate._differences import Differences


def test_one_kernel_for_optimize_and_nls():
    from stardust.optimize import _diff
    from stardust.regression import _nls

    assert _diff.Differences is Differences
    assert _nls._Differences is Differences
    assert not hasattr(sd.differentiate, "Differences")  # internal (C15)


def test_rounding_tolerance_is_sqrt_eps():
    # R175: 2^-26 is sqrt(eps) (eps = 2^-52), not sqrt(u) (u = 2^-53)
    assert _differences._ROUNDING_TOL == math.sqrt(2.0**-52) == 2.0**-26
    assert _differences._U == 2.0**-53


def _count(f, x0):
    n = [0]

    def g(v):
        n[0] += 1
        return f(float(v[0]))

    d = float(sd.optimize.approx_fprime([x0], g)[0])
    return d, n[0]


@pytest.mark.parametrize(
    ("x0", "calls"),
    [(0.0, 17), (1e-300, 25), (1e-30, 25), (1e-12, 61), (1e-8, 61), (1e-4, 13), (1e-2, 13), (1.0, 17)],
)
def test_exp_near_zero_counts(x0, calls):
    # f(x0) once, then 8 for the first two steps and 4 per further step; near 0 (not 0) the start is
    # at rounding level and the step grows by factors of 4 until the truncation error shows
    d, n = _count(math.exp, x0)
    assert n == calls
    # 2.6e-10 at 1e-4, where the start is not at rounding level and the search only goes down
    assert abs(d - math.exp(x0)) <= 1e-9


@pytest.mark.parametrize("f", [math.sin, math.atan, math.tanh, math.log1p])
def test_small_values_near_zero_count_less(f):
    # a function whose values are small near 0 is not at rounding level there
    for x0 in (1e-300, 1e-30, 1e-12, 1e-8):
        _, n = _count(f, x0)
        assert 9 <= n <= 25, (f, x0, n)
