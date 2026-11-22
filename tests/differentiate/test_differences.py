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
    [(0.0, 17), (1e-300, 25), (1e-30, 25), (1e-12, 61), (1e-8, 61), (3e-6, 57), (1e-5, 53), (1e-4, 49), (1e-2, 13),
     (1.0, 17)],
)  # fmt: skip
def test_exp_near_zero_counts(x0, calls):
    # f(x0) once, then 8 for the first two steps and 4 per further step; near 0 (not 0) the start is
    # at rounding level, or above 2^-36 of the column (C112), and the step grows by factors of 4
    # until the truncation error shows
    d, n = _count(math.exp, x0)
    assert n == calls
    # before C112 the start 2^-8 |x0| was kept when its rounding level was below sqrt(eps): 1.5e-8
    # at 3e-6, 5.1e-9 at 1e-5, 2.6e-10 at 1e-4 (the search only goes down)
    assert abs(d - math.exp(x0)) <= 1e-11 * math.exp(x0)


@pytest.mark.parametrize("f", [math.sin, math.atan, math.tanh, math.log1p])
def test_small_values_near_zero_count_less(f):
    # a function whose values are small near 0 is not at rounding level there
    for x0 in (1e-300, 1e-30, 1e-12, 1e-8):
        _, n = _count(f, x0)
        assert 9 <= n <= 25, (f, x0, n)


def _exp_model(t):
    import numpy as np

    x = np.linspace(0.01, 5.0, 50)
    return t[0] * np.exp(-t[1] * x)


def test_reused_buffer_and_strided_values_give_the_same_jacobian():
    # each returned array is read once, right after its call (csrc/differentiate/accurate.hpp):
    # a g that overwrites one buffer, or returns a strided view, gets the fresh arrays' Jacobian
    import numpy as np

    buf = sd.zeros(50)

    def fresh(t):
        return sd.asarray(_exp_model(t))

    def reused(t):
        buf[:] = sd.asarray(_exp_model(t))
        return buf

    def strided(t):
        return sd.asarray(np.repeat(_exp_model(t), 2))[::2]

    x, inf = [2.0, 0.8], [math.inf] * 2
    g0 = fresh(x)
    want, _ = Differences(fresh, 2, [-v for v in inf], inf).jacobian(x, g0, zero_noise=True)
    for g in (reused, strided):
        got, bad = Differences(g, 2, [-v for v in inf], inf).jacobian(x, g0, zero_noise=True)
        assert bad is None
        assert np.array_equal(np.asarray(got), np.asarray(want))


def test_second_differences_reuse_the_central_values():
    import numpy as np

    calls = [0]

    def g(t):
        calls[0] += 1
        return sd.asarray(_exp_model(t))

    x, inf = [2.0, 0.8], [math.inf] * 2
    g0 = g(x)
    D = Differences(g, 2, [-v for v in inf], inf)
    D.jacobian(x, g0)
    before = calls[0]
    c = sd.asarray(np.linspace(-1.0, 1.0, 50))
    S = D.second(x, g0, c, [0, 1])
    assert calls[0] - before == 4  # the diagonal from the Jacobian's central values, one 4-point cross
    assert S[0][1] == S[1][0]
