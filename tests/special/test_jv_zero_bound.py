"""jv's zero decision by DLMF 10.14.5 (user decision 2026-11-19, docs/design/special.md Q20):
|J_nu(nu sech a)| <= e^(-nu (a - tanh a)); below e^-746.2 (under half the smallest subnormal)
jv is 0. Arb (python-flint) checks the points next to the boundary."""

import math

import pytest
from flint import arb, ctx

import stardust as sd

sp = sd.special


def _log_bound(nu, x):
    """-nu (a - tanh a), x = nu sech a (0 < x < nu)."""
    t = math.sqrt((nu - x) / nu) * math.sqrt((nu + x) / nu)
    return -nu * (math.atanh(t) - t)


def _boundary(nu):
    lo, hi = 0.0, nu
    for _ in range(200):
        m = 0.5 * (lo + hi)
        if _log_bound(nu, m) < -746.2:
            lo = m
        else:
            hi = m
    return lo


def _arb_j(nu, x):
    """J_nu(x) as a float from Arb, its radius below 2^-60 of it (or certainly below 2^-1075)."""
    p = 256
    while p <= 140000:
        ctx.prec = p
        try:
            r = arb(x).bessel_j(arb(nu))
            if abs(r.mid()) + r.rad() < arb(2) ** -1075:
                return 0.0
            if r.rad() < abs(r.mid()) * arb(2) ** -60:
                return float(r.mid())
        finally:
            ctx.prec = 53
        p *= 2
    raise AssertionError("Arb did not settle")


def test_far_order_is_zero_not_nan():
    # e^-7.6e6: x = 0.63 nu is beyond the older (x/2)^nu / Gamma test (x < nu / 2); scipy 0
    assert sp.jv(3e7, 1.9e7) == 0.0 and sp.jve(3e7, 1.9e7) == 0.0
    assert sp.jv(3e7, 2.9e7) == 0.0
    # an order beyond 1e30 next to x: t from nu - x keeps the bound
    assert sp.jv(1e30, 1e30 - 1e21) == 0.0
    # negative integer orders: (-1)^n 0; a negative non-integer order needs Y (not decided here)
    assert sp.jv(-3e7, 1.9e7) == 0.0 and sp.jv(-30000001.0, 1.9e7) == 0.0
    # Y has no rigorous lower bound: still the 1e7-step recurrence, NaN (documented)
    assert math.isnan(sp.yv(3e7, 1.9e7))
    # (the older test, J and Y at once, is unchanged)
    assert sp.jv(3e7, 1.49e7) == 0.0 and sp.yv(3e7, 1.49e7) == -math.inf


def test_tiny_argument_is_not_decided_by_the_bound():
    # t rounds to 1 for a tiny x: the bound is taken from log(nu / x) + log1p(t) - t
    assert sp.jv(0.5, 5e-324) == pytest.approx(1.7735048886036274e-162, rel=1e-13)
    assert sp.jv(2.0, 1e-160) == pytest.approx(1.25e-321, abs=5e-323)  # (x/2)^2 / 2
    assert sp.jv(2.0, 1e-200) == 0.0


@pytest.mark.parametrize("nu", [60.0, 1e3, 1e4])
@pytest.mark.parametrize("f", [0.99, 0.9999999, 1.0000001, 1.001, 1.01, 1.03])
def test_next_to_the_boundary_against_arb(nu, f):
    x = _boundary(nu) * f
    got, want = sp.jv(nu, x), _arb_j(nu, x)
    if f < 1.0:
        assert got == 0.0 and want == 0.0
    # values: relative 1e-14, or the subnormals' absolute 45 2^-1074
    assert abs(got - want) <= max(1e-14 * abs(want), 45 * 2.0**-1074), (got, want)


def test_jv_far_below_order_is_zero_not_nan():
    # J_nu(x) <= (x/2)^nu / Gamma(nu + 1) (DLMF 10.14.4): e^-1.4e7 here, 0
    assert sp.jv(3e7, 1.49e7) == 0.0
    assert sp.yv(3e7, 1.49e7) == -math.inf
    assert sp.jv(-3e7, 1.49e7) == 0.0 or math.isinf(sp.jv(-3e7, 1.49e7))
    # x = v has a value; x > v > 1e7 below the Hankel reach is NaN (documented)
    assert sp.jv(2e7, 2e7) == pytest.approx(0.0016479, rel=1e-4)
    assert math.isnan(sp.jv(2e7, 3e7))
