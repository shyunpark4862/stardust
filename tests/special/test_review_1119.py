"""special: the Bessel order saturation, ndtri_exp at the end of the doubles and the
noncentral sums' complement (2026-11-19)."""

import math

import pytest

import stardust as sd

sp = sd.special


def test_jv_far_below_order_is_zero_not_nan():
    # J_nu(x) <= (x/2)^nu / Gamma(nu + 1) (DLMF 10.14.4): e^-1.4e7 here, 0
    assert sp.jv(3e7, 1.49e7) == 0.0
    assert sp.yv(3e7, 1.49e7) == -math.inf
    assert sp.jv(-3e7, 1.49e7) == 0.0 or math.isinf(sp.jv(-3e7, 1.49e7))
    # x = v has a value; x > v > 1e7 below the Hankel reach is NaN (documented)
    assert sp.jv(2e7, 2e7) == pytest.approx(0.0016479, rel=1e-4)
    assert math.isnan(sp.jv(2e7, 3e7))


@pytest.mark.parametrize("y", [-8.98e307, -8.99e307, -1e308, -1.7976931348623157e308])
def test_ndtri_exp_huge_negative(y):
    v = sp.ndtri_exp(y)
    want = -math.sqrt(2.0) * math.sqrt(-y)  # the asymptote's leading term, to ~log|y| / |y|
    assert v < 0 and math.isfinite(v)
    assert abs(v / want - 1.0) < 1e-15
    assert sd.stats.Normal().ilogcdf(y) == v
    assert sd.stats.Normal().ilogccdf(y) == -v
    assert sd.stats.Normal(mu=1.0, sigma=2.0).ilogcdf(y) == pytest.approx(1.0 + 2.0 * v, rel=1e-15)


def test_noncentral_chi_square_large_side_from_the_small():
    # the side near 1 ran out of terms from nc ~ 1.47e9; now 1 - the other side
    nc = 2.0e9
    s = math.sqrt(2.0 * (3.0 + 2.0 * nc))
    for k in (-6.0, -1.0, 1.0, 6.0):
        x = nc + 3.0 + k * s
        p, q = sp.chndtr(3.0, nc, x), sp.chndtrc(3.0, nc, x)
        assert math.isfinite(p) and math.isfinite(q)
        assert abs(p + q - 1.0) <= 4e-16
        assert p == pytest.approx(sp.ndtr(k), abs=2e-4)  # the normal limit, to its skewness
    assert math.isfinite(sp.chndtrix(3.0, 1.47e9, 0.5))
