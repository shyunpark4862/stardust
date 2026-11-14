"""S6: the Bessel functions of order 0 and 1, of real order, and airy (docs/api/special.md §8, §9).

Accuracy is tests/special/test_reference.py's (the oscillatory functions through the
envelope scale stored with their references). Here: the special values against scipy
except where docs/design/special.md §2.4 says otherwise (S6 user decision: true values),
the reflection and argument rules, and AiryResult."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd

S = sd.special
SUBNORMAL_ABS = 45 * 2.0**-1074
ORDER01 = ["i0", "i0e", "i1", "i1e", "j0", "j1", "y0", "y1", "k0", "k0e", "k1", "k1e"]


def same_bits(a, b):
    a, b = np.float64(a), np.float64(b)
    return bool(np.isnan(a) and np.isnan(b)) or a.view(np.uint64) == b.view(np.uint64)


@pytest.mark.parametrize("name", ORDER01)
def test_order01_edges(name):
    f, g = getattr(S, name), getattr(sp, name)
    for x in [0.0, -0.0, 1.0, -1.0, np.nan, 700.0, -700.0, 1e300, -1e300]:
        got, want = f(x), float(g(x))
        if want == 0 or not np.isfinite(want) or not np.isfinite(got):
            if name in ("j0", "j1", "y0", "y1") and abs(x) == 1e300 and not (name[0] == "y" and x < 0):
                assert abs(got) < 1e-149  # scipy's value at 1e300 is an O(1e-150) number; both are
                continue
            assert same_bits(got, want), (name, x, got, want)


def test_order01_true_values():
    # S6 user decision (scipy NaN, inf or 0 there)
    assert S.i0(np.inf) == np.inf and S.i0(-np.inf) == np.inf
    assert S.i1(np.inf) == np.inf and S.i1(-np.inf) == -np.inf
    for name in ["j0", "j1", "y0", "y1"]:
        assert getattr(S, name)(np.inf) == 0.0
    assert abs(S.i0(713.5) / 1.1051012081178279e308 - 1) < 1e-14  # scipy inf from 713
    assert S.k1(5e-324) == np.inf and S.k1e(5e-324) == np.inf  # scipy NaN
    assert abs(S.k0(5e-324) - 744.5560034370395) < 1e-12  # scipy inf (x/2 underflowed)


def test_order01_symmetry():
    x = np.array([0.3, 2.0, 9.0, 40.0])
    for name, parity in [("i0", 1), ("i1", -1), ("i0e", 1), ("i1e", -1), ("j0", 1), ("j1", -1)]:
        f = getattr(S, name)
        for v in x:
            assert f(-v) == parity * f(v), (name, v)
    for name in ["y0", "y1", "k0", "k1", "k0e", "k1e"]:
        assert np.isnan(getattr(S, name)(-1.0))


def test_real_order_matches_order01():
    for x in [0.1, 1.5, 7.0, 33.0, 400.0]:
        for n, (j, y, i, k) in enumerate([(S.j0, S.y0, S.i0, S.k0), (S.j1, S.y1, S.i1, S.k1)]):
            for got, want in [(S.jv(n, x), j(x)), (S.yv(n, x), y(x)), (S.iv(n, x), i(x)), (S.kv(n, x), k(x))]:
                assert abs(got - want) <= 1e-14 * max(abs(want), 1e-300) + 1e-14 / np.sqrt(x), (n, x, got, want)


def test_real_order_true_values():
    # S6 user decision: limits where scipy gives NaN or the wrong sign
    assert S.jv(0.5, np.inf) == 0.0 and S.yv(0.5, np.inf) == 0.0 and S.iv(0.5, np.inf) == np.inf
    assert S.yv(np.inf, 1.0) == -np.inf and S.kv(1e10, 1.0) == np.inf and S.kve(1e10, 1.0) == np.inf
    assert S.ive(1e10, 1.0) == 0.0 and S.yve(1e10, 1.0) == -np.inf
    assert S.iv(-1.5, 0.0) == -np.inf and S.iv(-0.5, 0.0) == np.inf
    assert S.yv(-0.5, 0.0) == 0.0 and S.yv(-1.5, 0.0) == 0.0
    assert S.jve(-0.5, 0.0) == S.jv(-0.5, 0.0) == np.inf and S.yve(0.5, 0.0) == -np.inf
    assert S.ive(-0.5, 0.0) == np.inf


def test_real_order_scipy_conventions():
    pts = [(0.0, 0.0), (1.0, 0.0), (0.5, 0.0), (-1.0, 0.0), (2.0, -1.0), (2.5, -1.0), (3.0, -2.0), (0.0, np.nan),
           (np.nan, 1.0), (1.0, -0.0)]
    for name in ["jv", "iv", "kv"]:
        f, g = getattr(S, name), getattr(sp, name)
        for v, x in pts:
            got, want = f(v, x), float(g(v, x))
            assert same_bits(got, want) or abs(got - want) <= 1e-14 * abs(want), (name, v, x, got, want)
    assert np.isnan(S.yv(2.0, -1.0)) and np.isnan(S.kv(2.0, -1.0)) and np.isnan(S.jv(2.5, -1.0))


def test_scaled_versions():
    rng = np.random.default_rng(3)
    for v, x in zip(rng.uniform(-20, 20, 50), np.exp(rng.uniform(-3, 5, 50))):
        assert S.jve(v, x) == S.jv(v, x) and S.yve(v, x) == S.yv(v, x)  # real x: exp(-|Im x|) = 1
        assert abs(S.ive(abs(v), x) - S.iv(abs(v), x) * np.exp(-x)) <= 1e-13 * S.ive(abs(v), x)
        assert abs(S.kve(v, x) - S.kv(v, x) * np.exp(x)) <= 1e-13 * S.kve(v, x)


def test_reflection():
    # J_-v = cos(v pi) J_v - sin(v pi) Y_v; integer orders: (-1)^n
    for v, x in [(0.3, 2.0), (2.7, 10.0), (11.5, 40.0)]:
        c, s = np.cos(np.pi * v), np.sin(np.pi * v)
        want = c * S.jv(v, x) - s * S.yv(v, x)
        assert abs(S.jv(-v, x) - want) <= 1e-13 * np.hypot(S.jv(v, x), S.yv(v, x))
    assert S.jv(-3, 2.0) == -S.jv(3, 2.0) and S.iv(-3, 2.0) == S.iv(3, 2.0) and S.kv(-2.5, 3.0) == S.kv(2.5, 3.0)


def test_airy_result():
    r = S.airy(1.5)
    assert type(r).__name__ == "AiryResult" and r._fields == ("ai", "aip", "bi", "bip")
    assert all(isinstance(v, float) for v in r)
    want = sp.airy(1.5)
    assert all(abs(a - b) <= 1e-14 * abs(b) for a, b in zip(r, want))
    a = S.airy(sd.asarray(np.array([-3.0, 0.0, 2.0])))
    assert np.asarray(a.bi).shape == (3,)


def test_airy_special_values():
    assert S.airy(0.0) == sp.airy(0.0)
    assert tuple(S.airy(np.inf)) == (0.0, -0.0, np.inf, np.inf)  # scipy NaN
    ai, aip, bi, bip = S.airy(-np.inf)
    assert ai == 0.0 and bi == 0.0 and np.isnan(aip) and np.isnan(bip)
    ai, aip, bi, bip = S.airy(105.0)  # scipy 0, 0, NaN, NaN
    assert 0 < ai < 1e-310 and aip < 0 and bi == np.inf and bip == np.inf
    assert all(np.isnan(v) for v in S.airy(np.nan))
