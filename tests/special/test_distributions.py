"""T1: the distribution functions (docs/api/special.md §13).

Accuracy against the Arb references is test_reference.py's job (tests/special/ref/*.npz);
here: the special values (true values and limits, T1 decision), agreement
with scipy where scipy is right, and regressions found while building T1.
"""

import math
import warnings

import numpy as np
import pytest
import scipy.special as sc

import stardust as sd

S = sd.special
INF, NAN = math.inf, math.nan


def eq(got, want):
    if isinstance(want, float) and math.isnan(want):
        assert math.isnan(got), got
    else:
        assert got == want, (got, want)


@pytest.mark.parametrize("call,want", [
    (lambda: S.stdtr(INF, 1.0), sc.ndtr(1.0)),
    (lambda: S.stdtr(2.0, INF), 1.0), (lambda: S.stdtr(2.0, -INF), 0.0), (lambda: S.stdtr(2.0, 0.0), 0.5),
    (lambda: S.stdtr(0.0, 1.0), NAN), (lambda: S.stdtr(-1.0, 1.0), NAN), (lambda: S.stdtr(NAN, 1.0), NAN),
    (lambda: S.stdtrit(2.0, 0.0), -INF), (lambda: S.stdtrit(2.0, 1.0), INF), (lambda: S.stdtrit(2.0, 0.5), 0.0),
    (lambda: S.stdtrit(2.0, 1.5), NAN), (lambda: S.stdtrit(INF, 0.975), S.ndtri(0.975)),
    (lambda: S.chdtr(0.0, 1.0), 1.0), (lambda: S.chdtr(0.0, 0.0), NAN), (lambda: S.chdtr(2.0, -1.0), NAN),
    (lambda: S.chdtrc(2.0, INF), 0.0), (lambda: S.chdtri(2.0, 1.0), 0.0), (lambda: S.chdtri(2.0, 0.0), INF),
    (lambda: S.fdtr(2.0, 3.0, 0.0), 0.0), (lambda: S.fdtrc(2.0, 3.0, INF), 0.0), (lambda: S.fdtr(0.0, 3.0, 1.0), NAN),
    (lambda: S.fdtr(INF, INF, 0.5), 0.0), (lambda: S.fdtr(INF, INF, 1.0), 1.0), (lambda: S.fdtrc(INF, INF, 2.0), 0.0),
    (lambda: S.fdtr(2.0, INF, 1.5), S.gammainc(1.0, 1.5)), (lambda: S.fdtr(INF, 4.0, 2.0), S.gammaincc(2.0, 1.0)),
    (lambda: S.fdtri(2.0, INF, 0.3), 2.0 * S.gammaincinv(1.0, 0.3) / 2.0),
    (lambda: S.fdtri(INF, 4.0, 0.3), 4.0 / (2.0 * S.gammainccinv(2.0, 0.3))), (lambda: S.fdtri(INF, INF, 0.3), 1.0),
    (lambda: S.pdtr(-1.0, 1.0), 0.0), (lambda: S.pdtrc(-0.5, 1.0), 1.0), (lambda: S.pdtr(2.7, 1.0), S.pdtr(2.0, 1.0)),
    (lambda: S.pdtr(2.0, -1.0), NAN), (lambda: S.pdtr(INF, 1.0), 1.0),
    (lambda: S.bdtr(-1.0, 5.0, 0.3), 0.0), (lambda: S.bdtrc(-1.0, 5.0, 0.3), 1.0),
    (lambda: S.bdtr(6.0, 5.0, 0.3), 1.0), (lambda: S.bdtrc(5.0, 5.0, 0.3), 0.0),
    (lambda: S.bdtr(2.0, 5.5, 0.3), NAN), (lambda: S.bdtr(2.0, INF, 0.3), NAN), (lambda: S.bdtr(2.0, 5.0, 1.5), NAN),
    (lambda: S.bdtr(2.7, 5.0, 0.3), S.bdtr(2.0, 5.0, 0.3)),
    (lambda: S.nbdtr(2.0, 5.5, 0.3), S.betainc(5.5, 3.0, 0.3)), (lambda: S.nbdtr(2.0, 0.0, 0.3), 1.0),
    (lambda: S.nbdtr(2.0, INF, 0.3), 0.0), (lambda: S.nbdtr(-1.0, 5.0, 0.3), 0.0), (lambda: S.nbdtrc(-1.0, 5.0, 0.3), 1.0),
    (lambda: S.ndtri_exp(0.0), INF), (lambda: S.ndtri_exp(-INF), -INF), (lambda: S.ndtri_exp(1.0), NAN),
    (lambda: S.ndtri_exp(NAN), NAN),
])
def test_special_values(call, want):
    eq(call(), float(want))


def test_scipy_errors_fixed():
    # T1 decision: the true value where scipy gives inf or NaN.
    t = S.stdtrit(3.0, 5e-324)
    assert math.isfinite(t) and t < -1e100
    assert S.stdtr(3.0, t) > 0
    assert S.ndtri_exp(-1e300) == -1.4142135623730951e150  # the asymptote, as scipy


RNG = np.random.default_rng(0)
N = 4000


def close_to_scipy(name, *args, rtol=1e-9):
    got = np.asarray(getattr(S, name)(*(sd.asarray(a) for a in args)))
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")  # scipy's deprecation of float n
        want = getattr(sc, name)(*args)
    ok = np.isfinite(want) & (want != 0)
    np.testing.assert_allclose(got[ok], want[ok], rtol=rtol, atol=0)
    assert np.array_equal(np.isnan(got), np.isnan(want))


def test_against_scipy_where_both_are_defined():
    df = np.exp(RNG.uniform(-3, 8, N))
    t = RNG.standard_normal(N) * np.exp(RNG.uniform(-3, 4, N))
    p = RNG.uniform(0, 1, N)
    close_to_scipy("stdtr", df, t)
    close_to_scipy("stdtrit", df, p)
    v, x = np.exp(RNG.uniform(-3, 6, N)), np.exp(RNG.uniform(-4, 7, N))
    for f in ("chdtr", "chdtrc"):
        close_to_scipy(f, v, x)
    close_to_scipy("chdtri", v, p)
    d1, d2, fx = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-3, 3, N))
    for f in ("fdtr", "fdtrc"):
        close_to_scipy(f, d1, d2, fx)
    close_to_scipy("fdtri", d1, d2, p, rtol=1e-8)
    k, m = np.floor(np.exp(RNG.uniform(0, 6, N))), np.exp(RNG.uniform(-2, 6, N))
    for f in ("pdtr", "pdtrc"):
        close_to_scipy(f, k, m)
    n, pp = k + np.floor(np.exp(RNG.uniform(0, 5, N))), RNG.uniform(0, 1, N)
    for f in ("bdtr", "bdtrc", "nbdtr", "nbdtrc"):
        close_to_scipy(f, k, n, pp)
    close_to_scipy("ndtri_exp", -np.exp(RNG.uniform(-10, 8, N)))


def test_betaincinv_small_a_and_b_regression():
    # T1 found betaincinv / betainccinv returning 0 or 1 for 8% of a, b < 1 (a bad
    # starting point read as an underflowing root). Now every root agrees with scipy.
    a, b, p = np.exp(RNG.uniform(-7, 0, 20000)), np.exp(RNG.uniform(-7, 0, 20000)), RNG.uniform(0, 1, 20000)
    for name in ("betaincinv", "betainccinv"):
        got = np.asarray(getattr(S, name)(sd.asarray(a), sd.asarray(b), sd.asarray(p)))
        want = getattr(sc, name)(a, b, p)
        ok = (want > 1e-290) & (want < 1 - 1e-12)
        np.testing.assert_allclose(got[ok], want[ok], rtol=1e-9)
    assert S.betaincinv(0.0379591, 0.00581817, 0.0879665) > 0


def test_python_scalars_give_floats_and_arrays_broadcast():
    assert type(S.bdtr(2, 5, 0.3)) is float
    got = S.pdtr(sd.asarray([0.0, 1.0, 2.0]), 1.5)
    assert np.allclose(np.asarray(got), sc.pdtr([0, 1, 2], 1.5), rtol=1e-14)


def test_inverses_beyond_the_double_range():
    # Roots outside float64: fdtri underflows to 0 only when even the smallest
    # subnormal already has fdtr >= p; stdtrit overflows to -inf only when even
    # -DBL_MAX still has stdtr >= p (T1: these points have no reference value).
    rng = np.random.default_rng(3)
    dfn, dfd, p = np.exp(rng.uniform(-4.6, 0, 3000)), np.exp(rng.uniform(-4.6, 9, 3000)), np.exp(rng.uniform(-230, -1, 3000))
    x = np.asarray(S.fdtri(sd.asarray(dfn), sd.asarray(dfd), sd.asarray(p)))
    zero = x == 0
    assert zero.any()
    assert np.all(np.asarray(S.fdtr(sd.asarray(dfn[zero]), sd.asarray(dfd[zero]), 5e-324)) >= p[zero])
    df, q = np.exp(rng.uniform(-4.6, 0.5, 3000)), np.exp(rng.uniform(-690, -1, 3000))
    t = np.asarray(S.stdtrit(sd.asarray(df), sd.asarray(q)))
    over = np.isinf(t)
    assert over.any() and np.all(t[over] < 0)
    assert np.all(np.asarray(S.stdtr(sd.asarray(df[over]), -1.7976931348623157e308)) >= q[over])
