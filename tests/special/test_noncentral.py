"""T2: the noncentral distributions (docs/api/special.md §14).

Accuracy against mpmath is test_reference.py's job (tests/special/ref/*.npz);
here: the special values (true values and limits, T1/T2 decision), the central
functions at nc = 0, agreement with scipy where scipy is right, the inverses'
round trips and their roots beyond the double range, and regressions found while
building T2. x comes last in every signature (T2 decision; scipy's chndtr takes
x first).
"""

import math
import warnings

import numpy as np
import pytest
import scipy.special as sc
import scipy.stats as st

import stardust as sd

S = sd.special
INF, NAN = math.inf, math.nan


def eq(got, want):
    if isinstance(want, float) and math.isnan(want):
        assert math.isnan(got), got
    else:
        assert got == want, (got, want)


@pytest.mark.parametrize("call,want", [
    # nc = 0: the central function, bit for bit
    (lambda: S.chndtr(2.5, 0.0, 3.0), S.chdtr(2.5, 3.0)), (lambda: S.chndtrc(2.5, 0.0, 3.0), S.chdtrc(2.5, 3.0)),
    (lambda: S.chndtrix(2.5, 0.0, 0.3), 2.0 * S.gammaincinv(1.25, 0.3)),
    (lambda: S.nctdtr(3.0, 0.0, 1.5), S.stdtr(3.0, 1.5)), (lambda: S.nctdtrc(3.0, 0.0, 1.5), S.stdtr(3.0, -1.5)),
    (lambda: S.nctdtrit(3.0, 0.0, 0.8), S.stdtrit(3.0, 0.8)),
    (lambda: S.ncfdtr(2.0, 3.0, 0.0, 1.5), S.fdtr(2.0, 3.0, 1.5)), (lambda: S.ncfdtrc(2.0, 3.0, 0.0, 1.5), S.fdtrc(2.0, 3.0, 1.5)),
    (lambda: S.ncfdtri(2.0, 3.0, 0.0, 0.4), S.fdtri(2.0, 3.0, 0.4)),
    # chi-square: domain ends, infinite parameters (the distribution runs off to infinity)
    (lambda: S.chndtr(2.0, 3.0, 0.0), 0.0), (lambda: S.chndtrc(2.0, 3.0, 0.0), 1.0),
    (lambda: S.chndtr(2.0, 3.0, INF), 1.0), (lambda: S.chndtrc(2.0, 3.0, INF), 0.0),
    (lambda: S.chndtr(INF, 3.0, 2.0), 0.0), (lambda: S.chndtr(2.0, INF, 2.0), 0.0), (lambda: S.chndtrc(2.0, INF, 2.0), 1.0),
    (lambda: S.chndtr(2.0, INF, INF), NAN), (lambda: S.chndtr(0.0, 3.0, 0.0), NAN),  # as chdtr(0, 0)
    (lambda: S.chndtr(-1.0, 3.0, 2.0), NAN), (lambda: S.chndtr(2.0, -1.0, 2.0), NAN), (lambda: S.chndtr(2.0, 3.0, -1.0), NAN),
    (lambda: S.chndtr(NAN, 3.0, 2.0), NAN),
    (lambda: S.chndtrix(2.0, 3.0, 0.0), 0.0), (lambda: S.chndtrix(2.0, 3.0, 1.0), INF), (lambda: S.chndtrix(2.0, 3.0, 1.5), NAN),
    (lambda: S.chndtrix(0.0, 3.0, 0.1), 0.0),  # inside the point mass e^-1.5 at 0
    (lambda: S.chndtrix(2.0, INF, 0.3), INF),
    # t: df = inf is the shifted normal; nc = +-inf
    (lambda: S.nctdtr(INF, 1.0, 0.5), S.ndtr(-0.5)), (lambda: S.nctdtrc(INF, 1.0, 0.5), S.ndtr(0.5)),
    (lambda: S.nctdtr(3.0, INF, 1.0), 0.0), (lambda: S.nctdtr(3.0, -INF, 1.0), 1.0),
    (lambda: S.nctdtr(3.0, 1.0, INF), 1.0), (lambda: S.nctdtr(3.0, 1.0, -INF), 0.0),
    (lambda: S.nctdtr(3.0, INF, INF), NAN), (lambda: S.nctdtr(3.0, INF, -INF), 0.0),
    (lambda: S.nctdtr(3.0, 2.0, 0.0), S.ndtr(-2.0)), (lambda: S.nctdtrc(3.0, 2.0, 0.0), S.ndtr(2.0)),
    (lambda: S.nctdtr(0.0, 1.0, 1.0), NAN), (lambda: S.nctdtr(-1.0, 1.0, 1.0), NAN),
    (lambda: S.nctdtrit(3.0, 2.0, 0.0), -INF), (lambda: S.nctdtrit(3.0, 2.0, 1.0), INF),
    (lambda: S.nctdtrit(INF, 2.0, 0.3), 2.0 + S.ndtri(0.3)), (lambda: S.nctdtrit(3.0, INF, 0.3), INF),
    (lambda: S.nctdtrit(3.0, -INF, 0.3), -INF),
    # F: dfd = inf is the noncentral chi-square of dfn F; dfn = inf the central F's limit
    (lambda: S.ncfdtr(2.0, INF, 1.0, 1.5), S.chndtr(2.0, 1.0, 3.0)), (lambda: S.ncfdtr(INF, 3.0, 1.0, 1.5), S.fdtr(INF, 3.0, 1.5)),
    (lambda: S.ncfdtr(2.0, 3.0, INF, 1.0), 0.0), (lambda: S.ncfdtrc(2.0, 3.0, INF, 1.0), 1.0),
    (lambda: S.ncfdtr(2.0, 3.0, 1.0, 0.0), 0.0), (lambda: S.ncfdtr(2.0, 3.0, 1.0, INF), 1.0),
    (lambda: S.ncfdtr(2.0, 3.0, INF, INF), NAN), (lambda: S.ncfdtr(0.0, 3.0, 1.0, 1.0), NAN),
    (lambda: S.ncfdtri(2.0, 3.0, 1.0, 0.0), 0.0), (lambda: S.ncfdtri(2.0, 3.0, 1.0, 1.0), INF),
    (lambda: S.ncfdtri(2.0, 3.0, INF, 0.3), INF),
])
def test_special_values(call, want):
    eq(call(), float(want))


def test_df_zero_is_a_point_mass_plus_the_rest():
    # chi-square(0, nc): mass e^(-nc/2) at 0; chndtr(0, 3, 2) from mpmath
    assert abs(S.chndtr(0.0, 3.0, 2.0) / 0.51205431668379600 - 1) < 1e-15
    assert S.chndtr(0.0, 3.0, 1e-300) > math.exp(-1.5) * (1 - 1e-15)


RNG = np.random.default_rng(0)
N = 3000


def call(name, *args):
    return np.asarray(getattr(S, name)(*(sd.asarray(a) for a in args)))


def scipy_quiet(f, *args):
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args)


def close(got, want, rtol):
    ok = np.isfinite(want) & (want > 1e-300)
    np.testing.assert_allclose(got[ok], want[ok], rtol=rtol, atol=0)


def test_against_scipy_where_scipy_is_right():
    # scipy (Boost underneath) holds ~1e-12 away from the far tails
    df, nc = np.exp(RNG.uniform(-1, 4, N)), np.exp(RNG.uniform(-2, 4, N))
    x = (df + nc) * np.exp(RNG.uniform(-0.7, 0.7, N))
    close(call("chndtr", df, nc, x), scipy_quiet(sc.chndtr, x, df, nc), 1e-11)
    close(call("chndtrc", df, nc, x), scipy_quiet(st.ncx2.sf, x, df, nc), 1e-11)
    tdf, tnc = np.exp(RNG.uniform(0, 4, N)), RNG.normal(0, 3, N)
    t = tnc + RNG.normal(0, 1, N)
    close(call("nctdtr", tdf, tnc, t), scipy_quiet(sc.nctdtr, tdf, tnc, t), 1e-10)
    d1, d2, fnc = np.exp(RNG.uniform(-1, 4, N)), np.exp(RNG.uniform(1, 4, N)), np.exp(RNG.uniform(-2, 4, N))
    f = (d1 + fnc) / d1 * np.exp(RNG.uniform(-0.7, 0.7, N))
    close(call("ncfdtr", d1, d2, fnc, f), scipy_quiet(sc.ncfdtr, d1, d2, fnc, f), 1e-10)


def test_the_two_tails_add_to_one():
    df, nc = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-3, 6, N))
    x = (df + nc) * np.exp(RNG.uniform(-2, 2, N))
    assert np.all(np.abs(call("chndtr", df, nc, x) + call("chndtrc", df, nc, x) - 1) < 1e-14)  # each tail within 1e-14 (the contract)
    tdf, tnc = np.exp(RNG.uniform(-2, 6, N)), RNG.normal(0, 10, N)
    t = tnc * np.exp(RNG.uniform(-1, 1, N)) + RNG.normal(0, 2, N)
    assert np.all(np.abs(call("nctdtr", tdf, tnc, t) + call("nctdtrc", tdf, tnc, t) - 1) < 1e-14)  # each tail within 1e-14 (the contract)
    d1, d2 = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-2, 6, N))
    f = np.exp(RNG.uniform(-3, 3, N))
    assert np.all(np.abs(call("ncfdtr", d1, d2, nc, f) + call("ncfdtrc", d1, d2, nc, f) - 1) < 1e-14)  # each tail within 1e-14 (the contract)


def test_inverses_round_trip():
    p = RNG.uniform(0, 1, N)
    df, nc = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-3, 6, N))
    np.testing.assert_allclose(call("chndtr", df, nc, call("chndtrix", df, nc, p)), p, rtol=1e-12)
    tdf, tnc = np.exp(RNG.uniform(-1, 6, N)), RNG.normal(0, 10, N)
    np.testing.assert_allclose(call("nctdtr", tdf, tnc, call("nctdtrit", tdf, tnc, p)), p, rtol=1e-12)
    d1, d2 = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-2, 6, N))
    np.testing.assert_allclose(call("ncfdtr", d1, d2, nc, call("ncfdtri", d1, d2, nc, p)), p, rtol=1e-12)


def test_inverses_beyond_the_double_range():
    # A root returned as 0 lies within the subnormal bound: F(45 * 2^-1074) >= p
    # (these points have no reference value, T1 and T2).
    rng = np.random.default_rng(3)
    df, nc, p = np.exp(rng.uniform(-4.6, 1, 3000)), np.exp(rng.uniform(-7, 2, 3000)), np.exp(rng.uniform(-690, -1, 3000))
    x = call("chndtrix", df, nc, p)
    zero = x == 0
    assert zero.any()
    assert np.all(call("chndtr", df[zero], nc[zero], np.full(zero.sum(), 2e-322)) >= p[zero])  # within 45 subnormal units
    d2 = np.exp(rng.uniform(-4.6, 4, 3000))
    x = call("ncfdtri", df, d2, nc, p)
    zero = x == 0
    assert zero.any()
    assert np.all(call("ncfdtr", df[zero], d2[zero], nc[zero], np.full(zero.sum(), 2e-322)) >= p[zero])  # within 45 subnormal units


def test_the_tail_away_from_the_noncentrality():
    # P(T > t) with nc < 0 < t: the series' two families cancel there (scipy: 1.6e-33 for
    # 1.4e-76, 0 below); an integral instead. Values from mpmath (both methods agree).
    assert abs(S.nctdtrc(0.1, -5.0, 100.0) / 1.2882566998160138e-07 - 1) < 1e-15
    assert abs(S.nctdtrc(12.35, -16.86, 4.509) / 1.4425759625117107e-76 - 1) < 1e-15
    assert abs(S.nctdtr(5.0, 3.0, -2.0) / 7.7996360918320379e-06 - 1) < 1e-15  # the mirror: T(nc) = -T(-nc)


def test_fdtr_with_x_near_one_from_a_division_regression():
    # T2 found fdtr(3218.14, 0.0129237, 1e20) off by 1.5e-9: w = 1 - 4e-26 from a division
    # carries only 1e-32 absolutely, and bgrat took log w from it (now from 1 - w). mpmath:
    assert abs(S.fdtr(3218.14, 0.0129237, 1e20) / 0.27853048221645325 - 1) < 1e-15
    assert abs(S.ncfdtr(3218.14, 0.0129237, 10.4689, 1e20) / 0.27851534065985444 - 1) < 1e-15


def test_x_last_and_python_scalars():
    assert S.chndtr(4.0, 3.0, 2.0) == float(sc.chndtr(2.0, 4.0, 3.0)) or abs(S.chndtr(4.0, 3.0, 2.0) / sc.chndtr(2.0, 4.0, 3.0) - 1) < 1e-13
    assert type(S.ncfdtr(2, 3, 1, 1)) is float


def test_df_zero_at_subnormal_x_review_regression():
    # x/2 underflowed to 0 and chdtr(0, x) became gammainc(0, 0) = NaN (review of T1/T2);
    # chdtr(0, x > 0) is 1, and the noncentral one keeps its point mass e^(-nc/2).
    assert S.chdtr(0.0, 5e-324) == 1.0 and S.chdtrc(0.0, 5e-324) == 0.0
    assert abs(S.chndtr(0.0, 3.0, 5e-324) / math.exp(-1.5) - 1) < 1e-15
    assert abs(S.chndtrc(0.0, 3.0, 5e-324) / (1 - math.exp(-1.5)) - 1) < 1e-15
    assert S.chndtrix(0.0, 3.0, 0.3) > 0  # above the point mass: a positive root
