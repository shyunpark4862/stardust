"""The noncentral distributions (docs/api/special.md §14).

Accuracy against the Arb references is test_reference.py's job (tests/special/ref/*.npz);
here: the special values (true values and limits, docs/design/special.md §2.4), the central
functions at nc = 0, agreement with scipy where scipy is right, the inverses'
round trips and their roots beyond the double range, and regressions found while
measuring them. x comes last in every signature (docs/design/special.md §2.4; scipy's chndtr takes
x first).
"""

import math
import warnings

import numpy as np
import pytest
import scipy.special as sc
import scipy.stats as st
from flint import arb, ctx

import stardust as sd
from tests._bits import bits_equal

S = sd.special
INF, NAN = math.inf, math.nan


def eq(got, want):
    if isinstance(want, float) and math.isnan(want):
        assert math.isnan(got), got
    else:
        assert got == want, (got, want)


@pytest.mark.parametrize(
    "call,want",
    [
        # nc = 0: the central function, bit for bit
        (lambda: S.chndtr(2.5, 0.0, 3.0), S.chdtr(2.5, 3.0)),
        (lambda: S.chndtrc(2.5, 0.0, 3.0), S.chdtrc(2.5, 3.0)),
        (lambda: S.chndtrix(2.5, 0.0, 0.3), 2.0 * S.gammaincinv(1.25, 0.3)),
        (lambda: S.nctdtr(3.0, 0.0, 1.5), S.stdtr(3.0, 1.5)),
        (lambda: S.nctdtrc(3.0, 0.0, 1.5), S.stdtr(3.0, -1.5)),
        (lambda: S.nctdtrit(3.0, 0.0, 0.8), S.stdtrit(3.0, 0.8)),
        (lambda: S.ncfdtr(2.0, 3.0, 0.0, 1.5), S.fdtr(2.0, 3.0, 1.5)),
        (lambda: S.ncfdtrc(2.0, 3.0, 0.0, 1.5), S.fdtrc(2.0, 3.0, 1.5)),
        (lambda: S.ncfdtri(2.0, 3.0, 0.0, 0.4), S.fdtri(2.0, 3.0, 0.4)),
        # chi-square: domain ends, infinite parameters (the distribution runs off to infinity)
        (lambda: S.chndtr(2.0, 3.0, 0.0), 0.0),
        (lambda: S.chndtrc(2.0, 3.0, 0.0), 1.0),
        (lambda: S.chndtr(2.0, 3.0, INF), 1.0),
        (lambda: S.chndtrc(2.0, 3.0, INF), 0.0),
        (lambda: S.chndtr(INF, 3.0, 2.0), 0.0),
        (lambda: S.chndtr(2.0, INF, 2.0), 0.0),
        (lambda: S.chndtrc(2.0, INF, 2.0), 1.0),
        (lambda: S.chndtr(2.0, INF, INF), NAN),
        (lambda: S.chndtr(0.0, 3.0, 0.0), NAN),  # as chdtr(0, 0)
        (lambda: S.chndtr(-1.0, 3.0, 2.0), NAN),
        (lambda: S.chndtr(2.0, -1.0, 2.0), NAN),
        (lambda: S.chndtr(2.0, 3.0, -1.0), NAN),
        (lambda: S.chndtr(NAN, 3.0, 2.0), NAN),
        (lambda: S.chndtrix(2.0, 3.0, 0.0), 0.0),
        (lambda: S.chndtrix(2.0, 3.0, 1.0), INF),
        (lambda: S.chndtrix(2.0, 3.0, 1.5), NAN),
        (lambda: S.chndtrix(0.0, 3.0, 0.1), 0.0),  # inside the point mass e^-1.5 at 0
        (lambda: S.chndtrix(2.0, INF, 0.3), INF),
        # t: df = inf is the shifted normal; nc = +-inf
        (lambda: S.nctdtr(INF, 1.0, 0.5), S.ndtr(-0.5)),
        (lambda: S.nctdtrc(INF, 1.0, 0.5), S.ndtr(0.5)),
        (lambda: S.nctdtr(3.0, INF, 1.0), 0.0),
        (lambda: S.nctdtr(3.0, -INF, 1.0), 1.0),
        (lambda: S.nctdtr(3.0, 1.0, INF), 1.0),
        (lambda: S.nctdtr(3.0, 1.0, -INF), 0.0),
        (lambda: S.nctdtr(3.0, INF, INF), NAN),
        (lambda: S.nctdtr(3.0, INF, -INF), 0.0),
        (lambda: S.nctdtr(3.0, 2.0, 0.0), S.ndtr(-2.0)),
        (lambda: S.nctdtrc(3.0, 2.0, 0.0), S.ndtr(2.0)),
        (lambda: S.nctdtr(0.0, 1.0, 1.0), NAN),
        (lambda: S.nctdtr(-1.0, 1.0, 1.0), NAN),
        (lambda: S.nctdtrit(3.0, 2.0, 0.0), -INF),
        (lambda: S.nctdtrit(3.0, 2.0, 1.0), INF),
        (lambda: S.nctdtrit(INF, 2.0, 0.3), 2.0 + S.ndtri(0.3)),
        (lambda: S.nctdtrit(3.0, INF, 0.3), INF),
        (lambda: S.nctdtrit(3.0, -INF, 0.3), -INF),
        # F: dfd = inf is the noncentral chi-square of dfn F; dfn = inf the central F's limit
        (lambda: S.ncfdtr(2.0, INF, 1.0, 1.5), S.chndtr(2.0, 1.0, 3.0)),
        (lambda: S.ncfdtr(INF, 3.0, 1.0, 1.5), S.fdtr(INF, 3.0, 1.5)),
        (lambda: S.ncfdtr(2.0, 3.0, INF, 1.0), 0.0),
        (lambda: S.ncfdtrc(2.0, 3.0, INF, 1.0), 1.0),
        (lambda: S.ncfdtr(2.0, 3.0, 1.0, 0.0), 0.0),
        (lambda: S.ncfdtr(2.0, 3.0, 1.0, INF), 1.0),
        (lambda: S.ncfdtr(2.0, 3.0, INF, INF), NAN),
        (lambda: S.ncfdtr(0.0, 3.0, 1.0, 1.0), NAN),
        (lambda: S.ncfdtri(2.0, 3.0, 1.0, 0.0), 0.0),
        (lambda: S.ncfdtri(2.0, 3.0, 1.0, 1.0), INF),
        (lambda: S.ncfdtri(2.0, 3.0, INF, 0.3), INF),
    ],
)
def test_special_values(call, want):
    eq(call(), float(want))


def test_df_zero_is_a_point_mass_plus_the_rest():
    # chi-square(0, nc): mass e^(-nc/2) at 0; chndtr(0, 3, 2) from Arb (tools/special_ref.py)
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
    assert np.all(
        np.abs(call("chndtr", df, nc, x) + call("chndtrc", df, nc, x) - 1) < 1e-14
    )  # each tail within 1e-14 (the contract)
    tdf, tnc = np.exp(RNG.uniform(-2, 6, N)), RNG.normal(0, 10, N)
    t = tnc * np.exp(RNG.uniform(-1, 1, N)) + RNG.normal(0, 2, N)
    assert np.all(
        np.abs(call("nctdtr", tdf, tnc, t) + call("nctdtrc", tdf, tnc, t) - 1) < 1e-14
    )  # each tail within 1e-14 (the contract)
    d1, d2 = np.exp(RNG.uniform(-2, 6, N)), np.exp(RNG.uniform(-2, 6, N))
    f = np.exp(RNG.uniform(-3, 3, N))
    assert np.all(
        np.abs(call("ncfdtr", d1, d2, nc, f) + call("ncfdtrc", d1, d2, nc, f) - 1) < 1e-14
    )  # each tail within 1e-14 (the contract)


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
    # (these points have no reference value).
    rng = np.random.default_rng(3)
    df, nc, p = (
        np.exp(rng.uniform(-4.6, 1, 3000)),
        np.exp(rng.uniform(-7, 2, 3000)),
        np.exp(rng.uniform(-690, -1, 3000)),
    )
    x = call("chndtrix", df, nc, p)
    zero = x == 0
    assert zero.any()
    assert np.all(
        call("chndtr", df[zero], nc[zero], np.full(zero.sum(), 2e-322)) >= p[zero]
    )  # within 45 subnormal units
    d2 = np.exp(rng.uniform(-4.6, 4, 3000))
    x = call("ncfdtri", df, d2, nc, p)
    zero = x == 0
    assert zero.any()
    assert np.all(
        call("ncfdtr", df[zero], d2[zero], nc[zero], np.full(zero.sum(), 2e-322)) >= p[zero]
    )  # within 45 subnormal units


def test_the_tail_away_from_the_noncentrality():
    # P(T > t) with nc < 0 < t: the series' two families cancel there (scipy: 1.6e-33 for
    # 1.4e-76, 0 below); an integral instead. Values from Arb (tools/special_ref.py).
    assert abs(S.nctdtrc(0.1, -5.0, 100.0) / 1.2882566998160138e-07 - 1) < 1e-15
    assert abs(S.nctdtrc(12.35, -16.86, 4.509) / 1.4425759625117107e-76 - 1) < 1e-15
    assert abs(S.nctdtr(5.0, 3.0, -2.0) / 7.7996360918320379e-06 - 1) < 1e-15  # the mirror: T(nc) = -T(-nc)


def test_fdtr_with_x_near_one_from_a_division_regression():
    # fdtr(3218.14, 0.0129237, 1e20) was off by 1.5e-9: w = 1 - 4e-26 from a division
    # carries only 1e-32 absolutely, and bgrat took log w from it (now from 1 - w). Arb:
    assert abs(S.fdtr(3218.14, 0.0129237, 1e20) / 0.27853048221645325 - 1) < 1e-15
    assert abs(S.ncfdtr(3218.14, 0.0129237, 10.4689, 1e20) / 0.27851534065985444 - 1) < 1e-15


def test_x_last_and_python_scalars():
    assert (
        S.chndtr(4.0, 3.0, 2.0) == float(sc.chndtr(2.0, 4.0, 3.0))
        or abs(S.chndtr(4.0, 3.0, 2.0) / sc.chndtr(2.0, 4.0, 3.0) - 1) < 1e-13
    )
    assert type(S.ncfdtr(2, 3, 1, 1)) is float


def test_df_zero_at_subnormal_x_review_regression():
    # x/2 underflowed to 0 and chdtr(0, x) became gammainc(0, 0) = NaN (a code review found it);
    # chdtr(0, x > 0) is 1, and the noncentral one keeps its point mass e^(-nc/2).
    assert S.chdtr(0.0, 5e-324) == 1.0 and S.chdtrc(0.0, 5e-324) == 0.0
    assert abs(S.chndtr(0.0, 3.0, 5e-324) / math.exp(-1.5) - 1) < 1e-15
    assert abs(S.chndtrc(0.0, 3.0, 5e-324) / (1 - math.exp(-1.5)) - 1) < 1e-15
    assert S.chndtrix(0.0, 3.0, 0.3) > 0  # above the point mass: a positive root


def test_tiny_noncentrality_and_tiny_argument_review_regression():
    # The far-end walk's frames: a step of 1 / mu (mu = nc / 2 near 1e-200) on top of one of
    # (s + 1) / y overflowed the scaled G, the sum became NaN and chndtr returned 1.0 for any
    # nc below about 1e-155 at a tiny x; a subnormal mu made 1 / mu infinite. Values from Arb
    # (the Poisson mixture's first terms at 700 bits).
    for df, nc, x, want in [
        (2.0, 1e-200, 1e-160, 5e-161),
        (0.5, 1e-200, 1e-160, 9.277296085790009e-41),
        (3.5, 1e-320, 1e-160, 1.8484784848877647e-281),
        (1.0, 1e-155, 1e-160, 7.978845608028654e-81),
    ]:
        assert abs(S.chndtr(df, nc, x) / want - 1) < 2e-15, (df, nc, x)
        assert S.chndtrc(df, nc, x) == 1.0
    assert abs(S.chndtr(0.0, 1e-200, 1e-160) - 1.0) <= 2.3e-16  # the point mass e^(-nc/2)
    assert abs(S.chndtrc(0.0, 1e-200, 1e-160) / 5e-201 - 1) < 2e-15
    assert abs(S.ncfdtr(2.0, 3.0, 1e-200, 1e-160) / 1e-160 - 1) < 2e-15
    assert abs(S.ncfdtr(0.5, 40.0, 1e-300, 1e-200) / 7.764613505692547e-51 - 1) < 2e-15
    assert S.ncfdtrc(2.0, 3.0, 1e-200, 1e-160) == 1.0
    assert S.nctdtr(3.0, 1e-200, 1e-160) == 0.5 and S.nctdtrc(3.0, 1e-200, 1e-160) == 0.5
    assert abs(S.nctdtr(3.0, -1e-170, 0.5) / 0.6742760175759245 - 1) < 2e-15
    # nc^2 / 2 below the doubles on the side away from the noncentrality (these were NaN)
    assert (
        abs(S.nctdtrc(0.2632616250534453, -2.276129563279541e-158, -2426.8530463003194) / 0.9540086053194111 - 1)
        < 2e-15
    )
    assert (
        abs(S.nctdtrc(26.849635091450086, -5.963356455454383e-159, -0.5917525269167969) / 0.7205167625224951 - 1)
        < 2e-15
    )


def test_every_call_returns_and_an_unfinished_sum_is_nan():
    # Every loop has a fixed most count of steps. A sum that would need more is NaN, never the
    # part summed so far: chndtr(3.5, 1e15, x) was 2.28e-9 at every x (1e8 of its 4e8 terms,
    # 1.7 s), and nc = 1e100 did not return (a bisection between neighbouring doubles).
    for f, args in [
        (S.chndtr, (3.5, 1e15, 1e15)),
        (S.chndtrc, (3.5, 1e15, 1e15)),
        (S.chndtrix, (3.5, 1e15, 0.5)),
        (S.chndtrix, (3.5, 1e100, 0.01)),
        (S.nctdtr, (9.0, 1e100, 0.5)),
        (S.nctdtrc, (9.0, 1e100, 1e30)),
        (S.nctdtr, (9.0, -1e8, -50.0)),
        (S.nctdtrit, (9.0, -1e8, 0.5)),
        (S.nctdtrit, (9.0, 1e100, 0.99)),
        (S.ncfdtr, (4.0, 13.0, 1e100, 1.0)),
        (S.ncfdtrc, (4.0, 13.0, 1e15, 1e16)),
        (S.ncfdtri, (4.0, 13.0, 1e100, 0.5)),
    ]:
        assert math.isnan(f(*args)), (f.__name__, args)
    # a chi-square tail below e^-1000 by Chernoff's bound is 0 (its complement 1) before any sum
    assert S.chndtr(3.5, 1e100, 1.0) == 0.0 and S.chndtrc(3.5, 1e100, 1.0) == 1.0
    assert S.chndtrc(3.5, 1e100, 1e308) == 0.0 and S.chndtr(3.5, 1e100, 1e308) == 1.0
    # the side near 1 that runs out of terms is 1 minus the other side where that one is
    # summed (here 0: x far beyond the bulk)
    assert S.chndtr(3.5, 1e15, 1e16) == 1.0 and S.chndtrc(3.5, 1e15, 0.5) == 1.0
    # inside the guaranteed parameters every x has its value: a tail beyond every term is 0 or 1
    assert S.chndtrc(3.5, 2.0, 1e300) == 0.0 and S.chndtr(3.5, 2.0, 1e300) == 1.0
    assert S.chndtrc(1e4, 1e4, 1e30) == 0.0 and S.chndtr(1e4, 1e4, 1e30) == 1.0
    # the t's tail away from nc at |t| next to the largest double (a t + root passed the
    # doubles: NaN). Arb.
    assert abs(S.nctdtr(0.1, 1.5, -1e308) / 8.255481062550066e-33 - 1) < 2e-15
    assert abs(S.nctdtrc(0.1, -1.5, 1e308) / 8.255481062550066e-33 - 1) < 2e-15
    assert S.nctdtrc(0.1, 1.5, -1e308) == 1.0
    # arrays give the scalar's bits, NaN included
    v = np.asarray(S.chndtr(sd.asarray([3.5, 3.5]), sd.asarray([1e15, 2.0]), sd.asarray([1e15, 1e300])))
    assert math.isnan(v[0]) and v[1] == 1.0


def test_noncentral_chi_square_large_side_from_the_small():
    # the side near 1 ran out of terms from nc ~ 1.47e9; now 1 - the other side
    nc = 2.0e9
    s = math.sqrt(2.0 * (3.0 + 2.0 * nc))
    for k in (-6.0, -1.0, 1.0, 6.0):
        x = nc + 3.0 + k * s
        p, q = S.chndtr(3.0, nc, x), S.chndtrc(3.0, nc, x)
        assert math.isfinite(p) and math.isfinite(q)
        assert abs(p + q - 1.0) <= 4e-16
        assert p == pytest.approx(S.ndtr(k), abs=2e-4)  # the normal limit, to its skewness
    assert math.isfinite(S.chndtrix(3.0, 1.47e9, 0.5))


# ---- noncentral chi-square tails beyond e^-1000 ----


def test_chndtr_obvious_tails():
    assert bits_equal(S.chndtr(1.0, 1e16, 1.0), 0.0)
    assert S.chndtrc(1.0, 1e16, 1.0) == 1.0
    assert S.chndtr(1.0, 1e16, 1e17) == 1.0
    assert bits_equal(S.chndtrc(1.0, 1e16, 1e17), 0.0)


# ---- degrees of freedom below 2^-1021


def test_special_underneath():
    d = 2.0**-1022
    # the upper tail was NaN where df / 2 is subnormal; now the df = 0 values
    for x in (1e-300, 1e-10, 1.0):
        assert S.chndtrc(d, 1.0, x) == S.chndtrc(0.0, 1.0, x)
        assert S.chndtr(d, 1.0, x) == S.chndtr(0.0, 1.0, x)
    # df > 0 has no point mass: chndtr(df, nc, 0) stays 0 (the function, not the T44 distribution)
    assert S.chndtr(d, 1.0, 0.0) == 0.0


@pytest.mark.parametrize("df", [5e-324, 2.0**-1022, 2.0**-1021, 1e-307, 1e-305, 1e-292])
def test_chndtrc_tiny_df(df):
    # R77: NaN for df up to ~1e-305 at small x; now the df = 0 law plus the j = 0 term's tail
    for x in (1e-300, 1e-100, 1e-20):
        assert S.chndtrc(df, 1.0, x) == pytest.approx(-math.expm1(-0.5), rel=1e-15)
        assert S.chndtr(df, 1.0, x) == pytest.approx(math.exp(-0.5), rel=1e-12)
    # nc tiny too: both terms count, mu e^-y and e^-mu (df/2) E1(y)
    want = 5e-301 + 0.5 * df * S.expn(1, 5e-301)
    assert S.chndtrc(df, 1e-300, 1e-300) == pytest.approx(want, rel=1e-14, abs=45 * 2.0**-1074)
    X = sd.stats.NoncentralChiSquared(df=max(df, 2.0**-1021), nc=1.0)
    assert X.ccdf(1e-300) == pytest.approx(-math.expm1(-0.5), rel=1e-15)
    assert X.logccdf(1e-20) == pytest.approx(math.log(-math.expm1(-0.5)), rel=1e-14)


def test_ncfdtri_root_beyond_the_doubles():
    # R78: the tail at the largest double is still short of the target: inf (was NaN)
    assert S.ncfdtri(5e-324, 5.0, 2.0, 0.5) == math.inf and S.ncfdtri(1e-320, 5.0, 2.0, 0.5) == math.inf
    assert S.ncfdtri(5e-324, 5.0, 2.0, 0.2) == 0.0  # inside the j = 0 mass, below every double
    assert S.ncfdtri(2.0**-1021, 5.0, 2.0, 0.5) == pytest.approx(1.8609659255940713e307, rel=1e-12)


def test_ncfdtr_smallest_dfn():
    # R82: dfn = 5e-324 halves to 0; NaN for f ~ 1e23-1e173
    for f in (1e10, 1e23, 1e100, 1e173, 1e200):
        assert S.ncfdtr(5e-324, 5.0, 2.0, f) == pytest.approx(math.exp(-1.0), rel=1e-15)
        assert S.ncfdtrc(5e-324, 5.0, 2.0, f) == pytest.approx(-math.expm1(-1.0), rel=1e-15)
        assert S.ncfdtr(5e-324, 5.0, 2.0, f) == S.ncfdtr(1e-323, 5.0, 2.0, f)


@pytest.mark.parametrize(
    "df,nc,x",
    [(5e-324, 5e-324, 1e-300), (1.5e-323, 1e-300, 1e-300), (1e-310, 1e-310, 1e-20), (2.0**-1022, 1e-300, 1e-300)],
)
def test_chndtrc_subnormal_df_against_arb(df, nc, x):
    # R83: the j = 0 term's half after the scaling (0.5 * df rounded first: 0 for 1.71e-321)
    ctx.prec = 4000
    try:
        a, mu, y = arb(df) / 2, arb(nc) / 2, arb(x) / 2
        want, p = arb(0), (-mu).exp()
        for j in range(60):
            want += p * y.gamma_upper(a + j, 1)
            p = p * mu / (j + 1)
        want = float(want)
    finally:
        ctx.prec = 53
    assert abs(S.chndtrc(df, nc, x) - want) <= max(2.0**-1074, 1e-14 * want)
