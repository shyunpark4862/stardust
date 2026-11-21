"""The distribution functions (docs/api/special.md §13).

Accuracy against the Arb references is test_reference.py's job (tests/special/ref/*.npz);
here: the special values (true values and limits, docs/design/special.md §2.4), agreement
with scipy where scipy is right, and regressions found while measuring them.
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


@pytest.mark.parametrize(
    "call,want",
    [
        (lambda: S.stdtr(INF, 1.0), sc.ndtr(1.0)),
        (lambda: S.stdtr(2.0, INF), 1.0),
        (lambda: S.stdtr(2.0, -INF), 0.0),
        (lambda: S.stdtr(2.0, 0.0), 0.5),
        (lambda: S.stdtr(0.0, 1.0), NAN),
        (lambda: S.stdtr(-1.0, 1.0), NAN),
        (lambda: S.stdtr(NAN, 1.0), NAN),
        (lambda: S.stdtrit(2.0, 0.0), -INF),
        (lambda: S.stdtrit(2.0, 1.0), INF),
        (lambda: S.stdtrit(2.0, 0.5), 0.0),
        (lambda: S.stdtrit(2.0, 1.5), NAN),
        (lambda: S.stdtrit(INF, 0.975), S.ndtri(0.975)),
        (lambda: S.chdtr(0.0, 1.0), 1.0),
        (lambda: S.chdtr(0.0, 0.0), NAN),
        (lambda: S.chdtr(2.0, -1.0), NAN),
        (lambda: S.chdtrc(2.0, INF), 0.0),
        (lambda: S.chdtri(2.0, 1.0), 0.0),
        (lambda: S.chdtri(2.0, 0.0), INF),
        (lambda: S.fdtr(2.0, 3.0, 0.0), 0.0),
        (lambda: S.fdtrc(2.0, 3.0, INF), 0.0),
        (lambda: S.fdtr(0.0, 3.0, 1.0), NAN),
        (lambda: S.fdtr(INF, INF, 0.5), 0.0),
        (lambda: S.fdtr(INF, INF, 1.0), 1.0),
        (lambda: S.fdtrc(INF, INF, 2.0), 0.0),
        (lambda: S.fdtr(2.0, INF, 1.5), S.gammainc(1.0, 1.5)),
        (lambda: S.fdtr(INF, 4.0, 2.0), S.gammaincc(2.0, 1.0)),
        (lambda: S.fdtri(2.0, INF, 0.3), 2.0 * S.gammaincinv(1.0, 0.3) / 2.0),
        (lambda: S.fdtri(INF, 4.0, 0.3), 4.0 / (2.0 * S.gammainccinv(2.0, 0.3))),
        (lambda: S.fdtri(INF, INF, 0.3), 1.0),
        (lambda: S.pdtr(-1.0, 1.0), 0.0),
        (lambda: S.pdtrc(-0.5, 1.0), 1.0),
        (lambda: S.pdtr(2.7, 1.0), S.pdtr(2.0, 1.0)),
        (lambda: S.pdtr(2.0, -1.0), NAN),
        (lambda: S.pdtr(INF, 1.0), 1.0),
        (lambda: S.bdtr(-1.0, 5.0, 0.3), 0.0),
        (lambda: S.bdtrc(-1.0, 5.0, 0.3), 1.0),
        (lambda: S.bdtr(6.0, 5.0, 0.3), 1.0),
        (lambda: S.bdtrc(5.0, 5.0, 0.3), 0.0),
        (lambda: S.bdtr(2.0, 5.5, 0.3), NAN),
        (lambda: S.bdtr(2.0, INF, 0.3), NAN),
        (lambda: S.bdtr(2.0, 5.0, 1.5), NAN),
        (lambda: S.bdtr(2.7, 5.0, 0.3), S.bdtr(2.0, 5.0, 0.3)),
        (lambda: S.nbdtr(2.0, 5.5, 0.3), S.betainc(5.5, 3.0, 0.3)),
        (lambda: S.nbdtr(2.0, 0.0, 0.3), 1.0),
        (lambda: S.nbdtr(2.0, INF, 0.3), 0.0),
        (lambda: S.nbdtr(-1.0, 5.0, 0.3), 0.0),
        (lambda: S.nbdtrc(-1.0, 5.0, 0.3), 1.0),
        (lambda: S.ndtri_exp(0.0), INF),
        (lambda: S.ndtri_exp(-INF), -INF),
        (lambda: S.ndtri_exp(1.0), NAN),
        (lambda: S.ndtri_exp(NAN), NAN),
    ],
)
def test_special_values(call, want):
    eq(call(), float(want))


def test_scipy_errors_fixed():
    # docs/design/special.md §2.4: the true value where scipy gives inf or NaN.
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
    # betaincinv / betainccinv returned 0 or 1 for 8% of a, b < 1 (a bad
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
    # -DBL_MAX still has stdtr >= p (these points have no reference value).
    rng = np.random.default_rng(3)
    dfn, dfd, p = (
        np.exp(rng.uniform(-4.6, 0, 3000)),
        np.exp(rng.uniform(-4.6, 9, 3000)),
        np.exp(rng.uniform(-230, -1, 3000)),
    )
    x = np.asarray(S.fdtri(sd.asarray(dfn), sd.asarray(dfd), sd.asarray(p)))
    zero = x == 0
    assert zero.any()
    assert np.all(np.asarray(S.fdtr(sd.asarray(dfn[zero]), sd.asarray(dfd[zero]), 5e-324)) >= p[zero])
    df, q = np.exp(rng.uniform(-4.6, 0.5, 3000)), np.exp(rng.uniform(-690, -1, 3000))
    t = np.asarray(S.stdtrit(sd.asarray(df), sd.asarray(q)))
    over = np.isinf(t)
    assert over.any() and np.all(t[over] < 0)
    assert np.all(np.asarray(S.stdtr(sd.asarray(df[over]), -1.7976931348623157e308)) >= q[over])


def test_fdtr_where_one_shape_is_past_2_900_review_regression():
    # dfd (or dfn) so large that the beta argument is below 2^-1000 while its product with
    # the shape is not small: the chi-square limit. The leading term alone, used before, gave
    # fdtr(2, 1e308, 1) = 1 (true: 1 - 1/e) and NaN at dfn = 200. Values: the series in Arb.
    rows = [
        (2.0, 1e308, 1.0, 0.6321205588285577, 0.36787944117144233),
        (200.0, 1e308, 1.0, 0.5132987982791487, 0.48670120172085135),
        (2e-5, 1e300, 1e-320, 0.992550258327848, 0.007449741672151913),
        (0.5, 1e305, 40.0, 0.9999979169695914, 2.083030408649431e-06),
        (1e308, 2.0, 1.0, 0.36787944117144233, 0.6321205588285577),
        (1e300, 2e-5, 1e305, 0.007106867638463068, 0.992893132361537),
        (7.0, 1e300, 1e-05, 2.1806304891469145e-17, 1.0),
        (16.419704325584377, 7.37960052026974e277, 5.776808771711861, 0.9999999999995333, 4.667669003019092e-13),
    ]
    for dfn, dfd, x, lo, up in rows:
        assert abs(S.fdtr(dfn, dfd, x) / lo - 1) < 2e-15, (dfn, dfd, x)
        assert abs(S.fdtrc(dfn, dfd, x) / up - 1) < 2e-15, (dfn, dfd, x)


@pytest.mark.parametrize("y", [-8.98e307, -8.99e307, -1e308, -1.7976931348623157e308])
def test_ndtri_exp_huge_negative(y):
    v = S.ndtri_exp(y)
    want = -math.sqrt(2.0) * math.sqrt(-y)  # the asymptote's leading term, to ~log|y| / |y|
    assert v < 0 and math.isfinite(v)
    assert abs(v / want - 1.0) < 1e-15
    assert sd.stats.Normal().ilogcdf(y) == v
    assert sd.stats.Normal().ilogccdf(y) == -v
    assert sd.stats.Normal(mu=1.0, sigma=2.0).ilogcdf(y) == pytest.approx(1.0 + 2.0 * v, rel=1e-15)


def test_fdtri_root_past_the_doubles_with_a_vanishing_dfd():
    # dfd / dfn underflowed to 0 while (1 - v) / v was inf (v = 0): inf x 0 = NaN. The root is
    # past the doubles (fdtr at the largest double is below p), so inf.
    rows = [(2, 5e-324, 0.1), (3, 5e-324, 0.5), (10, 1e-323, 0.1), (1e5, 1e-323, 1e-300), (1e20, 1e-310, 0.1)]
    for dfn, dfd, p in rows:
        assert S.fdtri(dfn, dfd, p) == math.inf, (dfn, dfd, p)
        assert S.fdtr(dfn, dfd, 1.7976931348623157e308) < p
    assert S.fdtri(1, 5e-324, 0.1) == math.inf
    # a normal quotient keeps the plain product's bits
    assert S.fdtri(3.0, 7.0, 0.3) == 0.49029710674478816


def test_ndtri_exp_near_minus_log_2():
    # p - 1/2 from expm1(y + log 2) with log 2 in two parts: exp(y) - 1/2 cancelled every bit
    # of q at y = -log 2 (gave 0). Values: sqrt(2) erfinv(2 exp(y) - 1) in Arb.
    rows = [
        (-0.6931471805599453, 2.9064941568900345e-17),
        (-0.6931471805599454, -1.10080879664688e-16),
        (-0.6931471805599452, 1.682107628024887e-16),
        (-0.69314718, 7.017874053400538e-10),
    ]
    for y, want in rows:
        assert abs(S.ndtri_exp(y) / want - 1) < 4e-16, y
    ys = sd.asarray(np.linspace(-2.7, -0.05, 20001))
    before = sd.get_num_threads()
    try:
        runs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            runs.append(np.asarray(S.ndtri_exp(ys)).view(np.uint64))
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        np.testing.assert_array_equal(r, runs[0])
    lanes = np.asarray(S.ndtri_exp(ys))
    assert [S.ndtri_exp(float(y)) for y in np.asarray(ys)[::997]] == list(lanes[::997])


# ---- argument names (user decision 2026-11-20): chdtr(df, x), chdtri(df, q), ncfdtr(..., x) ----


def test_renamed_arguments_in_messages():
    for f, args, name in (
        (S.chdtr, ("a", 1.0), "df"),
        (S.chdtrc, (1.0, "a"), "x"),
        (S.chdtri, (1.0, "a"), "q"),
        (S.ncfdtr, (1.0, 1.0, 1.0, "a"), "x"),
        (S.ncfdtrc, (1.0, 1.0, 1.0, "a"), "x"),
    ):
        with pytest.raises(TypeError, match=rf"^{f.__name__}\(\): '{name}' must be"):
            f(*args)
