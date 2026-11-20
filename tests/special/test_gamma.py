"""gamma rgamma gammaln digamma polygamma beta betaln binom zeta (docs/api/special.md §3, §7).

Accuracy is tests/special/test_reference.py's. Here: the special values, bit for bit against
scipy except where docs/design/special.md §2.4 says otherwise (true values), and the
argument conventions."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd
from tests._bits import bits_equal_any_nan

S = sd.special

EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    -2.0,
    0.5,
    -0.5,
    -2.5,
    np.inf,
    -np.inf,
    np.nan,
    1e-300,
    -1e-300,
    1e300,
    -1e300,
    171.7,
    -171.5,
    -180.5,
    23.0,
]


def special(v):
    return v == 0 or not np.isfinite(v)


@pytest.mark.parametrize("name", ["gamma", "rgamma", "gammaln", "digamma", "zeta"])
def test_special_values_unary(name):
    f, g = getattr(S, name), getattr(sp, name)
    for x in EDGES:
        got, want = f(x), g(x)
        if (name, x) in {("gamma", -171.5), ("rgamma", -171.5), ("rgamma", 171.7)}:
            continue  # subnormal tails, decision 3 below
        if special(got) or special(want):
            assert bits_equal_any_nan(got, want), (name, x, got, want)


@pytest.mark.parametrize("name", ["beta", "betaln", "binom", "polygamma"])
def test_special_values_binary(name):
    f, g = getattr(S, name), getattr(sp, name)
    pts = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.5, np.inf, 1e300, 3.0]
    for a, b in itertools.product(pts, pts):
        if name == "polygamma" and (a < 0 or a != np.floor(a) or a >= 2**53):
            continue  # beyond 2^53, n + 1 == n and scipy's (-1)^(n+1) flips: not a special value
        if name == "binom" and (a < 0 and a == np.floor(a) or (a, b) == (0.0, 1e300)):
            continue  # decision: the falling factorial; binom(0, 1e300) = 0 (scipy -2.9e-301)
        got, want = f(a, b), float(g(a, b))
        if name in ("beta", "betaln") and np.inf in (a, b) and 1e300 in (a, b):
            continue  # B(inf, 1e300) = 0: scipy's NaN comes from 1e6 * 1e300 overflowing inside it
        if name == "binom" and a < 0 and a != np.floor(a) and b == 1e300:
            continue  # (-1)^k binom(k - n - 1, k) with an even k: +inf (scipy -inf), test_binom_huge_k
        if name == "binom" and np.isfinite(a) and a >= 0 and np.isfinite(b) and b == np.floor(b) and b > max(a + 1, 30):
            assert got == 0.0, (a, b, got)  # k > n: 0 (scipy's sign of this zero is an accident of its path)
            continue
        if special(got) or special(want):
            assert bits_equal_any_nan(got, want), (name, a, b, got, want)


# ---- true values where scipy differs (docs/design/special.md §2.4)


def test_log_gamma_of_the_smallest_subnormal_is_finite():
    assert S.gammaln(5e-324) == pytest.approx(744.4400719213812, rel=1e-15)
    assert S.gammaln(-5e-324) == pytest.approx(744.4400719213812, rel=1e-15)
    assert S.betaln(5e-324, 1.0) == pytest.approx(744.4400719213812, rel=1e-15)
    assert sp.gammaln(5e-324) == np.inf


def test_rgamma_of_tiny_negatives_keeps_its_value():
    assert S.rgamma(-5e-324) == -5e-324 and S.rgamma(5e-324) == 5e-324
    assert S.rgamma(-0.0) == 0.0 and np.signbit(S.rgamma(-0.0))


def test_rgamma_subnormal_tail():
    assert S.rgamma(171.7) == pytest.approx(3.77039886193425e-309, rel=1e-12) and sp.rgamma(171.7) == 0.0
    assert S.binom(0, 1e300) == 0.0


def test_deep_negative_gamma_subnormals_and_signs():
    assert S.gamma(-171.5) == pytest.approx(1.9316265431712e-310, rel=1e-12)  # scipy: 0.0
    assert S.gamma(-1e10 - 0.5) == 0.0 and np.signbit(S.gamma(-1e10 - 0.5))  # scipy: +0.0
    assert S.rgamma(-1e10 - 0.5) == -np.inf  # scipy: inf
    assert S.gamma(-172.5) < 0 and S.gamma(-173.5) > 0


def test_nan_propagates_through_beta():
    for a in [0.0, -0.0, -1.0, -3.0, -np.inf]:
        assert np.isnan(S.beta(a, np.nan)) and np.isnan(S.beta(np.nan, a))
        assert np.isnan(S.betaln(a, np.nan)) and np.isnan(S.betaln(np.nan, a))
    assert sp.beta(0.0, np.nan) == np.inf


def test_zeta_at_infinity():
    assert S.zeta(np.inf, 2.0) == 0.0 and S.zeta(np.inf, 1.0) == 1.0 and S.zeta(np.inf, 0.5) == np.inf
    assert np.isnan(sp.zeta(np.inf, 2.0))


def test_binom_huge_k():
    # the reflection and a Gamma ratio keep these finite and signed (scipy: -5.1e-151, -inf)
    assert S.binom(-0.5, 1e300) == pytest.approx(5.641895835477563e-151, rel=1e-14)
    assert S.binom(-2.5, 1e300) == np.inf
    assert S.binom(2.5, 1e8) == pytest.approx(-1.0578555154332213e-28, rel=1e-13)


def test_binom_negative_integer_n():
    assert S.binom(-5, 2) == 15.0 and S.binom(-1, 0) == 1.0 and S.binom(-1, 3) == -1.0
    assert S.binom(-5, 40) == pytest.approx(float(sp.binom(44, 40)), rel=1e-13)  # (-1)^k binom(k - n - 1, k)
    assert np.isnan(S.binom(-5, 2.5)) and np.isnan(S.binom(-1, -1))  # scipy's NaN kept
    assert np.isnan(sp.binom(-5, 2))


# ---- conventions


def test_integers_are_exact():
    for n in range(1, 24):
        assert S.gamma(float(n)) == float(np.prod(np.arange(1, n, dtype=object))) if n > 1 else S.gamma(1.0) == 1.0
    assert S.gammaln(1.0) == 0.0 and S.gammaln(2.0) == 0.0
    assert S.binom(30, 15) == 155117520.0 and S.binom(5, 7) == 0.0 and S.binom(5, -1) == 0.0


def test_polygamma_order_must_be_a_non_negative_integer():
    assert np.isnan(S.polygamma(1.5, 1.0)) and np.isnan(S.polygamma(-1, 1.0)) and np.isnan(S.polygamma(np.inf, 1.0))
    assert S.polygamma(0, 2.5) == S.digamma(2.5)
    assert S.polygamma(np.int64(1), 1.0) == S.polygamma(1.0, 1.0)
    p = S.polygamma(sd.asarray(np.array([1, 2, 3])), 1.0)
    assert p.dtype == sd.float64


def test_zeta_forms():
    assert S.zeta(2.0) == pytest.approx(np.pi**2 / 6, rel=1e-15) and S.zeta(2.0, 1.0) == S.zeta(2.0, 1.0)
    assert S.zeta(-2.0) == 0.0 and S.zeta(0.0) == -0.5 and S.zeta(1.0) == np.inf
    assert np.isnan(S.zeta(0.5, 1.0))  # Hurwitz below 1: NaN (scipy)
    x = sd.asarray(np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(np.asarray(S.zeta(x)), sp.zeta([2.0, 3.0, 4.0]), rtol=1e-15)
    np.testing.assert_allclose(np.asarray(S.zeta(x, 2.0)), sp.zeta([2.0, 3.0, 4.0], 2.0), rtol=1e-14)
    out = sd.zeros(3)
    assert S.zeta(x, out=out) is out
    with pytest.raises(TypeError, match=r"^zeta\(\): 'q' must be"):
        S.zeta(2.0, "a")


# zeta(2), zeta(4), ..., zeta(52) rounded to the nearest double (Arb)
ZETA_EVEN = [
    1.6449340668482264,
    1.0823232337111381,
    1.0173430619844492,
    1.0040773561979444,
    1.000994575127818,
    1.000246086553308,
    1.0000612481350588,
    1.0000152822594086,
    1.000003817293265,
    1.0000009539620338,
    1.0000002384505027,
    1.000000059608189,
    1.0000000149015549,
    1.000000003725334,
    1.0000000009313275,
    1.000000000232831,
    1.0000000000582077,
    1.000000000014552,
    1.000000000003638,
    1.0000000000009095,
    1.0000000000002274,
    1.0000000000000568,
    1.0000000000000142,
    1.0000000000000036,
    1.0000000000000009,
    1.0000000000000002,
]


def test_zeta_even_integers():
    # the closed form 2^(n-1) pi^n |B_n| / n! loses up to 9 ULP (zeta(50) below 1): these are exact
    n = np.arange(2.0, 54.0, 2.0)
    assert [S.zeta(float(k)) for k in n] == ZETA_EVEN
    np.testing.assert_array_equal(np.asarray(S.zeta(sd.asarray(n))), ZETA_EVEN)
    assert S.zeta(50.0) == 1.0 + 2.0**-50 and S.zeta(52.0) == 1.0 + 2.0**-52


def test_zeta_large_argument_is_one_plus_two_to_minus_x():
    # zeta(x) = 1 + 2^-x + 3^-x + ...: never below 1, and from x = 20 the terms up to 8^-x are
    # the value to rounding (9^-20 is 2^-63)
    x = np.unique(
        np.concatenate(
            [
                np.arange(20.0, 201.0),
                np.arange(20.5, 200.0),
                np.linspace(20.0, 200.0, 20001),
                np.nextafter(np.arange(20.0, 60.0), 0),
                np.nextafter(np.arange(20.0, 60.0), 99),
            ]
        )
    )
    got = np.asarray(S.zeta(sd.asarray(x)))
    assert np.all(got >= 1.0) and np.all(got[x > 54.0] == 1.0)
    # the sum from its small end, each term exact to rounding: within 1 ULP of the truth
    want = 1.0 + (2.0**-x + (3.0**-x + (4.0**-x + (5.0**-x + (6.0**-x + (7.0**-x + 8.0**-x))))))
    assert np.all(np.abs(got - want) <= 2.0**-52)
    for xi in (20.0, 33.0, 50.0, 52.0, 53.0, 20.5, 47.25):
        assert S.zeta(xi) == got[np.searchsorted(x, xi)]


# (x, zeta(x) rounded to the nearest double by Arb): the points of (-60, -1) where rounding 1 - x
# before Gamma and a rounded 2 pi raised to x cost the most (65 ULP at -31.8), the neighbours of
# trivial zeros, and half-integers
ZETA_NEGATIVE = [
    (-58.95316528549029, 3.2015956501837933e32),
    (-49.657591587200635, -2.999410762012745e23),
    (-37.96167526092215, -4362222533570.371),
    (-31.803887378059255, 528796592.41541994),
    (-28.582168565967354, -8349832.126892958),
    (-15.488206159334117, 0.4998399508963755),
    (-9.296939563796487, -0.007685097995120685),
    (-1.1407067491725131, -0.06236808058755063),
    (-31.999999999999996, 1.3425985448140925e-05),
    (-32.00000000000001, -2.685197089628232e-05),
    (-57.99999999999999, -4.239351536427316e17),
    (-2.0000000000000004, 1.3521831236215423e-17),
    (-57.5, -8.820928901118726e30),
    (-28.5, -6571686.491569958),
    (-10.5, 0.011146122473942813),
]


def test_zeta_negative_argument_reflection():
    # relative accuracy, the trivial zeros' neighbours included: 10 ULP here, 45 the guarantee
    x = np.array([p[0] for p in ZETA_NEGATIVE])
    want = np.array([p[1] for p in ZETA_NEGATIVE])
    got = np.asarray(S.zeta(sd.asarray(x)))
    assert np.all(np.abs(got - want) <= 10 * np.spacing(np.abs(want)))
    assert [S.zeta(float(v)) for v in x] == list(got)


def test_same_bits_for_any_thread_count():
    before = sd.get_num_threads()
    rng = np.random.default_rng(3)
    x = sd.asarray(rng.uniform(-50, 50, 200_000))
    y = sd.asarray(rng.uniform(0.1, 30, 200_000))
    try:
        runs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            runs.append(
                [
                    np.asarray(r).view(np.uint64)
                    for r in (
                        S.gamma(x),
                        S.rgamma(x),
                        S.gammaln(x),
                        S.digamma(x),
                        S.polygamma(2.0, y),
                        S.beta(y, y),
                        S.betaln(x, y),
                        S.binom(y, x),
                        S.zeta(x),
                        S.zeta(y + 1.0, y),
                    )
                ]
            )
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        for a, b in zip(r, runs[0]):
            np.testing.assert_array_equal(a, b)


def test_binom_at_minus_infinite_k():
    # |binom(n, k)| ~ Gamma(n + 1) |k|^-(n+1) / pi as k -> -inf: 0 for finite n > -1, no limit below
    for n in (5.0, 5.5, 0.0, -0.5):
        v = float(S.binom(n, -np.inf))
        assert v == 0.0 and not np.signbit(v)
    for n in (-1.0, -1.5, -3.0, np.inf):
        assert np.isnan(float(S.binom(n, -np.inf)))
    assert np.isnan(float(S.binom(5.0, np.inf)))
