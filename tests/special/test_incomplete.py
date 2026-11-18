"""S4: gammainc gammaincc gammaincinv gammainccinv betainc betaincc betaincinv betainccinv
(docs/api/special.md §3).

Accuracy is tests/special/test_reference.py's. Here: the special values, bit for bit against
scipy except where docs/design/special.md §2.4 says otherwise (S4 user decision: true values
in place of scipy's errors), the identities between the pairs, and the inverses' round trips."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd

S = sd.special


def same_bits(a, b):
    a, b = np.float64(a), np.float64(b)
    return bool(np.isnan(a) and np.isnan(b)) or a.view(np.uint64) == b.view(np.uint64)


EDGE = [0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-300, 1e300, np.inf, np.nan]


@pytest.mark.parametrize("name", ["gammainc", "gammaincc"])
def test_gamma_edges_match_scipy(name):
    f, g = getattr(S, name), getattr(sp, name)
    for a, x in itertools.product(EDGE, EDGE):
        got, want = f(a, x), float(g(a, x))
        if want == 0 or not np.isfinite(want) or a in (0.0, np.inf) or x in (0.0, np.inf):
            assert same_bits(got, want), (name, a, x, got, want)


@pytest.mark.parametrize("name", ["betainc", "betaincc"])
def test_beta_edges_match_scipy(name):
    f, g = getattr(S, name), getattr(sp, name)
    ab = [0.0, 1.0, 2.5, 1e-300, 1e300, np.inf, -1.0, np.nan]
    xs = [0.0, -0.0, 0.5, 1.0, -0.1, 1.1, np.nan]
    for a, b, x in itertools.product(ab, ab, xs):
        if (a, b) == (1e300, 1e300) or (np.inf in (a, b) and x in (0.0, 1.0)):
            continue  # an ordinary value / scipy's order of edge rules, test_beta_edge_order
        got, want = f(a, b, x), float(g(a, b, x))
        if 0.0 in (a, b) or np.inf in (a, b) or x in (0.0, 1.0) or not np.isfinite(want):
            assert same_bits(got, want), (name, a, b, x, got, want)


def test_beta_edge_order():
    # x = 0 and x = 1 decide before a = 0, b = 0 (scipy): betainc(0, 1, 0) = 0, betainc(1, 0, 1) = 1
    assert S.betainc(0.0, 1.0, 0.0) == 0.0 and S.betainc(1.0, 0.0, 1.0) == 1.0
    assert S.betainc(0.0, 1.0, 1.0) == 1.0 and S.betainc(1.0, 0.0, 0.0) == 0.0
    assert np.isnan(S.betainc(0.0, 0.0, 0.5)) and np.isnan(S.betainc(np.inf, np.inf, 0.5))


@pytest.mark.parametrize("name", ["gammaincinv", "gammainccinv"])
def test_gamma_inverse_edges_match_scipy(name):
    f, g = getattr(S, name), getattr(sp, name)
    for a, p in itertools.product([0.0, 1.0, 2.5, np.inf, -1.0, np.nan], [0.0, -0.0, 0.5, 1.0, 1.5, -0.1, np.nan]):
        got, want = f(a, p), float(g(a, p))
        assert same_bits(got, want) or (np.isfinite(want) and want != 0 and abs(got - want) <= 1e-14 * want), (a, p)


@pytest.mark.parametrize("name", ["betaincinv", "betainccinv"])
def test_beta_inverse_edges_match_scipy(name):
    f, g = getattr(S, name), getattr(sp, name)
    v = [0.0, 2.0, 3.0, np.inf, -1.0, np.nan]
    for a, b, p in itertools.product(v, v, [0.0, 0.5, 1.0, 1.1, -0.1, np.nan]):
        got, want = f(a, b, p), float(g(a, b, p))
        assert same_bits(got, want) or abs(got - want) <= 1e-14 * abs(want), (name, a, b, p, got, want)


# ---- where scipy is wrong (S4 user decision: the true value) ----------------


def test_scipy_errors_get_true_values():
    assert S.gammainc(1e-300, 5e-324) == 1.0  # scipy 1.0000000000000238 (> 1)
    assert S.gammainc(5e-324, 0.5) == 1.0  # scipy 0
    assert S.gammainc(5e-324, 1e-300) == 1.0
    assert S.gammainc(1.0, 5e-324) == 5e-324  # scipy 0
    # scipy NaN; Arb gives the value below: x^2 b (b+1) / 2 (1 - 2 b x / 3 + ...), b x = 5e-24
    assert abs(S.betainc(2.0, 1e300, 5e-324) / 1.2205043120026404e-47 - 1) <= 1e-15
    assert S.gammaincinv(5e-324, 0.5) == 0.0  # scipy NaN: 0.5^(1/a) underflows
    x = S.betaincinv(2.0, 3.0, 5e-324)  # scipy NaN
    assert 0 < x < 1e-150 and abs(S.betainc(2.0, 3.0, x) - 5e-324) <= 45 * 2.0**-1074
    y = S.betainccinv(2.0, 3.0, 5e-324)
    assert y == 1.0  # 1 - x is ~1e-162: x rounds to 1
    assert S.betaincinv(0.0102, 36.2, 3.13e-169) == 0.0  # scipy 2.2e-308 (DBL_MIN): the root is 1e-16522


# ---- identities ---------------------------------------------------------------


def test_pairs_add_to_one():
    rng = np.random.default_rng(4)
    a = np.exp(rng.uniform(-5, 9, 2000))
    x = a * np.exp(rng.uniform(-3, 3, 2000))
    s = np.asarray(S.gammainc(sd.asarray(a), sd.asarray(x))) + np.asarray(S.gammaincc(sd.asarray(a), sd.asarray(x)))
    assert np.max(np.abs(s - 1)) <= 4e-16
    b = np.exp(rng.uniform(-5, 9, 2000))
    y = rng.uniform(0, 1, 2000)
    s = np.asarray(S.betainc(sd.asarray(a), sd.asarray(b), sd.asarray(y))) + \
        np.asarray(S.betaincc(sd.asarray(a), sd.asarray(b), sd.asarray(y)))
    # a, b <= 6 go to Boost (special/boost.hpp), which computes the two apart, each within the
    # contract (against Arb up to 9 ULP for a, b < 1) and at times off the same way: over 50
    # seeds of 1e5 draws of this distribution the sum was off by at most 1.0e-15 (a = 2.886,
    # b = 0.224, x = 0.943), 4.4e-16 at the 99.9th percentile
    assert np.max(np.abs(s - 1)) <= 2e-15


def test_beta_symmetry():
    # I_x(a, b) = 1 - I_(1-x)(b, a): with x = 1/2 both sides have the same exact inputs. Not
    # bit for bit: BRATIO picks its method by the order of (a, b)
    rng = np.random.default_rng(5)
    a, b = np.exp(rng.uniform(-4, 8, 500)), np.exp(rng.uniform(-4, 8, 500))
    for ai, bi in zip(a, b):
        u, v = S.betainc(ai, bi, 0.5), S.betaincc(bi, ai, 0.5)
        assert abs(u - v) <= 1e-14 * abs(v), (ai, bi, u, v)


def test_known_closed_forms():
    for x in [1e-10, 0.3, 1.0, 7.5, 40.0]:
        assert abs(S.gammaincc(1.0, x) / np.exp(-x) - 1) <= 4.5e-16  # Q(1, x) = e^-x (2 ULP)
    for x in [1e-8, 0.2, 0.5, 0.9]:
        assert abs(S.betainc(1.0, 3.0, x) / -np.expm1(3 * np.log1p(-x)) - 1) <= 4.5e-16  # 1 - (1-x)^3
        assert abs(S.betainc(2.0, 1.0, x) / x**2 - 1) <= 4.5e-16  # x**2 itself rounds


@pytest.mark.filterwarnings("ignore:overflow")
def test_inverse_round_trips():
    # |F(x) - p| <= the forward error + F' times the rounding of x to float64
    rng = np.random.default_rng(6)
    for _ in range(300):
        a = float(np.exp(rng.uniform(-6, 10)))
        p = float(rng.uniform(0, 1))
        x = S.gammaincinv(a, p)
        if x > 0:
            d = float(np.exp((a - 1) * np.log(x) - x - sp.gammaln(a)))
            assert abs(S.gammainc(a, x) - p) <= 1e-14 * p + d * np.spacing(x), (a, p, x)
        b = float(np.exp(rng.uniform(-6, 10)))
        y = S.betaincinv(a, b, p)
        if 0 < y < 1:
            d = float(np.exp((a - 1) * np.log(y) + (b - 1) * np.log1p(-y) - sp.betaln(a, b)))
            assert abs(S.betainc(a, b, y) - p) <= 1e-14 * p + d * np.spacing(y), (a, b, p, y)


def test_three_operand_broadcasting():
    a = sd.asarray(np.array([[0.5], [2.0], [30.0]]))
    b = sd.asarray(np.array([1.0, 4.0]))
    x = 0.25
    got = np.asarray(S.betainc(a, b, x))
    want = sp.betainc(np.array([[0.5], [2.0], [30.0]]), np.array([1.0, 4.0]), x)
    assert got.shape == (3, 2)
    assert np.allclose(got, want, rtol=1e-13, atol=0)


def test_a_zero_result_is_positive_zero_review_regression():
    # The prefix e^-E (1 - E.lo): from E = 2^53 the low part is no longer below 1, and the
    # factor turned the sign (gammainc(1e17, 1) was -0.0; scipy +0.0) or, from 1e300, made NaN.
    for a in [1e16, 1e17, 1e18, 1e100, 1e300]:
        got = [S.gammainc(a, 1.0), S.gammaincc(1.0, a), S.betainc(a, 2.0, 0.5), S.betaincc(2.0, a, 0.5),
               S.betainc(a, a, 0.25), S.betainc(a, 0.5, 0.5), S.betaincc(0.5, a, 0.5)]
        assert all(same_bits(v, 0.0) for v in got), (a, got)
        assert S.gammaincc(a, 1.0) == 1.0 and S.betaincc(a, 2.0, 0.5) == 1.0
    rng = np.random.default_rng(8)
    a, b = np.exp(rng.uniform(0, 690, 4000)), np.exp(rng.uniform(0, 690, 4000))
    x, u = a * np.exp(rng.uniform(-3, 3, 4000)), rng.uniform(0, 1, 4000)
    A, B, X, U = (sd.asarray(v) for v in (a, b, x, u))
    for v in (S.gammainc(A, X), S.gammaincc(A, X), S.betainc(A, B, U), S.betaincc(A, B, U)):
        v = np.asarray(v)
        assert not np.signbit(v).any() and np.all((v >= 0) & (v <= 1))


def test_betainc_where_one_shape_is_past_2_900():
    # The beta distribution's gamma limit, I_x(a, b) = P(a, b x): BRATIO's methods made NaN
    # of the first point. Values: the series of I_x in Arb (3000 bits).
    for a, b, x, lo, up in [(8.2, 3.7e277, 1.3e-276, 0.9999999999997419, 2.58155456806891e-13),
                            (0.3, 1e300, 1e-301, 0.5459128495917965, 0.45408715040820347),
                            (2.0, 1e290, 1e-310, 4.99999999999997e-41, 1.0)]:
        assert abs(S.betainc(a, b, x) / lo - 1) < 2e-15, (a, b, x)
        assert abs(S.betaincc(a, b, x) / up - 1) < 2e-15, (a, b, x)
        assert abs(S.betaincc(b, a, 1.0 - x) / lo - 1) < 2e-15 or x < 1e-16  # (1 - x rounds to 1 below that)
    assert S.betainc(1e300, 2.0, 0.5) == 0.0 and S.betaincc(1e300, 2.0, 0.5) == 1.0


def test_betaincc_small_shapes_on_the_side_boost_subtracts():
    # a, b <= 1, x < 0.02 and x^a in (0.6, 0.9]: Boost gives 1 - I as 1 minus a series near 1
    # (6.8e-15 off at the first point); that side is ibeta's own. Values from Arb.
    for a, b, x, lo, up in [(0.00039, 0.322, 1.54e-119, 0.8978643168072409, 0.10213568319275908),
                            (0.00027, 0.014, 1.4e-188, 0.872943608883056, 0.127056391116944),
                            (0.00636, 0.458, 2.36e-21, 0.7318146749103045, 0.26818532508969556),
                            (0.0861, 0.514, 0.0228, 0.6517547885654456, 0.3482452114345544)]:
        assert abs(S.betainc(a, b, x) / lo - 1) < 2.5e-15, (a, b, x)
        assert abs(S.betaincc(a, b, x) / up - 1) < 2.5e-15, (a, b, x)
        assert abs(S.betainc(b, a, 1.0 - x) / up - 1) < 2.5e-15 or x < 1e-16, (a, b, x)


def test_betainc_and_betaincc_add_to_one():
    # The two come from separate evaluations (Boost's ibeta and ibetac, or ibeta for one of
    # them): measured 4.5 ULP of 1 at most over 3.4e5 points of these shapes.
    rng = np.random.default_rng(12)
    n = 20000
    a, b = np.exp(rng.uniform(np.log(1e-5), np.log(6.0), n)), np.exp(rng.uniform(np.log(1e-5), np.log(6.0), n))
    x = np.where(np.arange(n) % 2 == 0, rng.uniform(0, 1, n), np.exp(np.log(rng.uniform(0.3, 1.0, n)) / a))
    ok = (x > 0) & (x < 1)
    A, B, X = (sd.asarray(v[ok]) for v in (a, b, x))
    lo, up = np.asarray(S.betainc(A, B, X)), np.asarray(S.betaincc(A, B, X))
    assert np.max(np.abs((lo - 0.5) + (up - 0.5))) <= 8 * 2.0**-52
