"""S3: gamma rgamma gammaln digamma polygamma beta betaln binom zeta (docs/api/special.md §3, §7).

Accuracy is tests/special/test_reference.py's. Here: the special values, bit for bit against
scipy except where docs/design/special.md §2.4 says otherwise (S3 user decisions), and the
argument conventions."""

import itertools

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd

S = sd.special

EDGES = [0.0, -0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, -2.5, np.inf, -np.inf, np.nan, 1e-300, -1e-300, 1e300, -1e300,
         171.7, -171.5, -180.5, 23.0]


def same_bits(a, b):
    a, b = np.float64(a), np.float64(b)
    return bool(np.isnan(a) and np.isnan(b)) or a.view(np.uint64) == b.view(np.uint64)


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
            assert same_bits(got, want), (name, x, got, want)


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
            assert same_bits(got, want), (name, a, b, got, want)


# ---- the S3 user decisions (docs/design/special.md §2.4)

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


def test_same_bits_for_any_thread_count():
    before = sd.get_num_threads()
    rng = np.random.default_rng(3)
    x = sd.asarray(rng.uniform(-50, 50, 200_000))
    y = sd.asarray(rng.uniform(0.1, 30, 200_000))
    try:
        runs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            runs.append([np.asarray(r).view(np.uint64) for r in
                         (S.gamma(x), S.rgamma(x), S.gammaln(x), S.digamma(x), S.polygamma(2.0, y), S.beta(y, y),
                          S.betaln(x, y), S.binom(y, x), S.zeta(x), S.zeta(y + 1.0, y))])
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        for a, b in zip(r, runs[0]):
            np.testing.assert_array_equal(a, b)
