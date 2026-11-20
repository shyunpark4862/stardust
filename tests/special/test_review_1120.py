"""special: the review fixes of 2026-11-20 -- extreme arguments (subnormal shapes and
degrees of freedom, huge exponents, huge orders), hyp2f1's polynomials, the NaN of erf, and
the GIL of a heavy call on few elements. Reference values are Arb's (python-flint) where a
closed form does not give them."""

import math
import struct
import threading
import time

import numpy as np
import pytest

import stardust as sd

S = sd.special
INF, NAN = math.inf, math.nan


def bits(v):
    return struct.unpack("<Q", struct.pack("<d", v))[0]


def same(got, want):
    """Equal including the sign of 0 and of inf."""
    return got == want and math.copysign(1.0, got) == math.copysign(1.0, want)


# ---- exp of a double-double past the doubles (stdtrit, fdtri) ----


@pytest.mark.parametrize("df", [1e-20, 1e-18, 1e-300])
def test_stdtrit_tiny_df_sign(df):
    # |t| = sqrt(df) (q a B)^(-1/df) is past the doubles; the sign is p's side of 1/2
    for p in (0.1, 0.3, 0.49):
        assert S.stdtrit(df, p) == -INF
    for p in (0.51, 0.7, 0.9):
        assert S.stdtrit(df, p) == INF


def test_fdtri_tiny_df_not_negative():
    assert S.fdtri(1.0, 1e-16, 0.99) == INF
    assert S.fdtri(1.0, 1e-18, 0.9) == INF
    assert same(S.fdtri(1e-17, 1.0, 0.1), 0.0)


# ---- subnormal degrees of freedom: the half carried exactly ----


def test_stdtr_subnormal_df_is_one_half():
    for df in (1e-310, 5e-324):
        for t in (-1e5, -1.0, 0.25, 1.0, 300.0):
            assert S.stdtr(df, t) == 0.5
    assert S.stdtr(5e-324, INF) == 1.0


def test_stdtrit_fdtri_subnormal_df():
    for df in (1e-310, 5e-324):
        assert S.stdtrit(df, 0.3) == -INF
        assert S.stdtrit(df, 0.7) == INF
        assert S.stdtrit(df, 0.5) == 0.0
    # a vanishing dfn: fdtr = 1 - a C(x) is above every p < 1 (mass at 0)
    assert same(S.fdtri(1e-310, 1.0, 0.5), 0.0)
    assert same(S.fdtri(5e-324, 1.0, 0.5), 0.0)
    # a vanishing dfd: mass at inf
    assert S.fdtri(1.0, 1e-310, 0.5) == INF
    assert S.fdtri(1.0, 5e-324, 0.5) == INF
    # both: point masses dfd / (dfn + dfd) at 0 and the rest at inf
    assert same(S.fdtri(5e-324, 1e-323, 0.3), 0.0)
    assert S.fdtri(5e-324, 1e-323, 0.8) == INF
    assert S.fdtri(5e-324, 5e-324, 0.5) == 1.0


def test_fdtri_subnormal_dfd_small_p():
    # fdtr with dfd = 5e-324 is linear in dfd/2: equal to p where the dfd one equals 2p.
    # Its root against the forward function.
    x = S.fdtri(1.0, 5e-324, 5e-324)
    assert 0 < x < INF
    assert S.fdtr(1.0, 5e-324, x) == pytest.approx(5e-324, abs=2e-323)


def test_beta_both_subnormal_is_inf():
    for a, b in ((5e-324, 5e-324), (1e-310, 1e-310), (1e-320, 2e-320)):
        assert S.beta(a, b) == INF
    assert S.beta(2.2250738585072014e-308, 2.2250738585072014e-308) == 8.98846567431158e307
    # (a + b) / (a b) where it is a double
    assert S.beta(2.0**-1023, 2.0**-1022) == pytest.approx(3 * 2.0**1022, rel=1e-15)


# ---- inverse incomplete beta with a subnormal shape ----


def test_betaincinv_subnormal_shape():
    # I_x(1, b) = 1 - (1 - x)^b: x = 1 - e^(-p/b)
    assert S.betaincinv(1.0, 5e-324, 5e-324) == pytest.approx(1 - math.exp(-1), rel=1e-15)
    assert S.betainccinv(5e-324, 1.0, 5e-324) == pytest.approx(math.exp(-1), rel=1e-15)
    # both subnormal: the same direction as both 1e-300 (mass 1/2 at each end)
    ps = (5e-324, 0.1, 0.9, 0.99)
    assert [S.betaincinv(5e-324, 5e-324, p) for p in ps] == [0.0, 0.0, 1.0, 1.0]
    assert [S.betaincinv(1e-300, 1e-300, p) for p in ps] == [0.0, 0.0, 1.0, 1.0]
    # the variable's own shape subnormal: the root underflows
    assert S.betaincinv(5e-324, 0.5, 0.3) == 0.0


@pytest.mark.parametrize(
    "a, b, p, v",
    [
        # v = 1 - x with I_x(a, b) = p; Arb roots (python-flint, 4000 bits)
        (0.5, 5e-324, 3e-323, 0.009866037165440194),
        (2.5, 1e-310, 3e-310, 0.014133468873666533),
        (0.3, 2e-320, 1e-319, 0.11535968177171663),
        (40.0, 3e-322, 5e-324, 0.06893397074948072),
    ],
)
def test_betainccinv_subnormal_other_shape_arb(a, b, p, v):
    assert S.betainccinv(b, a, p) == pytest.approx(v, rel=1e-14)


def test_betaincinv_huge_shapes_subnormal_p():
    # the normal limit's root (with the skewness term): sigma is below 1e-50 here
    for p in (5e-324, 1e-310, 1e-300):
        assert S.betaincinv(1e100, 1e100, p) == 0.5
    assert S.betaincinv(1e300, 1e300, 5e-324) == 0.5
    assert S.fdtri(1e300, 1e300, 5e-324) == 1.0
    # where the log tail still settles, the same values as before
    assert S.betaincinv(1e20, 1e20, 5e-324) == pytest.approx(0.49999999863997185, rel=1e-15)


# ---- chi-square with a subnormal v ----


def test_chdtr_subnormal_v_at_zero():
    for v in (5e-324, 1.5e-323, 1e-323):
        assert S.chdtrc(v, 0.0) == 1.0
        assert same(S.chdtr(v, 0.0), 0.0)
    assert S.chdtrc(5e-324, INF) == 0.0
    assert math.isnan(S.chdtrc(5e-324, NAN))


def test_chdtri_subnormal_v():
    # Q(v/2, y) = p with v/2 not a double: Q(v, y) = 2 p; E1(y) = 2 here (Q(a, y) ~ a E1(y))
    x = S.chdtri(5e-324, 5e-324)
    assert x == pytest.approx(0.1647440592414405, rel=1e-14)  # 2 y, E1(y) = 2 (Arb)
    assert S.chdtrc(5e-324, x) == pytest.approx(5e-324, abs=1e-323)
    for p in (1e-300, 0.5, 0.9):
        assert S.chdtri(5e-324, p) == 0.0
    assert S.chdtri(5e-324, 0.0) == INF
    assert math.isnan(S.chdtri(5e-324, NAN))


# ---- Hurwitz zeta and polygamma below 0 with a large order ----


def test_polygamma_negative_x_large_order():
    assert S.polygamma(3000.0, -2.25) == INF
    assert S.polygamma(7000.0, -0.5) == -INF
    assert S.polygamma(7000.0, -2.25) == INF
    for x in (-0.5, -2.25, -100.3, -0.49999999999999994):
        r = [S.polygamma(float(n), x) for n in range(1000, 7001, 37)]
        assert not any(math.isnan(v) for v in r), x


def test_hurwitz_negative_q_large_s():
    assert S.zeta(3001.0, -2.25) == -INF
    # the two terms next to 0 cancel exactly: zeta(s, 1.5) remains
    assert S.zeta(1025.0, -0.5) == S.zeta(1025.0, 1.5)
    assert 0 < S.zeta(1025.0, -0.5) < 1e-180
    # Arb (the terms below 0 summed directly, 600 bits): the two near terms nearly cancel
    assert S.zeta(1025.0, -0.5000000000000001) == pytest.approx(1.6365890538184702e296, rel=1e-14)
    assert S.zeta(1025.0, -1.4999999999999998) == pytest.approx(-3.2731781076369405e296, rel=1e-14)
    assert S.zeta(1025.0, -0.49999999999999994) == pytest.approx(-8.182945269092351e295, rel=1e-14)
    assert S.zeta(1025.0, -70.49999999999999) == pytest.approx(-2.094833988887642e298, rel=1e-14)


# ---- hyp2f1's polynomials for 0 < z < 1 ----


@pytest.mark.parametrize(
    "a, b, c, z, want",
    [
        (-200.0, 1.0, 2.0, 0.3, 0.01658374792703151),  # (1 - (1 - z)^(n + 1)) / ((n + 1) z)
        (-60.0, 2.5, 3.7, 0.8, 0.00026218446555434496),  # Arb
        (-120.0, 2.5, 3.7, 0.5, 0.00015580656311105288),  # Arb
        (1.0, -500.0, 2.0, 0.99, 0.0020161696808403393),  # Arb
    ],
)
def test_hyp2f1_polynomial_pfaff(a, b, c, z, want):
    assert S.hyp2f1(a, b, c, z) == pytest.approx(want, rel=1e-14)


def test_hyp2f1_polynomial_short_unchanged():
    # a short polynomial is summed directly as before
    assert S.hyp2f1(-3.0, 1.0, 2.0, 0.3) == 0.63325


# ---- Bessel at x = v past the step bound: NaN at once ----


def test_jv_at_order_huge_is_fast_nan():
    for v in (1e16, 1e20, 1e300):
        t = time.perf_counter()
        assert math.isnan(S.jv(v, v)) and math.isnan(S.yv(v, v))
        assert time.perf_counter() - t < 0.01  # (it walked 1e7 terms: 45 ms each)


# ---- noncentral chi-square tails beyond e^-1000 ----


def test_chndtr_obvious_tails():
    assert same(S.chndtr(1.0, 1e16, 1.0), 0.0)
    assert S.chndtrc(1.0, 1e16, 1.0) == 1.0
    assert S.chndtr(1.0, 1e16, 1e17) == 1.0
    assert same(S.chndtrc(1.0, 1e16, 1e17), 0.0)


# ---- erf's NaN ----


def test_erf_nan_is_the_input():
    for f in (S.erf, S.erfc):
        assert bits(f(NAN)) == 0x7FF8000000000000
        for n in (1, 2, 3, 1025):
            r = np.asarray(f(sd.asarray([NAN] * n))).view(np.uint64)
            assert (r == 0x7FF8000000000000).all()


# ---- the GIL of a heavy call on few elements ----


@pytest.mark.serial
def test_heavy_call_on_few_elements_releases_the_gil():
    # yv costs 750 ns per element by the table (capped weight 256): 16 elements are below the
    # parallel threshold at the capped weight but above it at the uncapped one. These take
    # ~12 ms each (a 5e5-step recurrence), and another Python thread must keep running.
    v = sd.full((16,), 1.5e6)
    x = sd.full((16,), 1e6)
    old = sd.get_num_threads()
    sd.set_num_threads(1)
    longest = [0.0]
    stop = threading.Event()

    def watch():
        last = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            longest[0] = max(longest[0], now - last)
            last = now

    th = threading.Thread(target=watch)
    th.start()
    try:
        time.sleep(0.01)
        longest[0] = 0.0
        t = time.perf_counter()
        r = S.yv(v, x)
        took = time.perf_counter() - t
        time.sleep(0.01)
    finally:
        stop.set()
        th.join()
        sd.set_num_threads(old)
    assert np.all(np.asarray(r) == -INF)
    assert took > 0.05
    assert longest[0] < 0.5 * took, (longest[0], took)


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


def test_beta_inverse_infinite_shape_is_the_limit():
    for p in (1e-300, 0.3, 0.9):
        assert S.betaincinv(INF, 2.0, p) == 1.0 and S.betainccinv(INF, 2.0, p) == 1.0
        assert S.betaincinv(2.0, INF, p) == 0.0 and S.betainccinv(2.0, INF, p) == 0.0
        assert math.isnan(S.betaincinv(INF, INF, p))


@pytest.mark.parametrize(
    "b, a, q, v, kappa",
    [
        # just above 2^-1022 (normal shapes): the Halley steps were up to 1.5e-12 off there, so a
        # tiny shape is scaled up to 2^-200 like a subnormal one. v from Arb (bisection on
        # 1 - I_v(b, a) = q, 3000 bits); kappa = q / (v |dq/dv|)
        (1.1665795231290236e-301, 0.136554210235461, 1.5690494586085367e-300, 0.0017765499324829284, 13.4),
        (1.5514140572237347e-301, 112.3179196887478, 3.875345277175781e-301, 0.00043316777388613387, 2.6),
        (1.770144725896944e-301, 28.69279061721251, 7.43810908607955e-301, 0.0003005168241676121, 4.2),
    ],
)
def test_betainccinv_tiny_normal_shape_arb(b, a, q, v, kappa):
    assert S.betainccinv(b, a, q) == pytest.approx(v, rel=1e-14 * kappa)
    # the same root from the lower function: x = 1 - v
    assert S.betaincinv(a, b, q) == pytest.approx(1.0 - v, rel=1e-15)
