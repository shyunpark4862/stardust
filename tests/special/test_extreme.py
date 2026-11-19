"""Extreme arguments (the 2026-11-17 sweep; docs/design/stats.md 1.3): every call returns in
bounded time, and a value is the right one or NaN. The expected values are Arb's
(tools/special_ref.py's references at the same points), rounded to a double."""

import math
import time

import pytest

import stardust as sd

S = sd.special
MAXD = 1.7976931348623157e308

# (function, arguments, Arb's value): points that were wrong, NaN or did not return
VALUES = [
    ("betaincinv", (14.38, 1e200, 0.617), 1.5190988385196275e-199),
    ("betainccinv", (14.38, 1e200, 0.617), 1.2964096042260577e-199),
    ("betainc", (14.38, 1e200, 4.314e-199), 0.9999998829522595),
    ("betaincc", (14.38, 1e250, 4.314000000000001e-249), 1.1704774050494255e-07),
    ("betainc", (100000.0, 1e50, 1e-45), 0.5004205221103728),
    ("betaincinv", (2.0, 1e-15, 1e-15), 0.8414056604369606),
    ("betainccinv", (1e-100, 2.0, 1e-100), 0.15859433956303937),
    ("betaincinv", (10.0, 14.38, 5e-324), 1.1336664780039086e-33),
    ("betainccinv", (3.0, 1e10, 5e-324), 7.570082864634377e-08),
    ("betainccinv", (2.5, 1e7, 1e-300), 7.002957184532962e-05),
    ("betaincinv", (14.38, 1e-10, 1e-300), 8.19621372187887e-21),
    ("gammainccinv", (50.5, 5e-324), 936.6704584017244),
    ("gammaincinv", (1e10, 1e-300), 9996295747.51846),
    ("chdtri", (1.5, 5e-324), 1485.16784168784),
    ("chdtrc", (1e-100, 5e-324), 3.7227800171851986e-98),
    ("chdtrc", (5e-324, 5e-324), 1.84e-321),
    ("fdtri", (3.0, 1e16, 5e-324), 2.338061758969395e-216),
    ("stdtr", (MAXD, 1e-10), 0.5000000000398942),
    ("binom", (0.1, 1000.0), -4.690261493098894e-05),
    ("binom", (1.5, 1000.0), 1.3406060425696815e-08),
    ("binom", (0.9999999999999999, 7.0), 2.6433881538694206e-18),
    ("binom", (1e-10, 1e100), -9.999999769164278e-111),
    ("beta", (-500.5, 1000.5), -5.9990466073745906e299),
    ("beta", (-14.38, 1e16), -1.678291660798581e219),
    ("betaln", (-5e-324, 2.0), 744.4400719213812),
    ("polygamma", (1e4, 3680.0), -0.0027537179881605865),
    ("polygamma", (1e6, 367880.0), -0.0015970388032333072),
    ("polygamma", (20.0, -2.5), -9182070.204945628),
    ("polygamma", (300.0, -50.5), -1.7106434084467737e99),
    ("polygamma", (20.0, 1e16), -1.2164510040883212e-303),
    ("zeta", (3.0, -100.5), 4.901360131359879e-05),
    ("jv", (0.5, 5e-324), 1.7735048886036274e-162),
    ("jv", (-0.5, 5e-324), 3.589613857049051e161),
    ("yv", (0.25, 5e-324), -9.205397297526877e80),
    ("yv", (0.0, 5e-324), -473.9990734230043),
    ("kv", (0.5, 5e-324), 5.63855226126471e161),
    ("kv", (0.49, 1e-300), 1.2696541839003133e147),
    ("kv", (0.0, 5e-324), 744.5560034370396),
    ("iv", (-0.25, 1e-310), 3.0688361644828e77),
    ("kve", (-1e50, 1e100), 2.0663656770612466e-50),
    ("kve", (-0.25, MAXD), 9.347643879329245e-155),
    ("ive", (100.5, MAXD), 2.9754474593158995e-155),
    ("ive", (1e10, 1e21), 1.200038948430136e-11),
    ("j1", (-3e-308,), -1.5000000000000004e-308),
    ("expi", (2e-323,), -742.4765618953599),
    ("hyp1f1", (-1e-300, 20.0, 700.0), 1.0),
    ("hyp1f1", (1e-300, -100.5, 2.0), 1.0),
    ("fdtri", (MAXD, 2.0, 0.5), 1.4426950408889634),
    ("fdtr", (1e-10, 1e-300, 5e-324), 9.999999961654802e-291),
    ("stdtr", (1e-300, -1.7e308), 0.5),
    ("stdtr", (1e-10, -1.7e308), 0.4999999639033558),
    # the loose region of hyp1f1: a series that kept its digits, the recurrence in a, the
    # expansions (Boost's values here were wrong by up to 148 orders of magnitude)
    ("hyp1f1", (-1000.0, -100.5, 0.999), 34799.14203063519),
    ("hyp1f1", (100.5, -100.5, -2.0), 7.70548315058123),
    ("hyp1f1", (20.0, 1e-10, -800.0), 3.619051865405504e-40),
    ("hyp1f1", (1.0, 1e-10, -1e-10), 9.9999999985e-11),
    ("hyp1f1", (-1e5, 20.0, 0.999), 4.238238016642356e-33),
    ("hyp1f1", (-1e5, 2.0, 0.9), 0.000125219456806466),
    ("hyp1f1", (-1000.0, 2.0, 0.9), 0.00230028182027408),
    ("hyp1f1", (-1000.5, 3.5, 12.0), 0.0003572830381750206),
    ("hyp1f1", (1e5, 2.0, -2.0), 2.1712476851990025e-05),
    ("hyp1f1", (3.0, 2.0, -700.0), -3.44102711377216e-302),
    ("hyp1f1", (0.010644339467115094, -2370.1352708166628, 731.0), -1.751244593765022e132),
    ("hyp1f1", (0.5, -8378.3086387652111, -43940.236563906175), 0.3329326928762033),
    # the guaranteed region of hyp1f1 beyond the measured shapes
    ("hyp1f1", (0.5, 1000.0, -1e5), 0.09946713465278335),
    ("hyp1f1", (1e5, 0.5, 0.9999999999999999), 3.871111604857811e274),
    ("hyp1f1", (0.5, 5e-324, -1e-100), -1.0120112665365531e223),
    # hyp2f1: Pfaff with (1 - z)^-a beyond the range, the transformation to 1 - z with scaled
    # Gammas and exact parameters, subnormal c
    ("hyp2f1", (20.0, 1.0, 1.0, -10.0), 1.486436280241437e-21),
    ("hyp2f1", (1.0, 20.0, 20.0, -1e16), 9.999999999999999e-17),
    ("hyp2f1", (1e-10, 1e-300, 3.0, -1e5), 1.0),
    ("hyp2f1", (1e100, 1e-100, -100.5, -1e-100), 1.0),
    ("hyp2f1", (10.0, 100.5, -1e-10, -10.0), 6.896165309945678e-24),
    ("hyp2f1", (1.5, 2.5, 3.25, 0.999), 463.05219796391196),
    ("hyp2f1", (0.5, 1.0, 5e-324, 1e-100), 1.0120112665365531e223),
    # (the guaranteed region with a large a: these two were NaN)
    ("hyp2f1", (16875.5, 0.5, 209.56858729337603, -0.11643960124986097), 0.3099924191781808),
    ("hyp2f1", (27119.0, 0.015613586964303486, 404.6943711783311, -0.99574664984313488), 0.9362831277073627),
    # binom with a huge n or k (n >= 0: the guaranteed region)
    ("binom", (0.1, 1e100), -9.357787209128716e-112),
    ("binom", (1e-5, 1e300), -9.931103159343954e-306),
    ("binom", (0.9, 2.5822498780869086e120), -1.5599300736602532e-230),
    ("binom", (1.30214365629727e-05, -1169176540410475.8), -1.9242237346658609e-16),
    ("binom", (1e7, -5e-324), 1.0),
    ("binom", (-2.5, 1e100), 7.5225277806367505e149),
    # shapes and degrees of freedom next to the smallest double
    ("betainc", (5e-324, 5e-324, 0.617), 0.5),
    ("betainc", (1e-300, 5e-324, 0.9999999999), 4.940656458412465e-24),
    ("betaincc", (5e-324, 1e-300, 1e-20), 4.940656458412465e-24),
    ("betaincc", (5e-324, 5e-324, 0.9), 0.5),
    ("fdtr", (1e-300, 5e-324, 0.5), 4.940656458412465e-24),
    ("fdtr", (MAXD, 5e-324, 1e300), 3.547e-321),
    ("fdtrc", (5e-324, 1e-100, 0.5), 4.940656458412466e-224),
    ("chdtrc", (5e-324, 1e-300), 1.705e-321),
    ("stdtr", (0.9, -MAXD), 1.1661168283467197e-278),
    ("betaln", (1e300, MAXD), -2.0007184997951635e301),
    # a noncentrality whose square is below the doubles
    ("nctdtr", (0.5, -5e-324, -1e300), 3.207009754142229e-151),
    ("nctdtr", (1e-300, 5e-324, 1e100), 0.5),
    ("nctdtrc", (1e-300, 1e-300, 1000.0), 0.5),
    ("nctdtrit", (2.0, 1e-300, 5e-324), -3.1812124520951964e161),
]


@pytest.mark.parametrize("name,args,want", VALUES, ids=[f"{n}{a}" for n, a, _ in VALUES])
def test_values(name, args, want):
    got = getattr(S, name)(*args)
    assert abs(got - want) <= 1e-13 * abs(want) + 5e-323, (got, want)


# hyp1f1 in its guaranteed region at x >= 16 with b < 11: Boost was up to 297 ULP off there
# (Arb truths)
HYP1F1_LARGE_X = [
    ((32.0, 0.0242, 515.2), 8.93725328738386e278),
    ((72.92, 1.834, 276.4), 3.066619017779241e196),
    ((21.7, 0.039, 567.7), 4.2397933336698174e288),
    ((8.19, 3.06, 18.56), 507708616608.90955),
]


@pytest.mark.parametrize("args,want", HYP1F1_LARGE_X, ids=[str(a) for a, _ in HYP1F1_LARGE_X])
def test_hyp1f1_guaranteed_large_x(args, want):
    got = S.hyp1f1(*args)
    assert abs(got - want) <= 1e-14 * abs(want), (got, want)


def test_infinite_and_zero_limits():
    assert S.iv(2.0, 1e16) == math.inf and S.iv(1000.0, MAXD) == math.inf
    assert S.kv(300.0, 8.452712498170644e270) == 0.0
    assert tuple(S.airy(1e206)) == (0.0, -0.0, math.inf, math.inf)
    assert S.hyp1f1(100.5, 1e16, 1e300) == math.inf
    assert S.hyp1f1(-1000.0, 2.0, -1e300) == math.inf
    assert S.polygamma(1e300, 2.0) == -math.inf and S.polygamma(1e7, 1e7) == 0.0
    assert S.polygamma(1e4, 3000.0) == -math.inf
    assert S.rgamma(-4503599627370494.5) == -math.inf
    assert S.fdtrc(MAXD, MAXD, 1e-10) == 1.0 and S.fdtrc(MAXD, MAXD, 2.0) == 0.0
    assert S.betainc(1e300, MAXD, 1e-10) == 0.0
    assert S.betainc(1e300, 1e300, 0.5) == 0.5 and S.betainc(3e200, 1e200, 0.75) == 0.5
    assert S.betaincinv(1e-10, 0.001, 0.9) == 0.0
    assert S.yv(MAXD, 2.0) == -math.inf and S.jv(MAXD, 2.0) == 0.0
    assert S.fdtr(1.7e308, 1.7e308, 0.5) == 0.0 and S.fdtr(1.7e308, 1.7e308, 1.5) == 1.0
    assert S.gammainc(1e100, 1e100) == 0.5 and S.gammainc(1e100, 1.0000000000000002e100) == 1.0
    assert S.gammaincinv(1e100, 0.3) == 1e100 and S.betaincinv(1e100, 1e100, 0.3) == 0.5
    assert S.hyp1f1(243.0, 1.5, -532863.0) == 0.0 and S.hyp1f1(-1.0, 200.0, 200.0) == 0.0
    assert S.hyp2f1(2832.767085636945, 0.0024580177239800797, 64.0, 0.67838080817888535) == math.inf
    assert S.hyp2f1(1e-25, 1e6, 1.0, 0.5) == math.inf
    assert S.hyp2f1(0.8056250417267026, -0.004406914025902325, -2019.1923970877576, 0.9145144646079915) == -math.inf
    assert S.binom(7224.5387284817061, 3578.5) == math.inf and S.binom(1e10 + 0.5, 77.25) == math.inf
    for n, k in ((6253.0, 15620.5), (9.1787364855220693e82, -179703.5), (6000.3, 1e7)):
        v = S.binom(n, k)
        assert v == 0.0 and math.copysign(1.0, v) == -1.0, (n, k, v)


def test_nan_where_no_value():
    # J, Y past the recurrences' reach; airy past the phase's; the loose hypergeometric regions
    for v in (
        S.jv(1e100, 1e100),
        S.jv(1e10, 1e12),
        S.yv(1e10, 1e16),
        S.airy(-1e21)[0],
        S.hyp1f1(-1e12 - 0.5, -2.5, -1e12 - 0.5),
        S.hyp2f1(10.0, -100.5, 1e-10, 0.9),
        S.hyp2f1(1e5, 1e5, 10.0, -100.5),
    ):
        assert math.isnan(v)
    # hyp1f1 where no form with a known loss applies (|a| or |b| above 100 with b < 0; Boost's
    # values there were wrong by orders of magnitude: 2.4e187 for 2.5e67 at the third point)
    for a, b, x in (
        (137.5, -33.5, -4.3177496217381481),
        (-11224.0, -102.60187231307339, 1.0),
        (8534.5, -102.5, -3.8173757356474955),
    ):
        assert math.isnan(S.hyp1f1(a, b, x)), (a, b, x)
    # hyp2f1 sums that lost every digit to cancellation (they were 2.0 - 1e-10 off, 0 and 5e-12 off)
    for a, b, c, z in (
        (2.0, -1e-300, 1e-300, -1e5),
        (-0.065882317250010827, -293.0, -1.5726074559891865, 0.30391397664010666),
        (20.0, 20.0, 0.5, -10.0),
    ):
        assert math.isnan(S.hyp2f1(a, b, c, z)), (a, b, c, z)
    # the noncentral F's small side as the complement of a side within 1/64 of 1 (they were
    # 2.2e-16 for 3.4e-298, 0 for 3.5e-298 and 4e7 ULP off)
    assert math.isnan(S.ncfdtr(1e7, 1e-300, 2.0, 0.1)) and math.isnan(S.ncfdtrc(1e-300, 100.5, 1e-300, 0.001))
    assert math.isnan(S.ncfdtr(100.5, 1e-10, 1.0, MAXD))
    # the noncentral inverses at a subnormal p (they were 9% and 16% off)
    assert math.isnan(S.chndtrix(14.38, 2.0, 5e-324)) and math.isnan(S.nctdtrit(2.0, 2.0, 5e-324))
    assert math.isnan(S.ncfdtri(14.38, 0.5, 1e-10, 5e-324))


def test_complex_ends():
    w = S.lambertw(complex(-MAXD, -MAXD), k=-1)
    assert abs(w - (703.5730396182713 - 8.627118545552033j)) < 1e-11
    w = S.lambertw(complex(5e-324, -5e-324), k=-1)
    assert abs(w - (-750.714537518813 - 3.9322287504391453j)) < 1e-11
    w = S.lambertw(complex(-1e-200, -1e-300), k=1)
    assert abs(w.real + 466.6626251653469) < 1e-11 and 0 < w.imag < 1e-99
    assert abs(S.lambertw(-1e-310, k=-1) + 720.3811592879879) < 1e-11
    z = S.loggamma(complex(-34.0, -5e-324))
    assert abs(z - (655.8592443791836 + 108.38494654884786j)) < 1e-11


# calls that did not return (or took seconds, or crashed the process): each is now well under a second
BOUNDED = [
    ("airy", (1e250,)),
    ("polygamma", (1e300, 100.5)),
    ("polygamma", (1e10, 1e5)),
    ("jv", (MAXD, 1e300)),
    ("jv", (1e10, 1e12)),
    ("jv", (1e100, 1e100)),
    ("yv", (MAXD, 0.5)),
    ("yv", (1e10, 1e16)),
    ("iv", (2.0, 1e300)),
    ("iv", (-0.5, 1e12)),
    ("kv", (0.5, 5e-324)),
    ("hyp1f1", (MAXD, -100.5, -1e16)),
    ("hyp1f1", (1e300, 2.0, MAXD)),
    ("hyp1f1", (1e-300, -100.5, 2.0)),
    ("hyp1f1", (-1e5, -MAXD, -1e300)),
    ("hyp1f1", (1e12, -1e5 - 0.5, -1e12 - 0.5)),
    ("hyp2f1", (1e300, 1e300, 1e5, -1e300)),
    ("hyp2f1", (1e5, 1e5, 10.0, -100.5)),
    ("hyp2f1", (1e16, 1e16, 3.0, -1e5)),
    ("zeta", (3.0, -1e15 - 0.5)),
    ("betaincinv", (1e100, 2.5, 0.3)),
    ("betaincinv", (1e300, 1e200, 1e-10)),
    ("gammaincinv", (1e300, 1e-200)),
    ("gammainccinv", (1e100, 5e-324)),
    ("hyp1f1", (-1e6, 2.5, 0.9)),
    ("hyp1f1", (-999999.5, 2.5, 0.9)),
    ("hyp1f1", (1e6, 0.5, 1e6)),
    ("hyp2f1", (1e5, 1e5, 0.5, 0.8999)),
    ("binom", (1e300, 1e299)),
    ("binom", (1e300, -1e299 - 0.5)),
]


def test_bounded_time():
    for name, args in BOUNDED:
        t = time.perf_counter()
        getattr(S, name)(*args)
        assert time.perf_counter() - t < 1.0, (name, args)
