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


def test_huge_order_debye():
    # code review (S10): nu >= 1e4 by Debye's expansion (was O(nu): iv(1e9, 1e9) took 26 s and
    # gave 0, kv gave inf)
    import time

    t = time.perf_counter()
    assert S.iv(1e9, 1e9) == np.inf and S.kv(1e9, 1e9) == 0.0
    assert time.perf_counter() - t < 0.1
    # nu eta = 0: I ~ 1/sqrt(2 pi s), K ~ sqrt(pi / (2 s)) up to the 1/nu corrections
    nu, x = 1e6, 662743.4193491816
    s = np.hypot(nu, x)
    assert abs(S.iv(nu, x) * np.sqrt(2 * np.pi * s) - 1) < 1e-6
    assert abs(S.kv(nu, x) / np.sqrt(np.pi / (2 * s)) - 1) < 1e-6
    # continuity with the recurrences at the switch (nu = 1e4; e^-x I is in range from x ~ 7e4):
    # measured within 7e-16
    for x in [1e5, 3e5, 1e6]:
        lo, hi = S.ive(np.nextafter(1e4, 0), x), S.ive(1e4, x)
        assert abs(hi / lo - 1) < 1e-11, x


# (v, x, J_v(x), Y_v(x)): x is the float64 nearest a zero of J_v or Y_v (the first two, the
# 20th and, for v >= 0, the 300th), the values Arb's (400 bits), rounded.
AT_ZEROS = [
    (0.0, 2.404825557695773, -6.10876525973673e-17, 0.509924383448479),
    (2.5, 5.76345919689455, 4.0664393837162815e-17, 0.34833142749597434),
    (0.5, 1.5707963267948966, 0.6366197723675814, -3.8981718325193755e-17),
    (0.0, 5.520078110286311, -2.7522649432621832e-17, -0.3389361307570227),
    (0.0, 62.048469190227166, -3.592096500597402e-16, -0.10129021163422375),
    (0.0, 941.6925306531796, 8.692286703552707e-16, -0.02600072560457681),
    (0.0, 0.8935769662791675, 0.8101238593535642, -2.3389279284062102e-17),
    (0.0, 3.957678419314858, -0.39960203885530415, -4.3331064642935194e-17),
    (0.0, 60.47772516422348, -0.10259706098256305, -3.116647178653208e-16),
    (0.0, 940.1217345481715, -0.02602243810181678, 4.4227196906619663e-16),
    (0.5, 3.141592653589793, 5.512847474009682e-17, 0.45015815807855303),
    (0.5, 6.283185307179586, -7.796343665038751e-17, -0.3183098861837907),
    (0.5, 62.83185307179586, -2.465420340294731e-16, -0.10065842420897407),
    (0.5, 942.4777960769379, -1.324191790744235e-15, -0.025989893374455873),
    (0.5, 4.71238898038469, -0.36755259694786135, 6.751831670557435e-17),
    (0.5, 61.26105674500097, -0.10194074882503562, -2.998086862518401e-16),
    (0.5, 940.9069997501431, -0.026011578729396218, -2.935006869280009e-16),
    (1.0, 3.8317059702075125, -6.149807356994906e-17, 0.41251739515882574),
    (1.0, 7.015586669815619, 2.825339409478929e-17, -0.30236208914434987),
    (1.0, 63.61135669848123, -1.7669881864973399e-16, -0.10004441540942134),
    (1.0, 943.2627966843023, -4.5746656633921745e-16, -0.02597908196509135),
    (1.0, 2.197141326031017, 0.5563684384218895, 2.513306678922122e-17),
    (1.0, 5.429681040794135, -0.34452506735398464, 1.4165786380203692e-16),
    (1.0, 62.040411147670696, -0.10130336645134913, -2.4255895646411714e-16),
    (1.0, 941.6919996943615, -0.02600074026468792, 1.2899556986391889e-15),
    (2.5, 9.095011330476355, -7.629939763902318e-18, -0.26949441809887503),
    (2.5, 65.92794150295865, 8.858273062465527e-17, -0.0983003881738278),
    (2.5, 945.6162161961014, -7.244800432166164e-16, -0.025946771927841478),
    (2.5, 3.9595279165010955, 0.4443364724009859, 4.556657243370653e-17),
    (2.5, 7.451610064214504, -0.30049805932384743, -4.4844496803163537e-17),
    (2.5, 64.35603372774236, -0.09949534713883523, -3.31223448098484e-16),
    (2.5, 944.0454145905115, -0.025968349578854907, 7.081519457853298e-16),
    (10.0, 14.475500686554541, -4.2222406802541416e-17, 0.24538837500296573),
    (10.0, 18.43346366696658, -2.409125566486351e-16, -0.20253888106008025),
    (10.0, 77.1067342468613, -1.3884920223748112e-16, -0.09124949661752954),
    (10.0, 957.3482637086654, 4.4047819717348346e-17, -0.025787954890758958),
    (10.0, 12.128927704415439, 0.29713912370722734, -8.918173475984897e-17),
    (10.0, 16.522284394784425, -0.21955859534393424, 2.854514713816143e-16),
    (10.0, 75.52231273850036, -0.09221838864859636, -2.560699814975261e-16),
    (10.0, 955.7773817548405, -0.025809140586777957, -9.146303746456285e-16),
    (30.0, 36.09833695674772, 2.2083181049886957e-16, 0.1771691662747889),
    (30.0, 41.092778663153425, -3.1675819670098036e-16, -0.15040131385317868),
    (30.0, 104.84989742653606, -4.324649610450311e-16, -0.07960225500215065),
    (30.0, 988.3610802852819, -8.612582483344021e-17, -0.02538529958733733),
    (30.0, 32.97879227445963, 0.2103272981470628, 1.3976738990563101e-16),
    (30.0, 38.71571834916654, -0.16095753136731394, 3.4504208507353673e-16),
    (30.0, 103.20943492269424, -0.08028975081254086, -2.309143574413192e-16),
    (30.0, 986.7895589025694, -0.02540552401470057, -1.3538512640492963e-16),
    (100.0, 108.83616589840977, -1.763995858069231e-18, 0.12112024959618509),
    (100.0, 115.73935123918876, 1.3051518954955542e-16, -0.10440888044777077),
    (100.0, 192.51777030049632, -5.128114463501077e-16, -0.06220736400150672),
    (100.0, 1094.1994073907124, -9.689447590145117e-17, -0.024171435859919356),
    (100.0, 104.38020425686611, 0.142301513463502, 2.345328790375787e-16),
    (100.0, 112.4859648302681, -0.11094570572441316, 1.2455709746517386e-16),
    (100.0, 190.67627547470977, -0.0626195774847141, -2.94616092933338e-16),
    (100.0, 1092.6220003717335, -0.024189024761628253, 2.6876519243914225e-15),
    (-0.3, 1.9228540150659374, -2.6874670848991233e-17, 0.5705790898328811),
    (-0.3, 5.042125633579608, 1.4821249841713562e-16, -0.354801960030825),
    (-0.3, 61.576515031230926, -8.921908520622531e-17, -0.10167821860447111),
    (-0.3, 0.42377215690603737, 1.149421017524423, -1.5387174201826995e-17),
    (-0.3, 3.4779469293073153, -0.42656229501431037, 1.9811198884424578e-17),
    (-0.3, 60.00575269916958, -0.10300037245092733, -7.131620969271681e-17),
    (-2.5, 3.9595279165010955, -4.556657243370653e-17, 0.4443364724009859),
    (-2.5, 7.451610064214504, 4.4844496803163537e-17, -0.30049805932384743),
    (-2.5, 64.35603372774236, 3.31223448098484e-16, -0.09949534713883523),
    (-2.5, 5.76345919689455, -0.34833142749597434, 4.0664393837162815e-17),
    (-2.5, 9.095011330476355, 0.26949441809887503, -7.629939763902318e-18),
    (-2.5, 65.92794150295865, 0.0983003881738278, 8.858273062465527e-17),
    (-7.7, 8.597857022469224, 1.1266152852816865e-16, -0.37490009313631706),
    (-7.7, 13.074207399502276, 1.3806142522865612e-16, 0.2447717837669003),
    (-7.7, 71.5294171642665, 7.145274759297878e-17, 0.09461445239318123),
    (-7.7, 11.046923363176187, 0.2811355296756186, 1.0079424296300159e-16),
    (-7.7, 14.949071641961222, -0.22259237090696013, 1.115669780410691e-16),
    (-7.7, 73.10915371328706, -0.09357499382975917, -5.827153617555984e-16),
]


def _neighbours(x, n=4):
    out, lo, hi = [x], x, x
    for _ in range(n):
        lo, hi = np.nextafter(lo, -np.inf), np.nextafter(hi, np.inf)
        out += [lo, hi]
    return out


def test_real_order_at_the_zeros():
    # At the float64 nearest a zero the ratio J'/J is infinite: the value is 0 or a few 1e-17,
    # never NaN, and within 1e-14 of the envelope M = sqrt(J^2 + Y^2) (docs/api/special.md §8.2).
    for v, x, j, y in AT_ZEROS:
        bound = 1e-14 * np.hypot(j, y)
        for name, want in (("jv", j), ("yv", y), ("jve", j), ("yve", y)):
            got = getattr(S, name)(v, x)
            assert abs(got - want) <= bound, (name, v, x, got, want)
    assert S.jv(0.0, 2.404825557695773) == 0.0 and S.jve(0.0, 2.404825557695773) == 0.0
    assert S.yv(0.0, 2.404825557695773) == pytest.approx(0.509924383448479, rel=1e-14)


def test_real_order_around_the_zeros():
    # the 4 doubles on each side of every zero above: no NaN, the scalar call's bits, and
    # J_v, Y_v next to their values at the middle point (4 ULP of x move them by ~1e-15 M; a
    # wrong sign of the one that is not at its zero would be 2 M away)
    for v, x, j, y in AT_ZEROS:
        xs = np.array(sorted(_neighbours(x)))
        vs = sd.asarray(np.full(xs.size, v))
        for name in ("jv", "yv", "jve", "yve", "iv", "kv", "ive", "kve"):
            got = np.asarray(getattr(S, name)(vs, sd.asarray(xs)))
            # iv overflows past x ~ 713; nothing else leaves the finite numbers
            assert not np.isnan(got).any() and (name == "iv" or np.isfinite(got).all()), (name, v, x, got)
            for g, xi in zip(got, xs):
                assert same_bits(g, getattr(S, name)(v, float(xi))), (name, v, xi)
        m = np.hypot(j, y)
        assert np.all(np.abs(np.asarray(S.jv(vs, sd.asarray(xs))) - j) <= 1e-10 * m), (v, x)
        assert np.all(np.abs(np.asarray(S.yv(vs, sd.asarray(xs))) - y) <= 1e-10 * m), (v, x)


def test_real_order_zeros_same_bits_for_any_thread_count():
    before = sd.get_num_threads()
    xs = np.array([t for _, x, _, _ in AT_ZEROS for t in _neighbours(x)] * 300)
    vs = np.array([v for v, _, _, _ in AT_ZEROS for _ in range(9)] * 300)
    try:
        runs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            runs.append([np.asarray(getattr(S, n)(sd.asarray(vs), sd.asarray(xs))).view(np.uint64)
                         for n in ("jv", "yv", "jve", "yve")])
    finally:
        sd.set_num_threads(before)
    for r in runs[1:]:
        for a, b in zip(r, runs[0]):
            np.testing.assert_array_equal(a, b)
