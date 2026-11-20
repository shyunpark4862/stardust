"""Complex arguments (docs/api/special.md §12): wofz, loggamma, and erf, erfc, erfcx and
lambertw of a complex argument.

Accuracy is tests/special/test_reference.py's (the files wofz, *_complex, loggamma and
lambertw_c*). Here: the special values against scipy except where docs/design/special.md
§2.4 says otherwise (true values and limits), the sides of the branch
cuts (the sign of a zero imaginary part), the result types, and lambertw's k."""

import itertools
import math

import numpy as np
import pytest
import scipy.special as sp

import stardust as sd

S = sd.special
INF, NAN = math.inf, math.nan
EDGE = [0.0, -0.0, 1.0, -1.0, INF, -INF, NAN, 5e-324]


def close_or_same(a, b, zero_sign=True):
    """Both finite and within 1e-13 of each other in modulus (the error measure of docs/api/special.md §12; zero
    parts must agree in sign unless zero_sign=False), or part by part the same NaN or infinity."""
    a, b = complex(a), complex(b)
    if all(map(math.isfinite, (a.real, a.imag, b.real, b.imag))):
        if zero_sign:
            for u, v in ((a.real, b.real), (a.imag, b.imag)):
                if u == 0 and v == 0 and math.copysign(1, u) != math.copysign(1, v):
                    return False
        return abs(a - b) <= 1e-13 * max(abs(a), abs(b))

    def part(u, v):
        if math.isnan(u) or math.isnan(v):
            return math.isnan(u) and math.isnan(v)
        return u == v and (not zero_sign or math.copysign(1, u) == math.copysign(1, v))

    return part(a.real, b.real) and part(a.imag, b.imag)


def close(a, b, tol=1e-14):
    return abs(complex(a) - complex(b)) <= tol * abs(complex(b))


def test_result_types():
    assert isinstance(S.wofz(1.0), complex)  # w of a real argument is complex (scipy)
    assert S.wofz(sd.asarray(np.array([0.5, 2.0]))).dtype == sd.complex128
    assert isinstance(S.erf(0.5), float) and isinstance(S.erf(0.5 + 0j), complex)
    assert S.erfc(sd.asarray(np.array([1 + 1j]))).dtype == sd.complex128
    assert isinstance(S.loggamma(2.5), float) and isinstance(S.loggamma(2.5 + 0j), complex)
    assert S.loggamma(sd.asarray(np.array([0.5, 3.0]))).dtype == sd.float64
    assert isinstance(S.lambertw(0.5), float) and isinstance(S.lambertw(0.5 + 0j), complex)
    with pytest.raises(TypeError, match=r"^erfinv\(\): not supported for complex128$"):
        S.erfinv(1j)


@pytest.mark.parametrize("name", ["wofz", "erf", "erfc", "erfcx"])
def test_erf_family_edges_match_scipy(name):
    # up to the sign of zero parts: scipy's is not consistent there (erfc(0+0j) has Im -0,
    # erfc(0.5+0j) +0); ours follows the derivative, below
    f, g = getattr(S, name), getattr(sp, name)
    for x, y in itertools.product(EDGE, EDGE):
        z = complex(x, y)
        assert close_or_same(f(z), g(z), zero_sign=False), (name, z, f(z), g(z))


def test_erf_family_zero_parts():
    # on the real axis Im f(x + iy) ~ y f'(x): the sign of y for erf, of -y for erfc and erfcx
    # (both decrease); w(iy) is real with the sign of Re z (scipy)
    for x in [0.0, 0.5, -3.0, 30.0]:
        for y in [0.0, -0.0]:
            z = complex(x, y)
            assert math.copysign(1, S.erf(z).imag) == math.copysign(1, y)
            assert math.copysign(1, S.erfc(z).imag) == -math.copysign(1, y)
            assert math.copysign(1, S.erfcx(z).imag) == -math.copysign(1, y)
            assert math.copysign(1, S.wofz(complex(y, x)).imag) == math.copysign(1, y)


def test_erf_family_true_values():
    # true values where scipy gives NaN (docs/design/special.md §2.4); and a scipy overflow where the value is finite
    assert S.erf(1e300 + 1e300j) == 1
    assert close(S.erf(5e-324 + 30j).real, 5.391989333430128e67, 1e-13)  # scipy inf
    # the axes are exact part by part: erf(iy) = i erfi(y), erfc(iy) = 1 - i erfi(y), w(iy) = erfcx(y)
    for y in [0.3, 2.0, 25.0, -4.0]:
        e = S.erf(complex(0.0, y))
        assert e.real == 0.0 and close(e.imag, sp.erfi(y))
        assert S.erfc(complex(0.0, y)) == complex(1.0, -e.imag)
        assert S.wofz(complex(0.0, y)) == complex(S.erfcx(y), 0.0)
    for x in [0.3, 2.0, -5.0]:
        assert S.erf(complex(x, 0.0)) == complex(S.erf(x), 0.0)
        assert S.erfc(complex(x, 0.0)) == complex(S.erfc(x), -0.0)


def test_erf_family_symmetries():
    rng = np.random.default_rng(8)
    for z in rng.uniform(-6, 6, 50) + 1j * rng.uniform(-6, 6, 50):
        z = complex(z)
        assert S.erf(-z) == -S.erf(z)
        assert close(S.erf(z.conjugate()), S.erf(z).conjugate(), 1e-15)
        assert close(S.wofz(-z.conjugate()), S.wofz(z).conjugate(), 1e-15)


def test_loggamma_real():
    x = np.array([0.1, 0.5, 1.0, 2.0, 7.5, 1e10, 1e300])
    assert np.array_equal(np.asarray(S.loggamma(sd.asarray(x))), np.asarray(S.gammaln(sd.asarray(x))))
    assert S.loggamma(0.0) == INF and S.loggamma(-0.0) == INF and S.loggamma(INF) == INF  # scipy
    assert all(math.isnan(S.loggamma(v)) for v in [-2.5, -1.0, -INF, NAN])


def test_loggamma_complex_special_values():
    for z in [0j, complex(-0.0, 0.0), complex(0.0, -0.0), -1 + 0j, complex(-1, -0.0), -3 + 0j]:
        v = S.loggamma(z)
        assert v.real == INF and math.isnan(v.imag), z  # a pole: Im depends on the direction (scipy NaN)
    # limits (docs/design/special.md §2.4; scipy NaN)
    assert close_or_same(S.loggamma(complex(INF, 0.0)), complex(INF, 0.0))
    assert close_or_same(S.loggamma(complex(INF, -0.0)), complex(INF, -0.0))
    assert close_or_same(S.loggamma(complex(INF, 1.0)), complex(INF, INF))
    assert close_or_same(S.loggamma(complex(1.0, INF)), complex(-INF, INF))
    assert close_or_same(S.loggamma(complex(1.0, -INF)), complex(-INF, -INF))
    assert close_or_same(S.loggamma(complex(-INF, 1.0)), complex(-INF, -INF))
    assert close_or_same(S.loggamma(complex(INF, INF)), complex(INF, INF))
    for z in [complex(-INF, 0.0), complex(NAN, 0.0), complex(0.0, NAN)]:
        assert all(math.isnan(p) for p in (S.loggamma(z).real, S.loggamma(z).imag)), z
    # the cut: Im = -3 pi on (-3, -2) from above, +3 pi from below (scipy)
    for x in [-2.5, -10.3, -0.5]:
        assert close_or_same(S.loggamma(complex(x, 0.0)), sp.loggamma(complex(x, 0.0)))
        assert close_or_same(S.loggamma(complex(x, -0.0)), sp.loggamma(complex(x, -0.0)))
    assert close(S.loggamma(complex(-2.5, 0.0)).imag, -3 * math.pi)
    assert S.loggamma(1 + 0j) == 0 and S.loggamma(2 + 0j) == 0


def test_loggamma_conjugate():
    rng = np.random.default_rng(9)
    for z in rng.uniform(-20, 20, 60) + 1j * rng.uniform(-10, 10, 60):
        z = complex(z)
        assert close(S.loggamma(z.conjugate()), S.loggamma(z).conjugate(), 1e-15), z


def test_lambertw_true_values():
    # true values where scipy gives inf+nanj or NaN (docs/design/special.md §2.4)
    assert close(S.lambertw(5e-324 + 0j, k=1), -751.0615683230378 + 3.1457810754352713j)
    assert close(S.lambertw(5e-324 + 0j, k=-1), -751.0615683230378 - 3.1457810754352713j)
    for k in [0, 1, -2]:
        assert close_or_same(S.lambertw(complex(0.0, INF), k=k), complex(INF, math.pi / 2 + 2 * math.pi * k))
        assert close_or_same(S.lambertw(complex(INF, 0.0), k=k), complex(INF, 2 * math.pi * k))
        assert close_or_same(S.lambertw(complex(-INF, 0.0), k=k), complex(INF, math.pi + 2 * math.pi * k))
    em = -0.36787944117144233  # the float64 nearest -1/e, 8e-18 below it: on the cut of W_0
    assert close(S.lambertw(complex(em, 0.0)), -1 + 8.220079714836618e-09j, 1e-14)
    assert close(S.lambertw(complex(em, -0.0)), -1 - 8.220079714836618e-09j, 1e-14)
    assert close(S.lambertw(complex(em, 0.0), k=-1), -1 - 8.220079714836618e-09j, 1e-14)
    assert S.lambertw(em) == -1.0  # the real argument: the branch point (docs/design/special.md §2.4)


def test_lambertw_zero_and_nan():
    assert close_or_same(S.lambertw(0j), 0j) and close_or_same(S.lambertw(complex(-0.0, -0.0)), complex(-0.0, -0.0))
    for k in [1, -1, 3]:
        assert close_or_same(S.lambertw(0j, k=k), complex(-INF, 0.0))  # scipy
    assert close_or_same(S.lambertw(complex(NAN, 0.0)), sp.lambertw(complex(NAN, 0.0)))


def test_lambertw_cut_sides():
    # the sign of a zero imaginary part picks the side (C99, docs/design/special.md §2.4)
    for x in [-1.0, -5.0, -1e10]:
        for k in [0, 1, -1, 2]:
            assert close_or_same(S.lambertw(complex(x, 0.0), k=k), sp.lambertw(complex(x, 0.0), k)), (x, k)
            assert close_or_same(S.lambertw(complex(x, -0.0), k=k), sp.lambertw(complex(x, -0.0), k)), (x, k)
    # on (-1/e, 0): W_-1 is real from above; from below it is W_1's conjugate. scipy gives the
    # real value from both sides.
    x = -0.1
    assert S.lambertw(complex(x, 0.0), k=-1) == complex(S.lambertw(x, k=-1), 0.0)
    assert S.lambertw(complex(x, -0.0), k=1) == complex(S.lambertw(x, k=-1), -0.0)
    assert close(S.lambertw(complex(x, -0.0), k=-1), S.lambertw(complex(x, 0.0), k=1).conjugate())
    assert S.lambertw(complex(x, 0.0)) == complex(S.lambertw(x), 0.0)


def test_lambertw_k():
    with pytest.raises(TypeError, match=r"'k' must be an int"):
        S.lambertw(1j, k=1.0)
    with pytest.raises(TypeError, match=r"'k' must be an int"):
        S.lambertw(1j, k=True)
    with pytest.raises(ValueError, match=r"'k' must be 0 or -1 for a real argument"):
        S.lambertw(0.5, k=1)
    z = sd.asarray(np.array([[1 + 1j, -2.0 + 0j], [0.5j, 3.0 + 0j]]))
    got = np.asarray(S.lambertw(z, k=2))
    assert got.shape == (2, 2)
    assert all(close(got[i, j], sp.lambertw(np.asarray(z)[i, j], 2)) for i in range(2) for j in range(2))
    for k in [0, 1, -1, 5]:
        w = S.lambertw(2 - 3j, k=k)
        assert close(w * np.exp(w), 2 - 3j, 1e-14)
