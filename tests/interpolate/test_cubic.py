"""The cubic family's construction (docs/design/interpolate.md §3.3, §4; milestone IP2): the node slopes against the
exact ones (tests/interpolate/ref/slopes.npz, exact rationals rounded once) in units of their contracts
(tests/interpolate/refdata.py), the one Hermite routine (decisions IP15, IP42), the boundary conditions' special
cases, batches, axes and complex values, against scipy 1.18.1 where the behaviour is not rounding."""

from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
CORE = sd._core.interpolate
SLOPES = refdata.ref("slopes")
U = refdata.U
BC = {"natural": ((2, 0.0), (2, 0.0)), "clamped": ((1, 0.0), (1, 0.0)), "not_a_knot": ("not-a-knot", "not-a-knot"),
      "periodic": ("periodic", "periodic")}  # fmt: skip
KIND = {"not-a-knot": 0, "periodic": 3}


def _ref_slopes(method, cid):
    m = SLOPES.methods[method]
    pos = np.flatnonzero(SLOPES.inputs["set"][m.index] == cid)
    idx = m.index[pos]
    order = np.argsort(SLOPES.inputs["index"][idx], kind="stable")
    return m.hi[pos[order]], m.lo[pos[order]]


def _native_spline_slopes(x, y, bc):
    args = []
    for b in bc:
        if isinstance(b, str):
            args += [KIND[b], None]
        else:
            args += [b[0], sd.asarray([float(b[1])])]
    s, bad = CORE.spline_slopes(sd.asarray(x), sd.asarray(y.reshape(-1, 1).copy()), *args)
    assert bad == -1
    return np.asarray(s)[:, 0]


def _finite_curves():
    """The real curves whose divided differences are doubles (subnormal/x's overflow: the IP36 test)."""
    return [c for c in S.real_curve_ids() if np.all(np.isfinite(np.diff(S.curve(c).y) / np.diff(S.curve(c).x)))]


def _units(got, hi, lo, bound):
    err = np.abs((got - hi) - lo)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = np.where(err == 0, 0.0, err / bound)
    return u


@pytest.mark.parametrize("method", ["natural", "clamped", "not_a_knot", "periodic"])
def test_spline_slopes_contract(method):
    worst = 0.0
    for cid in _finite_curves():
        c = S.curve(cid)
        if method == "periodic" and "periodic" not in c.tags:
            continue
        got = _native_spline_slopes(c.x, c.y, BC[method])
        hi, lo = _ref_slopes(method, cid)
        u = _units(got, hi, lo, refdata.spline_bound(c.x, c.y, BC[method], got))
        assert np.all(u <= 1), (cid, float(u.max()))
        worst = max(worst, float(u.max()))
        if method in ("natural", "clamped", "periodic"):  # the corollary of diagonal dominance (§3.3)
            M = np.max(np.abs(np.diff(c.y) / np.diff(c.x)))
            assert np.all(np.abs((got - hi) - lo) <= 150 * U * M), cid
    assert worst > 0


def test_pchip_slopes_contract():
    for cid in _finite_curves():
        c = S.curve(cid)
        got = np.asarray(CORE.pchip_slopes(sd.asarray(c.x), sd.asarray(c.y.reshape(-1, 1).copy())))[:, 0]
        hi, lo = _ref_slopes("pchip", cid)
        u = _units(got, hi, lo, refdata.pchip_bound(c.x, c.y))
        assert np.all(u <= 1), (cid, float(u.max()))


@pytest.mark.parametrize("method", ["akima", "makima"])
def test_akima_slopes_contract(method):
    for cid in _finite_curves():
        c = S.curve(cid)
        got = np.asarray(CORE.akima_slopes(sd.asarray(c.x), sd.asarray(c.y.reshape(-1, 1).copy()),
                                           method == "makima"))[:, 0]  # fmt: skip
        hi, lo = _ref_slopes(method, cid)
        u = _units(got, hi, lo, refdata.akima_bound(c.x, c.y, method == "makima"))
        assert np.all(u <= 1), (cid, float(u.max()))


def test_the_classes_use_these_slopes_and_one_hermite_routine():
    """IP15: the same slopes give the same coefficients, whatever the class."""
    for cid in _finite_curves():
        c = S.curve(cid)
        if cid == "large/1e300":
            continue  # the coefficients overflow there (IP36)
        objs = {"not_a_knot": IP.CubicSpline(c.x, c.y), "natural": IP.CubicSpline(c.x, c.y, bc_type="natural"),
                "pchip": IP.PchipInterpolator(c.x, c.y), "akima": IP.Akima1DInterpolator(c.x, c.y),
                "makima": IP.Akima1DInterpolator(c.x, c.y, method="makima")}  # fmt: skip
        for meth, obj in objs.items():
            cc = np.asarray(obj.c)
            # the slopes: c1 at the nodes but the last; the last node's from the derivative there
            s = np.append(cc[2], obj(c.x[-1], nu=1))
            herm = IP.CubicHermiteSpline(c.x, c.y, s)
            same = np.asarray(herm.c)
            assert np.array_equal(same[:, :-1], cc[:, :-1]), (cid, meth)


def test_hermite_coefficients_contract():
    for cid in _finite_curves():
        c = S.curve(cid)
        if cid == "large/1e300":
            continue
        obj = IP.CubicSpline(c.x, c.y, bc_type="natural")
        cc = np.asarray(obj.c)
        s = np.append(cc[2], np.nan)
        s[-1] = _native_spline_slopes(c.x, c.y, BC["natural"])[-1]
        for i in range(c.n - 1):
            X0, X1, Y0, Y1 = (Fraction(float(v)) for v in (c.x[i], c.x[i + 1], c.y[i], c.y[i + 1]))
            S0, S1 = Fraction(float(s[i])), Fraction(float(s[i + 1]))
            h = X1 - X0
            D = (Y1 - Y0) / h
            c3 = (S0 + S1 - 2 * D) / h**2
            c2 = (3 * D - 2 * S0 - S1) / h
            b3, b2 = refdata.hermite_bound(float(h), s[i], s[i + 1], float(D))
            assert abs(Fraction(float(cc[0, i])) - c3) <= Fraction(b3), (cid, i)
            assert abs(Fraction(float(cc[1, i])) - c2) <= Fraction(b2), (cid, i)
            assert cc[2, i] == s[i] and cc[3, i] == c.y[i]


def test_overflowing_coefficients_are_an_error_naming_the_interval():
    """IP36: scipy's large/1e300 spline has coefficients past 1e308 and NaN even at the nodes; sd refuses it."""
    c = S.curve("large/1e300")
    with pytest.raises(ValueError, match=r"^CubicSpline\(\): the coefficients overflow on interval \d+, "):
        IP.CubicSpline(c.x, c.y, bc_type="clamped")
    c = S.curve("subnormal/x")  # the divided differences themselves overflow
    for cls in (IP.CubicSpline, IP.PchipInterpolator, IP.Akima1DInterpolator):
        with pytest.raises(ValueError, match=r"the coefficients overflow on interval 0, "):
            cls(c.x, c.y)


@LIVE
def test_special_cases_follow_scipy():
    x2, y2 = np.array([0.0, 1.0]), np.array([2.0, 7.0])
    for bc in ("not-a-knot", "natural", "clamped", "periodic"):
        yy = y2 if bc != "periodic" else np.array([2.0, 2.0])
        mine, theirs = IP.CubicSpline(x2, yy, bc_type=bc), si.CubicSpline(x2, yy, bc_type=bc)
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-15, atol=1e-15), bc
        assert mine.extrapolate == theirs.extrapolate
    x3 = np.array([0.0, 1.0, 3.0])
    for bc in ("not-a-knot", ("not-a-knot", "natural"), ("clamped", "not-a-knot"), "periodic", ((2, 1.5), (1, -2.0))):
        y3 = np.array([0.0, 1.0, 0.0])
        mine, theirs = IP.CubicSpline(x3, y3, bc_type=bc), si.CubicSpline(x3, y3, bc_type=bc)
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-14, atol=1e-15), bc
    for n in (2, 3):
        x, y = np.arange(n, dtype=float), np.array([0.0, 1.0, 3.0])[:n]
        assert np.allclose(np.asarray(IP.PchipInterpolator(x, y).c), si.PchipInterpolator(x, y).c, atol=1e-15)
        for meth in ("akima", "makima"):
            assert np.allclose(np.asarray(IP.Akima1DInterpolator(x, y, method=meth).c),
                               si.Akima1DInterpolator(x, y, method=meth).c, atol=1e-15)  # fmt: skip


@LIVE
def test_pchip_and_akima_are_scipys_bits():
    """The same formulas in the same order of operations: the same coefficients as scipy, bit for bit."""
    for cid in _finite_curves():
        c = S.curve(cid)
        if cid == "large/1e300":
            continue
        assert np.array_equal(np.asarray(IP.PchipInterpolator(c.x, c.y).c), si.PchipInterpolator(c.x, c.y).c), cid
        for meth in ("akima", "makima"):
            a = np.asarray(IP.Akima1DInterpolator(c.x, c.y, method=meth).c)
            assert np.array_equal(a, si.Akima1DInterpolator(c.x, c.y, method=meth).c), (cid, meth)


@LIVE
def test_batches_axes_and_complex_values():
    g = np.random.default_rng(7)
    x = np.sort(g.random(9))
    y = g.standard_normal((2, 9, 3))
    q = g.random((4, 5)) * 1.4 - 0.2
    for bc in ("not-a-knot", "natural", "clamped", ((1, np.ones((2, 3))), (2, np.zeros((2, 3))))):
        mine = IP.CubicSpline(x, y, axis=1, bc_type=bc)
        theirs = si.CubicSpline(x, y, axis=1, bc_type=bc)
        got, want = np.asarray(mine(q)), theirs(q)
        assert got.shape == want.shape == (2, 4, 5, 3)
        assert np.allclose(got, want, rtol=1e-13, atol=1e-13)
        assert np.asarray(mine.c).shape == theirs.c.shape
        # each column alone gives the same bits as in the batch
        one = IP.CubicSpline(x, y[1, :, 2], bc_type=bc if isinstance(bc, str) else ((1, 1.0), (2, 0.0)))
        assert np.array_equal(np.asarray(one(q)), got[1, :, :, 2])
    yc = y[0] + 1j * y[1]
    mine, theirs = IP.CubicSpline(x, yc), si.CubicSpline(x, yc)
    assert np.allclose(np.asarray(mine(q)), theirs(q), rtol=1e-13, atol=1e-13)
    real = IP.CubicSpline(x, y[0])
    assert np.array_equal(np.asarray(mine.c).real, np.asarray(real.c))  # real and imaginary parts apart
    mine = IP.CubicSpline(x, y[0, :, 0], bc_type=((1, 1j), "natural"))
    assert np.asarray(mine.c).dtype == np.complex128
    for cls in (IP.PchipInterpolator, IP.Akima1DInterpolator):
        assert np.allclose(np.asarray(cls(x, y, axis=1)(q)), getattr(si, cls.__name__)(x, y, axis=1)(q),
                           rtol=1e-13, atol=1e-13, equal_nan=True)  # fmt: skip


def test_periodic_tolerance_is_scipys():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    ok = IP.CubicSpline(x, [1.0, 2.0, 0.0, 1.0 + 1e-16], bc_type="periodic")
    assert ok.extrapolate == "periodic"
    with pytest.raises(ValueError, match="for bc_type='periodic'"):
        IP.CubicSpline(x, [1.0, 2.0, 0.0, 1.0 + 1e-14], bc_type="periodic")


def test_defaults_of_extrapolate():
    x, y = np.arange(5.0), np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    assert IP.CubicSpline(x, y).extrapolate is True
    assert IP.CubicSpline(x, y, bc_type="periodic").extrapolate == "periodic"
    assert IP.PchipInterpolator(x, y).extrapolate is True
    assert IP.Akima1DInterpolator(x, y).extrapolate is False
    assert np.isnan(IP.Akima1DInterpolator(x, y)(5.0))
    assert IP.CubicHermiteSpline(x, y, y).extrapolate is True


def test_coefficients_are_read_only_copies():
    x, y = np.arange(5.0), np.array([0.0, 1.0, 0.0, 2.0, 0.0])
    p = IP.CubicSpline(x, y)
    with pytest.raises(ValueError):
        p.c[0, 0] = 1.0
    with pytest.raises(ValueError):
        p.x[0] = 1.0
    xs = sd.asarray(x)
    q = IP.PPoly(np.asarray(p.c), xs)
    xs[0] = -5.0
    assert float(q.x[0]) == 0.0


def test_pchip_interpolate_is_the_object_called():
    x, y = np.arange(6.0), np.array([0.0, 1.0, 1.5, 1.5, 3.0, 2.0])
    q = np.linspace(-1, 6, 29)
    p = IP.PchipInterpolator(x, y)
    assert np.array_equal(np.asarray(IP.pchip_interpolate(x, y, q)), np.asarray(p(q)))
    d = IP.pchip_interpolate(x, y, q, der=[0, 2])
    assert isinstance(d, list) and np.array_equal(np.asarray(d[1]), np.asarray(p(q, nu=2)))
    assert IP.pchip_interpolate(x, y, 2.5) == p(2.5)
