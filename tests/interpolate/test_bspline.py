"""BSpline (docs/api/interpolate.md §4; docs/design/interpolate.md §2.5, §3.4, §4; milestone IP3): the evaluation
contract against the exact value of the computed coefficients (refdata.bspline_bound) for every degree, derivative
order and outside rule; k = 1 is sd.interp's linear kernel; the calculus (derivative, antiderivative, integrate)
against exact rationals and scipy 1.18.1; basis_element, design_matrix, insert_knot, from_power_basis and
PPoly.from_spline; the rules where sd differs from scipy (decisions IP52-IP61)."""

import math
import time
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate


def _splines():
    """(curve, method, spline) for every real curve and make_interp_spline method of bspline.npz it allows."""
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        for method in refdata.BSPLINE_METHODS:
            k, bc = refdata.bspline_args(method)
            if bc == "periodic" and "periodic" not in c.tags:
                continue
            try:
                b = IP.make_interp_spline(c.x, c.y, k=k, bc_type=bc)
            except ValueError:
                continue  # too few points for k, or subnormal/x's overflowing end derivatives
            yield c, method, b


def test_evaluation_contract():
    """Every degree and derivative order (nu = 0 .. k + 1) at every curve's evaluation points, the end pieces
    extended outside: |y^ - sum c_j B_j^(nu)(x)| within gamma_{6k+1} sum |c_j| B~_j + floor (k = 1, nu = 0 is the
    linear kernel's: test_reference.py)."""
    worst = 0.0
    for c, method, b in _splines():
        t, cc, k = np.asarray(b.t), np.asarray(b.c), b.k
        pts = c.xe[np.isfinite(c.xe)][::2]
        for nu in range(k + 2):
            if k == 1 and nu == 0:
                continue
            got = np.asarray(b(pts, nu=nu, extrapolate=True))
            for x, g in zip(pts, got):
                want = refdata.bspline_exact(t, cc, k, x, nu)
                bound = refdata.bspline_bound(t, cc, k, x, nu)
                if bound == math.inf:  # the terms overflow (large/1e300's high derivatives): no finite contract
                    continue
                assert math.isfinite(g), (c.id, method, nu, x)
                err = abs(Fraction(float(g)) - want)
                if err:
                    u = float(err) / bound
                    assert u <= 1, (c.id, method, nu, float(x), u)
                    worst = max(worst, u)
    assert worst > 0


def test_k1_is_the_linear_kernel():
    """§1.4: make_interp_spline(xp, fp, k=1) and BSpline(t, c, 1) give sd.interp's bits on [xp[0], xp[-1]] and the
    linear path's outside; at the nodes the data itself (scipy's de Boor misses some)."""
    lin = sd._core.interpolate.linear
    for cid in S.curve_ids():
        c = S.curve(cid)
        if c.n < 2 or np.asarray(c.y).ndim != 1:
            continue
        b = IP.make_interp_spline(c.x, c.y, k=1)
        inside = c.xe[(c.xe >= c.x[0]) & (c.xe <= c.x[-1])]
        want = np.asarray(sd.interp(sd.asarray(inside), sd.asarray(c.x), sd.asarray(c.y)))
        assert np.array_equal(np.asarray(b(inside)), want, equal_nan=True), cid
        assert np.array_equal(np.asarray(b(c.x)), c.y), cid
        col = sd.asarray(np.ascontiguousarray(np.asarray(c.y).reshape(-1, 1)))
        for ext, rule in ((True, 1), (False, 0), ("periodic", 2)):
            want = np.asarray(lin(sd.asarray(c.xe), sd.asarray(c.x), col, rule))[:, 0]
            assert np.array_equal(np.asarray(b(c.xe, extrapolate=ext)), want, equal_nan=True), (cid, ext)
        same = IP.BSpline(b.t, b.c, 1)
        assert np.array_equal(np.asarray(same(c.xe)), np.asarray(b(c.xe)), equal_nan=True)
    # repeated inner knots (a jump): de Boor, continuous from the right
    jump = IP.BSpline([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [0.0, 1.0, 3.0, 4.0], 1)
    assert np.asarray(jump([0.5, 1.0, 1.5])).tolist() == [0.5, 3.0, 3.5]


@LIVE
def test_scipy_k1_misses_nodes():
    c = S.curve("random/50")
    assert np.sum(si.make_interp_spline(c.x, c.y, k=1)(c.x) != c.y) > 0
    assert np.sum(np.asarray(IP.make_interp_spline(c.x, c.y, k=1)(c.x)) != c.y) == 0


def test_outside_rules_nan_points_and_high_orders():
    """§2.3: False is NaN outside [t[k], t[n]], 'periodic' wraps with numpy's remainder; a NaN point is NaN; nu > k
    is 0 except where the value itself is NaN (IP52)."""
    x = np.array([0.0, 0.5, 1.5, 2.0, 3.0, 4.0])
    b = IP.make_interp_spline(x, np.sin(x))
    q = np.array([-1.0, 0.0, 2.2, 4.0, 4.5, math.nan])
    f = np.asarray(b(q, extrapolate=False))
    assert np.isnan(f[[0, 4, 5]]).all() and np.isfinite(f[1:4]).all()
    w = 0.0 + np.remainder(q - 0.0, 4.0)
    assert np.array_equal(np.asarray(b(q, extrapolate="periodic")), np.asarray(b(w)), equal_nan=True)
    hi = np.asarray(b(q, nu=4, extrapolate=False))
    assert hi[1:4].tolist() == [0.0, 0.0, 0.0] and np.isnan(hi[[0, 4, 5]]).all()
    assert np.asarray(b(q, nu=7))[:5].tolist() == [0.0] * 5
    assert math.isnan(b(math.nan))
    assert isinstance(b(1.0), float) and isinstance(IP.make_interp_spline(x, x + 1j)(1.0), complex)
    per = IP.BSpline(b.t, b.c, 3, extrapolate="periodic")
    assert np.array_equal(np.asarray(per(q)), np.asarray(b(w)), equal_nan=True)


def test_evaluation_order_does_not_matter_and_search_is_logarithmic():
    """§3.1: any order of the points gives the same values; 10^5 random points on 10^6 knots take milliseconds
    (scipy walks from the previous interval: minutes at this size)."""
    g = np.random.default_rng(41)
    n = 10**6
    x = np.arange(n) / (n - 1)
    b = IP.make_interp_spline(x, g.standard_normal(n))
    q = g.random(10**5)
    perm = g.permutation(q.size)
    srt = np.argsort(q)
    v = np.asarray(b(q))
    assert np.array_equal(np.asarray(b(q[perm])), v[perm]) and np.array_equal(np.asarray(b(q[srt])), v[srt])
    t0 = time.perf_counter()
    b(q)
    assert time.perf_counter() - t0 < 0.5


def _exact_integral(b, lo, hi):
    """The integral of one column of b over [lo, hi] (inside one extension rule: the end pieces), exactly."""
    t, c, k = np.asarray(b.t), np.asarray(b.c), b.k
    a = b.antiderivative()
    return refdata.bspline_exact(np.asarray(a.t), np.asarray(a.c), a.k, hi) - refdata.bspline_exact(
        np.asarray(a.t), np.asarray(a.c), a.k, lo
    ), (t, c, k)


def test_calculus_against_exact_rationals():
    """The derivative's and antiderivative's coefficients are scipy's formulas (splder, splantider): the derivative
    of the antiderivative is the spline again (to rounding), and the antiderivative is zero at t[0] and continuous."""
    g = np.random.default_rng(42)
    x = np.sort(g.random(14))
    for k in range(6):
        b = IP.make_interp_spline(x, g.standard_normal(14), k=k)
        a = b.antiderivative(nu=2)
        assert a.k == k + 2 and np.asarray(a.t).shape[0] == np.asarray(b.t).shape[0] + 4
        q = np.linspace(x[0], x[-1], 37)
        if k:  # k = 0's last knot is double: its antiderivative's end knots repeat k + 2 times (scipy: the same error)
            back = a.derivative(nu=2)
            assert np.allclose(np.asarray(back(q)), np.asarray(b(q)), rtol=1e-12, atol=1e-12)
        else:
            with pytest.raises(ValueError, match="repeated internal knots"):
                a.derivative(nu=2)
        assert a(float(a.t[0])) == 0.0
        for nu in range(1, k + 1):
            d = b.derivative(nu=nu)
            assert np.allclose(np.asarray(d(q)), np.asarray(b(q, nu=nu)), rtol=1e-10, atol=1e-10 * 10**nu)
        # integrate: F(b) - F(a) of the antiderivative, against the exact value of the computed antiderivative
        F = b.antiderivative()
        for lo, hi in ((x[1], x[-2]), (x[0] - 0.1, x[-1] + 0.2)):
            want = refdata.bspline_exact(F.t, F.c, F.k, hi) - refdata.bspline_exact(F.t, F.c, F.k, lo)
            got = b.integrate(lo, hi)
            assert abs(Fraction(got) - want) <= 1e-14 * max(1.0, abs(float(want))), (k, lo, hi)


@LIVE
def test_calculus_follows_scipy():
    g = np.random.default_rng(43)
    x = np.sort(g.random(12))
    y = g.standard_normal((12, 2))
    yp = np.r_[y[:-1], y[:1]]
    for k in (0, 1, 2, 3, 5):
        mine, theirs = IP.make_interp_spline(x, y, k=k), si.make_interp_spline(x, y, k=k)
        for nu in range(1, k + 1):
            d1, d2 = mine.derivative(nu=nu), theirs.derivative(nu)
            assert np.array_equal(np.asarray(d1.t), d2.t) and d1.k == d2.k
            assert np.allclose(np.asarray(d1.c), d2.c, rtol=1e-12, atol=1e-12)
        for nu in (1, 2):
            a1, a2 = mine.antiderivative(nu=nu), theirs.antiderivative(nu)
            assert np.array_equal(np.asarray(a1.t), a2.t) and a1.k == a2.k
            assert np.allclose(np.asarray(a1.c), a2.c, rtol=1e-12, atol=1e-13)
        for lo, hi in ((0.2, 0.7), (-0.5, 1.4), (0.9, 0.1)):
            for ext in (True, False):
                assert np.allclose(np.asarray(mine.integrate(lo, hi, extrapolate=ext)),
                                   theirs.integrate(lo, hi, extrapolate=ext), rtol=1e-12, atol=1e-13)  # fmt: skip
        if k >= 2:
            mp = IP.make_interp_spline(x, yp, k=k, bc_type="periodic")
            tp = si.make_interp_spline(x, yp, k=k, bc_type="periodic")
            for lo, hi in ((-2.3, 3.1), (0.1, 0.5), (0.8, 1.6)):
                assert np.allclose(np.asarray(mp.integrate(lo, hi)), tp.integrate(lo, hi), rtol=1e-11, atol=1e-12)
            assert mp.antiderivative().extrapolate is False  # scipy
    # one column with scipy's splint path: both ends outside on one side integrate nothing
    one = IP.make_interp_spline(x, y[:, 0])
    assert one.integrate(-3.0, -1.0, extrapolate=False) == 0.0 == si.make_interp_spline(x, y[:, 0]).integrate(
        -3.0, -1.0, extrapolate=False)  # fmt: skip
    assert np.asarray(IP.make_interp_spline(x, y).integrate(-3.0, -1.0, extrapolate=False)).tolist() == [0.0, 0.0]
    assert math.isnan(one.integrate(math.nan, 1.0))  # IP49's rule


def test_derivative_errors_follow_scipy():
    b = IP.make_interp_spline(np.arange(6.0), np.arange(6.0) ** 2)
    with pytest.raises(ValueError, match=r"^derivative\(\): 'nu' must be at most the degree k = 3, not 4"):
        b.derivative(nu=4)
    rep = IP.BSpline([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], np.arange(6.0), 2)
    with pytest.raises(ValueError, match="repeated internal knots"):
        rep.derivative(nu=2)
    assert b.derivative(nu=-1).k == 4 and b.antiderivative(nu=-1).k == 2 and b.derivative(nu=0).k == 3


@LIVE
def test_basis_element_and_design_matrix():
    for t in ([0, 1, 2, 3, 4], [0, 1, 1, 2], [0, 1], [0.0, 0.5, 2.0]):
        mine, theirs = IP.BSpline.basis_element(t), si.BSpline.basis_element(t)
        assert np.array_equal(np.asarray(mine.t), theirs.t) and np.array_equal(np.asarray(mine.c), theirs.c)
        q = np.linspace(-1, 5, 25)
        assert np.allclose(np.asarray(mine(q)), theirs(q), atol=1e-15)
    assert IP.BSpline.basis_element([0, 1, 2], extrapolate=False).extrapolate is False
    t = np.r_[0, 0, 0, 0, 1, 2, 3, 3, 3, 3.0]
    x = np.array([0.0, 0.5, 1.0, 2.5, 3.0])
    mine, theirs = IP.BSpline.design_matrix(x, t, 3), si.BSpline.design_matrix(x, t, 3)
    assert isinstance(mine, sd.sparse.csr_array) and mine.shape == theirs.shape == (5, 6)
    assert np.array_equal(np.asarray(mine.indices), theirs.indices) and np.array_equal(np.asarray(mine.indptr),
                                                                                         theirs.indptr)  # fmt: skip
    assert np.allclose(np.asarray(mine.data), theirs.data, atol=1e-16) and mine.nnz == 20  # zeros stored
    for ext in (True, "periodic"):
        q = np.array([-1.0, 4.0, 1.5])
        assert np.allclose(np.asarray(IP.BSpline.design_matrix(q, t, 3, extrapolate=ext).toarray()),
                           si.BSpline.design_matrix(q, t, 3, extrapolate=ext).toarray(), atol=1e-13)  # fmt: skip
    # the matrix times the coefficients is the spline
    b = IP.make_interp_spline(np.arange(7.0), np.cos(np.arange(7.0)))
    q = np.linspace(0, 6, 31)
    dm = IP.BSpline.design_matrix(q, b.t, 3)
    assert np.allclose(np.asarray(dm @ b.c), np.asarray(b(q)), atol=1e-14)
    assert IP.BSpline.design_matrix(np.zeros(0), t, 3).shape == (0, 6)


@LIVE
def test_insert_knot_follows_scipy():
    g = np.random.default_rng(44)
    x = np.sort(g.random(10))
    y = g.standard_normal((10, 2))
    q = np.linspace(-0.2, 1.2, 41)
    for k in (1, 2, 3, 5):
        mine = IP.make_interp_spline(x, y, k=k)
        theirs = si.BSpline(np.asarray(mine.t), np.asarray(mine.c), k)  # the same coefficients: the same bits
        for xv, m in ((0.5, 1), (float(x[3]), 2), (float(x[0]), 1), (float(x[-1]), 1)):
            a, b = mine.insert_knot(xv, m=m), theirs.insert_knot(xv, m)
            assert np.array_equal(np.asarray(a.t), b.t) and np.array_equal(np.asarray(a.c), b.c), (k, xv, m)
            inside = q[(q >= x[0]) & (q <= x[-1])]  # a knot at an end changes the extension (scipy too)
            assert np.allclose(np.asarray(a(inside)), np.asarray(mine(inside)), atol=1e-12)
    yp = np.r_[y[:-1], y[:1]]
    mine = IP.make_interp_spline(x, yp, bc_type="periodic")
    theirs = si.BSpline(np.asarray(mine.t), np.asarray(mine.c), 3, extrapolate="periodic")
    a, b = mine.insert_knot(0.5), theirs.insert_knot(0.5)
    assert np.array_equal(np.asarray(a.t), b.t) and np.allclose(np.asarray(a.c), b.c, atol=1e-15)


@LIVE
def test_power_basis_conversions():
    """BSpline.from_power_basis (scipy's Marsden identity) and PPoly.from_spline (IP54: the breakpoints are the
    distinct knots of the base interval, so the polynomial is the spline under every outside rule)."""
    g = np.random.default_rng(45)
    x = np.sort(g.random(9))
    q = np.linspace(-0.3, 1.3, 47)
    for bc in ("not-a-knot", "natural", "clamped", "periodic"):
        y = g.standard_normal(9)
        if bc == "periodic":
            y[-1] = y[0]
        mine = IP.BSpline.from_power_basis(IP.CubicSpline(x, y, bc_type=bc), bc_type=bc)
        theirs = si.BSpline.from_power_basis(si.CubicSpline(x, y, bc_type=bc), bc_type=bc)
        assert np.array_equal(np.asarray(mine.t), theirs.t) and mine.extrapolate == theirs.extrapolate
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-11, atol=1e-12), bc
        assert np.allclose(np.asarray(mine(q)), np.asarray(IP.CubicSpline(x, y, bc_type=bc)(q)), atol=1e-11), bc
    batch = IP.CubicSpline(x, g.standard_normal((9, 3)))
    assert np.asarray(IP.BSpline.from_power_basis(batch).c).shape == (9, 3)  # scipy takes one column only
    for k in range(6):
        b = IP.make_interp_spline(x, g.standard_normal(9), k=k)
        p = IP.PPoly.from_spline(b)
        assert np.array_equal(np.asarray(p.x), np.unique(np.asarray(b.t)))  # clamped knots: scipy's minus repeats
        pt = si.PPoly.from_spline(si.make_interp_spline(x, np.asarray(b.c) if k <= 1 else g.standard_normal(9), k=k))
        assert pt.c.shape[0] == np.asarray(p.c).shape[0]
        # k = 0: the spline's value at x[-1] and past it is the empty last interval's c[-1]; a PPoly has no empty
        # pieces (IP43) and continues its last one
        qq = q[q < x[-1]] if k == 0 else q
        assert np.allclose(np.asarray(p(qq)), np.asarray(b(qq)), rtol=1e-12, atol=1e-12), k
        for ext in (False, "periodic"):
            assert np.allclose(np.asarray(p(qq, extrapolate=ext)), np.asarray(b(qq, extrapolate=ext)), rtol=1e-12,
                               atol=1e-12, equal_nan=True), (k, ext)  # fmt: skip
    # knots past the base interval: scipy's breakpoints run over all knots, sd's over [t[k], t[n]]
    u = IP.BSpline(np.arange(8.0), [1.0, 2.0, 3.0, 4.0], 3, extrapolate="periodic")
    p = IP.PPoly.from_spline(u)
    assert np.asarray(p.x).tolist() == [3.0, 4.0] and p.extrapolate == "periodic"
    assert np.allclose(np.asarray(p(q * 8 - 1)), np.asarray(u(q * 8 - 1)), atol=1e-12)
    assert IP.PPoly.from_spline((np.asarray(u.t), np.asarray(u.c), 3)).extrapolate is True
    pp = IP.PPoly.from_spline((u.t, np.c_[u.c, 2 * np.asarray(u.c)], 3), extrapolate=False)
    assert np.asarray(pp.c).shape == (4, 1, 2)


def test_constructor_rules():
    t = np.arange(8.0)
    b = IP.BSpline(t, np.ones((2, 5)), 3, axis=1)  # the interpolation axis to the front; extra coefficients kept
    assert np.asarray(b.c).shape == (5, 2) and b.axis == 1
    assert np.asarray(b(np.array([3.5, 4.0]))).shape == (2, 2)
    assert IP.BSpline(t, np.ones(4), 3, extrapolate=None).extrapolate is True  # IP52: the class default
    assert b.tck[0] is b.t and b.tck[2] == 3
    with pytest.raises(ValueError):
        b.c[0, 0] = 2.0
    with pytest.raises(ValueError):
        b.t[0] = 2.0
    tt = sd.asarray(t)
    c = IP.BSpline(tt, np.ones(4), 3)
    tt[0] = -5.0
    assert float(c.t[0]) == 0.0  # a copy
    assert repr(c) == "BSpline(degree=3, knots=8, extrapolate=True)"
    with pytest.raises(TypeError, match=r"^BSpline\(\): 't' must be real, not complex128"):
        IP.BSpline(t + 0j, np.ones(4), 3)
