"""LinearNDInterpolator, CloughTocher2DInterpolator and griddata (docs/api/interpolate.md §7; docs/design/interpolate.md
§3.8, §4; docs/design/spatial_geometry.md §2.6, §3.8, §3.9; milestone IP9): the contracts against the exact references
(tests/interpolate/ref/scattered.npz), the values at the data points to the bit, affine data, outside the hull and
NaN points, equal points, complex values and batches, the shapes of xi, griddata's three methods, rescale, a
Delaunay as the points, the gradient estimate's stopping, the errors, and scipy 1.18.1 on the same data."""

import math
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
REF = refdata.ref("scattered")
CT_EVAL = sd._core.interpolate.ct_eval


def _field(p):
    return np.sin(3 * p[:, 0]) + np.cos(2 * p[:, -1]) + 0.1 * p.sum(axis=1)


def _bits(v):
    a = np.asarray(v)
    return a.dtype, a.shape, a.tobytes()


def _set_points(method, sid):
    m = REF.methods[method]
    pos = np.flatnonzero(REF.inputs["set"][m.index] == sid)
    return pos, m.index[pos]


# ---- the contracts against the exact references --------------------------------------------------


def test_linear_contract():
    """|y^ - y| <= gamma_{d+5} sum_k lambda_k |f_k| + (d + 2) 2^-1074 on every set (sd's triangulation, the exact
    value of its piecewise linear function), and the data points exact."""
    m, scale = REF.methods["linear"], REF.methods["linear_scale"]
    worst = 0.0
    for sid in sorted(set(REF.inputs["set"].tolist())):
        sc = S.tri_set(sid)
        d = sc.y.shape[1]
        L = IP.LinearNDInterpolator(sc.y, sc.d)
        got = np.asarray(L(sc.xi))
        pos, idx = _set_points("linear", sid)
        found = np.asarray(L.tri.find_simplex(sc.xi[REF.inputs["index"][idx]]))
        assert np.array_equal(found, REF.inputs["simplex"][idx])
        for p, k in zip(pos, idx):
            i = int(REF.inputs["index"][k])
            err = abs(Fraction(float(got[i])) - Fraction(m.hi[p]) - Fraction(m.lo[p]))
            units = float(err) / refdata.linear_nd_bound(scale.hi[p], d)
            assert units <= 1, (sid, i, units)
            worst = max(worst, units)
            if i >= 64:
                assert got[i] == sc.d[i - 64], (sid, i)
    assert 0.1 < worst


def test_clough_tocher_contract():
    """For the gradients of the reference: |y^ - y| <= 256 u K + 32 2^-1074, K = (1 + max|g_k|) (max|f_k| + max
    |dx f_i| |e_x| + |dy f_i| |e_y|); the data points exact."""
    m, scale = REF.methods["clough_tocher"], REF.methods["ct_scale"]
    sets = REF.group_inputs["set"].tolist()
    worst = 0.0
    for sid in sorted(set(REF.inputs["set"][m.index].tolist())):
        sc = S.tri_set(sid)
        grad = REF.group_inputs["grad"][sets.index(sid)]
        C = IP.CloughTocher2DInterpolator(sc.y, sc.d)
        got = np.asarray(CT_EVAL(C.tri._tri, sd.asarray(sc.xi), C._v, sd.asarray(grad.reshape(-1, 1, 2).copy()),
                                 math.nan, 0.0))[:, 0]  # fmt: skip
        pos, idx = _set_points("clough_tocher", sid)
        for p, k in zip(pos, idx):
            i = int(REF.inputs["index"][k])
            err = abs(Fraction(float(got[i])) - Fraction(m.hi[p]) - Fraction(m.lo[p]))
            units = float(err) / refdata.ct_bound(scale.hi[p])
            assert units <= 1, (sid, i, units)
            worst = max(worst, units)
            if i >= 64:
                assert got[i] == sc.d[i - 64], (sid, i)
    assert worst > 0.001


# ---- values ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("d", [2, 3])
def test_data_points_are_exact(d):
    g = np.random.default_rng(10 + d)
    p = g.random((300, d))
    f = _field(p)
    f[7] = -0.0
    f[11] = 1e-310
    got = np.asarray(IP.LinearNDInterpolator(p, f)(p))
    assert _bits(got) == _bits(f)
    if d == 2:
        assert _bits(IP.CloughTocher2DInterpolator(p, f)(p)) == _bits(f)
    # a value that is not finite does not reach the other data points
    f2 = f.copy()
    f2[3] = np.nan
    f2[4] = np.inf
    got = np.asarray(IP.LinearNDInterpolator(p, f2)(p))
    assert _bits(got) == _bits(f2)


@pytest.mark.parametrize("d", [2, 3])
def test_affine_data(d):
    g = np.random.default_rng(20 + d)
    p = g.random((200, d))
    a = np.array([2.0, 3.0, -1.0][:d])
    f = p @ a + 1.0
    x = g.random((500, d)) * 0.8 + 0.1
    want = x @ a + 1.0
    got = np.asarray(IP.LinearNDInterpolator(p, f)(x))
    ok = np.isfinite(got)
    assert ok.sum() > 400 and np.max(np.abs(got[ok] - want[ok])) < 1e-14
    if d == 2:
        c = IP.CloughTocher2DInterpolator(p, f)
        assert c.converged and np.max(np.abs(np.asarray(c.grad) - a)) < 1e-6
        got = np.asarray(c(x))
        assert np.max(np.abs(got[ok] - want[ok])) < 1e-7
        tight = IP.CloughTocher2DInterpolator(p, f, tol=1e-15, maxiter=2000)
        assert np.max(np.abs(np.asarray(tight(x))[ok] - want[ok])) < 1e-13


def _with_axis(rows, axis, w):
    """Rows of d - 1 coordinates with the column w * (row's flag) inserted at `axis`: that axis's extent is w."""
    return np.array([np.insert(np.array(r[1:], dtype=float), axis, w * r[0]) for r in rows])


@pytest.mark.parametrize("w", [2.0**-1074, 2.0**-1073, 2.0**-1030])
def test_tiny_extent_axis(w):
    """An axis whose extent is subnormal (2^-1074: half of it rounds to 0) on the sets of every axis, d = 2 and 3:
    the data points to the bit, and points of exact barycentric coordinates (the simplex's extents differ by
    2^1000 and more, so the determinants are those of copies scaled axis by axis)."""
    for axis in (0, 1):
        p = _with_axis([[0, 0.0], [1, 1.0], [0, 2.0]], axis, w)
        f = np.array([1.0, 2.0, 3.0])
        x = np.concatenate([p, _with_axis([[0, 1.0], [0, 0.5], [0, 1.5]], axis, w)])
        want = np.concatenate([f, [2.0, 1.5, 2.5]])
        assert _bits(IP.LinearNDInterpolator(p, f)(x)) == _bits(want)
        assert _bits(IP.griddata(p, f, x)) == _bits(want)
        assert _bits(IP.CloughTocher2DInterpolator(p, f)(p)) == _bits(f)
    small = np.array([[w, 0.0], [0.0, w], [0.0, 0.0]])
    f = np.array([1.0, 2.0, 3.0])
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        assert _bits(cls(small, f)(small)) == _bits(f)
    for axis in (0, 1, 2):
        p = _with_axis([[0, 0.0, 0.0], [1, 1.0, 0.0], [0, 2.0, 0.0], [0, 1.0, 1.0]], axis, w)
        f = np.array([1.0, 2.0, 3.0, 4.0])
        x = np.concatenate([p, _with_axis([[0, 1.0, 0.5], [0, 1.0, 0.0]], axis, w)])
        want = np.concatenate([f, [3.0, 2.0]])  # (0, 1, 0.5) = (v0 + v2) / 4 + v3 / 2
        assert _bits(IP.LinearNDInterpolator(p, f)(x)) == _bits(want)
    small = np.array([[0.0, 0.0, 0.0], [w, 0.0, 0.0], [0.0, w, 0.0], [0.0, 0.0, w]])
    f = np.array([1.0, 2.0, 3.0, 4.0])
    assert _bits(IP.LinearNDInterpolator(small, f)(small)) == _bits(f)


def test_outside_and_nan_points():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f = np.array([1.0, 2.0, 3.0, 4.0])
    x = np.array([[0.5, 0.5], [2.0, 0.5], [np.nan, 0.5], [np.inf, 0.0], [1.0, 1.0], [0.5, 0.0], [-1e-300, 0.5]])
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        got = np.asarray(cls(p, f)(x))
        assert np.isnan(got[1]) and np.isnan(got[2]) and np.isnan(got[3]) and np.isnan(got[6])
        assert got[4] == 4.0
        got = np.asarray(cls(p, f, fill_value=-7.0)(x))
        assert got[1] == -7.0 and np.isnan(got[2]) and got[3] == -7.0 and got[6] == -7.0
        assert got[5] == 1.5 if cls is IP.LinearNDInterpolator else np.isfinite(got[5])
        assert cls(p, f).fill_value != cls(p, f).fill_value  # NaN
        assert cls(p, f, fill_value=3).fill_value == 3.0 and type(cls(p, f, fill_value=3).fill_value) is float
    fc = f * (1 + 2j)
    got = np.asarray(IP.LinearNDInterpolator(p, fc, fill_value=1 - 1j)(x))
    assert got.dtype == np.complex128 and got[1] == 1 - 1j and np.isnan(got[2].real) and np.isnan(got[2].imag)


def test_equal_points():
    g = np.random.default_rng(31)
    p = g.random((30, 2))
    p[12] = p[4]
    p[20] = p[4]
    f = _field(p)
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        obj = cls(p, f)
        assert _bits(obj(p)) == _bits(f)
        f2 = f.copy()
        f2[20] = 99.0
        with pytest.raises(ValueError, match=rf"^{cls.__name__}\(\): data points 4 and 20 are equal and their values "
                                             "differ$"):  # fmt: skip
            cls(p, f2)
    # bit equality: a NaN with the same bits is the same value, -0.0 and 0.0 are not
    f3 = f.copy()
    f3[[4, 12, 20]] = np.nan
    IP.LinearNDInterpolator(p, f3)
    f3[[4, 12, 20]] = [0.0, -0.0, 0.0]
    with pytest.raises(ValueError, match="data points 4 and 12"):
        IP.LinearNDInterpolator(p, f3)
    c = IP.CloughTocher2DInterpolator(p, f)
    grad = np.asarray(c.grad)
    assert np.array_equal(grad[12], grad[4]) and np.array_equal(grad[20], grad[4])


def test_complex_and_batches():
    g = np.random.default_rng(41)
    p = g.random((60, 2))
    x = g.random((7, 3, 2))
    re, im = _field(p), np.cos(5 * p[:, 0])
    batch = np.stack([re, im, re * im], axis=1).reshape(60, 3, 1)
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        c = cls(p, re + 1j * im, fill_value=-1.0)  # outside, a complex fill_value is (fill_value, 0) as scipy's
        got = np.asarray(c(x))
        assert got.shape == (7, 3) and got.dtype == np.complex128
        assert _bits(got.real) == _bits(np.asarray(cls(p, re, fill_value=-1.0)(x)))
        inside = got.real != -1.0
        assert inside.sum() > 5 and np.all(got.imag[~inside] == 0.0)
        assert _bits(got.imag[inside]) == _bits(np.asarray(cls(p, im)(x))[inside])
        b = cls(p, batch)
        got = np.asarray(b(x))
        assert got.shape == (7, 3, 3, 1)
        for k in range(3):
            assert _bits(got[..., k, 0]) == _bits(np.asarray(cls(p, batch[:, k, 0])(x)))
        assert np.asarray(b.values).shape == (60, 3, 1)
    cc = IP.CloughTocher2DInterpolator(p, re + 1j * im)
    grad = np.asarray(cc.grad)
    assert grad.shape == (60, 2) and grad.dtype == np.complex128
    assert _bits(grad.real) == _bits(np.asarray(IP.CloughTocher2DInterpolator(p, re).grad))
    assert np.asarray(IP.CloughTocher2DInterpolator(p, batch).grad).shape == (60, 3, 1, 2)


def test_shapes_of_xi():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f = np.array([1.0, 2.0, 3.0, 4.0])
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        obj = cls(p, f)
        one = obj([0.25, 0.5])
        assert type(one) is float
        assert type(obj((0.25, 0.5))) is float and obj((0.25, 0.5)) == one
        gx, gy = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
        got = np.asarray(obj((gx, gy)))
        assert got.shape == (3, 4)
        assert _bits(got) == _bits(np.asarray(obj(np.stack([gx, gy], axis=-1))))
        assert np.asarray(obj((np.linspace(0, 1, 5), 0.5))).shape == (5,)  # broadcast
        assert np.asarray(obj(np.zeros((0, 2)))).shape == (0,)
        with pytest.raises(ValueError, match=r"'xi' must have 2 entries on its last axis"):
            obj([0.1, 0.2, 0.3])
        with pytest.raises(ValueError, match=r"a tuple 'xi' must have 2 coordinate arrays, not 3"):
            obj((0.1, 0.2, 0.3))
        with pytest.raises(ValueError, match=r"'xi' must have at least 1 dimension"):
            obj(np.float64(0.5))
        with pytest.raises(TypeError, match=r"'xi' must be real"):
            obj([0.1 + 1j, 0.2])


def test_points_forms():
    g = np.random.default_rng(51)
    p = g.random((50, 2))
    f = _field(p)
    x = g.random((40, 2))
    want = _bits(IP.LinearNDInterpolator(p, f)(x))
    tri = sd.spatial.Delaunay(p)
    assert _bits(IP.LinearNDInterpolator(tri, f)(x)) == want
    assert _bits(IP.LinearNDInterpolator((p[:, 0], p[:, 1]), f)(x)) == want
    assert IP.LinearNDInterpolator(tri, f).tri is tri
    assert _bits(IP.CloughTocher2DInterpolator(tri, f)(x)) == _bits(IP.CloughTocher2DInterpolator(p, f)(x))
    with pytest.raises(ValueError, match=r"^LinearNDInterpolator\(\): 'rescale' must be False when 'points' is a "
                                         "Delaunay$"):  # fmt: skip
        IP.LinearNDInterpolator(tri, f, rescale=True)
    with pytest.raises(ValueError, match="coordinate arrays of a tuple 'points' must be 1-D and of one length"):
        IP.LinearNDInterpolator((p[:, 0], p[:-1, 1]), f[:-1])


def test_rescale():
    g = np.random.default_rng(61)
    p = g.random((80, 2)) * np.array([1e6, 1e-3])
    f = _field(p / np.array([1e6, 1e-3]))
    x = g.random((50, 2)) * np.array([1e6, 1e-3])
    off = np.asarray(sd.mean(sd.asarray(p), axis=0))
    scale = p.max(0) - p.min(0)
    for cls in (IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator):
        r = cls(p, f, rescale=True)
        manual = cls((p - off) / scale, f)
        assert _bits(r(x)) == _bits(manual((x - off) / scale))
        assert _bits(r.points) == _bits(p)


def test_clough_tocher_gradient_stopping():
    g = np.random.default_rng(71)
    p = g.random((200, 2))
    f = _field(p)
    c = IP.CloughTocher2DInterpolator(p, f)
    assert c.converged and 1 <= c.niter < 400
    one = IP.CloughTocher2DInterpolator(p, f, maxiter=1)
    assert not one.converged and one.niter == 1
    zero = IP.CloughTocher2DInterpolator(p, f, tol=0.0, maxiter=30)
    assert not zero.converged and zero.niter == 30
    loose = IP.CloughTocher2DInterpolator(p, f, tol=1e300)
    assert loose.converged and loose.niter == 1
    assert _bits(loose.grad) == _bits(one.grad)
    # a batch: the most sweeps of its columns, converged when all are
    b = IP.CloughTocher2DInterpolator(p, np.stack([f, 1e3 * f], axis=1))
    assert b.niter >= c.niter and b.converged


def test_griddata():
    g = np.random.default_rng(81)
    p = g.random((70, 2))
    f = _field(p)
    gx, gy = np.meshgrid(np.linspace(0, 1, 6), np.linspace(0, 1, 5))
    xi = (gx, gy)
    assert _bits(IP.griddata(p, f, xi)) == _bits(IP.LinearNDInterpolator(p, f)(xi))
    assert _bits(IP.griddata(p, f, xi, method="cubic", fill_value=0.0)) == _bits(
        IP.CloughTocher2DInterpolator(p, f, fill_value=0.0)(xi))  # fmt: skip
    assert _bits(IP.griddata(p, f, xi, method="nearest", fill_value=5.0)) == _bits(IP.NearestNDInterpolator(p, f)(xi))
    assert _bits(IP.griddata((p[:, 0], p[:, 1]), f, xi, rescale=True)) == _bits(
        IP.LinearNDInterpolator(p, f, rescale=True)(xi))  # fmt: skip
    tri = sd.spatial.Delaunay(p)
    assert _bits(IP.griddata(tri, f, xi, method="nearest")) == _bits(IP.NearestNDInterpolator(p, f)(xi))
    p3 = g.random((60, 3))
    assert _bits(IP.griddata(p3, _field(p3), p3[:5])) == _bits(_field(p3)[:5])
    t = np.linspace(0, 1, 7)
    assert _bits(IP.griddata(t, t**2, [0.3, 0.9], method="nearest")) == _bits(np.array([t[2] ** 2, t[5] ** 2]))
    p5 = g.random((30, 5))
    assert np.asarray(IP.griddata(p5, p5[:, 0], p5[:3], method="nearest")).shape == (3,)
    with pytest.raises(ValueError, match=r"^griddata\(\): 1-D points: use sd.interp \(method 'linear'\) or CubicSpline "
                                         r"\(method 'cubic'\)$"):  # fmt: skip
        IP.griddata(t, t, [0.5])
    with pytest.raises(ValueError, match=r"^griddata\(\): method 'cubic' is for 2-D points, not 3-D$"):
        IP.griddata(p3, p3[:, 0], p3[:2], method="cubic")
    with pytest.raises(ValueError, match=r"^griddata\(\): 'points' must have 2 or 3 columns, not 5$"):
        IP.griddata(p5, p5[:, 0], p5[:2])
    with pytest.raises(ValueError, match=r"^griddata\(\): 'method' must be 'nearest', 'linear' or 'cubic', not "
                                         "'spline'$"):  # fmt: skip
        IP.griddata(p, f, xi, method="spline")


def test_errors():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f = np.array([1.0, 2.0, 3.0, 4.0])
    L, C = IP.LinearNDInterpolator, IP.CloughTocher2DInterpolator
    cases = [
        (
            lambda: L(p[:, 0], f),
            ValueError,
            r"^LinearNDInterpolator\(\): 'points' must have 2 dimensions \(points, "
            r"coordinates\), not 1$",
        ),
        (lambda: L(np.zeros((4, 4)), f), ValueError, r"'points' must have 2 or 3 columns, not 4$"),
        (
            lambda: C(np.eye(4)[:, :3], f),
            ValueError,
            r"^CloughTocher2DInterpolator\(\): 'points' must have 2 columns, "
            "not 3$",
        ),
        (
            lambda: L([[0, 0], [1, 1], [2, 2]], [1, 2, 3]),
            ValueError,
            r"^LinearNDInterpolator\(\): 'points' are "
            "collinear$",
        ),
        (lambda: L([[0, 0], [1, 1], [0, 0]], [1, 2, 1]), ValueError, r"needs at least 3 distinct points, got 2$"),
        (lambda: L([[0, 0], [1, np.nan], [0, 1]], [1, 2, 3]), ValueError, r"'points' must be finite$"),
        (lambda: L(p + 0j, f), TypeError, r"'points' must be real, not complex128$"),
        (lambda: L(p, f[:3]), ValueError, r"'values' must have one row per data point \(4\), not 3$"),
        (lambda: L(p, np.float64(1.0)), ValueError, r"'values' must have at least 1 dimension$"),
        (lambda: L(p, f, fill_value=None), TypeError, r"'fill_value' must be a number, not NoneType$"),
        (lambda: L(p, f, fill_value=True), TypeError, r"'fill_value' must be a number, not bool$"),
        (
            lambda: L(p, f, fill_value=1j),
            ValueError,
            r"'fill_value' must be a real number for real values, not "
            "complex$",
        ),
        (lambda: L(p, f, rescale=1), TypeError, r"'rescale' must be True or False, not int$"),
        (
            lambda: C(p, [1.0, np.nan, 2.0, 3.0]),
            ValueError,
            r"^CloughTocher2DInterpolator\(\): 'values' must be "
            "finite$",
        ),
        (lambda: C(p, f, tol=-1.0), ValueError, r"'tol' must be a finite number >= 0, not -1.0$"),
        (lambda: C(p, f, tol=np.nan), ValueError, r"'tol' must be a finite number >= 0, not nan$"),
        (lambda: C(p, f, tol="1"), TypeError, r"'tol' must be an int or a float, not str$"),
        (lambda: C(p, f, maxiter=0), ValueError, r"'maxiter' must be >= 1, not 0$"),
        (lambda: C(p, f, maxiter=2.0), TypeError, r"'maxiter' must be an int, not float$"),
        (
            lambda: L(p, f, 0.0),
            TypeError,
            r"^LinearNDInterpolator\(\): takes 2 positional arguments but 3 were "
            "given$",
        ),
        (lambda: L(p, f)(p, 1), TypeError, r"^LinearNDInterpolator\(\):"),
        (lambda: IP.griddata(p, f), TypeError, r"^griddata\(\): missing 1 required positional argument: 'xi'$"),
    ]
    for fn, kind, pattern in cases:
        with pytest.raises(kind, match=pattern):
            fn()


def test_attributes_and_repr():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    f = np.array([1.0, 2.0, 3.0, 4.0])
    c = IP.CloughTocher2DInterpolator(p, f)
    for a in (c.points, c.values, c.grad):
        assert not a._flags()["writeable"]
    assert isinstance(c.tri, sd.spatial.Delaunay) and c.tri.npoints == 4
    assert repr(c) == "CloughTocher2DInterpolator(points=4, dimensions=2)"
    assert repr(IP.LinearNDInterpolator(np.eye(4)[:, :3], f)) == "LinearNDInterpolator(points=4, dimensions=3)"


# ---- scipy 1.18.1 on the same data ---------------------------------------------------------------


@LIVE
@pytest.mark.parametrize("sid", [s for s in S.tri_ids() if "random" in S.tri_set(s).tags])
def test_scipy_values_in_general_position(sid):
    """On the random sets the triangulations are the same and the values agree to rounding; outside the hull both give
    fill_value (scipy's tolerance puts a few points within 1e-15 of the hull inside: none here)."""
    import scipy.interpolate as si

    sc = S.tri_set(sid)
    scale = np.max(np.abs(sc.d))
    got = np.asarray(IP.LinearNDInterpolator(sc.y, sc.d)(sc.xi))
    want = si.LinearNDInterpolator(sc.y, sc.d)(sc.xi)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    ok = ~np.isnan(got)
    assert np.max(np.abs(got[ok] - want[ok])) <= 64 * 2**-53 * scale
    if sc.y.shape[1] == 2:
        c = IP.CloughTocher2DInterpolator(sc.y, sc.d)
        cs = si.CloughTocher2DInterpolator(sc.y, sc.d)
        # the same estimate, stopped by another rule: within the tolerance
        assert np.max(np.abs(np.asarray(c.grad) - cs.grad[:, 0, :])) <= 1e-5 * np.max(np.abs(cs.grad))
        got, want = np.asarray(c(sc.xi)), cs(sc.xi)
        assert np.array_equal(np.isnan(got), np.isnan(want))
        assert np.max(np.abs(got[ok] - want[ok])) <= 1e-6 * scale
        assert _bits(IP.CloughTocher2DInterpolator(sc.y, sc.d, tol=1e-14, maxiter=1000).grad) != _bits(c.grad)


@LIVE
def test_scipy_differences():
    """scipy keeps one value of equal points without a word, is off by an ULP at data points, puts NaN points to
    fill_value only as NaN, and stops the gradients with a warning (sd: converged)."""
    import warnings

    import scipy.interpolate as si

    g = np.random.default_rng(91)
    p = g.random((40, 2))
    f = _field(p)
    p2 = np.vstack([p, p[:1]])
    assert si.LinearNDInterpolator(p2, np.append(f, 99.0))(p[:1])[0] in (f[0], 99.0)  # one of them, no error
    with pytest.raises(ValueError, match="are equal and their values differ"):
        IP.LinearNDInterpolator(p2, np.append(f, 99.0))
    big = g.random((2000, 2))
    fb = _field(big)
    assert np.any(si.LinearNDInterpolator(big, fb)(big) != fb)
    assert _bits(IP.LinearNDInterpolator(big, fb)(big)) == _bits(fb)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        si.CloughTocher2DInterpolator(p, f, maxiter=1)
    assert any("did not converge" in str(x.message) for x in w)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not IP.CloughTocher2DInterpolator(p, f, maxiter=1).converged
