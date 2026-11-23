"""RegularGridInterpolator and interpn (docs/design/interpolate.md §2.7, §3.7, §4; milestone IP5).

- the linear paths are one: in 1-D the linear and slinear grids give sd.interp's bits inside and the k = 1 B-spline's
  outside (fill_value=None); in d dimensions linear is the 1-D linear path folded axis by axis from the last to the
  first, slinear is linear, interpn is the object;
- cubic and quintic are make_interp_spline in 1-D (bits), and in d dimensions meet the construction contract of the
  tensor coefficients (the 1-D contract of every axis's solve carried through the later axes) and the evaluation
  contract of the tensor product (refdata.tensor_collocation_bound, refdata.tensor_bound), end to end against the
  exact interpolant of grid.npz;
- pchip is the 1-D PchipInterpolator folded axis by axis (bits): scipy's algorithm with sd's 1-D interpolant, so the
  same function as scipy's, whose values it meets within rounding;
- linear meets the stacked linear contract (refdata.grid_linear_exact), nearest picks scipy's node;
- the outside rules, NaN points (always NaN, decision IP24), axes of one node, complex values, shapes, nu, the errors;
- the same bytes at 1, 2, 3, 7 and 10 threads.
"""

import math
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
RGI = IP.RegularGridInterpolator
A = sd.asarray
LINEAR = sd._core.interpolate.linear
EXTEND = 1
METHODS = ("linear", "nearest", "slinear", "cubic", "quintic", "pchip")
NEED = {"linear": 1, "nearest": 1, "slinear": 2, "cubic": 4, "quintic": 6, "pchip": 4}
DEGREE = {"cubic": 3, "quintic": 5}


def _np(a):
    return np.asarray(a)


def _ascending(G):
    """The grid's points ascending and its values flipped with them (scipy's rule, RegularGridInterpolator's)."""
    pts, vals = [], np.asarray(G.values)
    for a, p in enumerate(G.points):
        if p.size > 1 and p[0] > p[-1]:
            p = p[::-1].copy()
            vals = np.flip(vals, axis=a)
        pts.append(p)
    return pts, np.ascontiguousarray(vals)


def _grids(meth, finite=False):
    """The sample grids the method takes; finite: the ones with finite values and one column (the references')."""
    for gid in S.GRID_IDS:
        G = S.grid(gid)
        if any(p.size < NEED[meth] for p in G.points):
            continue
        if finite and ({"nan", "batch"} & G.tags):
            continue
        yield G


def _outside(G, g, q=12):
    """Points beyond the grid on some axes (and inside on the others)."""
    lo = np.array([p.min() for p in G.points])
    hi = np.array([p.max() for p in G.points])
    span = np.where(hi > lo, hi - lo, 1.0)
    return lo - 0.4 * span + 1.8 * span * g.random((q, len(G.points)))


# ---------------------------------------------------------------- one linear path


def test_1d_linear_and_slinear_are_the_linear_path():
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        if c.n < 2:
            continue
        inside = c.xe[(c.xe >= c.x[0]) & (c.xe <= c.x[-1])]
        want = sd.interp(A(inside), A(c.x), A(c.y))
        k1 = IP.make_interp_spline(c.x, c.y, k=1)
        for meth in ("linear", "slinear"):
            got = RGI((c.x,), c.y, method=meth)(inside)
            assert bits_equal(got, want), (cid, meth)
            assert bits_equal(IP.interpn((c.x,), c.y, inside, method=meth), want), (cid, meth)
            ext = RGI((c.x,), c.y, method=meth, bounds_error=False, fill_value=None)(c.xe)
            assert bits_equal(ext, k1(c.xe)), (cid, meth)
            path = _np(LINEAR(A(c.xe), A(c.x), A(c.y.reshape(-1, 1)), EXTEND))[:, 0]
            assert bits_equal(ext, path), (cid, meth)


def _fold_linear(points, values, pt):
    """The 1-D linear path (_core.interpolate.linear, extended) along every axis from the last to the first."""
    arr = np.asarray(values)
    for a in reversed(range(len(points))):
        p = points[a]
        if p.size == 1:
            arr = arr.take(0, axis=a)
            continue
        lines = np.ascontiguousarray(np.moveaxis(arr, a, 0)).reshape(p.size, -1)
        arr = _np(LINEAR(A([pt[a]]), A(p), A(lines), EXTEND))[0].reshape(np.moveaxis(arr, a, 0).shape[1:])
    return arr


def test_nd_linear_is_the_1d_path_folded():
    g = np.random.default_rng(51)
    for G in _grids("linear"):
        pts, vals = _ascending(G)
        q = np.vstack([G.xi, _outside(G, g)])
        lin = RGI(G.points, G.values, bounds_error=False, fill_value=None)
        got = _np(lin(q))
        for i, pt in enumerate(q):
            assert bits_equal(got[i], _fold_linear(pts, vals, pt)), (G.id, i)
        assert bits_equal(lin(G.xi), IP.interpn(G.points, G.values, G.xi)), G.id
        if all(p.size >= 2 for p in G.points):
            sl = RGI(G.points, G.values, method="slinear", bounds_error=False, fill_value=None)
            assert bits_equal(sl(q), got), G.id
            assert bits_equal(sl(q, nu=[0] * len(G.points)), got), G.id


def test_a_decreasing_axis_is_the_flipped_ascending_one():
    G = S.grid("grid/2d-desc")
    pts, vals = _ascending(G)
    for meth in METHODS:
        a = RGI(G.points, G.values, method=meth)
        b = RGI(tuple(pts), vals, method=meth)
        assert bits_equal(a(G.xi), b(G.xi)), meth
        assert bits_equal(a.values, vals) and all(bits_equal(u, v) for u, v in zip(a.grid, pts)), meth


def test_columns_and_complex_parts():
    G = S.grid("grid/batch")
    v = G.values  # (6, 5, 2)
    z = v[..., 0] + 1j * v[..., 1]
    for meth in METHODS:
        if any(p.size < NEED[meth] for p in G.points):
            continue
        whole = _np(RGI(G.points, v, method=meth)(G.xi))
        for col in range(2):
            alone = RGI(G.points, np.ascontiguousarray(v[..., col]), method=meth)(G.xi)
            assert bits_equal(whole[:, col], alone), (meth, col)
        if meth != "pchip":
            gz = _np(RGI(G.points, z, method=meth)(G.xi))
            assert bits_equal(gz.real, whole[:, 0]) and bits_equal(gz.imag, whole[:, 1]), meth
    with pytest.raises(ValueError, match="takes real values"):
        RGI(G.points, z, method="pchip")


# ---------------------------------------------------------------- splines


def test_1d_cubic_and_quintic_are_make_interp_spline():
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        if not np.all(np.isfinite(c.y)):
            continue
        for meth, k in DEGREE.items():
            if c.n < k + 1:
                continue
            b = IP.make_interp_spline(c.x, c.y, k=k)
            r = RGI((c.x,), c.y, method=meth, bounds_error=False, fill_value=None)
            assert bits_equal(r(c.xe), b(c.xe)), (cid, meth)
            for nu in (1, 2, k, k + 1):
                assert bits_equal(r(c.xe, nu=(nu,)), b(c.xe, nu=nu)), (cid, meth, nu)
        if c.n >= 2:
            b1 = IP.make_interp_spline(c.x, c.y, k=1)
            r1 = RGI((c.x,), c.y, method="slinear", bounds_error=False, fill_value=None)
            assert bits_equal(r1(c.xe, nu=(1,)), b1(c.xe, nu=1)), cid


def _spline_points(G, g):
    return np.vstack([G.xi, _outside(G, g, 6)])


@pytest.mark.parametrize("meth", ["cubic", "quintic"])
def test_spline_contracts(meth):
    """The construction contract of the coefficients, the evaluation contract (nu = 0 and derivatives, inside and
    outside) and end to end the exact interpolant of grid.npz."""
    k = DEGREE[meth]
    ref = refdata.ref("grid")
    m = ref.methods[meth]
    g = np.random.default_rng(52)
    worst = {"coefficients": 0.0, "evaluation": 0.0, "end to end": 0.0}
    for G in _grids(meth, finite=True):
        r = RGI(G.points, G.values, method=meth, bounds_error=False, fill_value=None)
        pts = [np.asarray(p) for p in r.grid]
        vals = np.asarray(r.values)
        knots, c2 = r._spline(k, "test")
        knots = [np.asarray(t) for t in knots]
        c = np.asarray(c2).reshape(vals.shape)
        exact = refdata.tensor_coefficients(pts, knots, k, vals)
        cb = refdata.tensor_collocation_bound(pts, knots, k, vals)
        for idx in np.ndindex(c.shape):
            err = abs(float(exact[idx] - Fraction(float(c[idx]))))
            assert err <= cb[idx], (G.id, idx, err, cb[idx])
            worst["coefficients"] = max(worst["coefficients"], err / cb[idx] if err else 0.0)
        q = _spline_points(G, g)
        d = len(pts)
        for nu in ([0] * d, [1] + [0] * (d - 1), [0] * (d - 1) + [2], [k + 1] + [0] * (d - 1)):
            got = _np(r(q, nu=nu))
            for i, pt in enumerate(q):
                want = refdata.tensor_exact(knots, c, k, pt, nu)
                err = abs(float(Fraction(float(got[i])) - want))
                b = refdata.tensor_bound(knots, c, k, pt, nu)
                assert err <= b, (G.id, nu, i, err, b)
                worst["evaluation"] = max(worst["evaluation"], err / b if err else 0.0)
        # end to end at the points of grid.npz (inside the grid)
        pos = np.flatnonzero(ref.inputs["set"][m.index] == G.id)
        got = _np(r(G.xi))
        for p in pos:
            i = int(ref.inputs["index"][m.index[p]])
            err = abs(float(Fraction(float(got[i])) - Fraction(float(m.hi[p])) - Fraction(float(m.lo[p]))))
            b = refdata.tensor_bound(knots, c, k, G.xi[i], None, c_bound=cb)
            assert err <= b, (G.id, i, err, b)
            worst["end to end"] = max(worst["end to end"], err / b if err else 0.0)
    assert all(v < 1 for v in worst.values()), worst


def test_cubic_is_exact_on_a_quadratic():
    """scipy's iterative solve gives 24.999999999999986 here (tests/interpolate/data/scipy.json rgi/cubic_quadratic,
    decision IP23); the direct solve reproduces the quadratic within rounding."""
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    r = RGI((gx, gx), q, method="cubic")
    assert abs(r([[3.0, 4.0]])[0] - 25.0) <= 4 * 2.0**-52 * 25
    assert abs(r([[1.5, 2.5]], nu=(1, 0))[0] - 3.0) <= 64 * 2.0**-52 * 3
    assert abs(RGI((gx, gx), q, method="quintic")([[3.3, 4.1]])[0] - (3.3**2 + 4.1**2)) <= 1e-12


# ---------------------------------------------------------------- pchip


def test_1d_pchip_is_pchip_interpolator():
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        if c.n < 4 or not np.all(np.isfinite(c.y)):
            continue
        try:
            p = IP.PchipInterpolator(c.x, c.y)
        except ValueError:  # coefficients that overflow (IP36): the grid computes them locally
            continue
        r = RGI((c.x,), c.y, method="pchip", bounds_error=False, fill_value=None)
        assert bits_equal(r(c.xe), p(c.xe)), cid


def _fold_pchip(points, values, pt):
    """scipy's algorithm (RegularGridInterpolator._evaluate_spline) with sd's PchipInterpolator: along the last axis
    first, then the folded values along each axis down to the first."""
    arr = np.asarray(values)
    for a in reversed(range(len(points))):
        moved = np.ascontiguousarray(np.moveaxis(arr, a, 0))
        arr = _np(IP.PchipInterpolator(points[a], moved, axis=0)(pt[a]))
        arr = np.asarray(arr, float).reshape(moved.shape[1:])
    return arr


def test_nd_pchip_is_pchip_interpolator_folded():
    g = np.random.default_rng(53)
    for G in _grids("pchip", finite=True):
        pts, vals = _ascending(G)
        q = np.vstack([G.xi, _outside(G, g)])
        got = _np(RGI(G.points, G.values, method="pchip", bounds_error=False, fill_value=None)(q))
        for i, pt in enumerate(q):
            assert bits_equal(got[i], _fold_pchip(pts, vals, pt)), (G.id, i)


@LIVE
def test_pchip_is_scipys_function():
    """The same coefficients as scipy's (IP42) and the same intervals: the values differ by the evaluation's rounding
    only (sd's Horner with fma, scipy's PPoly evaluation)."""
    import scipy.interpolate as si

    g = np.random.default_rng(54)
    worst = 0.0
    for G in _grids("pchip", finite=True):
        q = np.vstack([G.xi, _outside(G, g, 40)])
        kw = {"bounds_error": False, "fill_value": None}
        got = _np(RGI(G.points, G.values, method="pchip", **kw)(q))
        want = si.RegularGridInterpolator(G.points, G.values, method="pchip", **kw)(q)
        # the end cubics grow outside: a point's scale is max|v| times (1 + its distance / the end width)^3 per axis
        scale = np.full(len(q), float(np.max(np.abs(G.values))))
        for a, p in enumerate(G.points):
            ps = np.sort(p)
            below, above = np.maximum(ps[0] - q[:, a], 0), np.maximum(q[:, a] - ps[-1], 0)
            dist = below / (ps[1] - ps[0]) + above / (ps[-1] - ps[-2])
            scale *= (1 + dist) ** 3
        worst = max(worst, float(np.max(np.abs(got - want) / scale)))
    assert worst < 2.0**-47, worst


def test_pchip_end_to_end():
    ref = refdata.ref("grid")
    m = ref.methods["pchip"]
    for G in _grids("pchip", finite=True):
        got = _np(RGI(G.points, G.values, method="pchip")(G.xi))
        pos = np.flatnonzero(ref.inputs["set"][m.index] == G.id)
        assert pos.size
        for p in pos:
            i = int(ref.inputs["index"][m.index[p]])
            err = abs((got[i] - m.hi[p]) - m.lo[p])
            assert err <= 2.0**-45 * np.max(np.abs(G.values)), (G.id, i, err)


# ---------------------------------------------------------------- linear contract, nearest


def test_linear_contract():
    ref = refdata.ref("grid")
    m = ref.methods["linear"]
    g = np.random.default_rng(55)
    worst = 0.0
    for G in _grids("linear", finite=True):
        pts, vals = _ascending(G)
        q = np.vstack([G.xi, _outside(G, g)])
        got = _np(RGI(G.points, G.values, bounds_error=False, fill_value=None)(q))
        for i, pt in enumerate(q):
            exact, b = refdata.grid_linear_exact(pts, vals, pt)
            err = abs(float(Fraction(float(got[i])) - exact))
            assert err <= b, (G.id, i, err, b)
            worst = max(worst, err / b if err else 0.0)
        pos = np.flatnonzero(ref.inputs["set"][m.index] == G.id)
        for p in pos:  # grid.npz's linear reference is the same exact value
            i = int(ref.inputs["index"][m.index[p]])
            exact, _ = refdata.grid_linear_exact(pts, vals, G.xi[i])
            dd = Fraction(float(m.hi[p])) + Fraction(float(m.lo[p]))
            assert abs(float(exact - dd)) <= 2.0**-100 * max(1.0, abs(float(exact))), (G.id, i)
    assert worst < 1, worst


def test_linear_is_exact_at_the_nodes_and_between_the_corners():
    for G in _grids("linear", finite=True):
        pts, vals = _ascending(G)
        r = RGI(G.points, G.values)
        nodes = np.array(np.meshgrid(*pts, indexing="ij")).reshape(len(pts), -1).T
        assert bits_equal(r(nodes), vals.reshape((-1,) + vals.shape[len(pts) :])), G.id
        got = _np(r(G.xi))
        for i, pt in enumerate(G.xi):
            idx = []
            for p, v in zip(pts, pt):
                j = min(max(int(np.searchsorted(p, v, side="right")) - 1, 0), max(p.size - 2, 0))
                idx.append([j] if p.size == 1 else [j, j + 1])
            corner = vals[np.ix_(*idx)]
            assert corner.min() <= got[i] <= corner.max(), (G.id, i)


def test_nearest_picks_scipys_node():
    for G in _grids("nearest"):
        pts, vals = _ascending(G)
        g = np.random.default_rng(56)
        q = np.vstack([G.xi, _outside(G, g)])
        got = _np(RGI(G.points, G.values, method="nearest", bounds_error=False, fill_value=None)(q))
        for i, pt in enumerate(q):
            idx = []
            for p, v in zip(pts, pt):
                if p.size == 1:
                    idx.append(0)
                    continue
                j = min(max(int(np.searchsorted(p, v, side="right")) - 1, 0), p.size - 2)
                t = (v - p[j]) / (p[j + 1] - p[j])
                idx.append(j if t <= 0.5 else j + 1)
            assert bits_equal(got[i], vals[tuple(idx)]), (G.id, i)
    gx = np.arange(6.0)
    tie = RGI((gx,), gx * 10, method="nearest")([2.5, 3.5])
    assert _np(tie).tolist() == refdata.data("scipy")["rgi/nearest_tie"]["value"]["array"]


# ---------------------------------------------------------------- outside, NaN, special axes


def test_nan_points_are_nan_and_lie_nowhere():
    G = S.grid("grid/2d")
    q = np.array([[math.nan, 0.5], [0.5, math.nan], [math.nan, 9.0], [0.5, 0.5]])
    for meth in METHODS:
        for kw in ({}, {"bounds_error": False}, {"bounds_error": False, "fill_value": 5.0},
                   {"bounds_error": False, "fill_value": None}):  # fmt: skip
            got = _np(RGI(G.points, G.values, method=meth, **kw)(q))
            assert np.isnan(got[:3]).all() and np.isfinite(got[3]), (meth, kw)
        if meth in ("slinear", "cubic", "quintic"):
            got = _np(RGI(G.points, G.values, method=meth)(q, nu=[1, 0]))
            assert np.isnan(got[:3]).all(), meth
    gz = _np(RGI(G.points, G.values + 1j)(q))
    assert np.isnan(gz[0].real) and np.isnan(gz[0].imag)


def test_outside_rules():
    G = S.grid("grid/3d")
    inside = G.xi[:5]
    out = np.array([[0.5, 0.5, 1.5], [0.5, -0.1, 1.5], [2.0, 0.5, 0.5], [0.5, math.inf, 0.5], [-math.inf, 0.5, 0.5]])
    for meth in METHODS:
        if any(p.size < NEED[meth] for p in G.points):
            continue
        r = RGI(G.points, G.values, method=meth)
        with pytest.raises(ValueError, match=r"RegularGridInterpolator\(\): one of the requested xi is out of bounds "
                                             r"in dimension 1"):  # fmt: skip
            r(np.vstack([inside, out[1:2], out[0:1]]))
        with pytest.raises(ValueError, match="in dimension 0"):
            IP.interpn(G.points, G.values, out[2:3], method=meth)
        f = _np(RGI(G.points, G.values, method=meth, bounds_error=False, fill_value=-7.5)(np.vstack([inside, out])))
        assert (f[5:] == -7.5).all() and np.isfinite(f[:5]).all(), meth
        n = _np(RGI(G.points, G.values, method=meth, bounds_error=False)(out))
        assert np.isnan(n).all(), meth
        e = RGI(G.points, G.values, method=meth, bounds_error=False, fill_value=None)
        assert bits_equal(e(inside), r(inside)), meth
        assert np.isfinite(_np(e(out[:3]))).all(), meth


def test_axes_of_one_node():
    rec = refdata.data("scipy")
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    r = RGI((gx[:1], gx), q[:1])
    assert _np(r([[0.0, 2.5]])).tolist() == rec["rgi/one_point_linear"]["value"]["array"]
    with pytest.raises(ValueError, match="dimension 0"):
        r([[0.5, 2.5]])
    e = RGI((gx[:1], gx), q[:1], bounds_error=False, fill_value=None)
    assert _np(e([[7.0, 2.5], [-3.0, 2.5]])).tolist() == [6.5, 6.5]
    assert _np(RGI((gx[:1], gx), q[:1], method="nearest")([[0.0, 2.4]])).tolist() == [4.0]
    G = S.grid("grid/axis1")
    assert bits_equal(RGI(G.points, G.values)(G.xi), RGI((G.points[1],), G.values[0])(G.xi[:, 1:]))
    for meth in ("slinear", "cubic", "quintic", "pchip"):
        with pytest.raises(ValueError, match=f"method '{meth}' needs at least"):
            RGI(G.points, G.values, method=meth)


def test_values_that_are_not_finite():
    G = S.grid("grid/nan")  # one NaN node
    i = int(np.flatnonzero(np.isnan(G.values.reshape(-1)))[0])
    node = np.unravel_index(i, G.values.shape)
    lin = RGI(G.points, G.values)
    # the NaN reaches the points whose weight on it is not 0 (the 1-D linear path's rule)
    at = np.array([[G.points[0][node[0]], G.points[1][node[1]]]])
    assert np.isnan(_np(lin(at))).all()
    far = np.array([[G.points[0][0], G.points[1][0]]]) if node != (0, 0) else np.array([[1.0, 1.0]])
    assert np.isfinite(_np(lin(far))).all()
    assert np.isfinite(_np(RGI(G.points, G.values, method="slinear")(far))).all()
    for meth in ("cubic", "quintic", "pchip"):
        if any(p.size < NEED[meth] for p in G.points):
            continue
        with pytest.raises(ValueError, match=f"'values' must be finite for method '{meth}'"):
            RGI(G.points, G.values, method=meth)
        with pytest.raises(ValueError, match="must be finite"):
            lin(far, method=meth)
    inf = np.array(G.values)
    inf[node] = math.inf
    assert np.isinf(_np(RGI(G.points, inf)(at))).all()


def test_nu():
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    r = RGI((gx, gx), q, method="cubic", bounds_error=False)
    assert _np(r([[1.5, 2.5]], nu=(4, 0))).tolist() == [0.0]
    assert np.isnan(_np(r([[9.0, 2.5]], nu=(1, 0)))[0])
    assert _np(RGI((gx, gx), q, method="cubic", bounds_error=False, fill_value=2.0)([[9.0, 2.5]], nu=(1, 0)))[0] == 2.0
    assert _np(RGI((gx, gx), q, method="slinear")([[1.5, 2.5]], nu=[1, 0])).tolist() == [3.0]
    assert _np(RGI((gx, gx), q, method="slinear")([[1.5, 2.5]], nu=[2, 0])).tolist() == [0.0]
    for meth in ("linear", "nearest", "pchip"):
        with pytest.raises(ValueError, match="'nu' is for the methods"):
            RGI((gx, gx), q, method=meth)([[1.5, 2.5]], nu=(1, 0))
    with pytest.raises(ValueError, match="one order per dimension"):
        r([[1.5, 2.5]], nu=(1,))
    with pytest.raises(ValueError, match=">= 0"):
        r([[1.5, 2.5]], nu=(-1, 0))
    with pytest.raises(TypeError, match="must be an int"):
        r([[1.5, 2.5]], nu=(1.0, 0))
    with pytest.raises(TypeError, match="sequence"):
        r([[1.5, 2.5]], nu=1)
    # a method other than the object's: its coefficients on the first call
    lin = RGI((gx, gx), q)
    assert bits_equal(lin([[1.5, 2.5]], method="cubic", nu=(1, 0)), r([[1.5, 2.5]], nu=(1, 0)))


# ---------------------------------------------------------------- shapes and arguments


def test_shapes():
    G = S.grid("grid/batch")  # (6, 5) and a batch of 2
    r = RGI(G.points, G.values)
    assert r(np.full((3, 4, 2), 0.5)).shape == (3, 4, 2)
    assert r([0.5, 0.5, 0.25, 0.25]).shape == (2, 2)  # 1-D: reshape(-1, d)
    assert r((np.array([0.1, 0.2, 0.3]), 0.5)).shape == (3, 2)
    assert r(np.zeros((0, 2))).shape == (0, 2)
    one = RGI(G.points, np.ascontiguousarray(G.values[..., 0]))
    v = one((0.5, 0.25))
    assert isinstance(v, float)  # the point of a tuple of scalars: a Python scalar (no 0-d arrays)
    assert v == _np(one([[0.5, 0.25]]))[0]
    assert one((np.array([[0.1], [0.2]]), np.array([0.3, 0.4, 0.5]))).shape == (2, 3)
    assert IP.interpn(G.points, G.values, [[0.5, 0.5]], method="nearest").shape == (1, 2)
    rec = refdata.data("scipy")["rgi/tuple_xi"]["value"]["array"]
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    assert _np(RGI((gx, gx), q)((np.array([1.5, 2.0]), np.array([0.5, 4.0])))).tolist() == rec


def test_attributes():
    G = S.grid("grid/2d-desc")
    r = RGI(G.points, G.values, method="pchip", bounds_error=False, fill_value=3)
    assert r.method == "pchip" and r.bounds_error is False and r.fill_value == 3.0
    assert isinstance(r.grid, tuple) and len(r.grid) == 2
    assert np.all(np.diff(np.asarray(r.grid[1])) > 0)
    assert r.values.shape == G.values.shape
    with pytest.raises(ValueError):
        r.values[0, 0] = 1.0
    assert RGI(G.points, G.values, fill_value=None).fill_value is None
    assert RGI(G.points, G.values + 0j, fill_value=1j).fill_value == 1j
    assert "RegularGridInterpolator" in repr(r)


def test_errors():
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    cases = [
        (lambda: RGI((gx, gx), q, method="Linear"), ValueError, "'method' must be"),
        (lambda: RGI((gx, gx), q, method=None), TypeError, "'method' must be"),
        (lambda: RGI((gx, gx), q, bounds_error="no"), TypeError, "'bounds_error' must be True or False"),
        (lambda: RGI((gx, gx), q, fill_value=1j), ValueError, "'fill_value' must be None or a real number"),
        (lambda: RGI((gx, gx), q, fill_value="a"), TypeError, "'fill_value' must be None or a number"),
        (lambda: RGI((gx, gx), q, fill_value=True), TypeError, "'fill_value' must be None or a number"),
        (lambda: RGI((gx, gx), q, fill_value=np.array([1.0])), TypeError, "'fill_value'"),
        (lambda: RGI(gx, q), TypeError, "'points' must be a tuple or a list"),
        (lambda: RGI((), q), ValueError, "an axis or more"),
        (lambda: RGI((gx, gx[[0, 2, 1, 3, 4, 5]]), q), ValueError, "dimension 1 must be strictly increasing or"),
        (lambda: RGI((gx, np.r_[gx[:-1], np.inf]), q), ValueError, "dimension 1 must be finite"),
        (lambda: RGI((gx, q), q), ValueError, "dimension 1 must be 1-D"),
        (lambda: RGI((gx[:5], gx), q), ValueError, "there are 5 points and 6 values in dimension 0"),
        (lambda: RGI((gx, gx, gx), q), ValueError, "there are 3 point arrays, but 'values' has 2 dimensions"),
        (lambda: RGI((gx[:3], gx), q[:3], method="cubic"), ValueError, "method 'cubic' needs at least 4 points"),
        (lambda: RGI((gx, gx), q, method="quintic")([[1.0, 1.0]], method="cubc"), ValueError, "'method' must"),
        (lambda: RGI((gx, gx), q)([[0.5, 0.2, 0.1]]), ValueError, "dimension 3, but the grid has dimension 2"),
        (lambda: RGI((gx, gx), q)([0.5, 0.2, 0.1]), ValueError, "not a whole number of 2-D points"),
        (lambda: RGI((gx,), gx)(0.5), ValueError, "at least 1 dimension"),
        (lambda: RGI((gx, gx), q)([[0.5 + 1j, 0.2]]), TypeError, "must be real"),
        (lambda: RGI((gx, gx), q)((0.5, 1j)), TypeError, "must be real"),
        (lambda: RGI((gx, gx), q)((np.zeros(2), np.zeros(3))), ValueError, "do not broadcast"),
        (lambda: RGI((gx, gx), q)((0.5,) * 3), ValueError, "has 3 coordinate arrays"),
        (lambda: RGI((gx, gx), q, solver=None), TypeError, "unexpected keyword argument 'solver'"),
        (lambda: RGI((gx, gx), q, "linear"), TypeError, "positional"),
        (lambda: IP.interpn((gx, gx), q, [[1.0, 1.0]], method="splinef2d"), ValueError, r"interpn\(\): 'method'"),
        (lambda: IP.interpn((gx, gx), q, [[9.0, 1.0]]), ValueError, r"interpn\(\): one of the requested xi"),
        (lambda: IP.interpn((gx, gx), q, [[1.0, 1.0]], "linear"), TypeError, r"interpn\(\)"),
    ]
    for f, exc, msg in cases:
        with pytest.raises(exc, match=msg):
            f()
    with pytest.raises(ValueError, match=r"^RegularGridInterpolator\(\): "):
        RGI((gx, gx), q, method="Linear")


@LIVE
def test_scipy_agrees_on_shapes_values_and_errors():
    """Non-rounding behaviour against scipy 1.18.1 (API.md §4 rule 2): shapes, nearest's bits, the other methods within
    rounding (cubic and quintic against scipy's exact-solve methods `*_legacy`; its default solve is iterative), the
    outside rules, and errors of the same kind."""
    import scipy.interpolate as si

    g = np.random.default_rng(57)
    for G in (S.grid(i) for i in S.GRID_IDS):
        for meth in METHODS:
            if any(p.size < NEED[meth] for p in G.points):
                continue
            finite = np.all(np.isfinite(G.values))
            if meth in ("slinear", "cubic", "quintic", "pchip") and not finite:  # scipy: errors (slinear: sd's NaN)
                continue
            for kw in ({}, {"bounds_error": False}, {"bounds_error": False, "fill_value": None}):
                q = G.xi if not kw else np.vstack([G.xi, _outside(G, g, 8)])
                got = _np(RGI(G.points, G.values, method=meth, **kw)(q))
                smeth = meth + "_legacy" if meth in ("cubic", "quintic") else meth
                want = si.RegularGridInterpolator(G.points, G.values, method=smeth, **kw)(q)
                assert got.shape == want.shape, (G.id, meth)
                if meth == "nearest":
                    assert bits_equal(got, want), (G.id, kw)
                    continue
                ok = np.isfinite(want)
                assert np.array_equal(np.isnan(got), np.isnan(want)) or not finite, (G.id, meth, kw)
                scale = np.nanmax(np.abs(G.values)) * (1 + np.max(np.abs(q))) ** DEGREE.get(meth, 1)
                tol = {"cubic": 1e-10, "quintic": 1e-8}.get(meth, 2.0**-46)
                assert np.all(np.abs(got[ok] - want[ok]) <= tol * scale), (G.id, meth, kw)
    gx = np.arange(6.0)
    q = np.add.outer(gx**2, gx**2)
    for call in (
        lambda R: R((gx, gx), q, method="Linear"),
        lambda R: R((gx[:3], gx), q[:3], method="cubic"),
        lambda R: R((gx, gx), q)([[9.0, 1.0]]),
        lambda R: R((gx, gx), q)([[0.5, 0.2, 0.1]]),
        lambda R: R((gx[:5], gx), q),
        lambda R: R((gx, gx), q, fill_value=1j),
        lambda R: R((gx, gx), q, method="linear")([[1.0, 1.0]], nu=(1, 0)),
    ):
        kinds = []
        for R in (RGI, si.RegularGridInterpolator):
            try:
                call(R)
                kinds.append(None)
            except Exception as e:  # noqa: BLE001
                kinds.append(type(e))
        assert kinds[0] is kinds[1] is ValueError, kinds


# ---------------------------------------------------------------- threads


def _thread_results():
    g = np.random.default_rng(58)
    out = []
    ax = [np.sort(g.random(300)), np.sort(g.random(240))]
    v = g.standard_normal((300, 240))
    q = g.random((200_000, 2)) * 1.2 - 0.1
    q[::997, 0] = np.nan
    for meth in METHODS:
        r = RGI(tuple(ax), v, method=meth, bounds_error=False, fill_value=None)
        out.append(r(q[:60_000] if meth in ("quintic", "pchip") else q))
    out.append(RGI(tuple(ax), v + 1j * v[::-1], bounds_error=False, fill_value=2.0)(q))
    out.append(RGI(tuple(ax), v, method="slinear", bounds_error=False)(q, nu=(1, 0)))
    out.append(RGI(tuple(ax), v, method="cubic", bounds_error=False)(q[:60_000], nu=(1, 2)))
    big = g.standard_normal((400, 350))
    out.append(RGI((np.arange(400.0), np.sort(g.random(350))), big, method="cubic")._spline(3, "t")[1])
    batch = g.standard_normal((40, 30, 50, 3))
    a3 = (np.sort(g.random(40)), np.sort(g.random(30)), np.sort(g.random(50)))
    q3 = g.random((50_000, 3))
    for meth in ("linear", "nearest", "cubic", "pchip"):
        out.append(RGI(a3, batch, method=meth, bounds_error=False, fill_value=None)(q3))
    return out


@pytest.mark.usefixtures("restore_threads")
def test_thread_counts():
    sd.set_num_threads(1)
    base = _thread_results()
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        got = _thread_results()
        assert all(bits_equal(a, b) for a, b in zip(base, got)), t


def test_linear_blocks_are_the_pointwise_kernel():
    # real values take the block path (corner values gathered, each step in base/lerp.hpp's block kernel; decision
    # IP99), complex values the point-by-point one: the same bits in the real part, under every outside rule, with
    # NaN, outside and infinite points and non-finite values
    g = np.random.default_rng(99)
    for shape in ((40,), (17, 23), (6, 7, 5)):
        pts = tuple(np.sort(g.random(n)) for n in shape)
        v = g.standard_normal(shape)
        v.flat[[1, 5, 9]] = [math.inf, -math.inf, math.nan]
        xi = np.stack([g.uniform(-0.2, 1.2, 3000) for _ in shape], axis=1)
        xi[::101, 0] = math.nan
        xi[::173, -1] = math.inf
        for fill in (None, math.nan, 2.5):
            real = _np(RGI(pts, v, bounds_error=False, fill_value=fill)(xi))
            cplx = _np(RGI(pts, v + 0j, bounds_error=False, fill_value=fill)(xi))
            assert bits_equal(real, np.ascontiguousarray(cplx.real)), (shape, fill)
