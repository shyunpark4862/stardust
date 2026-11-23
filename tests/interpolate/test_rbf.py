"""RBFInterpolator and NearestNDInterpolator (docs/api/interpolate.md §7; docs/design/interpolate.md §2.8, §3.8, §4;
milestone IP6).

- The contracts (refdata's RBF part): the solve's backward error componentwise within gamma_{4N} (|A^| + |L||D||L^T|)
  |c^| against the computed system and its sytrf factors, and the evaluation within the kernel and summation
  rounding of the computed coefficients' exact function (Arb), on every reference set and kernel, smoothing and
  complex values included; end to end against the exact interpolant (tests/interpolate/ref/rbf.npz) as scipy.
- neighbors=k: each point's value is the global interpolant of its k nearest data points (sorted by (distance,
  index), so ties go to the smaller index) to the bit, whatever the other points, their order, the split into calls
  or the thread count; k >= n is the global interpolant to the bit.
- NearestNDInterpolator: the brute force's nearest point under (distance, index), rescale as scipy's.
- The scipy differences of §2.8 (negative smoothing, duplicate points, a rank-deficient tail, a low degree)."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
CORE = sd._core.interpolate
REF = refdata.ref("rbf")
SETS = np.asarray(REF.inputs["set"])
A = sd.asarray


def _points(method, sid):
    """Positions of a set's points in the method's arrays, in the set's order."""
    M = REF.methods[method]
    pos = np.flatnonzero(SETS[M.index] == sid)
    order = np.argsort(np.asarray(REF.inputs["index"])[M.index[pos]], kind="stable")
    return pos[order]


def _cases():
    for method, M in REF.methods.items():
        for sid in sorted(set(SETS[M.index])):
            yield method, sid


def _build(method, sc, **kw):
    kernel, eps, sm, nb = refdata.rbf_options(method)
    return IP.RBFInterpolator(sc.y, sc.d, kernel=kernel, epsilon=eps, smoothing=sm, neighbors=nb, **kw)


# ---------------------------------------------------------------- contracts


def _solve_units(r, y, d):
    c = np.asarray(r._coeffs)
    if np.iscomplexobj(c):
        c = np.ascontiguousarray(c).view(float)
        d = np.ascontiguousarray(d.reshape(len(d), -1)).view(float)
    A_ = np.asarray(CORE.rbf_matrix(A(y), r._smoothing, r._kid, r._epsilon, r._powers))
    L, D, _ = sd.linalg.ldl(A(A_))
    res = refdata.rbf_residual(A_, c, d.reshape(len(d), -1))
    bound = refdata.rbf_backward_bound(A_, c, np.asarray(L), np.asarray(D))
    return float(np.max(res / bound))


def test_backward_error_of_the_solve():
    worst = 0.0
    for method, sid in _cases():
        if method.endswith("_k8"):
            continue
        sc = S.scatter(sid)
        u = _solve_units(_build(method, sc), sc.y, sc.d)
        assert u <= 1, (method, sid, u)
        worst = max(worst, u)
    assert worst > 0  # the residuals round: the contract is not vacuous
    # complex values and a smoothing array: two real columns of the same real system
    sc = S.scatter("scatter/uniform3d")
    d = np.stack([sc.d, np.cos(sc.d)], axis=1) * (1 - 2j)
    sm = np.linspace(0, 0.2, len(sc.d))
    assert _solve_units(IP.RBFInterpolator(sc.y, d, smoothing=sm, kernel="cubic"), sc.y, d) <= 1


def _eval_units(r, y, x, got, col=0):
    c = np.asarray(r._coeffs)[:, col]
    pw, sh, scl = np.asarray(r._powers), np.asarray(r._shift), np.asarray(r._scale)
    worst = 0.0
    for i, xi in enumerate(x):
        f, phis, monos, rs = refdata.rbf_exact(r._kernel, r._epsilon, y, xi, c, pw, sh, scl)
        bound = refdata.rbf_eval_bound(r._kernel, y.shape[1], c, phis, monos, rs, pw)
        err = abs(float(got[i]) - float(f.mid()))
        assert float(f.rad()) < 1e-3 * bound  # the reference is settled far below the bound
        worst = max(worst, err / bound)
    return worst


def test_evaluation_contract():
    worst = 0.0
    for method, sid in _cases():
        if method.endswith("_k8"):
            continue
        sc = S.scatter(sid)
        r = _build(method, sc)
        x = np.vstack([sc.xi, sc.y[:3]])  # data points too: phi(0)
        u = _eval_units(r, sc.y, x, np.asarray(r(x)))
        assert u <= 1, (method, sid, u)
        worst = max(worst, u)
    assert worst > 0


@LIVE
def test_end_to_end_as_scipy():
    # the exact interpolant of the float data (Arb): sd's error is scipy's size (both solve the same system by
    # Bunch-Kaufman; the conditioning decides, from 1e-16 (linear) to 2e-7 (gaussian, epsilon 3) relatively)
    import scipy.interpolate as si

    for method, sid in _cases():
        sc = S.scatter(sid)
        pos = _points(method, sid)
        M = REF.methods[method]
        kernel, eps, sm, nb = refdata.rbf_options(method)
        got = np.asarray(_build(method, sc)(sc.xi))
        want = si.RBFInterpolator(sc.y, sc.d, kernel=kernel, epsilon=eps, smoothing=sm, neighbors=nb)(sc.xi)
        scale = np.max(np.abs(M.hi[pos]))
        mine = np.max(np.abs((got - M.hi[pos]) - M.lo[pos])) / scale
        theirs = np.max(np.abs((want - M.hi[pos]) - M.lo[pos])) / scale
        assert mine <= 8 * theirs + 2.0**-48, (method, sid, mine, theirs)


def test_residual_at_the_data_points():
    # the interpolant meets the data within the solve's backward error (smoothing 0)
    for sid in ("scatter/uniform2d", "scatter/lattice2d"):
        sc = S.scatter(sid)
        for kernel in ("thin_plate_spline", "linear", "inverse_quadratic"):
            r = IP.RBFInterpolator(sc.y, sc.d, kernel=kernel, epsilon=2.0)
            got = np.asarray(r(sc.y))
            assert np.max(np.abs(got - sc.d)) <= 1e-12 * np.max(np.abs(sc.d)), (sid, kernel)


@LIVE
def test_values_and_shapes_as_scipy():
    import scipy.interpolate as si

    g = np.random.default_rng(61)
    y = g.random((30, 3))
    d = np.stack([np.sin(y.sum(1)), y[:, 0] * 1j + y[:, 1]], axis=1).reshape(30, 2, 1)
    x = g.random((4, 5, 3))
    r = IP.RBFInterpolator(y, d, kernel="quintic", smoothing=1e-3)
    s = si.RBFInterpolator(y, d, kernel="quintic", smoothing=1e-3)
    got = np.asarray(r(x))
    assert got.shape == (4, 5, 2, 1) and got.dtype == np.complex128
    assert np.allclose(got, s(x.reshape(-1, 3)).reshape(4, 5, 2, 1), rtol=1e-11, atol=1e-11)
    one = IP.RBFInterpolator(y, d[:, 0, 0].real)
    assert isinstance(one(x[0, 0]), float)  # one point, scalar values: a Python float
    assert np.asarray(one(x[0])).shape == (5,)
    assert np.asarray(one(np.empty((0, 3)))).shape == (0,)
    # a NaN point is NaN; the global interpolant computes its formula at an infinite point as scipy
    out = np.asarray(
        IP.RBFInterpolator(y, d[:, 0, 0].real, kernel="gaussian", epsilon=1.0)([[np.nan, 0, 0], [np.inf, 0, 0]])
    )
    ref = si.RBFInterpolator(y, d[:, 0, 0].real, kernel="gaussian", epsilon=1.0)([[np.nan, 0, 0], [np.inf, 0, 0]])
    assert np.isnan(out[0]) and np.allclose(out[1], ref[1], rtol=1e-12)


def test_degree_minus_one_and_one_point():
    y = np.array([[0.3, 0.1]])
    r = IP.RBFInterpolator(y, [2.5], kernel="gaussian", epsilon=1.0, degree=-1)
    assert r([0.3, 0.1]) == 2.5  # phi(0) = 1: c = 2.5
    assert IP.RBFInterpolator(y, [2.5], kernel="linear")([1.0, 1.0]) == 2.5  # the constant tail alone


# ---------------------------------------------------------------- neighbors


def _nearest_sets(y, x, k):
    """Each point's k nearest data points by (distance, index), sorted: the brute force of sd's distance kernel."""
    dist = np.asarray(sd.spatial.cdist(A(x), A(y)))
    return [np.sort(np.lexsort((np.arange(len(y)), row))[:k]) for row in dist]


def _local_by_global(y, d, x, k, **kw):
    out = []
    for xi, s in zip(x, _nearest_sets(y, x, k)):
        out.append(IP.RBFInterpolator(y[s], d[s], **kw)(xi))
    return np.array(out)


def test_neighbors_is_the_global_interpolant_of_each_set():
    g = np.random.default_rng(62)
    y = g.random((200, 2))
    d = np.sin(4 * y[:, 0]) * y[:, 1]
    x = g.random((60, 2))
    for k, kw in ((8, {}), (12, {"kernel": "gaussian", "epsilon": 4.0}), (6, {"kernel": "linear"})):
        got = np.asarray(IP.RBFInterpolator(y, d, neighbors=k, **kw)(x))
        assert bits_equal(got, _local_by_global(y, d, x, k, **kw)), k


def test_neighbor_ties_never_change_results():
    # the lattice of samples.scatter: many equal distances; centres of the cells and midpoints of the edges have
    # 4 and 2 equidistant neighbours, so k = 1, 2, 3, 5 cut through ties
    sc = S.scatter("scatter/lattice2d")
    y, d = sc.y, sc.d
    h = 0.25
    x = np.array([[i * h + h / 2, j * h + h / 2] for i in range(4) for j in range(4)]
                 + [[i * h + h / 2, j * h] for i in range(4) for j in range(5)])  # fmt: skip
    for k, kw in ((1, {"kernel": "gaussian", "epsilon": 2.0}), (2, {"kernel": "linear"}), (3, {"kernel": "linear"}),
                  (5, {}), (9, {"kernel": "inverse_quadratic", "epsilon": 3.0})):  # fmt: skip
        r = IP.RBFInterpolator(y, d, neighbors=k, **kw)
        got = np.asarray(r(x))
        assert bits_equal(got, _local_by_global(y, d, x, k, **kw)), k  # the smaller indices win the ties
        perm = np.random.default_rng(k).permutation(len(x))
        assert bits_equal(np.asarray(r(x[perm])), got[perm]), k  # the order of the points
        assert bits_equal(np.asarray(r(np.vstack([x, x]))), np.concatenate([got, got])), k  # repeated points
        assert bits_equal(np.array([r(p) for p in x]), got), k  # one point a call: other points change nothing


def test_neighbors_at_least_n_is_global():
    sc = S.scatter("scatter/uniform3d")
    g = IP.RBFInterpolator(sc.y, sc.d, kernel="cubic")
    for k in (len(sc.y), len(sc.y) + 5):
        assert bits_equal(np.asarray(IP.RBFInterpolator(sc.y, sc.d, kernel="cubic", neighbors=k)(sc.xi)),
                          np.asarray(g(sc.xi)))  # fmt: skip


def test_neighbors_nonfinite_points_and_failures():
    sc = S.scatter("scatter/uniform2d")
    r = IP.RBFInterpolator(sc.y, sc.d, neighbors=10)
    x = sc.xi[:4].copy()
    x[1, 0], x[3, 1] = np.nan, np.inf
    got = np.asarray(r(x))
    assert np.isnan(got[1]) and np.isnan(got[3])  # no neighbourhood (scipy: ValueError)
    assert bits_equal(got[[0, 2]], np.asarray(r(x[[0, 2]])))
    assert np.isnan(np.asarray(r(np.full((2, 2), np.nan)))).all()
    from stardust.linalg import LinAlgError

    t = np.linspace(0, 1, 30)
    y = np.column_stack([t, np.where(t < 0.5, 0.0, (t - 0.5) ** 2)])  # collinear at the left, a parabola at the right
    rr = IP.RBFInterpolator(y, t, neighbors=6)
    assert np.isfinite(np.asarray(rr([[0.9, 0.16]]))).all()
    with pytest.raises(LinAlgError, match=r"the 6 neighbors of point 1 of 'x' do not have full column rank \(2/3\)"):
        rr([[0.9, 0.16], [0.1, 0.0], [0.2, 0.0]])


# ---------------------------------------------------------------- NearestNDInterpolator


def test_nearest_is_the_brute_force_with_ties_to_the_smaller_index():
    for sid in S.SCATTER_IDS:
        sc = S.scatter(sid)
        vals = np.arange(len(sc.y), dtype=float)
        got = np.asarray(IP.NearestNDInterpolator(sc.y, vals)(sc.xi))
        assert np.array_equal(got, [s[0] for s in _nearest_sets(sc.y, sc.xi, 1)]), sid
    sc = S.scatter("scatter/lattice2d")
    mid = np.array([[0.125, 0.125], [0.375, 0.625], [0.5, 0.375], [1.0, 0.875]])  # 4, 4, 2 and 2 at one distance
    got = np.asarray(IP.NearestNDInterpolator(sc.y, np.arange(25.0))(mid))
    assert got.tolist() == [0.0, 7.0, 11.0, 23.0]


@LIVE
def test_nearest_as_scipy():
    import scipy.interpolate as si

    for sid in S.SCATTER_IDS:
        sc = S.scatter(sid)
        for rescale in (False, True):
            stretch = [1, 50] if sc.y.shape[1] == 2 else [1, 1, 50]  # rescale changes the nearest points
            yy, xx = sc.y * stretch, sc.xi * stretch
            got = np.asarray(IP.NearestNDInterpolator(yy, sc.d, rescale=rescale)(xx))
            want = si.NearestNDInterpolator(yy, sc.d, rescale=rescale)(xx)
            # where two points are nearest scipy's tree picks either; sd the smaller index (the brute force)
            dist = np.sort(np.asarray(sd.spatial.cdist(A(xx), A(yy))), axis=1)
            unique = dist[:, 0] < dist[:, 1]
            assert np.array_equal(got[unique], want[unique]), (sid, rescale)
            if not rescale:
                assert np.array_equal(got, sc.d[[s[0] for s in _nearest_sets(yy, xx, 1)]]), sid


def test_nearest_shapes_and_nonfinite_points():
    g = np.random.default_rng(63)
    y = g.random((50, 2))
    v = g.random((50, 3)) + 1j
    n = IP.NearestNDInterpolator(y, v)
    x = g.random((4, 6, 2))
    got = np.asarray(n(x))
    assert got.shape == (4, 6, 3) and got.dtype == np.complex128
    assert bits_equal(np.asarray(n((x[..., 0], x[..., 1]))), got)  # a tuple of coordinate arrays
    assert bits_equal(np.asarray(n((x[..., 0], 0.5))), np.asarray(n(np.stack([x[..., 0], np.full((4, 6), 0.5)], -1))))
    s = IP.NearestNDInterpolator(y, v[:, 0].real)
    assert isinstance(s([0.2, 0.3]), float) and s((0.2, 0.3)) == s([0.2, 0.3])
    out = np.asarray(s([[np.nan, 0.1], [0.2, np.inf], [0.2, 0.3]]))
    assert np.isnan(out[:2]).all() and out[2] == s([0.2, 0.3])  # scipy: ValueError
    one_d = IP.NearestNDInterpolator(np.array([[0.0], [1.0], [2.0]]), [5.0, 6.0, 7.0])  # scipy needs 2 coordinates
    assert np.asarray(one_d([[0.4], [1.5], [1.6]])).tolist() == [5.0, 6.0, 7.0]


# ---------------------------------------------------------------- scipy's differences (§2.8)


@LIVE
def test_scipy_differences():
    import warnings

    import scipy.interpolate as si

    from stardust.linalg import LinAlgError

    sc = S.scatter("scatter/uniform2d")
    with pytest.raises(ValueError, match="'smoothing' must be finite and >= 0"):
        IP.RBFInterpolator(sc.y, sc.d, smoothing=-1.0)
    assert np.isfinite(si.RBFInterpolator(sc.y, sc.d, smoothing=-1.0)(sc.xi[:2])).all()  # scipy takes it
    dup = S.scatter("scatter/duplicate2d")
    with pytest.raises(LinAlgError, match="data points 2 and 7 are equal"):
        IP.RBFInterpolator(dup.y, dup.d)
    assert np.isfinite(si.RBFInterpolator(dup.y, dup.d)(dup.xi[:2])).all()  # scipy: values without a word
    IP.RBFInterpolator(dup.y, dup.d, smoothing=1e-3)  # smoothing makes it solvable
    col = S.scatter("scatter/collinear2d")
    with pytest.raises(LinAlgError, match=r"do not have full column rank \(2/3\)"):
        IP.RBFInterpolator(col.y, col.d)
    assert np.max(np.abs(si.RBFInterpolator(col.y, col.d)(col.xi))) > 1e9  # scipy: 1e11 values
    IP.RBFInterpolator(col.y, col.d, kernel="linear")  # degree 0: full rank
    for deg in (0, -1):
        with pytest.raises(ValueError, match="'degree' must be at least 1 for kernel 'thin_plate_spline'"):
            IP.RBFInterpolator(sc.y, sc.d, degree=deg)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        si.RBFInterpolator(sc.y, sc.d, degree=0)
        si.RBFInterpolator(sc.y, sc.d, degree=-1)
    assert len(w) == 1  # scipy warns for 0 and takes -1 silently
    with pytest.raises(ValueError, match="'epsilon' must be finite and positive"):
        IP.RBFInterpolator(sc.y, sc.d, kernel="gaussian", epsilon=-3.0)
    # scipy's epsilon multiplies the points, so a negative one acts as its absolute value
    a = si.RBFInterpolator(sc.y, sc.d, kernel="gaussian", epsilon=-3.0)(sc.xi)
    b = np.asarray(IP.RBFInterpolator(sc.y, sc.d, kernel="gaussian", epsilon=3.0)(sc.xi))
    assert np.allclose(a, b, rtol=1e-5)


# ---------------------------------------------------------------- threads


@pytest.mark.usefixtures("restore_threads")
def test_thread_counts():
    def results():
        g = np.random.default_rng(64)
        y = g.random((500, 2))
        d = np.stack([np.sin(5 * y[:, 0]), y[:, 1] ** 2], axis=1)
        x = g.random((20_000, 2))
        out = [IP.RBFInterpolator(y, d)(x), IP.RBFInterpolator(y, d, kernel="gaussian", epsilon=3.0)(x[:5000])]
        out.append(CORE.rbf_matrix(A(y), A(np.zeros(500)), 1, 1.0, A(np.array([[0, 0], [1, 0], [0, 1]]))))
        yb = g.random((3000, 3))
        out.append(IP.RBFInterpolator(yb, np.cos(yb.sum(1)), neighbors=20)(g.random((4000, 3))))
        out.append(IP.RBFInterpolator(yb, np.cos(yb.sum(1)) * 1j, neighbors=7, kernel="linear")(g.random((4000, 3))))
        yl = S.scatter("scatter/lattice2d").y
        out.append(IP.RBFInterpolator(yl, yl[:, 0], neighbors=3, kernel="linear")(g.random((3000, 2))))
        yn = g.random((10_000, 2))
        out.append(IP.NearestNDInterpolator(yn, g.random(10_000), rescale=True)(g.random((100_000, 2))))
        return out

    sd.set_num_threads(1)
    base = results()
    for t in (3, 10):
        sd.set_num_threads(t)
        for k, (a, b) in enumerate(zip(base, results())):
            assert bits_equal(a, b), (t, k)


def test_monomial_powers_are_scipys():
    from scipy.interpolate._rbfinterp_common import _monomial_powers_impl

    from stardust.interpolate._rbf import monomial_powers

    for m in (1, 2, 3, 4):
        for deg in (-1, 0, 1, 2, 3):
            want = _monomial_powers_impl(m, deg) if deg >= 0 else []
            assert np.asarray(monomial_powers(m, deg)).tolist() == want, (m, deg)
    assert math.comb(3 + 2, 2) == np.asarray(monomial_powers(2, 3)).shape[0]


@pytest.mark.parametrize("kernel", ["linear", "thin_plate_spline", "cubic", "quintic", "multiquadric",
                                    "inverse_multiquadric", "inverse_quadratic", "gaussian"])  # fmt: skip
def test_lanes_and_single_points_give_the_same_bits(kernel):
    # four points at a time in NEON lanes, a single point alone (decision IP100): the same bits, the data points
    # themselves (spatial's careful path in their lane) and far points (exp past its lanes) included
    g = np.random.default_rng(100)
    y = g.random((60, 2))
    eps = {} if kernel in ("linear", "thin_plate_spline", "cubic", "quintic") else {"epsilon": 2.0}
    r = IP.RBFInterpolator(y, np.stack([np.sin(4 * y[:, 0]), y[:, 1]], axis=1), kernel=kernel, **eps)
    x = np.concatenate([g.random((37, 2)), y[:9], [[40.0, -3.0], [1e-170, 1e-170]]])
    got = np.asarray(r(x))
    one = np.stack([np.asarray(r(x[i : i + 1]))[0] for i in range(len(x))])
    assert bits_equal(got, one)
