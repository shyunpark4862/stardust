"""make_interp_spline (docs/design/interpolate.md §2.5, §3.4, §4; milestone IP3): the knots are scipy's bit for bit
(tests/interpolate/ref/bspline.npz holds them), the coefficients within the construction contract of the exact ones
(exact rationals rounded once; refdata.collocation_bound), the stability of elimination without pivoting with the
derivative rows after their end's interpolation row (decision IP35; exact growth of the systems), when the solve
pivots instead (decision IP53), every bc_type, batches, axes and complex values against scipy 1.18.1."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
REF = refdata.ref("bspline")


def _ref(method, cid):
    """(group, hi, lo) of a curve's reference coefficients, in coefficient order (None: no reference)."""
    m = REF.methods[method]
    pos = np.flatnonzero(REF.inputs["set"][m.index] == cid)
    if not pos.size:
        return None
    order = np.argsort(REF.inputs["index"][m.index[pos]], kind="stable")
    group = int(np.flatnonzero(REF.group_inputs["set"] == cid)[0])
    return group, m.hi[pos[order]], m.lo[pos[order]]


@pytest.mark.parametrize("method", refdata.BSPLINE_METHODS)
def test_knots_and_the_construction_contract(method):
    k, bc = refdata.bspline_args(method)
    worst, seen = 0.0, 0
    for cid in S.real_curve_ids():
        got_ref = _ref(method, cid)
        if got_ref is None:
            continue
        g, hi, lo = got_ref
        c = S.curve(cid)
        if cid == "subnormal/x" and bc in ("natural", "clamped"):
            with pytest.raises(ValueError, match="the derivatives of the end conditions overflow"):
                IP.make_interp_spline(c.x, c.y, k=k, bc_type=bc)
            continue
        b = IP.make_interp_spline(c.x, c.y, k=k, bc_type=bc)
        assert np.array_equal(np.asarray(b.t), REF.group_inputs["t_" + method][g]), cid  # scipy's knots
        got = np.asarray(b.c)
        err = np.abs((got - hi) - lo)
        seen += 1
        if k <= 1:
            assert np.array_equal(got, c.y), cid  # the data itself
            continue
        bound = refdata.collocation_bound(c.x, b.t, k, bc, got)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(err == 0, 0.0, err / bound)
        assert np.all(u <= 1), (cid, float(u.max()))
        worst = max(worst, float(u.max()))
    assert seen
    assert k <= 1 or worst > 0  # the contract is not vacuous


def _growth(rows):
    """max (|L||U|)_ij / |A_ij| of exact elimination without pivoting (inf: a zero pivot or fill outside A)."""
    n = len(rows)
    U = [list(r) for r in rows]
    L = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    for j in range(n):
        if U[j][j] == 0:
            return math.inf
        for i in range(j + 1, n):
            if U[i][j]:
                L[i][j] = U[i][j] / U[j][j]
                for c in range(j, n):
                    U[i][c] -= L[i][j] * U[j][c]
    worst = Fraction(0)
    for i in range(n):
        for j in range(n):
            s = sum((abs(L[i][p]) * abs(U[p][j]) for p in range(min(i, j) + 1)), Fraction(0))
            if rows[i][j] == 0:
                if s:
                    return math.inf
                continue
            worst = max(worst, s / abs(rows[i][j]))
    return float(worst)


def _rows(x, k, left, right):
    """The collocation rows of the augmented knots in IP35's order (exact): x[0], the left derivative orders, the
    inner nodes, the right ones, x[-1]."""
    t = np.r_[(x[0],) * k, x, (x[-1],) * k]
    T = [Fraction(float(v)) for v in t]
    X = [Fraction(float(v)) for v in x]
    nt = len(T) - k - 1

    def row(v, nu):
        r = [Fraction(0)] * nt
        l = refdata.knot_interval(T, k, v)
        for s, b in enumerate(refdata._basis(T, k, l, v, nu, Fraction(0))):
            r[l - k + s] = b
        return r

    return ([row(X[0], 0)] + [row(X[0], o) for o in left] + [row(v, 0) for v in X[1:-1]]
            + [row(X[-1], o) for o in right] + [row(X[-1], 0)])  # fmt: skip


def test_elimination_without_pivoting_is_stable_in_ip35_order():
    """§3.4: the componentwise growth max (|L||U|)_ij / |A_ij| is 1 for the not-a-knot systems (totally nonnegative)
    and for derivative orders 1 .. r at each end (a triangular end block), and below 2 for k = 3's natural condition
    (1 + (1 - r)/(1 + r) at each end, r the ratio of the end spacing to the two end spacings); scipy's order (the
    derivative rows first) has a zero pivot."""
    g = np.random.default_rng(31)
    worst = {}
    for _ in range(60):
        n = int(g.integers(2, 9))
        h = 10.0 ** g.uniform(-5, 5, n - 1)
        x = np.cumsum(np.r_[0.0, h])
        x /= x[-1]
        if len(set(x)) < n:
            continue
        for k, left, right in ((3, [2], [2]), (3, [1], [1]), (5, [1, 2], [2, 1]), (4, [1, 2], [1]), (3, [2], [1])):
            if len(left) + len(right) != k - 1:
                continue
            gr = _growth(_rows(x, k, left, right))
            worst[(k, tuple(left), tuple(right))] = max(worst.get((k, tuple(left), tuple(right)), 0.0), gr)
            if (k, left, right) == (3, [2], [2]) and n >= 3:
                hh = np.diff(x)
                r0, r1 = hh[0] / (hh[0] + hh[1]), hh[-1] / (hh[-1] + hh[-2])
                assert gr <= max(1 + (1 - r0) / (1 + r0), 1 + (1 - r1) / (1 + r1)) * (1 + 1e-12)
        for k in (3, 5):  # not-a-knot: totally nonnegative
            if n >= k + 1:
                t = np.r_[(x[0],) * (k + 1), x[(k + 1) // 2 : -((k + 1) // 2)], (x[-1],) * (k + 1)]
                T = [Fraction(float(v)) for v in t]
                rows = []
                for v in x:
                    V = Fraction(float(v))
                    l = refdata.knot_interval(T, k, V)
                    r = [Fraction(0)] * (len(T) - k - 1)
                    for s, b in enumerate(refdata._basis(T, k, l, V, 0, Fraction(0))):
                        r[l - k + s] = b
                    rows.append(r)
                assert _growth(rows) == 1.0
    assert worst[(3, (2,), (2,))] < 2
    for key in ((3, (1,), (1,)), (5, (1, 2), (2, 1)), (4, (1, 2), (1,))):
        assert worst[key] == 1.0, key
    x = np.linspace(0, 1, 7)
    rows = _rows(x, 3, [2], [2])
    scipy_order = [rows[1], rows[0], *rows[2:-2], rows[-1], rows[-2]]
    assert _growth(scipy_order) == math.inf


def test_other_derivative_orders_pivot():
    """IP53: orders that are not 1 .. r (nor k = 3's second derivative alone) pivot; without pivoting their growth
    grows like 1/h^2 with the first spacing h (1.1e8 here)."""
    x = np.array([0.0, 1e-8, 1.0, 2.0, 3.0, 4.0, 5.0]) / 5
    assert _growth(_rows(x, 5, [2, 3], [3, 2])) > 1e8
    x = np.linspace(0, 1, 7) ** 2
    y = np.sin(3 * x)
    for k, bc in ((5, ([(2, 0.5), (3, -1.0)], [(3, 0.2), (2, 0.0)])), (5, ([(1, 0.5), (3, 1.0)], [(3, 0.2), (1, 0.0)])),
                  (4, ([(2, 1.0)], [(1, 0.0), (2, 0.3)]))):  # fmt: skip
        mine = IP.make_interp_spline(x, y, k=k, bc_type=bc)
        theirs = si.make_interp_spline(x, y, k=k, bc_type=bc)
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-12, atol=1e-12)
    # knots given with end conditions pivot too (the proof is for make_interp_spline's own knots)
    t = np.r_[(0.0,) * 4, 0.05, 0.2, 0.35, 0.6, (1.0,) * 4]
    mine = IP.make_interp_spline(x, y, k=3, t=t, bc_type=([(1, 0.0)], None))
    assert np.allclose(np.asarray(mine(x)), y, atol=1e-13)


@LIVE
def test_every_bc_type_follows_scipy():
    g = np.random.default_rng(32)
    x = np.sort(g.random(11))
    y = g.standard_normal(11)
    yp = np.r_[y[:-1], y[0]]
    q = np.linspace(-0.3, 1.3, 61)
    cases = [(k, None) for k in range(6)] + [
        (3, "not-a-knot"),
        (3, "natural"),
        (3, "clamped"),
        (3, ([(1, 2.0)], [(2, -1.0)])),
        (3, ("natural", [(1, 0.5)])),
        (5, ([(1, 0.0), (2, 1.0)], [(2, 0.0), (1, 3.0)])),
        (2, ([(1, 1.0)], None)),
        (4, (None, [(1, 0.0), (2, 0.0), (3, 1.0)])),
        (3, ([(1, 1.0), (2, 0.0)], None)),
    ]
    for k, bc in cases + [(k, "periodic") for k in range(6)]:
        yy = yp if bc == "periodic" else y
        mine, theirs = IP.make_interp_spline(x, yy, k=k, bc_type=bc), si.make_interp_spline(x, yy, k=k, bc_type=bc)
        assert np.array_equal(np.asarray(mine.t), theirs.t), (k, bc)
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-12, atol=1e-12), (k, bc)
        assert mine.extrapolate == theirs.extrapolate and mine.k == theirs.k, (k, bc)
        for nu in range(k + 1):
            assert np.allclose(np.asarray(mine(q, nu=nu)), theirs(q, nu=nu), rtol=1e-9, atol=1e-9), (k, bc, nu)
    # periodic with n <= k: scipy's full system
    xs, ys = np.array([0.0, 1.0, 2.5, 3.0]), np.array([0.0, 1.0, 2.0, 0.0])
    for k in (3, 4, 5, 6):
        mine = IP.make_interp_spline(xs, ys, k=k, bc_type="periodic")
        theirs = si.make_interp_spline(xs, ys, k=k, bc_type="periodic")
        assert np.allclose(np.asarray(mine.c), theirs.c, rtol=1e-12, atol=1e-12), k
    # knots given
    t = np.r_[(x[0],) * 4, x[2:-2], (x[-1],) * 4]
    assert np.allclose(np.asarray(IP.make_interp_spline(x, y, k=3, t=t).c), si.make_interp_spline(x, y, k=3, t=t).c,
                       atol=1e-12)  # fmt: skip
    t1 = np.r_[x[0], x, x[-1]]
    assert np.array_equal(np.asarray(IP.make_interp_spline(x, y, k=1, t=t1).c), y)


@LIVE
def test_batches_axes_and_complex_values():
    g = np.random.default_rng(33)
    x = np.sort(g.random(9))
    y = g.standard_normal((2, 9, 3))
    q = g.random((4, 5)) * 1.4 - 0.2
    for k, bc in ((3, None), (3, "natural"), (5, None), (3, ([(1, np.ones((2, 3)))], [(2, np.zeros((2, 3)))]))):
        mine = IP.make_interp_spline(x, y, k=k, bc_type=bc, axis=1)
        theirs = si.make_interp_spline(x, y, k=k, bc_type=bc, axis=1)
        got = np.asarray(mine(q))
        assert got.shape == theirs(q).shape == (2, 4, 5, 3) and mine.axis == theirs.axis == 1
        assert np.allclose(got, theirs(q), rtol=1e-12, atol=1e-12)
        assert np.asarray(mine.c).shape == theirs.c.shape
        # a column alone gives the bits it gets in the batch
        bc1 = bc if not isinstance(bc, tuple) else ([(1, 1.0)], [(2, 0.0)])
        one = IP.make_interp_spline(x, y[1, :, 2], k=k, bc_type=bc1)
        assert np.array_equal(np.asarray(one.c), np.asarray(mine.c)[:, 1, 2])
        assert np.array_equal(np.asarray(one(q)), got[1, :, :, 2])
    yp = y.copy()
    yp[:, -1] = yp[:, 0]
    mine = IP.make_interp_spline(x, yp, k=3, bc_type="periodic", axis=1)
    assert np.allclose(
        np.asarray(mine(q)), si.make_interp_spline(x, yp, k=3, bc_type="periodic", axis=1)(q), atol=1e-12
    )
    yc = y[0, :, 0] + 1j * y[1, :, 0]
    mine = IP.make_interp_spline(x, yc)
    assert np.allclose(np.asarray(mine(q)), si.make_interp_spline(x, yc)(q), atol=1e-12)
    real = IP.make_interp_spline(x, y[0, :, 0])
    assert np.array_equal(np.asarray(mine.c).real, np.asarray(real.c))  # the parts apart
    cplx_bc = IP.make_interp_spline(x, y[0, :, 0], bc_type=([(1, 1j)], [(1, 0.0)]))
    assert np.asarray(cplx_bc.c).dtype == np.complex128


def test_periodic_check_and_defaults():
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    ok = IP.make_interp_spline(x, [1.0, 2.0, 0.0, 3.0, 1.0 + 1e-16], bc_type="periodic")
    assert ok.extrapolate == "periodic"
    with pytest.raises(ValueError, match="for bc_type='periodic'"):  # IP28: CubicSpline's tolerance
        IP.make_interp_spline(x, [1.0, 2.0, 0.0, 3.0, 1.0 + 1e-14], bc_type="periodic")
    # scipy: k = 0, 1 ignore 'periodic' after the check and extrapolate
    for k in (0, 1):
        assert IP.make_interp_spline(x, [1.0, 2.0, 0.0, 3.0, 1.0], k=k, bc_type="periodic").extrapolate is True
    assert IP.make_interp_spline(x, x**2).extrapolate is True


def test_k0_and_k1():
    x = np.array([0.0, 1.0, 3.0, 4.0])
    y = np.array([1.0, -2.0, 5.0, 0.5])
    b0 = IP.make_interp_spline(x, y, k=0)
    assert np.asarray(b0.t).tolist() == [0.0, 1.0, 3.0, 4.0, 4.0]
    got = np.asarray(b0([-1.0, 0.0, 0.5, 1.0, 3.999, 4.0, 5.0]))
    assert got.tolist() == [1.0, 1.0, 1.0, -2.0, 5.0, 0.5, 0.5]  # a step continuous from the left; y[-1] at x[-1]
    b1 = IP.make_interp_spline(x, y, k=1)
    assert np.asarray(b1.t).tolist() == [0.0, 0.0, 1.0, 3.0, 4.0, 4.0]
    assert np.array_equal(np.asarray(b1(x)), y)
    assert b1(5.0) == -4.0  # the end piece extended


def test_construction_is_direct_and_quick_for_a_long_column():
    n = 200_000
    x = np.arange(n) / (n - 1)
    y = np.sin(40 * x)
    b = IP.make_interp_spline(x, y)
    q = (np.arange(1000) + 0.5) / 1000
    assert np.max(np.abs(np.asarray(b(q)) - np.sin(40 * q))) < 1e-12


def test_knots_given_without_conditions_solve_without_pivoting():
    """IP114: with no derivative condition the collocation matrix is totally nonnegative whatever the knots, so
    the knots given are solved without pivoting too: make_interp_spline's own knots given as t give the bits of
    t=None, and on spacings that span 10^10 (where partial pivoting leaves the contract by up to 10^6) the
    coefficients stay within the construction contract against the exact rational solve."""
    core = sd._core.interpolate
    g = np.random.default_rng(41)
    for k in (2, 3, 4, 5):
        x = np.cumsum(np.r_[0.0, 10.0 ** g.uniform(-5, 5, 11)])
        y = g.standard_normal(12)
        t = np.asarray(core.bspline_knots(sd.asarray(x), k, 0))
        mine = IP.make_interp_spline(x, y, k=k, t=t)
        assert np.asarray(mine.c).tobytes() == np.asarray(IP.make_interp_spline(x, y, k=k).c).tobytes(), k
        T = [Fraction(float(v)) for v in t]
        n = len(x)
        A = []
        for v in x:
            V = Fraction(float(v))
            row = [Fraction(0)] * n
            l = refdata.knot_interval(T, k, V)
            for s, b in enumerate(refdata._basis(T, k, l, V, 0, Fraction(0))):
                row[l - k + s] = b
            A.append(row)
        b = [Fraction(float(v)) for v in y]
        for i in range(n):  # exact elimination (any nonzero pivot)
            p = next(r for r in range(i, n) if A[r][i] != 0)
            A[i], A[p], b[i], b[p] = A[p], A[i], b[p], b[i]
            for r in range(i + 1, n):
                if A[r][i] != 0:
                    f = A[r][i] / A[i][i]
                    A[r] = [a - f * c for a, c in zip(A[r], A[i], strict=True)]
                    b[r] -= f * b[i]
        exact = [Fraction(0)] * n
        for i in range(n - 1, -1, -1):
            exact[i] = (b[i] - sum(A[i][j] * exact[j] for j in range(i + 1, n))) / A[i][i]
        c = np.asarray(mine.c)
        err = np.abs(c - np.array([float(v) for v in exact]))
        assert np.all(err <= refdata.collocation_bound(x, t, k, None, c)), k
