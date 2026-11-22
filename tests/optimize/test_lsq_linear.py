"""sd.optimize.lsq_linear and sd.optimize.nnls (docs/api/optimize.md §3, docs/design/optimize.md
§3.4, §7): scipy 1.18.1 to the bit for lsq_linear, nnls to rounding with scipy's zeros (the recorded
runs, tests/optimize/data/scipy_lsq_linear.json, and random problems live), the optimality (KKT)
conditions, the iteration caps, the result and the errors."""

import json
import math
import pathlib
import warnings

import numpy as np
import pytest
from scipy.optimize import lsq_linear as scipy_lsq_linear
from scipy.optimize import nnls as scipy_nnls

import stardust as sd

from .functions import LSQ_LINEAR

DATA = pathlib.Path(__file__).parent / "data"
REF = json.loads((DATA / "scipy_lsq_linear.json").read_text())


def problem(pid):
    A, b, bounds = LSQ_LINEAR[pid]
    return np.array(A), np.array(b), bounds


def clipped(rs, bounds, method):
    """scipy's x as sd returns it: within the bounds, and for 'bvls' a variable on a bound
    (active_mask +-1) at that bound exactly ('bvls' leaves a bound reached by a step at its
    rounded value; 'trf''s active_mask only says x is near a bound)."""
    n = rs.x.shape[0]
    lb, ub = (np.broadcast_to(np.asarray(v, dtype=float), (n,)) for v in bounds)
    x = np.clip(rs.x, lb, ub)
    if method == "bvls":
        am = np.asarray(rs.active_mask)
        x = np.where(am < 0, lb, np.where(am > 0, ub, x))
    return x


def nnls_close(x, xs, A):
    """sd's nnls x against scipy's: the same entries at 0 (the same final set of free columns)
    and the free ones within rounding of their least squares solve. sd's inner products and
    one-column reflections are short plain loops where scipy calls BLAS and LAPACK (csrc/optimize/
    linear.cpp, nnls_dot), so the bits differ in the last places: allowed 64 eps cond(A_free)
    of |x|_inf."""
    x, xs = np.asarray(x, dtype=float), np.asarray(xs, dtype=float)
    if not np.array_equal(x == 0, xs == 0):
        return False
    free = xs != 0
    if not free.any():
        return True
    cond = float(np.linalg.cond(np.asarray(A, dtype=float)[:, free]))
    tol = 64 * np.finfo(float).eps * max(cond, 1.0) * max(1.0, float(np.max(np.abs(xs))))
    return bool(np.all(np.abs(x - xs) <= tol))


def same_as_scipy(r, rs, bounds=None):
    """Every field of sd's lsq_linear result is scipy's, to the bit; where scipy's x is outside
    its bounds by rounding (bounds given), sd's x is the clipped one and fun, cost and optimality
    are those of that x (to rounding)."""
    if bounds is not None and rs.status != 3:
        x = clipped(rs, bounds, r.method)
        if x.tolist() != rs.x.tolist():
            A_x_b = r.fun  # A x - b at sd's x
            return (
                r.x.tolist() == x.tolist()
                and math.isclose(r.cost, 0.5 * float(np.dot(A_x_b.tolist(), A_x_b.tolist())), rel_tol=1e-14)
                and math.isclose(r.cost, rs.cost, rel_tol=1e-12, abs_tol=1e-300)
                and r.active_mask.tolist() == [int(v) for v in rs.active_mask]
                and (r.status, r.message, r.n_iter, r.converged) == (rs.status, rs.message, rs.nit, rs.success)
            )
    return (
        r.x.tolist() == rs.x.tolist()
        and r.fun.tolist() == rs.fun.tolist()
        and r.cost == rs.cost
        and r.optimality == rs.optimality
        and r.active_mask.tolist() == [int(v) for v in rs.active_mask]
        and r.status == rs.status
        and r.message == rs.message
        and r.n_iter == rs.nit
        and r.converged == rs.success
    )


def random_problems(seed, count, wide=True):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        m = int(rng.integers(1, 50))
        n = int(rng.integers(1, 40 if wide else m + 1))
        A = rng.standard_normal((m, n))
        b = 3 * rng.standard_normal(m)
        kind = int(rng.integers(0, 4))
        if kind == 0:
            lb, ub = np.zeros(n), np.full(n, np.inf)
        elif kind == 1:
            lb, ub = np.full(n, -0.3), np.full(n, 0.3)
        elif kind == 2:
            lb = rng.uniform(-1, 0, n)
            ub = lb + rng.uniform(0.01, 1, n)
            lb[rng.random(n) < 0.3] = -np.inf
            ub[rng.random(n) < 0.3] = np.inf
        else:  # rank-deficient
            r = max(1, min(m, n) // 2)
            A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
            lb, ub = np.full(n, -1.0), np.full(n, 0.5)
        yield A, b, (lb, ub)


# ---- scipy 1.18.1 (O9)


@pytest.mark.parametrize("method", ["trf", "bvls"])
def test_recorded_scipy(method):
    for pid in LSQ_LINEAR:
        A, b, bounds = problem(pid)
        rec = REF[pid][method]
        r = sd.optimize.lsq_linear(A, b, bounds=bounds, method=method)
        assert r.x.tolist() == rec["x"], pid
        assert r.fun.tolist() == rec["fun"], pid
        assert (r.cost, r.optimality, r.status, r.message, r.n_iter) == (
            rec["cost"],
            rec["optimality"],
            rec["status"],
            rec["message"],
            rec["nit"],
        ), pid
        assert r.active_mask.tolist() == rec["active_mask"] and r.converged == rec["success"], pid


def test_recorded_scipy_nnls():
    for pid in LSQ_LINEAR:
        A, b, _ = problem(pid)
        rec = REF[pid]["nnls"]
        r = sd.optimize.nnls(A, b)
        assert nnls_close(r.x.tolist(), rec["x"], A), pid
        # scipy's rnorm is ||b'|| of its rotated right-hand side; cost is 1/2 ||A x - b||^2
        # (the same up to rounding at the scale of b; scipy's is 0 when P has m columns)
        assert abs(math.sqrt(2 * r.cost) - rec["rnorm"]) <= 1e-10 * max(1.0, float(np.linalg.norm(b))), pid


@pytest.mark.parametrize("method", ["trf", "bvls"])
@pytest.mark.parametrize("seed", [3, 11])
def test_random_problems_are_scipy_to_the_bit(method, seed):
    # wide, tall, rank-deficient; half-bounded, boxed, mixed infinite bounds
    for A, b, bounds in random_problems(seed, 150):
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            rs = scipy_lsq_linear(A, b, bounds=bounds, method=method)
        r = sd.optimize.lsq_linear(A, b, bounds=bounds, method=method)
        assert same_as_scipy(r, rs, bounds), (A.shape, method)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (3, 1), (2, 2), (1, 2), (2, 1)])
def test_small_shapes_are_scipy_to_the_bit(shape):
    # numpy's dot takes other paths for a row, a column or one element
    rng = np.random.default_rng(5)
    m, n = shape
    for _ in range(20):
        A, b = rng.standard_normal((m, n)), 3 * rng.standard_normal(m)
        for bounds in ((np.zeros(n), np.full(n, np.inf)), (np.full(n, -0.2), np.full(n, 0.2))):
            for method in ("trf", "bvls"):
                rs = scipy_lsq_linear(A, b, bounds=bounds, method=method)
                assert same_as_scipy(sd.optimize.lsq_linear(A, b, bounds=bounds, method=method), rs, bounds)
        assert nnls_close(sd.optimize.nnls(A, b).x.tolist(), scipy_nnls(A, b)[0], A)


@pytest.mark.parametrize("shape", [(1000, 100), (300, 120)])
def test_larger_problems_are_scipy_to_the_bit(shape):
    rng = np.random.default_rng(9)
    m, n = shape
    A = rng.standard_normal((m, n))
    b = A @ rng.standard_normal(n) + rng.standard_normal(m)
    bounds = (np.zeros(n), np.full(n, np.inf))
    for method in ("trf", "bvls"):
        assert same_as_scipy(
            sd.optimize.lsq_linear(A, b, bounds=bounds, method=method),
            scipy_lsq_linear(A, b, bounds=bounds, method=method),
            bounds,
        )
    assert nnls_close(sd.optimize.nnls(A, b).x.tolist(), scipy_nnls(A, b)[0], A)


def test_nnls_is_scipy_to_rounding():
    # x on random problems (nnls_close: the same zeros, the rest to rounding): tall, wide, rank-deficient,
    # nonnegative A, b in A's cone;
    # with an iteration cap scipy raises where the status is 0 (D-26); scipy's maxiter stops one
    # iteration short of the cap (iter >= maxiter), so its maxiter = cap + 1 is max_iter = cap
    rng = np.random.default_rng(7)
    for k in range(300):
        m, n = int(rng.integers(1, 80)), int(rng.integers(1, 80))
        A = rng.standard_normal((m, n))
        kind = k % 4
        if kind == 1:
            r = max(1, min(m, n) // 3)
            A = rng.standard_normal((m, r)) @ rng.standard_normal((r, n))
        if kind == 2:
            A = np.abs(A)
        b = A @ np.abs(rng.standard_normal(n)) if kind == 3 else rng.standard_normal(m)
        cap = int(rng.integers(1, 2 * n + 1)) if k % 5 == 0 else None
        try:
            want = scipy_nnls(A, b, maxiter=None if cap is None else cap + 1)[0].tolist()
        except RuntimeError:
            want = None
        r = sd.optimize.nnls(A, b, max_iter=cap)
        if want is None:
            assert r.status == 0 and not r.converged and r.n_iter == cap
            assert r.message == "The maximum number of iterations is exceeded."
        else:
            assert r.status == 1 and r.converged and nnls_close(r.x.tolist(), want, A), (m, n, cap)


# ---- optimality


def kkt(A, b, x, lb, ub, tol):
    """The first-order conditions at x: within the bounds; the gradient A^T (A x - b) vanishes on
    the free variables and points outward at the active bounds (relative tol)."""
    g = A.T @ (A @ x - b)
    scale = max(1.0, np.max(np.abs(A.T @ b)))
    xs = max(1.0, np.max(np.abs(x)))
    assert np.all(x >= lb - 1e-15 * xs) and np.all(x <= ub + 1e-15 * xs)
    for j in range(len(x)):
        at_lo = math.isfinite(lb[j]) and x[j] - lb[j] <= 1e-12 * max(xs, abs(lb[j]))
        at_up = math.isfinite(ub[j]) and ub[j] - x[j] <= 1e-12 * max(xs, abs(ub[j]))
        if at_lo:
            assert g[j] >= -tol * scale
        elif at_up:
            assert g[j] <= tol * scale
        else:
            assert abs(g[j]) <= tol * scale


def test_kkt():
    # bvls and nnls end at a KKT point; trf (an interior method) stops when its Coleman-Li scaled
    # gradient is below tol or the cost stops changing, at bvls's cost to 1e-9
    for A, b, (lb, ub) in random_problems(21, 100, wide=False):
        rb = sd.optimize.lsq_linear(A, b, bounds=(lb, ub), method="bvls")
        assert rb.converged
        kkt(A, b, np.array(rb.x.tolist()), lb, ub, 1e-9)
        r = sd.optimize.lsq_linear(A, b, bounds=(lb, ub), method="trf")
        assert r.converged
        x = np.array(r.x.tolist())
        assert np.all(x >= lb) and np.all(x <= ub)
        assert r.cost - rb.cost <= 1e-9 * max(rb.cost, 1e-300) + 1e-30
        x = np.array(sd.optimize.nnls(A, b).x.tolist())
        kkt(A, b, x, np.zeros_like(x), np.full_like(x, np.inf), 1e-9)


# ---- the result, the caps, the errors


def test_unbounded_solution():
    A, b, _ = problem("cos.30x8")
    r = sd.optimize.lsq_linear(A, b)
    assert r.status == 3 and r.n_iter == 0 and r.message == "The unconstrained solution is optimal."
    assert r.x.tolist() == np.linalg.lstsq(A, b, rcond=-1)[0].tolist()
    assert r.active_mask.tolist() == [0] * 8


def test_iteration_caps():
    for pid, method, cap in (("hilbert.12x6.nonneg", "trf", 5), ("sin.8x12.box", "bvls", 1)):
        A, b, bounds = problem(pid)
        r = sd.optimize.lsq_linear(A, b, bounds=bounds, method=method, max_iter=cap)
        rs = scipy_lsq_linear(A, b, bounds=bounds, method=method, max_iter=cap)
        assert same_as_scipy(r, rs, bounds)
        assert r.status == 0 and not r.converged and r.message == "The maximum number of iterations is exceeded."
    # nnls (D-26): at most max_iter iterations (status 0 past it, lsq_linear's message); the
    # default 3n - 1 stops where scipy's maxiter = 3n does
    A, b, _ = problem("hilbert.12x6.nonneg")
    r = sd.optimize.nnls(A, b, max_iter=1)
    assert (r.status, r.converged, r.n_iter) == (0, False, 1)
    assert r.message == "The maximum number of iterations is exceeded."
    with pytest.raises(RuntimeError):
        scipy_nnls(A, b, maxiter=2)
    # a problem solved in one iteration finishes with max_iter = 1 (scipy needs maxiter = 2)
    r = sd.optimize.nnls(np.eye(1), [2.0], max_iter=1)
    assert (r.x.tolist(), r.status, r.converged, r.n_iter) == ([2.0], 1, True, 1)
    with pytest.raises(RuntimeError):
        scipy_nnls(np.eye(1), [2.0], maxiter=1)
    full = sd.optimize.nnls(A, b)
    assert full.n_iter <= 3 * 6 - 1 and nnls_close(full.x.tolist(), scipy_nnls(A, b)[0], A)


def test_result_fields():
    A, b, bounds = problem("scipy.dense.lb0")
    r = sd.optimize.lsq_linear(A, b, bounds=bounds)
    for name in ("x", "cost", "fun", "optimality", "active_mask", "converged", "status", "message", "n_iter", "method"):
        assert hasattr(r, name), name
    # O81: no unbounded_sol
    for name in ("unbounded_sol", "jac", "grad", "nfev", "njev"):
        assert not hasattr(r, name), name
    assert r.active_mask.dtype == sd.int64 and r.method == "trf"
    assert repr(r).startswith("OptimizeResult(method='trf', converged=True, cost=")
    r = sd.optimize.nnls(A, b)
    for name in ("x", "cost", "fun", "converged", "status", "message", "n_iter", "method"):
        assert hasattr(r, name), name
    for name in ("rnorm", "optimality", "active_mask", "nfev"):
        assert not hasattr(r, name), name
    assert r.method == "nnls" and repr(r).startswith("OptimizeResult(method='nnls', converged=True")
    assert r.message == "The solution has been computed successfully."
    assert r.fun.tolist() == (A @ np.array(r.x.tolist()) - b).tolist()


def test_inputs():
    A, b, bounds = problem("scipy.dense.lb0")
    want = sd.optimize.lsq_linear(A, b, bounds=bounds)
    for AA, bb in ((A.tolist(), b.tolist()), (sd.asarray(A), sd.asarray(b)), (np.asfortranarray(A), b)):
        r = sd.optimize.lsq_linear(AA, bb, bounds=bounds)
        assert r.x.tolist() == want.x.tolist()
    r = sd.optimize.lsq_linear(np.array([[1, 0], [0, 1]]), np.array([2, -1]), bounds=(0, 1))
    assert r.x.tolist() == [1.0, 0.0] or max(abs(r.x[0] - 1.0), abs(r.x[1])) < 1e-9


def test_same_bits_with_1_and_10_threads():
    rng = np.random.default_rng(4)
    A = rng.standard_normal((800, 60))
    b = rng.standard_normal(800)
    old = sd.get_num_threads()
    out = []
    try:
        for t in (1, 10):
            sd.set_num_threads(t)
            out.append(
                (
                    sd.optimize.lsq_linear(A, b, bounds=(0, np.inf)).x.tolist(),
                    sd.optimize.lsq_linear(A, b, bounds=(0, np.inf), method="bvls").x.tolist(),
                    sd.optimize.nnls(A, b).x.tolist(),
                )
            )
    finally:
        sd.set_num_threads(old)
    assert out[0] == out[1]


def test_errors():
    A, b = [[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0]
    for func in (sd.optimize.lsq_linear, sd.optimize.nnls):
        name = func.__name__
        with pytest.raises(ValueError, match=rf"{name}\(\): 'A' must be 2-dimensional, not shape \(2,\)"):
            func([1.0, 2.0], b)
        with pytest.raises(ValueError, match=rf"{name}\(\): 'A' must be 2-dimensional, not a scalar"):
            func(1.0, b)
        with pytest.raises(ValueError, match=rf"{name}\(\): 'b' must be 1-dimensional, not shape \(2, 1\)"):
            func(A, [[1.0], [2.0]])
        with pytest.raises(ValueError, match=rf"{name}\(\): 'b' must have 2 values \(one per row of 'A'\), not 3"):
            func(A, [1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match=rf"{name}\(\): 'A' must be finite"):
            func([[1.0, math.nan], [0.0, 1.0]], b)
        with pytest.raises(ValueError, match=rf"{name}\(\): 'b' must be finite"):
            func(A, [1.0, math.inf])
        with pytest.raises(ValueError, match=rf"{name}\(\): 'A' is empty, shape \(2, 0\)"):
            func(np.zeros((2, 0)), b)
        with pytest.raises(TypeError, match=rf"{name}\(\): 'A' must be real"):
            func([[1j, 0.0], [0.0, 1.0]], b)
        with pytest.raises(TypeError, match=rf"{name}\(\): 'max_iter' must be an int, not float"):
            func(A, b, max_iter=2.0)
        with pytest.raises(ValueError, match=rf"{name}\(\): 'max_iter' must be >= 1"):
            func(A, b, max_iter=0)
    f = sd.optimize.lsq_linear
    with pytest.raises(ValueError, match=r"lsq_linear\(\): 'method' must be 'trf' or 'bvls', not 'dogbox'"):
        f(A, b, method="dogbox")
    with pytest.raises(ValueError, match=r"lsq_linear\(\): 'method' must be lowercase: 'bvls', not 'BVLS'"):
        f(A, b, method="BVLS")
    with pytest.raises(ValueError, match=r"lsq_linear\(\): each lower bound must be less than its upper bound"):
        f(A, b, bounds=(1.0, 1.0))
    with pytest.raises(ValueError, match=r"lsq_linear\(\): 'bounds' must be a pair \(lower, upper\), not 3 values"):
        f(A, b, bounds=(0, 1, 2))
    with pytest.raises(ValueError, match=r"lsq_linear\(\): 'tol' must be a finite number >= 0"):
        f(A, b, tol=-1.0)
    with pytest.raises(TypeError, match=r"lsq_linear\(\): unexpected keyword argument 'lsq_solver'"):
        f(A, b, lsq_solver="lsmr")


def test_bvls_optimality_is_plus_zero():
    # API-10: np.max picks +0.0 over -0.0 (numpy's maximum), as scipy's optimality
    rng = np.random.default_rng(0)
    for m, n in ((10, 3), (30, 10), (5, 8), (1, 1)):  # (the draws before the case)
        for _ in range(4):
            rng.standard_normal((m, n)), rng.standard_normal(m), rng.uniform(-0.3, 0, n), rng.uniform(0, 0.3, n)
    A, b = rng.standard_normal((1, 4)), rng.standard_normal(1)
    r = sd.optimize.lsq_linear(A, b, bounds=(0, np.inf), method="bvls")
    s = scipy_lsq_linear(A, b, bounds=(0, np.inf), method="bvls")
    assert r.optimality == 0.0 and math.copysign(1.0, r.optimality) == 1.0
    assert r.x.tolist() == s.x.tolist() and math.copysign(1.0, s.optimality) == 1.0


def test_nnls_infinite_cost_is_not_converged():
    # API-5: as lsq_linear, a cost that is not finite is not convergence
    A = [[1e200, 0.0], [0.0, 1e200], [1.0, 1.0]]
    b = [-1e200, 1e200, 1e200]
    r = sd.optimize.nnls(A, b)
    assert math.isinf(r.cost) and not r.converged


# ---- x within the bounds; the bounds' form


def test_bvls_x_is_within_the_bounds():
    # a variable that the step x + alpha (z - x) takes to its lower bound 0 is a tiny negative
    # number in scipy (-1.39e-17 and the like) and exactly 0 in sd. The data are numpy's (its
    # exp and generator), so which fits show it does not follow the last bits of sd.exp; about
    # half of these 20 do
    t = np.linspace(0, 5, 40)
    A = np.exp(-t[:, None] * (0.25 * 1.25 ** np.arange(14)))
    shown = 0
    for seed in range(20):
        b = 2 * np.exp(-0.5 * t) + np.exp(-3 * t) + 0.005 * np.random.default_rng(seed).standard_normal(40)
        r = sd.optimize.lsq_linear(A, b, bounds=(0, sd.inf), method="bvls")
        x, am = r.x.tolist(), r.active_mask.tolist()
        assert all(v >= 0 for v in x) and all((v == 0.0) == (a == -1) for v, a in zip(x, am)), seed
        assert r.fun.tolist() == (sd.asarray(A) @ r.x - sd.asarray(b)).tolist(), seed
        rs = scipy_lsq_linear(A, b, bounds=(0, np.inf), method="bvls")
        below = np.nonzero(rs.x < 0)[0].tolist()
        assert all(x[j] == 0.0 and am[j] == -1 for j in below), seed
        assert same_as_scipy(r, rs, (0, np.inf)), seed
        shown += bool(below)
    assert shown >= 5, shown


@pytest.mark.parametrize("method", ["trf", "bvls"])
def test_random_problems_stay_within_the_bounds(method):
    clipped_runs = 0
    for seed in (3, 11):
        for A, b, (lb, ub) in random_problems(seed, 150):
            r = sd.optimize.lsq_linear(A, b, bounds=(lb, ub), method=method)
            x = np.array(r.x.tolist())
            assert np.all(x >= lb) and np.all(x <= ub)
            if method == "bvls":
                am = np.array(r.active_mask.tolist())
                assert np.all(x[am == -1] == lb[am == -1]) and np.all(x[am == 1] == ub[am == 1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rs = scipy_lsq_linear(A, b, bounds=(lb, ub), method=method)
                clipped_runs += x.tolist() != rs.x.tolist()
        r = sd.optimize.nnls(A, b)
        assert all(v >= 0 for v in r.x.tolist())
    if method == "bvls":
        assert clipped_runs > 0  # the case this guards occurs in these problems


def test_bounds_messages():
    A, b = [[1.0, 0.0], [0.0, 1.0]], [1.0, 2.0]
    msg = r"^lsq_linear\(\): 'bounds' must have 2 values \(one per column of 'A'\), not 3$"
    with pytest.raises(ValueError, match=msg):
        sd.optimize.lsq_linear(A, b, bounds=([0, 0, 0], 1))
    with pytest.raises(ValueError, match=r"^lsq_linear\(\): 'b' must have 1 value \(one per row of 'A'\), not 2$"):
        sd.optimize.lsq_linear([[1.0, 2.0]], b)
    msg = r"^lsq_linear\(\): 'bounds' must be a pair \(lower, upper\), not a sequence of \(low, high\) pairs$"
    with pytest.raises(ValueError, match=msg):
        sd.optimize.lsq_linear([[1.0, 0.0, 0.0]] * 3, [1.0, 2.0, 3.0], bounds=[(0, 1), (0, 1), (0, 1)])
