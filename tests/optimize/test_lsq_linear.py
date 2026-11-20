"""sd.optimize.lsq_linear and sd.optimize.nnls (docs/api/optimize.md §3, docs/design/optimize.md
§3.4, §7): scipy 1.18.1 to the bit (the recorded runs, tests/optimize/data/scipy_lsq_linear.json,
and random problems live), the optimality (KKT) conditions, the iteration caps, the result and the
errors."""

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


def same_as_scipy(r, rs):
    """Every field of sd's lsq_linear result is scipy's, to the bit."""
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
        assert r.x.tolist() == rec["x"], pid
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
        assert same_as_scipy(r, rs), (A.shape, method)


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
                assert same_as_scipy(sd.optimize.lsq_linear(A, b, bounds=bounds, method=method), rs)
        assert sd.optimize.nnls(A, b).x.tolist() == scipy_nnls(A, b)[0].tolist()


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
        )
    assert sd.optimize.nnls(A, b).x.tolist() == scipy_nnls(A, b)[0].tolist()


def test_nnls_is_scipy_to_the_bit():
    # x's bits on random problems: tall, wide, rank-deficient, nonnegative A, b in A's cone;
    # with an iteration cap scipy raises where the status is 3 (O77)
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
            want = scipy_nnls(A, b, maxiter=cap)[0].tolist()
        except RuntimeError:
            want = None
        r = sd.optimize.nnls(A, b, max_iter=cap)
        if want is None:
            assert r.status == 3 and not r.converged
            assert r.message == "Maximum number of iterations reached."
        else:
            assert r.status == 1 and r.converged and r.x.tolist() == want, (m, n, cap)


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
        assert same_as_scipy(r, rs)
        assert r.status == 0 and not r.converged and r.message == "The maximum number of iterations is exceeded."
    # nnls's default cap is 3n (scipy's)
    A, b, _ = problem("hilbert.12x6.nonneg")
    r = sd.optimize.nnls(A, b, max_iter=1)
    assert (r.status, r.converged, r.n_iter) == (3, False, 1)
    with pytest.raises(RuntimeError):
        scipy_nnls(A, b, maxiter=1)


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
    for name in ("x", "cost", "fun", "converged", "status", "message", "n_iter"):
        assert hasattr(r, name), name
    for name in ("rnorm", "optimality", "active_mask", "method", "nfev"):
        assert not hasattr(r, name), name
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
    with pytest.raises(TypeError, match=r"lsq_linear\(\) got an unexpected keyword argument 'lsq_solver'"):
        f(A, b, lsq_solver="lsmr")
