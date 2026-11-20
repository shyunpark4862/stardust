"""Linear least squares with bounds: lsq_linear and nnls (docs/api/optimize.md §3,
docs/design/optimize.md §3.4).

Both run in the native kernel stardust._core.optimize (lsq_linear_solve, nnls_solve), the same
steps as scipy 1.18.1: lsq_linear's 'trf' (trust region reflective) and 'bvls' (bounded-variable
least squares) from numpy's unbounded solution, nnls scipy's Lawson-Hanson. A is a dense
float64 matrix (no sparse matrices or LinearOperator; no lsmr, O81).
"""

import math

from .. import _core as _sd
from ._common import _as_array, _bounds, _check_int, _check_method, _check_tol
from ._result import OptimizeResult

_k = _sd.optimize

_METHODS = ("trf", "bvls")
# scipy 1.18.1's lsq_linear TERMINATION_MESSAGES
_MESSAGES = {
    -1: "The algorithm was not able to make progress on the last iteration.",
    0: "The maximum number of iterations is exceeded.",
    1: "The first-order optimality measure is less than `tol`.",
    2: "The relative change of the cost function is less than `tol`.",
    3: "The unconstrained solution is optimal.",
}
# nnls (O77): scipy's __nnls info; 1 has no text in scipy (Lawson and Hanson's mode 1), 3 is the
# text of scipy's RuntimeError
_NNLS_MESSAGES = {
    1: "The solution has been computed successfully.",
    3: "Maximum number of iterations reached.",
}


def _system(A, b, func):
    """A as a float64 sd.ndarray (m, n) and b (m,), not empty."""
    a = _as_array(A, func, "A") if not _is_scalar(A) else None
    if a is None or a.ndim != 2:
        shape = "a scalar" if a is None else f"shape {tuple(a.shape)}"
        raise ValueError(f"{func}(): 'A' must be 2-dimensional, not {shape}")
    m, n = a.shape
    if m == 0 or n == 0:
        raise ValueError(f"{func}(): 'A' is empty, shape {(m, n)}")
    if _is_scalar(b):
        raise ValueError(f"{func}(): 'b' must be 1-dimensional, not a scalar")
    bb = _as_array(b, func, "b")
    if bb.ndim != 1:
        raise ValueError(f"{func}(): 'b' must be 1-dimensional, not shape {tuple(bb.shape)}")
    if bb.shape[0] != m:
        raise ValueError(f"{func}(): 'b' must have {m} values (one per row of 'A'), not {bb.shape[0]}")
    # (the kernel checks that A and b are finite as it reads them)
    return a, bb, m, n


def _is_scalar(v):
    import numbers

    return isinstance(v, numbers.Number) or (hasattr(v, "ndim") and v.ndim == 0)


def lsq_linear(A, b, /, *, bounds=None, method="trf", tol=1e-10, max_iter=None):
    """Minimizes 1/2 ||A x - b||^2 within bounds: method='trf' (trust region reflective) or
    'bvls' (bounded-variable least squares) (docs/api/optimize.md §3)."""
    func = "lsq_linear"
    a, bb, m, n = _system(A, b, func)
    _check_method(method, func, _METHODS)
    lb, ub = _bounds(bounds, func, n)
    tol = _check_tol(tol, func, "tol")
    cap = -1 if max_iter is None else _check_int(max_iter, func, "max_iter", 1)
    x, f, cost, optimality, active, status, n_iter = _k.lsq_linear_solve(
        func, a, bb, lb, ub, 0 if method == "trf" else 1, tol, cap
    )
    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f,
        optimality=optimality,
        active_mask=_sd.asarray(active, dtype=_sd.int64),
        converged=status > 0 and math.isfinite(cost),
        status=status,
        message=_MESSAGES[status],
        n_iter=n_iter,
        method=method,
    )


def nnls(A, b, /, *, max_iter=None):
    """Minimizes ||A x - b|| subject to x >= 0 (Lawson and Hanson's NNLS)
    (docs/api/optimize.md §3)."""
    func = "nnls"
    a, bb, m, n = _system(A, b, func)
    cap = 3 * n if max_iter is None else _check_int(max_iter, func, "max_iter", 1)
    x, f, cost, status, n_iter = _k.nnls_solve(func, a, bb, cap)
    return OptimizeResult(
        x=x,
        cost=cost,
        fun=f,
        converged=status == 1,
        status=status,
        message=_NNLS_MESSAGES[status],
        n_iter=n_iter,
    )
