"""stardust.optimize: root finding, minimization, least squares and numerical differentiation
(docs/api/optimize.md).

Usage: ``import stardust as sd`` then ``r = sd.optimize.minimize(f, x0)`` and ``r.x``,
``r.converged``, ``r.nfev``. The functions and their arguments are scipy.optimize's names, the
inputs and options follow this library's rules and every result is an OptimizeResult whose
field names are the other areas' (``converged``, ``n_iter``). The decisions are
docs/design/optimize.md §1.1. The module has numerical differentiation and nonlinear least
squares (O1), scalar roots and minima and their batched versions (O2), minimize (O3, O4), and
systems of equations and linear least squares with bounds (O5).
"""

from ._diff import approx_fprime, check_grad
from ._elementwise import bracket_minimum, bracket_root, find_minimum, find_root
from ._linear import lsq_linear, nnls
from ._lsq import least_squares
from ._minimize import minimize
from ._result import OptimizeResult
from ._root import root
from ._scalar import minimize_scalar, root_scalar

__all__ = [
    "OptimizeResult",
    "approx_fprime",
    "bracket_minimum",
    "bracket_root",
    "check_grad",
    "find_minimum",
    "find_root",
    "least_squares",
    "lsq_linear",
    "minimize",
    "minimize_scalar",
    "nnls",
    "root",
    "root_scalar",
]
