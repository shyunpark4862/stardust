"""stardust.optimize: root finding, minimization, least squares and numerical differentiation
(docs/api/optimize.md).

Usage: ``import stardust as sd`` then ``r = sd.optimize.minimize(f, x0)`` and ``r.x``,
``r.converged``, ``r.nfev``. The functions and their arguments are scipy.optimize's names, the
inputs and options follow this library's rules and every result is an OptimizeResult whose
field names are the other areas' (``converged``, ``n_iter``). The decisions are
docs/design/optimize.md §1.1. The solvers come in milestones O1-O5; this module has numerical
differentiation and nonlinear least squares so far (O1).
"""

from ._diff import approx_fprime, check_grad
from ._lsq import least_squares
from ._result import OptimizeResult

__all__ = ["OptimizeResult", "approx_fprime", "check_grad", "least_squares"]
