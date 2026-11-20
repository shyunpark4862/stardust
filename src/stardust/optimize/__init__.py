"""stardust.optimize: root finding, minimization, least squares and numerical differentiation
(docs/api/optimize.md).

Usage: ``import stardust as sd`` then ``r = sd.optimize.minimize(f, x0)`` and ``r.x``,
``r.converged``, ``r.nfev``. The functions and their arguments are scipy.optimize's names, the
inputs and options follow this library's rules and every result is an OptimizeResult whose
field names are the other areas' (``converged``, ``n_iter``). The decisions are
docs/design/optimize.md §1.1. The module has numerical differentiation, nonlinear and linear
least squares, scalar roots and minima and their batched versions, minimize, and systems of
equations.
"""

from .. import _calls
from ._diff import approx_fprime, check_grad
from ._elementwise import bracket_minimum, bracket_root, find_minimum, find_root
from ._linear import lsq_linear, nnls
from ._lsq import least_squares
from ._minimize import minimize
from ._result import OptimizeResult
from ._root import root
from ._scalar import minimize_scalar, root_scalar

# a wrong call shape in the library's form, "minimize(): unexpected keyword argument 'foo'"
# (API.md §2.1)
approx_fprime = _calls.function(approx_fprime)
check_grad = _calls.function(check_grad)
bracket_minimum = _calls.function(bracket_minimum)
bracket_root = _calls.function(bracket_root)
find_minimum = _calls.function(find_minimum)
find_root = _calls.function(find_root)
lsq_linear = _calls.function(lsq_linear)
nnls = _calls.function(nnls)
least_squares = _calls.function(least_squares)
minimize = _calls.function(minimize)
root = _calls.function(root)
minimize_scalar = _calls.function(minimize_scalar)
root_scalar = _calls.function(root_scalar)
del _calls

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
