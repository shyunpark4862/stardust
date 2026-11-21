"""stardust.differentiate: numerical differentiation (docs/api/differentiate.md,
docs/design/integrate.md, which covers this module and sd.integrate together).

Usage: ``import stardust as sd`` then ``r = sd.differentiate.derivative(f, x)`` and ``r.df``,
``r.error``, ``r.converged``; ``sd.differentiate.jacobian(f, x).df`` and
``sd.differentiate.hessian(f, x).ddf``. The functions and their arguments are
scipy.differentiate's names; the inputs, options and results follow this library's rules
(decisions C1-C38 in docs/design/integrate.md §1.1): one result class, DifferentiateResult, with
the other areas' field names (converged, n_iter, nfev).
"""

from .. import _calls
from ._derivative import derivative, hessian, jacobian
from ._result import DifferentiateResult

# a wrong call shape in the library's form, "derivative(): unexpected keyword argument 'foo'"
# (API.md §2.1)
derivative = _calls.function(derivative)
jacobian = _calls.function(jacobian)
hessian = _calls.function(hessian)
del _calls

__all__ = ["DifferentiateResult", "derivative", "hessian", "jacobian"]
