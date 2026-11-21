"""stardust.integrate: numerical integration (docs/api/integrate.md, docs/design/integrate.md).

Usage: ``import stardust as sd`` then ``sd.integrate.trapezoid(y, x)``,
``sd.integrate.fixed_quad(f, 0, 1, n=10)``. The functions and their arguments are
scipy.integrate's names; the inputs, options and results follow this library's rules
(decisions C1-C54 in docs/design/integrate.md §1.1): one result class, IntegrateResult, with the
other areas' field names (integral, error, converged, nfev).

The public functions come milestone by milestone (docs/design/integrate.md §8): C3 brought the
sample rules (trapezoid, simpson, romb, cumulative_trapezoid, cumulative_simpson) and the fixed
rules (fixed_quad, newton_cotes).
"""

from .. import _calls
from ._result import IntegrateResult
from ._rules import fixed_quad, newton_cotes
from ._samples import cumulative_simpson, cumulative_trapezoid, romb, simpson, trapezoid

# a wrong call shape in the library's form, "trapezoid(): unexpected keyword argument 'foo'"
# (API.md §2.1)
trapezoid = _calls.function(trapezoid)
simpson = _calls.function(simpson)
romb = _calls.function(romb)
cumulative_trapezoid = _calls.function(cumulative_trapezoid)
cumulative_simpson = _calls.function(cumulative_simpson)
fixed_quad = _calls.function(fixed_quad)
newton_cotes = _calls.function(newton_cotes)
del _calls

__all__ = [
    "IntegrateResult",
    "cumulative_simpson",
    "cumulative_trapezoid",
    "fixed_quad",
    "newton_cotes",
    "romb",
    "simpson",
    "trapezoid",
]
