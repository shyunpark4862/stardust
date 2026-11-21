"""stardust.integrate: numerical integration (docs/api/integrate.md, docs/design/integrate.md).

Usage: ``import stardust as sd`` then ``integral, error = sd.integrate.quad(f, 0, 1)``. The
functions and their arguments are scipy.integrate's names; the inputs, options and results
follow this library's rules (decisions C1-C17 in docs/design/integrate.md §1.1): one result
class, IntegrateResult, with the other areas' field names (integral, error, converged, nfev).

The public functions come milestone by milestone (docs/design/integrate.md §8): C0 prepared the
area and only the result class is public so far.
"""

from ._result import IntegrateResult

__all__ = ["IntegrateResult"]
