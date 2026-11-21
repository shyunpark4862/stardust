"""stardust.differentiate: numerical differentiation (docs/api/differentiate.md,
docs/design/integrate.md, which covers this module and sd.integrate together).

Usage: ``import stardust as sd`` then ``r = sd.differentiate.derivative(f, x)`` and ``r.df``,
``r.error``, ``r.converged``. The functions and their arguments are scipy.differentiate's names;
the inputs, options and results follow this library's rules (decisions C1-C17 in
docs/design/integrate.md §1.1): one result class, DifferentiateResult, with the other areas'
field names (converged, n_iter, nfev).

The public functions come with milestone C2 (docs/design/integrate.md §8): C0 prepared the area
and only the result class is public so far.
"""

from ._result import DifferentiateResult

__all__ = ["DifferentiateResult"]
