"""The accurate difference Jacobian (docs/design/integrate.md §6.1, decisions C3, C15): the 5-point
differences of sd.optimize's approx_fprime and check_grad (their default; optimize/_diff.py, which
keeps the public functions) and of sd.regression.nls's Jacobian at the solution (regression G70,
G88).

The accurate differences are 5-point stencils with a step per variable chosen once: from
2^-8 |x_j| down by factors of 4 while consecutive estimates get closer by more than 8 times
(the truncation error, O(h^4), still dominates), powers of two so that x_j +- k h is exact;
one-sided 5-point stencils at a bound or where the function is not finite on one side
(bench/regression/g6/accuracy.md at commit 4151d2a). A start at rounding level (|x_j| far below the
scale on which the function changes, as x_j = 1e-12 next to 1) is too small: the step grows until
the estimates differ by more than their rounding, then the same search starts from there
(optimize O104). The caller counts the evaluations and words the errors; g returns the values as
a float64 sd.ndarray.
"""

import math

from .. import _core as _sd

# The method's constants, as csrc/differentiate/accurate.hpp has them (kept here to name them):
# the steps: powers of two from 2^-8 |x_j| down to 2^-30 |x_j|
_FD_START, _FD_END = -8, -30
# a difference column whose 2-norm is at most this times its rounding level is taken as 0
# (regression G88): f's own evaluation errors are a few ulps (sd's vForce functions up to 8),
# on top of the rounding of each value that the level counts
_NOISE_FACTOR = 16.0
_U = 2.0**-53  # the unit roundoff: rounding a value moves it by at most u |f|
# a column whose rounding level is above sqrt(eps) = 2^-26 of its 2-norm (eps = 2u, the machine
# epsilon) has lost half its digits to rounding: its step is too small for the variable's scale
_ROUNDING_TOL = 2.0**-26
# a start whose rounding level is above 2^-36 of its column grows first (C112)
_GROW_TOL = 2.0**-36
# the largest start of a search: x_j = 0's, 2^-7 (frexp(1) = 0.5 * 2^1, so 2^(1 - 8))
_FD_CAP = math.ldexp(1.0, math.frexp(1.0)[1] + _FD_START)
# a search whose last two estimates differ by more than this (relative) never settled
_SETTLED = 2.0**-20


def Differences(g, p, lb, ub, weights=None, *, array_x=False, check=None):
    """5-point difference Jacobians of g(x) -> (m,) values (a float64 sd.ndarray, or a float
    when m = 1), x a list of p floats within [lb, ub] (a new float64 sd.ndarray of p values with
    array_x), with each variable's step chosen at its first column and kept (steps). weights
    (None or (m,)) scale the rows of the Jacobian. check (None or a callable) converts a value
    g returned that is not such an array (or raises). The loop is csrc/differentiate/accurate.hpp
    (the method above, in its constants: steps 2^-8 |x_j| down to 2^-30 |x_j|, a column within
    16 times its rounding level u |f| taken as 0, a start above 2^-36 of rounding grown first,
    the search from 2^-7 when one from below never settled to 2^-20).

    Methods: jacobian(x, g0, zero_noise=False) -> (J (m, p) or None, bad variable or None);
    forward(x, g0, eps) -> scipy's forward differences with steps eps; second(x, g0, c, idx) ->
    S = sum_i c_i d^2 g_i / dx^2 over the variables idx by central differences with the
    Jacobian's steps (a list of p lists; None when a point is outside the bounds or not
    finite). Attributes: steps (settable), too_narrow (no stencil fitted within the bounds at
    any step in the last failed jacobian())."""
    return _sd.differentiate.AccurateDiff(g, p, [float(v) for v in lb], [float(v) for v in ub], weights, array_x, check)
