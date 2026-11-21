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
from .. import linalg as _linalg

# the steps: powers of two from 2^-8 |x_j| down to 2^-30 |x_j|
_FD_START, _FD_END = -8, -30
# a difference column whose 2-norm is at most this times its rounding level is taken as 0
# (regression G88): f's own evaluation errors are a few ulps (sd's vForce functions up to 8),
# on top of the rounding of each value that the level counts
_NOISE_FACTOR = 16.0
_U = 2.0**-53  # the unit roundoff: rounding a value moves it by at most u |f|
# a column whose rounding level is above sqrt(eps) = 2^-26 of its 2-norm (eps = 2u, the machine
# epsilon) has lost half its digits to rounding: its step is too small for the variable's scale
# (the start 2^-8 |x_j| with x_j near 0)
_ROUNDING_TOL = 2.0**-26
# the largest start of that search: x_j = 0's (2^-8 of a variable of size 1)
_FD_CAP = math.ldexp(1.0, math.frexp(1.0)[1] + _FD_START)


class Differences:
    """5-point difference Jacobians of g(x) -> (m,) values (a float64 sd.ndarray), x a list
    of p floats within [lb, ub], with each variable's step chosen at its first column and
    kept (steps). weights (None or (m,)) scale the rows of the Jacobian. central keeps the
    last Jacobian's central values (h, g(x - h e_j), g(x + h e_j)) for second(); noise the
    rounding level of each column (column())."""

    # the 5-point stencils: central, and one-sided at a bound or where g is not finite on one
    # side (offsets in steps h, weights, the denominator's multiple of h)
    _STENCILS = (
        ((-2, -1, 1, 2), (1.0, -8.0, 8.0, -1.0)),
        ((0, 1, 2, 3, 4), (-25.0, 48.0, -36.0, 16.0, -3.0)),
        ((0, -1, -2, -3, -4), (25.0, -48.0, 36.0, -16.0, 3.0)),
    )

    def __init__(self, g, p, lb, ub, weights=None):
        self.g, self.p, self.lb, self.ub, self.weights = g, p, lb, ub, weights
        self.steps = [None] * p  # the step per variable, chosen once
        self.central = [None] * p  # (h, g(x - h e_j), g(x + h e_j)) of the last Jacobian
        self.noise = [0.0] * p  # the rounding level of the last difference column
        self.fits = False  # some stencil of the last column asked fitted within the bounds
        # a failed jacobian(): no stencil fitted within the bounds at any step (else g was not
        # finite at the points of every stencil that fitted)
        self.too_narrow = False

    def column(self, x, j, h, g0):
        """dg/dx_j (unweighted) by the first feasible 5-point stencil with step h, or None;
        the central stencil's values are kept for second(). Also sets noise[j], the 2-norm
        (weighted as the Jacobian) of the column's rounding level: each value
        g_i(x + o h e_j) carries its rounding, up to u |g_i|, and the stencil
        sum_o w_o g_o / (12 h) adds those with the weights, so element i is uncertain by up to
        (sum_o |w_o|) / (12 h) u max_o |g_i(x + o h e_j)|."""
        t = x[j]
        self.fits = False
        for kind, (offs, wts) in enumerate(self._STENCILS):
            if not (self.lb[j] <= t + min(offs) * h and t + max(offs) * h <= self.ub[j]):
                continue
            self.fits = True
            acc = None
            vals = {}
            for o, w in zip(offs, wts):
                if o == 0:
                    vr = g0
                else:
                    th = list(x)
                    th[j] = t + o * h
                    vr = self.g(th)
                    if not bool(_sd.all(_sd.isfinite(vr))):
                        acc = None
                        break
                vals[o] = vr
                acc = w * vr if acc is None else acc + w * vr
            if acc is not None:
                if kind == 0:
                    self.central[j] = (h, vals[-1], vals[1])
                big = None
                for o, w in zip(offs, wts):
                    if w != 0.0:
                        a = _sd.abs(vals[o])
                        big = a if big is None else _sd.maximum(big, a)
                if self.weights is not None:
                    big = big * self.weights
                wsum = sum(abs(w) for w in wts)
                self.noise[j] = wsum / (12.0 * h) * _U * float(_linalg.vector_norm(big)) if big.shape[0] else 0.0
                return acc / (12.0 * h)
        return None

    def _wnorm(self, col):
        """The 2-norm of a column weighted as the Jacobian (noise[j]'s scale)."""
        return float(_linalg.vector_norm(col if self.weights is None else col * self.weights))

    def _rounding(self, col, j):
        """noise[j] (of the column just computed) relative to the column's size: inf for 0."""
        size = self._wnorm(col)
        return self.noise[j] / size if size > 0 else math.inf

    def choose_step(self, x, j, g0):
        """The step of variable j, chosen once: from 2^-8 |x_j| down by factors of 4 while
        consecutive estimates get closer by more than 8 times; the last such estimate and its
        step. A start at rounding level is moved up first (_grow); when no larger step leaves
        rounding level (a column that is 0 or noise) the search is the plain one from the start."""
        base = math.frexp(abs(x[j]) if x[j] != 0 else 1.0)[1]
        e0 = base + _FD_START
        col0 = self.column(x, j, math.ldexp(1.0, e0), g0)
        if col0 is None or math.ldexp(1.0, e0) >= _FD_CAP or self._rounding(col0, j) <= _ROUNDING_TOL:
            return self._descend(x, j, g0, e0, (col0, self.fits))
        fits0 = self.fits
        e = self._grow(x, j, g0, e0, col0)
        if e is not None:
            best, hbest = self._descend(x, j, g0, e)
            if best is not None:
                return best, hbest
        best, hbest = self._descend(x, j, g0, e0, strict=False)
        self.fits = self.fits or fits0
        return best, hbest

    def _grow(self, x, j, g0, e0, col0):
        """The exponent of the step to start from when the start 2^e0 is at rounding level:
        moved by the factor that brings the rounding to sqrt(eps) of the column (at most to
        _FD_CAP), then up by factors of 4 until consecutive estimates differ by more than their
        rounding levels (the truncation error shows) or _FD_CAP; None when no such step leaves
        rounding level (a column that is 0 or noise at every step: the usual search decides)."""
        cap = math.frexp(_FD_CAP)[1] - 1
        r = self._rounding(col0, j)
        jump = cap - e0 if math.isinf(r) else math.ceil(math.log2(r / _ROUNDING_TOL))
        e = min(cap, e0 + max(jump, 2))
        col = self.column(x, j, math.ldexp(1.0, e), g0)
        if col is None:
            return None
        lvl = self.noise[j]
        while e + 2 <= cap:
            nxt = self.column(x, j, math.ldexp(1.0, e + 2), g0)
            if nxt is None:
                break
            nlvl = self.noise[j]
            e += 2
            shows = self._wnorm(nxt - col) > lvl + nlvl
            col, lvl = nxt, nlvl
            if shows:
                break
        size = self._wnorm(col)
        if not (size > 0 and lvl / size <= _ROUNDING_TOL):
            return None
        return e

    def _descend(self, x, j, g0, e0, first=None, strict=True):
        """The search from the step 2^e0 down by factors of 4 (12 steps): the estimate where
        consecutive ones stop getting closer by more than 8 times. first: (column, fits) at
        2^e0 when just computed. strict: the second estimate is not taken over the first when
        it is rounding noise (within _NOISE_FACTOR of its rounding level, regression G88's test)
        and the first is not."""
        best = hbest = None
        dprev = math.inf
        prev = None
        prev_ok = False  # the first estimate is not rounding noise
        fits = False
        for e in range(e0, e0 + _FD_END - _FD_START - 1, -2):
            h = math.ldexp(1.0, e)
            if first is not None and e == e0:
                col, self.fits = first
            else:
                col = self.column(x, j, h, g0)
            fits = fits or self.fits
            if col is None:
                if best is not None:
                    break
                continue
            if prev is None:
                best, hbest = col, h
                prev_ok = strict and self._rounding(col, j) * _NOISE_FACTOR < 1
            else:
                if prev_ok and dprev == math.inf and self._rounding(col, j) * _NOISE_FACTOR >= 1:
                    break
                sc = float(_sd.max(_sd.abs(col)))
                dd = float(_sd.max(_sd.abs(col - prev))) / sc if sc > 0 else 0.0
                if dd >= dprev / 8:
                    break
                best, hbest = col, h
                dprev = dd
                if dd == 0:
                    break
            prev = col
        self.fits = fits
        return best, hbest

    def jacobian(self, x, g0, zero_noise=False):
        """(J, bad): the (m, p) Jacobian at x (g0 = g(x)), rows scaled by the weights, with the
        steps chosen at the first call; bad is the first variable whose column no stencil
        gives (J is None then). zero_noise: a column within _NOISE_FACTOR of its rounding level
        is set to 0 (regression G88)."""
        self.central = [None] * self.p
        self.too_narrow = False
        cols = []
        for j in range(self.p):
            col = None
            fits = False
            if self.steps[j] is not None:
                col = self.column(x, j, self.steps[j], g0)
                fits = self.fits
            if col is None:
                col, self.steps[j] = self.choose_step(x, j, g0)
                fits = fits or self.fits
            if col is None:
                self.too_narrow = not fits
                return None, j
            cols.append(col)
        J = _sd.stack(cols, axis=1)
        if self.weights is not None:
            J = J * _sd.reshape(self.weights, (self.weights.shape[0], 1))
        if zero_noise:
            # kept, a noise column has a direction of its own and can take the place of an
            # identified parameter's column (a exp(-k t) + c at a large k: the k column was
            # rounding noise constant over the rows, the direction of c's)
            zero = [j for j in range(self.p) if float(_linalg.vector_norm(J[:, j])) <= _NOISE_FACTOR * self.noise[j]]
            if zero:
                keep = _sd.asarray([0.0 if j in zero else 1.0 for j in range(self.p)])
                J = J * keep
        return J, None

    def second(self, x, g0, c, idx):
        """S = sum_i c_i d^2 g_i / dx^2 over the variables idx (0 elsewhere), by central
        differences with the Jacobian's steps (its kept values for the diagonal); None when a
        point is outside the bounds or not finite."""
        p = self.p
        hs = []
        for j in range(p):
            if self.central[j] is not None:
                hs.append(self.central[j][0])
            elif self.steps[j] is not None:
                hs.append(self.steps[j])
            else:
                hs.append(math.ldexp(1.0, math.frexp(abs(x[j]) if x[j] != 0 else 1.0)[1] + _FD_START - 2))
        if not all(self.lb[j] <= x[j] - hs[j] and x[j] + hs[j] <= self.ub[j] for j in idx):
            return None
        c0 = float(_linalg.vecdot(c, g0))

        def phi(steps):
            th = list(x)
            for j, s in steps:
                th[j] = x[j] + s * hs[j]
            vr = self.g(th)
            if not bool(_sd.all(_sd.isfinite(vr))):
                raise ArithmeticError
            return float(_linalg.vecdot(c, vr))

        S = [[0.0] * p for _ in range(p)]
        try:
            for a, j in enumerate(idx):
                if self.central[j] is not None:
                    fm, fp = (float(_linalg.vecdot(c, v)) for v in self.central[j][1:])
                else:
                    fm, fp = phi([(j, -1)]), phi([(j, 1)])
                S[j][j] = (fp - 2 * c0 + fm) / (hs[j] * hs[j])
                for k in idx[:a]:
                    v = (
                        phi([(j, 1), (k, 1)])
                        - phi([(j, 1), (k, -1)])
                        - phi([(j, -1), (k, 1)])
                        + phi([(j, -1), (k, -1)])
                    ) / (4 * hs[j] * hs[k])
                    S[j][k] = S[k][j] = v
        except ArithmeticError:
            return None
        return S
