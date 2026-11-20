"""Reference derivatives for the optimize accuracy measurement (bench/accuracy, area optimize).

    .venv/bin/python tools/optimize_accuracy_ref.py            # both files
    .venv/bin/python tools/optimize_accuracy_ref.py minima     # scalar_minima.json only

writes

  tests/optimize/ref/derivatives.npz: for each MGH case with factor 1 (the 47 cases of
tests/optimize/test_diff.py), at its starting point x0, the Jacobian J of the residuals and the
gradient g = 2 J^T F of f = sum F_i^2, from the problems' analytic formulas (tests/optimize/mgh.py)
evaluated in Arb on the double x0 (tests/optimize/ad.py's ARB), settled to 2^-110 of the largest
entry (approx_fprime's error is measured normwise), as hi + lo. Keys per case i: g<i>_id, x0,
g_hi, g_lo, J_hi, J_lo (J row-major, m x n).

  tests/optimize/ref/scalar_minima.json: for each minimize_scalar case (tests/optimize/
functions.py MIN_CASES), the local minimizers of f as [hi, lo] pairs: "bracket", those inside
the bracket's outer points (brent and golden search there; scipy's two-point bracket is
widened downhill, its cases' minimizer is the known one), and "bounds", those of f on [lower,
upper] (bounded), the end points included where f' points out of the interval. A minimizer is
a sign change of f' from - to + (f' in double, by forward-mode differentiation, on a grid of
GRID points), refined by bisection on the sign of the exact f' in Arb balls (forward-mode
over arb) until the interval is narrower than 2^-112 |x|; the pair is its midpoint.

The other optimize references are files already: the Arb roots (tests/optimize/data/
scalar_roots.json, tools/optimize_ref.py), NIST StRD's certified values
(tests/regression/data/nls), MGH's optima (tests/optimize/mgh.py KNOWN).
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
from flint import arb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
from arbref import MAX_PREC, START_PREC, NotSettled, precision, settled, to_dd  # noqa: E402

OUT = ROOT / "tests" / "optimize" / "ref"


def _settled_vec(fn):
    p = START_PREC
    while p <= MAX_PREC:
        with precision(p):
            vs = [v if isinstance(v, arb) else arb(v) for v in fn()]
            top = max((abs(float(v.mid())) for v in vs), default=0.0)
            if all(settled(v, floor=top) for v in vs):
                dd = [to_dd(v) for v in vs]
                return np.array([h for h, _ in dd]), np.array([lo for _, lo in dd])
        p *= 2
    raise NotSettled(fn)


GRID = 20001


class _ArbDual:
    """v + d e, e^2 = 0, with arb parts: the exact derivative of the scalar test functions."""

    __slots__ = ("v", "d")

    def __init__(self, v, d):
        self.v, self.d = v, d

    @staticmethod
    def _c(o):
        return o if isinstance(o, _ArbDual) else _ArbDual(arb(o), arb(0))

    def __add__(self, o):
        o = self._c(o)
        return _ArbDual(self.v + o.v, self.d + o.d)

    __radd__ = __add__

    def __sub__(self, o):
        o = self._c(o)
        return _ArbDual(self.v - o.v, self.d - o.d)

    def __rsub__(self, o):
        return self._c(o) - self

    def __neg__(self):
        return _ArbDual(-self.v, -self.d)

    def __mul__(self, o):
        o = self._c(o)
        return _ArbDual(self.v * o.v, self.d * o.v + self.v * o.d)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._c(o)
        return _ArbDual(self.v / o.v, (self.d * o.v - self.v * o.d) / (o.v * o.v))

    def __rtruediv__(self, o):
        return self._c(o) / self

    def __pow__(self, k):
        k = float(k)
        if not k.is_integer():
            raise ValueError("integer powers only")
        k = int(k)
        return _ArbDual(self.v**k, k * self.v ** (k - 1) * self.d if k else arb(0))


class _ArbDualM:
    def exp(self, x):
        e = x.v.exp()
        return _ArbDual(e, e * x.d)


def _fprime_sign(f, x):
    """The sign of f'(x) at an arb point x, from the exact derivative (0 if the ball holds 0)."""
    d = f(_ArbDual(arb(x), arb(1)), _ArbDualM()).d
    return 1 if d > 0 else (-1 if d < 0 else 0)


def _bisect_min(f, a, b):
    """The minimizer in (a, b), f'(a) < 0 < f'(b), as an arb midpoint narrower than 2^-112 |x|."""
    from flint import ctx

    old = ctx.prec
    ctx.prec = 320
    try:
        lo, hi = arb(a), arb(b)
        for _ in range(4000):
            mid = (lo + hi) / 2
            w = abs(hi - lo)
            if w <= abs(mid) * arb(2) ** -112 or w <= arb(2) ** -1100:
                break
            s = _fprime_sign(f, mid)
            if s == 0:  # the ball holds 0: f' is 0 to the working precision
                break
            lo, hi = (mid, hi) if s < 0 else (lo, mid)
        return to_dd((lo + hi) / 2)
    finally:
        ctx.prec = old


def _local_minima(f, a, b, ends=False):
    """The local minimizers of f on [a, b] as [hi, lo] pairs (ends: the end points too, where f'
    points out of the interval)."""
    from tests.optimize import ad

    xs = np.linspace(a, b, GRID)
    with np.errstate(all="ignore"):
        d = np.array([float(f(ad.Dual2.variables([float(x)])[0], ad.AD).g[0]) for x in xs])
    out = []
    if ends and d[0] > 0:
        out.append([float(a), 0.0])
    s = np.sign(d)
    for i in range(GRID - 1):
        if s[i] < 0 and s[i + 1] >= 0:
            j = i + 1
            while j < GRID - 1 and s[j] == 0:
                j += 1
            if s[j] > 0 or j == GRID - 1:
                out.append(list(_bisect_min(f, float(xs[i]), float(xs[j]))))
    if ends and d[-1] < 0:
        out.append([float(b), 0.0])
    return out


def minima():
    import json

    from tests.optimize import functions

    doc = {}
    for c in functions.MIN_CASES:
        lo, hi = c.bounds
        if c.xmin is not None:  # scipy's (x - 1.5)^2 - 0.8: brent and golden end at 1.5
            bracket = [[float(c.xmin), 0.0]]
        else:
            bracket = _local_minima(c.f, float(min(c.bracket)), float(max(c.bracket)))
        doc[c.id] = {"bracket": bracket, "bounds": _local_minima(c.f, float(lo), float(hi), ends=True),
                     "interval": [float(lo), float(hi)]}  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scalar_minima.json").write_text(json.dumps(doc, indent=1) + "\n")
    n = sum(len(v["bracket"]) + len(v["bounds"]) for v in doc.values())
    print(f"{len(doc)} cases, {n} minimizers -> {(OUT / 'scalar_minima.json').relative_to(ROOT)}")


def main():
    if sys.argv[1:] == ["minima"]:
        return minima()
    from tests.optimize import ad, mgh

    arrays = {}
    cases = [c for c in mgh.CASES if c.factor == 1.0]
    for i, c in enumerate(cases):
        x0 = [float(v) for v in c.x0()]

        def jac(c=c, x0=x0):
            x = [arb(v) for v in x0]
            return [e for row in c.jac(x, M=ad.ARB) for e in row]

        def grad(c=c, x0=x0):
            x = [arb(v) for v in x0]
            J, F = c.jac(x, M=ad.ARB), c.fun(x, M=ad.ARB)
            return [2 * sum((arb(J[k][j]) * F[k] for k in range(c.m)), arb(0)) for j in range(c.n)]

        jh, jl = _settled_vec(jac)
        gh, gl = _settled_vec(grad)
        arrays.update({f"g{i}_id": c.id, f"g{i}_x0": np.array(x0), f"g{i}_g_hi": gh, f"g{i}_g_lo": gl,
                       f"g{i}_J_hi": jh.reshape(c.m, c.n), f"g{i}_J_lo": jl.reshape(c.m, c.n)})  # fmt: skip
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "derivatives.npz", **arrays)
    print(f"{len(cases)} cases -> {(OUT / 'derivatives.npz').relative_to(ROOT)}")
    minima()


if __name__ == "__main__":
    main()
