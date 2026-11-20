"""The Arb references of sd.optimize (docs/design/optimize.md §7), in tests/optimize/ref/:

    .venv/bin/python tools/optimize_ref.py                  # all three
    .venv/bin/python tools/optimize_ref.py roots minima     # only these

  roots        scalar_roots.json: for each scalar root case of tests/optimize/functions.py, the
               exact root of the function as written (its constants are the doubles in the code)
               located to one ULP: {"kind": "bracket", "a", "b"} (adjacent doubles with f of
               opposite signs) or {"kind": "exact", "x"} (f exactly 0 there)
  minima       scalar_minima.json: for each minimize_scalar case (MIN_CASES), the local
               minimizers of f as [hi, lo] pairs: "bracket", those inside the bracket's outer
               points (brent and golden search there; scipy's two-point bracket is widened
               downhill, its cases' minimizer is the known one), and "bounds", those of f on
               [lower, upper] (bounded), the end points included where f' points out of the interval
  derivatives  derivatives.npz (layout: tests/reffile.py): for each MGH case with factor 1 (the 47
               cases of tests/optimize/test_diff.py), at its starting point x0, the Jacobian of the
               residuals and the gradient g = 2 J^T F of f = sum F_i^2, from the problems'
               analytic formulas (tests/optimize/mgh.py) evaluated in Arb on the double x0
               (tests/optimize/ad.py's ARB), settled to 2^-110 of the largest entry (approx_fprime's
               error is measured normwise), as hi + lo

A root: the sign of f at a double x is decided by evaluating the case's code in Arb balls (x is
exact) at rising precision until the ball excludes 0. Bisection runs over the doubles themselves
(in the order of their bit patterns), so it ends after at most 64 halvings with adjacent doubles
a < b where f(a) and f(b) have opposite signs, or at a double where f is exactly 0; a sign that no
precision up to ROOT_PREC settles raises. A minimizer is a sign change of f' from - to + (f' in
double, by forward-mode differentiation, on a grid of GRID points), refined by bisection on the
sign of the exact f' in Arb balls (forward-mode over arb) until the interval is narrower than
2^-112 |x|; the pair is its midpoint. The JSON files carry "_provenance" (tests/optimize/refdata.py
leaves it out). The other optimize references are files already: NIST StRD's certified values
(tests/regression/data/nls) and MGH's optima (tests/optimize/mgh.py KNOWN). An unknown name is a
usage error (exit status 2); the console output also goes to build/logs/optimize_ref/. No mpmath.
"""

from __future__ import annotations

import json
import pathlib
import struct
import sys

import numpy as np
from flint import arb, ctx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import arbref  # noqa: E402
from arbref import MAX_PREC, START_PREC, NotSettled, precision, settled, to_dd  # noqa: E402

OUT = ROOT / "tests" / "optimize" / "ref"
ROOT_PREC = 4096  # the roots' sign: the largest precision tried
TOOL = "tools/optimize_ref.py"


def _provenance(bits):
    return {**arbref.provenance(bits), "tool": TOOL, "n_requested": -1}


def _ordered(x):
    """A double's position among all doubles (monotone in x, -0 and +0 together)."""
    (i,) = struct.unpack("<q", struct.pack("<d", float(x)))
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)


def _double(k):
    """The inverse of _ordered."""
    bits = k if k >= 0 else ((-k) | (1 << 63))
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def sign(f, x):
    """The sign of f at the double x: -1, 0 (exactly zero) or 1."""
    from tests.optimize.ad import ARB

    old = ctx.prec
    try:
        prec = 64
        while prec <= ROOT_PREC:
            ctx.prec = prec
            v = f(arb(float(x)), ARB)
            v = v if isinstance(v, arb) else arb(v)
            if v.is_zero():
                return 0
            if v > 0:
                return 1
            if v < 0:
                return -1
            prec *= 2
        raise ArithmeticError(f"the sign of f at {x!r} does not settle by {ROOT_PREC} bits")
    finally:
        ctx.prec = old


def root_ulp(f, a, b):
    """The root of f in [a, b] (f(a), f(b) of opposite signs) to one ULP."""
    sa, sb = sign(f, a), sign(f, b)
    if sa == 0:
        return {"kind": "exact", "x": float(a)}
    if sb == 0:
        return {"kind": "exact", "x": float(b)}
    if sa == sb:
        raise ValueError(f"f has the same sign at both ends of [{a!r}, {b!r}]")
    ka, kb = _ordered(a), _ordered(b)
    while kb - ka > 1:
        km = (ka + kb) // 2
        xm = _double(km)
        sm = sign(f, xm)
        if sm == 0:
            return {"kind": "exact", "x": xm}
        if sm == sa:
            ka = km
        else:
            kb = km
    return {"kind": "bracket", "a": _double(ka), "b": _double(kb)}


def roots():
    from tests.optimize import functions as fn

    out = {"_provenance": _provenance(52)}  # a bracket of adjacent doubles: the root to 2^-52
    for c in fn.ROOT_CASES:
        out[c.id] = root_ulp(c.f, *c.bracket)
    lines = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()]
    (OUT / "scalar_roots.json").write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(len(out) - 1, "root cases ->", (OUT / "scalar_roots.json").relative_to(ROOT))


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
    from tests.optimize import functions

    doc = {"_provenance": _provenance(112)}
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
    n = sum(len(v["bracket"]) + len(v["bounds"]) for k, v in doc.items() if k != "_provenance")
    print(f"{len(doc) - 1} cases, {n} minimizers -> {(OUT / 'scalar_minima.json').relative_to(ROOT)}")


def derivatives():
    from tests import reffile
    from tests.optimize import ad, mgh

    cases = [c for c in mgh.CASES if c.factor == 1.0]
    x0s, gs, js = [], [], []
    for c in cases:
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
        x0s.append(np.array(x0))
        gs.append((gh, gl))
        js.append((jh.reshape(c.m, c.n), jl.reshape(c.m, c.n)))
    idx = np.arange(len(cases))
    reffile.write(
        OUT / "derivatives.npz",
        **arbref.provenance(),
        tool=TOOL,
        n_requested=len(cases),
        inputs={"id": np.array([c.id for c in cases]), "x0": x0s},
        methods={
            "gradient": {"over": "point", "index": idx, "hi": [g[0] for g in gs], "lo": [g[1] for g in gs]},
            "jacobian": {"over": "point", "index": idx, "hi": [j[0] for j in js], "lo": [j[1] for j in js]},
        },
    )
    print(f"{len(cases)} cases -> {(OUT / 'derivatives.npz').relative_to(ROOT)}")


PARTS = {"roots": roots, "minima": minima, "derivatives": derivatives}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(prog="tools/optimize_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="NAME", help=", ".join(PARTS))
    a = ap.parse_args(argv)
    bad = [n for n in a.names if n not in PARTS]
    if bad:
        ap.error(f"unknown reference {bad[0]!r}")
    OUT.mkdir(parents=True, exist_ok=True)
    with arbref.logged("optimize_ref"):
        for n in a.names or PARTS:
            PARTS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
