"""Reference roots for the bracketing methods of sd.optimize (docs/design/optimize.md §7):
for each scalar root case of tests/optimize/functions.py, the exact root of the function as
written (its constants are the doubles in the code) located to one ULP, with Arb.

The sign of f at a double x is decided by evaluating the case's code in Arb balls (x is exact)
at rising precision until the ball excludes 0. Bisection runs over the doubles themselves (in
the order of their bit patterns), so it ends after at most 64 halvings with adjacent doubles
a < b where f(a) and f(b) have opposite signs, or at a double where f is exactly 0 (a ball that
is exactly zero, e.g. x^4 - 1 at 1, or a branch of the code that returns 0). No mpmath.

The result per case is {"kind": "bracket", "a": a, "b": b} or {"kind": "exact", "x": x}; a
sign that no precision up to MAX_PREC settles raises.

Run: uv run --no-sync python tools/optimize_ref.py   (writes tests/optimize/data/scalar_roots.json)
"""

import json
import pathlib
import struct
import sys

from flint import arb, ctx

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "optimize" / "data" / "scalar_roots.json"
MAX_PREC = 4096


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
        while prec <= MAX_PREC:
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
        raise ArithmeticError(f"the sign of f at {x!r} does not settle by {MAX_PREC} bits")
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


def main():
    from tests.optimize import functions as fn
    out = {}
    for c in fn.ROOT_CASES:
        out[c.id] = root_ulp(c.f, *c.bracket)
    lines = [f"{json.dumps(k)}: {json.dumps(v)}" for k, v in out.items()]
    OUT.write_text("{\n" + ",\n".join(lines) + "\n}\n")
    print(len(out), "cases")


if __name__ == "__main__":
    main()
