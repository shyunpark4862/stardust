"""Compare the inverses before and after on one point set, and judge both by Arb.

`inverse_sweep.cpp` prints `method,p,p0,p1,x` per line; built against the new headers and
against the old ones it gives two files with the same points. For every point this computes
with Arb the probability at the returned x on the side that was asked (the smaller of the
target and its complement) and its relative distance from the target: the backward error. An
inverse whose backward error is below 1e-14 is inside the contract 1e-14 max(1, kappa) (T24).

    .venv/bin/python bench/stats/opt/b/inverse_check.py new.csv old.csv
"""

import math
import sys
from collections import defaultdict

from arb_check import CDF, A
from flint import ctx

ctx.prec = 400


def backward(method, p, p0, p1, x):
    dist, m = method.split("_")
    if not (math.isfinite(x) and x == x):
        return None
    lo, up = CDF[dist](A(x), A(p0), A(p1))
    if m in ("ilogcdf", "ilogccdf"):
        target = A(p).exp()  # p is the log
        m = m.replace("ilog", "i")
    else:
        target = A(p)
    got = lo if m == "icdf" else up
    other = up if m == "icdf" else lo
    if target > 0.5:  # the complement is the smaller side
        target, got = 1 - target, other
    if not (target > 0) or not got.is_finite() or float(target.mid()) < 1e-290:
        return None
    e = abs(got - target) / target
    if float(e.rad()) > 1e-18:
        return None
    return float(e.upper())


def main():
    new, old = open(sys.argv[1]).readlines(), open(sys.argv[2]).readlines()
    w = defaultdict(lambda: [0, 0, 0.0, 0.0, 0.0, None])
    for ln, lo_ in zip(new, old):
        method, p, p0, p1, xn = ln.strip().split(",")
        xo = lo_.strip().split(",")[4]
        p, p0, p1, xn, xo = map(float, (p, p0, p1, xn, xo))
        r = w[method]
        r[0] += 1
        if xn != xo:
            r[1] += 1
            if xo != 0 and math.isfinite(xo) and math.isfinite(xn):
                r[2] = max(r[2], abs(xn / xo - 1))
            en, eo = backward(method, p, p0, p1, xn), backward(method, p, p0, p1, xo)
            if en is not None and en > r[3]:
                r[3], r[5] = en, (p, p0, p1)
            if eo is not None:
                r[4] = max(r[4], eo)
    print(
        "| method | points | changed | max rel change of x | backward error, new (changed points) | old | new's worst (p, p0, p1) |"
    )
    print("|---|---:|---:|---:|---:|---:|---|")
    for m, (n, ch, d, en, eo, pt) in w.items():
        print(f"| `{m}` | {n} | {ch} | {d:.3g} | {en:.3g} | {eo:.3g} | {pt} |")


if __name__ == "__main__":
    main()
