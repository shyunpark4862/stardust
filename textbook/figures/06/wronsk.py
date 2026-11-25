"""Wronskian residual of stardust's jv, yv over the order: r = |x (J_{v+1} Y_v - J_v Y_{v+1}) pi/2 - 1| divided by
the size of the products x (|J_{v+1} Y_v| + |J_v Y_{v+1}|) pi/2 (1 when they do not cancel), for v in [0.25, 1000]
at x = 10, 100, 300, across the float64 path (v < 500) and the double-double path. Output: wronsk.dat."""
import math
import numpy as np
import stardust as sd

def res(nu, x):
    v = np.array([nu, nu + 1.0]); xs = np.array([x, x])
    J = np.asarray(sd.special.jv(sd.asarray(v), sd.asarray(xs))); Y = np.asarray(sd.special.yv(sd.asarray(v), sd.asarray(xs)))
    a, b = J[1] * Y[0], J[0] * Y[1]
    if not all(np.isfinite([a, b])) or a == 0 and b == 0: return float('nan')
    w = x * (a - b) * math.pi / 2 - 1
    return abs(w) / max(x * (abs(a) + abs(b)) * math.pi / 2, 1.0)

nus = np.arange(0.25, 1000.25, 7.0)  # exact: v + 1 is the next order (0.3 would not be across 512)
with open("wronsk.dat", "w") as f:
    f.write("nu x10 x100 x300\n")
    for nu in nus:
        r = [res(float(nu), x) for x in (10.0, 100.0, 300.0)]
        f.write("%.1f %s\n" % (nu, " ".join(("%.4e" % max(v, 1e-17)) if v == v else "nan" for v in r)))
a = np.genfromtxt("wronsk.dat", names=True)
for c in ("x10", "x100", "x300"):
    v = a[c]; m = np.isfinite(v)
    lo = a["nu"] < 500
    print(c, "max nu<500 %.1e, nu>=500 %.1e, valid %d" % (np.nanmax(v[lo]), np.nanmax(v[~lo]) if np.any(np.isfinite(v[~lo])) else float('nan'), m.sum()))
