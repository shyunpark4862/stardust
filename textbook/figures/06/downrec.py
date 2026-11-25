"""The downward recurrence J_{v-1} = (2v/x) J_v - J_{v+1} from exact J_286.3(100), J_287.3(100) down to order 0.3:
float64 against double-double (csrc/special/ops/bessel.hpp jy_temme keeps it in double-double: "86 ULP at nu = 286,
x = 100"). Errors of J_v relative to the envelope sqrt(2/(pi x)) for v > x... reported relative to |J_v| (u)."""
import math
from fractions import Fraction as F
from dref import jv as jref, dec, D
from longrec import dd_add, dd_mul

def run(mu=0.3, x=100.0, n=286):
    nu = mu + n  # a double; the orders nu - j are exact doubles (the library's choice)
    ex = {k: jref(dec(nu) - (n - k), x) for k in range(0, n + 2)}
    xr = F(1) / F(x); xh = float(xr); xl = float(xr - F(xh))
    p0, p1 = float(ex[n]), float(ex[n + 1])          # plain: J_v, J_{v+1}
    def todd(d):
        h = float(d); return (h, float(d - D(repr(h))))
    d0, d1 = todd(ex[n]), todd(ex[n + 1])
    worst_p = worst_d = 0; out = []
    for k in range(n, 0, -1):
        v = nu - (n - k)
        p0, p1 = 2 * v / x * p0 - p1, p0
        cf = dd_mul((2 * v, 0.0), (xh, xl))
        t = dd_mul(cf, d0); d0, d1 = dd_add(t, (-d1[0], -d1[1])), d0
        e = ex[k - 1]; sc = max(abs(e), abs(ex[k]))  # the scale of the pair (a zero of J_v alone would blow up)
        rp = float(abs(dec(p0) - e) / sc) / 2 ** -53
        rd = float(abs(dec(d0[0]) - e) / sc) / 2 ** -53
        out.append((k - 1, rp, rd))
    with open("downrec.dat", "w") as f:
        f.write("v plain dd\n")
        for o in out: f.write("%d %.4f %.4f\n" % (o[0], max(o[1], 0.01), max(o[2], 0.01)))
    for o in out[::-40]: print("order %3d: float64 %.1f u, double-double %.2f u (relative to max(|J_v|, |J_v+1|))" % o)
    import numpy as np
    a = np.array(out)
    m = a[:, 0] > 0  # avoid zeros of J (relative error) by reporting the median over orders below x as well
    print("max over orders > x: float64 %.1f u, dd %.2f u" % (a[a[:, 0] > x, 1].max(), a[a[:, 0] > x, 2].max()))
    print("at order 0.3: float64 %.1f u, dd %.2f u" % (a[-1, 1], a[-1, 2]))

if __name__ == "__main__":
    run()
