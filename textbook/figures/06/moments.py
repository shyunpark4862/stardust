"""QUADPACK's modified Chebyshev moments of cos(p x) (dqc25f, csrc/integrate/quadpack.hpp): v_j for the even
Chebyshev degrees 2(j-1). Forward recurrence in binary64 against the same recurrence in 100-digit Decimal (exact up
to that rounding), for p = 10 (forward unstable: the library solves a tridiagonal boundary-value problem) and
p = 30 (forward stable, the library's branch). The boundary-value path is a transcription of qc25f + dgtsl.
Also stardust's integrate.quad with weight 'cos' on a known integral. Output: moments.dat."""
import math
from decimal import Decimal as D, getcontext
getcontext().prec = 100
import numpy as np

def dsin_cos(p):
    from dref import PI
    p = D(p) % (2 * PI); x2 = p * p; s = p; c = D(1); ts = p; tc = D(1); k = 1
    while True:
        ts = -ts * x2 / ((2 * k) * (2 * k + 1)); tc = -tc * x2 / ((2 * k - 1) * (2 * k)); s += ts; c += tc; k += 1
        if abs(ts) < D(10) ** -110 and abs(tc) < D(10) ** -110: break
    return s, c

def forward(p, n, dec=False):
    if dec:
        P = D(repr(p)); sp, cp = dsin_cos(P); one = D(1)
    else:
        P = p; sp, cp = math.sin(p), math.cos(p); one = 1.0
    p2 = P * P; p22 = p2 + 2
    v = [None] * (n + 1)
    v[1] = 2 * sp / P
    v[2] = (8 * cp + (p2 + p2 - 8) * sp / P) / p2
    v[3] = (32 * (p2 - 12) * cp + 2 * ((p2 - 80) * p2 + 192) * sp / P) / (p2 * p2)
    ac = 8 * cp; as_ = 24 * P * sp
    an = 4 * one
    for i in range(4, n + 1):
        an2 = an * an
        t = (an2 - 4) * (2 * (p22 - an2 - an2) * v[i - 1] - ac) + as_
        v[i] = (-(p2 * (an + 1) * (an + 2)) * v[i - 2] + t) / (p2 * (an - 1) * (an - 2))
        an += 2
    return v

def bvp(p):
    """qc25f's |p| <= 24 branch for the cosine moments: 25 equations, the last from the asymptotic asap."""
    par2 = p * p; par22 = par2 + 2; sinpar = math.sin(p); cospar = math.cos(p)
    v = [0.0] * 29
    v[1] = 2 * sinpar / p
    v[2] = (8 * cospar + (par2 + par2 - 8) * sinpar / p) / par2
    v[3] = (32 * (par2 - 12) * cospar + (2 * ((par2 - 80) * par2 + 192) * sinpar) / p) / (par2 * par2)
    ac = 8 * cospar; as_ = 24 * p * sinpar
    d = [0.0] * 27; d1 = [0.0] * 27; d2 = [0.0] * 27
    an = 6.0
    for k in range(1, 25):
        an2 = an * an
        d[k] = -2 * (an2 - 4) * (par22 - an2 - an2)
        d2[k] = (an - 1) * (an - 2) * par2
        d1[k + 1] = (an + 3) * (an + 4) * par2
        v[k + 3] = as_ - (an2 - 4) * ac
        an += 2
    an2 = an * an
    d[25] = -2 * (an2 - 4) * (par22 - an2 - an2)
    v[28] = as_ - (an2 - 4) * ac
    v[4] = v[4] - 56 * par2 * v[3]
    ass = p * sinpar
    asap = (((210 * par2 - 1) * cospar - (105 * par2 - 63) * ass) / an2 - (1 - 15 * par2) * cospar + 15 * ass) / an2
    asap = ((asap - cospar + 3 * ass) / an2 - cospar) / an2
    v[28] = v[28] - 2 * asap * par2 * (an - 1) * (an - 2)
    # dgtsl on (c = d1, d = d, e = d2, b = v[3:]) 1-based
    c = d1; e = d2; b = [0.0] + v[4:29]  # b[1..25] = v[4..28]
    n = 25
    for k in range(1, n - 1):
        if abs(d[k]) < abs(c[k + 1]):
            t = -d[k] / c[k + 1]; dk1 = d[k + 1]
            d[k] = c[k + 1]; d[k + 1] = t * dk1 + e[k]; c[k + 1] = e[k + 1]; e[k] = dk1; e[k + 1] = t * e[k + 1]
            bk = b[k]; b[k] = b[k + 1]; b[k + 1] = t * b[k + 1] + bk
        else:
            t = -c[k + 1] / d[k]; d[k + 1] = t * e[k] + d[k + 1]; b[k + 1] = t * b[k] + b[k + 1]; c[k + 1] = 0.0
    b[n] = b[n] / d[n]; b[n - 1] = (b[n - 1] - e[n - 1] * b[n]) / d[n - 1]
    for k in range(n - 2, 0, -1): b[k] = (b[k] - e[k] * b[k + 1] - c[k + 1] * b[k + 2]) / d[k]
    return v[:4] + b[1:14]  # v[1..3] and v[4..16]

if __name__ == "__main__":
    rows = []
    for p in (10.0, 30.0):
        ref = forward(p, 16, dec=True); fw = forward(p, 16)
        bv = bvp(p) if p <= 24 else None
        for j in range(1, 14):
            r = ref[j]
            ef = float(abs(D(repr(fw[j])) - r) / abs(r))
            eb = float(abs(D(repr(bv[j])) - r) / abs(r)) if bv else float('nan')
            rows.append((p, 2 * (j - 1), max(ef, 1e-17), max(eb, 1e-17) if bv else float('nan'), float(r)))
    with open("moments.dat", "w") as f:
        f.write("p deg fwd bvp ref\n")
        for r in rows: f.write("%g %d %.4e %.4e %.6e\n" % r)
    for r in rows: print("p %g deg %2d fwd %.2e bvp %.2e  value %.3e" % r)

def library_quad():
    """stardust integrate.quad(exp, 0, 2, weight='cos', wvar=w): one 25-point step; exact by Decimal."""
    import stardust as sd
    getcontext().prec = 60
    for w in (10.0, 20.0, 40.0, 60.0, 200.0):
        r = float(sd.integrate.quad(lambda x: math.exp(x), 0.0, 2.0, weight='cos', wvar=w).integral)
        W = D(repr(w)); s2, c2 = dsin_cos(2 * W); e2 = D(2).exp()
        ex = (e2 * (c2 + W * s2) - 1) / (1 + W * W)
        print("quad weight cos w=%g (p = w h = %g): rel err %.2e" % (w, w, float(abs(D(repr(r)) - ex) / abs(ex))))
