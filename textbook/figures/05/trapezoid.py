"""Exponential convergence of the trapezoid rule.
(1) periodic: int_0^{2pi} e^{cos t} dt = 2 pi I_0(1), N equispaced points (sd.exp, sd.cos, sd.integrate.trapezoid),
    and the nonperiodic int_0^1 e^x dx for contrast. Reference: Decimal series.
(2) the real line: int exp(-k (e^{2v} - 1 - 2v)) dv = e^k Gamma(k) / (2 k^k) (s = e^{2v}), whose integrand is
    integrable on lines Im v = y only for |y| < pi/4: error against h, with the prediction e^{-2 pi (pi/4) / h}.
(3) sd.stats.NoncentralT(df, nc < 0).ccdf(t): the library's sinh-mapped trapezoid (special/ops/noncentral.hpp)
    against the same integral by the plain trapezoid rule at h = 1/200 in fsum (math.erfc).
Output: trap_periodic.dat, trap_line.dat."""
import math
import numpy as np
import stardust as sd
from dec import D, getcontext, PI
getcontext().prec = 50

def I0_1():
    s, t, k = D(0), D(1), 0
    while t > D(10) ** -45:
        s += t; k += 1; t = t / (4 * k * k)
    return s

def periodic():
    ex = float(2 * PI * I0_1())
    ex2 = math.e - 1
    rows = []
    for N in range(1, 25):
        t = np.linspace(0, 2 * np.pi, N + 1)
        y = np.asarray(sd.exp(sd.cos(sd.asarray(t))))
        q = float(np.asarray(sd.integrate.trapezoid(sd.asarray(y), sd.asarray(t))))
        x = np.linspace(0, 1, N + 1)
        q2 = float(np.asarray(sd.integrate.trapezoid(sd.exp(sd.asarray(x)), sd.asarray(x))))
        rows.append((N, max(abs(q - ex) / ex, 1e-17), abs(q2 - ex2) / ex2))
    with open("trap_periodic.dat", "w") as f:
        f.write("N per lin\n")
        for r in rows: f.write("%d %.4e %.4e\n" % r)
    for r in rows[:18:2]: print("N %2d periodic %.2e  [0,1] %.2e" % r)

def pred(k, h):
    """exp(-max_d [2 pi d / h + k log cos 2d]), d < pi/4: the strip bound of the real-line theorem with
    M(d) ~ (cos 2d)^{-k}, the integrand's largest modulus on the line Im v = d (at e^{2x} cos 2d = 1)."""
    ds = np.linspace(1e-4, math.pi / 4 - 1e-4, 4000)
    return math.exp(-np.max(2 * math.pi * ds / h + k * np.log(np.cos(2 * ds))))

def line():
    rows = []
    for k in (0.5, 2.5, 50.0):
        ex = math.exp(k + math.lgamma(k) - k * math.log(k)) / 2
        f = lambda v: math.exp(-k * (math.expm1(2 * v) - 2 * v)) if v < 30 else 0.0
        for h in np.geomspace(0.05, 1.5, 40):
            # nodes j h on both sides until the terms fade
            terms = [f(0.0)]
            for sgn in (1, -1):
                j = 1
                while True:
                    v = f(sgn * j * h); terms.append(v)
                    if v < 1e-40 * terms[0] and j > 5: break
                    j += 1
            q = h * math.fsum(terms)
            rows.append((k, h, max(abs(q - ex) / ex, 1e-17)))
    with open("trap_line.dat", "w") as f:
        f.write("k h err pred\n")
        for k, h, e in rows: f.write("%.1f %.5f %.4e %.4e\n" % (k, h, e, pred(k, h)))
    for k in (0.5, 2.5, 50.0):
        sub = [r for r in rows if r[0] == k]
        for r in sub[::8]: print("k %.1f h %.3f err %.2e pred %.2e" % (r[0], r[1], r[2], pred(r[0], r[1])))

def nct_ref(df, a, t):
    # Q = P(T > t | nc = -a) = int Phi(-(a + t s)) f_S(s) ds, s = e^v
    lc = math.log(2) + (df / 2) * math.log(df / 2) - math.lgamma(df / 2)
    def g(v):
        s = math.exp(v)
        if s > 1e150: return 0.0
        lf = lc + (df - 1) * v - df * s * s / 2 + v
        return 0.5 * math.erfc((a + t * s) / math.sqrt(2)) * math.exp(lf)
    h = 1 / 200
    return h * math.fsum(g(j * h) for j in range(-20000 * 4, 4000))

def nct():
    worst = 0
    for df in (1.0, 5.0, 30.0):
        for nc in (-1.0, -5.0):
            for t in (0.5, 2.0, 8.0):
                lib = float(np.asarray(sd.stats.NoncentralT(df=df, nc=nc).ccdf(t)))
                r = nct_ref(df, -nc, t)
                e = abs(lib - r) / r; worst = max(worst, e)
                print("df %4.1f nc %4.1f t %4.1f  sf %.6e  rel %.1e" % (df, nc, t, lib, e))
    print("nct worst", worst)

if __name__ == "__main__":
    periodic(); line(); nct()
