"""(1) Taylor coefficients of f(z) = e^z / (z - 2)... no: of f(z) = 1 / (1 - z/2) * e^z? Keep it exact:
f(z) = e^z, a_k = 1/k!, from the trapezoid rule on |z| = r with N points (sd.fft of the samples):
a_k ~ (1/N) r^-k sum f(r w^j) w^(-jk). Error against 1/k! over r (truncation: aliasing a_{k+N} r^N, rounding
~ u max|f| / r^k). (2) f'(x) for f = e^x sin x / (cos^3 x + sin^3 x)? Use Squire-Trapp's test function
f(x) = e^x / sqrt(sin^3 x + cos^3 x) at x = 1.5: forward difference, central difference, complex step,
all through stardust's elementwise complex functions. Reference: Decimal. Output: cauchy_r.dat, cstep.dat."""
import math
import numpy as np
import stardust as sd
from dec import D, cos_sin, getcontext
from fractions import Fraction

def coeffs(r, N):
    j = np.arange(N)
    w = np.exp(2j * np.pi * j / N)  # nodes: np only for the points; the values through sd
    z = sd.asarray(r * w)
    fz = np.asarray(sd.exp(z))
    F = np.asarray(sd.fft.fft(sd.asarray(fz))) / N
    return F  # a_k r^k

def run_r():
    N = 32
    rows = []
    for r in np.geomspace(1e-3, 30, 60):
        F = coeffs(r, N)
        errs = []
        for k in (1, 5, 10):
            ak = F[k].real / r ** k
            ex = 1 / math.factorial(k)
            errs.append(abs(ak - ex) / ex)
        rows.append((r, *errs))
    with open("cauchy_r.dat", "w") as f:
        f.write("r e1 e5 e10\n")
        for row in rows: f.write("%.6e %.4e %.4e %.4e\n" % row)
    a = np.array(rows)
    for i, k in enumerate((1, 5, 10)):
        m = a[:, i + 1].argmin(); print("k", k, "best r", a[m, 0], "err", a[m, i + 1])

# f(x) = e^x / sqrt(sin^3 x + cos^3 x), Squire and Trapp 1998
def f_sd(z):
    s, c = sd.sin(z), sd.cos(z)
    return sd.exp(z) / sd.sqrt(s * s * s + c * c * c)

def dfdx_exact(x):
    getcontext().prec = 60
    X = D(Fraction(x).numerator) / D(Fraction(x).denominator)
    c, s = cos_sin(X)
    g = s ** 3 + c ** 3
    dg = 3 * s * s * c - 3 * c * c * s
    e = X.exp()
    return float(e / g.sqrt() - e * dg / (2 * g * g.sqrt()))

def run_h():
    x = 1.5
    ex = dfdx_exact(x)
    rows = []
    for h in np.geomspace(1e-300, 1e-1, 120):
        cs = float(np.asarray(f_sd(sd.asarray(np.array([complex(x, h)])))).imag[0]) / h
        fw = (float(np.asarray(f_sd(sd.asarray(np.array([x + h]))))[0]) - float(np.asarray(f_sd(sd.asarray(np.array([x]))))[0])) / h if h > 1e-17 else float('nan')
        ce = (float(np.asarray(f_sd(sd.asarray(np.array([x + h]))))[0]) - float(np.asarray(f_sd(sd.asarray(np.array([x - h]))))[0])) / (2 * h) if h > 1e-17 else float('nan')
        rows.append((h, abs(fw - ex) / abs(ex), abs(ce - ex) / abs(ex), abs(cs - ex) / abs(ex)))
    with open("cstep.dat", "w") as f:
        f.write("h fwd ctr cstep\n")
        for row in rows:
            f.write("%.4e %s %s %s\n" % (row[0], *[("%.4e" % max(v, 1e-17)) if v == v else "nan" for v in row[1:]]))
    a = np.array(rows)
    print("exact", ex)
    print("fwd best", np.nanmin(a[:, 1]), "ctr best", np.nanmin(a[:, 2]), "cstep at 1e-20", a[np.argmin(abs(a[:, 0] - 1e-20)), 3], "cstep max for h<1e-8", a[a[:, 0] < 1e-8, 3].max())

if __name__ == "__main__":
    run_r(); run_h()
