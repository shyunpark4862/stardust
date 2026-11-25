"""loggamma's branch against an independent reference.
Reference (Decimal, about 50 digits): Stirling's series at z + N (N = 40, 24 terms) minus the sum of the principal
logarithms Log(z + j), j < N -- the continuation as a sum of logs, with no crossings counted. The library: the
log of a product with its crossings counted (special/ops/loggamma.hpp).
(1) error |lib - ref| / max(1, |ref|) on a grid in [-10, 10]^2 and on both sides of the cut (y = +-0);
(2) the staircase Im loggamma(x +- 0i) for x in [-6, 1]: loggamma_stair.dat;
(3) the residual loggamma(z + 1) - loggamma(z) - log z on the grid, and the same with log(Gamma(z)) for contrast."""
import math, cmath
from fractions import Fraction as F
import numpy as np
import stardust as sd
from dec import D, getcontext, PI, atan2, cos_sin
getcontext().prec = 60

def bern(n):
    A = [F(0)] * (n + 1)
    B = []
    for m in range(n + 1):
        A[m] = F(1, m + 1)
        for j in range(m, 0, -1): A[j - 1] = j * (A[j - 1] - A[j])
        B.append(A[0])
    return B  # B_1 = +1/2 convention irrelevant here (even ones only)
BN = bern(60)

def cdec(z): 
    a, b = F(z.real), F(z.imag)
    return (D(a.numerator) / D(a.denominator), D(b.numerator) / D(b.denominator))

def cmul(a, b): return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])
def cinv(a):
    d = a[0] * a[0] + a[1] * a[1]; return (a[0] / d, -a[1] / d)
def clog(a, negzero=False):
    re = ((a[0] * a[0] + a[1] * a[1]).ln()) / 2
    if a[1] == 0 and a[0] < 0: return (re, -PI if negzero else PI)
    return (re, atan2(a[1], a[0]))

def ref(z):
    x, y = cdec(z); neg = math.copysign(1, z.imag) < 0
    N = max(40, 40 - int(math.floor(z.real)))  # z + N well inside the right half-plane
    w = (x + N, y)
    lw = clog(w)
    zh = (w[0] - D("0.5"), w[1])
    s = cmul(zh, lw)
    s = (s[0] - w[0] + (2 * PI).ln() / 2, s[1] - w[1])
    rw = cinv(w); rw2 = cmul(rw, rw); p = rw
    for k in range(1, 25):
        c = BN[2 * k] / (2 * k * (2 * k - 1)); cD = D(c.numerator) / D(c.denominator)
        s = (s[0] + cD * p[0], s[1] + cD * p[1]); p = cmul(p, rw2)
    for j in range(N):
        l = clog((x + j, y), neg)
        s = (s[0] - l[0], s[1] - l[1])
    return complex(float(s[0]), float(s[1]))

def lib(zs):
    return np.asarray(sd.special.loggamma(sd.asarray(np.asarray(zs, complex))))

def grid():
    rng = np.random.default_rng(1)
    zs = list(rng.uniform(-10, 10, 1500) + 1j * rng.uniform(-10, 10, 1500))
    zs += list(rng.uniform(-10, 10, 300) + 1j * rng.uniform(-1e-3, 1e-3, 300))
    xs = rng.uniform(-10, 0, 200)
    zs += [complex(x, 0.0) for x in xs] + [complex(x, -0.0) for x in xs]
    zs = [z for z in zs if min(abs(z + k) for k in range(0, 12)) > 1e-6]
    L = lib(zs)
    errs = []
    for z, l in zip(zs, L):
        r = ref(z); errs.append((z, abs(l - r) / max(1.0, abs(r)), abs(l - r) / abs(r)))
    e = np.array([t[1] for t in errs])
    rp = np.array([t[2] for t in errs if t[0].real > 0])
    print("loggamma vs ref: n", len(zs), "max err/max(1,|ref|)", e.max(), "Re z > 0 max rel", rp.max())
    side = [t for t in errs if t[0].imag == 0]
    print("  on the cut (y = +-0): max", max(t[1] for t in side))

def stair():
    xs = np.linspace(-6, 1, 1401)
    xs = xs[np.min(abs(xs[:, None] + np.arange(0, 7)[None, :]), axis=1) > 1e-9]
    up = lib([complex(x, 0.0) for x in xs]).imag
    dn = lib([complex(x, -0.0) for x in xs]).imag
    g = np.array([math.gamma(x) for x in xs])
    lg = np.where(g < 0, math.pi, 0.0)  # Im Log Gamma(x): the principal log of the real Gamma
    xh = np.linspace(-6, 1, 1401)
    lh = lib(xh + 0.5j)
    arg = np.angle(np.exp(lh))  # Arg Gamma(x + i/2): the principal value, wrapping at +-pi
    with open("loggamma_half.dat", "w") as f:
        f.write("x im arg\n")
        for row in zip(xh, lh.imag / math.pi, arg / math.pi): f.write("%.5f %.6f %.6f\n" % row)
    with open("loggamma_stair.dat", "w") as f:
        f.write("x up dn logg\n")
        for row in zip(xs, up / math.pi, dn / math.pi, lg / math.pi): f.write("%.5f %.6f %.6f %.6f\n" % row)

def residual():
    rng = np.random.default_rng(2)
    zs = rng.uniform(-8, 8, 4000) + 1j * rng.uniform(-8, 8, 4000)
    a = lib(zs); b = lib(zs + 1); lz = np.log(zs)
    r = b - a - lz
    print("residual loggamma(z+1)-loggamma(z)-log z: max |.|", abs(r).max(), "max |Im|/2pi", abs(r.imag).max() / (2 * math.pi))
    # log Gamma(z) principal from the library's gamma, if complex gamma exists
    try:
        g = np.asarray(sd.special.gamma(sd.asarray(zs)))
        g1 = np.asarray(sd.special.gamma(sd.asarray(zs + 1)))
        rr = np.log(g1) - np.log(g) - lz
        k = np.round(rr.imag / (2 * math.pi))
        print("  with Log Gamma: fraction off by 2 pi k != 0:", np.mean(k != 0))
    except Exception as ex:
        print("  no complex gamma:", ex)

def far_left():
    """Stirling's series used directly where |Im z| > 7, also far to the left (arg z near pi)."""
    rng = np.random.default_rng(3)
    zs = list(rng.uniform(-300, 0, 300) + 1j * rng.choice([-1, 1], 300) * rng.uniform(7.0001, 12, 300))
    L = lib(zs)
    e = [abs(l - ref(z)) / max(1.0, abs(ref(z))) for z, l in zip(zs, L)]
    print("far left, 7 < |Im z| < 12, -300 < Re z < 0: max err/max(1,|ref|)", max(e))

if __name__ == "__main__":
    grid(); stair(); residual(); far_left()
