"""Smith division's special values, lambertw's branches on the cut, AAA zeros, and the nodes of the
noncentral t tail's sinh-mapped trapezoid (a transcription of the node loop with the reference integrand)."""
import math, cmath
import numpy as np
import stardust as sd

def one(f, *zs):
    return complex(np.asarray(f(*[sd.asarray(np.array([z])) for z in zs]))[0])

def smith():
    inf = math.inf
    cases = [(complex(1, 1), complex(inf, inf)), (complex(1, 0), complex(0, 0)), (complex(1, 1), complex(1e300, 1e300)),
             (complex(1e-300, 1e-300), complex(1e-300, 2e-300)), (complex(inf, 0), complex(1, 1))]
    for a, b in cases:
        lib = one(sd.divide, a, b) if hasattr(sd, "divide") else one(lambda x, y: x / y, a, b)
        try: py = a / b
        except ZeroDivisionError: py = "ZeroDivisionError"
        naive = None
        try:
            d = b.real * b.real + b.imag * b.imag
            naive = complex((a.real * b.real + a.imag * b.imag) / d, (a.imag * b.real - a.real * b.imag) / d)
        except ZeroDivisionError: naive = "ZeroDivisionError"
        print("smith", a, "/", b, "lib", lib, "python", py, "naive", naive)

def lambertw():
    for z in (complex(-0.2, 0.0), complex(-0.2, -0.0)):
        for k in (-1, 0, 1):
            v = complex(np.asarray(sd.special.lambertw(sd.asarray(np.array([z])), k=k))[0])
            print("lambertw", z, "k", k, v)

def aaa_roots():
    x = np.linspace(-1, 1, 1000)
    f = 1 / (x - 1.5) - 3 / (x + 2) + 2 * x / (x * x + 0.25)
    r = sd.interpolate.AAA(sd.asarray(x), sd.asarray(f))
    print("exact rational roots", np.round(np.asarray(r.roots()), 6))
    # true zeros: numerator of the rational
    # r = [ (x+2)(x^2+1/4) - 3 (x-3/2)(x^2+1/4) + 2x (x-3/2)(x+2) ] / ...
    p = np.polyadd(np.polyadd(np.polymul([1, 2], [1, 0, 0.25]), np.polymul([-3, 4.5], [1, 0, 0.25])), np.polymul([2, 0], np.polymul([1, -1.5], [1, 2])))
    print("  true zeros", np.roots(p))
    r2 = sd.interpolate.AAA(sd.asarray(x), sd.asarray(np.tan(x)))
    print("tan roots", np.round(np.sort_complex(np.asarray(r2.roots())), 6))

def nct_nodes(df=5.0, a=5.0, t=2.0):
    lc = math.log(2) + (df / 2) * math.log(df / 2) - math.lgamma(df / 2)
    def g(v):
        s = math.exp(v)
        if s > 1e150: return 0.0
        return 0.5 * math.erfc((a + t * s) / math.sqrt(2)) * math.exp(lc + df * v - df * s * s / 2)
    # v*: stationary point of the Gaussian part (the library's closed form)
    sstar = 2 * df / (a * t + math.hypot(t * math.sqrt(a * a + 4 * df), 2 * df))
    vs = math.log(sstar)
    p = t * sstar; z = a + p
    kappa = p * p + z * p + 2 * df * sstar * sstar
    sigma = 1 / math.sqrt(kappa)
    c, A, kap = min(sigma, 0.4), 8.0, 0.25
    nodes = [(vs, g(vs))]
    S = g(vs)
    for d in (1, -1):
        j = 1
        while True:
            s = d * j * kap
            v = vs + c * A * math.sinh(s / A)
            q = g(v) * math.cosh(s / A)
            S += q; nodes.append((v, g(v)))
            if not (q > 2 ** -60 * S) and j > 8: break
            j += 1
    print("nct nodes df=%g a=%g t=%g: sigma %.4f c %.4f nodes %d, v range [%.2f, %.2f]" % (df, a, t, sigma, c, len(nodes), min(v for v, _ in nodes), max(v for v, _ in nodes)))
    return vs, sorted(nodes)

if __name__ == "__main__":
    smith(); lambertw(); aaa_roots()
    for df, a, t in [(1, 1, 0.5), (5, 5, 2), (30, 5, 8), (1, 5, 0.5), (100, 1, 2)]:
        nct_nodes(df, a, t)
    vs, nodes = nct_nodes(5.0, 5.0, 2.0)
    gmax = max(gv for _, gv in nodes)
    with open("nct_nodes.dat", "w") as f:
        f.write("v g\n")
        for v, gv in nodes:
            f.write("%.5f %.5e\n" % (v - vs, max(gv / gmax, 1e-30)))
