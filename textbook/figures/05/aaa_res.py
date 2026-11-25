"""Poles and residues of AAA fits (sd.interpolate.AAA: arrowhead pencil eigenvalues, N(p)/D'(p)).
(1) tan x on 1000 points of [-1, 1]: poles (k + 1/2) pi, residues -1.
(2) Gamma(x) on 1000 points of [1/2, 4]: poles at -k, residues (-1)^k / k!.
Output: aaa_tan.dat, aaa_gamma.dat (pole, |pole error|, |residue error| / |residue|), aaa_poles.dat (all poles)."""
import math
import numpy as np
import stardust as sd

def fit(x, f):
    r = sd.interpolate.AAA(sd.asarray(x), sd.asarray(f))
    return np.asarray(r.poles()), np.asarray(r.residues())

def run():
    x = np.linspace(-1, 1, 1000)
    p, res = fit(x, np.tan(x))
    rows = []
    for k in range(-4, 4):
        t = (k + 0.5) * math.pi
        i = np.argmin(abs(p - t))
        rows.append((t, abs(p[i] - t), abs(res[i] + 1)))
    print("tan: #poles", len(p))
    for r in rows: print("  pole %.4f  err %.2e  res err %.2e" % r)
    with open("aaa_tan.dat", "w") as f:
        f.write("pole perr rerr\n")
        for r in rows: f.write("%.6f %.3e %.3e\n" % r)
    with open("aaa_tan_poles.dat", "w") as f:
        f.write("re im\n")
        for z in p: f.write("%.6f %.6f\n" % (z.real, z.imag))
    x = np.linspace(0.5, 4, 1000)
    p, res = fit(x, np.array([math.gamma(t) for t in x]))
    rows = []
    for k in range(0, 8):
        i = np.argmin(abs(p + k))
        ex = (-1) ** k / math.factorial(k)
        rows.append((-k, abs(p[i] + k), abs(res[i] - ex) / abs(ex)))
    print("gamma: #poles", len(p), "real poles", sorted(z.real for z in p if abs(z.imag) < 1e-6)[:10])
    for r in rows: print("  pole %d  err %.2e  res rel err %.2e" % r)
    with open("aaa_gamma.dat", "w") as f:
        f.write("pole perr rerr\n")
        for r in rows: f.write("%d %.3e %.3e\n" % r)
    with open("aaa_gamma_poles.dat", "w") as f:
        f.write("re im\n")
        for z in p: f.write("%.6f %.6f\n" % (z.real, z.imag))

def exact_rational():
    """r(x) = 1/(x - 3/2) - 3/(x + 2) + 2x/(x^2 + 1/4): poles 3/2, -2, +-i/2 with residues 1, -3, 1, 1."""
    x = np.linspace(-1, 1, 1000)
    f = 1 / (x - 1.5) - 3 / (x + 2) + 2 * x / (x * x + 0.25)
    r = sd.interpolate.AAA(sd.asarray(x), sd.asarray(f))
    p, res = np.asarray(r.poles()), np.asarray(r.residues())
    true = [(1.5, 1), (-2, -3), (0.5j, 1), (-0.5j, 1)]
    print("exact rational: support points", len(np.asarray(r.support_points)), "poles", len(p))
    for t, rt in true:
        i = np.argmin(abs(p - t))
        print("  pole", t, "err %.2e res err %.2e" % (abs(p[i] - t), abs(res[i] - rt) / abs(rt)))

if __name__ == "__main__":
    run(); exact_rational()
