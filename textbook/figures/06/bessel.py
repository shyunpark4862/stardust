"""Directions of the Bessel recurrence y_{n+1} = (2n/x) y_n - y_{n-1}.
(1) forward from exact J_0(1), J_1(1) in binary64, Miller from N = 30 normalised by J_0 + 2 sum J_2k = 1, and
    stardust's jv(n, 1): relative errors for n = 0..25 (bessel_dir.dat).
(2) |J_n(10)|, |Y_n(10)| for n = 0..40 (stardust) (bessel_mag.dat).
(3) Miller with Neumann's sum for nu = mu + n, mu = 0.3: error of f_0 against N - top (miller_n.dat).
(4) stardust jv on a grid against the Decimal series; Wronskian residual x (J_{nu+1} Y_nu - J_nu Y_{nu+1}) pi/2 - 1."""
import math
import numpy as np
import stardust as sd
from dref import jv as jref, dec, D, gamma as gref

def lib_jv(nu, x):
    return float(np.asarray(sd.special.jv(sd.asarray(np.array([float(nu)])), sd.asarray(np.array([float(x)]))))[0])

def lib_yv(nu, x):
    return float(np.asarray(sd.special.yv(sd.asarray(np.array([float(nu)])), sd.asarray(np.array([float(x)]))))[0])

def direction():
    x = 1.0
    ex = [jref(n, x) for n in range(0, 27)]
    f = [float(ex[0]), float(ex[1])]
    for n in range(1, 26): f.append(2 * n / x * f[n] - f[n - 1])
    N = 30
    g = [0.0] * (N + 2); g[N + 1] = 0.0; g[N] = 1.0
    for k in range(N, 0, -1): g[k - 1] = 2 * k / x * g[k] - g[k + 1]
    norm = g[0] + 2 * sum(g[2 * k] for k in range(1, N // 2 + 1))
    m = [v / norm for v in g]
    rows = []
    for n in range(0, 26):
        e = lambda v: max(float(abs(dec(v) - ex[n]) / abs(ex[n])), 1e-17)
        rows.append((n, min(e(f[n]), 1e3), e(m[n]), e(lib_jv(n, x))))
    with open("bessel_dir.dat", "w") as fo:
        fo.write("n fwd miller lib\n")
        for r in rows: fo.write("%d %.4e %.4e %.4e\n" % r)
    for r in rows[::5]: print("n %2d fwd %.2e miller %.2e lib %.2e" % r)
    print("J_20(1) exact %.3e forward %.3e" % (float(ex[20]), f[20]))

def magnitude():
    with open("bessel_mag.dat", "w") as fo:
        fo.write("n J Y\n")
        for n in range(0, 41):
            fo.write("%d %.6e %.6e\n" % (n, abs(lib_jv(n, 10.0)), abs(lib_yv(n, 10.0))))

def miller_neumann(mu, x, n, N):
    """f_n/(normalised) by the library's scheme with a chosen start N (even), float64 without dd coefficients."""
    fk1, fk = 0.0, 1.0; vals = {N: 1.0, N + 1: 0.0}
    for k in range(N, 0, -1):
        f = 2 * (mu + k) / x * fk - fk1; fk1, fk = fk, f; vals[k - 1] = f
        if abs(fk) > 2.0 ** 600:
            fk *= 2.0 ** -600; fk1 *= 2.0 ** -600
            for kk in vals: vals[kk] *= 2.0 ** -600
    # Neumann: (x/2)^mu / Gamma(mu+1) = J_mu + sum_k (mu + 2k) (mu+1)_{k-1} / k! J_{mu+2k}
    s = vals[0]; coef = 1.0
    for k in range(1, N // 2 + 1):
        if k > 1: coef *= (mu + k - 1) / k
        else: coef = 1.0
        s += (mu + 2 * k) * coef * vals[2 * k]
    lhs = (x / 2) ** mu / math.gamma(mu + 1)
    return vals[n] * lhs / s

def miller_start():
    rows = []
    mu = 0.3
    for x, n in ((10.0, 0), (10.0, 20), (50.0, 0)):
        top = max(n + mu, x)
        ex = jref(dec(mu) + n, x)
        for d in range(0, 60, 2):
            N = int(top) + d; N += N % 2
            v = miller_neumann(mu, x, n, N)
            rows.append((x, n, N - top, max(float(abs(dec(v) - ex) / abs(ex)), 1e-17)))
        lib_d = 10 * top ** (1 / 3) + 8
        print("x %g n %d: library's N - top = %.1f; 7 top^(1/3) + 3 = %.1f" % (x, n, lib_d, 7 * top ** (1 / 3) + 3))
    with open("miller_n.dat", "w") as fo:
        fo.write("x n d err\n")
        for r in rows: fo.write("%g %d %.3f %.4e\n" % r)
    for x, n in ((10.0, 0), (10.0, 20)):
        with open("miller_n_x%d_n%d.dat" % (x, n), "w") as fo:
            fo.write("d err\n")
            for r in rows:
                if r[0] == x and r[1] == n: fo.write("%.3f %.4e\n" % (r[2], r[3]))
    for r in rows:
        if r[3] < 1e-15: print("first d with err < 1e-15:", r); break

def accuracy():
    rng = np.random.default_rng(4)
    worst = 0; wr = 0; cnt = 0
    for _ in range(300):
        nu = round(float(rng.uniform(0, 40)) * 1024) / 1024; x = float(rng.uniform(0.5, 30))  # nu + 1 exact
        ex = jref(nu, x)
        if abs(ex) < D(10) ** -250: continue
        v = lib_jv(nu, x)
        e = float(abs(dec(v) - ex) / abs(ex)); worst = max(worst, e)
        env = math.hypot(float(ex), lib_yv(nu, x)); wenv = max(globals().get("wenv", 0), float(abs(dec(v) - ex)) / env); globals()["wenv"] = wenv
        # Wronskian x (J_{nu+1} Y_nu - J_nu Y_{nu+1}) = 2/pi
        a, b, c, d = lib_jv(nu + 1, x), lib_yv(nu, x), lib_jv(nu, x), lib_yv(nu + 1, x)
        w = x * (a * b - c * d) * math.pi / 2 - 1
        scale = x * (abs(a * b) + abs(c * d)) * math.pi / 2
        wr = max(wr, abs(w) / max(scale, 1.0)); cnt += 1
    print("jv vs Decimal series, %d points: max rel %.2e, max |err|/sqrt(J^2+Y^2) %.2e; Wronskian residual / scale max %.2e" % (cnt, worst, globals()["wenv"], wr))

if __name__ == "__main__":
    direction(); magnitude(); miller_start(); accuracy()
