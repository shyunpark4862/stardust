"""Multivariate normal logpdf with an ill-conditioned covariance (Section 4.8): stardust's
MultivariateNormal(mu, cov).logpdf (double-double Cholesky, stats/dd_linalg.hpp) against a
float64 Cholesky computation (numpy), both against the exact value: the quadratic form
x^T cov^-1 x and det(cov) in exact rationals (cov = the float64 Hilbert matrix), then
-(d log 2pi + log det + Q)/2 with the logs of big integers (math.log on ints is accurate to a
double). Run with the stardust clone's python. Writes mvn.dat."""
from fractions import Fraction as F
import math
import numpy as np
import stardust as sd


def exact_solve_det(A, b):
    k = len(b)
    M = [[F(A[i][j]) for j in range(k)] + [F(b[i])] for i in range(k)]
    det = F(1)
    for c in range(k):
        p = next(r for r in range(c, k) if M[r][c] != 0)
        if p != c:
            M[c], M[p] = M[p], M[c]; det = -det
        det *= M[c][c]
        for r in range(c + 1, k):
            if M[r][c] != 0:
                f = M[r][c] / M[c][c]
                M[r] = [a - f * bb for a, bb in zip(M[r], M[c])]
    x = [F(0)] * k
    for i in range(k - 1, -1, -1):
        x[i] = (M[i][k] - sum(M[i][j] * x[j] for j in range(i + 1, k))) / M[i][i]
    return x, det


def flog(q):  # log of a positive Fraction
    return math.log(q.numerator) - math.log(q.denominator)


from hilbert_kappa_lib import kappa2  # sigma_max(H) sigma_max(exact H^-1)

rng = np.random.default_rng(8)
rows = []
for d in range(2, 12):
    H = np.array([[1.0 / (i + j + 1) for j in range(d)] for i in range(d)])
    x = rng.standard_normal(d)
    sol, det = exact_solve_det(H.tolist(), x.tolist())
    Q = sum(F(a) * s for a, s in zip(x.tolist(), sol))
    exact = -(d * math.log(2 * math.pi) + flog(det) + float(Q)) / 2
    # a float64 Cholesky route
    L = np.linalg.cholesky(H)
    z = np.linalg.solve(L, x)
    f64 = -(d * math.log(2 * math.pi) + 2 * np.sum(np.log(np.diag(L))) + z @ z) / 2
    lib = float(sd.stats.MultivariateNormal(mu=sd.asarray([0.0] * d), cov=sd.asarray(H)).logpdf(sd.asarray(x)))
    kap = kappa2(H)
    rows.append((d, kap, abs(f64 - exact) / abs(exact), max(abs(lib - exact) / abs(exact), 1e-18)))
    print(f"d={d} kappa={kap:.1e} float64 rel err {rows[-1][2]:.1e}, stardust {rows[-1][3]:.1e}, Q={float(Q):.3e}")
with open("mvn.dat", "w") as f:
    f.write("d kappa f64 lib\n" + "".join(f"{a} {b:.4e} {c:.4e} {e:.4e}\n" for a, b, c, e in rows))
