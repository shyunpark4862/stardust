"""Gram (discrete orthogonal) polynomials on s_i = i - (N-1)/2, i < N: the degree recurrence
p_{j+1}(s) = s p_j(s) - beta_j p_{j-1}(s) evaluated in binary64 at the window's end s_0 against exact rationals
(relative error over j), and the Savitzky-Golay smoothing coefficients h_i = sum_{j<=d} p_j(s_i) p_j(0)/||p_j||^2
from stardust's savgol_coeffs (which runs the recurrence along i in double-double) against the exact ones and
against the same sum with the degree recurrence in binary64. Output: gram.dat."""
from fractions import Fraction as F
import numpy as np
import stardust as sd

def exact_gram(N, d):
    s = [F(2 * i - (N - 1), 2) for i in range(N)]
    P = [[F(1)] * N, list(s)]
    norms = [F(N), sum(v * v for v in s)]
    betas = [None, norms[1] / norms[0]]
    for j in range(1, d):
        b = norms[j] / norms[j - 1]
        nxt = [s[i] * P[j][i] - b * P[j - 1][i] for i in range(N)]
        P.append(nxt); norms.append(sum(v * v for v in nxt))
    return s, P, norms

def main(N=51, d=45):
    s, P, norms = exact_gram(N, d)
    beta = [None] + [float(norms[j] / norms[j - 1]) for j in range(1, d + 1)]
    rows = []
    # degree recurrence at s_0 and at the centre, binary64, normalised by the exact norm
    for where in (0, (N - 1) // 2):
        x = float(s[where]); p0, p1 = 1.0, x
        vals = [p0, p1]
        for j in range(1, d):
            p0, p1 = p1, x * p1 - beta[j] * p0; vals.append(p1)
        rows.append([abs(F(vals[j]) - P[j][where]) / (abs(P[j][where]) if P[j][where] != 0 else 1) for j in range(d + 1)])
    with open("gram.dat", "w") as f:
        f.write("j end centre\n")
        for j in range(d + 1):
            f.write("%d %.4e %.4e\n" % (j, max(float(rows[0][j]), 1e-17), max(float(rows[1][j]), 1e-17)))
    jj = next(j for j in range(d + 1) if rows[0][j] > 1e-8)
    print("N=%d: degree recurrence at the window end passes 1e-8 relative error at j=%d (N/4=%.1f); at j=%d: %.1e" % (N, jj, N / 4, d, float(rows[0][d])))
    # savgol coefficients, centre, deriv 0
    for dd in (10, 20, 30, 40):
        hx = [sum(P[j][i] * P[j][(N - 1) // 2] / norms[j] for j in range(dd + 1)) for i in range(N)]
        lib = np.asarray(sd.smooth.savgol_coeffs(N, dd, use='dot'))
        el = max(abs(F(float(lib[i])) - hx[i]) for i in range(N)) / max(abs(v) for v in hx)
        # naive: the degree recurrence in binary64 at every point, the norms exact-rounded
        nv = []
        for i in range(N):
            x = float(s[i]); p0, p1 = 1.0, x; vals = [p0, p1]
            for j in range(1, dd): p0, p1 = p1, x * p1 - beta[j] * p0; vals.append(p1)
            nv.append(vals)
        cen = nv[(N - 1) // 2]
        hn = [sum(nv[i][j] * cen[j] / float(norms[j]) for j in range(dd + 1)) for i in range(N)]
        en = max(abs(F(hn[i]) - hx[i]) for i in range(N)) / max(abs(v) for v in hx)
        print("savgol N=%d d=%d: stardust max err/max|h| %.1e, binary64 degree recurrence %.1e" % (N, dd, float(el), float(en)))

if __name__ == "__main__":
    main()
