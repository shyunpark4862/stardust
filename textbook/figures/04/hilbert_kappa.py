"""kappa_2 of the binary64 Hilbert matrices without trusting a computed sigma_min: the exact
rational inverse of the stored matrix, rounded to double, has its largest singular value
1/sigma_min accurate to a few u (largest singular values are well conditioned); kappa_2 =
sigma_max(H) sigma_max(H^-1). Rewrites the kappa column of hilbert.dat (numpy needed)."""
from fractions import Fraction as F
import numpy as np


def exact_inverse(A):
    k = len(A)
    M = [[F(A[i][j]) for j in range(k)] + [F(int(i == j)) for j in range(k)] for i in range(k)]
    for c in range(k):
        p = next(r for r in range(c, k) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        piv = M[c][c]
        M[c] = [a / piv for a in M[c]]
        for r in range(k):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [a - f * b for a, b in zip(M[r], M[c])]
    return [[float(M[i][k + j]) for j in range(k)] for i in range(k)]


rows = [l.split() for l in open("hilbert.dat").read().split("\n")[1:] if l.strip()]
out = ["n kappa err"]
for n, _, err in rows:
    n = int(n)
    H = np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])
    Hi = np.array(exact_inverse(H.tolist()))
    kap = np.linalg.svd(H, compute_uv=False)[0] * np.linalg.svd(Hi, compute_uv=False)[0]
    out.append(f"{n} {kap:.4e} {err}")
    print(n, f"{kap:.3e}")
open("hilbert.dat", "w").write("\n".join(out) + "\n")
