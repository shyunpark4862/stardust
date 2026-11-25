"""kappa_2 of a float64 matrix from the largest singular values of H and of its exact rational
inverse rounded to double (no computed sigma_min)."""
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


def kappa2(H):
    Hi = np.array(exact_inverse(np.asarray(H).tolist()))
    return np.linalg.svd(H, compute_uv=False)[0] * np.linalg.svd(Hi, compute_uv=False)[0]
