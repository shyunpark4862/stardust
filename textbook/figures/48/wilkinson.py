"""Proposition 48.4: growth 2^(n-1) for W_n with partial pivoting (scipy getrf)."""
import numpy as np
from scipy import linalg
for n in (10, 30, 53, 60):
    w = np.tril(-np.ones((n, n)), -1) + np.eye(n); w[:, -1] = 1
    p, l, u = linalg.lu(w)
    print(n, np.abs(u).max(), 2.0 ** (n - 1), np.allclose(p, np.eye(n)))
