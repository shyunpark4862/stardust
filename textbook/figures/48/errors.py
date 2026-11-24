"""Figure 48.3: forward and backward errors of LU solves against kappa_2(A), n = 100.
A = U diag(s) V^T with Haar U, V and s geometric from 1 to 1/kappa; x random; b = A x.
Backward error by Rigal-Gaches in the 2-norm.
Run: uv run --no-project --with numpy --with scipy python errors.py > errors.dat"""
import numpy as np
from scipy import linalg

rng = np.random.default_rng(4801)
n = 100
def haar(n):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))
print("kappa fwd bwd")
for e in np.linspace(0, 16, 33):
    kappa = 10.0 ** e
    for _ in range(3):
        s = kappa ** (-np.linspace(0, 1, n))
        a = haar(n) @ np.diag(s) @ haar(n).T
        x = rng.standard_normal(n)
        b = a @ x
        xh = linalg.solve(a, b)
        k2 = np.linalg.cond(a)
        r = b - a @ xh
        bwd = np.linalg.norm(r) / (np.linalg.norm(a, 2) * np.linalg.norm(xh) + np.linalg.norm(b))
        fwd = np.linalg.norm(xh - x) / np.linalg.norm(x)
        print(f"{k2:.4e} {max(fwd,1e-18):.4e} {max(bwd,1e-18):.4e}")
