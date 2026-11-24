"""Section 48.10: the normwise backward error of LU solves in units of the residual contract
||A x - b||_inf / (n eps (||A||_inf ||x||_inf + ||b||_inf)), random Gaussian A and b."""
import numpy as np
from scipy import linalg
rng = np.random.default_rng(4810); eps = np.finfo(float).eps
print("n reps median max")
for n, reps in [(10, 2000), (100, 300), (1000, 20)]:
    v = []
    for _ in range(reps):
        a = rng.standard_normal((n, n)); b = rng.standard_normal(n)
        x = linalg.solve(a, b)
        inf = lambda m: np.abs(m).sum(axis=-1).max() if m.ndim == 2 else np.abs(m).max()
        v.append(inf(a @ x - b) / (n * eps * (inf(a) * inf(x) + inf(b))))
    print(n, reps, f"{np.median(v):.4f}", f"{np.max(v):.4f}")
