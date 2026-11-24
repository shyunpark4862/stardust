"""Figure 48.2: growth factor of partial pivoting on random Gaussian matrices.
rho_n = max over all stages k of |a_ij^(k)| / max |a_ij|, by our own right-looking elimination;
also the same maximum divided by the entries' standard deviation (1), as Trefethen and Schreiber normalize.
Run: uv run --no-project --with numpy python growth.py > growth.dat"""
import numpy as np

def growth(a):
    a = a.copy(); n = a.shape[0]; m0 = np.abs(a).max(); mx = m0
    for k in range(n - 1):
        p = k + np.argmax(np.abs(a[k:, k]))
        if p != k: a[[k, p]] = a[[p, k]]
        l = a[k + 1:, k] / a[k, k]
        a[k + 1:, k + 1:] -= np.outer(l, a[k, k + 1:])
        a[k + 1:, k] = 0
        mx = max(mx, np.abs(a[k + 1:, k + 1:]).max())
    return mx / m0, mx

rng = np.random.default_rng(48)
print("n median max sigmed reps")
for n, reps in [(10, 2000), (20, 1000), (50, 400), (100, 200), (200, 60), (500, 40), (1000, 12)]:
    g = np.array([growth(rng.standard_normal((n, n))) for _ in range(reps)])
    print(n, f"{np.median(g[:,0]):.3f}", f"{np.max(g[:,0]):.3f}", f"{np.median(g[:,1]):.3f}", reps, flush=True)
