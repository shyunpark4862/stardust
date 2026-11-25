"""kappa(A_P) for the four problems of figures/08/nnls_kkt.py (same seed and draws)."""
import numpy as np, stardust as sd
rng = np.random.default_rng(4)
for (m, n, cond) in ((30, 10, 1e1), (60, 30, 1e3), (100, 50, 1e6), (200, 80, 1e8)):
    U, _ = np.linalg.qr(rng.normal(size=(m, n))); V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    A = (U * s) @ V.T
    b = rng.normal(size=m)
    r = sd.optimize.nnls(sd.asarray(A), sd.asarray(b))
    x = np.array(r.x.tolist()); P = x > 0
    print(m, n, f"{cond:.0e}", "|P|", int(P.sum()), "kappa(A_P)", f"{np.linalg.cond(A[:, P]):.2e}")
