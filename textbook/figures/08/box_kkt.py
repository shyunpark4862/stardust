"""상자 제약의 KKT: L-BFGS-B의 사영 기울기 노름과 lsq_linear의 optimality를 직접 계산해 견준다(복제본)."""
import numpy as np, stardust as sd
rng = np.random.default_rng(7)
n = 50
Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
H = (Q * np.linspace(1, 100, n)) @ Q.T
c = rng.normal(size=n) * 20
lo, hi = -np.ones(n), np.ones(n)
f = lambda v: float(0.5 * np.asarray(v) @ H @ np.asarray(v) - c @ np.asarray(v))
g = lambda v: sd.asarray(H @ np.asarray(v) - c)
for pg in (1e-5, 1e-8, 1e-11):
    r = sd.optimize.minimize(f, sd.asarray(np.zeros(n)), method="l-bfgs-b", jac=g, bounds=(sd.asarray(lo), sd.asarray(hi)),
                             options={"gtol": pg, "ftol": 0.0, "max_iter": 10000})
    x = np.array(r.x.tolist()); gr = H @ x - c
    pgrad = x - np.clip(x - gr, lo, hi)
    act = int(((x == lo) | (x == hi)).sum())
    print(f"l-bfgs-b gtol {pg:.0e}: {r.message!s} status {r.status} n_iter {int(r.n_iter)} active {act} |P(x-g)-x|_inf {np.max(np.abs(pgrad)):.2e}")
# lsq_linear: 0 <= x <= 1
m = 80
A = rng.normal(size=(m, n)); b = rng.normal(size=m) * 3
for meth in ("trf", "bvls"):
    r = sd.optimize.lsq_linear(sd.asarray(A), sd.asarray(b), bounds=(0.0, 1.0), method=meth)
    x = np.array(r.x.tolist()); gr = A.T @ (A @ x - b)
    pgrad = x - np.clip(x - gr, 0.0, 1.0)
    print(f"lsq_linear {meth}: {r.message!s} status {r.status} n_iter {int(r.n_iter)} optimality {float(r.optimality):.2e} |P(x-g)-x|_inf {np.max(np.abs(pgrad)):.2e} active {int(np.abs(np.array(r.active_mask.tolist())).sum())}")
