import numpy as np, stardust as sd
rng = np.random.default_rng(7)
n = 50
Q, _ = np.linalg.qr(rng.normal(size=(n, n)))
H = (Q * np.linspace(1, 100, n)) @ Q.T
c = rng.normal(size=n) * 20
lo, hi = -np.ones(n), np.ones(n)
f = lambda v: float(0.5 * np.asarray(v) @ H @ np.asarray(v) - c @ np.asarray(v))
g = lambda v: sd.asarray(H @ np.asarray(v) - c)
for ftol in (None, 0.0):
  for pg in (1e-5, 1e-8, 1e-11):
    opts = {"gtol": pg, "max_iter": 10000}
    if ftol is not None: opts["ftol"] = ftol
    r = sd.optimize.minimize(f, sd.asarray(np.zeros(n)), method="l-bfgs-b", jac=g, bounds=(sd.asarray(lo), sd.asarray(hi)), options=opts)
    x = np.array(r.x.tolist()); gr = H @ x - c
    pgrad = x - np.clip(x - gr, lo, hi)
    free = (x > lo) & (x < hi)
    Hf = H[np.ix_(free, free)]; ev = np.linalg.eigvalsh(Hf)
    print(f"ftol {ftol} gtol {pg:.0e}: {r.message} conv {r.converged} it {int(r.n_iter)} pg {np.max(np.abs(pgrad)):.2e} f {float(r.fun):.6g} free-eig [{ev[0]:.1f},{ev[-1]:.1f}] sqrt(2u|f|lmax) {np.sqrt(2*2**-53*abs(float(r.fun))*ev[-1]):.1e}")
