"""걸음 수 대 조건수: 이차 함수 f = x^T A x / 2 (A 대각, 고유값 1..kappa 고르게, n = 100)에서
f - f* 가 처음의 1e-10배가 될 때까지의 걸음. 경사 하강(걸음 1/L)과 CG는 여기서 짠 것,
BFGS와 L-BFGS-B는 따로 빌드한 stardust (max_iter를 바꿔 가며 처음 닿는 반복을 찾는다)."""
import numpy as np, stardust as sd, sys
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
n = 100
rng = np.random.default_rng(1)
x0 = rng.normal(size=n)
def run(kappa):
    lam = np.linspace(1.0, kappa, n)
    f = lambda x: 0.5 * float(np.dot(lam * x, x))
    f0 = f(x0); tol = 1e-10 * f0
    # 경사 하강, alpha = 1/L
    x = x0.copy(); k = 0
    while f(x) > tol:
        x = x - lam * x / kappa; k += 1
    gd = k
    # CG (x0에서, b = 0)
    x = x0.copy(); r = -lam * x; p = r.copy(); k = 0
    while f(x) > tol:
        Ap = lam * p; a = r @ r / (p @ Ap); x = x + a * p; rn = r - a * Ap
        p = rn + (rn @ rn) / (r @ r) * p; r = rn; k += 1
    cg = k
    res = {}
    for meth in ("bfgs", "l-bfgs-b"):
        fun = lambda v: 0.5 * float(np.dot(lam * np.asarray(v), np.asarray(v)))
        jac = lambda v: sd.asarray(lam * np.asarray(v))
        def reached(m):
            r_ = sd.optimize.minimize(fun, sd.asarray(x0), method=meth, jac=jac,
                                      options={"max_iter": m, "gtol": 0.0} if meth == "bfgs" else {"max_iter": m, "gtol": 0.0, "ftol": 0.0})
            return fun(r_.x) <= tol, int(r_.n_iter)
        lo, hi = 0, 1
        while not reached(hi)[0]:
            lo, hi = hi, hi * 2
            if hi > 200000: break
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if reached(mid)[0]: hi = mid
            else: lo = mid
        res[meth] = hi
    return gd, cg, res["bfgs"], res["l-bfgs-b"]
with open(f"{OUT}/kappa.dat", "w") as o:
    o.write("kappa gd cg bfgs lbfgsb bound\n")
    for kappa in (3, 10, 30, 100, 300, 1000, 3000, 10000):
        gd, cg, b, lb = run(float(kappa))
        bound = np.log(1e10 * 1.0) / -np.log(1 - 1 / kappa) if kappa > 1 else 0
        print(kappa, gd, cg, b, lb, round(bound))
        o.write(f"{kappa} {gd} {cg} {b} {lb} {bound:.1f}\n")
