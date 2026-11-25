"""라소의 쌍대 간격: 순진한 식(P - D)과 stardust의 소거 없는 식을 정확한 유리수 간격과 견준다.
목적 함수는 stardust의 꼴(절편 없음, 표준화 없음): (1/(2n)) ||y - X b||^2 + alpha ||b||_1."""
import numpy as np, stardust as sd, sys
from fractions import Fraction as F
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(11)
n, p = 40, 12
X = rng.normal(size=(n, p))
beta = np.zeros(p); beta[:4] = [3.0, -2.0, 1.5, 1.0]
y = X @ beta + 0.5 * rng.normal(size=n) + 3000.0 * X[:, 5]   # 큰 신호: v^T v가 잔차보다 훨씬 크다
alpha = 0.3
r = sd.regression.lasso(sd.asarray(y), sd.asarray(X), alpha=alpha, intercept=False, tol=1e-12)
bs = np.array(r.params.tolist())
print("params nonzero", int((bs != 0).sum()), "stardust dual_gap", float(r.dual_gap), "converged", r.converged)
om = 1.0 / n
def parts_float(b):
    res = y - X @ b
    R2 = om * float(res @ res)
    xta = om * (X.T @ res)
    dn = float(np.max(np.abs(xta)))
    s = alpha / dn if dn > alpha else 1.0
    l1n = float(np.abs(b).sum())
    vv = om * float(y @ y)
    D = om * float(res @ (y - res))
    naive_P = 0.5 * R2 + alpha * l1n
    naive_D = 0.5 * vv - 0.5 * om * float((y - s * res) @ (y - s * res))
    naive = naive_P - naive_D
    stable = 0.5 * R2 * (1 - s) ** 2 + alpha * l1n - s * D
    return naive, stable, 0.5 * R2 + alpha * l1n
def exact(b):
    Xf = [[F(float(v)) for v in row] for row in X]; yf = [F(float(v)) for v in y]; bf = [F(float(v)) for v in b]
    on = F(1, n); a = F(alpha)
    res = [yf[i] - sum(Xf[i][j] * bf[j] for j in range(p)) for i in range(n)]
    R2 = on * sum(v * v for v in res)
    xta = [on * sum(Xf[i][j] * res[i] for i in range(n)) for j in range(p)]
    dn = max(abs(v) for v in xta)
    s = a / dn if dn > a else F(1)
    Pv = R2 / 2 + a * sum(abs(v) for v in bf)
    Dv = on * sum(v * v for v in yf) / 2 - on * sum((yf[i] - s * res[i]) ** 2 for i in range(n)) / 2
    return Pv - Dv, Pv
g0, P0 = exact(bs)
print("exact gap at stardust's solution", float(g0), "relative to P", float(g0 / P0), "vv/P", float(om * float(y @ y)) / float(P0))
supp = bs != 0
d = rng.normal(size=p) * supp
with open(f"{OUT}/lasso_gap.dat", "w") as o:
    o.write("eps exact naive stable\n")
    for k in range(0, 33):
        eps = 10.0 ** (-k / 2)
        b = bs + eps * d
        ge, Pe = exact(b)
        nv, st, Pf = parts_float(b)
        o.write(f"{eps:.3e} {float(ge/Pe):.6e} {abs(nv)/Pf:.6e} {abs(st)/Pf:.6e}\n")
        if k % 4 == 0: print(f"eps {eps:.1e} exact {float(ge/Pe):.3e} naive {nv/Pf:.3e} stable {st/Pf:.3e}")
