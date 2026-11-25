"""The lasso gap as stardust's covariance (Gram) path computes it (p <= kCovMaxCols = 1024):
R2 = vv - 2 b.c + b.Gb, xta = c - Gb, D = b.xta, then eq. gap-stable; against the exact gap.
Same problem as figures/08/lasso_gap.py."""
import numpy as np, stardust as sd
from fractions import Fraction as F
rng = np.random.default_rng(11)
n, p = 40, 12
X = rng.normal(size=(n, p))
beta = np.zeros(p); beta[:4] = [3.0, -2.0, 1.5, 1.0]
y = X @ beta + 0.5 * rng.normal(size=n) + 3000.0 * X[:, 5]
alpha = 0.3
r = sd.regression.lasso(sd.asarray(y), sd.asarray(X), alpha=alpha, intercept=False, tol=1e-12)
bs = np.array(r.params.tolist())
om = 1.0 / n
G = om * (X.T @ X); c = om * (X.T @ y); vv = om * float(y @ y)
def gram_gap(b):
    Gb = G @ b
    R2 = max(0.0, vv - 2.0 * float(b @ c) + float(b @ Gb))
    xta = c - Gb
    D = float(b @ xta)
    dn = float(np.max(np.abs(xta)))
    s = alpha / dn if dn > alpha else 1.0
    return 0.5 * R2 * (1 - s) ** 2 + alpha * float(np.abs(b).sum()) - s * D, 0.5 * R2 + alpha * float(np.abs(b).sum()), R2
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
    return Pv - Dv, Pv, R2
supp = bs != 0
d = rng.normal(size=p) * supp
print("stardust dual_gap", float(r.dual_gap))
for k in range(0, 33, 2):
    eps = 10.0 ** (-k / 2)
    b = bs + eps * d
    ge, Pe, R2e = exact(b)
    gg, Pf, R2f = gram_gap(b)
    print(f"eps {eps:.1e} exact {float(ge/Pe):.3e} gram {gg/Pf:.3e}  R2 relerr {abs(R2f-float(R2e))/float(R2e):.1e}")
