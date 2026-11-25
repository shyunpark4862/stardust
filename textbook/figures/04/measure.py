"""Chapter 4 measurements on stardust itself (a clone of the public repository built with
`uv pip install .`; run with that environment's python, which has numpy 2.5.5). Exact references
in Python Fractions/integers. Writes norm_scale.dat, svd_err.dat, hilbert.dat, jordan.dat and
prints the tables' numbers. Seeds fixed."""
from fractions import Fraction as F
import math
import time
import numpy as np
import stardust as sd

rng = np.random.default_rng(4)


def exact_norm(x):  # sqrt(sum x_i^2) correctly rounded to double (x a float64 array)
    S = sum(F(float(v)) ** 2 for v in x)
    if S == 0:
        return 0.0
    n, d = S.numerator, S.denominator
    K = 200 + max(0, (d.bit_length() - n.bit_length()) // 2)
    r = math.isqrt(n * 4**K // d)
    return float(F(r, 2**K))


def rel(v, r):
    if not math.isfinite(v) or r == 0:
        return 1.0 if v != r else 0.0
    return max(abs(v - r) / r, 1e-18)


# ---- 1. 2-norm across scales
v = rng.standard_normal(1000)
with open("norm_scale.dat", "w") as f:
    f.write("k naive numpy stardust\n")
    for k in range(-1070, 1011, 10):  # the true norm stays below the largest double
        x = np.ldexp(v, k)
        r = exact_norm(x)
        naive = float(np.sqrt(np.sum(x * x)))
        npn = float(np.linalg.norm(x))
        sdn = float(sd.linalg.vector_norm(sd.asarray(x.tolist())))
        f.write(f"{k} {rel(naive, r):.3e} {rel(npn, r):.3e} {rel(sdn, r):.3e}\n")
print("norm: written norm_scale.dat")
xx = np.array([3e200, 4e200]); print("numpy norm(3e200,4e200) =", np.linalg.norm(xx), " stardust =", sd.linalg.vector_norm(sd.asarray([3e200, 4e200])))
yy = np.array([3e-200, 4e-200]); print("numpy norm(3e-200,4e-200) =", np.linalg.norm(yy), " stardust =", sd.linalg.vector_norm(sd.asarray([3e-200, 4e-200])))

# ---- 2. matrix_norm by layout
n = 4096
A = rng.standard_normal((n, n))
As = sd.asarray(A)
def best(f, reps=5):
    t = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); f(); t = min(t, time.perf_counter() - t0)
    return t * 1e3
for ordv in (1, float("inf"), "fro"):
    tc = best(lambda: sd.linalg.matrix_norm(As, ord=ordv))
    tf = best(lambda: sd.linalg.matrix_norm(As.mT, ord=ordv))
    tn = best(lambda: np.linalg.norm(A, ord=ordv))
    print(f"matrix_norm ord={ordv}: stardust C {tc:.2f} ms, transposed view {tf:.2f} ms, numpy {tn:.2f} ms")

# ---- 3. singular values with prescribed values (rational orthogonal Householder factors)
m = 40
def householder(vec):
    vv = sum(F(a) * a for a in vec)
    return [[F(int(i == j)) - 2 * F(vec[i]) * vec[j] / vv for j in range(m)] for i in range(m)]
def matmul(X, Y):
    return [[sum(X[i][k] * Y[k][j] for k in range(m)) for j in range(m)] for i in range(m)]
v1 = [int(a) for a in rng.integers(-9, 10, m)]; v2 = [int(a) for a in rng.integers(-9, 10, m)]
ks = [round(52 * i / (m - 1)) for i in range(m)]
sig = [F(1, 2**k) for k in ks]
Q1, Q2 = householder(v1), householder(v2)
D = [[sig[i] if i == j else F(0) for j in range(m)] for i in range(m)]
Aex = matmul(matmul(Q1, D), Q2)
Ad = [[float(a) for a in row] for row in Aex]
sv = sd.linalg.svdvals(sd.asarray(Ad)).tolist()
with open("svd_err.dat", "w") as f:
    f.write("i sigma relerr bound\n")
    for i in range(m):
        s = float(sig[i])
        f.write(f"{i+1} {s:.6e} {max(abs(sv[i]-s)/s,1e-18):.3e} {2**-53 * float(sig[0]) / s:.3e}\n")
print("svd: written svd_err.dat; smallest sigma", float(sig[-1]), "computed", sv[-1])

# ---- 4. Hilbert systems
def solve_exact(Hm, b):
    k = len(b)
    M = [[F(Hm[i][j]) for j in range(k)] + [F(b[i])] for i in range(k)]
    for c in range(k):
        p = next(r for r in range(c, k) if M[r][c] != 0)
        M[c], M[p] = M[p], M[c]
        for r in range(k):
            if r != c and M[r][c] != 0:
                fct = M[r][c] / M[c][c]
                M[r] = [a - fct * bb for a, bb in zip(M[r], M[c])]
    return [M[i][k] / M[i][i] for i in range(k)]
with open("hilbert.dat", "w") as f:
    f.write("n kappa err\n")
    for nn in range(2, 14):
        H = np.array([[1.0 / (i + j + 1) for j in range(nn)] for i in range(nn)])
        b = H @ np.ones(nn)
        xe = solve_exact(H.tolist(), b.tolist())
        xs = sd.linalg.solve(sd.asarray(H), sd.asarray(b)).tolist()
        err = math.sqrt(sum(float((F(a) - e) ** 2) for a, e in zip(xs, xe))) / math.sqrt(sum(float(e * e) for e in xe))
        kap = float(sd.linalg.cond(sd.asarray(H), ord=2))  # hilbert_kappa.py rewrites this column with an exact-inverse kappa
        f.write(f"{nn} {kap:.4e} {max(err,1e-18):.4e}\n")
        print(f"hilbert n={nn}: kappa {kap:.2e}, error {err:.2e}")

# ---- 5. Jordan block
nj, eps = 16, 1e-16
J = np.diag(np.ones(nj - 1), 1)
Jp = J.copy(); Jp[nj - 1, 0] = eps
ev = sd.linalg.eigvals(sd.asarray(Jp)).tolist()
ev0 = sd.linalg.eigvals(sd.asarray(J)).tolist()
with open("jordan.dat", "w") as f:
    f.write("re im re0 im0\n")
    for a, b0 in zip(ev, ev0):
        f.write(f"{a.real:.6e} {a.imag:.6e} {b0.real:.6e} {b0.imag:.6e}\n")
print("jordan: |eigs| of perturbed", min(abs(a) for a in ev), max(abs(a) for a in ev), "eps^(1/16) =", eps ** (1 / 16))
print("jordan: max |eig| unperturbed", max(abs(a) for a in ev0))

# ---- 6. Sylvester vs Kronecker
for mm in (20, 40):
    A_ = rng.standard_normal((mm, mm)) + mm * np.eye(mm)
    B_ = rng.standard_normal((mm, mm)) + mm * np.eye(mm)
    C_ = rng.standard_normal((mm, mm))
    t1 = best(lambda: sd.linalg.solve_sylvester(sd.asarray(A_), sd.asarray(B_), sd.asarray(C_)), 20)
    X1 = np.array(sd.linalg.solve_sylvester(sd.asarray(A_), sd.asarray(B_), sd.asarray(C_)).tolist())
    K = np.kron(np.eye(mm), A_) + np.kron(B_.T, np.eye(mm))
    t2 = best(lambda: sd.linalg.solve(sd.asarray(K), sd.asarray(C_.reshape(-1, order="F"))), 5)
    X2 = np.array(sd.linalg.solve(sd.asarray(K), sd.asarray(C_.reshape(-1, order="F"))).tolist()).reshape(mm, mm, order="F")
    r1 = np.linalg.norm(A_ @ X1 + X1 @ B_ - C_) / np.linalg.norm(C_)
    r2 = np.linalg.norm(A_ @ X2 + X2 @ B_ - C_) / np.linalg.norm(C_)
    print(f"sylvester m={mm}: Bartels-Stewart {t1:.3f} ms residual {r1:.1e}; Kronecker solve {t2:.3f} ms residual {r2:.1e}")

# ---- 7. kronsum: NaN and the identity products
Am = rng.standard_normal((4, 4)); Bm = rng.standard_normal((3, 3)); Am[1, 2] = np.nan; Bm[0, 1] = np.inf
ks_ = np.array(sd.linalg.kronsum(sd.asarray(Am), sd.asarray(Bm)).tolist())
sp = np.array(sd.sparse.kronsum(sd.asarray(Am), sd.asarray(Bm)).toarray().tolist()) if hasattr(sd.sparse.kronsum(sd.asarray(Am), sd.asarray(Bm)), "toarray") else None
viaprod = np.kron(np.eye(3), Am) + np.kron(Bm, np.eye(4))
print("kronsum: NaN entries dense", int(np.isnan(ks_).sum()), "inf entries", int(np.isinf(ks_).sum()), "; via kron products: NaN", int(np.isnan(viaprod).sum()), "inf", int(np.isinf(viaprod).sum()))
if sp is not None:
    same = np.array_equal(ks_.view(np.int64), sp.view(np.int64))
    print("kronsum dense vs sparse bit-identical:", same)
