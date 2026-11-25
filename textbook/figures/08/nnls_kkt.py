"""stardust nnls의 KKT 잔차와 활성 집합의 정확성(정확한 유리수로 검증). 복제본으로 실행."""
import numpy as np, stardust as sd
from fractions import Fraction as F
rng = np.random.default_rng(4)
def exact_on_set(A, b, P):
    # P 위의 정규방정식 A_P^T A_P z = A_P^T b를 유리수로 푼다 (Gauss 소거)
    Ap = [[F(float(A[i, j])) for j in P] for i in range(A.shape[0])]
    bb = [F(float(v)) for v in b]
    k = len(P)
    M = [[sum(Ap[i][r] * Ap[i][c] for i in range(len(bb))) for c in range(k)] + [sum(Ap[i][r] * bb[i] for i in range(len(bb)))] for r in range(k)]
    for c in range(k):
        piv = next(r for r in range(c, k) if M[r][c] != 0); M[c], M[piv] = M[piv], M[c]
        for r in range(k):
            if r != c and M[r][c] != 0:
                q = M[r][c] / M[c][c]; M[r] = [a - q * bq for a, bq in zip(M[r], M[c])]
    return [M[r][k] / M[r][r] for r in range(k)]
rows = []
for (m, n, cond) in ((30, 10, 1e1), (60, 30, 1e3), (100, 50, 1e6), (200, 80, 1e8)):
    U, _ = np.linalg.qr(rng.normal(size=(m, n))); V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    s = np.logspace(0, -np.log10(cond), n)
    A = (U * s) @ V.T
    b = rng.normal(size=m)
    r = sd.optimize.nnls(sd.asarray(A), sd.asarray(b))
    x = np.array(r.x.tolist())
    P = [j for j in range(n) if x[j] > 0]
    w = A.T @ (b - A @ x)
    scale = np.linalg.norm(A, 2) * np.linalg.norm(b)
    dual_free = max(abs(w[j]) for j in P) / scale if P else 0.0     # 활성이 아닌(양수) 변수의 기울기
    dual_bound = max([w[j] for j in range(n) if j not in P] + [0.0]) / scale  # 0인 변수의 기울기 양수 부분
    z = exact_on_set(A, b, P)
    # 정확한 해에서 KKT가 성립하는지(같은 활성 집합이 정답인지)
    xs = [F(0)] * n
    for j, v in zip(P, z): xs[j] = v
    rr = [F(float(b[i])) - sum(F(float(A[i, j])) * xs[j] for j in P) for i in range(m)]
    we = [sum(F(float(A[i, j])) * rr[i] for i in range(m)) for j in range(n)]
    kkt_ok = all(v > 0 for v in z) and all(we[j] <= 0 for j in range(n) if j not in P)
    err = max(abs(F(float(x[j])) - xs[j]) for j in range(n)) / max(abs(v) for v in xs)
    print(m, n, f"{cond:.0e}", "iters", int(r.n_iter), "|P|", len(P), "free", f"{dual_free:.1e}", "bound", f"{dual_bound:.1e}", "exact KKT", kkt_ok, "x relerr", f"{float(err):.1e}")
