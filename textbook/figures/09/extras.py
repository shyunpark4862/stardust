"""9장 보충 그림 자료: Borel-Cantelli, 꼬리 부등식, Irwin-Hall 합성곱, 특성함수(경험 특성함수는 복제본 표본)."""
import math, sys, numpy as np
OUT = sys.argv[1] if len(sys.argv) > 1 else "."
# 1. Borel-Cantelli: 독립 사건 A_n = {U_n <= p_n}, 누적 개수
try:
    import stardust as sd
    rng = sd.random.default_rng(31)
    N = 10**6
    u = np.array(rng.random((N,)).tolist())
except ImportError:
    sys.exit("needs stardust")
n = np.arange(1, N + 1)
c1 = np.cumsum(u <= 1.0 / n); c2 = np.cumsum(u <= 1.0 / n**2)
idx = np.unique(np.round(np.logspace(0, 6, 300)).astype(int))
with open(f"{OUT}/bc.dat", "w") as o:
    o.write("n inv inv2 logn\n")
    for k in idx: o.write(f"{k} {c1[k-1]} {c2[k-1]} {math.log(k)+0.5772:.4f}\n")
print("BC final", c1[-1], c2[-1], math.log(N) + 0.5772)
# 2. 꼬리 부등식: 정규 P(|Z|>=a), Chebyshev 1/a^2; 지수 P(X>=a), Markov 1/a
with open(f"{OUT}/tailbounds.dat", "w") as o:
    o.write("a normal cheb expo markov\n")
    for i in range(1, 81):
        a = i / 10
        o.write(f"{a} {math.erfc(a/math.sqrt(2)):.6e} {min(1,1/a**2):.6e} {math.exp(-a):.6e} {min(1,1/a):.6e}\n")
# 3. Irwin-Hall: 균등 n개 합의 표준화 밀도
def ih(x, k):
    return sum((-1)**j * math.comb(k, j) * max(x - j, 0)**(k - 1) for j in range(k + 1)) / math.factorial(k - 1)
with open(f"{OUT}/irwin.dat", "w") as o:
    o.write("z n1 n2 n3 n6\n")
    for i in range(-400, 401):
        z = i / 100; row = []
        for k in (1, 2, 3, 6):
            mu, sd_ = k / 2, math.sqrt(k / 12)
            x = mu + z * sd_
            row.append(ih(x, k) * sd_ if 0 < x < k else 0.0)
        o.write(f"{z} " + " ".join(f"{v:.6e}" for v in row) + "\n")
# 4. 특성함수: 정규, Cauchy, 균등(-1,1), 경험 특성함수(정규 표본 200개)
x = np.array(sd.stats.Normal().sample((200,), rng=rng).tolist())
with open(f"{OUT}/cf.dat", "w") as o:
    o.write("t normal cauchy unif emp\n")
    for i in range(0, 301):
        t = i / 30
        emp = abs(np.mean(np.exp(1j * t * x)))
        o.write(f"{t:.4f} {math.exp(-t*t/2):.6e} {math.exp(-t):.6e} {(math.sin(t)/t if t else 1.0):.6e} {emp:.6e}\n")
