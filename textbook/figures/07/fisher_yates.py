"""Fisher--Yates 균등성의 카이제곱 시험(따로 빌드한 stardust로; 참값은 균등분포)."""
import math, itertools, numpy as np, stardust as sd

def chi2_sf(x, k):  # Wilson--Hilferty 근사로 카이제곱 위꼬리
    z = ((x / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return 0.5 * math.erfc(z / math.sqrt(2))

def run(label, draw, cells, M):
    idx = {c: i for i, c in enumerate(cells)}
    cnt = np.zeros(len(cells))
    for _ in range(M):
        cnt[idx[draw()]] += 1
    e = M / len(cells)
    x = float(((cnt - e) ** 2 / e).sum())
    k = len(cells) - 1
    print(f"{label}: cells {len(cells)}, draws {M}, chi2 {x:.1f} (df {k}), p {chi2_sf(x, k):.3f}")

g = sd.random.default_rng(20261125)
run("permutation(5)", lambda: tuple(int(v) for v in g.permutation(5).tolist()),
    list(itertools.permutations(range(5))), 600_000)
run("choice(10, shape=(3,), replace=False)", lambda: tuple(int(v) for v in g.choice(10, shape=(3,), replace=False).tolist()),
    list(itertools.permutations(range(10), 3)), 720_000)
run("choice(64, shape=(2,), replace=False) [해시 경로]", lambda: tuple(int(v) for v in g.choice(64, shape=(2,), replace=False).tolist()),
    list(itertools.permutations(range(64), 2)), 2_016_000)
