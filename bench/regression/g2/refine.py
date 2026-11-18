"""G2: is the refinement needed in every IRLS iteration, or only in the final solve?
(docs/design/regression.md §3.4). Results: bench/regression/g2/refine.md.

Accuracy: logistic and poisson regressions with X = U diag(s) V^T (n = 200, k = 6 with the
intercept column among them: no separate intercept), s from 1 to 1/kappa geometrically, and
beta chosen so that X beta has unit scale (beta itself grows with kappa). The coefficients'
normwise relative error against Arb (tools/regref.py's glm, the same double inputs) for
  glm: the iterations before the deviance's convergence by the normal equations (Cholesky,
       G40), the polishing iterations after it and the final solve by QR with up to 3
       corrections
  QR inner: as glm, but the iterations before the convergence by QR without refinement
  every: QR with up to 3 corrections in every iteration
  no final: as glm but without the final solve's refinement
and the standard errors' (glm). Three seeds, the largest error.
Time: the logistic (10^5, 10) and (10^6, 50) fits of bench/regression/g2/speed.py with the
first two settings, 1 thread, best of 3.

Run: uv run --no-sync python bench/regression/g2/refine.py
"""

import os
import sys
import time

import numpy as np

import stardust as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import regref  # noqa: E402

K = sd._core.regression
FAM = {"binomial": 1, "poisson": 2}
LINK = {"binomial": 2, "poisson": 1}


def fit(y, X, family, inner, final, intercept=False, chol=True):
    n = y.shape[-1]
    aux = np.zeros((4, n))
    aux[:3] = 1.0
    raw = K.glm_fit("glm", sd.asarray(y), sd.asarray(X), sd.asarray(aux), None, 0, FAM[family], LINK[family], False,
                    False, False, False, intercept, -1.0, 0, -1, 100, 1e-8, 0, 0.0, inner, final, chol)
    st = np.asarray(raw[6])
    return np.asarray(raw[0]), np.sqrt(np.diag(np.asarray(raw[1])) * st[5]), int(st[8])


def problem(kappa, family, seed):
    rng = np.random.default_rng(seed)
    n, k = 200, 6
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    c *= 0.8 / np.linalg.norm(U @ c * np.sqrt(n)) * np.sqrt(n)
    beta = V @ (c / s)
    eta = X @ beta
    if family == "binomial":
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = rng.poisson(np.exp(0.5 + eta)).astype(float)
        X[:, 0] += 0.0
    return y, X


def nerr(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return float(np.linalg.norm((got - hi) - lo) / np.linalg.norm(hi + lo))


def rerr(got, ref):
    return float(max(abs((g - h) - l) / abs(h + l) for g, (h, l) in zip(got, ref)))


def main():
    rows = []
    for family in ("binomial", "poisson"):
        for kappa in (1e0, 1e4, 1e6, 1e8, 1e10, 1e12):
            worst = {}
            its = set()
            for seed in range(3):
                y, X = problem(kappa, family, seed)
                b0, se0, it0 = fit(y, X, family, 0, 3)
                bq, _, _ = fit(y, X, family, 0, 3, chol=False)
                b3, _, _ = fit(y, X, family, 3, 3, chol=False)
                bn, _, _ = fit(y, X, family, 0, 0)
                its.add(it0)
                ref, _ = regref.glm(y.tolist(), X.tolist(), family=family, link="logit" if family == "binomial" else "log",
                                    intercept=False, beta0=b0.tolist())
                for key, v in (("0/3", nerr(b0, ref["params"])), ("qr", nerr(bq, ref["params"])),
                               ("3/3", nerr(b3, ref["params"])),
                               ("0/0", nerr(bn, ref["params"])), ("bse", rerr(se0, ref["bse"]))):
                    worst[key] = max(worst.get(key, 0.0), v)
            rows.append((family, kappa, worst, sorted(its)))
            print(rows[-1], flush=True)
    print()
    print("| family | kappa | glm | QR inner | every | no final | bse (glm) | iterations |")
    print("|---|---:|---:|---:|---:|---:|---|")
    for family, kappa, w, its in rows:
        print(f"| {family} | {kappa:.0e} | {w['0/3']:.1e} | {w['qr']:.1e} | {w['3/3']:.1e} | {w['0/0']:.1e} | {w['bse']:.1e} | "
              f"{', '.join(map(str, its))} |")

    sd.set_num_threads(1)
    rng = np.random.default_rng(0)
    print()
    print("| (n, k) | glm (ms) | QR inner (ms) | every (ms) |")
    for n, k in ((100_000, 10), (1_000_000, 50)):
        X = rng.standard_normal((n, k))
        beta = rng.standard_normal(k) * (0.6 / np.sqrt(k))
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.2 + X @ beta)))).astype(float)
        aux = sd.asarray(np.r_[np.ones((3, n)), np.zeros((1, n))])
        Xs, ys = sd.asarray(X), sd.asarray(y)
        t = {}
        for key, inner, chol in (("glm", 0, True), ("qr", 0, False), ("every", 3, False)):
            best = []
            for _ in range(3):
                t0 = time.perf_counter()
                K.glm_fit("glm", ys, Xs, aux, None, 0, 1, 2, False, False, False, False, True, -1.0, 0, -1, 100, 1e-8,
                          0, 0.0, inner, 3, chol)
                best.append(time.perf_counter() - t0)
            t[key] = min(best)
        print(f"| ({n:,}, {k}) | {t['glm'] * 1e3:.0f} | {t['qr'] * 1e3:.0f} | {t['every'] * 1e3:.0f} |", flush=True)


if __name__ == "__main__":
    main()
