"""G2 (G42): Newton's polishing and final step through the thin Q for the non-canonical links
(docs/design/regression.md §3.4). Results: bench/regression/g2/noncanonical.md.

Accuracy: X = U diag(s) V^T (n = 200, k = 5 and the intercept, s from 1 to 1/kappa
geometrically), eta = base + U c scaled per link so that mu stays in its range, y drawn from
the family. The coefficients' normwise relative error against Arb (tools/regref.py's glm),
two seeds, the largest, for glm (Newton through the thin Q) and for the same fit with Fisher's
polishing and final step (the kernel's newton=False), with glm's n_iter.
Slow Fisher scoring (G44): the gamma-log problem of bench/regression/g2/refine.py's form at
kappa 1e4-1e10 (n = 200, k = 6, no intercept), glm against Fisher's steps throughout.
Time: probit and gamma-log fits at (10^5, 10) and (10^6, 50), 1 and 10 threads, best of 3,
with and without Newton's steps; logit and poisson-log (canonical: unchanged) for reference.

Run: uv run --no-sync python bench/regression/g2/noncanonical.py
"""

import os
import sys
import time

import numpy as np
from scipy.stats import norm

import stardust as sd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "tools"))
import regref  # noqa: E402

K = sd._core.regression
FAM = {"gaussian": 0, "binomial": 1, "poisson": 2, "gamma": 3}
LINK = {"identity": 0, "log": 1, "logit": 2, "probit": 3, "cloglog": 4, "sqrt": 5, "inverse": 6}
LINKS = [
    ("binomial", "probit", 0.3, 1.0),
    ("binomial", "cloglog", -0.5, 1.0),
    ("binomial", "log", -1.5, 0.4),
    ("poisson", "identity", 5.0, 2.0),
    ("poisson", "sqrt", 2.0, 0.5),
    ("gamma", "log", 0.5, 1.0),
    ("gamma", "identity", 4.0, 1.5),
    ("gaussian", "log", 0.8, 0.5),
    ("binomial", "logit", 0.2, 1.0),
    ("poisson", "log", 0.5, 1.0),
]
INV = {
    "probit": norm.cdf,
    "cloglog": lambda e: -np.expm1(-np.exp(e)),
    "log": np.exp,
    "identity": lambda e: e,
    "sqrt": lambda e: e * e,
    "logit": lambda e: 1 / (1 + np.exp(-e)),
}


def draw(rng, family, mu):
    n = mu.shape[0]
    return {
        "binomial": lambda: (rng.uniform(size=n) < mu).astype(float),
        "poisson": lambda: rng.poisson(mu).astype(float),
        "gamma": lambda: rng.gamma(3.0, mu / 3.0),
        "gaussian": lambda: mu + 0.3 * rng.standard_normal(n),
    }[family]()


def problem(family, link, base, scale, kappa, seed):
    rng = np.random.default_rng(seed)
    n, k = 200, 5
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    eta = base + U @ c * (scale / np.linalg.norm(U @ c)) * np.sqrt(n) / 4
    return draw(rng, family, INV[link](eta)), X


def slow_problem(kappa, seed):
    rng = np.random.default_rng(int(np.log10(kappa)) + 7 + 100 * seed)
    n, k = 200, 6
    U, _ = np.linalg.qr(rng.standard_normal((n, k)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    s = kappa ** (-np.arange(k) / (k - 1))
    X = (U * s) @ V.T * np.sqrt(n)
    c = rng.standard_normal(k)
    c *= 0.8 / np.linalg.norm(U @ c)
    eta = X @ (V @ (c / s))
    return rng.gamma(3.0, np.exp(0.5 + eta) / 3.0), X


def fit(y, X, family, link, newton, threads_aux=None, intercept=True):
    n = y.shape[-1]
    aux = threads_aux if threads_aux is not None else sd.asarray(np.r_[np.ones((3, n)), np.zeros((1, n))])
    raw = K.glm_fit(
        "glm",
        sd.asarray(y) if not isinstance(y, sd.ndarray) else y,
        sd.asarray(X) if not isinstance(X, sd.ndarray) else X,
        aux,
        None,
        0,
        FAM[family],
        LINK[link],
        False,
        False,
        False,
        False,
        intercept,
        -1.0,
        0,
        -1,
        100,
        1e-8,
        0,
        0.0,
        0,
        3,
        True,
        2 if newton else 0,
    )
    st = np.asarray(raw[6])
    return np.asarray(raw[0]), int(st[8]), bool(st[9])


def main():
    kappas = (1e4, 1e6, 1e8, 1e10)
    print("| family | link | " + " | ".join(f"kappa {k:.0e}: glm / Fisher (iterations)" for k in kappas) + " |")
    print("|---|---|" + "---:|" * len(kappas))
    for family, link, base, scale in LINKS:
        cells = []
        for kappa in kappas:
            eg = ef = 0.0
            its = []
            for seed in range(2):
                y, X = problem(family, link, base, scale, kappa, seed)
                bg, it, cg = fit(y, X, family, link, True)
                bf, _, _ = fit(y, X, family, link, False)
                ref, _ = regref.glm(y.tolist(), X.tolist(), family=family, link=link, beta0=bg.tolist())
                hi = np.array([v[0] + v[1] for v in ref["params"]])
                eg = max(eg, np.linalg.norm(bg - hi) / np.linalg.norm(hi))
                ef = max(ef, np.linalg.norm(bf - hi) / np.linalg.norm(hi))
                its.append(f"{it}" + ("" if cg else "*"))
            cells.append(f"{eg:.1e} / {ef:.1e} ({', '.join(its)})")
        print(f"| {family} | {link} | " + " | ".join(cells) + " |", flush=True)

    print()
    print("| kappa | seed | glm: error (iterations) | Fisher: error (iterations) |")
    print("|---:|---:|---:|---:|")
    for kappa in (1e4, 1e6, 1e8, 1e10):
        for seed in range(2):
            y, X = slow_problem(kappa, seed)
            bg, ig, cg = fit(y, X, "gamma", "log", True, intercept=False)
            bf, i_f, cf = fit(y, X, "gamma", "log", False, intercept=False)
            ref, _ = regref.glm(y.tolist(), X.tolist(), family="gamma", link="log", intercept=False, beta0=bg.tolist())
            hi = np.array([v[0] + v[1] for v in ref["params"]])
            e = lambda b: np.linalg.norm(b - hi) / np.linalg.norm(hi)
            print(
                f"| {kappa:.0e} | {seed} | {e(bg):.1e} ({ig}{'' if cg else ', not converged'}) | "
                f"{e(bf):.1e} ({i_f}{'' if cf else ', not converged'}) |",
                flush=True,
            )

    print()
    print("| (n, k) | family, link | threads | glm (ms, iterations) | Fisher's polishing (ms, iterations) | added |")
    print("|---|---|---:|---:|---:|---:|")
    rng = np.random.default_rng(0)
    for n, k in ((100_000, 10), (1_000_000, 50)):
        X = rng.standard_normal((n, k))
        beta = rng.standard_normal(k) * (0.6 / np.sqrt(k))
        eta = X @ beta
        Xs = sd.asarray(X)
        aux = sd.asarray(np.r_[np.ones((3, n)), np.zeros((1, n))])
        for family, link, base in (
            ("binomial", "probit", 0.1),
            ("gamma", "log", 0.5),
            ("binomial", "logit", 0.2),
            ("poisson", "log", 0.3),
        ):
            y = sd.asarray(draw(rng, family, INV[link](base + eta * (0.5 if family == "poisson" else 1.0))))
            for threads in (1, 10):
                sd.set_num_threads(threads)
                t = {}
                for newton in (True, False):
                    best = []
                    for _ in range(3):
                        t0 = time.perf_counter()
                        _, it, _ = fit(y, Xs, family, link, newton, aux)
                        best.append(time.perf_counter() - t0)
                    t[newton] = (min(best), it)
                (a, ia), (b, ib) = t[True], t[False]
                print(
                    f"| ({n:,}, {k}) | {family} {link} | {threads} | {a * 1e3:,.1f} ({ia}) | {b * 1e3:,.1f} ({ib}) | "
                    f"{(a / b - 1) * 100:+.0f}% |",
                    flush=True,
                )


if __name__ == "__main__":
    main()
