"""Group C: the fast pmf / logpmf paths against Arb (python-flint, 256 bits), where the comparison with
the double-double definitions (sweep.cpp) is not sharp enough: those take the log of their
slowly varying factor in float64 (an error of |log factor| ULP, 2e-15 for n ~ 1e15), so the
largest fast-vs-precise differences are the precise side's own error.

Points: parameters over many magnitudes, k within `zmax` sd of the mean ("bulk", |z| <= 4:
the fast paths' range for every distribution here except the hypergeometric's pmf, which
keeps to 2.2 sd) and beyond ("tail": mostly the double-double definitions). Prints the
largest error in contract units (relative for pmf; relative to max(1, |value|) for logpmf).

    .venv/bin/python bench/stats/opt/c/truth.py [points per distribution, default 1500]
"""

import math
import sys

import numpy as np
from flint import arb, ctx

import stardust as sd

S = sd.stats
ctx.prec = 256  # the log-gammas of 1e16 cancel 60 bits in a log pmf of size 1


def lg(v):
    return arb(v).lgamma()


def error(got, ref, mixed=False):
    """An upper bound of |got - ref| / |ref| (mixed: / max(1, |ref|)), ref an Arb ball."""
    return float((abs(arb(float(got)) - ref) / (max(arb(1), abs(ref)) if mixed else abs(ref))).upper())


def lchoose(n, k):
    return lg(n + 1) - lg(k + 1) - lg(n - k + 1)


def draw(rng, name):
    z = rng.uniform(-4, 4) if rng.uniform() < 0.7 else rng.uniform(-12, 12)
    p = float(rng.uniform(0.001, 0.999))
    if name == "binomial":
        n = float(math.floor(10 ** rng.uniform(1.5, 15.9)))
        k = min(max(math.floor(n * p + z * math.sqrt(n * p * (1 - p))), 0), n)
        t = lchoose(int(n), int(k)) + int(k) * arb(p).log() + int(n - k) * (-arb(p)).log1p()
        return S.Binomial(n=n, p=p), float(k), t, z
    if name == "negativebinomial":
        r = float(10 ** rng.uniform(0.5, 14))
        k = max(math.floor(r * (1 - p) / p + z * math.sqrt(r * (1 - p)) / p), 0)
        rm = arb(r)
        t = (k + rm).lgamma() - rm.lgamma() - lg(k + 1) + rm * arb(p).log() + k * (-arb(p)).log1p()
        return S.NegativeBinomial(n=r, p=p), float(k), t, z
    if name == "poisson":
        mu = float(10 ** rng.uniform(0, 15.5))
        k = max(math.floor(mu + z * math.sqrt(mu)), 0)
        return S.Poisson(mu=mu), float(k), k * arb(mu).log() - arb(mu) - lg(k + 1), z
    if name == "hypergeometric":
        N = math.floor(10 ** rng.uniform(2.2, 15.5))
        K, n = math.floor(N * rng.uniform(0.02, 0.98)), math.floor(N * rng.uniform(0.02, 0.98))
        X = S.Hypergeometric(total=float(N), successes=float(K), draws=float(n))
        lo, hi = X.support()
        k = int(min(max(math.floor(X.mean() + z * X.standard_deviation()), lo), hi))
        return X, float(k), lchoose(K, k) + lchoose(N - K, n - k) - lchoose(N, n), z
    if name == "geometric":
        pg = float(10 ** rng.uniform(-9, -0.01))
        k = max(1, math.ceil(-math.log(rng.uniform()) * (3 if abs(z) > 4 else 1) / pg))
        return S.Geometric(p=pg), float(k), (k - 1) * (-arb(pg)).log1p() + arb(pg).log(), z
    pl = float(1 - 10 ** rng.uniform(-9, -0.01))
    k = max(1, math.ceil(-math.log(rng.uniform()) * (3 if abs(z) > 4 else 1) / (1 - pl)))
    plm = arb(pl)
    return S.LogSeries(p=pl), float(k), k * plm.log() - arb(k).log() - (-(-plm).log1p()).log(), z


def main():
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rng = np.random.default_rng(11)
    print("| distribution | region | points | pmf max | logpmf max |")
    print("|---|---|---:|---:|---:|")
    for name in ("binomial", "negativebinomial", "poisson", "hypergeometric", "geometric", "logseries"):
        worst = {"bulk": [0, 0.0, 0.0], "tail": [0, 0.0, 0.0]}
        for _ in range(count):
            X, k, t, z = draw(rng, name)
            arr = sd.asarray(np.full(9, k))  # an array: the chunk kernel, past the memo's minimum
            got, got_l = float(np.asarray(X.pmf(arr))[0]), float(np.asarray(X.logpmf(arr))[0])
            w = worst["bulk" if abs(z) <= 4 else "tail"]
            w[0] += 1
            if got > 1e-290:
                w[1] = max(w[1], error(got, t.exp()))
            w[2] = max(w[2], error(got_l, t, mixed=True))
        for region, (n, a, b) in worst.items():
            print(f"| {name} | {region} | {n} | {a:.2e} | {b:.2e} |", flush=True)


if __name__ == "__main__":
    main()
