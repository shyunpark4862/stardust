"""Group C (discrete distributions): single-thread speed against scipy.stats, two sets.

"t5": the parameters of bench/stats/t5/speed.py (small supports: k repeats in an array).
"wide": large parameters, where nearly every k in a chunk is different, so a row shows the
cost of one evaluation and not of a table lookup.

    .venv/bin/python bench/stats/opt/c/speed.py [t5|wide] [Name ...]

Columns: ns per element for sd (1 thread) and scipy, and scipy / sd.
"""

import sys
import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
SCIPY = {
    "pmf": "pmf",
    "logpmf": "logpmf",
    "cdf": "cdf",
    "logcdf": "logcdf",
    "ccdf": "sf",
    "logccdf": "logsf",
    "icdf": "ppf",
    "iccdf": "isf",
}

SETS = {
    "t5": [
        (S.Binomial, {"n": 40, "p": 0.3}, st.binom(40, 0.3)),
        (S.Poisson, {"mu": 4.5}, st.poisson(4.5)),
        (S.Geometric, {"p": 0.2}, st.geom(0.2)),
        (S.NegativeBinomial, {"n": 2.5, "p": 0.3}, st.nbinom(2.5, 0.3)),
        (S.Hypergeometric, {"total": 500, "successes": 120, "draws": 200}, st.hypergeom(500, 120, 200)),
        (S.Bernoulli, {"p": 0.3}, st.bernoulli(0.3)),
        (S.DiscreteUniform, {"low": 2, "high": 90}, st.randint(2, 90)),
        (S.BetaBinomial, {"n": 40, "a": 2.5, "b": 1.7}, st.betabinom(40, 2.5, 1.7)),
        (S.LogSeries, {"p": 0.7}, st.logser(0.7)),
        (S.Zipf, {"a": 2.5}, st.zipf(2.5)),
    ],
    "wide": [
        (S.Binomial, {"n": 10**9, "p": 0.3}, st.binom(10**9, 0.3)),
        (S.Poisson, {"mu": 3e8}, st.poisson(3e8)),
        (S.Geometric, {"p": 1e-6}, st.geom(1e-6)),
        (S.NegativeBinomial, {"n": 2.5e6, "p": 1e-3}, st.nbinom(2.5e6, 1e-3)),
        (
            S.Hypergeometric,
            {"total": 5 * 10**6, "successes": 12 * 10**5, "draws": 2 * 10**6},
            st.hypergeom(5 * 10**6, 12 * 10**5, 2 * 10**6),
        ),
        (S.DiscreteUniform, {"low": 2, "high": 10**9}, st.randint(2, 10**9)),
        (S.BetaBinomial, {"n": 20000, "a": 2.5, "b": 1.7}, st.betabinom(20000, 2.5, 1.7)),
        (S.LogSeries, {"p": 1 - 1e-7}, st.logser(1 - 1e-7)),
        (S.Zipf, {"a": 1.05}, st.zipf(1.05)),
    ],
}


def best(f, budget=0.5):
    t = timeit.timeit(f, number=1)
    if t > 2.0:
        return t
    reps = max(1, min(5, int(budget / max(t, 1e-9))))
    return min([t] + timeit.repeat(f, number=1, repeat=reps))


def main():
    warnings.simplefilter("ignore")
    which = sys.argv[1] if len(sys.argv) > 1 else "t5"
    names = set(sys.argv[2:])
    rng = np.random.default_rng(0)
    N0 = 100_000 if which == "t5" else 20_000
    sd.set_num_threads(1)
    print(f"set {which}: {N0} points, 1 thread, ns per element\n")
    print("| distribution | method | sd | scipy | scipy / sd |")
    print("|---|---|---:|---:|---:|")
    for cls, params, ref in SETS[which]:
        if names and cls.__name__ not in names:
            continue
        # the sums of the wide hypergeometric and beta-binomial cdfs run to thousands of terms
        N = 2000 if which == "wide" and cls in (S.Hypergeometric, S.BetaBinomial) else N0
        p = rng.uniform(0, 1, N)
        X = cls(**params)
        x = np.asarray(X.icdf(sd.asarray(p)))
        sx, sp = sd.asarray(x), sd.asarray(p)
        for m in ("pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf"):
            arg, npa = (sp, p) if m in ("icdf", "iccdf") else (sx, x)
            f = getattr(X, m)
            t1 = best(lambda: f(arg)) / N * 1e9
            g = getattr(ref, SCIPY[m])
            small = npa[: max(1, N // 50)]  # scipy's slow rows (sums, searches): a sample
            try:
                if which == "wide" and cls is S.Zipf and m not in ("pmf", "logpmf"):
                    raise RuntimeError  # scipy's zipf cdf at k ~ 1e20 does not return
                ts = timeit.timeit(lambda: g(small), number=1) / len(small) * 1e9
                if ts * N < 2e9:
                    ts = best(lambda: g(npa)) / N * 1e9
                row = f"{ts:.1f} | {ts / t1:.2f}"
            except Exception:  # scipy's zipf ppf: "updating stopped, endless loop"
                row = "— | —"
            print(f"| {cls.__name__} | `{m}` | {t1:.1f} | {row} |", flush=True)


if __name__ == "__main__":
    main()
