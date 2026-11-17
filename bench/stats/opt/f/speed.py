"""Single-thread ns per element of the beta and F tails at ordinary shapes (the before/after
check of the tiny-shape change), 1e5 points at the distribution's own quantiles.

    .venv/bin/python bench/stats/opt/f/speed.py
"""

import timeit

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
CASES = [(S.Beta(a=2.5, b=0.7), st.beta(2.5, 0.7)), (S.F(dfn=3.0, dfd=11.0), st.f(3.0, 11.0)),
         (S.Beta(a=0.3, b=0.9), st.beta(0.3, 0.9)), (S.Beta(a=2.0, b=3.0), st.beta(2.0, 3.0))]
NAMES = ["Beta(2.5, 0.7)", "F(3, 11)", "Beta(0.3, 0.9)", "Beta(2, 3)"]


def main():
    n = 100_000
    p = np.random.default_rng(0).uniform(0, 1, n)
    before = sd.get_num_threads()
    sd.set_num_threads(1)
    try:
        for name, (X, ref) in zip(NAMES, CASES):
            x = sd.asarray(ref.ppf(p))
            row = []
            for m in ("cdf", "ccdf", "logcdf", "logccdf"):
                f = getattr(X, m)
                row.append(f"{m} {min(timeit.repeat(lambda: f(x), number=1, repeat=15)) / n * 1e9:6.1f}")
            print(f"{name:16s} " + "  ".join(row), flush=True)
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
