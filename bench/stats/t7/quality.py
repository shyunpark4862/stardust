"""T7: the samples against their distributions at 10^6 samples per case (the cases of
tests/stats/test_sample.py): the KS p-value against the distribution's own cdf (continuous)
and the chi-square p-value over its cdf steps (discrete). Uniform p-values are what a correct
sampler gives; one below 1e-3 in ~70 cases would be a flag.

    .venv/bin/python bench/stats/t7/quality.py > bench/stats/t7/quality.md
"""

import pathlib
import sys
import zlib

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
import stardust as sd  # noqa: E402
from tests.stats.test_sample import CONTINUOUS, DISCRETE  # noqa: E402

N = 10**6


def chi_square(D, x):
    finite = x[np.isfinite(x)]
    edges = np.unique(np.quantile(finite, np.linspace(0.0, 0.999, 1000)).round())
    probs = np.diff(np.concatenate([[0.0], np.asarray(D.cdf(sd.asarray(edges))), [1.0]]))
    counts = np.histogram(x, bins=np.concatenate([[-np.inf], edges + 0.5, [np.inf]]))[0]
    keep = probs * x.size >= 5
    e, o = probs[keep] * x.size, counts[keep]
    re, ro = x.size - e.sum(), x.size - o.sum()
    chi, dof = ((o - e) ** 2 / e).sum(), keep.sum() - 1
    if re >= 5:
        chi += (ro - re) ** 2 / re
        dof += 1
    return st.chi2.sf(chi, max(dof, 1)), dof


def main():
    print(f"{N:,} samples per case (seed: crc32 of the case).\n")
    print("| distribution | parameters | test | p-value |")
    print("|---|---|---|---:|")
    for kind, cases in (("KS", CONTINUOUS), ("chi2", DISCRETE)):
        for name, kw in cases:
            D = getattr(sd.stats, name)(**kw)
            params = ", ".join(f"{k}={v:g}" for k, v in kw.items())
            x = np.asarray(D.sample(N, rng=sd.random.default_rng(zlib.crc32(f"{name}{params}".encode()))))
            if kind == "KS":
                p = st.kstest(x, lambda v: np.asarray(D.cdf(sd.asarray(v)))).pvalue
                label = "KS"
            else:
                p, dof = chi_square(D, x)
                label = f"chi2 ({dof} dof)"
            print(f"| {name} | {params} | {label} | {p:.3g} |", flush=True)


if __name__ == "__main__":
    main()
