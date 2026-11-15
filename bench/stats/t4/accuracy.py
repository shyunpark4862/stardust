"""T4: accuracy of the 12 continuous distributions against the mpmath references
(tests/stats/ref), and scipy's error at the same points.

Errors are in the contract's units (docs/design/stats.md §2.3): relative error (absolute
for a subnormal reference), divided by max(1, kappa) for the inverses, and
|error| / max(1, |value|) for logpdf, entropy and the Weibull skewness and kurtosis.
"in" is the guaranteed region, "out" the rest of the measured points.

    .venv/bin/python bench/stats/t4/accuracy.py > bench/stats/t4/accuracy.md
"""

import math
import pathlib
import sys
import warnings

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from tests.stats import refdata  # noqa: E402

# scipy's classic form of each distribution from the native parameter arrays
SCIPY = {
    "normal": lambda mu, s: st.norm(mu, s),
    "studentt": lambda df: st.t(df),
    "chisquared": lambda df: st.chi2(df),
    "f": lambda m, n: st.f(m, n),
    "gamma": lambda a, s: st.gamma(a, scale=s),
    "beta": lambda a, b: st.beta(a, b),
    "exponential": lambda s: st.expon(scale=s),
    "lognormal": lambda mu, s: st.lognorm(s, scale=np.exp(mu)),
    "uniform": lambda a, b: st.uniform(a, b - a),
    "cauchy": lambda mu, s: st.cauchy(mu, s),
    "laplace": lambda mu, s: st.laplace(mu, s),
    "weibull": lambda c, s: st.weibull_min(c, scale=s),
}
METHOD = {"pdf": "pdf", "logpdf": "logpdf", "cdf": "cdf", "logcdf": "logcdf", "ccdf": "sf", "logccdf": "logsf",
          "icdf": "ppf", "iccdf": "isf"}


def fmt(v):
    if not np.isfinite(v):
        return "inf"
    return f"{v:.1e}"


def main():
    warnings.simplefilter("ignore")
    print("| method | points | sd max (in) | sd max (out) | scipy max (in) |")
    print("|---|---:|---:|---:|---:|")
    for name in refdata.names():
        dist, method = refdata.split(name)
        xs, hi, lo, g, sc = refdata.load(name)
        got = refdata.call(name, xs)
        e = refdata.contract_error(name, got, hi, lo, sc)
        e = np.where(np.isnan(e), np.inf, e)
        ein = e[g].max() if g.any() else math.nan
        eout = e[~g].max() if (~g).any() else math.nan
        sp = "—"
        if method in METHOD:
            nparams = len(xs) - 1
            with np.errstate(all="ignore"):
                ref = SCIPY[dist](*xs[:nparams])
                try:
                    v = getattr(ref, METHOD[method])(xs[-1])
                except (OverflowError, ValueError):  # scipy raises on some points: per point then
                    v = np.empty(xs[-1].size)
                    for i in range(v.size):
                        try:
                            v[i] = getattr(SCIPY[dist](*(x[i] for x in xs[:nparams])), METHOD[method])(xs[-1][i])
                        except (OverflowError, ValueError):
                            v[i] = np.nan
            es = refdata.contract_error(name, np.asarray(v, dtype=np.float64), hi, lo, sc)
            es = np.where(np.isnan(es), np.inf, es)
            sp = fmt(es[g].max()) if g.any() else "—"
        out = fmt(eout) if not math.isnan(eout) else "—"
        print(f"| `{name}` | {hi.size} | {fmt(ein)} | {out} | {sp} |", flush=True)


if __name__ == "__main__":
    main()
