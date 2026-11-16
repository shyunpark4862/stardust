"""T6: accuracy of the extreme value, lifetime and noncentral distributions against the
Arb references (tests/stats/ref), and scipy's error at the same points.

Errors are in the contract's units (docs/design/stats.md §2.3): relative error (absolute
for a subnormal reference), divided by max(1, kappa) for the inverses, |error| /
max(1, |value|) for logpdf, entropy and GEV's skewness and kurtosis. "in" is the
guaranteed region, "out" the rest of the measured points (for the noncentral three: tails
below 1e-300 and their entropies, T26).

    .venv/bin/python bench/stats/t6/accuracy.py > bench/stats/t6/accuracy.md
"""

import math
import pathlib
import sys
import warnings

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from tests.stats import refdata  # noqa: E402

DISTS = ("logistic", "gumbel", "gev", "pareto", "rayleigh", "inversegamma", "noncentralchisquared", "noncentralt",
         "noncentralf")

# scipy's classic form of each distribution from the native parameter arrays
SCIPY = {
    "logistic": lambda mu, s: st.logistic(mu, s),
    "gumbel": lambda mu, s: st.gumbel_r(mu, s),
    "gev": lambda c, mu, s: st.genextreme(c, mu, s),
    "pareto": lambda b, s: st.pareto(b, scale=s),
    "rayleigh": lambda s: st.rayleigh(scale=s),
    "inversegamma": lambda a, s: st.invgamma(a, scale=s),
    "noncentralchisquared": lambda df, nc: st.ncx2(df, nc),
    "noncentralt": lambda df, nc: st.nct(df, nc),
    "noncentralf": lambda dfn, dfd, nc: st.ncf(dfn, dfd, nc),
}
METHOD = {"pdf": "pdf", "logpdf": "logpdf", "cdf": "cdf", "logcdf": "logcdf", "ccdf": "sf", "logccdf": "logsf",
          "icdf": "ppf", "iccdf": "isf"}
# scipy's noncentral inverses take minutes on 4000 points (a root search per point)
NO_SCIPY_INVERSE = {"noncentralchisquared", "noncentralt", "noncentralf"}


def fmt(v):
    if not np.isfinite(v):
        return "inf"
    return f"{v:.1e}"


def scipy_values(dist, method, xs):
    nparams = len(xs) - 1
    with np.errstate(all="ignore"):
        try:
            return np.asarray(getattr(SCIPY[dist](*xs[:nparams]), METHOD[method])(xs[-1]), dtype=np.float64)
        except (OverflowError, ValueError, MemoryError):  # per point then
            v = np.empty(xs[-1].size)
            for i in range(v.size):
                try:
                    v[i] = getattr(SCIPY[dist](*(x[i] for x in xs[:nparams])), METHOD[method])(xs[-1][i])
                except Exception:  # noqa: BLE001 (scipy's own failures on a point: NaN, an error for it)
                    v[i] = np.nan
            return v


def main():
    warnings.simplefilter("ignore")
    print("| method | points | sd max (in) | sd max (out) | scipy max (in) |")
    print("|---|---:|---:|---:|---:|")
    for name in refdata.names():
        dist, method = refdata.split(name)
        if dist not in DISTS:
            continue
        xs, hi, lo, g, sc = refdata.load(name)
        got = refdata.call(name, xs)
        e = refdata.contract_error(name, got, hi, lo, sc)
        e = np.where(np.isnan(e), np.inf, e)
        ein = e[g].max() if g.any() else math.nan
        eout = e[~g].max() if (~g).any() else math.nan
        sp = "—"
        if method in METHOD and not (method in ("icdf", "iccdf") and dist in NO_SCIPY_INVERSE):
            es = refdata.contract_error(name, scipy_values(dist, method, xs), hi, lo, sc)
            es = np.where(np.isnan(es), np.inf, es)
            sp = fmt(es[g].max()) if g.any() else "—"
        out = fmt(eout) if not math.isnan(eout) else "—"
        ein_s = fmt(ein) if not math.isnan(ein) else "—"
        print(f"| `{name}` | {hi.size} | {ein_s} | {out} | {sp} |", flush=True)


if __name__ == "__main__":
    main()
