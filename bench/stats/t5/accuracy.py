"""T5: accuracy of the 10 discrete distributions against the Arb references
(tests/stats/ref), and scipy's error at the same points.

Errors are in the contract's units (docs/design/stats.md §2.3): relative error (absolute
for a subnormal reference), |error| / max(1, |value|) for entropy. The inverses (and
median) are held to a backward error (T25): the answer must be the exact quantile of some
target within relative 1e-14; for them the table gives the share of points whose answer
lies in that interval (sd and scipy), and "max" is 0 when all do. "in" is the guaranteed
region, "out" the rest of the measured points.

    .venv/bin/python bench/stats/t5/accuracy.py > bench/stats/t5/accuracy.md
"""

import math
import pathlib
import sys
import warnings

import numpy as np
import scipy.stats as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from tests.stats import refdata  # noqa: E402

DISCRETE = ("binomial", "bernoulli", "poisson", "geometric", "negativebinomial", "hypergeometric",
            "discreteuniform", "betabinomial", "logseries", "zipf")

# scipy's classic form of each distribution from the native parameter arrays
SCIPY = {
    "binomial": lambda n, p: st.binom(n, p),
    "bernoulli": lambda p: st.bernoulli(p),
    "poisson": lambda mu: st.poisson(mu),
    "geometric": lambda p: st.geom(p),
    "negativebinomial": lambda n, p: st.nbinom(n, p),
    "hypergeometric": lambda total, successes, draws: st.hypergeom(total, successes, draws),
    "discreteuniform": lambda lo, hi: st.randint(lo, hi),
    "betabinomial": lambda n, a, b: st.betabinom(n, a, b),
    "logseries": lambda p: st.logser(p),
    "zipf": lambda a: st.zipf(a),
}
METHOD = {"pmf": "pmf", "logpmf": "logpmf", "cdf": "cdf", "logcdf": "logcdf", "ccdf": "sf", "logccdf": "logsf",
          "icdf": "ppf", "iccdf": "isf"}
# scipy's ppf for these runs out of memory or time far in the tail (it steps k up one by one)
NO_SCIPY_INVERSE = {"zipf", "logseries"}
# scipy's zipf cdf and sf sum the pmf over 1..k (an array of k terms): past this k scipy is
# not called (NaN in its column; the k reach 3e307, and the run held 8 GB at the first one)
ZIPF_SCIPY_KMAX = 1e6


def fmt(v):
    if not np.isfinite(v):
        return "inf"
    return f"{v:.1e}"


def share(e, mask):
    return f"{100 * float((e[mask] <= refdata.GUARANTEED).mean()):.2f}%" if mask.any() else "—"


def scipy_values(dist, method, xs):
    nparams = len(xs) - 1
    if dist == "zipf" and method in ("cdf", "logcdf", "ccdf", "logccdf"):
        keep = xs[-1] <= ZIPF_SCIPY_KMAX
        v = np.full(xs[-1].size, np.nan)
        if keep.any():
            v[keep] = scipy_values("zipf_small", method, tuple(x[keep] for x in xs))
        return v
    dist = "zipf" if dist == "zipf_small" else dist
    with np.errstate(all="ignore"):
        try:
            return np.asarray(getattr(SCIPY[dist](*xs[:nparams]), METHOD[method])(xs[-1]), dtype=np.float64)
        except (OverflowError, ValueError, MemoryError):  # per point then
            v = np.empty(xs[-1].size)
            for i in range(v.size):
                try:
                    v[i] = getattr(SCIPY[dist](*(x[i] for x in xs[:nparams])), METHOD[method])(xs[-1][i])
                except (OverflowError, ValueError, MemoryError):
                    v[i] = np.nan
            return v


def main():
    warnings.simplefilter("ignore")
    print("| method | points | sd max (in) | sd max (out) | sd in interval | scipy max (in) | scipy in interval |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for name in refdata.names():
        dist, method = refdata.split(name)
        if dist not in DISCRETE:
            continue
        xs, hi, lo, g, sc = refdata.load(name)
        bounds = refdata.interval(name)
        got = refdata.call(name, xs)
        e = refdata.contract_error(name, got, hi, lo, sc, bounds)
        e = np.where(np.isnan(e), np.inf, e)
        ein = e[g].max() if g.any() else math.nan
        eout = e[~g].max() if (~g).any() else math.nan
        sp, sp_share = "—", "—"
        if method in METHOD and not (bounds is not None and dist in NO_SCIPY_INVERSE):
            es = refdata.contract_error(name, scipy_values(dist, method, xs), hi, lo, sc, bounds)
            es = np.where(np.isnan(es), np.inf, es)
            sp = fmt(es[g].max()) if g.any() else "—"
            if bounds is not None:
                sp_share = share(es, g)
        out = fmt(eout) if not math.isnan(eout) else "—"
        sd_share = share(e, g) if bounds is not None else "—"
        print(f"| `{name}` | {hi.size} | {fmt(ein)} | {out} | {sd_share} | {sp} | {sp_share} |", flush=True)


if __name__ == "__main__":
    main()
