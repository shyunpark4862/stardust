"""Reference values for sd.stats' distributions (tools/stats_ref.py) and the contract's
error measure (docs/design/stats.md §2.3; the format and the error functions are
tests/special/refdata.py's).

A file tests/stats/ref/<distribution>_<method>.npz holds the native function's inputs
x0, x1, ... (the parameters, then the method's input), the reference hi + lo, the
guaranteed region and the bound's multiplier `scale` (max(1, kappa) for the inverses).
"""

import pathlib

import numpy as np

import stardust as sd

from ..special import refdata as _special

REF = pathlib.Path(__file__).parent / "ref"
GUARANTEED = 1e-14
SUBNORMAL_ABS = 45 * 2.0**-1074

# Methods held to |error| <= 1e-14 max(1, |value|) instead of a relative error: logpdf
# (T20) and entropy (T24), log quantities that cross 0, and the Weibull skewness and
# kurtosis, which cross 0 (near c = 3.60 and c = 2.25, 5.77; T24).
MIXED = {"logpdf", "entropy"}
MIXED_FILES = {"weibull_skewness", "weibull_kurtosis"}

CLASSES = {"binomial": "Binomial", "bernoulli": "Bernoulli", "poisson": "Poisson", "geometric": "Geometric",
           "negativebinomial": "NegativeBinomial", "hypergeometric": "Hypergeometric", "discreteuniform": "DiscreteUniform",
           "betabinomial": "BetaBinomial", "logseries": "LogSeries", "zipf": "Zipf",
           "normal": "Normal", "exponential": "Exponential", "uniform": "Uniform", "laplace": "Laplace",
           "cauchy": "Cauchy", "lognormal": "LogNormal", "weibull": "Weibull", "studentt": "StudentT",
           "chisquared": "ChiSquared", "f": "F", "gamma": "Gamma", "beta": "Beta"}


def names():
    return sorted(p.stem for p in REF.glob("*.npz"))


def split(name):
    """('normal', 'cdf') from 'normal_cdf' (the method may hold underscores)."""
    for dist in sorted(CLASSES, key=len, reverse=True):
        if name.startswith(dist + "_"):
            return dist, name[len(dist) + 1:]
    raise KeyError(name)


def load(name):
    """(inputs, hi, lo, guaranteed, scale)."""
    with np.load(REF / f"{name}.npz") as f:
        nin = int(f["nin"])
        return tuple(f[f"x{k}"] for k in range(nin)), f["hi"], f["lo"], f["guaranteed"], f["scale"]


def interval(name):
    """(kmin, kmax) of a discrete inverse's file (the answers its backward-error contract
    allows: the exact quantiles of the targets within relative 1e-14; T25), else None."""
    with np.load(REF / f"{name}.npz") as f:
        return (f["kmin"], f["kmax"]) if "kmin" in f else None


def call(name, xs):
    """The method through the public API: the class built from the parameter arrays."""
    dist, method = split(name)
    cls = getattr(sd.stats, CLASSES[dist])
    nparams = len(cls._params)
    conv = (lambda v: v) if isinstance(xs[0], float) else sd.asarray  # Python floats stay scalars
    params = {p: conv(x) for p, x in zip(cls._params, xs[:nparams])}
    X = cls(**params)
    if method in ("support_lo", "support_hi"):
        return np.asarray(X.support()[method == "support_hi"])
    if len(xs) > nparams:
        return np.asarray(getattr(X, method)(conv(xs[nparams])))
    return np.asarray(getattr(X, method)())


def contract_error(name, got, hi, lo, scale, bounds=None):
    """The error in units where 1e-14 is the bound: relative (absolute for a subnormal
    reference), divided by the inverses' scale; the mixed measure for MIXED methods; for a
    discrete inverse with `bounds` (kmin, kmax), 0 inside and 1e-14 plus the relative
    distance to the interval outside."""
    if bounds is not None:
        kmin, kmax = bounds
        inside = (got >= kmin) & (got <= kmax)
        with np.errstate(invalid="ignore", divide="ignore"):
            near = np.where(got < kmin, kmin, kmax)
            dist = np.where(got == near, 0.0, np.abs(got - near) / np.abs(near))
        return np.where(inside, 0.0, GUARANTEED + np.nan_to_num(dist, nan=np.inf))
    method = split(name)[1]
    if method in MIXED or name in MIXED_FILES:
        return _special.mixed_error(got, hi, lo)
    rel = _special.rel_error(got, hi, lo)
    sub = np.isfinite(hi) & (np.abs(hi) < 2.0**-1022)
    with np.errstate(invalid="ignore", over="ignore"):
        absolute = _special._diff(got, hi, lo) / SUBNORMAL_ABS * GUARANTEED
    return np.where(sub, absolute, rel) / scale
