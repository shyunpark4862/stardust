"""Reference values for sd.stats' distributions (tools/stats_ref.py) and the contract's
error measure (docs/design/stats.md §2.3; the error functions are tests/special/refdata.py's).

A file tests/stats/ref/<distribution>_<kind>.npz (layout: tests/reffile.py) holds one input
set of a distribution -- kind x (the points of the densities and cdfs), p (the inverses'
probabilities), logp (their logarithms) or param (the moments: parameters only) -- as the
native function's inputs x0, x1, ... (the parameters, then the method's input, if any), and
per method of that kind the reference hi + lo at its points, the guaranteed region and the
bound's multiplier `scale` (max(1, kappa) for the inverses); the discrete inverses also hold
kmin, kmax (the answers their backward-error contract allows). A method is named
<distribution>_<method> (normal_cdf).
"""

import functools
import pathlib

import numpy as np

import stardust as sd

from .. import reffile
from ..special import refdata as _special

REF = pathlib.Path(__file__).parent / "ref"
GUARANTEED = 1e-14
SUBNORMAL_ABS = 45 * 2.0**-1074

# Methods held to |error| <= 1e-14 max(1, |value|) instead of a relative error: logpdf and
# entropy, log quantities that cross 0, and the Weibull skewness and kurtosis, which cross 0
# (near c = 3.60 and c = 2.25, 5.77; docs/design/stats.md §2.3).
MIXED = {"logpdf", "entropy"}
MIXED_FILES = {"weibull_skewness", "weibull_kurtosis", "gev_skewness", "gev_kurtosis"}  # GEV's W = Y^c is a Weibull

CLASSES = {
    "binomial": "Binomial",
    "bernoulli": "Bernoulli",
    "poisson": "Poisson",
    "geometric": "Geometric",
    "negativebinomial": "NegativeBinomial",
    "hypergeometric": "Hypergeometric",
    "discreteuniform": "DiscreteUniform",
    "betabinomial": "BetaBinomial",
    "logseries": "LogSeries",
    "zipf": "Zipf",
    "normal": "Normal",
    "exponential": "Exponential",
    "uniform": "Uniform",
    "laplace": "Laplace",
    "cauchy": "Cauchy",
    "lognormal": "LogNormal",
    "weibull": "Weibull",
    "studentt": "StudentT",
    "chisquared": "ChiSquared",
    "f": "F",
    "gamma": "Gamma",
    "beta": "Beta",
    "logistic": "Logistic",
    "gumbel": "Gumbel",
    "gev": "GEV",
    "pareto": "Pareto",
    "rayleigh": "Rayleigh",
    "inversegamma": "InverseGamma",
    "noncentralchisquared": "NoncentralChiSquared",
    "noncentralt": "NoncentralT",
    "noncentralf": "NoncentralF",
}


PARAM_METHODS = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy")


def kind(method):
    """The input set a method is evaluated on: "param", "p", "logp" or "x"."""
    if method in PARAM_METHODS:
        return "param"
    if method in ("icdf", "iccdf"):
        return "p"
    if method in ("ilogcdf", "ilogccdf"):
        return "logp"
    return "x"


def path(name):
    """The file of a method <distribution>_<method>."""
    dist, method = split(name)
    return REF / f"{dist}_{kind(method)}.npz"


@functools.lru_cache(maxsize=4)
def _file(p):
    return reffile.read(p)


def names():
    """Every <distribution>_<method> with references."""
    out = []
    for p in REF.glob("*.npz"):
        dist = split(p.stem + "_")[0]
        out += [f"{dist}_{m}" for m in reffile.read(p).methods]
    return sorted(out)


def split(name):
    """('normal', 'cdf') from 'normal_cdf' (the method may hold underscores)."""
    for dist in sorted(CLASSES, key=len, reverse=True):
        if name.startswith(dist + "_"):
            return dist, name[len(dist) + 1 :]
    raise KeyError(name)


def _method(name):
    r = _file(path(name))
    return r, r.methods[split(name)[1]]


def load(name):
    """(inputs, hi, lo, guaranteed, scale) at the method's points."""
    r, m = _method(name)
    f = m.fields
    return r.args(split(name)[1]), m.hi.copy(), m.lo.copy(), f["guaranteed"].copy(), f["scale"].copy()


def interval(name):
    """(kmin, kmax) of a discrete inverse (the answers its backward-error contract allows: the
    exact quantiles of the targets within relative 1e-14; docs/design/stats.md §2.3), else None."""
    m = _method(name)[1]
    return (m.fields["kmin"].copy(), m.fields["kmax"].copy()) if "kmin" in m.fields else None


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


# ---------------------------------------------------------------- long tail sums
# tests/stats/ref/tails/<distribution>.npz (tools/stats_ref.py <distribution>_tails): the cdfs of
# the hypergeometric and the beta-binomial where a side of the pmf has 1e4 to 1e7 terms, a group
# per parameter set.

TAILS = REF / "tails"
TAIL_DISTRIBUTIONS = ("hypergeometric", "betabinomial")


def tail_groups(dist):
    """[(parameters, k, {method: (hi, lo)})]: the points k (float64) of each parameter set."""
    r = reffile.read(TAILS / f"{dist}.npz")
    *params, k = r.args()
    out = []
    for i in range(r.ngroups):
        sel = np.flatnonzero(r.group == i)
        out.append(
            (
                tuple(float(p[sel[0]]) for p in params),
                k[sel],
                {m: (mm.hi[sel], mm.lo[sel]) for m, mm in r.methods.items()},
            )
        )
    return out


# ---------------------------------------------------------------- multivariate
# tests/stats/ref/mv/<family>.npz (the densities, entropies and pmfs of the multivariate and
# matrix distributions: a group per distribution, its points x) and tests/stats/ref/mv/<set>.npz
# (the cdfs: one method, cdf, at points h and the correlations), both from tools/stats_ref.py.

MV = REF / "mv"
# family -> (class name, methods)
MV_FAMILIES = {
    "mvnormal": ("MultivariateNormal", ("logpdf", "pdf", "entropy")),
    "mvt": ("MultivariateT", ("logpdf", "pdf", "entropy")),
    "wishart": ("Wishart", ("logpdf", "entropy")),
    "invwishart": ("InverseWishart", ("logpdf", "entropy")),
    "matnormal": ("MatrixNormal", ("logpdf", "entropy")),
    "multinomial": ("Multinomial", ("logpmf", "entropy")),
    "dirichlet": ("Dirichlet", ("logpdf", "entropy")),
}
MV_CDF_SETS = ("bvn", "tvn", "bvt", "tvt", "factor_n", "factor_t")


def mv_groups(family):
    """A density family's groups: {parameter: value, "x": the points, "<method>_hi", "_lo"}
    (a per-group method, the entropy, as a 1-element array)."""
    r = reffile.read(MV / f"{family}.npz")
    out = []
    for i in range(r.ngroups):
        sel = np.flatnonzero(r.group == i)
        g = {k: v[i] for k, v in r.group_inputs.items()}
        x = r.inputs["x"]
        g["x"] = x[sel] if isinstance(x, np.ndarray) else np.stack([x[j] for j in sel])
        for m, mm in r.methods.items():
            at = np.flatnonzero(np.isin(mm.index, [i] if mm.over == "group" else sel))
            g[f"{m}_hi"], g[f"{m}_lo"] = mm.hi[at], mm.lo[at]
        out.append(g)
    return out


def mv_distribution(family, g):
    """The sd distribution of a density family's group."""
    S, a = sd.stats, sd.asarray
    if family == "mvnormal":
        return S.MultivariateNormal(mu=a(g["mu"]), cov=a(g["cov"]))
    if family == "mvt":
        return S.MultivariateT(mu=a(g["mu"]), shape=a(g["cov"]), df=float(g["df"]))
    if family == "wishart":
        return S.Wishart(df=float(g["df"]), scale=a(g["scale"]))
    if family == "invwishart":
        return S.InverseWishart(df=float(g["df"]), scale=a(g["scale"]))
    if family == "matnormal":
        return S.MatrixNormal(mu=a(g["mu"]), rowcov=a(g["rowcov"]), colcov=a(g["colcov"]))
    if family == "multinomial":
        return S.Multinomial(n=float(g["n"]), p=a(g["p"]))
    return S.Dirichlet(alpha=a(g["alpha"]))


def mv_values(family, method, g):
    """sd's values of a method at a density group's points (the entropy: one value)."""
    D = mv_distribution(family, g)
    if method == "entropy":
        return np.atleast_1d(np.asarray(D.entropy(), dtype=float))
    return np.atleast_1d(np.asarray(getattr(D, method)(sd.asarray(g["x"])), dtype=float))


def mv_cdf(name):
    """A cdf set's inputs by name (h, and k, r, nu, lam, d as the set has them) and hi, lo."""
    r = reffile.read(MV / f"{name}.npz")
    out = dict(r.inputs)
    out["hi"], out["lo"] = r.methods["cdf"].hi, r.methods["cdf"].lo
    return out


# The contracts of the multivariate references (docs/api/stats.md §3, docs/design/stats.md), per
# point: each function returns {region: (mask, error, bound)}, the points of the region, their
# error in the contract's measure and the bound it is held to (error <= bound passes). The
# accuracy measurement and the tests judge with these.

MV_EPS = 2.0**-52
T_EXACT_REL = {2: 5e-15, 3: 1e-14}  # the exact t cdfs (bvt, tvt; the one-factor t at d <= 3)


def mv_density_contract(method, got, hi, lo, logpdf_hi=None):
    """A density family's method: |error| <= 1e-14 max(1, |value|) for the log densities and
    entropies (the mixed measure); for pdf, that error carried through exp, relative error <=
    1e-14 max(1, |logpdf|), at the points where the density is a normal double (|logpdf| < 700)."""
    got = np.asarray(got, dtype=float)
    with np.errstate(all="ignore"):
        if method == "pdf":
            mask = np.abs(logpdf_hi) < 700
            err = _special.rel_error(got, hi, lo) / np.maximum(1.0, np.abs(logpdf_hi))
        else:
            mask = np.ones(hi.shape, bool)
            err = _special.mixed_error(got, hi, lo)
    return {"guaranteed": (mask, err, GUARANTEED)}


def mv_cdf_eval(cov, x, df=np.inf, atol=1e-6, max_points=0, batch=None):
    """values, errors of P(X <= x) from the native kernel; cov (d, d) or (B, d, d), x (d,) or (M, d)."""
    from stardust._core import stats as C

    cov = np.asarray(cov, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if cov.ndim == 2:
        cov = cov[None]
    if batch is None:
        batch = np.zeros(len(x), np.int64) if len(cov) == 1 else np.arange(len(x), dtype=np.int64)
    df = np.broadcast_to(np.asarray(df, dtype=float), (len(cov),)).copy()
    v, e = C.mv_cdf(
        sd.asarray(cov), sd.asarray(x), sd.asarray(np.asarray(batch, np.int64)), sd.asarray(df), atol, max_points
    )
    return np.asarray(v), np.asarray(e)


def corr(R3):
    """(M, 3) correlations (r12, r13, r23) as (M, 3, 3) matrices."""
    M = len(R3)
    S = np.tile(np.eye(3), (M, 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        S[:, a, b] = S[:, b, a] = R3[:, q]
    return S


def cdf_rel(v, hi, lo):
    """|v - (hi + lo)| / hi (the cdfs are positive)."""
    return np.abs((v - hi) - lo) / hi


def tvn_cond(h, R):
    """The trivariate normal's condition number at each point: the sum over the six inputs of
    |d log P / d log x|, by central differences of the kernel's own values (inf where a
    perturbed value underflows)."""
    st_ = 1e-6
    c = np.zeros(len(h))
    for i in range(6):
        Hp, Hm, Rp, Rm = h.copy(), h.copy(), R.copy(), R.copy()
        if i < 3:
            Hp[:, i] *= 1 + st_
            Hm[:, i] *= 1 - st_
        else:
            Rp[:, i - 3] *= 1 + st_
            Rm[:, i - 3] *= 1 - st_
        with np.errstate(divide="ignore", invalid="ignore"):
            c += np.abs((np.log(mv_cdf_eval(corr(Rp), Hp)[0]) - np.log(mv_cdf_eval(corr(Rm), Hm)[0])) / (2 * st_))
    return c


def _exact_cdf_regions(v, e, hi, lo, guaranteed, loose):
    """guaranteed (relative 1e-14), loose (the rest of the normal range: loose(rel, mask) in
    units of its bound 1) and tiny (P < 1e-300: absolute 1e-300); an exact path reports an
    error estimate of 0, any other estimate fails."""
    with np.errstate(all="ignore"):
        rel = cdf_rel(v, hi, lo)
        est = np.where(e == 0, 0.0, np.inf)
        norm = hi >= 1e-300
        g = norm & guaranteed
        m = norm & ~g
        out = {"guaranteed": (g, np.maximum(rel, est), GUARANTEED)}
        lo_err = np.zeros(hi.shape)
        lo_err[m] = loose(rel[m], m)
        out["loose"] = (m, np.maximum(lo_err, est), 1.0)
        out["tiny"] = (~norm, np.abs(v - hi), 1e-300)
    return out


def bvn_contract(v, e, hi, lo):
    """Phi2: relative 1e-14 down to P = 1e-15; below, 10 eps max(1, -ln P) (the rounding of the
    exponent q ~ -ln P inside exp); below 1e-300, absolute 1e-300."""
    with np.errstate(divide="ignore"):
        lnp = -np.log(hi)
    return _exact_cdf_regions(v, e, hi, lo, hi >= 1e-15, lambda rel, m: rel / (10 * MV_EPS * np.maximum(1.0, lnp[m])))


def tvn_guaranteed(hi, R, cond):
    """The trivariate normal's guaranteed region: P >= 1e-15 with at most one negative
    correlation, or P >= 1e-10 with a condition number <= 100."""
    negs = (R < 0).sum(axis=1)
    return ((hi >= 1e-15) & (negs <= 1)) | ((hi >= 1e-10) & (cond <= 100))


def tvn_contract(v, e, hi, lo, R, cond):
    """Phi3: relative 1e-14 in tvn_guaranteed; elsewhere within 8 u cond of the inputs'
    rounding where cond is finite (else unbounded); below 1e-300, absolute 1e-300."""

    def loose(rel, m):
        c = cond[m]
        return np.where(np.isfinite(c), rel / (8 * 2.0**-53 * np.maximum(c, 1.0)), 0.0)

    return _exact_cdf_regions(v, e, hi, lo, tvn_guaranteed(hi, R, cond), loose)


def t_exact_contract(d, v, e, hi, lo):
    """The exact t cdf at d = 2, 3 (the scale mixture): relative T_EXACT_REL[d] at every point."""
    with np.errstate(all="ignore"):
        err = np.maximum(cdf_rel(v, hi, lo), np.where(e == 0, 0.0, np.inf))
    return {"guaranteed": (np.ones(hi.shape, bool), err, T_EXACT_REL[d])}


def qmc_contract(err, est, atol, slack):
    """A QMC result against a reference: the estimate <= atol, the actual error <= 1.5 atol and
    <= 2 estimate + slack (the estimate is statistical: 3 standard errors over 12 shifts), as
    one ratio to 1."""
    with np.errstate(all="ignore"):
        return np.maximum.reduce([est / atol, err / (1.5 * atol), err / (2 * est + slack)])
