"""The comparator libraries of the accuracy tables (python -m bench.accuracy run --compare).

Each area names, per sweep, the libraries that compute the same quantity (bench/speed's set:
numpy, scipy, statsmodels, scikit-learn, and R through rsession), as callables over the sweep's
own inputs. Only the same function counts: where a library has none, the sweep has no entry
for it and the table shows "—". A library's points where it returns NaN or inf while the
reference is finite (outside its definition, or a failure) are left out of its measures and
counted (the "제외" column).
"""

from __future__ import annotations

import importlib.metadata as md

import numpy as np

from . import rsession

LIBS = ("scipy", "numpy", "statsmodels", "scikit-learn")


def versions() -> dict:
    out = {}
    for lib in LIBS:
        try:
            out[lib] = md.version(lib)
        except md.PackageNotFoundError:
            out[lib] = None
    out["R"] = rsession.version()
    if out["R"]:
        for pkg in ("mvtnorm", "reticulate", "minpack.lm", "car", "lmtest", "leaps", "irlba"):
            out[f"R {pkg}"] = _r_package_version(pkg)
    from . import boost

    out["Boost"] = boost.version()
    out["Boost in sd.special"] = boost.vendored()
    return out


def _r_package_version(pkg):
    import subprocess

    r = subprocess.run(
        ["Rscript", "--vanilla", "-e", f'cat(as.character(packageVersion("{pkg}")))'], capture_output=True, text=True
    )
    return r.stdout.strip() or None


def _boost(name):
    """A Boost.Math comparator (boost.py) by its name in boost_comparator.cpp; None if unknown."""
    from . import boost

    if not boost.available():
        return "not run"
    known = boost.names()
    if name not in known and name.split(":")[0] + ":" not in known:
        return None

    def fn(xs):
        return boost.evaluate(name, *xs)

    return fn


def _r(expr, packages=()):
    """An R comparator over the sweep's inputs (a, b, c, ... in order); None without R."""
    if not rsession.available():
        return "not run"

    def fn(xs):
        return rsession.evaluate(expr, *xs, packages=packages)

    return fn


# ------------------------------------------------------------------------------ special
# R's expressions over sd's argument order (a, b, c, d = the sweep's inputs)
R_SPECIAL = {
    "gamma": "gamma(a)", "gammaln": "lgamma(a)", "beta": "beta(a, b)", "betaln": "lbeta(a, b)",
    "digamma": "digamma(a)", "polygamma": "psigamma(b, deriv = a)", "binom": "choose(a, b)",
    "gammainc": "pgamma(b, a)", "gammaincc": "pgamma(b, a, lower.tail = FALSE)",
    "gammaincinv": "qgamma(b, a)", "gammainccinv": "qgamma(b, a, lower.tail = FALSE)",
    "betainc": "pbeta(c, a, b)", "betaincc": "pbeta(c, a, b, lower.tail = FALSE)",
    "betaincinv": "qbeta(c, a, b)", "betainccinv": "qbeta(c, a, b, lower.tail = FALSE)",
    "ndtr": "pnorm(a)", "log_ndtr": "pnorm(a, log.p = TRUE)", "ndtri": "qnorm(a)",
    "ndtri_exp": "qnorm(a, log.p = TRUE)",
    "stdtr": "pt(b, a)", "stdtrit": "qt(b, a)",
    "chdtr": "pchisq(b, a)", "chdtrc": "pchisq(b, a, lower.tail = FALSE)",
    "chdtri": "qchisq(b, a, lower.tail = FALSE)",
    "fdtr": "pf(c, a, b)", "fdtrc": "pf(c, a, b, lower.tail = FALSE)", "fdtri": "qf(c, a, b)",
    "bdtr": "pbinom(a, b, c)", "bdtrc": "pbinom(a, b, c, lower.tail = FALSE)",
    "nbdtr": "pnbinom(a, b, c)", "nbdtrc": "pnbinom(a, b, c, lower.tail = FALSE)",
    "pdtr": "ppois(a, b)", "pdtrc": "ppois(a, b, lower.tail = FALSE)",
    "chndtr": "pchisq(c, a, ncp = b)", "chndtrc": "pchisq(c, a, ncp = b, lower.tail = FALSE)",
    "chndtrix": "qchisq(c, a, ncp = b)",
    "ncfdtr": "pf(d, a, b, ncp = c)", "ncfdtrc": "pf(d, a, b, ncp = c, lower.tail = FALSE)",
    "ncfdtri": "qf(d, a, b, ncp = c)",
    "nctdtr": "pt(c, a, ncp = b)", "nctdtrc": "pt(c, a, ncp = b, lower.tail = FALSE)",
    "nctdtrit": "qt(c, a, ncp = b)",
    "expit": "plogis(a)", "logit": "qlogis(a)", "log_expit": "plogis(a, log.p = TRUE)",
    "jv": "besselJ(b, a)", "yv": "besselY(b, a)", "iv": "besselI(b, a)",
    "ive": "besselI(b, a, expon.scaled = TRUE)", "kv": "besselK(b, a)",
    "kve": "besselK(b, a, expon.scaled = TRUE)",
    "j0": "besselJ(a, 0)", "j1": "besselJ(a, 1)", "y0": "besselY(a, 0)", "y1": "besselY(a, 1)",
    "i0": "besselI(a, 0)", "i1": "besselI(a, 1)", "i0e": "besselI(a, 0, expon.scaled = TRUE)",
    "i1e": "besselI(a, 1, expon.scaled = TRUE)", "k0": "besselK(a, 0)", "k1": "besselK(a, 1)",
    "k0e": "besselK(a, 0, expon.scaled = TRUE)", "k1e": "besselK(a, 1, expon.scaled = TRUE)",
}  # fmt: skip


# The functions whose scipy counterpart takes sd's arguments in another order, or exists only as
# a scipy.stats upper tail: name -> (the dotted scipy callable, the order: its k-th argument is
# sd's argument ORDER[k]). The speed measurement (bench/speed) calls the same counterparts.
SCIPY_REORDERED = {
    "chndtr": ("scipy.special.chndtr", (2, 0, 1)),  # chndtr(x, df, nc); sd's chndtr(df, nc, x)
    "chndtrix": ("scipy.special.chndtrix", (2, 0, 1)),  # chndtrix(p, df, nc)
    "chndtrc": ("scipy.stats.ncx2.sf", (2, 0, 1)),  # sf(x, df, nc)
    "nctdtrc": ("scipy.stats.nct.sf", (2, 0, 1)),  # sf(t, df, nc); sd's nctdtrc(df, nc, t)
    "ncfdtrc": ("scipy.stats.ncf.sf", (3, 0, 1, 2)),  # sf(f, dfn, dfd, nc); sd's ncfdtrc(dfn, dfd, nc, f)
}


def scipy_reordered(name):
    """The scipy counterpart of `name` over sd's argument order, or None if SCIPY_REORDERED has none."""
    import importlib

    if name not in SCIPY_REORDERED:
        return None
    dotted, order = SCIPY_REORDERED[name]
    parts = dotted.split(".")
    obj = importlib.import_module(".".join(parts[:2]))  # scipy.special or scipy.stats
    for a in parts[2:]:
        obj = getattr(obj, a)
    return lambda *args, f=obj, order=order: f(*(args[k] for k in order))


def special(name) -> dict:
    import scipy.special as ss

    from tests.special import refdata
    from tests.special import test_reference as T

    out = {}
    fn = None
    if name in SCIPY_REORDERED:
        f = scipy_reordered(name)
        fn = lambda xs, f=f: f(*xs)  # noqa: E731
    elif name in refdata.KW:
        base, kw = refdata.KW[name]
        f = getattr(ss, base, None)
        fn = f and (lambda xs, f=f, kw=kw: f(*xs, **kw))
    elif name in refdata.PART:
        base, i = refdata.PART[name]
        f = getattr(ss, base, None)
        fn = f and (lambda xs, f=f, i=i: f(*xs)[i])
    else:
        f = getattr(ss, T.ALIAS.get(name, name), None) if not name.startswith("_") else None
        fn = f and (lambda xs, f=f: f(*xs))
    if fn:
        out["scipy"] = fn
    if name in R_SPECIAL:
        out["R"] = _r(R_SPECIAL[name])
    b = _boost(name) if name in BOOST_SPECIAL else None
    if b is not None:
        out["Boost"] = b
    return out


# the names boost_comparator.cpp evaluates (the same function, over sd's argument order)
BOOST_SPECIAL = {
    "gamma", "gammaln", "beta", "digamma", "polygamma", "gammainc", "gammaincc", "gammaincinv", "gammainccinv",
    "betainc", "betaincc", "betaincinv", "betainccinv", "erf", "erfc", "erfinv", "erfcinv", "expi", "expn",
    "jv", "yv", "iv", "kv", "j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1", "airy_ai", "airy_aip", "airy_bi",
    "airy_bip", "lambertw_m1", "zeta", "hyp1f1", "ndtr", "log_ndtr", "ndtri", "expit", "log_expit", "logit",
    "stdtr", "stdtrit", "chdtr", "chdtrc", "chdtri", "fdtr", "fdtrc", "fdtri", "bdtr", "bdtrc", "nbdtr",
    "nbdtrc", "pdtr", "pdtrc", "chndtr", "chndtrc", "chndtrix", "ncfdtr", "ncfdtrc", "ncfdtri", "nctdtr",
    "nctdtrc", "nctdtrit",
}  # fmt: skip
BOOST_CONT = {"pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf", "icdf", "iccdf"}
BOOST_DISC = {"pmf", "logpmf", "cdf", "ccdf", "logcdf", "logccdf"}  # (its discrete quantiles round otherwise)
BOOST_MOMENTS = {"mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode"}
BOOST_DISCRETE_DISTS = {"binomial", "bernoulli", "poisson", "negativebinomial", "geometric"}
# the distributions boost_comparator.cpp and bench/speed/boost_speed.cpp build (as "<dist>:<method>"
# names); a static table, so listing the speed cases builds nothing (a test checks both libraries)
BOOST_DISTS = frozenset({
    "normal", "studentt", "chisquared", "f", "gamma", "beta", "exponential", "lognormal", "uniform", "cauchy",
    "laplace", "logistic", "weibull", "gumbel", "pareto", "rayleigh", "inversegamma", "noncentralchisquared",
    "noncentralt", "noncentralf", "binomial", "bernoulli", "poisson", "negativebinomial", "geometric",
})  # fmt: skip


# ------------------------------------------------------------------------------ stats
# sd's parameters (tests/stats/refdata.CLASSES order) -> scipy.stats' classic frozen distribution
SCIPY_DIST = {
    "binomial": lambda st, n, p: st.binom(n, p),
    "bernoulli": lambda st, p: st.bernoulli(p),
    "poisson": lambda st, mu: st.poisson(mu),
    "geometric": lambda st, p: st.geom(p),
    "negativebinomial": lambda st, n, p: st.nbinom(n, p),
    "hypergeometric": lambda st, total, succ, draws: st.hypergeom(total, succ, draws),
    "discreteuniform": lambda st, low, high: st.randint(low, high),
    "betabinomial": lambda st, n, a, b: st.betabinom(n, a, b),
    "logseries": lambda st, p: st.logser(p),
    "zipf": lambda st, a: st.zipf(a),
    "normal": lambda st, mu, s: st.norm(mu, s),
    "exponential": lambda st, scale: st.expon(scale=scale),
    "uniform": lambda st, a, b: st.uniform(a, b - a),
    "laplace": lambda st, mu, s: st.laplace(mu, s),
    "cauchy": lambda st, mu, s: st.cauchy(mu, s),
    "lognormal": lambda st, mu, s: st.lognorm(s, scale=np.exp(mu)),
    "weibull": lambda st, c, scale: st.weibull_min(c, scale=scale),
    "studentt": lambda st, df: st.t(df),
    "chisquared": lambda st, df: st.chi2(df),
    "f": lambda st, dfn, dfd: st.f(dfn, dfd),
    "gamma": lambda st, a, scale: st.gamma(a, scale=scale),
    "beta": lambda st, a, b: st.beta(a, b),
    "logistic": lambda st, mu, s: st.logistic(mu, s),
    "gumbel": lambda st, mu, s: st.gumbel_r(mu, s),
    "gev": lambda st, c, mu, s: st.genextreme(c, mu, s),
    "pareto": lambda st, b, scale: st.pareto(b, scale=scale),
    "rayleigh": lambda st, scale: st.rayleigh(scale=scale),
    "inversegamma": lambda st, a, scale: st.invgamma(a, scale=scale),
    "noncentralchisquared": lambda st, df, nc: st.ncx2(df, nc),
    "noncentralt": lambda st, df, nc: st.nct(df, nc),
    "noncentralf": lambda st, dfn, dfd, nc: st.ncf(dfn, dfd, nc),
}
SCIPY_METHOD = {
    "pdf": "pdf", "logpdf": "logpdf", "pmf": "pmf", "logpmf": "logpmf", "cdf": "cdf", "logcdf": "logcdf",
    "ccdf": "sf", "logccdf": "logsf", "icdf": "ppf", "iccdf": "isf", "mean": "mean", "median": "median",
    "variance": "var", "standard_deviation": "std", "entropy": "entropy",
}  # fmt: skip
NEW_STYLE = {"ilogcdf", "ilogccdf", "mode", "skewness", "kurtosis"}  # scipy's make_distribution has these

# sd's parameters -> R's d/p/q stem and arguments; shift: x of R = x of sd - shift (R's geometric counts failures)
R_DIST = {
    "binomial": ("binom", "size = {0}, prob = {1}"),
    "bernoulli": ("binom", "size = 1, prob = {0}"),
    "poisson": ("pois", "lambda = {0}"),
    "geometric": ("geom", "prob = {0}"),
    "negativebinomial": ("nbinom", "size = {0}, prob = {1}"),
    "hypergeometric": ("hyper", "m = {1}, n = {0} - {1}, k = {2}"),
    "normal": ("norm", "mean = {0}, sd = {1}"),
    "exponential": ("exp", "rate = 1 / {0}"),
    "uniform": ("unif", "min = {0}, max = {1}"),
    "cauchy": ("cauchy", "location = {0}, scale = {1}"),
    "lognormal": ("lnorm", "meanlog = {0}, sdlog = {1}"),
    "weibull": ("weibull", "shape = {0}, scale = {1}"),
    "studentt": ("t", "df = {0}"),
    "chisquared": ("chisq", "df = {0}"),
    "f": ("f", "df1 = {0}, df2 = {1}"),
    "gamma": ("gamma", "shape = {0}, scale = {1}"),
    "beta": ("beta", "shape1 = {0}, shape2 = {1}"),
    "logistic": ("logis", "location = {0}, scale = {1}"),
    "noncentralchisquared": ("chisq", "df = {0}, ncp = {1}"),
    "noncentralt": ("t", "df = {0}, ncp = {1}"),
    "noncentralf": ("f", "df1 = {0}, df2 = {1}, ncp = {2}"),
}
R_SHIFT = {"geometric": 1.0}
R_METHOD = {  # (prefix, extra arguments, is the input an x (shifted) or a probability)
    "pdf": ("d", "", "x"), "pmf": ("d", "", "x"), "logpdf": ("d", ", log = TRUE", "x"),
    "logpmf": ("d", ", log = TRUE", "x"), "cdf": ("p", "", "x"), "logcdf": ("p", ", log.p = TRUE", "x"),
    "ccdf": ("p", ", lower.tail = FALSE", "x"), "logccdf": ("p", ", lower.tail = FALSE, log.p = TRUE", "x"),
    "icdf": ("q", "", "p"), "iccdf": ("q", ", lower.tail = FALSE", "p"),
    "ilogcdf": ("q", ", log.p = TRUE", "p"), "ilogccdf": ("q", ", lower.tail = FALSE, log.p = TRUE", "p"),
}  # fmt: skip


def stats(name) -> dict:
    import stardust as sd
    from tests.stats import refdata

    dist, method = refdata.split(name)
    npar = len(getattr(sd.stats, refdata.CLASSES[dist])._params)
    out = {}
    if dist in SCIPY_DIST and (method in SCIPY_METHOD or method in NEW_STYLE):
        out["scipy"] = lambda xs, d=dist, m=method: _scipy_stats(d, m, xs, npar)
    if dist in R_DIST and method in R_METHOD:
        stem, args = R_DIST[dist]
        pre, extra, kind = R_METHOD[method]
        names = "abcdefgh"
        params = args.format(*names[:npar])
        x = names[npar]
        shift = R_SHIFT.get(dist, 0.0)
        if kind == "x" and shift:
            x = f"({x} - {shift})"
        expr = f"{pre}{stem}({x}, {params}{extra})"
        if kind == "p" and shift:
            expr = f"({expr} + {shift})"
        out["R"] = _r(expr)
    ok = BOOST_MOMENTS | (BOOST_DISC if dist in BOOST_DISCRETE_DISTS else BOOST_CONT)
    b = _boost(f"{dist}:{method}") if method in ok else None
    if b is not None:
        out["Boost"] = b
    return out


def _scipy_stats(dist, method, xs, nparams):
    import scipy.stats as st

    params, rest = xs[:nparams], xs[nparams:]
    with np.errstate(all="ignore"):
        if method in NEW_STYLE:
            return _scipy_new(st, dist, method, params, rest)
        D = SCIPY_DIST[dist](st, *params)
        if method in ("mean", "median", "variance", "standard_deviation", "entropy"):
            return np.asarray(getattr(D, SCIPY_METHOD[method])(), dtype=float)
        return np.asarray(getattr(D, SCIPY_METHOD[method])(rest[0]), dtype=float)


def _scipy_new(st, dist, method, params, rest):
    """make_distribution's ilogcdf, ilogccdf, mode, skewness, kurtosis (excess), point by point
    over the parameter sets (a new-style distribution takes array parameters too, but its
    loc/scale come from the classic one's keywords)."""
    classic = SCIPY_DIST[dist](st, *params)
    rv, shapes, kw = classic.dist, classic.args, classic.kwds
    D = st.make_distribution(rv)
    names = [p.name for p in (rv._shape_info() or [])]
    X = D(**dict(zip(names, shapes))) if names else D()
    loc, scale = kw.get("loc", 0.0), kw.get("scale", 1.0)
    if np.any(np.asarray(scale) != 1.0):
        X = X * scale
    if np.any(np.asarray(loc) != 0.0):
        X = X + loc
    if method == "kurtosis":
        return np.asarray(X.kurtosis(convention="excess"), dtype=float)
    if method in ("mode", "skewness"):
        return np.asarray(getattr(X, method)(), dtype=float)
    return np.asarray(getattr(X, method)(rest[0]), dtype=float)
