"""sd.stats against scipy.stats and numpy's Generator. From bench/stats/t4/speed.py
(continuous), t5/speed.py (discrete: small supports and the "wide" set), t6/speed.py
(extreme value, lifetime, noncentral), t7/speed.py (sample vs numpy), t8/speed.py
(descriptive), t9/speed.py and t9/cdf_speed.py (multivariate; mvn / mvt cdf) and
t10/speed.py (Wishart, matrix normal).

The distribution methods (t4-t6) take 10^5 points spread over the distribution by its
quantiles: scipy's own ppf for the continuous ones (t4), sd's icdf for the discrete ones
(scipy's zipf ppf stops with "endless loop") and for t6 (scipy's noncentral ppf takes
minutes for 10^5). scipy is timed in its classic form (scipy.stats.<name>.<method> on a
frozen distribution) with the matching names (sf for ccdf, ppf for icdf, isf for iccdf),
and in its new form (scipy.stats.Normal, Uniform, Logistic, Binomial: NEW) where the class
exists; the scalar cdf rows also against one scipy.special function where there is one.

Every array case of a distribution feeds csrc/stats/cost_table.hpp with the tag
"<class>.<method>", the geometric mean of its cases: the wide discrete set, the continuous
other-parameter group (OTHER) and the rows above. The small-support discrete rows (where the engine's
memo applies) have their own key "<class>.<method>@memo".
The parameter-array and mid-size groups are not in the table.

The wide discrete set (nearly every k different) takes 20,000 points (2,000 for the
hypergeometric and beta-binomial, whose cdf sums run to thousands of terms). Where
scipy's call over all of them would take more than about 2 s, the old script timed it
on the first N/50 points: those comparators carry that n (SMALL_SAMPLED and OTHER_SAMPLED
do the same with 2,000 points). scipy is left out where it fails or does not return (zipf
ppf "endless loop", zipf cdf at k ~ 1e20, logser ppf).

The scalar groups time one Python float per call (cdf, a few logpdf and icdf; one sample
in t7), one thread. The small inputs (10^3, 10^4 points; 100 for the descriptive
statistics) rotate fresh inputs (harness.rotate), so the branch predictor cannot learn one.
"""

from __future__ import annotations

import math

from ..harness import fresh_count, rotate
from ..registry import Case, Ref

N = 100_000

SCIPY = {
    "pdf": "pdf",
    "logpdf": "logpdf",
    "pmf": "pmf",
    "logpmf": "logpmf",
    "cdf": "cdf",
    "logcdf": "logcdf",
    "ccdf": "sf",
    "logccdf": "logsf",
    "icdf": "ppf",
    "iccdf": "isf",
}
CONT = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf")
DISC = ("pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf")

# (class, parameters, scipy frozen classic distribution from scipy.stats)
T4 = [
    ("Normal", {"mu": 1.5, "sigma": 2.0}, lambda st: st.norm(1.5, 2.0)),
    ("StudentT", {"df": 3.5}, lambda st: st.t(3.5)),
    ("ChiSquared", {"df": 4.5}, lambda st: st.chi2(4.5)),
    ("F", {"dfn": 3.0, "dfd": 11.0}, lambda st: st.f(3.0, 11.0)),
    ("Gamma", {"a": 2.5, "scale": 3.0}, lambda st: st.gamma(2.5, scale=3.0)),
    ("Beta", {"a": 2.5, "b": 0.7}, lambda st: st.beta(2.5, 0.7)),
    ("Exponential", {"scale": 2.0}, lambda st: st.expon(scale=2.0)),
    ("LogNormal", {"mu": 0.5, "sigma": 0.8}, lambda st: st.lognorm(0.8, scale=math.exp(0.5))),
    ("Uniform", {"a": -1.0, "b": 3.0}, lambda st: st.uniform(-1.0, 4.0)),
    ("Cauchy", {"mu": 0.5, "sigma": 2.0}, lambda st: st.cauchy(0.5, 2.0)),
    ("Laplace", {"mu": 0.5, "sigma": 2.0}, lambda st: st.laplace(0.5, 2.0)),
    ("Weibull", {"c": 1.7, "scale": 2.0}, lambda st: st.weibull_min(1.7, scale=2.0)),
]

T5 = [
    ("Binomial", {"n": 40, "p": 0.3}, lambda st: st.binom(40, 0.3)),
    ("Poisson", {"mu": 4.5}, lambda st: st.poisson(4.5)),
    ("Geometric", {"p": 0.2}, lambda st: st.geom(0.2)),
    ("NegativeBinomial", {"n": 2.5, "p": 0.3}, lambda st: st.nbinom(2.5, 0.3)),
    ("Hypergeometric", {"total": 500, "successes": 120, "draws": 200}, lambda st: st.hypergeom(500, 120, 200)),
    ("Bernoulli", {"p": 0.3}, lambda st: st.bernoulli(0.3)),
    ("DiscreteUniform", {"low": 2, "high": 90}, lambda st: st.randint(2, 90)),
    ("BetaBinomial", {"n": 40, "a": 2.5, "b": 1.7}, lambda st: st.betabinom(40, 2.5, 1.7)),
    ("LogSeries", {"p": 0.7}, lambda st: st.logser(0.7)),
    ("Zipf", {"a": 2.5}, lambda st: st.zipf(2.5)),
]
T5_NO_SCIPY = {("Zipf", "icdf"), ("Zipf", "iccdf")}  # scipy's zipf ppf: "updating stopped, endless loop"

T5_WIDE = [
    ("Binomial", {"n": 10**9, "p": 0.3}, lambda st: st.binom(10**9, 0.3)),
    ("Poisson", {"mu": 3e8}, lambda st: st.poisson(3e8)),
    ("Geometric", {"p": 1e-6}, lambda st: st.geom(1e-6)),
    ("NegativeBinomial", {"n": 2.5e6, "p": 1e-3}, lambda st: st.nbinom(2.5e6, 1e-3)),
    (
        "Hypergeometric",
        {"total": 5 * 10**6, "successes": 12 * 10**5, "draws": 2 * 10**6},
        lambda st: st.hypergeom(5 * 10**6, 12 * 10**5, 2 * 10**6),
    ),
    ("DiscreteUniform", {"low": 2, "high": 10**9}, lambda st: st.randint(2, 10**9)),
    ("BetaBinomial", {"n": 20000, "a": 2.5, "b": 1.7}, lambda st: st.betabinom(20000, 2.5, 1.7)),
    ("LogSeries", {"p": 1 - 1e-7}, lambda st: st.logser(1 - 1e-7)),
    ("Zipf", {"a": 1.05}, lambda st: st.zipf(1.05)),
]
WIDE_SHORT = ("Hypergeometric", "BetaBinomial")  # 2,000 points instead of 20,000
# scipy's rows over 2 s for all the points: the old script timed them on the first N/50
WIDE_SAMPLED = {
    ("Poisson", "icdf"),
    ("Poisson", "iccdf"),
    ("Hypergeometric", "logcdf"),
    ("Hypergeometric", "logccdf"),
    ("BetaBinomial", "icdf"),
    ("BetaBinomial", "iccdf"),
    ("LogSeries", "cdf"),
    ("LogSeries", "logcdf"),
}
WIDE_NO_SCIPY = (
    {("LogSeries", "icdf"), ("LogSeries", "iccdf")}  # scipy's logser ppf fails
    | {("Zipf", m) for m in DISC if m not in ("pmf", "logpmf")}
)  # zipf cdf at k ~ 1e20 does not return

T6 = [
    ("Logistic", {"mu": 0.5, "sigma": 2.0}, lambda st: st.logistic(0.5, 2.0)),
    ("Gumbel", {"mu": 0.5, "sigma": 2.0}, lambda st: st.gumbel_r(0.5, 2.0)),
    ("GEV", {"c": 0.25, "mu": 1.0, "sigma": 2.0}, lambda st: st.genextreme(0.25, 1.0, 2.0)),
    ("Pareto", {"b": 2.5, "scale": 1.5}, lambda st: st.pareto(2.5, scale=1.5)),
    ("Rayleigh", {"scale": 2.0}, lambda st: st.rayleigh(scale=2.0)),
    ("InverseGamma", {"a": 4.5, "scale": 2.0}, lambda st: st.invgamma(4.5, scale=2.0)),
    ("NoncentralChiSquared", {"df": 3.5, "nc": 2.0}, lambda st: st.ncx2(3.5, 2.0)),
    ("NoncentralT", {"df": 9.0, "nc": 1.5}, lambda st: st.nct(9.0, 1.5)),
    ("NoncentralF", {"dfn": 4.0, "dfd": 13.0, "nc": 3.0}, lambda st: st.ncf(4.0, 13.0, 3.0)),
]

QUICK_DIST = {
    ("연속 분포", "Normal.cdf"),
    ("연속 분포", "Normal.icdf"),
    ("연속 분포", "Gamma.icdf"),
    ("연속 분포", "StudentT.cdf"),
    ("이산 분포", "Binomial.cdf"),
    ("이산 분포(넓은 지지)", "Poisson.cdf"),
    ("극값·수명·비중심 분포", "NoncentralT.cdf"),
    ("연속 분포(스칼라)", "Normal.cdf"),
    ("이산 분포", "Zipf.icdf"),
    ("이산 분포(넓은 지지)", "Hypergeometric.cdf"),
    ("이산 분포(넓은 지지)", "BetaBinomial.icdf"),
    ("연속 분포(다른 매개변수)", "Gamma(a=0.3).cdf"),
    ("매개변수 배열", "Binomial.cdf"),
}

# The inverses of the log probabilities, for a subset (their cost keys "<class>.ilogcdf").
ILOG = ("ilogcdf", "ilogccdf")
ILOG_DISTS = {"Normal", "Gamma", "Binomial", "Poisson"}

# scipy's new-style distributions (scipy.stats.Normal, ...), built from sd's parameters
# (scalars or arrays): a second comparator column wherever the class exists.
NEW_LABEL = "scipy 새 방식"
NEW = {
    "Normal": lambda st, kw: st.Normal(mu=kw["mu"], sigma=kw["sigma"]),
    "Uniform": lambda st, kw: st.Uniform(a=kw["a"], b=kw["b"]),
    "Logistic": lambda st, kw: kw["sigma"] * st.Logistic() + kw["mu"],
    "Binomial": lambda st, kw: st.Binomial(n=kw["n"], p=kw["p"]),
}

# The scalar cdf as one plain scipy.special function, where there is one (the
# negative binomial's nbdtr rounds a non-integer n down, so it is left out).
SPECIAL_LABEL = "scipy.special"
SPECIAL_CDF = {
    "Normal": lambda sc, kw: lambda v: sc.ndtr((v - kw["mu"]) / kw["sigma"]),
    "LogNormal": lambda sc, kw: lambda v: sc.ndtr((math.log(v) - kw["mu"]) / kw["sigma"]),
    "StudentT": lambda sc, kw: lambda v: sc.stdtr(kw["df"], v),
    "ChiSquared": lambda sc, kw: lambda v: sc.chdtr(kw["df"], v),
    "F": lambda sc, kw: lambda v: sc.fdtr(kw["dfn"], kw["dfd"], v),
    "Gamma": lambda sc, kw: lambda v: sc.gdtr(1.0 / kw.get("scale", 1.0), kw["a"], v),
    "Beta": lambda sc, kw: lambda v: sc.betainc(kw["a"], kw["b"], v),
    "Logistic": lambda sc, kw: lambda v: sc.expit((v - kw["mu"]) / kw["sigma"]),
    "InverseGamma": lambda sc, kw: lambda v: sc.gammaincc(kw["a"], kw.get("scale", 1.0) / v),
    "NoncentralChiSquared": lambda sc, kw: lambda v: sc.chndtr(v, kw["df"], kw["nc"]),
    "NoncentralT": lambda sc, kw: lambda v: sc.nctdtr(kw["df"], kw["nc"], v),
    "NoncentralF": lambda sc, kw: lambda v: sc.ncfdtr(kw["dfn"], kw["dfd"], kw["nc"], v),
    "Binomial": lambda sc, kw: lambda v: sc.bdtr(v, kw["n"], kw["p"]),
    "Poisson": lambda sc, kw: lambda v: sc.pdtr(v, kw["mu"]),
}

# More scalar methods than the cdf for a few distributions.
SCALAR_MORE = {
    "Normal": ("logpdf", "icdf"),
    "Gamma": ("logpdf", "icdf"),
    "Beta": ("logpdf", "icdf"),
    "Binomial": ("logpmf", "icdf"),
    "Poisson": ("logpmf", "icdf"),
}

# Small-support rows whose scipy call takes seconds over all N points: timed on the first
# SAMPLE_N points, as WIDE_SAMPLED.
SAMPLE_N = 2000
SMALL_SAMPLED = {
    ("Hypergeometric", "logcdf"),
    ("Hypergeometric", "logccdf"),
    ("Hypergeometric", "icdf"),
    ("Hypergeometric", "iccdf"),
    ("BetaBinomial", "icdf"),
    ("BetaBinomial", "iccdf"),
    ("LogSeries", "icdf"),
    ("LogSeries", "iccdf"),
}

# Other parameters of the continuous distributions (the bulk and the edges of a shape):
# cdf, icdf, logpdf on N points from sd's icdf. They feed the same cost keys as the rows
# above, so the table's cost is the geometric mean over the parameter sets.
OTHER = [
    ("Gamma", {"a": 2.3}, lambda st: st.gamma(2.3)),
    ("Gamma", {"a": 0.3}, lambda st: st.gamma(0.3)),
    ("Gamma", {"a": 50.0}, lambda st: st.gamma(50.0)),
    ("InverseGamma", {"a": 4.3}, lambda st: st.invgamma(4.3)),
    ("StudentT", {"df": 30.0}, lambda st: st.t(30.0)),
    ("Beta", {"a": 0.5, "b": 0.5}, lambda st: st.beta(0.5, 0.5)),
    ("Beta", {"a": 20.0, "b": 30.0}, lambda st: st.beta(20.0, 30.0)),
    ("F", {"dfn": 5.0, "dfd": 40.0}, lambda st: st.f(5.0, 40.0)),
    ("NoncentralT", {"df": 9.0, "nc": 20.0}, lambda st: st.nct(9.0, 20.0)),
    ("NoncentralChiSquared", {"df": 3.5, "nc": 200.0}, lambda st: st.ncx2(3.5, 200.0)),
]
OTHER_METHODS = ("cdf", "icdf", "logpdf")
OTHER_SAMPLED = {("NoncentralT", "icdf"), ("NoncentralChiSquared", "icdf")}  # scipy's noncentral ppf: seconds


def _label(name, params):
    return f"{name}({', '.join(f'{k}={v:g}' for k, v in params.items())})" if params else name


# ---------------------------------------------------------------- t4-t6: distribution methods


def _dist_data(name, params, scipy_ref, p, points):
    """The case data for points p (uniform in (0, 1)): points "scipy" (the reference's own
    ppf, t4) or "sd" (sd's icdf, t5, t6 and the other groups)."""
    import numpy as np
    import scipy.special as sc
    import scipy.stats as st

    import stardust as sd

    X = getattr(sd.stats, name)(**params)
    ref = scipy_ref(st)
    x = ref.ppf(p) if points == "scipy" else np.asarray(X.icdf(sd.asarray(p)))
    lp = np.log(p)
    n = len(p)
    return {
        "X": X,
        "ref": ref,
        "new": NEW[name](st, params) if name in NEW else None,
        "special": SPECIAL_CDF[name](sc, params) if name in SPECIAL_CDF else None,
        "p": p,
        "x": x,
        "lp": lp,
        "sp": sd.asarray(p),
        "sx": sd.asarray(x),
        "slp": sd.asarray(lp),
        "v": float(x[n // 3]),
        "pv": float(p[n // 3]),
    }


def _dist_setup(name, params, scipy_ref, points):
    def setup():
        import numpy as np

        return _dist_data(name, params, scipy_ref, np.random.default_rng(0).uniform(0, 1, N), points)

    return setup


def _wide_setup(index):
    name, params, scipy_ref = T5_WIDE[index]

    def setup():
        import numpy as np

        rng = np.random.default_rng(0)  # one generator over the distributions in order, as the old script
        for nm, _, _ in T5_WIDE[: index + 1]:
            p = rng.uniform(0, 1, _wide_n(nm))
        return _dist_data(name, params, scipy_ref, p, "sd")

    return setup


def _wide_n(name):
    return 2000 if name in WIDE_SHORT else 20_000


def _arg_keys(m):
    """(sd's data key, numpy's data key) of method m's argument: the points, p for the
    inverses, log p for the inverses of the log probabilities."""
    if m in ILOG:
        return "slp", "lp"
    if m in ("icdf", "iccdf"):
        return "sp", "p"
    return "sx", "x"


def _sd_call(m):
    key = _arg_keys(m)[0]
    return lambda d: lambda f=getattr(d["X"], m), a=d[key]: f(a)


def _ref_call(obj, m, k=None):
    """obj's method m (scipy's classic name for obj "ref") on the case data, on its first k
    points when k is given (sliced outside the timing)."""
    key = _arg_keys(m)[1]
    meth = SCIPY[m] if obj == "ref" else m

    def make(d):
        a = d[key] if k is None else d[key][:k].copy()
        return lambda g=getattr(d[obj], meth), a=a: g(a)

    return make


def _refs(name, m, *, no_scipy=False, sampled=None, new_sampled=None):
    """scipy's classic call (not for the ilog methods, which it has not) and the new-style
    one where the class exists; sampled / new_sampled: their point counts when timed on a
    sample."""
    refs = []
    if not no_scipy and m not in ILOG:
        refs.append(Ref("scipy", _ref_call("ref", m, sampled), n=sampled))
    if name in NEW:
        refs.append(Ref(NEW_LABEL, _ref_call("new", m, new_sampled), n=new_sampled, lib="scipy"))
    return tuple(refs)


def _methods(name, methods):
    return methods + ILOG if name in ILOG_DISTS else methods


def _ilog_note(name, m):
    return "scipy에 맞는 함수가 없다" if m in ILOG and name not in NEW else ""


def _dist_cases(group, specs, methods, points, *, no_scipy=frozenset(), sampled=frozenset(), memo=False):
    """memo: the small-support discrete rows, whose cost key is "<class>.<method>@memo"."""
    out = []
    for name, params, scipy_ref in specs:
        setup = _dist_setup(name, params, scipy_ref, points)
        for m in _methods(name, methods):
            k = SAMPLE_N if (name, m) in sampled else None
            note = _ilog_note(name, m)
            if (name, m) in no_scipy:
                note = "scipy의 ppf가 'endless loop'로 멈춘다"
            out.append(
                Case(
                    area="stats",
                    group=group,
                    name=f"{name}.{m}",
                    size="10^5",
                    n=N,
                    setup=setup,
                    sd=_sd_call(m),
                    refs=_refs(name, m, no_scipy=(name, m) in no_scipy, sampled=k),
                    cost=("stats", f"{name.lower()}.{m}" + ("@memo" if memo else "")),
                    quick=(group, f"{name}.{m}") in QUICK_DIST,
                    note=note,
                )
            )
        out += _scalar_cases(f"{group}(스칼라)", name, params, setup)
    return out


def _scalar_cases(group, name, params, setup):
    """One Python float per call, one thread: the cdf (with scipy.special where it has the
    function) and, for a few distributions, SCALAR_MORE."""
    out = []
    for m in ("cdf", *SCALAR_MORE.get(name, ())):
        key = "pv" if m in ("icdf", "iccdf") else "v"
        refs = [Ref("scipy", lambda d, m=m, key=key: lambda g=getattr(d["ref"], SCIPY[m]), v=d[key]: g(v))]
        if name in NEW:
            refs.append(
                Ref(NEW_LABEL, lambda d, m=m, key=key: lambda g=getattr(d["new"], m), v=d[key]: g(v), lib="scipy")
            )
        if m == "cdf" and name in SPECIAL_CDF:
            refs.append(Ref(SPECIAL_LABEL, lambda d: lambda g=d["special"], v=d["v"]: g(v), lib="scipy"))
        out.append(
            Case(
                area="stats",
                group=group,
                name=f"{name}.{m}",
                setup=setup,
                sd=lambda d, m=m, key=key: lambda f=getattr(d["X"], m), v=d[key]: f(v),
                refs=tuple(refs),
                threads=(1,),
                quick=(group, f"{name}.{m}") in QUICK_DIST,
            )
        )
    return out


def _wide_cases():
    group = "이산 분포(넓은 지지)"
    out = []
    for i, (name, _, _) in enumerate(T5_WIDE):
        setup = _wide_setup(i)
        n = _wide_n(name)
        for m in _methods(name, DISC):
            no_scipy = (name, m) in WIDE_NO_SCIPY
            note = _ilog_note(name, m)
            if no_scipy:
                note = (
                    "scipy의 ppf가 실패한다"
                    if m in ("icdf", "iccdf")
                    else "scipy의 zipf cdf는 k ~ 1e20에서 끝나지 않는다"
                )
            sampled = max(1, n // 50) if (name, m) in WIDE_SAMPLED else None
            new_sampled = max(1, n // 100) if m in ILOG else None  # scipy.stats.Binomial's ilogcdf: 2 ms a point
            out.append(
                Case(
                    area="stats",
                    group=group,
                    name=f"{name}.{m}",
                    size=f"{n:,}",
                    n=n,
                    setup=setup,
                    sd=_sd_call(m),
                    refs=_refs(name, m, no_scipy=no_scipy, sampled=sampled, new_sampled=new_sampled),
                    cost=("stats", f"{name.lower()}.{m}"),
                    quick=(group, f"{name}.{m}") in QUICK_DIST,
                    note=note,
                )
            )
    return out


def _other_cases():
    group = "연속 분포(다른 매개변수)"
    out = []
    for name, params, scipy_ref in OTHER:
        setup = _dist_setup(name, params, scipy_ref, "sd")
        label = _label(name, params)
        for m in OTHER_METHODS:
            k = SAMPLE_N if (name, m) in OTHER_SAMPLED else None
            out.append(
                Case(
                    area="stats",
                    group=group,
                    name=f"{label}.{m}",
                    size="10^5",
                    n=N,
                    setup=setup,
                    sd=_sd_call(m),
                    refs=_refs(name, m, sampled=k),
                    cost=("stats", f"{name.lower()}.{m}"),
                    quick=(group, f"{label}.{m}") in QUICK_DIST,
                )
            )
    return out


# ---------------------------------------------------------------- parameter arrays

# (class, scipy classic name, rng -> sd's parameters as arrays of N, the scipy frozen
# distribution of those arrays, note); one parameter set per point, the points from sd's icdf.
PARAM_ARRAY = [
    (
        "Normal",
        lambda rng: {"mu": rng.uniform(-2, 2, N), "sigma": rng.uniform(0.5, 3, N)},
        lambda st, kw: st.norm(kw["mu"], kw["sigma"]),
        "mu는 −2–2, sigma는 0.5–3에서 점마다 균등하게 뽑는다",
    ),
    (
        "Gamma",
        lambda rng: {"a": rng.uniform(0.5, 20, N)},
        lambda st, kw: st.gamma(kw["a"]),
        "a는 0.5–20에서 점마다 균등하게 뽑는다",
    ),
    (
        "StudentT",
        lambda rng: {"df": rng.uniform(1, 30, N)},
        lambda st, kw: st.t(kw["df"]),
        "df는 1–30에서 점마다 균등하게 뽑는다",
    ),
    (
        "Beta",
        lambda rng: {"a": rng.uniform(0.5, 20, N), "b": rng.uniform(0.5, 20, N)},
        lambda st, kw: st.beta(kw["a"], kw["b"]),
        "a, b는 0.5–20에서 점마다 균등하게 뽑는다",
    ),
    (
        "Poisson",
        lambda rng: {"mu": rng.uniform(0.5, 50, N)},
        lambda st, kw: st.poisson(kw["mu"]),
        "mu는 0.5–50에서 점마다 균등하게 뽑는다",
    ),
    (
        "Binomial",
        lambda rng: {"n": rng.integers(1, 201, N).astype(float), "p": rng.uniform(0.05, 0.95, N)},
        lambda st, kw: st.binom(kw["n"], kw["p"]),
        "n은 1–200의 정수, p는 0.05–0.95에서 점마다 균등하게 뽑는다",
    ),
]
PARAM_ARRAY_METHODS = {
    "Normal": ("logpdf", "cdf"),
    "Gamma": ("logpdf", "cdf"),
    "StudentT": ("cdf",),
    "Beta": ("cdf",),
    "Poisson": ("logpmf", "cdf"),
    "Binomial": ("logpmf", "cdf"),
}


def _param_array_setup(name, draw, scipy_ref):
    def setup():
        import numpy as np
        import scipy.stats as st

        import stardust as sd

        rng = np.random.default_rng(0)
        kw = draw(rng)
        p = rng.uniform(0, 1, N)
        X = getattr(sd.stats, name)(**{k: sd.asarray(v) for k, v in kw.items()})
        x = np.asarray(X.icdf(sd.asarray(p)))
        return {
            "X": X,
            "ref": scipy_ref(st, kw),
            "new": NEW[name](st, kw) if name in NEW else None,
            "x": x,
            "sx": sd.asarray(x),
        }

    return setup


def _param_array_cases():
    group = "매개변수 배열"
    out = []
    for name, draw, scipy_ref, note in PARAM_ARRAY:
        setup = _param_array_setup(name, draw, scipy_ref)
        for m in PARAM_ARRAY_METHODS[name]:
            out.append(
                Case(
                    area="stats",
                    group=group,
                    name=f"{name}.{m}",
                    size="10^5",
                    n=N,
                    setup=setup,
                    sd=_sd_call(m),
                    refs=_refs(name, m),
                    quick=(group, f"{name}.{m}") in QUICK_DIST,
                    note=note,
                )
            )
    return out


# ---------------------------------------------------------------- mid sizes

MID = [
    ("Normal", {"mu": 1.5, "sigma": 2.0}, lambda st: st.norm(1.5, 2.0), "cdf"),
    ("Normal", {"mu": 1.5, "sigma": 2.0}, lambda st: st.norm(1.5, 2.0), "logpdf"),
    ("Gamma", {"a": 2.3}, lambda st: st.gamma(2.3), "cdf"),
    ("Binomial", {"n": 40, "p": 0.3}, lambda st: st.binom(40, 0.3), "cdf"),
    ("StudentT", {"df": 3.5}, lambda st: st.t(3.5), "icdf"),
]
MID_SIZES = ((1000, "10^3"), (10_000, "10^4"))


def _mid_setup(name, params, scipy_ref, n):
    """fresh_count(n) point sets of n, called in turn (harness.rotate)."""

    def setup():
        import numpy as np

        rng = np.random.default_rng(0)
        sets = [_dist_data(name, params, scipy_ref, rng.uniform(0, 1, n), "sd") for _ in range(fresh_count(n))]
        return {"sets": sets}

    return setup


def _mid_cases():
    group = "점 10^3·10^4개"
    out = []
    for name, params, scipy_ref, m in MID:
        for n, size in MID_SIZES:
            refs = [Ref("scipy", lambda d, m=m: rotate([_ref_call("ref", m)(s) for s in d["sets"]]))]
            if name in NEW:
                refs.append(
                    Ref(NEW_LABEL, lambda d, m=m: rotate([_ref_call("new", m)(s) for s in d["sets"]]), lib="scipy")
                )
            out.append(
                Case(
                    area="stats",
                    group=group,
                    name=f"{_label(name, params)}.{m}",
                    size=size,
                    n=n,
                    setup=_mid_setup(name, params, scipy_ref, n),
                    sd=lambda d, m=m: rotate([_sd_call(m)(s) for s in d["sets"]]),
                    refs=tuple(refs),
                    note=f"점 {fresh_count(n)}벌을 돌려 쓴다",
                )
            )
    return out


# ---------------------------------------------------------------- t7: sample

# (class, parameters, numpy Generator call g(n) or None)
T7 = [
    ("Normal", {}, lambda g, n: g.standard_normal(n)),
    ("LogNormal", {}, lambda g, n: g.lognormal(size=n)),
    ("Exponential", {}, lambda g, n: g.standard_exponential(n)),
    ("Uniform", dict(a=0.0, b=1.0), lambda g, n: g.random(n)),
    ("Laplace", {}, lambda g, n: g.laplace(size=n)),
    ("Cauchy", {}, lambda g, n: g.standard_cauchy(n)),
    ("Logistic", {}, lambda g, n: g.logistic(size=n)),
    ("Gumbel", {}, lambda g, n: g.gumbel(size=n)),
    ("GEV", dict(c=0.1), None),
    ("Weibull", dict(c=2.0), lambda g, n: g.weibull(2.0, n)),
    ("Pareto", dict(b=3.0), lambda g, n: g.pareto(3.0, n) + 1.0),
    ("Rayleigh", {}, lambda g, n: g.rayleigh(size=n)),
    ("Gamma", dict(a=3.0), lambda g, n: g.standard_gamma(3.0, n)),
    ("Gamma", dict(a=0.5), lambda g, n: g.standard_gamma(0.5, n)),
    ("ChiSquared", dict(df=5.0), lambda g, n: g.chisquare(5.0, n)),
    ("InverseGamma", dict(a=3.0), None),
    ("Beta", dict(a=2.0, b=3.0), lambda g, n: g.beta(2.0, 3.0, n)),
    ("Beta", dict(a=0.5, b=0.5), lambda g, n: g.beta(0.5, 0.5, n)),
    ("StudentT", dict(df=5.0), lambda g, n: g.standard_t(5.0, n)),
    ("F", dict(dfn=5.0, dfd=7.0), lambda g, n: g.f(5.0, 7.0, n)),
    ("NoncentralChiSquared", dict(df=5.0, nc=2.0), lambda g, n: g.noncentral_chisquare(5.0, 2.0, n)),
    ("NoncentralT", dict(df=5.0, nc=1.0), None),
    ("NoncentralF", dict(dfn=5.0, dfd=7.0, nc=2.0), lambda g, n: g.noncentral_f(5.0, 7.0, 2.0, n)),
    ("Bernoulli", dict(p=0.3), lambda g, n: g.binomial(1, 0.3, n)),
    ("DiscreteUniform", dict(low=0.0, high=10.0), lambda g, n: g.integers(0, 10, n)),
    ("Geometric", dict(p=0.1), lambda g, n: g.geometric(0.1, n)),
    ("Poisson", dict(mu=3.0), lambda g, n: g.poisson(3.0, n)),
    ("Poisson", dict(mu=100.0), lambda g, n: g.poisson(100.0, n)),
    ("Binomial", dict(n=20.0, p=0.3), lambda g, n: g.binomial(20, 0.3, n)),
    ("Binomial", dict(n=1e4, p=0.3), lambda g, n: g.binomial(10000, 0.3, n)),
    ("NegativeBinomial", dict(n=5.0, p=0.3), lambda g, n: g.negative_binomial(5.0, 0.3, n)),
    (
        "Hypergeometric",
        dict(total=1000.0, successes=300.0, draws=100.0),
        lambda g, n: g.hypergeometric(300, 700, 100, n),
    ),
    ("BetaBinomial", dict(n=20.0, a=2.0, b=3.0), None),
    ("LogSeries", dict(p=0.9), lambda g, n: g.logseries(0.9, n)),
    ("Zipf", dict(a=2.0), lambda g, n: g.zipf(2.0, n)),
]
T7_N = 10**6
T7_QUICK = {"Normal.sample", "Gamma(a=0.5).sample", "Binomial(n=10000, p=0.3).sample"}
# The rows numpy's Generator has no sampler for: scipy's rvs, drawing from a Generator.
T7_SCIPY = {
    "GEV": lambda st, kw: st.genextreme(kw["c"]),
    "InverseGamma": lambda st, kw: st.invgamma(kw["a"]),
    "NoncentralT": lambda st, kw: st.nct(kw["df"], kw["nc"]),
    "BetaBinomial": lambda st, kw: st.betabinom(int(kw["n"]), kw["a"], kw["b"]),
}


def _sample_setup(name, kw):
    def setup():
        import numpy as np
        import scipy.stats as st

        import stardust as sd

        return {
            "D": getattr(sd.stats, name)(**kw),
            "rng": sd.random.default_rng(1),
            "g": np.random.default_rng(1),
            "ref": T7_SCIPY[name](st, kw) if name in T7_SCIPY else None,
        }

    return setup


def _sample_refs(name, npf, size):
    if npf:
        return (Ref("numpy", lambda d: lambda g=d["g"]: npf(g, size)),)
    if name in T7_SCIPY:
        return (Ref("scipy", lambda d: lambda r=d["ref"], g=d["g"]: r.rvs(size=size, random_state=g)),)
    return ()


def _sample_cases():
    out = []
    for name, kw, npf in T7:
        params = ", ".join(f"{k}={v:g}" for k, v in kw.items())
        label = f"{name}({params}).sample" if params else f"{name}.sample"
        setup = _sample_setup(name, kw)
        refs = _sample_refs(name, npf, T7_N)
        out.append(
            Case(
                area="stats",
                group="표본 추출",
                name=label,
                size="10^6",
                n=T7_N,
                setup=setup,
                sd=lambda d: lambda D=d["D"], r=d["rng"]: D.sample(T7_N, rng=r),
                refs=refs,
                quick=label in T7_QUICK,
            )
        )
        refs1 = _sample_refs(name, npf, None)
        out.append(
            Case(
                area="stats",
                group="표본 추출(스칼라)",
                name=label,
                setup=setup,
                sd=lambda d: lambda D=d["D"], r=d["rng"]: D.sample(rng=r),
                refs=refs1,
                threads=(1,),
            )
        )
    return out


# ---------------------------------------------------------------- t8: descriptive

T8 = [
    ("skew", lambda S, X, ax: S.skew(X, axis=ax), lambda st, x, ax: st.skew(x, axis=ax)),
    ("kurtosis", lambda S, X, ax: S.kurtosis(X, axis=ax), lambda st, x, ax: st.kurtosis(x, axis=ax)),
    ("describe", lambda S, X, ax: S.describe(X, axis=ax), lambda st, x, ax: st.describe(x, axis=ax)),
    ("moment", lambda S, X, ax: S.moment(X, order=3, axis=ax), lambda st, x, ax: st.moment(x, order=3, axis=ax)),
    ("sem", lambda S, X, ax: S.sem(X, axis=ax), lambda st, x, ax: st.sem(x, axis=ax)),
    ("zscore", lambda S, X, ax: S.zscore(X, axis=ax), lambda st, x, ax: st.zscore(x, axis=ax)),
    ("rankdata", lambda S, X, ax: S.rankdata(X, axis=ax), lambda st, x, ax: st.rankdata(x, axis=ax)),
    ("iqr", lambda S, X, ax: S.iqr(X, axis=ax), lambda st, x, ax: st.iqr(x, axis=ax)),
    ("mode", lambda S, X, ax: S.mode(X, axis=ax), lambda st, x, ax: st.mode(x, axis=ax)),
    ("entropy", lambda S, X, ax: S.entropy(X, axis=ax), lambda st, x, ax: st.entropy(x, axis=ax)),
    (
        "differential_entropy",
        lambda S, X, ax: S.differential_entropy(X, axis=ax),
        lambda st, x, ax: st.differential_entropy(x, axis=ax),
    ),
    (
        "skew omit",
        lambda S, X, ax: S.skew(X, axis=ax, nan_policy="omit"),
        lambda st, x, ax: st.skew(x, axis=ax, nan_policy="omit"),
    ),
]
T8_QUICK = {
    ("skew", None, "10^6"),
    ("rankdata", 0, "(1000, 1000)"),
    ("iqr", None, "10^6"),
    ("describe", 0, "(1000, 1000)"),
}
# (axis, shape, size label): the whole array, down the columns, along short rows
T8_SHAPES = ((None, (10**6,), "10^6"), (0, (1000, 1000), "(1000, 1000)"), (1, (100_000, 10), "(100000, 10)"))
T8_SMALL = ("skew", "describe", "zscore", "iqr")  # also on 100 elements (fresh inputs in turn)
T8_SMALL_N = 100
T8_ROWS_SCIPY = 10_000  # skew omit along short rows: scipy loops over the rows with a NaN; timed on these first rows


def _describe_data(fname, shape, seed=0):
    import numpy as np

    n = math.prod(shape)
    x1 = np.random.default_rng(seed).gamma(2.0, size=n)
    x = x1.reshape(shape)
    if fname == "mode":
        x = np.round(x1 * 4).reshape(shape)  # repeated values
    if fname == "skew omit":
        x = x.copy()
        x.reshape(-1)[::1000] = np.nan
    return x


def _describe_setup(fname, shape):
    def setup():
        import scipy.stats as st

        import stardust as sd

        x = _describe_data(fname, shape)
        return {"S": sd.stats, "st": st, "x": x, "X": sd.asarray(x)}

    return setup


def _describe_small_setup(fname):
    def setup():
        import scipy.stats as st

        import stardust as sd

        xs = [_describe_data(fname, (T8_SMALL_N,), seed=i) for i in range(fresh_count(T8_SMALL_N))]
        return {"S": sd.stats, "st": st, "xs": xs, "Xs": [sd.asarray(x) for x in xs]}

    return setup


def _describe_ref(g, ax, sampled=False):
    if not sampled:
        return Ref("scipy", lambda d: lambda: g(d["st"], d["x"], ax))
    return Ref(
        "scipy",
        lambda d: lambda x=d["x"][:T8_ROWS_SCIPY].copy(): g(d["st"], x, ax),
        n=T8_ROWS_SCIPY * 10,
    )


def _describe_cases():
    out = []
    for fname, f, g in T8:
        base, _, omit = fname.partition(" ")

        def call(ax):
            return (
                ["x"]
                + (["order=3"] if base == "moment" else [])
                + ([] if ax is None else [f"axis={ax}"])
                + (['nan_policy="omit"'] if omit else [])
            )

        for ax, shape, size in T8_SHAPES:
            out.append(
                Case(
                    area="stats",
                    group="기술 통계",
                    name=f"{base}({', '.join(call(ax))})",
                    size=size,
                    n=10**6,
                    setup=_describe_setup(fname, shape),
                    sd=lambda d, f=f, ax=ax: lambda: f(d["S"], d["X"], ax),
                    refs=(_describe_ref(g, ax, sampled=bool(omit) and ax == 1),),
                    quick=(fname, ax, size) in T8_QUICK,
                    note="원소 1000개마다 NaN" if omit else "",
                )
            )
        if fname in T8_SMALL:
            out.append(
                Case(
                    area="stats",
                    group="기술 통계",
                    name=f"{base}({', '.join(call(None))})",
                    size=str(T8_SMALL_N),
                    n=T8_SMALL_N,
                    setup=_describe_small_setup(fname),
                    sd=lambda d, f=f: rotate([lambda X=X: f(d["S"], X, None) for X in d["Xs"]]),
                    refs=(Ref("scipy", lambda d, g=g: rotate([lambda x=x: g(d["st"], x, None) for x in d["xs"]])),),
                    note=f"입력 {fresh_count(T8_SMALL_N)}벌을 돌려 쓴다",
                )
            )
    return out


# ---------------------------------------------------------------- t9: multivariate

T9_N = 10**5


def _mv_setup(which):
    """The t9 rows in the old script's order (one generator over them); `which` picks one."""

    def setup():
        import numpy as np
        import scipy.stats as st

        import stardust as sd

        S = sd.stats
        rng = np.random.default_rng(0)
        for d in (3, 10, 50):
            a = rng.normal(size=(d, d))
            cov = a @ a.T + d * np.eye(d)
            mu = rng.normal(size=d)
            x = rng.normal(size=(T9_N, d))
            if which == ("MultivariateNormal", d):
                return {
                    "D": S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)),
                    "ref": st.multivariate_normal(mu, cov),
                    "x": x,
                    "X": sd.asarray(x),
                }
            if which == ("MultivariateT", d):
                return {
                    "D": S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=5.0),
                    "ref": st.multivariate_t(mu, cov, df=5.0),
                    "x": x,
                    "X": sd.asarray(x),
                }
        p = np.array([0.1, 0.2, 0.3, 0.4])
        xm = rng.multinomial(20, p, size=T9_N).astype(float)
        if which[0] == "Multinomial":
            return {
                "D": S.Multinomial(n=20.0, p=sd.asarray(p)),
                "ref": st.multinomial(20, p),
                "x": xm,
                "X": sd.asarray(xm),
            }
        al = np.array([0.5, 2.0, 3.0, 1.5])
        xd = rng.dirichlet(al, size=T9_N)
        return {"D": S.Dirichlet(alpha=sd.asarray(al)), "ref": st.dirichlet(al), "x": xd, "X": sd.asarray(xd)}

    return setup


def _mv_cases():
    rows = []
    for d in (3, 10, 50):
        rows += [
            (("MultivariateNormal", d), f"MultivariateNormal(d={d})", "logpdf"),
            (("MultivariateT", d), f"MultivariateT(d={d}, df=5)", "logpdf"),
        ]
    rows += [(("Multinomial", 4), "Multinomial(n=20, K=4)", "logpmf"), (("Dirichlet", 4), "Dirichlet(K=4)", "logpdf")]
    quick = {"MultivariateNormal(d=10).logpdf", "Dirichlet(K=4).sample"}
    out = []
    for which, label, meth in rows:
        setup = _mv_setup(which)
        if which[0] == "Dirichlet":
            ref = lambda d: lambda r=d["ref"], xt=d["x"].T: r.logpdf(xt)  # noqa: E731
        else:
            ref = lambda d, meth=meth: lambda f=getattr(d["ref"], meth), x=d["x"]: f(x)  # noqa: E731
        out.append(
            Case(
                area="stats",
                group="다변량 분포",
                name=f"{label}.{meth}",
                size="10^5 점",
                n=T9_N,
                setup=setup,
                sd=lambda d, meth=meth: lambda f=getattr(d["D"], meth), X=d["X"]: f(X),
                refs=(Ref("scipy", ref),),
                quick=f"{label}.{meth}" in quick,
            )
        )
        out.append(
            Case(
                area="stats",
                group="다변량 분포",
                name=f"{label}.sample",
                size="10^5 점",
                n=T9_N,
                setup=lambda setup=setup: _with_rng(setup()),
                sd=lambda d: lambda D=d["D"], r=d["rng"]: D.sample(T9_N, rng=r),
                refs=(Ref("scipy", lambda d: lambda r=d["ref"], g=d["g"]: r.rvs(size=T9_N, random_state=g)),),
                quick=f"{label}.sample" in quick,
            )
        )
    return out


def _with_rng(d):
    """sd's generator and, for scipy's rvs, a numpy Generator: an int random_state would
    seed a new one on every call."""
    import numpy as np

    import stardust as sd

    d["rng"] = sd.random.default_rng(1)
    d["g"] = np.random.default_rng(1)
    return d


# ---------------------------------------------------------------- t9: mvn / mvt cdf


# Per (d, df): sd's points, scipy's points (the old script timed scipy on X[:n_sc], which
# is all of X when sd has fewer), the atols. d <= 3: the exact formulas.
def _mvcdf_rows():
    rows = []
    for d in (2, 3, 5, 10):
        for df in (math.inf, 4.0):
            exact = d <= 3
            n_ours = (20000 if math.isinf(df) else 200) if exact else (40 if math.isinf(df) else 20)
            n_sc = min(n_ours, 400 if exact or not math.isinf(df) else 40)
            for atol in (1e-6,) if exact else (1e-6, 1e-5, 1e-4):
                rows.append((d, df, n_ours, n_sc, atol))
    return rows


# sd at one thread above 5 s per call in the old table (timed once per round)
MVCDF_SLOW = {(10, math.inf, 1e-6), (10, 4.0, 1e-6)}


def _mvcdf_setup(d, df):
    def setup():
        import numpy as np
        import scipy.stats as st

        import stardust as sd

        rng = np.random.default_rng(d)
        A = rng.normal(size=(d, d + 2))
        S = A @ A.T
        s = np.sqrt(np.diag(S))
        S = S / s[:, None] / s[None, :]  # a random correlation matrix
        X = None
        for dfi in (math.inf, 4.0):  # the old script drew both point sets from one generator, in this order
            exact = d <= 3
            n = (20000 if math.isinf(dfi) else 200) if exact else (40 if math.isinf(dfi) else 20)
            Xi = rng.normal(size=(n, d)) + 0.5
            if dfi == df:
                X = Xi
                break
        n_sc = min(len(X), 400 if d <= 3 or not math.isinf(df) else 40)
        if math.isinf(df):
            D = sd.stats.MultivariateNormal(mu=sd.asarray(np.zeros(d)), cov=sd.asarray(S))
            ref = None
        else:
            D = sd.stats.MultivariateT(mu=sd.asarray(np.zeros(d)), shape=sd.asarray(S), df=df)
            ref = st.multivariate_t(np.zeros(d), S, df=df, seed=0)
        return {"D": D, "X": sd.asarray(X), "xs": X[:n_sc].copy(), "S": S, "ref": ref, "st": st, "d": d}

    return setup


def _scipy_mvt_cdf(d):
    import numpy as np

    return lambda: np.array([d["ref"].cdf(x) for x in d["xs"]])


def _mvcdf_cases():
    out = []
    quick = {(5, math.inf, 1e-5), (3, math.inf, 1e-6), (5, 4.0, 1e-4)}
    for d, df, n_ours, n_sc, atol in _mvcdf_rows():
        cls = "MultivariateNormal" if math.isinf(df) else "MultivariateT"
        what = f"d={d}" if math.isinf(df) else f"d={d}, df=4"
        slow = (d, df, atol) in MVCDF_SLOW
        note = "정확한 공식" if d <= 3 else ""
        if slow:
            note = "sd 1스레드 한 번에 5초가 넘는다"
        if not math.isinf(df):
            note = ", ".join(filter(None, (note, "scipy의 mvt cdf에는 허용오차 인자가 없다")))
        out.append(
            Case(
                area="stats",
                group="다변량 정규·t cdf",
                name=f"{cls}({what}).cdf(x, atol={atol:.0e})",
                size=f"{n_ours:,} 점",
                n=n_ours,
                setup=_mvcdf_setup(d, df),
                sd=lambda dd, atol=atol: lambda D=dd["D"], X=dd["X"]: D.cdf(X, atol=atol),
                refs=(_mvcdf_ref(d, df, atol, n_sc),),
                quick=(d, df, atol) in quick,
                note=note,
            )
        )
    return out


MVN_LABEL = "scipy abseps=atol"
# scipy's mvn cdf at abseps=1e-6 takes about 18 ms a point at d=3 (even): 100 points. At
# d=10 it takes 4 ms to 3.5 s a point (the hard ones run to maxpts), so no sample of the 40
# is representative: all 40 (about 22 s a call, timed once a round).
MVN_SCIPY_N = {(3, 1e-6): 100}


def _mvcdf_ref(dim, df, atol, n_sc):
    """scipy's mvn cdf at sd's accuracy (abseps=atol, releps=0); its mvt cdf takes no
    tolerance and runs at its own."""
    if not math.isinf(df):
        return Ref("scipy", _scipy_mvt_cdf, n=n_sc)
    k = MVN_SCIPY_N.get((dim, atol), n_sc)

    def make(d):
        import numpy as np

        st, xs, S = d["st"], d["xs"][:k].copy(), d["S"]
        mu = np.zeros(d["d"])
        return lambda: st.multivariate_normal.cdf(xs, mu, S, abseps=atol, releps=0, rng=0)

    return Ref(MVN_LABEL, make, n=k, lib="scipy")


# ---------------------------------------------------------------- t10: matrix distributions

T10_N = 10**4


def _matrix_setup(which):
    def setup():
        import numpy as np
        import scipy.stats as st

        import stardust as sd

        S = sd.stats
        rng = np.random.default_rng(0)
        for p in (3, 10):
            a = rng.normal(size=(p, p))
            V = a @ a.T + p * np.eye(p)
            df = p + 2.5
            if which == ("Wishart", p):
                ref = st.wishart(df, V)
                x = ref.rvs(T10_N, random_state=1)
                return {
                    "D": S.Wishart(df=df, scale=sd.asarray(V)),
                    "ref": ref,
                    "x": x,
                    "X": sd.asarray(x),
                    "xm": np.moveaxis(x, 0, -1),
                }
            if which == ("InverseWishart", p):
                ref = st.invwishart(df, V)
                x = ref.rvs(T10_N, random_state=1)
                return {
                    "D": S.InverseWishart(df=df, scale=sd.asarray(V)),
                    "ref": ref,
                    "x": x,
                    "X": sd.asarray(x),
                    "xm": np.moveaxis(x, 0, -1),
                }
        for n, p in ((3, 2), (10, 5)):
            M = rng.normal(size=(n, p))
            U, V = np.eye(n) + 0.3, np.eye(p) + 0.2
            if which == ("MatrixNormal", (n, p)):
                ref = st.matrix_normal(M, U, V)
                x = ref.rvs(T10_N, random_state=1)
                return {
                    "D": S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V)),
                    "ref": ref,
                    "x": x,
                    "X": sd.asarray(x),
                    "xm": x,
                }
        raise ValueError(which)

    return setup


def _matrix_cases():
    rows = []
    for p in (3, 10):
        rows += [
            (("Wishart", p), f"Wishart(p={p}, df={p + 2.5:g})"),
            (("InverseWishart", p), f"InverseWishart(p={p}, df={p + 2.5:g})"),
        ]
    rows += [(("MatrixNormal", (3, 2)), "MatrixNormal(3x2)"), (("MatrixNormal", (10, 5)), "MatrixNormal(10x5)")]
    quick = {"Wishart(p=3, df=5.5).logpdf"}
    out = []
    for which, label in rows:
        setup = _matrix_setup(which)
        out.append(
            Case(
                area="stats",
                group="행렬 분포",
                name=f"{label}.logpdf",
                size="10^4 행렬",
                n=T10_N,
                setup=setup,
                sd=lambda d: lambda D=d["D"], X=d["X"]: D.logpdf(X),
                refs=(Ref("scipy", lambda d: lambda r=d["ref"], x=d["xm"]: r.logpdf(x)),),
                quick=f"{label}.logpdf" in quick,
            )
        )
        out.append(
            Case(
                area="stats",
                group="행렬 분포",
                name=f"{label}.sample",
                size="10^4 행렬",
                n=T10_N,
                setup=lambda setup=setup: _with_rng(setup()),
                sd=lambda d: lambda D=d["D"], r=d["rng"]: D.sample(T10_N, rng=r),
                refs=(Ref("scipy", lambda d: lambda r=d["ref"], g=d["g"]: r.rvs(T10_N, random_state=g)),),
            )
        )
    return out


# ---------------------------------------------------------------- all


def cases() -> list[Case]:
    return [
        *_dist_cases("연속 분포", T4, CONT, "scipy"),
        *_other_cases(),
        *_dist_cases("이산 분포", T5, DISC, "sd", no_scipy=T5_NO_SCIPY, sampled=SMALL_SAMPLED, memo=True),
        *_wide_cases(),
        *_dist_cases("극값·수명·비중심 분포", T6, CONT, "sd"),
        *_param_array_cases(),
        *_mid_cases(),
        *_sample_cases(),
        *_describe_cases(),
        *_mv_cases(),
        *_mvcdf_cases(),
        *_matrix_cases(),
    ]
