"""Golden cases for sd.stats (tools/golden.py has the contract).

Univariate distributions (every class of __all__), each in three forms:
  typ     one typical parameter set: the object itself (parameters, moments, median, mode,
          entropy, support: every method without arguments), every input method at 41
          points spread over the distribution (its own icdf of a grid of p) and at a Python
          scalar, pdf/cdf/icdf at many points (the multithreaded loop), samples
  grid    every combination of special parameter values (zero, tiny, huge, inf, NaN, outside
          the domain), one axis per parameter, against 16 special inputs on another axis
  errors  the documented constructor and method errors
Multivariate and matrix distributions get a few parameter sets each, batches included.
The descriptive functions run on 1-d, 2-d, NaN-holding, tied, constant, empty and large
inputs over their axis, correction, bias, method and nan_policy options.
"""

import math
import zlib

import numpy as np

NAN, INF = math.nan, math.inf

# name: (typical parameters, {parameter: special values}); grids keep prod(lengths) <= 64.
_LOC = [-1e300, -3.0, 0.0, 2.5, 1e300, INF, NAN]
_SCALE = [-1.0, 0.0, 1e-300, 0.5, 1.0, 1e10, INF, NAN]
_DF = [-1.0, 0.0, 5e-324, 1e-10, 0.5, 1.0, 2.0, 3.5, 30.0, 1e10, 1e300, INF, NAN]
_POS = [-1.0, 0.0, 5e-324, 1e-300, 0.5, 1.0, 1e300, INF, NAN]
_SHAPE = [-1.0, 0.0, 1e-300, 1e-3, 0.5, 1.0, 7.5, 1e10]
_SCALE8 = [NAN, 0.0, 1e-300, 0.5, 1.0, 3.0, 1e300, INF]
_PROB = [-0.1, 0.0, 1e-10, 0.3, 0.5, 1 - 1e-10, 1.0, NAN]

CONTINUOUS = {
    "Normal": ({"mu": 1.5, "sigma": 2.0}, {"mu": _LOC, "sigma": _SCALE}),
    "StudentT": ({"df": 4.5}, {"df": _DF}),
    "ChiSquared": ({"df": 3.5}, {"df": _DF}),
    "F": (
        {"dfn": 3.0, "dfd": 7.5},
        {"dfn": [NAN, 0.0, 1e-10, 0.5, 1.0, 3.0, 1e10, INF], "dfd": [0.0, 1e-10, 0.5, 2.0, 4.5, 100.0, 1e10, INF]},
    ),
    "Gamma": ({"a": 2.5, "scale": 1.5}, {"a": _SHAPE, "scale": _SCALE8}),
    "Beta": (
        {"a": 2.0, "b": 3.5},
        {"a": [0.0, 1e-10, 0.5, 1.0, 2.0, 10.0, 1e5, INF], "b": [NAN, 1e-10, 0.5, 1.0, 3.0, 10.0, 1e8, 0.0]},
    ),
    "Exponential": ({"scale": 2.0}, {"scale": _POS}),
    "LogNormal": (
        {"mu": 0.5, "sigma": 0.75},
        {"mu": [-1e3, -3.0, 0.0, 2.5, 700.0, INF, NAN], "sigma": [-1.0, 0.0, 1e-300, 0.25, 1.0, 30.0, INF, NAN]},
    ),
    "Uniform": (
        {"a": -1.0, "b": 3.0},
        {"a": [-INF, -1e300, -1.0, 0.0, 1.0, NAN], "b": [-1.0, 0.0, 1.0, 2.0, 1e300, INF, NAN]},
    ),
    "Cauchy": ({"mu": -0.5, "sigma": 1.5}, {"mu": _LOC, "sigma": _SCALE}),
    "Laplace": ({"mu": 0.25, "sigma": 0.5}, {"mu": _LOC, "sigma": _SCALE}),
    "Weibull": ({"c": 1.7, "scale": 2.0}, {"c": [-1.0, 0.0, 1e-10, 0.5, 1.0, 2.0, 30.0, INF], "scale": _SCALE8}),
    "Logistic": ({"mu": 1.0, "sigma": 0.3}, {"mu": _LOC, "sigma": _SCALE}),
    "Gumbel": ({"mu": -2.0, "sigma": 3.0}, {"mu": _LOC, "sigma": _SCALE}),
    "GEV": (
        {"c": -0.2, "mu": 1.0, "sigma": 2.0},
        {"c": [-2.0, -0.5, 0.0, 1e-10, 0.3, 1.0, 4.0, NAN], "mu": [0.0, 2.0], "sigma": [0.0, 0.5, 1.0, INF]},
    ),
    # Not scale=0: Pareto(scale=0).pdf(+-0) and .logpdf(+-0) never return (2026-11-20, any b).
    # Put 0.0 back in place of 5e-324 once that is fixed.
    "Pareto": (
        {"b": 2.5, "scale": 1.5},
        {"b": [-1.0, 0.0, 1e-10, 0.5, 1.0, 2.5, 30.0, INF], "scale": [NAN, 5e-324, 1e-300, 0.5, 1.0, 3.0, 1e300, INF]},
    ),
    "Rayleigh": ({"scale": 1.5}, {"scale": _POS}),
    "InverseGamma": ({"a": 3.5, "scale": 2.0}, {"a": _SHAPE, "scale": _SCALE8}),
    "NoncentralChiSquared": (
        {"df": 4.0, "nc": 2.5},
        {"df": [0.0, 1e-10, 0.5, 1.0, 4.0, 100.0, 1e5, NAN], "nc": [-1.0, 0.0, 1e-10, 0.5, 5.0, 100.0, 1e4, INF]},
    ),
    "NoncentralT": (
        {"df": 6.0, "nc": 1.5},
        {"df": [0.0, 1e-10, 0.5, 1.0, 4.0, 100.0, 1e10, INF], "nc": [NAN, -30.0, -1.0, 0.0, 0.5, 5.0, 40.0, INF]},
    ),
    "NoncentralF": (
        {"dfn": 3.0, "dfd": 12.0, "nc": 2.0},
        {"dfn": [0.5, 3.0, 100.0, INF], "dfd": [1e-10, 2.0, 30.0, INF], "nc": [NAN, 0.0, 2.5, 50.0]},
    ),
}
DISCRETE = {
    "Binomial": ({"n": 20, "p": 0.3}, {"n": [0.0, 1.0, 5.0, 30.0, 1000.0, 1e9, 2.5, NAN], "p": _PROB}),
    "Poisson": ({"mu": 3.5}, {"mu": [-1.0, 0.0, 1e-300, 1e-10, 0.5, 3.0, 40.0, 1e5, 1e15, INF, NAN]}),
    "Geometric": ({"p": 0.25}, {"p": [-0.1, 0.0, 1e-300, 1e-10, 0.3, 0.5, 1 - 1e-10, 1.0, 1.5, NAN]}),
    "NegativeBinomial": ({"n": 4.5, "p": 0.4}, {"n": [0.0, 1e-10, 0.5, 1.0, 5.0, 100.0, 1e8, NAN], "p": _PROB}),
    "Hypergeometric": (
        {"total": 60, "successes": 20, "draws": 15},
        {"total": [0.0, 10.0, 50.0, 1e6], "successes": [0.0, 3.0, 10.0, 2.5], "draws": [0.0, 4.0, 10.0, 60.0]},
    ),
    "Bernoulli": ({"p": 0.3}, {"p": [-0.1, 0.0, 1e-300, 0.3, 0.5, 1.0, NAN]}),
    "DiscreteUniform": (
        {"low": -3, "high": 12},
        {"low": [-1e18, -5.0, 0.0, 3.0, 2.5, NAN], "high": [-5.0, 0.0, 3.0, 10.0, 1e18, INF]},
    ),
    "BetaBinomial": (
        {"n": 15, "a": 2.0, "b": 3.5},
        {"n": [0.0, 1.0, 10.0, 1e6], "a": [1e-10, 0.5, 2.0, 1e5], "b": [NAN, 0.5, 3.0, 1e6]},
    ),
    "LogSeries": ({"p": 0.7}, {"p": [-0.1, 0.0, 1e-300, 1e-10, 0.3, 0.5, 0.99, 1 - 1e-10, 1.0, NAN]}),
    "Zipf": ({"a": 2.5}, {"a": [-1.0, 0.0, 1.0, 1 + 1e-10, 1.5, 2.0, 3.5, 30.0, INF, NAN]}),
}
HEAVY = {"NoncentralChiSquared", "NoncentralT", "NoncentralF", "BetaBinomial", "Hypergeometric"}

XC = [-INF, -1e10, -3.0, -1.0, -0.0, 0.0, 5e-324, 1e-10, 0.25, 0.5, 1.0, 2.0, 10.0, 1e10, INF, NAN]
XD = [-INF, -1.0, -0.0, 0.0, 1.0, 2.0, 3.0, 4.5, 7.0, 10.0, 30.0, 100.0, 1e6, 1e18, INF, NAN]
PG = [-0.5, 0.0, 5e-324, 1e-300, 1e-20, 1e-10, 0.01, 0.25, 0.5, 0.75, 0.99, 1 - 1e-10, 1 - 2**-53, 1.0, 1.5, NAN]
LPG = [-INF, -1e300, -1e5, -745.0, -100.0, -30.0, -5.0, -1.0, -0.5, -1e-5, -1e-20, -5e-324, -0.0, 0.0, 0.5, NAN]
CONT_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf")
DISC_METHODS = ("pmf", "logpmf", "cdf", "logcdf", "ccdf", "logccdf")
INV_METHODS = (("icdf", PG), ("iccdf", PG), ("ilogcdf", LPG), ("ilogccdf", LPG))


def seed_of(name):
    return zlib.crc32(name.encode())


def cases(L):
    yield from univariate(L)
    yield from univariate_errors(L)
    yield from multivariate(L)
    yield from descriptive(L)


def univariate(L):
    sd = L.sd
    st = sd.stats
    for discrete, table in ((False, CONTINUOUS), (True, DISCRETE)):
        methods = DISC_METHODS if discrete else CONT_METHODS
        xs_special = XD if discrete else XC
        for name, (typ, grid) in table.items():
            cls = getattr(st, name)
            seed = seed_of(name)
            big = 4101 if name in HEAVY else 2**15 + 5

            def mk(cls=cls, typ=typ):
                return cls(**typ)

            def mkgrid(cls=cls, grid=grid):
                k = len(grid)
                kw = {}
                for i, (p, vals) in enumerate(grid.items()):
                    shape = [1] * k
                    shape[i] = len(vals)
                    kw[p] = L.arr(np.array(vals, dtype=np.float64).reshape(shape))
                return cls(**kw)

            def pts(mk=mk, discrete=discrete):
                X = mk()
                x = X.icdf(L.arr(np.linspace(0.0025, 0.9975, 41)))
                return x if not discrete else x + L.arr(np.where(np.arange(41) % 4 == 3, 0.5, 0.0))

            def special_x(k, xs=xs_special):
                return L.arr(np.array(xs).reshape((len(xs),) + (1,) * k))

            yield f"{name} typ", mk
            yield f"{name} typ repr", lambda mk=mk: repr(mk())
            yield f"{name} grid", mkgrid
            k = len(grid)
            for m in methods:
                yield f"{name} typ {m}", lambda mk=mk, m=m, pts=pts: getattr(mk(), m)(pts())
                yield f"{name} typ {m} scalar", lambda mk=mk, m=m: getattr(mk(), m)(mk().median())
                yield f"{name} grid {m}", lambda mkgrid=mkgrid, m=m, k=k, sx=special_x: getattr(mkgrid(), m)(sx(k))
            for m, g in INV_METHODS:
                yield f"{name} typ {m}", lambda mk=mk, m=m, g=g: getattr(mk(), m)(L.arr(np.array(g)))
                yield (
                    f"{name} grid {m}",
                    lambda mkgrid=mkgrid, m=m, g=g, k=k: getattr(mkgrid(), m)(
                        L.arr(np.array(g).reshape((len(g),) + (1,) * k))
                    ),
                )
            for m in (methods[0], "cdf", "icdf"):

                def large(mk=mk, m=m, seed=seed, big=big):
                    X = mk()
                    p = L.arr(L.rng(seed).uniform(0, 1, big))
                    return getattr(X, m)(p if m == "icdf" else X.icdf(p))

                yield f"{name} typ {m} large", large
            yield (
                f"{name} typ cdf out",
                lambda mk=mk, pts=pts: (lambda o: (mk().cdf(pts(), out=o) is o, o))(sd.zeros((41,))),
            )
            yield f"{name} typ cdf ints", lambda mk=mk: mk().cdf(sd.asarray([0, 1, 2, 5]))
            yield f"{name} typ cdf bool", lambda mk=mk: mk().cdf(sd.asarray([True, False]))
            yield (
                f"{name} typ cdf 2d strided",
                lambda mk=mk, pts=pts: mk().cdf(sd.reshape(pts()[:40], (8, 5))[::-2, ::2]),
            )
            yield f"{name} typ cdf empty", lambda mk=mk: mk().cdf(sd.zeros((0, 2)))
            yield f"{name} typ cdf complex", lambda mk=mk: mk().cdf(sd.asarray([1 + 1j]))
            yield f"{name} typ cdf str", lambda mk=mk: mk().cdf("a")
            yield f"{name} typ cdf list", lambda mk=mk: mk().cdf([0.5])
            yield f"{name} typ cdf pyint big", lambda mk=mk: mk().cdf(2**63)

            def sample(mk, shape, seed, k=None):
                rng = sd.random.default_rng(seed)
                s = mk().sample(shape, rng=rng) if k is None else mk().sample(shape, rng=rng)
                return {"sample": s, "state": rng.state}

            yield f"{name} typ sample", lambda mk=mk, seed=seed: sample(mk, (1000,), seed)
            yield f"{name} typ sample None", lambda mk=mk, seed=seed: sample(mk, None, seed)
            yield f"{name} typ sample int", lambda mk=mk, seed=seed: sample(mk, 5, seed)
            yield f"{name} typ sample 2d", lambda mk=mk, seed=seed: sample(mk, (3, 4), seed)
            yield f"{name} typ sample large", lambda mk=mk, seed=seed, big=big: sample(mk, (big,), seed + 1)
            yield f"{name} grid sample", lambda mkgrid=mkgrid, seed=seed: sample(mkgrid, (2,), seed + 2)
            yield f"{name} grid sample None", lambda mkgrid=mkgrid, seed=seed: sample(mkgrid, None, seed + 3)
            yield f"{name} typ sample no rng", lambda mk=mk: mk().sample(3)
            yield f"{name} typ sample rng int", lambda mk=mk: mk().sample(3, rng=1)
            yield f"{name} typ sample shape float", lambda mk=mk: mk().sample(3.0, rng=sd.random.default_rng(0))
            yield f"{name} typ sample shape negative", lambda mk=mk: mk().sample((-1,), rng=sd.random.default_rng(0))


def univariate_errors(L):
    st = L.sd.stats
    yield "err Normal positional", lambda: st.Normal(1.0)
    yield "err Normal unknown keyword", lambda: st.Normal(cool=1.0)
    yield "err Gamma missing", lambda: st.Gamma()
    yield "err Normal str", lambda: st.Normal(mu="a")
    yield "err Normal complex", lambda: st.Normal(mu=1j)
    yield "err Normal list", lambda: st.Normal(mu=[1.0])
    yield "err Normal broadcast", lambda: st.Normal(mu=L.sd.zeros((2,)), sigma=L.sd.ones((3,)))
    yield "err Normal pyint big", lambda: st.Normal(mu=2**63)
    yield "Normal bool param", lambda: st.Normal(mu=True)
    yield "Normal int params", lambda: st.Normal(mu=L.sd.asarray([1, 2]), sigma=3)
    yield "err ContinuousDistribution", lambda: st.ContinuousDistribution()
    yield "err DiscreteDistribution", lambda: st.DiscreteDistribution()
    yield "err Normal setattr", lambda: setattr(st.Normal(), "mu", 2.0)
    yield (
        "Normal isinstance",
        lambda: (
            isinstance(st.Normal(), st.ContinuousDistribution),
            isinstance(st.Poisson(mu=1.0), st.DiscreteDistribution),
            isinstance(st.Poisson(mu=1.0), st.ContinuousDistribution),
        ),
    )
    yield "err Poisson pdf", lambda: st.Poisson(mu=1.0).pdf(1.0)
    yield "err Normal pmf", lambda: st.Normal().pmf(1.0)


def spd(r, d, cond=10.0):
    q, _ = np.linalg.qr(r.standard_normal((d, d)))
    ev = np.geomspace(1.0, cond, d)
    return (q * ev) @ q.T


def multivariate(L):
    sd = L.sd
    st = sd.stats
    r = L.rng(101)
    a = L.arr
    cov2 = np.array([[2.0, 0.6], [0.6, 1.0]])
    cov3 = np.array([[1.0, 0.5, -0.3], [0.5, 2.0, 0.2], [-0.3, 0.2, 1.5]])
    cov5 = spd(r, 5, 100.0)
    covbatch = np.stack(
        [
            cov2,
            np.array([[1.0, -0.9], [-0.9, 1.0]]),
            np.array([[1.0, 0.0], [0.0, 1e-8]]),
            np.array([[1.0, 2.0], [2.0, 1.0]]),
        ]
    )  # the last is not positive definite
    sets = {
        "MVN d2": lambda: st.MultivariateNormal(mu=a(np.array([0.5, -1.0])), cov=a(cov2)),
        "MVN d3": lambda: st.MultivariateNormal(mu=a(np.array([0.0, 1.0, -0.5])), cov=a(cov3)),
        "MVN d5": lambda: st.MultivariateNormal(mu=a(np.zeros(5)), cov=a(cov5)),
        "MVN batch": lambda: st.MultivariateNormal(mu=a(np.zeros((4, 2))), cov=a(covbatch)),
        "MVT d2": lambda: st.MultivariateT(mu=a(np.array([0.5, -1.0])), shape=a(cov2), df=4.5),
        "MVT d3": lambda: st.MultivariateT(mu=a(np.zeros(3)), shape=a(cov3), df=2.5),
        "MVT d5": lambda: st.MultivariateT(mu=a(np.zeros(5)), shape=a(cov5), df=7.0),
        "MVT df inf": lambda: st.MultivariateT(mu=a(np.zeros(2)), shape=a(cov2), df=INF),
        "MVT batch df": lambda: st.MultivariateT(
            mu=a(np.zeros(2)), shape=a(cov2), df=a(np.array([0.5, 1.0, 1.5, 2.0, 3.0, 1e10, INF, NAN, 0.0]))
        ),
    }
    x2 = r.standard_normal((30, 2)) * 1.5
    x2[0] = [INF, 0.0]
    x2[1] = [-INF, 1.0]
    x2[2] = [NAN, 0.0]
    x3 = r.standard_normal((20, 3))
    x5 = r.standard_normal((6, 5))
    xin = {"d2": x2, "d3": x3, "d5": x5, "batch": x2[:, None, :], "df inf": x2, "df": x2[:5, None, :]}
    for nm, mk in sets.items():
        x = next(v for k, v in xin.items() if nm.endswith(k))
        yield nm, mk
        yield f"{nm} repr", lambda mk=mk: repr(mk())
        yield f"{nm} pdf", lambda mk=mk, x=x: mk().pdf(a(x))
        yield f"{nm} logpdf", lambda mk=mk, x=x: mk().logpdf(a(x))
        yield f"{nm} logpdf one", lambda mk=mk, x=x: mk().logpdf(a(x[3 % len(x)]))
        yield f"{nm} sample", lambda mk=mk: (lambda g: (mk().sample((50,), rng=g), g.state))(sd.random.default_rng(7))
        yield f"{nm} sample None", lambda mk=mk: mk().sample(rng=sd.random.default_rng(8))
        cx = x[:4] if x.ndim == 2 else x[:2]
        yield f"{nm} cdf", lambda mk=mk, cx=cx: mk().cdf(a(cx))
        yield f"{nm} cdf error", lambda mk=mk, cx=cx: mk().cdf(a(cx), atol=1e-4, return_error=True)
        yield f"{nm} cdf max_points", lambda mk=mk, cx=cx: mk().cdf(a(cx), max_points=2000, return_error=True)
    yield "MVN logpdf large", lambda: sets["MVN d3"]().logpdf(a(L.rng(102).standard_normal((40000, 3))))
    yield "MVN sample large", lambda: sets["MVN d3"]().sample((40000,), rng=sd.random.default_rng(9))
    yield "err MVN list", lambda: st.MultivariateNormal(mu=[0.0, 0.0], cov=a(cov2))
    yield "err MVN cov shape", lambda: st.MultivariateNormal(mu=a(np.zeros(2)), cov=a(cov3))
    yield "err MVN x components", lambda: sets["MVN d2"]().cdf(a(x3))
    yield "err MVN max_points float", lambda: sets["MVN d2"]().cdf(a(x2[:2]), max_points=1.0)
    yield "err MVN x list", lambda: sets["MVN d2"]().logpdf([0.0, 0.0])

    pk = np.array([0.2, 0.3, 0.5])
    disc = {
        "Multinomial": lambda: st.Multinomial(n=10, p=a(pk)),
        "Multinomial batch": lambda: st.Multinomial(
            n=a(np.array([0.0, 1.0, 10.0, 2.5, 1e6])[:, None]),
            p=a(np.array([[0.2, 0.3, 0.5], [0.0, 0.0, 1.0], [0.5, 0.5, 0.5]])),
        ),
        "Dirichlet": lambda: st.Dirichlet(alpha=a(np.array([0.5, 2.0, 3.5]))),
        "Dirichlet batch": lambda: st.Dirichlet(
            alpha=a(np.array([[1.0, 1.0, 1.0], [1e-3, 0.5, 2.0], [0.0, 1.0, 2.0]]))
        ),
    }
    kx = np.array(
        [[2.0, 3.0, 5.0], [0.0, 0.0, 10.0], [10.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.5, 2.5, 5.0], [-1.0, 5.0, 6.0]]
    )
    dx = np.array(
        [[0.2, 0.3, 0.5], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0], [0.4, 0.4, 0.4], [-0.1, 0.6, 0.5], [1 / 3, 1 / 3, 1 / 3]]
    )
    for nm, mk in disc.items():
        x = kx if nm.startswith("Multinomial") else dx
        if "batch" in nm:
            x = x[:, None, None, :] if nm.startswith("Multinomial") else x[:, None, :]
        yield nm, mk
        yield f"{nm} repr", lambda mk=mk: repr(mk())
        m = "pmf" if nm.startswith("Multinomial") else "pdf"
        yield f"{nm} {m}", lambda mk=mk, x=x, m=m: getattr(mk(), m)(a(x))
        yield f"{nm} log{m}", lambda mk=mk, x=x, m=m: getattr(mk(), "log" + m)(a(x))
        yield f"{nm} sample", lambda mk=mk: (lambda g: (mk().sample((40,), rng=g), g.state))(sd.random.default_rng(10))
    yield "Multinomial sample large", lambda: disc["Multinomial"]().sample((40000,), rng=sd.random.default_rng(11))

    s2 = np.array([[2.0, 0.3], [0.3, 0.5]])
    w = r.standard_normal((5, 2, 2))
    wx = np.einsum("nij,nkj->nik", w, w) + 0.1 * np.eye(2)
    mats = {
        "Wishart": lambda: st.Wishart(df=4.5, scale=a(s2)),
        "Wishart batch df": lambda: st.Wishart(df=a(np.array([0.5, 1.0, 1.5, 3.0, 1e9, 1e30, INF, NAN])), scale=a(s2)),
        "InverseWishart": lambda: st.InverseWishart(df=6.0, scale=a(s2)),
        "InverseWishart batch df": lambda: st.InverseWishart(
            df=a(np.array([1.5, 3.0, 3.5, 5.0, 6.0, 1e9, INF])), scale=a(s2)
        ),
        "MatrixNormal": lambda: st.MatrixNormal(mu=a(np.zeros((3, 2))), rowcov=a(cov3), colcov=a(s2)),
    }
    mx = {"Wishart": wx, "InverseWishart": wx, "MatrixNormal": r.standard_normal((4, 3, 2))}
    for nm, mk in mats.items():
        x = next(v for k, v in mx.items() if nm.startswith(k))
        if "batch" in nm:
            x = x[:, None]
        yield nm, mk
        yield f"{nm} repr", lambda mk=mk: repr(mk())
        yield f"{nm} pdf", lambda mk=mk, x=x: mk().pdf(a(x))
        yield f"{nm} logpdf", lambda mk=mk, x=x: mk().logpdf(a(x))
        yield f"{nm} sample", lambda mk=mk: (lambda g: (mk().sample((6,), rng=g), g.state))(sd.random.default_rng(12))
    yield "Wishart logpdf large", lambda: mats["Wishart"]().logpdf(a(np.broadcast_to(wx[0], (40000, 2, 2)).copy()))
    yield "err Wishart unknown keyword", lambda: st.Wishart(df=3.0, scale=a(s2), q=1)


def descriptive(L):
    sd = L.sd
    st = sd.stats
    a = L.arr
    r = L.rng(201)
    data = {
        "1d": r.standard_normal(25) * 3 + 1,
        "2d": r.standard_normal((8, 9)),
        "ties": r.integers(-3, 4, (6, 10)).astype(np.float64),
        "nan": np.where(r.random((7, 6)) < 0.2, np.nan, r.standard_normal((7, 6))),
        "inf": np.array([1.0, INF, 2.0, -INF, 3.0, 0.5]),
        "const": np.full(12, 2.5),
        "offset": r.standard_normal(2000) + 1e8,
        "one": np.array([4.0]),
        "large": L.rng(202).standard_normal(2**15 + 5),
        "large2d": L.rng(203).standard_normal((40000, 3)),
    }
    for dn, src in data.items():
        axes = (None, 0, 1, -1, (0, 1)) if src.ndim == 2 else (None, 0)
        for axis in axes:
            ax = f"axis={axis}"
            x = lambda src=src: a(src)  # noqa: E731
            yield f"describe {dn} {ax}", lambda x=x, axis=axis: st.describe(x(), axis=axis)
            yield f"skew {dn} {ax}", lambda x=x, axis=axis: st.skew(x(), axis=axis)
            yield f"kurtosis {dn} {ax}", lambda x=x, axis=axis: st.kurtosis(x(), axis=axis)
            yield f"sem {dn} {ax}", lambda x=x, axis=axis: st.sem(x(), axis=axis)
            yield f"zscore {dn} {ax}", lambda x=x, axis=axis: st.zscore(x(), axis=axis)
            yield f"iqr {dn} {ax}", lambda x=x, axis=axis: st.iqr(x(), axis=axis)
            yield f"mode {dn} {ax}", lambda x=x, axis=axis: st.mode(x(), axis=axis)
            yield f"differential_entropy {dn} {ax}", lambda x=x, axis=axis: st.differential_entropy(x(), axis=axis)
            for order in (1, 2, 3, 4):
                yield (
                    f"moment {dn} {ax} order={order}",
                    lambda x=x, axis=axis, o=order: st.moment(x(), order=o, axis=axis),
                )
            if axis is None or isinstance(axis, int):
                yield f"rankdata {dn} {ax}", lambda x=x, axis=axis: st.rankdata(x(), axis=axis)
        for pol in ("omit", "raise"):
            x = lambda src=src: a(src)  # noqa: E731
            yield f"describe {dn} nan_policy={pol}", lambda x=x, pol=pol: st.describe(x(), axis=0, nan_policy=pol)
            yield f"skew {dn} nan_policy={pol}", lambda x=x, pol=pol: st.skew(x(), axis=0, nan_policy=pol)
            yield f"rankdata {dn} nan_policy={pol}", lambda x=x, pol=pol: st.rankdata(x(), axis=0, nan_policy=pol)
            yield f"mode {dn} nan_policy={pol}", lambda x=x, pol=pol: st.mode(x(), axis=0, nan_policy=pol)
            yield f"iqr {dn} nan_policy={pol}", lambda x=x, pol=pol: st.iqr(x(), axis=0, nan_policy=pol)
            yield (
                f"differential_entropy {dn} nan_policy={pol}",
                lambda x=x, pol=pol: st.differential_entropy(x(), axis=0, nan_policy=pol),
            )
    x2 = lambda: a(data["ties"])  # noqa: E731
    xn = lambda: a(data["nan"])  # noqa: E731
    for corr in (0, 1, 2, 0.5):
        yield f"describe correction={corr}", lambda c=corr: st.describe(x2(), axis=1, correction=c)
        yield f"sem correction={corr}", lambda c=corr: st.sem(x2(), axis=1, correction=c)
        yield f"zscore correction={corr}", lambda c=corr: st.zscore(x2(), axis=1, correction=c)
    for bias in (True, False):
        yield f"describe bias={bias}", lambda b=bias: st.describe(x2(), axis=1, bias=b)
        yield f"skew bias={bias}", lambda b=bias: st.skew(x2(), axis=1, bias=b)
        for fisher in (True, False):
            yield (
                f"kurtosis bias={bias} fisher={fisher}",
                lambda b=bias, f=fisher: st.kurtosis(x2(), axis=1, bias=b, fisher=f),
            )
    for keep in (False, True):
        for fn in ("skew", "kurtosis", "sem", "iqr", "mode", "differential_entropy"):
            yield f"{fn} keepdims={keep}", lambda fn=fn, k=keep: getattr(st, fn)(x2(), axis=1, keepdims=k)
        yield f"moment keepdims={keep}", lambda k=keep: st.moment(x2(), order=2, axis=1, keepdims=k)
        yield f"entropy keepdims={keep}", lambda k=keep: st.entropy(sd.abs(x2()), axis=1, keepdims=k)
    for center in (None, 0.0, 1.5):
        yield f"moment center={center}", lambda c=center: st.moment(x2(), order=3, center=c, axis=0)
    yield "moment order tuple", lambda: st.moment(x2(), order=(1, 2, 3), axis=0)
    yield "moment order 0", lambda: st.moment(x2(), order=0)
    yield "err moment order empty", lambda: st.moment(x2(), order=[])
    yield "err moment order missing", lambda: st.moment(x2())
    yield "err moment order float", lambda: st.moment(x2(), order=1.5)
    for method in ("average", "min", "max", "dense", "ordinal", "x"):
        yield f"rankdata method={method}", lambda m=method: st.rankdata(x2(), method=m, axis=1)
        yield f"rankdata nan method={method}", lambda m=method: st.rankdata(xn(), method=m, axis=0, nan_policy="omit")
    for method in ("linear", "lower", "higher", "midpoint", "nearest", "x"):
        yield f"iqr method={method}", lambda m=method: st.iqr(x2(), method=m, axis=1)
    for q in ((0.25, 0.75), (0.1, 0.9), (0.0, 1.0), (0.75, 0.25), (0.5,), (-0.1, 0.5)):
        yield f"iqr q={q}", lambda q=q: st.iqr(x2(), q=q, axis=0)
    for scale in (1.0, 2.0, 0.0, "normal", -1.0):
        yield f"iqr scale={scale}", lambda s=scale: st.iqr(x2(), scale=s, axis=0)
    for method in ("auto", "vasicek", "van es", "ebrahimi", "correa", "Vasicek"):
        for wl in (None, 1, 3, 20):
            yield (
                f"differential_entropy method={method} window={wl}",
                lambda m=method, w=wl: st.differential_entropy(a(data["1d"]), method=m, window_length=w),
            )
    for base in (None, 2.0, 10.0, 1.0, 0.0):
        yield f"differential_entropy base={base}", lambda b=base: st.differential_entropy(a(data["1d"]), base=b)
        yield f"entropy base={base}", lambda b=base: st.entropy(a(np.array([0.2, 0.3, 0.5, 0.0])), base=b)
    pk = np.array([[0.2, 0.3, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 2.0, 4.0, 1.0]])
    qk = np.array([[0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    for axis in (None, 0, 1):
        yield f"entropy pk axis={axis}", lambda ax=axis: st.entropy(a(pk), axis=ax)
        yield f"entropy pk qk axis={axis}", lambda ax=axis: st.entropy(a(pk), a(qk), axis=ax)
    yield "entropy negative", lambda: st.entropy(a(np.array([0.5, -0.1, 0.6])))
    yield "entropy nan omit", lambda: st.entropy(a(np.array([0.5, NAN, 0.5])), nan_policy="omit")
    yield "entropy large", lambda: st.entropy(a(L.rng(204).random(2**15 + 5)))
    yield "err entropy shapes", lambda: st.entropy(a(pk), a(qk[:, :3]))
    yield "describe empty", lambda: st.describe(sd.zeros((0,)))
    yield "skew empty axis", lambda: st.skew(sd.zeros((0, 3)), axis=0)
    yield "rankdata empty", lambda: st.rankdata(sd.zeros((0,)))
    yield "mode empty", lambda: st.mode(sd.zeros((0,)))
    yield "describe ints", lambda: st.describe(sd.asarray([1, 2, 2, 7]))
    yield "rankdata ints", lambda: st.rankdata(sd.asarray([3, 1, 3, 2]))
    yield "mode bool", lambda: st.mode(sd.asarray([True, False, True]))
    yield "err describe complex", lambda: st.describe(sd.asarray([1j]))
    yield "err skew list", lambda: st.skew([1.0, 2.0])
    yield "err skew nan_policy", lambda: st.skew(x2(), nan_policy="x")
    yield "err skew nan_policy None", lambda: st.skew(x2(), nan_policy=None)
    yield "err skew axis repeated", lambda: st.skew(x2(), axis=(0, 0))
    yield "err skew axis float", lambda: st.skew(x2(), axis=1.0)
    yield "err rankdata axis tuple", lambda: st.rankdata(x2(), axis=(0, 1))
    yield "err rankdata axis float", lambda: st.rankdata(x2(), axis=1.0)
    yield "err describe correction str", lambda: st.describe(x2(), correction="a")
    yield "err describe bias int", lambda: st.describe(x2(), bias=1)
    yield "err nan raise", lambda: st.describe(xn(), nan_policy="raise")
    yield "err differential_entropy window", lambda: st.differential_entropy(a(data["1d"]), window_length=0)
    yield "err differential_entropy window big", lambda: st.differential_entropy(a(data["1d"]), window_length=13)
