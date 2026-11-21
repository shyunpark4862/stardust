"""Golden cases for sd.regression (tools/golden.py has the contract).

Every fitting function of __all__ on a few fixed data sets (well conditioned, collinear,
heavy-tailed, with NaN, batched, large enough for the multithreaded kernels, frame input),
over its options (cov_type with maxlags and groups, families and links, penalties and paths,
cross-validation, norms and scales, quantiles, bandwidths, nan_policy, rtol), and the tests on
fitted models (linear hypotheses, comparisons, anova, the diagnostic tests) and the variable
selections (stepwise and best_subset over their models, directions and criteria). A result object
is stored whole: every public field and every method without arguments (summary(),
cov_params(), conf_int(), jarque_bera(), breusch_pagan(), ...); predict and predict_interval
get new rows. nls runs the 27 NIST StRD problems from both starts with both methods, with
differences and analytic Jacobians, and bounded, plus weighted, capped and non-finite cases.
"""

import math

import numpy as np

from tests.regression import nist_nls as N

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {
    "OLSResult": "a result class: the cases of the function that returns it store it whole",
    "WLSResult": "a result class: the cases of the function that returns it store it whole",
    "GLSResult": "a result class: the cases of the function that returns it store it whole",
    "GLMResult": "a result class: the cases of the function that returns it store it whole",
    "LinregressResult": "a result class: the cases of the function that returns it store it whole",
    "JarqueBeraResult": "a result class: the cases of the function that returns it store it whole",
    "BreuschPaganResult": "a result class: the cases of the function that returns it store it whole",
    "RidgeResult": "a result class: the cases of the function that returns it store it whole",
    "LassoResult": "a result class: the cases of the function that returns it store it whole",
    "ElasticNetResult": "a result class: the cases of the function that returns it store it whole",
    "RidgeCVResult": "a result class: the cases of the function that returns it store it whole",
    "LassoCVResult": "a result class: the cases of the function that returns it store it whole",
    "ElasticNetCVResult": "a result class: the cases of the function that returns it store it whole",
    "PenalizedGLMResult": "a result class: the cases of the function that returns it store it whole",
    "RLMResult": "a result class: the cases of the function that returns it store it whole",
    "TheilslopesResult": "a result class: the cases of the function that returns it store it whole",
    "SiegelslopesResult": "a result class: the cases of the function that returns it store it whole",
    "QuantRegResult": "a result class: the cases of the function that returns it store it whole",
    "NLSResult": "a result class: the cases of the function that returns it store it whole",
    "TTestResult": "a result class: the cases of the function that returns it store it whole",
    "WaldTestResult": "a result class: the cases of the function that returns it store it whole",
    "CompareFTestResult": "a result class: the cases of the function that returns it store it whole",
    "CompareLRTestResult": "a result class: the cases of the function that returns it store it whole",
    "AnovaResult": "a result class: the cases of the function that returns it store it whole",
    "WhiteTestResult": "a result class: the cases of the function that returns it store it whole",
    "GoldfeldQuandtResult": "a result class: the cases of the function that returns it store it whole",
    "ResetTestResult": "a result class: the cases of the function that returns it store it whole",
    "StepwiseResult": "a result class: the cases of the function that returns it store it whole",
    "BestSubsetResult": "a result class: the cases of the function that returns it store it whole",
}

NAN, INF = math.nan, math.inf


def data(L):
    r = L.rng(301)
    n, k = 60, 3
    X = r.standard_normal((n, k))
    beta = np.array([1.5, -2.0, 0.5])
    y = 0.7 + X @ beta + 0.5 * r.standard_normal(n)
    d = {"X": X, "y": y}
    d["w"] = r.uniform(0.2, 3.0, n)
    d["w0"] = d["w"].copy()
    d["w0"][[3, 17]] = 0.0
    rho = 0.6
    d["sigma"] = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    d["sigma_diag"] = r.uniform(0.5, 2.0, n)
    d["groups"] = np.repeat(np.arange(12), 5)
    Xc = X.copy()
    Xc[:, 2] = Xc[:, 0] + Xc[:, 1]  # exactly collinear
    d["Xc"] = Xc
    d["yt"] = 0.7 + X @ beta + r.standard_t(1.5, n)  # heavy tails
    d["yt"][[5, 40]] += [30.0, -25.0]
    ynan = y.copy()
    ynan[[2, 9]] = NAN
    d["ynan"] = ynan
    eta = 0.3 + X @ np.array([0.8, -0.6, 0.2])
    p = 1 / (1 + np.exp(-eta))
    d["ybin"] = (r.random(n) < p).astype(np.float64)
    d["ypois"] = r.poisson(np.exp(0.5 * eta)).astype(np.float64)
    d["ygamma"] = r.gamma(2.0, np.exp(0.2 * eta) / 2.0)
    d["trials"] = r.integers(1, 10, n).astype(np.float64)
    d["yprop"] = np.minimum(r.binomial(d["trials"].astype(int), p) / d["trials"], 1.0)
    d["offset"] = r.uniform(-0.5, 0.5, n)
    d["exposure"] = r.uniform(0.5, 2.0, n)
    d["Xnew"] = r.standard_normal((5, k))
    # batch: 4 regressions sharing X
    d["yb"] = 0.7 + (X @ beta)[None, :] + r.standard_normal((4, n))
    # large: the multithreaded paths
    rl = L.rng(302)
    nl = 2**15 + 7
    d["XL"] = rl.standard_normal((nl, 4))
    d["yL"] = d["XL"] @ np.array([1.0, -1.0, 2.0, 0.5]) + rl.standard_normal(nl)
    d["ybinL"] = (rl.random(nl) < 1 / (1 + np.exp(-d["XL"][:, 0]))).astype(np.float64)
    # simple regression
    d["x1"] = np.sort(r.uniform(0, 10, 30))
    d["y1"] = 2.0 - 0.5 * d["x1"] + r.standard_normal(30)
    d["y1"][[4, 22]] += [6.0, -8.0]
    return d


def cases(L):
    yield from linear(L)
    yield from glm(L)
    yield from penalized(L)
    yield from robust(L)
    yield from quantile(L)
    yield from nls(L)
    yield from model_tests(L)
    yield from selection(L)


def fit_and_predict(L, res, Xnew):
    """The result object and what its prediction methods give on new rows."""
    out = {"result": res}
    for name, call in (
        ("predict", lambda: res.predict(Xnew)),
        ("predict_interval", lambda: res.predict_interval(Xnew)),
        ("predict_interval obs", lambda: res.predict_interval(Xnew, kind="obs", alpha=0.1)),
        ("conf_int 0.01", lambda: res.conf_int(alpha=0.01)),
        ("summary 0.1", lambda: res.summary(alpha=0.1)),
    ):
        try:
            out[name] = call()
        except Exception as e:  # noqa: BLE001
            out[name] = e
    return out


def linear(L):
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, y, Xn = (lambda: a(D["X"])), (lambda: a(D["y"])), (lambda: a(D["Xnew"]))
    fits = {
        "ols": lambda **kw: R.ols(y(), X(), **kw),
        "wls": lambda **kw: R.wls(y(), X(), weights=a(D["w"]), **kw),
        "wls zero weights": lambda **kw: R.wls(y(), X(), weights=a(D["w0"]), **kw),
        "gls": lambda **kw: R.gls(y(), X(), sigma=a(D["sigma"]), **kw),
        "gls diag": lambda **kw: R.gls(y(), X(), sigma=a(np.diag(D["sigma_diag"])), **kw),
    }
    covs = [
        ("nonrobust", {}),
        ("HC0", {}),
        ("HC1", {}),
        ("HC2", {}),
        ("HC3", {}),
        ("HAC", {}),
        ("HAC maxlags=3", {"maxlags": 3}),
        ("cluster", {"groups": a(D["groups"])}),
        ("cluster list", {"groups": [f"g{v % 7}" for v in range(60)]}),
    ]
    for fn, fit in fits.items():
        for cname, kw in covs:
            ct = cname.split()[0]
            yield f"{fn} {cname}", lambda fit=fit, ct=ct, kw=kw: fit_and_predict(L, fit(cov_type=ct, **kw), Xn())
        yield f"{fn} no intercept", lambda fit=fit: fit_and_predict(L, fit(intercept=False), Xn())
        yield (
            f"{fn} nan omit",
            lambda fn=fn: fit_and_predict(L, _with_y(R, fn, a(D["ynan"]), X(), D, a, nan_policy="omit"), Xn()),
        )
        yield f"{fn} nan propagate", lambda fn=fn: _with_y(R, fn, a(D["ynan"]), X(), D, a)
        yield f"{fn} nan raise", lambda fn=fn: _with_y(R, fn, a(D["ynan"]), X(), D, a, nan_policy="raise")
        yield f"{fn} collinear", lambda fn=fn: fit_and_predict(L, _with_y(R, fn, y(), a(D["Xc"]), D, a), Xn())
        yield f"{fn} collinear rtol", lambda fn=fn: _with_y(R, fn, y(), a(D["Xc"]), D, a, rtol=1e-3)
        yield f"{fn} batch", lambda fn=fn: fit_and_predict(L, _with_y(R, fn, a(D["yb"]), X(), D, a), Xn())
    yield "ols large", lambda: R.ols(a(D["yL"]), a(D["XL"]))
    yield "ols large HC3", lambda: R.ols(a(D["yL"]), a(D["XL"]), cov_type="HC3").bse
    yield "ols large HAC", lambda: R.ols(a(D["yL"]), a(D["XL"]), cov_type="HAC").bse
    yield "ols frame", lambda: _frame_fit(L, D, R.ols)
    yield "ols 1 column X", lambda: R.ols(y(), a(D["X"][:, :1]))
    yield "ols const y", lambda: R.ols(sd.full((60,), 3.0), X())
    yield "ols empty", lambda: R.ols(sd.zeros((0,)), sd.zeros((0, 3)))
    yield "ols n < k", lambda: R.ols(a(D["y"][:3]), a(D["X"][:3]))
    yield "ols inf", lambda: R.ols(a(np.where(np.arange(60) == 7, INF, D["y"])), X())
    yield "err ols rows", lambda: R.ols(a(D["y"][:59]), X())
    yield "err ols cov_type", lambda: R.ols(y(), X(), cov_type="x")
    yield "err ols rtol", lambda: R.ols(y(), X(), rtol=-1.0)
    yield "err ols maxlags", lambda: R.ols(y(), X(), cov_type="HAC", maxlags=-1)
    yield "err ols groups missing", lambda: R.ols(y(), X(), cov_type="cluster")
    yield "err ols list", lambda: R.ols([1.0, 2.0], X())
    yield "err ols complex", lambda: R.ols(sd.asarray([1j] * 60), X())
    yield "err ols keyword y", lambda: R.ols(y=y(), X=X())
    yield "err wls weights negative", lambda: R.wls(y(), X(), weights=a(-D["w"]))
    yield "err gls sigma not pd", lambda: R.gls(y(), X(), sigma=a(-D["sigma"]))
    yield "err result setattr", lambda: setattr(R.ols(y(), X()), "cov_type", "HC0")
    yield "ols breusch_pagan studentize False", lambda: R.ols(y(), X()).breusch_pagan(studentize=False)
    yield "ols param_names copy", lambda: (lambda r: (r.param_names.append("z"), r.param_names)[1])(R.ols(y(), X()))
    # linregress
    x1, y1 = (lambda: a(D["x1"])), (lambda: a(D["y1"]))
    yield "linregress", lambda: R.linregress(x1(), y1())
    yield (
        "linregress nan omit",
        lambda: R.linregress(a(np.where(np.arange(30) == 3, NAN, D["x1"])), y1(), nan_policy="omit"),
    )
    yield "linregress const x", lambda: R.linregress(sd.full((30,), 2.0), y1())
    yield "linregress two points", lambda: R.linregress(sd.asarray([1.0, 2.0]), sd.asarray([3.0, 5.0]))
    yield "linregress one point", lambda: R.linregress(sd.asarray([1.0]), sd.asarray([3.0]))
    yield "linregress large", lambda: R.linregress(a(D["XL"][:, 0]), a(D["yL"]))
    yield "err linregress lengths", lambda: R.linregress(x1(), a(D["y1"][:29]))


def _with_y(R, fn, y, X, D, a, **kw):
    base = fn.split()[0]
    if base == "ols":
        return R.ols(y, X, **kw)
    if base == "wls":
        return R.wls(y, X, weights=a(D["w0"] if "zero" in fn else D["w"]), **kw)
    sigma = np.diag(D["sigma_diag"]) if "diag" in fn else D["sigma"]
    return R.gls(y, X, sigma=a(sigma), **kw)


def _frame_fit(L, D, fn):
    F = L.sd.frame
    df = F.DataFrame({"a": D["X"][:, 0].tolist(), "b": D["X"][:, 1].tolist(), "c": D["X"][:, 2].tolist()})
    ys = F.Series(D["y"].tolist(), name="y")
    r = fn(ys, df)
    return {"result": r, "predict": r.predict(df.iloc[[0, 1, 2]])}


def glm(L):
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, Xn = (lambda: a(D["X"])), (lambda: a(D["Xnew"]))
    fams = {
        "gaussian": ("y", [None, "identity", "log", "inverse"]),  # inverse: an error
        "binomial": ("ybin", [None, "logit", "probit", "cloglog", "log"]),
        "poisson": ("ypois", [None, "log", "identity", "sqrt"]),
        "gamma": ("ygamma", [None, "inverse", "log", "identity"]),
    }
    for fam, (yk, links) in fams.items():
        for link in links:
            yield (
                f"glm {fam} link={link}",
                lambda fam=fam, yk=yk, link=link: fit_and_predict(L, R.glm(a(D[yk]), X(), family=fam, link=link), Xn()),
            )
        for ct, kw in (("HC0", {}), ("HC1", {}), ("HAC", {"maxlags": 2}), ("cluster", {"groups": a(D["groups"])})):
            yield (
                f"glm {fam} cov_type={ct}",
                lambda fam=fam, yk=yk, ct=ct, kw=kw: R.glm(a(D[yk]), X(), family=fam, cov_type=ct, **kw),
            )
        yield f"glm {fam} var_weights", lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, var_weights=a(D["w"]))
        yield (
            f"glm {fam} freq_weights",
            lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, freq_weights=a(np.round(D["w"]))),
        )
        yield f"glm {fam} offset", lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, offset=a(D["offset"]))
        yield f"glm {fam} no intercept", lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, intercept=False)
        yield f"glm {fam} max_iter=2", lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, max_iter=2)
        yield f"glm {fam} scale", lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam, scale=1.5)
        yield f"glm {fam} collinear", lambda fam=fam, yk=yk: R.glm(a(D[yk]), a(D["Xc"]), family=fam)
    yield "glm binomial trials", lambda: R.glm(a(D["yprop"]), X(), family="binomial", trials=a(D["trials"]))
    yield "glm poisson exposure", lambda: R.glm(a(D["ypois"]), X(), family="poisson", exposure=a(D["exposure"]))
    yield "glm binomial separation", lambda: R.glm(a((D["X"][:, 0] > 0).astype(float)), X(), family="binomial")
    yield "glm binomial large", lambda: R.glm(a(D["ybinL"]), a(D["XL"]), family="binomial").params
    yield "glm nan omit", lambda: R.glm(a(D["ynan"]), X(), family="gaussian", nan_policy="omit")
    yield "glm batch", lambda: R.glm(a(D["yb"]), X(), family="gaussian")
    yield "err glm family", lambda: R.glm(a(D["y"]), X(), family="x")
    yield "err glm link", lambda: R.glm(a(D["ybin"]), X(), family="binomial", link="identity")
    yield "err glm y range", lambda: R.glm(a(D["y"]), X(), family="binomial")
    yield "err glm family object", lambda: R.glm(a(D["y"]), X(), family=None)


def penalized(L):
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, y, Xn = (lambda: a(D["X"])), (lambda: a(D["y"])), (lambda: a(D["Xnew"]))
    for fn in ("ridge", "lasso", "elastic_net"):
        f = getattr(R, fn)
        for alpha in (0.0, 0.01, 0.3, 5.0, 1e6):
            yield f"{fn} alpha={alpha}", lambda f=f, al=alpha: fit_and_predict(L, f(y(), X(), alpha=al), Xn())
        yield f"{fn} path", lambda f=f: f(y(), X(), alpha=a(np.array([2.0, 0.5, 0.1, 0.01])))
        yield f"{fn} standardize", lambda f=f: f(y(), X(), alpha=0.1, standardize=True)
        yield f"{fn} weights", lambda f=f: f(y(), X(), alpha=0.1, weights=a(D["w0"]))
        yield f"{fn} no intercept", lambda f=f: f(y(), X(), alpha=0.1, intercept=False)
        yield f"{fn} collinear", lambda f=f: f(y(), a(D["Xc"]), alpha=0.1)
        yield f"{fn} nan omit", lambda f=f: f(a(D["ynan"]), X(), alpha=0.1, nan_policy="omit")
        yield f"{fn} batch", lambda f=f: f(a(D["yb"]), X(), alpha=0.1)
        yield f"{fn} large", lambda f=f: f(a(D["yL"]), a(D["XL"]), alpha=0.05).params
        yield f"err {fn} alpha negative", lambda f=f: f(y(), X(), alpha=-1.0)
        yield f"err {fn} alpha missing", lambda f=f: f(y(), X())
    for l1 in (0.0, 0.25, 1.0):
        yield f"elastic_net l1_ratio={l1}", lambda l1=l1: R.elastic_net(y(), X(), alpha=0.1, l1_ratio=l1)
    yield "lasso max_iter=1", lambda: R.lasso(y(), X(), alpha=0.01, max_iter=1)
    yield "lasso tol", lambda: R.lasso(y(), X(), alpha=0.01, tol=1e-12)
    yield "err elastic_net l1_ratio", lambda: R.elastic_net(y(), X(), alpha=0.1, l1_ratio=1.5)
    rng = sd.random.default_rng
    for fn in ("ridge_cv", "lasso_cv", "elastic_net_cv"):
        f = getattr(R, fn)
        # 5 folds given (ridge_cv's default is cv='gcv', D-18)
        yield f"{fn}", lambda f=f: fit_and_predict(L, f(y(), X(), rng=rng(5), cv=5), Xn())
        yield f"{fn} 1se", lambda f=f: f(y(), X(), rng=rng(6), cv=5, select="1se")
        yield f"{fn} cv=3 alphas=10", lambda f=f: f(y(), X(), rng=rng(7), cv=3, alphas=10)
        yield f"{fn} alphas array", lambda f=f: f(y(), X(), rng=rng(8), cv=5, alphas=a(np.array([1.0, 0.1, 0.01])))
        yield (
            f"{fn} standardize weights",
            lambda f=f: f(y(), X(), rng=rng(9), cv=5, standardize=True, weights=a(D["w"])),
        )
        yield f"err {fn} rng int", lambda f=f: f(y(), X(), rng=1, cv=5)
        yield f"err {fn} cv too big", lambda f=f: f(y(), X(), rng=rng(1), cv=61)
    yield "ridge_cv gcv", lambda: R.ridge_cv(y(), X(), cv="gcv")
    yield "ridge_cv default", lambda: R.ridge_cv(y(), X())
    yield "err ridge_cv no rng", lambda: R.ridge_cv(y(), X(), cv=5)
    yield "err ridge_cv gcv rng", lambda: R.ridge_cv(y(), X(), cv="gcv", rng=rng(1))
    yield "err ridge_cv gcv 1se", lambda: R.ridge_cv(y(), X(), cv="gcv", select="1se")
    yield "err ridge_cv cv str", lambda: R.ridge_cv(y(), X(), cv="x")
    for fam, yk in (("binomial", "ybin"), ("poisson", "ypois")):
        for l1 in (0.0, 0.5, 1.0):
            yield (
                f"penalized_glm {fam} l1_ratio={l1}",
                lambda fam=fam, yk=yk, l1=l1: fit_and_predict(
                    L, R.penalized_glm(a(D[yk]), X(), family=fam, alpha=0.05, l1_ratio=l1), Xn()
                ),
            )
        yield (
            f"penalized_glm {fam} path",
            lambda fam=fam, yk=yk: R.penalized_glm(a(D[yk]), X(), family=fam, alpha=a(np.array([0.5, 0.05, 0.005]))),
        )
        yield (
            f"penalized_glm {fam} standardize weights",
            lambda fam=fam, yk=yk: R.penalized_glm(
                a(D[yk]), X(), family=fam, alpha=0.05, standardize=True, weights=a(D["w"])
            ),
        )
        yield (
            f"penalized_glm {fam} all zero",
            lambda fam=fam: R.penalized_glm(sd.zeros((60,)), X(), family=fam, alpha=0.05),
        )
    yield (
        "penalized_glm predict linear",
        lambda: R.penalized_glm(a(D["ybin"]), X(), family="binomial", alpha=0.05).predict(Xn(), which="linear"),
    )
    yield "err penalized_glm family", lambda: R.penalized_glm(a(D["ybin"]), X(), family="gamma", alpha=0.1)
    yield "err penalized_glm y", lambda: R.penalized_glm(a(D["y"]), X(), family="binomial", alpha=0.1)


def robust(L):
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, yt, Xn = (lambda: a(D["X"])), (lambda: a(D["yt"])), (lambda: a(D["Xnew"]))
    for norm in (None, "huber", "bisquare", "hampel"):
        for scale in (None, "mad", "huber"):
            yield (
                f"rlm M norm={norm} scale_est={scale}",
                lambda norm=norm, scale=scale: fit_and_predict(L, R.rlm(yt(), X(), norm=norm, scale_est=scale), Xn()),
            )
    yield "rlm M c", lambda: R.rlm(yt(), X(), norm="huber", c=2.0)
    yield "rlm M hampel c", lambda: R.rlm(yt(), X(), norm="hampel", c=(1.5, 3.0, 6.0))
    yield "rlm M max_iter=2", lambda: R.rlm(yt(), X(), max_iter=2)
    yield "rlm M no intercept", lambda: R.rlm(yt(), X(), intercept=False)
    yield "rlm M collinear", lambda: R.rlm(yt(), a(D["Xc"]))
    yield "rlm M nan omit", lambda: R.rlm(a(D["ynan"]), X(), nan_policy="omit")
    yield "rlm M batch", lambda: R.rlm(a(D["yb"]), X())
    yield "rlm M large", lambda: R.rlm(a(D["yL"]), a(D["XL"])).params
    for seed in (1, 2):
        yield (
            f"rlm MM seed={seed}",
            lambda seed=seed: fit_and_predict(L, R.rlm(yt(), X(), method="MM", rng=sd.random.default_rng(seed)), Xn()),
        )
    yield "rlm MM c", lambda: R.rlm(yt(), X(), method="MM", c=4.0, rng=sd.random.default_rng(3))
    yield "err rlm MM no rng", lambda: R.rlm(yt(), X(), method="MM")
    yield "err rlm M rng", lambda: R.rlm(yt(), X(), rng=sd.random.default_rng(1))
    yield "err rlm MM huber", lambda: R.rlm(yt(), X(), method="MM", norm="huber", rng=sd.random.default_rng(1))
    yield "err rlm MM scale_est", lambda: R.rlm(yt(), X(), method="MM", scale_est="mad", rng=sd.random.default_rng(1))
    yield "err rlm norm", lambda: R.rlm(yt(), X(), norm="x")
    x1, y1 = (lambda: a(D["x1"])), (lambda: a(D["y1"]))
    for alpha in (0.95, 0.9, 0.05):
        for method in ("separate", "joint"):
            yield (
                f"theilslopes alpha={alpha} {method}",
                lambda al=alpha, m=method: R.theilslopes(y1(), x1(), alpha=al, method=m),
            )
    yield "theilslopes no x", lambda: R.theilslopes(y1())
    yield (
        "theilslopes nan omit",
        lambda: R.theilslopes(a(np.where(np.arange(30) == 3, NAN, D["y1"])), x1(), nan_policy="omit"),
    )
    yield "theilslopes large", lambda: R.theilslopes(a(D["yL"][:3000]), a(D["XL"][:3000, 0]))
    yield (
        "theilslopes ties",
        lambda: R.theilslopes(sd.asarray([1.0, 1.0, 2.0, 2.0, 3.0]), sd.asarray([0.0, 0.0, 1.0, 1.0, 2.0])),
    )
    for method in ("hierarchical", "separate"):
        yield f"siegelslopes {method}", lambda m=method: R.siegelslopes(y1(), x1(), method=m)
    yield "siegelslopes no x", lambda: R.siegelslopes(y1())
    yield "siegelslopes large", lambda: R.siegelslopes(a(D["yL"][:2000]), a(D["XL"][:2000, 0]))
    yield "err theilslopes method", lambda: R.theilslopes(y1(), x1(), method="x")
    yield "err theilslopes alpha big int", lambda: R.theilslopes(y1(), x1(), alpha=10**400)
    yield "theilslopes default alpha", lambda: R.theilslopes(y1(), x1())
    yield "theilslopes one point", lambda: R.theilslopes(sd.asarray([1.0]), sd.asarray([2.0]))
    yield "siegelslopes one point", lambda: R.siegelslopes(sd.asarray([1.0]))
    yield "err siegelslopes lengths", lambda: R.siegelslopes(y1(), a(D["x1"][:5]))


def quantile(L):
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, y, Xn = (lambda: a(D["X"])), (lambda: a(D["yt"])), (lambda: a(D["Xnew"]))
    for q in (0.5, 0.1, 0.9, 0.3):
        for ct in ("nid", "iid", "ker"):
            for bw in ("hall-sheather", "bofinger"):
                yield (
                    f"quantreg q={q} {ct} {bw}",
                    lambda q=q, ct=ct, bw=bw: fit_and_predict(
                        L, R.quantreg(y(), X(), q=q, cov_type=ct, bandwidth=bw), Xn()
                    ),
                )
    yield "quantreg q array", lambda: R.quantreg(y(), X(), q=a(np.array([0.1, 0.25, 0.5, 0.75, 0.9])))
    yield "quantreg no intercept", lambda: R.quantreg(y(), X(), intercept=False)
    yield "quantreg collinear", lambda: R.quantreg(y(), a(D["Xc"]))
    yield "quantreg integer y dummies", lambda: R.quantreg(a(np.round(D["yt"])), a((D["X"] > 0).astype(float)))
    yield "quantreg even median", lambda: R.quantreg(sd.asarray([1.0, 2.0, 3.0, 4.0]), sd.ones((4, 1)), intercept=False)
    yield "quantreg max_iter=1", lambda: R.quantreg(y(), X(), max_iter=1)
    yield "quantreg nan omit", lambda: R.quantreg(a(D["ynan"]), X(), nan_policy="omit")
    yield "quantreg batch", lambda: R.quantreg(a(D["yb"]), X())
    yield "quantreg large", lambda: R.quantreg(a(D["yL"][:5000]), a(D["XL"][:5000])).params
    yield "quantreg const y", lambda: R.quantreg(sd.full((60,), 2.0), X())
    yield "err quantreg q", lambda: R.quantreg(y(), X(), q=1.0)
    yield "err quantreg cov_type", lambda: R.quantreg(y(), X(), cov_type="x")
    yield "err quantreg bandwidth", lambda: R.quantreg(y(), X(), bandwidth="x")


def decay(sd):
    return lambda x, a, b, c: a * sd.exp(-b * x) + c


def decay_jac(sd):
    def jac(x, a, b, c):
        e = sd.exp(-b * x)
        return sd.stack([e, -a * x * e, x * 0.0 + 1.0], axis=1)

    return jac


def nls(L):
    sd = L.sd
    R = sd.regression
    for name in N.NAMES:
        d = N.load(name)

        def setup(name=name, d=d):
            x, y = N.data(d, sd)
            f, jac = N.model(name, sd)
            return x, y, f, jac

        for start in ("start1", "start2"):
            p0 = [float(v) for v in d[start]]
            for method in ("lm", "trf"):
                for uj in (False, True):

                    def run(setup=setup, p0=p0, method=method, uj=uj):
                        x, y, f, jac = setup()
                        return _nls_record(R.nls(f, x, y, p0=p0, jac=jac if uj else None, method=method), x)

                    yield f"nls {name}/{start}/{method}/{'jac' if uj else 'diff'}", run
            lb = [v - 0.5 * abs(v) - 1e-3 for v in p0]
            ub = [v + 2 * abs(v) + 1e-3 for v in p0]
            for uj in (False, True):

                def run_b(setup=setup, p0=p0, uj=uj, lb=lb, ub=ub):
                    x, y, f, jac = setup()
                    return _nls_record(R.nls(f, x, y, p0=p0, jac=jac if uj else None, bounds=(lb, ub)), x)

                yield f"nls {name}/{start}/bounded/{'jac' if uj else 'diff'}", run_b
    rng = L.rng(1)
    xn = np.linspace(0, 5, 40)
    yn = 3.0 * np.exp(-0.7 * xn) + 0.2 - 0.05 * xn + 0.03 * rng.standard_normal(40)
    w = rng.uniform(0.5, 2.0, 40)
    w[[2, 11, 30]] = 0.0
    f, fj = decay(sd), decay_jac(sd)
    x, y, ww = (lambda: L.arr(xn)), (lambda: L.arr(yn)), (lambda: L.arr(w))
    nonfinite = lambda x, a, b, c: sd.where(b > 3, math.nan, a * sd.exp(-b * x) + c)  # noqa: E731
    odd = {
        "b1": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], bounds=([0, 0, 0.5], [10, 10, 10])),
        "b2": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], bounds=([0, 0.9, -1], [2.8, 10, 10])),
        "b3": lambda: R.nls(f, x(), y(), p0=[1.0, 0.4, 1.0], bounds=([-INF, 0, -INF], [INF, 0.5, INF])),
        "b4": lambda: R.nls(f, x(), y(), p0=[1.0, 0.0, 1.0], bounds=(0, 10)),
        "w": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], weights=ww()),
        "wj": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], weights=ww(), jac=fj),
        "wtrf": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], weights=ww(), method="trf"),
        "maxnfev": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], max_nfev=7),
        "maxnfev_trf": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], max_nfev=9, method="trf"),
        "loose": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], ftol=1e-3, xtol=1e-3, gtol=1e-3),
        "nonfinite": lambda: R.nls(nonfinite, x(), y(), p0=[1.0, 2.9, 1.0]),
        "nonfinite_trf": lambda: R.nls(nonfinite, x(), y(), p0=[1.0, 2.9, 1.0], method="trf"),
        "start_nonfinite": lambda: R.nls(lambda x, a, b, c: x * math.nan, x(), y(), p0=[1.0, 1.0, 1.0]),
        "aliased": lambda: R.nls(lambda x, a, b, c: a * x + b * x + c, x(), y(), p0=[1.0, 1.0, 1.0]),
        "err p0 missing": lambda: R.nls(f, x(), y()),
        "err lm bounds": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], method="lm", bounds=(0, 10)),
        "err p0 outside bounds": lambda: R.nls(f, x(), y(), p0=[20.0, 1.0, 1.0], bounds=(0, 10)),
        "err method": lambda: R.nls(f, x(), y(), p0=[1.0, 1.0, 1.0], method="dogbox"),
    }
    for k, fn in odd.items():
        yield f"nls case/{k}", lambda fn=fn: _nls_record(fn(), x())


def _nls_record(r, x):
    """The result object (every public field and method without arguments) and its predictions."""
    out = {"result": r}
    try:
        out["predict"] = r.predict(x[:5])
        out["predict_interval"] = r.predict_interval(x[:5])
    except Exception as e:  # noqa: BLE001
        out["predict"] = e
    return out


def model_tests(L):
    """The tests on fitted models: t_test and wald_test (linear with each covariance, GLM, rlm,
    quantreg, nls), compare_f_test and compare_lr_test, anova (one fit of Type I and III, with
    terms, several fits), white_test, goldfeld_quandt and reset_test over their options, and the
    errors of each."""
    sd = L.sd
    R = sd.regression
    D = data(L)
    a = L.arr
    X, y, Xs = (lambda: a(D["X"])), (lambda: a(D["y"])), (lambda: a(D["X"][:, :1]))
    R2 = [[0.0, 1.0, -1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    row = [0.0, 1.0, 1.0, 0.0]
    fits = {
        "ols": lambda **kw: R.ols(y(), X(), **kw),
        "wls zero weights": lambda **kw: R.wls(y(), X(), weights=a(D["w0"]), **kw),
        "gls": lambda **kw: R.gls(y(), X(), sigma=a(D["sigma"]), **kw),
    }
    small = {
        "ols": lambda: R.ols(y(), Xs()),
        "wls zero weights": lambda: R.wls(y(), Xs(), weights=a(D["w0"])),
        "gls": lambda: R.gls(y(), Xs(), sigma=a(D["sigma"])),
    }
    for fn, fit in fits.items():
        yield f"t_test {fn}", lambda fit=fit: fit().t_test(a(R2), q=[0.5, 0.0])
        yield f"t_test {fn} one row", lambda fit=fit: fit().t_test(row, q=0.25)
        yield f"wald_test {fn}", lambda fit=fit: fit().wald_test(a(R2))
        yield f"wald_test {fn} HC3", lambda fit=fit: fit(cov_type="HC3").wald_test(a(R2), q=[0.5, 0.0])
        yield f"compare_f_test {fn}", lambda fit=fit, fn=fn: fit().compare_f_test(small[fn]())
        yield f"compare_lr_test {fn}", lambda fit=fit, fn=fn: fit().compare_lr_test(small[fn]())
        yield f"anova {fn}", lambda fit=fit: R.anova(fit())
        yield f"anova {fn} typ=3", lambda fit=fit: R.anova(fit(), typ=3)
        yield f"anova {fn} two fits", lambda fit=fit, fn=fn: R.anova(small[fn](), fit())
        yield f"white_test {fn} cross_terms=False", lambda fit=fit: fit().white_test(cross_terms=False)
        yield f"reset_test {fn} regressor", lambda fit=fit: fit().reset_test(power=(2, 3, 4), kind="regressor")
    ols = fits["ols"]
    yield "t_test ols cluster", lambda: ols(cov_type="cluster", groups=a(D["groups"])).t_test(a(R2))
    yield "wald_test ols HAC", lambda: ols(cov_type="HAC", maxlags=2).wald_test(a(R2))
    yield "t_test ols collinear", lambda: R.ols(y(), a(D["Xc"])).t_test(a(R2))
    yield "wald_test ols collinear", lambda: R.ols(y(), a(D["Xc"])).wald_test(a(R2))
    yield "wald_test ols dependent rows", lambda: ols().wald_test(a([R2[0], R2[1], [0.0, 2.0, 0.0, 1.0], [0.0] * 4]))
    yield "t_test ols batch", lambda: R.ols(a(D["yb"]), X()).t_test(a(R2))
    yield "wald_test ols batch", lambda: R.ols(a(D["yb"]), X()).wald_test(a(R2))
    yield "anova ols terms", lambda: R.anova(ols(), terms=[[0], [1, 2]])
    yield "anova ols terms typ=3", lambda: R.anova(ols(), typ=3, terms=[[0, 1], [2]])
    yield "anova ols HC1 typ=3", lambda: R.anova(ols(cov_type="HC1"), typ=3)
    mid = lambda: R.ols(y(), a(D["X"][:, :2]))  # noqa: E731
    yield "anova ols three fits", lambda: R.anova(R.ols(y(), Xs()), mid(), ols())
    yield "anova ols three fits Chisq", lambda: R.anova(R.ols(y(), Xs()), mid(), ols(), test="Chisq")
    yield "anova ols no intercept", lambda: R.anova(ols(intercept=False))
    yield "anova ols collinear", lambda: R.anova(R.ols(y(), a(D["Xc"])))
    yield "anova summary", lambda: R.anova(ols()).summary()
    yield "goldfeld_quandt ols", lambda: ols().goldfeld_quandt()
    for name, kw in (
        ("split=20", {"split": 20}),
        ("split=0.4 drop=0.2", {"split": 0.4, "drop": 0.2}),
        ("split=25 drop=10 order_by=0", {"split": 25, "drop": 10, "order_by": 0}),
        ("order_by=values less", {"order_by": a(D["X"][:, 1]), "alternative": "less"}),
        ("two-sided", {"alternative": "two-sided"}),
    ):
        yield f"goldfeld_quandt ols {name}", lambda kw=kw: ols().goldfeld_quandt(**kw)
    yield "goldfeld_quandt wls zero weights", lambda: fits["wls zero weights"]().goldfeld_quandt(split=15, drop=4)
    yield "white_test ols batch", lambda: R.ols(a(D["yb"]), X()).white_test()
    yield "reset_test ols HC3", lambda: ols(cov_type="HC3").reset_test()
    yield "reset_test ols batch", lambda: R.ols(a(D["yb"]), X()).reset_test()
    yield "white_test ols perfect fit", lambda: R.ols(a(D["X"] @ [1.0, 2.0, 3.0]), X()).white_test()
    # GLM
    for fam, yk in (("binomial", "ybin"), ("poisson", "ypois"), ("gamma", "ygamma")):
        big = lambda fam=fam, yk=yk: R.glm(a(D[yk]), X(), family=fam)  # noqa: E731
        sm = lambda fam=fam, yk=yk: R.glm(a(D[yk]), Xs(), family=fam)  # noqa: E731
        yield f"t_test glm {fam}", lambda big=big: big().t_test(a(R2))
        yield f"wald_test glm {fam}", lambda big=big: big().wald_test(a(R2))
        yield f"compare_lr_test glm {fam}", lambda big=big, sm=sm: big().compare_lr_test(sm())
        yield f"anova glm {fam}", lambda big=big: R.anova(big())
        yield f"anova glm {fam} typ=3", lambda big=big: R.anova(big(), typ=3)
        yield f"anova glm {fam} two fits", lambda big=big, sm=sm: R.anova(sm(), big())
    gbig = lambda: R.glm(a(D["ygamma"]), X(), family="gamma")  # noqa: E731
    gsmall = lambda: R.glm(a(D["ygamma"]), Xs(), family="gamma")  # noqa: E731
    yield "compare_f_test glm gamma", lambda: gbig().compare_f_test(gsmall())
    yield "anova glm gamma two fits Chisq", lambda: R.anova(gsmall(), gbig(), test="Chisq")
    # rlm, quantreg, nls
    yield "t_test rlm", lambda: R.rlm(a(D["yt"]), X()).t_test(a(R2))
    yield "wald_test rlm", lambda: R.rlm(a(D["yt"]), X()).wald_test(a(R2))
    yield "t_test quantreg", lambda: R.quantreg(a(D["yt"]), X(), q=[0.25, 0.5]).t_test(a(R2))
    yield "wald_test quantreg", lambda: R.quantreg(a(D["yt"]), X(), q=0.5).wald_test(a(R2))
    xn = np.linspace(0, 5, 40)
    yn = 3.0 * np.exp(-0.7 * xn) + 0.2 + 0.03 * L.rng(2).standard_normal(40)
    nfit = lambda: R.nls(decay(sd), a(xn), a(yn), p0=[1.0, 1.0, 1.0])  # noqa: E731
    yield "t_test nls", lambda: nfit().t_test([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]], q=[3.0, 0.0])
    yield "wald_test nls", lambda: nfit().wald_test([[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]], q=[3.0, 0.0])
    # errors
    yield "err t_test R columns", lambda: ols().t_test([[0.0, 1.0, 0.0]])
    yield "err t_test q length", lambda: ols().t_test(a(R2), q=[1.0, 2.0, 3.0])
    yield "err t_test R nan", lambda: ols().t_test([0.0, NAN, 0.0, 0.0])
    yield "err compare_f_test type", lambda: ols().compare_f_test(fits["wls zero weights"]())
    yield "err compare_f_test data", lambda: ols().compare_f_test(R.ols(a(D["yt"]), Xs()))
    yield "err compare_f_test not nested", lambda: R.ols(y(), Xs()).compare_f_test(ols())
    yield "err compare_f_test robust", lambda: ols(cov_type="HC3").compare_f_test(R.ols(y(), Xs(), cov_type="HC3"))
    yield (
        "err compare_f_test glm fixed scale",
        lambda: R.glm(a(D["ybin"]), X(), family="binomial").compare_f_test(
            R.glm(a(D["ybin"]), Xs(), family="binomial")
        ),
    )
    yield (
        "err anova glm fixed scale F",
        lambda: R.anova(
            R.glm(a(D["ybin"]), Xs(), family="binomial"), R.glm(a(D["ybin"]), X(), family="binomial"), test="F"
        ),
    )
    yield "err anova typ=2", lambda: R.anova(ols(), typ=2)
    yield "err anova rlm", lambda: R.anova(R.rlm(a(D["yt"]), X()))
    yield "err anova none", lambda: R.anova()
    yield "err anova typ1 robust", lambda: R.anova(ols(cov_type="HC1"))
    yield "err anova terms", lambda: R.anova(ols(), terms=[[0], [1]])
    yield "err anova batch fits", lambda: R.anova(R.ols(a(D["yb"]), Xs()), R.ols(a(D["yb"]), X()))
    yield "err goldfeld_quandt gls", lambda: fits["gls"]().goldfeld_quandt()
    yield "err goldfeld_quandt alternative", lambda: ols().goldfeld_quandt(alternative="x")
    yield "err reset_test power int", lambda: ols().reset_test(power=2)
    yield "err reset_test power 1", lambda: ols().reset_test(power=(1, 2))


def selection(L):
    """stepwise (each direction and criterion, ols, wls with zero weights, gls, glm, terms,
    include, start, an aliased column, NaN, max_steps) and best_subset (each criterion, the
    linear models and a GLM), and their errors. The results are stored whole (fit included)."""
    sd = L.sd
    R = sd.regression
    a = L.arr
    D = data(L)
    r = L.rng(907)
    n, k = 80, 6
    Z = r.standard_normal((n, k))
    X6 = Z + 0.4 * Z[:, [1, 2, 3, 4, 5, 0]]
    y6 = 1.0 + X6 @ np.array([0.8, 0.0, -0.45, 0.0, 0.25, 0.1]) + r.standard_normal(n)
    yb = (r.random(n) < 1 / (1 + np.exp(-(0.2 + X6[:, :4] @ np.array([0.9, 0.0, -0.5, 0.2]))))).astype(np.float64)
    w = r.uniform(0.3, 3.0, n)
    w[[3, 30]] = 0.0
    sigma = 0.4 ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    X, y = (lambda: a(X6)), (lambda: a(y6))
    for d in ("both", "forward", "backward"):
        for c in ("aic", "bic", "pvalue"):
            yield f"stepwise ols {d} {c}", lambda d=d, c=c: R.stepwise(y(), X(), direction=d, criterion=c)
        yield (
            f"stepwise ols {d} cv",
            lambda d=d: R.stepwise(y(), X(), direction=d, criterion="cv", rng=sd.random.default_rng(5)),
        )
    yield "stepwise wls zero weights", lambda: R.stepwise(y(), X(), model="wls", weights=a(w), cov_type="HC1")
    yield (
        "stepwise wls cv",
        lambda: R.stepwise(y(), X(), model="wls", weights=a(w), criterion="cv", cv=4, rng=sd.random.default_rng(2)),
    )
    yield "stepwise gls", lambda: R.stepwise(y(), X(), model="gls", sigma=a(sigma))
    yield "stepwise no intercept", lambda: R.stepwise(y(), X(), intercept=False)
    yield "stepwise terms include", lambda: R.stepwise(y(), X(), terms=[[0, 1], [2], [3, 4], [5]], include=[2])
    yield "stepwise start forward", lambda: R.stepwise(y(), X(), start=[3], direction="forward", criterion="bic")
    yield "stepwise aliased", lambda: R.stepwise(y(), a(np.c_[X6, X6[:, 0] + X6[:, 1]]))
    yield "stepwise max_steps", lambda: R.stepwise(y(), X(), max_steps=1)
    yield "stepwise pvalue alphas", lambda: R.stepwise(y(), X(), criterion="pvalue", alpha_enter=0.2, alpha_remove=0.3)
    ynan = y6.copy()
    ynan[[5]] = NAN
    Xnan = X6.copy()
    Xnan[7, 2] = NAN
    yield "stepwise nan propagate", lambda: R.stepwise(a(ynan), X())
    yield "stepwise nan omit X", lambda: R.stepwise(y(), a(Xnan), nan_policy="omit")
    yield "stepwise frame", lambda: R.stepwise(y(), sd.frame.DataFrame({f"v{j}": X6[:, j].tolist() for j in range(k)}))
    yield "stepwise glm binomial", lambda: R.stepwise(a(yb), X(), model="glm", family="binomial")
    yield "stepwise glm binomial bic", lambda: R.stepwise(a(yb), X(), model="glm", family="binomial", criterion="bic")
    yield (
        "stepwise glm binomial pvalue",
        lambda: R.stepwise(a(yb), X(), model="glm", family="binomial", criterion="pvalue"),
    )
    yield (
        "stepwise glm binomial cv",
        lambda: R.stepwise(
            a(yb), X(), model="glm", family="binomial", criterion="cv", cv=4, rng=sd.random.default_rng(3)
        ),
    )
    yield "stepwise glm poisson", lambda: R.stepwise(a(D["ypois"]), a(D["X"]), model="glm", family="poisson")
    yield "stepwise glm gamma", lambda: R.stepwise(a(D["ygamma"]), a(D["X"]), model="glm", family="gamma", link="log")
    for c in ("aic", "bic", "cp", "adjr2"):
        yield f"best_subset ols {c}", lambda c=c: R.best_subset(y(), X(), criterion=c)
    yield "best_subset ols cv", lambda: R.best_subset(y(), X(), criterion="cv", rng=sd.random.default_rng(4))
    yield "best_subset ols max_size include", lambda: R.best_subset(y(), X(), max_size=2, include=[0])
    yield "best_subset ols terms", lambda: R.best_subset(y(), X(), terms=[[0, 1], [2], [3, 4], [5]])
    yield "best_subset wls", lambda: R.best_subset(y(), X(), model="wls", weights=a(w))
    yield "best_subset gls", lambda: R.best_subset(y(), X(), model="gls", sigma=a(sigma), criterion="bic")
    yield "best_subset no intercept", lambda: R.best_subset(y(), X(), intercept=False)
    yield "best_subset glm binomial", lambda: R.best_subset(a(yb), X(), model="glm", family="binomial")
    yield (
        "best_subset glm binomial cv",
        lambda: R.best_subset(
            a(yb), X(), model="glm", family="binomial", criterion="cv", cv=4, rng=sd.random.default_rng(6)
        ),
    )
    yield "best_subset nan propagate", lambda: R.best_subset(a(ynan), X())
    yield "err stepwise model", lambda: R.stepwise(y(), X(), model="rlm")
    yield "err stepwise option", lambda: R.stepwise(y(), X(), foo=1)
    yield "err stepwise weights missing", lambda: R.stepwise(y(), X(), model="wls")
    yield "err stepwise cv rng", lambda: R.stepwise(y(), X(), criterion="cv")
    yield "err stepwise rng aic", lambda: R.stepwise(y(), X(), rng=sd.random.default_rng(1))
    yield "err stepwise alphas", lambda: R.stepwise(y(), X(), alpha_enter=0.2)
    yield "err stepwise batch", lambda: R.stepwise(a(np.stack([y6, y6])), X())
    yield "err stepwise no parameters", lambda: R.stepwise(y(), X(), intercept=False, direction="forward")
    yield (
        "err stepwise gls cv",
        lambda: R.stepwise(y(), X(), model="gls", sigma=a(sigma), criterion="cv", rng=sd.random.default_rng(1)),
    )
    yield "err best_subset cp glm", lambda: R.best_subset(a(yb), X(), model="glm", family="binomial", criterion="cp")
    yield (
        "err best_subset glm terms",
        lambda: R.best_subset(a(yb), a(np.zeros((n, 16))), model="glm", family="binomial"),
    )
    yield "err best_subset terms", lambda: R.best_subset(y(), X(), terms=[[0, 1]])
