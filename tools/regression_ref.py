"""Arb references of sd.regression (docs/design/regression.md §6), from tools/regref.py's
formulas: tests/regression/ref/<kind>.json, read by tests/regression/refdata*.py (load) and the
accuracy measurement.

    .venv/bin/python tools/regression_ref.py                  # every kind
    .venv/bin/python tools/regression_ref.py glm robust       # only these
    .venv/bin/python tools/regression_ref.py linear --jobs 4  # linear's problems on 4 processes

  linear     ols and wls: random problems (with and without the intercept, zero weights, an
             ill-conditioned one with a large residual, the contract's edge kappa 1e11) and the
             11 NIST StRD problems as a user gives them
  glm        every family and link on random data, binomial proportions with trials, poisson
             with an offset and an exposure, gamma with both kinds of weights, no intercept,
             scale='dev'
  penalized  the lasso, the elastic net and ridge (weights, standardize, no intercept, a
             constant column, an ill-conditioned design) and the penalized binomial and poisson
             GLMs; the active set given to the reference is sd's, which the reference certifies
  robust     rlm's three norms with the MAD and Huber's proposal 2 scales, Brownlee's stack
             loss, and the MM estimator's M step; the pieces of psi given to the reference are
             sd's, which the reference certifies
  model_tests  the tests on fitted models (docs/design/regression.md §3.8): t and Wald tests,
             the nested F and likelihood ratio comparisons, Type I and III tables, VIF, DFFITS,
             DFBETAS, White, Goldfeld-Quandt and RESET of OLS, WLS with zero weights, a fit
             without the intercept, a factor's dummies as one term and a design with kappa near
             1e8; and of binomial (logit, probit), poisson and gamma GLMs with their influence
             measures; the GLMs' Newton starts are sd's fits, which the reference certifies

Each file holds per problem its inputs as doubles, the reference values as [hi, lo] pairs and
"_provenance" (the Arb and the target, as the npz references' fields). A fit that does not
settle by 8192 bits stops the build. An unknown kind is a usage error (exit status 2); the
console output also goes to build/logs/regression_ref/.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import arbref  # noqa: E402
import regref  # noqa: E402

from tests.regression import refdata_glm as G  # noqa: E402
from tests.regression import refdata_penalized as P  # noqa: E402
from tests.regression import refdata_robust as Rb  # noqa: E402

OUT = ROOT / "tests" / "regression" / "ref"


# ---------------------------------------------------------------- linear


def linear_problems():
    import numpy as np

    from tests.regression import nist

    out = {}
    rng = np.random.default_rng(1)
    X = rng.standard_normal((30, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + rng.standard_normal(30)
    # cluster labels drawn after the data, so the problems' data are the earlier ones
    g = [int(v) for v in np.random.default_rng(7).integers(0, 6, 30)]
    gz = list(g)
    gz[4] = gz[17] = 9  # the zero-weight rows' own cluster
    out["ols_int"] = (y.tolist(), X.tolist(), None, True, g)
    out["ols_noint"] = (y.tolist(), X.tolist(), None, False, g)
    w = rng.uniform(0.2, 3.0, 30)
    w[[4, 17]] = 0.0
    out["wls_zero"] = (y.tolist(), X.tolist(), w.tolist(), True, gz)
    w2 = rng.uniform(0.2, 3.0, 30)
    out["wls_noint"] = (y.tolist(), X.tolist(), w2.tolist(), False, g)
    # kappa 1e5 with a residual as large as X b (R^2 = 1/2), an intercept
    n, k = 50, 4
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Xc = (U[:, :k] * 1e5 ** (-np.arange(k) / (k - 1))) @ V.T
    b = rng.standard_normal(k)
    r = U[:, k:] @ rng.standard_normal(n - k)
    yc = Xc @ b + r * (np.linalg.norm(Xc @ b) / np.linalg.norm(r))
    out["ols_cond"] = (yc.tolist(), Xc.tolist(), None, False, [int(v) for v in np.arange(n) % 7])
    # the contract's edge (docs/design/regression.md §3.1): kappa 1e11 with a residual 100 times X b (R^2 = 1e-4)
    k = 6
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((k, k)))
    Xe = (U[:, :k] * 1e11 ** (-np.arange(k) / (k - 1))) @ V.T
    b = rng.standard_normal(k)
    r = U[:, k:] @ rng.standard_normal(n - k)
    ye = Xe @ b + r * (100.0 * np.linalg.norm(Xe @ b) / np.linalg.norm(r))
    out["ols_kappa11"] = (ye.tolist(), Xe.tolist(), None, False, None)
    for name in nist.NAMES:
        d = nist.load(name)
        ys, Xs = nist.design_strings(d)
        out["nist_" + name] = (ys, Xs, None, d["intercept"], None)
    return out


def _linear_one(item):
    name, (y, X, w, intercept, groups) = item
    import scipy.stats as st

    k = len(X[0]) + (1 if intercept else 0)
    m = sum(1 for v in w if v > 0) if w is not None else len(y)
    ref, meta = regref.ols(
        y, X, weights=w, intercept=intercept, t_guess=float(st.t.ppf(0.975, max(m - k, 1))), groups=groups
    )
    vals = {key: ([list(p) for p in v] if isinstance(v, list) else list(v)) for key, v in ref.items()}
    return name, {"y": y, "X": X, "weights": w, "intercept": intercept, "groups": groups, "meta": meta, "ref": vals}


# ---------------------------------------------------------------- glm


def glm_problems():
    import numpy as np

    import stardust as sd

    out = {}
    rng = np.random.default_rng(11)
    n = 60
    g = [int(v) for v in np.random.default_rng(5).integers(0, 6, n)]
    base = dict(var_weights=None, freq_weights=None, trials=None, offset=None, intercept=True, scale=None, groups=g)
    for family, link in G.LINKS:
        X = rng.standard_normal((n, 2))
        y = G._response(rng, family, link, X)
        out[f"{family}_{link}"] = dict(base, y=y.tolist(), X=X.tolist(), family=family, link=link)
    # binomial proportions with trials
    X = rng.standard_normal((n, 2))
    tr = rng.integers(1, 12, n).astype(float)
    p = 1 / (1 + np.exp(-(0.2 + 0.7 * X[:, 0] - 0.4 * X[:, 1])))
    y = rng.binomial(tr.astype(int), p) / tr
    out["binomial_trials"] = dict(base, y=y.tolist(), X=X.tolist(), family="binomial", link="logit", trials=tr.tolist())
    # poisson with an offset and an exposure (log exposure added as glm() adds it)
    X = rng.standard_normal((n, 2))
    expo = rng.uniform(0.5, 4.0, n)
    off = 0.1 * rng.standard_normal(n)
    y = rng.poisson(expo * np.exp(off + 0.3 + 0.4 * X[:, 0] - 0.2 * X[:, 1])).astype(float)
    tot = sd.asarray(off) + sd.log(sd.asarray(expo))
    out["poisson_exposure"] = dict(
        base,
        y=y.tolist(),
        X=X.tolist(),
        family="poisson",
        link="log",
        offset=[float(v) for v in tot.tolist()],
        _offset=off.tolist(),
        _exposure=expo.tolist(),
    )
    # gamma with var_weights (one zero) and freq_weights
    X = rng.standard_normal((n, 2))
    y = rng.gamma(3.0, np.exp(0.5 + 0.4 * X[:, 0] - 0.3 * X[:, 1]) / 3.0)
    vw = rng.uniform(0.5, 2.0, n)
    vw[7] = 0.0
    fw = rng.integers(1, 4, n).astype(float)
    out["gamma_weights"] = dict(
        base, y=y.tolist(), X=X.tolist(), family="gamma", link="log", var_weights=vw.tolist(), freq_weights=fw.tolist()
    )
    # no intercept; scale from the deviance
    X = rng.standard_normal((n, 2))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.9 * X[:, 0] - 0.6 * X[:, 1])))).astype(float)
    out["logit_noint"] = dict(base, y=y.tolist(), X=X.tolist(), family="binomial", link="logit", intercept=False)
    X = rng.standard_normal((n, 2))
    y = rng.gamma(3.0, np.exp(0.5 + 0.4 * X[:, 0] - 0.3 * X[:, 1]) / 3.0)
    out["gamma_dev"] = dict(base, y=y.tolist(), X=X.tolist(), family="gamma", link="log", scale="dev")
    return out


def _glm_one(item):
    name, p = item

    r = G.fit(p)
    beta0 = [float(v) for v in r.params.tolist()]
    nb0 = 0.0
    if p["offset"] is not None and p["link"] == "log":  # the intercept-only fit's start
        nb0 = math.log(sum(p["y"]) / sum(math.exp(o) for o in p["offset"]))
    ref, meta = regref.glm(
        p["y"],
        p["X"],
        family=p["family"],
        link=p["link"],
        var_weights=p["var_weights"],
        freq_weights=p["freq_weights"],
        trials=p["trials"],
        offset=p["offset"],
        intercept=p["intercept"],
        scale=p["scale"],
        beta0=beta0,
        groups=p["groups"],
        null_beta0=nb0,
    )
    vals = {key: ([list(v) for v in x] if isinstance(x, list) else list(x)) for key, x in ref.items()}
    return name, dict(p, meta=meta, ref=vals)


# ---------------------------------------------------------------- penalized


def penalized_problems():
    import numpy as np

    out = {}
    rng = np.random.default_rng(23)
    n, p = 60, 6
    beta = np.array([1.5, 0.0, -2.0, 0.0, 0.5, 0.2])

    def lin(X):
        return X @ beta + 1.0 + rng.standard_normal(X.shape[0])

    base = dict(weights=None, intercept=True, standardize=False, family="gaussian")
    X = rng.standard_normal((n, p)) * rng.uniform(0.5, 3.0, p) + 1.5
    y = lin(X)
    w = rng.uniform(0.2, 2.0, n)
    w[[3, 11]] = 0.0
    for name, kind, alpha, rho, extra in [
        ("lasso", "lasso", 0.1, 1.0, {}),
        ("lasso_small_alpha", "lasso", 0.002, 1.0, {}),
        ("lasso_weights_std", "lasso", 0.1, 1.0, dict(weights=w.tolist(), standardize=True)),
        ("enet", "elastic_net", 0.2, 0.3, {}),
        ("enet_weights", "elastic_net", 0.05, 0.8, dict(weights=w.tolist())),
        ("enet_noint", "elastic_net", 0.2, 0.7, dict(intercept=False)),
        ("enet_std", "elastic_net", 0.3, 0.5, dict(standardize=True)),
        ("ridge", "ridge", 0.05, 0.0, {}),
        ("ridge_weights_std", "ridge", 0.5, 0.0, dict(weights=w.tolist(), standardize=True)),
        ("ridge_noint", "ridge", 0.05, 0.0, dict(intercept=False)),
    ]:
        out[name] = dict(base, kind=kind, y=y.tolist(), X=X.tolist(), alpha=alpha, l1_ratio=rho, **extra)
    # a constant column (out of the fit)
    Xc = X.copy()
    Xc[:, 1] = 2.5
    out["lasso_constant_column"] = dict(base, kind="lasso", y=y.tolist(), X=Xc.tolist(), alpha=0.05, l1_ratio=1.0)
    # an ill-conditioned design: singular values 1 .. 1e-6 after centering
    U, _ = np.linalg.qr(rng.standard_normal((n, p)))
    U -= U.mean(axis=0)
    V, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Xk = (U * np.geomspace(1.0, 1e-6, p)) @ V.T * 10.0 + 3.0
    yk = Xk @ rng.standard_normal(p) + 0.1 * rng.standard_normal(n)
    out["lasso_kappa1e6"] = dict(base, kind="lasso", y=yk.tolist(), X=Xk.tolist(), alpha=1e-4, l1_ratio=1.0)
    # an active set whose Gram block has condition number about 1e9: the polish refines with the data
    out["enet_kappa1e6_refined"] = dict(
        base, kind="elastic_net", y=yk.tolist(), X=Xk.tolist(), alpha=3e-5, l1_ratio=0.5, max_iter=100000
    )
    out["ridge_kappa1e6"] = dict(base, kind="ridge", y=yk.tolist(), X=Xk.tolist(), alpha=1e-8, l1_ratio=0.0)
    # the penalized GLMs
    Xg = rng.standard_normal((n, p))
    eta = Xg @ (0.5 * beta)
    yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    yp = rng.poisson(np.exp(0.3 * eta)).astype(float)
    for fam, yy in (("binomial", yb), ("poisson", yp)):
        for rho, alpha in ((1.0, 0.02), (0.5, 0.03), (0.0, 0.05)):
            out[f"{fam}_{rho:g}"] = dict(
                base, kind="glm", family=fam, y=yy.tolist(), X=Xg.tolist(), alpha=alpha, l1_ratio=rho
            )
    out["binomial_weights_std"] = dict(
        base,
        kind="glm",
        family="binomial",
        y=yb.tolist(),
        X=Xg.tolist(),
        alpha=0.02,
        l1_ratio=0.7,
        weights=w.tolist(),
        standardize=True,
    )
    return out


def _penalized_one(item):
    name, p = item

    r = P.fit(p)
    beta0 = [float(v) for v in r.params.tolist()]
    ref = regref.penalized(
        p["y"],
        p["X"],
        alpha=p["alpha"],
        l1_ratio=p["l1_ratio"],
        beta0=beta0,
        family=p["family"],
        weights=p["weights"],
        intercept=p["intercept"],
        standardize=p["standardize"],
    )
    vals = {key: ([list(v) for v in x] if isinstance(x, list) else list(x)) for key, x in ref.items()}
    return name, dict(p, ref=vals)


# ---------------------------------------------------------------- robust


def robust_problems():
    import numpy as np

    out = {}
    rng = np.random.default_rng(41)

    def outliers(n, p, frac, shift=8.0):
        X = rng.standard_normal((n, p)) * rng.uniform(0.5, 2.0, p) + 1.0
        y = X @ np.linspace(1.0, -1.0, p) + 0.5 + rng.standard_normal(n)
        m = int(frac * n)
        y[:m] += shift * (1 + rng.uniform(size=m))
        return y.tolist(), X.tolist()

    y60, X60 = outliers(60, 2, 0.1)
    y61, X61 = outliers(61, 3, 0.15)
    for norm in ("huber", "bisquare", "hampel"):
        for se in ("mad", "huber"):
            out[f"{norm}_{se}_even"] = dict(method="M", norm=norm, scale_est=se, intercept=True, y=y60, X=X60)
            out[f"{norm}_{se}_odd"] = dict(method="M", norm=norm, scale_est=se, intercept=True, y=y61, X=X61)
    out["huber_mad_noint"] = dict(method="M", norm="huber", scale_est="mad", intercept=False, y=y61, X=X61)
    yb, Xb = outliers(400, 4, 0.05)
    out["bisquare_mad_400"] = dict(method="M", norm="bisquare", scale_est="mad", intercept=True, y=yb, X=Xb)
    for norm in ("huber", "bisquare", "hampel"):
        out[f"stackloss_{norm}"] = dict(
            method="M",
            norm=norm,
            scale_est="mad",
            intercept=True,
            y=[float(v) for v in Rb.STACKLOSS_Y],
            X=[[float(v) for v in r] for r in Rb.STACKLOSS_X],
        )
    # MM: 20% leverage outliers
    n, p = 120, 2
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, 2.0]) + 0.5 + 0.5 * rng.standard_normal(n)
    m = int(0.2 * n)
    X[:m] = rng.normal(5.0, 0.5, (m, p))
    y[:m] = rng.normal(-10.0, 1.0, m)
    out["mm_leverage"] = dict(method="MM", norm="bisquare", intercept=True, seed=7, y=y.tolist(), X=X.tolist())
    out["mm_stackloss"] = dict(
        method="MM",
        norm="bisquare",
        intercept=True,
        seed=11,
        y=[float(v) for v in Rb.STACKLOSS_Y],
        X=[[float(v) for v in r] for r in Rb.STACKLOSS_X],
    )
    return out


def _robust_one(item):
    name, p = item

    r = Rb.fit(p)
    beta0 = [float(v) for v in r.params.tolist()]
    if p["method"] == "MM":
        # the S estimate: the certified fixed point near sd's (regref.s_estimate), rounded;
        # the M step's reference starts from it
        S = regref.s_estimate(
            p["y"],
            p["X"],
            beta0=[float(v) for v in r._raw[8].tolist()],
            scale0=float(r.scale),
            intercept=p["intercept"],
        )
        init = [float(h) for h, _ in S["params"]]
        p = dict(p, init_params=init, s_scale=float(S["scale"][0]))
        ref = regref.rlm(
            p["y"],
            p["X"],
            beta0=beta0,
            norm="bisquare",
            c=(4.685061,),
            intercept=p["intercept"],
            scale=p["s_scale"],
            init_params=init,
        )
    else:
        ref = regref.rlm(
            p["y"],
            p["X"],
            beta0=beta0,
            norm=p["norm"],
            c=Rb.C_REF[p["norm"]],
            scale_est=p["scale_est"],
            intercept=p["intercept"],
        )
    vals = {key: ([list(v) for v in x] if isinstance(x, list) else list(x)) for key, x in ref.items()}
    return name, dict(p, ref=vals)


# ---------------------------------------------------------------- tests on fitted models


def model_tests_problems():
    import numpy as np

    out = {}
    rng = np.random.default_rng(71)
    n = 40
    X = np.round(rng.standard_normal((n, 3)), 6)
    y = np.round(1.0 + X @ [0.7, -0.4, 0.1] + rng.standard_normal(n) * (0.4 + np.abs(X[:, 0])), 6)
    R = [[0, 1, -1, 0], [0, 0, 1, 1], [0, 1, 1, 0]]
    out["ols"] = dict(
        kind="lin",
        y=y,
        X=X,
        weights=None,
        intercept=True,
        R=R,
        q=[0.3, 0.0, 0.1],
        small=1,
        terms=[[0], [1], [2]],
        gq=(None, 0, None),
    )
    w = np.round(rng.uniform(0.3, 3.0, n), 6)
    w[[5, 22]] = 0.0
    out["wls_zero"] = dict(
        kind="lin",
        y=y,
        X=X,
        weights=w,
        intercept=True,
        R=[[0, 1, 1, 1]],
        q=[0.0],
        small=2,
        terms=[[0, 1], [2]],
        gq=(15, 4, 1),
    )
    Xn = np.round(rng.standard_normal((n, 3)) + 1.0, 6)
    yn = np.round(Xn @ [0.5, 0.5, -0.3] + rng.standard_normal(n), 6)
    out["noint"] = dict(
        kind="lin",
        y=yn,
        X=Xn,
        weights=None,
        intercept=False,
        R=[[1, -1, 0], [0, 0, 1]],
        q=[0.0, 0.2],
        small=2,
        terms=[[0], [1], [2]],
        gq=(None, 6, 2),
    )
    # nearly collinear columns: kappa of the design near 1e8
    x1 = np.round(rng.standard_normal(50), 6)
    x2 = x1 + np.round(1e-7 * rng.standard_normal(50), 13)
    x3 = np.round(rng.standard_normal(50), 6)
    Xc = np.column_stack([x1, x2, x3])
    yc = np.round(1.0 + x1 + 0.5 * x3 + 0.3 * rng.standard_normal(50), 6)
    out["cond"] = dict(
        kind="lin",
        y=yc,
        X=Xc,
        weights=None,
        intercept=True,
        R=[[0, 1, -1, 0], [0, 1, 1, 0]],
        q=[0.0, 1.0],
        small=1,
        terms=[[0], [1], [2]],
        gq=(None, 0, 2),
    )
    nf = 45
    g = rng.integers(0, 3, nf)
    Xf = np.column_stack([np.round(rng.standard_normal((nf, 2)), 6), (g == 1).astype(float), (g == 2).astype(float)])
    yf = np.round(0.5 + Xf @ [1.0, 0.3, 0.6, -0.2] + rng.standard_normal(nf), 6)
    out["factor"] = dict(
        kind="lin",
        y=yf,
        X=Xf,
        weights=None,
        intercept=True,
        R=[[0, 0, 0, 1, -1], [0, 1, 0, 0, 0]],
        q=[0.0, 0.0],
        small=2,
        terms=[[0], [1], [2, 3]],
        gq=(None, 5, 0),
    )
    ng = 60
    Xg = np.round(rng.standard_normal((ng, 2)), 6)
    eta = 0.3 + Xg @ [0.9, -0.6]
    Rg = [[0, 1, 1], [0, 0, 1]]
    out["glm_logit"] = dict(
        kind="glm",
        y=(rng.uniform(size=ng) < 1 / (1 + np.exp(-eta))).astype(float),
        X=Xg,
        family="binomial",
        link="logit",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_probit"] = dict(
        kind="glm",
        y=(rng.uniform(size=ng) < 1 / (1 + np.exp(-eta))).astype(float),
        X=Xg,
        family="binomial",
        link="probit",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_poisson"] = dict(
        kind="glm",
        y=rng.poisson(np.exp(eta)).astype(float),
        X=Xg,
        family="poisson",
        link="log",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    out["glm_gamma"] = dict(
        kind="glm",
        y=np.round(rng.gamma(3.0, np.exp(eta) / 3.0), 6),
        X=Xg,
        family="gamma",
        link="log",
        R=Rg,
        small=1,
        terms=[[0], [1]],
    )
    return out


def _plain(v):
    import numpy as np

    return np.asarray(v).tolist() if hasattr(v, "shape") else v


def _model_tests_one(item):
    name, p = item

    p = {k: _plain(v) for k, v in p.items()}
    if p["kind"] == "lin":
        ref, meta = regref.model_tests_linear(
            p["y"],
            p["X"],
            weights=p["weights"],
            intercept=p["intercept"],
            R=p["R"],
            q=p["q"],
            small=p["small"],
            terms=p["terms"],
            gq=tuple(p["gq"]),
        )
    else:
        import stardust as sd

        def start(cols):
            return sd.regression.glm(
                p["y"], [[r[c] for c in cols] for r in p["X"]], family=p["family"], link=p["link"]
            ).params.tolist()

        k = len(p["X"][0])
        acc, seq = [], []
        for t in p["terms"][:-1]:
            acc = acc + t
            seq.append(start(acc))
        starts = {"big": start(list(range(k))), "small": start(list(range(p["small"]))), "seq": seq}
        ref, meta = regref.model_tests_glm(
            p["y"],
            p["X"],
            family=p["family"],
            link=p["link"],
            R=p["R"],
            small=p["small"],
            terms=p["terms"],
            starts=starts,
        )
    vals = {key: ([list(x) for x in v] if isinstance(v, list) else list(v)) for key, v in ref.items()}
    return name, p | {"meta": meta, "ref": vals}


# ---------------------------------------------------------------- selection

SEL_RUNS = [(d, c) for d in ("both", "forward", "backward") for c in ("aic", "bic", "pvalue", "cv")]
SEL_BEST = ("aic", "bic", "cp", "adjr2", "cv")
SEL_TIE = 1e-9  # a step or size whose best and runner-up (or the move threshold) are this close is flagged


def selection_problems():
    import numpy as np

    import stardust as sd

    def labels(n, seed, cv=5):
        return np.asarray(sd.take(sd.asarray([i % cv for i in range(n)], dtype=sd.int64),
                                  sd.random.default_rng(seed).permutation(n)).tolist())  # fmt: skip

    out = {}
    rng = np.random.default_rng(1121)
    n = 80
    Z = rng.standard_normal((n, 6))
    X = np.round(Z + 0.4 * Z[:, [1, 2, 3, 4, 5, 0]], 6)
    y = np.round(1.0 + X @ [0.8, 0.0, -0.45, 0.0, 0.25, 0.1] + rng.standard_normal(n), 6)
    out["ols"] = dict(kind="lin", y=y, X=X, weights=None, intercept=True, labels=labels(n, 11), seed=11)
    w = np.round(rng.uniform(0.3, 3.0, n), 6)
    w[[3, 30, 61]] = 0.0
    out["wls_zero"] = dict(kind="lin", y=y, X=X, weights=w, intercept=True, labels=labels(n, 12), seed=12)
    Xn = np.round(rng.standard_normal((60, 5)) + 0.5, 6)
    yn = np.round(Xn @ [0.6, 0.0, 0.35, -0.3, 0.0] + rng.standard_normal(60), 6)
    out["noint"] = dict(kind="lin", y=yn, X=Xn, weights=None, intercept=False, labels=labels(60, 13), seed=13)
    # two nearly equal columns: kappa of the design near 1e6
    x1 = np.round(rng.standard_normal(70), 6)
    Xc = np.column_stack(
        [x1, x1 + np.round(1e-5 * rng.standard_normal(70), 11), np.round(rng.standard_normal((70, 3)), 6)]
    )
    yc = np.round(0.5 + Xc @ [0.7, 0.0, 0.0, 0.4, -0.2] + rng.standard_normal(70), 6)
    out["cond"] = dict(kind="lin", y=yc, X=Xc, weights=None, intercept=True, labels=labels(70, 14), seed=14)
    ng = 70
    Xg = np.round(rng.standard_normal((ng, 4)), 6)
    eta = 0.2 + Xg @ [0.9, 0.0, -0.5, 0.2]
    out["glm_logit"] = dict(kind="glm", y=(rng.uniform(size=ng) < 1 / (1 + np.exp(-eta))).astype(float), X=Xg,
                            family="binomial", link="logit", intercept=True, weights=None, labels=None)  # fmt: skip
    out["glm_poisson"] = dict(kind="glm", y=rng.poisson(np.exp(0.5 * eta)).astype(float), X=Xg, family="poisson",
                              link="log", intercept=True, weights=None, labels=None)  # fmt: skip
    out["glm_gamma"] = dict(kind="glm", y=np.round(rng.gamma(3.0, np.exp(0.5 * eta) / 3.0), 6), X=Xg, family="gamma",
                            link="log", intercept=True, weights=None, labels=None)  # fmt: skip
    return out


def _sel_val(dd):
    return dd[0] + dd[1]


def _sel_arb(dd):
    from flint import arb

    return arb(dd[0]) + arb(dd[1])


def _sel_pvalue(p, ref, big, small, glm):
    """The p-value of `small` within `big` (masks), in Arb: the partial F (big's scale) or the
    likelihood ratio."""
    from flint import arb

    dr = bin(big).count("1") - bin(small).count("1")
    if glm:
        lr = -2 * (_sel_arb(ref["llf"][small]) - _sel_arb(ref["llf"][big]))
        if lr < 0:
            lr = arb(0)
        return float(regref._chi2_upper(lr, arb(dr)).mid())
    k = bin(big).count("1") + (1 if p["intercept"] else 0)
    df = p["meta"]["nobs"] - k
    sb, ss = _sel_arb(ref["ssr"][big]), _sel_arb(ref["ssr"][small])
    f = ((ss - sb) / dr) / (sb / df)
    if f < 0:
        f = arb(0)
    return float(regref._f_upper(f, arb(dr), arb(df)).mid())


def _sel_simulate(p, ref, direction, crit, glm):
    """stepwise's path with the reference values (docs/design/regression.md §3.8, G167-G170):
    ([(action, column)], the smallest gap met, the masks of the path)."""
    k = len(p["X"][0])
    full = (1 << k) - 1
    intercept = p["intercept"]
    if direction == "forward":
        cur = 0 if intercept else 1
    else:
        cur = full
    if direction == "forward" and not intercept:
        cur = 1  # the tests start it from the first column (start=[0])
    vals = ref.get(crit)
    moves, path, visited = [], [cur], {cur}
    gap = float("inf")
    while True:
        cands = []
        for j in range(k):
            if cur >> j & 1:
                if direction in ("both", "backward") and (intercept or bin(cur).count("1") > 1):
                    cands.append((j, True))
            elif direction in ("both", "forward"):
                cands.append((j, False))
        nxt = [cur & ~(1 << j) if d else cur | (1 << j) for j, d in cands]
        pick = None
        if crit == "pvalue":
            pd = [(_sel_pvalue(p, ref, cur, m, glm), j, d, m) for (j, d), m in zip(cands, nxt) if d]
            over = [x for x in pd if x[0] > 0.10]
            for x in pd:
                gap = min(gap, abs(x[0] - 0.10) / 0.10)
            if over:
                pick = max(over, key=lambda x: (x[0], -x[1]))
            else:
                pa = [(_sel_pvalue(p, ref, m, cur, glm), j, d, m) for (j, d), m in zip(cands, nxt) if not d]
                for x in pa:
                    gap = min(gap, abs(x[0] - 0.05) / 0.05)
                under = [x for x in pa if x[0] <= 0.05]
                if under:
                    pick = min(under, key=lambda x: (x[0], x[1]))
            if pick is not None:
                pick = pick[1:]
        else:
            cv = _sel_val(vals[cur])
            ds = [(_sel_val(vals[m]) - cv, j, d, m) for (j, d), m in zip(cands, nxt)]
            if ds:
                ds.sort(key=lambda x: (x[0], x[1]))
                scale = max(1.0, abs(cv))
                if len(ds) > 1:
                    gap = min(gap, (ds[1][0] - ds[0][0]) / scale)
                limit = -1e-7 if crit in ("aic", "bic") else 0.0
                gap = min(gap, abs(ds[0][0] - limit) / scale)
                if ds[0][0] < limit:
                    pick = ds[0][1:]
        if pick is None:
            break
        j, d, m = pick
        if m in visited:
            break
        moves.append(("drop" if d else "add", j))
        cur = m
        visited.add(m)
        path.append(m)
    return moves, gap, path


def _sel_best(p, ref, crit, glm):
    """best_subset's choice with the reference values: (per size the best mask or None, the size
    picked, the smallest gap, per size the criterion as an arb or None)."""
    from flint import arb

    k = len(p["X"][0])
    intercept = p["intercept"]
    nobs = p["meta"]["nobs"]
    icpt = 1 if intercept else 0
    gap = float("inf")
    sizes = list(range(k + 1))

    def masks_of(s):
        return [m for m in range(2**k) if bin(m).count("1") == s and (intercept or m)]

    def ranked(key, masks, sign=1.0):
        nonlocal gap
        vs = sorted((sign * _sel_val(key(m)), m) for m in masks)
        if len(vs) > 1:
            gap = min(gap, (vs[1][0] - vs[0][0]) / max(1.0, abs(vs[0][0])))
        return vs[0][1]

    if glm:
        best = [ranked(lambda m: ref["llf"][m], masks_of(s), -1.0) if masks_of(s) else None for s in sizes]
    else:
        best = [ranked(lambda m: ref["ssr"][m], masks_of(s)) if masks_of(s) else None for s in sizes]
    vals = []
    for s, m in zip(sizes, best):
        if m is None:
            vals.append(None)
        elif crit in ("aic", "bic"):
            vals.append(_sel_arb(ref[crit][m]))
        elif crit == "cp":
            full = (1 << k) - 1
            s2 = _sel_arb(ref["ssr"][full]) / (nobs - k - icpt)
            vals.append(_sel_arb(ref["ssr"][m]) / s2 - nobs + 2 * (s + icpt))
        elif crit == "adjr2":
            if s == 0 and intercept:
                vals.append(arb(0))
            else:
                tss = _sel_arb(ref["ssr"][0]) if intercept else arb(p["meta"]["utss"])
                vals.append(1 - arb(nobs - icpt) / (nobs - s - icpt) * (_sel_arb(ref["ssr"][m]) / tss))
        else:  # cv: per fold the size's best by the training rows, its test error; the folds' mean
            c = p["meta"]["folds"]
            errs = []
            for f in range(c):
                mf = ranked(lambda q: ref["fold_train_ssr"][q * c + f], masks_of(s))
                errs.append(_sel_arb(ref["fold_mse"][mf * c + f]))
            vals.append(sum(errs, arb(0)) / c)
    fin = [(float(v.mid()), s) for s, v in zip(sizes, vals) if v is not None]
    order = sorted(fin, key=lambda x: ((-x[0] if crit == "adjr2" else x[0]), x[1]))
    if len(order) > 1:
        gap = min(gap, abs(order[1][0] - order[0][0]) / max(1.0, abs(order[0][0])))
    return best, order[0][1], gap, vals


def _selection_one(item):
    name, p = item
    import numpy as np

    p = {k: _plain(v) for k, v in p.items()}
    glm = p["kind"] == "glm"
    k = len(p["X"][0])
    if glm:
        import stardust as sd

        starts = []
        for mask in range(2**k):
            cols = [j for j in range(k) if mask >> j & 1]
            Xs = np.asarray(p["X"])[:, cols]
            starts.append(sd.regression.glm(p["y"], Xs, family=p["family"], link=p["link"]).params.tolist())
        ref, meta = regref.selection_glm(p["y"], p["X"], family=p["family"], link=p["link"], starts=starts)
    else:
        ref, meta = regref.selection_linear(p["y"], p["X"], weights=p["weights"], intercept=p["intercept"],
                                            labels=p["labels"])  # fmt: skip
        if not p["intercept"]:
            meta["utss"] = float(sum(
                (wi if p["weights"] else 1.0) * yi * yi
                for yi, wi in zip(p["y"], p["weights"] or [1.0] * len(p["y"])) if (wi if p["weights"] else 1.0) > 0
            ))  # fmt: skip
    p["meta"] = meta
    with arbref.precision(256):  # the decisions' p-values and the derived values from the [hi, lo] pairs
        runs, best, derived = _sel_derive(p, ref, glm)
    for key in ("fold_train_ssr", "fold_mse"):
        ref.pop(key, None)
    return name, p | {"subsets": ref, "runs": runs, "best": best, "ref": derived}


def _sel_derive(p, ref, glm):
    runs, best = {}, {}
    derived = {}
    for d, c in SEL_RUNS:
        if glm and c == "cv":
            continue
        moves, gap, path = _sel_simulate(p, ref, d, c, glm)
        runs[f"{d}/{c}"] = {"moves": moves, "near_tie": gap < SEL_TIE, "gap": gap}
        if c != "pvalue" and gap >= SEL_TIE:
            derived[f"step_{d}_{c}"] = [ref[c][m] for m in path]
    for c in SEL_BEST:
        if glm and c in ("cp", "adjr2", "cv"):
            continue
        sets, pick, gap, vals = _sel_best(p, ref, c, glm)
        best[c] = {"sets": sets, "pick": pick, "near_tie": gap < SEL_TIE, "gap": gap}
        if gap >= SEL_TIE:
            derived[f"best_{c}"] = [list(arbref.to_dd(v)) for v in vals if v is not None]
    return runs, best, derived


KINDS = {
    "linear": (linear_problems, _linear_one),
    "glm": (glm_problems, _glm_one),
    "penalized": (penalized_problems, _penalized_one),
    "robust": (robust_problems, _robust_one),
    "model_tests": (model_tests_problems, _model_tests_one),
    "selection": (selection_problems, _selection_one),
}


def generate(kind, jobs):
    problems, one = KINDS[kind]
    items = list(problems().items())
    if kind in (
        "linear",
        "model_tests",
        "selection",
    ):  # the slow kinds (NIST's fits at high precision, n refits): in processes
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(jobs, len(items)), maxtasksperchild=4) as pool:
            data = dict(pool.map(one, items, chunksize=1))
    else:  # each problem takes well under a second
        data = dict(one(item) for item in items)
    prov = {**arbref.provenance(), "tool": "tools/regression_ref.py", "n_requested": len(items)}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{kind}.json"
    with open(path, "w") as fh:
        json.dump({"_provenance": prov, **data}, fh)
    print(f"{kind}: {len(data)} problems -> {path.relative_to(ROOT)} ({path.stat().st_size} bytes)", flush=True)


def main(argv=None):
    import argparse

    from bench import cli

    ap = argparse.ArgumentParser(prog="tools/regression_ref.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="KIND", help=", ".join(KINDS))
    cli.add_jobs(ap, default=6)
    a = ap.parse_args(argv)
    bad = [n for n in a.names if n not in KINDS]
    if bad:
        ap.error(f"unknown kind {bad[0]!r}")
    with arbref.logged("regression_ref"):
        for k in a.names or KINDS:
            generate(k, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
