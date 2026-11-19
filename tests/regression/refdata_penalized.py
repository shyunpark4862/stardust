"""Arb references for the penalized regressions' tests (tools/regref.py's penalized;
docs/design/regression.md §3.5).

tests/regression/ref/penalized.json holds, per problem, the inputs as doubles (y, X rows,
weights, alpha, l1_ratio, family, intercept, standardize) and regref.penalized's certified
solution (params and ssr or deviance) as [hi, lo] pairs: the lasso, the elastic net and ridge
with and without weights (zeros among them), standardize, no intercept, a constant column, an
ill-conditioned design (condition number 1e6 after centering; in one of its fits the polish
refines), and the penalized binomial and poisson GLMs at l1_ratio 1, 0.5 and 0. The active set
and signs given to the reference are those of sd.regression's solution; the reference
certifies them (the strict KKT inequalities of the inactive columns), so a wrong active set
would fail to generate. Regenerate with

    uv run --no-sync python -m tests.regression.refdata_penalized
"""

import json
import os
import pathlib
import sys

REF = pathlib.Path(__file__).parent / "ref" / "penalized.json"


def problems():
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


def fit(p):
    """sd.regression's fit of a problem."""
    import stardust as sd

    R = sd.regression
    kw = dict(alpha=p["alpha"], weights=p["weights"], intercept=p["intercept"], standardize=p["standardize"])
    it = {"max_iter": p["max_iter"]} if "max_iter" in p else {}
    if p["kind"] == "lasso":
        return R.lasso(p["y"], p["X"], **kw, **it)
    if p["kind"] == "ridge":
        return R.ridge(p["y"], p["X"], **kw)
    if p["kind"] == "elastic_net":
        return R.elastic_net(p["y"], p["X"], l1_ratio=p["l1_ratio"], **kw, **it)
    return R.penalized_glm(p["y"], p["X"], family=p["family"], l1_ratio=p["l1_ratio"], **kw)


def _one(item):
    name, p = item
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import regref

    r = fit(p)
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


def generate():
    data = dict(_one(item) for item in problems().items())
    REF.parent.mkdir(exist_ok=True)
    with open(REF, "w") as fh:
        json.dump(data, fh)


def load():
    with open(REF) as fh:
        return json.load(fh)


if __name__ == "__main__":
    generate()
    print(REF, os.path.getsize(REF), "bytes")
