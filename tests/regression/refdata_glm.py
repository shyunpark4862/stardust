"""Arb references for the GLM tests (tools/regref.py's glm; docs/design/regression.md §3.4, §6).

tests/regression/ref/glm.json holds, per problem, the inputs as doubles (y, X rows without the
constant column, family, link, var_weights, freq_weights, trials, offset (with log exposure
already added, as glm() adds it), intercept, scale, cluster labels) and regref.glm's values as
[hi, lo] pairs: every family and link of G31 on random data, binomial proportions with trials,
poisson with an offset and an exposure, gamma with both kinds of weights (one zero), a fit
without the intercept and one with scale='dev'. Regenerate with

    uv run --no-sync python -m tests.regression.refdata_glm
"""

import json
import math
import os
import pathlib
import sys

REF = pathlib.Path(__file__).parent / "ref" / "glm.json"

LINKS = [("gaussian", "identity"), ("gaussian", "log"), ("binomial", "logit"), ("binomial", "probit"),
         ("binomial", "cloglog"), ("binomial", "log"), ("poisson", "log"), ("poisson", "identity"),
         ("poisson", "sqrt"), ("gamma", "inverse"), ("gamma", "log"), ("gamma", "identity")]


def _response(rng, family, link, X):
    import numpy as np

    n = X.shape[0]
    x1, x2 = X[:, 0], X[:, 1]
    if family == "gaussian":
        mean = 2.0 + 0.5 * x1 - 0.3 * x2 if link == "identity" else np.exp(0.5 + 0.3 * x1 - 0.2 * x2)
        return mean + 0.4 * rng.standard_normal(n)
    if family == "binomial":
        if link == "log":
            p = 0.3 * np.exp(0.2 * x1 - 0.1 * x2)
        else:
            p = 1 / (1 + np.exp(-(0.3 + 0.8 * x1 - 0.5 * x2)))
        return (rng.uniform(size=n) < p).astype(float)
    if family == "poisson":
        mean = {"log": np.exp(0.4 + 0.5 * x1 - 0.3 * x2), "identity": 4.0 + 0.8 * x1 - 0.5 * x2,
                "sqrt": (2.0 + 0.4 * x1 - 0.3 * x2) ** 2}[link]
        return rng.poisson(mean).astype(float)
    mean = {"inverse": 1 / (1.0 + 0.1 * x1 ** 2 + 0.05 * x2 ** 2), "log": np.exp(0.5 + 0.4 * x1 - 0.3 * x2),
            "identity": 4.0 + 0.6 * x1 - 0.4 * x2}[link]
    return rng.gamma(3.0, mean / 3.0)


def problems():
    import numpy as np

    import stardust as sd

    out = {}
    rng = np.random.default_rng(11)
    n = 60
    g = [int(v) for v in np.random.default_rng(5).integers(0, 6, n)]
    base = dict(var_weights=None, freq_weights=None, trials=None, offset=None, intercept=True, scale=None, groups=g)
    for family, link in LINKS:
        X = rng.standard_normal((n, 2))
        y = _response(rng, family, link, X)
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
    out["poisson_exposure"] = dict(base, y=y.tolist(), X=X.tolist(), family="poisson", link="log",
                                   offset=[float(v) for v in tot.tolist()], _offset=off.tolist(), _exposure=expo.tolist())
    # gamma with var_weights (one zero) and freq_weights
    X = rng.standard_normal((n, 2))
    y = rng.gamma(3.0, np.exp(0.5 + 0.4 * X[:, 0] - 0.3 * X[:, 1]) / 3.0)
    vw = rng.uniform(0.5, 2.0, n)
    vw[7] = 0.0
    fw = rng.integers(1, 4, n).astype(float)
    out["gamma_weights"] = dict(base, y=y.tolist(), X=X.tolist(), family="gamma", link="log", var_weights=vw.tolist(),
                                freq_weights=fw.tolist())
    # no intercept; scale from the deviance
    X = rng.standard_normal((n, 2))
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.9 * X[:, 0] - 0.6 * X[:, 1])))).astype(float)
    out["logit_noint"] = dict(base, y=y.tolist(), X=X.tolist(), family="binomial", link="logit", intercept=False)
    X = rng.standard_normal((n, 2))
    y = rng.gamma(3.0, np.exp(0.5 + 0.4 * X[:, 0] - 0.3 * X[:, 1]) / 3.0)
    out["gamma_dev"] = dict(base, y=y.tolist(), X=X.tolist(), family="gamma", link="log", scale="dev")
    return out


def fit(p, **kw):
    """sd.regression.glm on a problem's inputs (exposure as given, when there is one)."""
    import stardust as sd

    extra = {}
    if "_exposure" in p:
        extra = dict(offset=p["_offset"], exposure=p["_exposure"])
    elif p["offset"] is not None:
        extra = dict(offset=p["offset"])
    return sd.regression.glm(p["y"], p["X"], family=p["family"], link=p["link"], var_weights=p["var_weights"],
                             freq_weights=p["freq_weights"], trials=p["trials"], intercept=p["intercept"],
                             scale=p["scale"], **extra, **kw)


def _one(item):
    name, p = item
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import regref

    r = fit(p)
    beta0 = [float(v) for v in r.params.tolist()]
    nb0 = 0.0
    if p["offset"] is not None and p["link"] == "log":  # the intercept-only fit's start
        nb0 = math.log(sum(p["y"]) / sum(math.exp(o) for o in p["offset"]))
    ref, meta = regref.glm(p["y"], p["X"], family=p["family"], link=p["link"], var_weights=p["var_weights"],
                           freq_weights=p["freq_weights"], trials=p["trials"], offset=p["offset"],
                           intercept=p["intercept"], scale=p["scale"], beta0=beta0, groups=p["groups"], null_beta0=nb0)
    vals = {key: ([list(v) for v in x] if isinstance(x, list) else list(x)) for key, x in ref.items()}
    return name, dict(p, meta=meta, ref=vals)


def generate():
    # one process: each problem takes well under a second
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
