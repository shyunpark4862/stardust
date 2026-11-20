"""Arb references for the GLM tests (tools/regression_ref.py from tools/regref.py's glm;
docs/design/regression.md §3.4, §6).

tests/regression/ref/glm.json holds, per problem, the inputs as doubles (y, X rows without the
constant column, family, link, var_weights, freq_weights, trials, offset (with log exposure
already added, as glm() adds it), intercept, scale, cluster labels) and regref.glm's values as
[hi, lo] pairs: every family and link (docs/design/regression.md §3.4) on random data, binomial proportions with trials,
poisson with an offset and an exposure, gamma with both kinds of weights (one zero), a fit
without the intercept and one with scale='dev'. Regenerate with

    .venv/bin/python tools/regression_ref.py glm
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "glm.json"

LINKS = [
    ("gaussian", "identity"),
    ("gaussian", "log"),
    ("binomial", "logit"),
    ("binomial", "probit"),
    ("binomial", "cloglog"),
    ("binomial", "log"),
    ("poisson", "log"),
    ("poisson", "identity"),
    ("poisson", "sqrt"),
    ("gamma", "inverse"),
    ("gamma", "log"),
    ("gamma", "identity"),
]


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
        mean = {
            "log": np.exp(0.4 + 0.5 * x1 - 0.3 * x2),
            "identity": 4.0 + 0.8 * x1 - 0.5 * x2,
            "sqrt": (2.0 + 0.4 * x1 - 0.3 * x2) ** 2,
        }[link]
        return rng.poisson(mean).astype(float)
    mean = {
        "inverse": 1 / (1.0 + 0.1 * x1**2 + 0.05 * x2**2),
        "log": np.exp(0.5 + 0.4 * x1 - 0.3 * x2),
        "identity": 4.0 + 0.6 * x1 - 0.4 * x2,
    }[link]
    return rng.gamma(3.0, mean / 3.0)


def fit(p, **kw):
    """sd.regression.glm on a problem's inputs (exposure as given, when there is one)."""
    import stardust as sd

    extra = {}
    if "_exposure" in p:
        extra = dict(offset=p["_offset"], exposure=p["_exposure"])
    elif p["offset"] is not None:
        extra = dict(offset=p["offset"])
    return sd.regression.glm(
        p["y"],
        p["X"],
        family=p["family"],
        link=p["link"],
        var_weights=p["var_weights"],
        freq_weights=p["freq_weights"],
        trials=p["trials"],
        intercept=p["intercept"],
        scale=p["scale"],
        **extra,
        **kw,
    )


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data
