"""Arb references for rlm's tests (tools/regref.py's rlm; docs/design/regression.md §3.6).

tests/regression/ref/robust.json holds, per problem, the inputs as doubles (y, X rows, norm,
tuning constants, scale rule, intercept; for MM the rng seed and the S estimate's params and
scale that sd.regression found with it) and regref.rlm's certified solution (params, scale,
weights, bse) as [hi, lo] pairs: the three norms with the MAD and Huber's proposal 2 scales
on data with outliers (an even and an odd n, no intercept, a larger fit), Brownlee's stack
loss, and the MM estimator's M step on contaminated data. The pieces of psi (and the median's
rows) given to the reference are those of sd.regression's solution; the reference certifies
them on a box around the root, so a solution on a wrong piece would fail to generate.
Regenerate with

    uv run --no-sync python -m tests.regression.refdata_robust
"""

import json
import os
import pathlib
import sys

REF = pathlib.Path(__file__).parent / "ref" / "robust.json"

# Brownlee's stack loss (statsmodels' datasets.stackloss): air flow, water temperature, acid
# concentration; stack loss
STACKLOSS_X = [[80, 27, 89], [80, 27, 88], [75, 25, 90], [62, 24, 87], [62, 22, 87], [62, 23, 87], [62, 24, 93],
               [62, 24, 93], [58, 23, 87], [58, 18, 80], [58, 18, 89], [58, 17, 88], [58, 18, 82], [58, 19, 93],
               [50, 18, 89], [50, 18, 86], [50, 19, 72], [50, 19, 79], [50, 20, 80], [56, 20, 82], [70, 20, 91]]
STACKLOSS_Y = [42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15]

C_REF = {"huber": (1.345,), "bisquare": (4.685,), "hampel": (2.0, 4.0, 8.0)}


def problems():
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
        out[f"stackloss_{norm}"] = dict(method="M", norm=norm, scale_est="mad", intercept=True,
                                        y=[float(v) for v in STACKLOSS_Y], X=[[float(v) for v in r] for r in STACKLOSS_X])
    # MM: 20% leverage outliers
    n, p = 120, 2
    X = rng.standard_normal((n, p))
    y = X @ np.array([1.0, 2.0]) + 0.5 + 0.5 * rng.standard_normal(n)
    m = int(0.2 * n)
    X[:m] = rng.normal(5.0, 0.5, (m, p))
    y[:m] = rng.normal(-10.0, 1.0, m)
    out["mm_leverage"] = dict(method="MM", norm="bisquare", intercept=True, seed=7, y=y.tolist(), X=X.tolist())
    out["mm_stackloss"] = dict(method="MM", norm="bisquare", intercept=True, seed=11,
                               y=[float(v) for v in STACKLOSS_Y], X=[[float(v) for v in r] for r in STACKLOSS_X])
    return out


def fit(p):
    """sd.regression's fit of a problem."""
    import stardust as sd

    if p["method"] == "MM":
        return sd.regression.rlm(p["y"], p["X"], method="MM", intercept=p["intercept"],
                                 rng=sd.random.default_rng(p["seed"]))
    return sd.regression.rlm(p["y"], p["X"], norm=p["norm"], scale_est=p["scale_est"], intercept=p["intercept"])


def _one(item):
    name, p = item
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
    import regref

    r = fit(p)
    beta0 = [float(v) for v in r.params.tolist()]
    if p["method"] == "MM":
        init = [float(v) for v in r._raw[8].tolist()]
        p = dict(p, init_params=init, s_scale=float(r.scale))
        ref = regref.rlm(p["y"], p["X"], beta0=beta0, norm="bisquare", c=(4.685061,), intercept=p["intercept"],
                         scale=r.scale, init_params=init)
    else:
        ref = regref.rlm(p["y"], p["X"], beta0=beta0, norm=p["norm"], c=C_REF[p["norm"]], scale_est=p["scale_est"],
                         intercept=p["intercept"])
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
