"""Arb references for rlm's tests (tools/regression_ref.py from tools/regref.py's rlm;
docs/design/regression.md §3.6).

tests/regression/ref/robust.json holds, per problem, the inputs as doubles (y, X rows, norm,
tuning constants, scale rule, intercept; for MM the rng seed and the S estimate's params and
scale: regref.s_estimate's certified fixed point near the one sd.regression found with it,
rounded to doubles) and regref.rlm's certified solution (params, scale,
weights, bse) as [hi, lo] pairs: the three norms with the MAD and Huber's proposal 2 scales
on data with outliers (an even and an odd n, no intercept, a larger fit), Brownlee's stack
loss, and the MM estimator's M step on contaminated data. The pieces of psi (and the median's
rows) given to the reference are those of sd.regression's solution; the reference certifies
them on a box around the root, so a solution on a wrong piece would fail to generate.
Regenerate with

    .venv/bin/python tools/regression_ref.py robust
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "robust.json"

# Brownlee's stack loss (statsmodels' datasets.stackloss): air flow, water temperature, acid
# concentration; stack loss
STACKLOSS_X = [
    [80, 27, 89],
    [80, 27, 88],
    [75, 25, 90],
    [62, 24, 87],
    [62, 22, 87],
    [62, 23, 87],
    [62, 24, 93],
    [62, 24, 93],
    [58, 23, 87],
    [58, 18, 80],
    [58, 18, 89],
    [58, 17, 88],
    [58, 18, 82],
    [58, 19, 93],
    [50, 18, 89],
    [50, 18, 86],
    [50, 19, 72],
    [50, 19, 79],
    [50, 20, 80],
    [56, 20, 82],
    [70, 20, 91],
]
STACKLOSS_Y = [42, 37, 37, 28, 18, 18, 19, 20, 15, 14, 14, 13, 11, 12, 8, 7, 8, 8, 9, 15, 15]

C_REF = {"huber": (1.345,), "bisquare": (4.685,), "hampel": (2.0, 4.0, 8.0)}


def fit(p):
    """sd.regression's fit of a problem."""
    import stardust as sd

    if p["method"] == "MM":
        return sd.regression.rlm(
            p["y"], p["X"], method="MM", intercept=p["intercept"], rng=sd.random.default_rng(p["seed"])
        )
    return sd.regression.rlm(p["y"], p["X"], norm=p["norm"], scale_est=p["scale_est"], intercept=p["intercept"])


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data
