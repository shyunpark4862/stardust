"""Arb references for variable selection (tools/regression_ref.py from tools/regref.py's
selection_linear and selection_glm; docs/design/regression.md §3.8, §6).

tests/regression/ref/selection.json holds, per problem, the inputs as doubles (with the fold
labels of sd.random.default_rng(seed).permutation, lasso_cv's folds), every subset's reference
values as [hi, lo] pairs ("subsets": residual sum of squares, llf, AIC, BIC, the cross-validation
error), the paths stepwise takes with those values ("runs": per direction and criterion the moves
as (action, column), and whether a step met a near-tie), best_subset's choice ("best": per
criterion each size's best subset as a bit mask and the size picked) and the derived references
of the values sd reports ("ref": the criterion along each path and of each size's best). Problems:
OLS with correlated columns, WLS with zero weights, no intercept, two nearly equal columns
(kappa near 1e6), and binomial (logit), poisson (log) and gamma (log) GLMs. Regenerate with

    .venv/bin/python tools/regression_ref.py selection
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "selection.json"
CV = 5


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data


def options(p):
    """stepwise's and best_subset's model options of a problem."""
    if p["kind"] == "glm":
        return dict(model="glm", family=p["family"], link=p["link"])
    kw = dict(intercept=p["intercept"])
    if p["weights"] is not None:
        kw.update(model="wls", weights=p["weights"])
    return kw


def rng(p):
    import stardust as sd

    return sd.random.default_rng(p["seed"])


def stepwise(p, direction, criterion):
    """sd's stepwise on a problem as the reference ran it (without the intercept, forward starts
    from the first column)."""
    import stardust as sd

    kw = options(p) | dict(direction=direction, criterion=criterion)
    if criterion == "cv":
        kw.update(cv=CV, rng=rng(p))
    if direction == "forward" and not p["intercept"]:
        kw["start"] = [0]
    return sd.regression.stepwise(p["y"], p["X"], **kw)


def best_subset(p, criterion):
    import stardust as sd

    kw = options(p) | dict(criterion=criterion)
    if criterion == "cv":
        kw.update(cv=CV, rng=rng(p))
    return sd.regression.best_subset(p["y"], p["X"], **kw)


def mask(cols):
    return sum(1 << c for c in cols)


def values(name, p) -> dict:
    """sd's value of each reference field of a problem, {field: list}: the criterion along each
    path (stepwise's steps) and of each size's best subset (best_subset's values where a size has
    one)."""
    import math

    out = {}
    for field in p["ref"]:
        kind, rest = field.split("_", 1)
        if kind == "step":
            direction, criterion = rest.split("_")
            out[field] = [s.criterion for s in stepwise(p, direction, criterion).steps]
        else:
            vals = best_subset(p, rest).values.tolist()
            out[field] = [v for v in vals if not math.isnan(v)]
    return out
