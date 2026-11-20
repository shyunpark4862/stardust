"""Arb references for the penalized regressions' tests (tools/regression_ref.py from tools/regref.py's
penalized;
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

    .venv/bin/python tools/regression_ref.py penalized
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "penalized.json"


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


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data
