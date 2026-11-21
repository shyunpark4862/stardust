"""Variable selection against R (tests/regression/data/selection_r.json, made once by
tools/gen_selection_r.py with R 4.6.2, MASS 7.3.66 and leaps 3.2; R is not needed here).

stepwise takes R's step and MASS::stepAIC's path: the same moves in the same order, the same
final terms, and AIC values that differ from R's extractAIC by one constant per problem (R drops
the likelihood's constants: n log(RSS/n) + 2 edf for lm). best_subset takes leaps::regsubsets'
best subset of each size and summary's rss, cp and adjr2. The partial F and likelihood ratio
p-values of the 'pvalue' criterion are R's add1 and drop1 (test = "F", "Chisq"). Where sd and R
differ by definition (docs/design/regression.md §3.8): the gamma GLM's AIC (sd's llf takes the
Pearson scale, R's the deviance / n; G174), which here still picks R's model.
"""

import json
import math
import pathlib

import numpy as np
import pytest

import stardust as sd

R = sd.regression
REF = json.loads((pathlib.Path(__file__).parent / "data" / "selection_r.json").read_text())


def _terms(p):
    """The problem's terms (the factor's two dummies as one term named 'f') and their names."""
    k = len(p["X"][0])
    if p.get("meta", [0, 0])[1] == 1:
        return [[j] for j in range(k - 2)] + [[k - 2, k - 1]], [f"x{j + 1}" for j in range(k - 2)] + ["f"]
    return [[j] for j in range(k)], [f"x{j + 1}" for j in range(k)]


def _kw(p):
    w = np.asarray(p["w"])
    icpt = p["meta"][0] == 1
    kw = dict(intercept=icpt, terms=_terms(p)[0])
    if p["meta"][2]:
        kw["include"] = list(range(int(p["meta"][2])))
    if not np.all(w == 1):
        kw.update(model="wls", weights=w)
    return kw


def _moves(r, names, terms):
    """sd's path as R's labels ('-x2', '+f')."""
    label = {" + ".join(f"x{c + 1}" for c in t): nm for t, nm in zip(terms, names)}
    return [("-" if s.action == "drop" else "+") + label[s.term] for s in r.steps[1:]]


def _final(r, names, terms):
    return [nm for t, nm in zip(terms, names) if all(c in r.selected for c in t)]


LM_RUNS = {
    # R name: stepwise options
    "both": dict(),
    "both_bic": dict(criterion="bic"),
    "backward": dict(direction="backward"),
    "stepaic": dict(),
    "forward": dict(direction="forward"),
    "forward_bic": dict(direction="forward", criterion="bic"),
    "both_empty": dict(direction="both", start=[]),
}


@pytest.mark.parametrize("name", sorted(REF["lm"]))
@pytest.mark.parametrize("run", sorted(LM_RUNS))
def test_lm_step_path(name, run):
    p = REF["lm"][name]
    ref = p["R"]
    kw = _kw(p) | LM_RUNS[run]
    terms, names = _terms(p)
    if not kw["intercept"] and run in ("forward", "forward_bic", "both_empty"):
        kw["start"] = [0]  # R's smallest model there is y ~ 0 + x1
    r = R.stepwise(p["y"], p["X"], **kw)
    assert _moves(r, names, terms) == ref[f"{run}_moves"]
    assert _final(r, names, terms) == ref[f"{run}_terms"]
    # the AIC (or BIC) path: a constant apart from R's extractAIC, n log(RSS / n) + k edf, whose
    # n counts the rows of weight 0 too (length(residuals)); sd's n is nobs (G171): the log parts
    # of the changes are in the ratio of the two n
    got = np.array([s.criterion for s in r.steps])
    dg = got - got[0]
    dr = np.asarray(ref[f"{run}_aic"]) - ref[f"{run}_aic"][0]
    w = np.asarray(p["w"])
    n_r, n = len(w), float(np.sum(w > 0))
    k = math.log(n) if "bic" in run else 2.0
    sizes = {" + ".join(f"x{c + 1}" for c in t): len(t) for t in terms}
    dedf = np.cumsum([0.0] + [sizes[s_.term] * (-1.0 if s_.action == "drop" else 1.0) for s_ in r.steps[1:]])
    assert np.max(np.abs((dg - k * dedf) / n - (dr - k * dedf) / n_r)) <= 1e-11 * np.max(np.abs(got))


@pytest.mark.parametrize("name", sorted(REF["lm"]))
def test_lm_pvalues(name):
    """The 'pvalue' criterion's tests are add1's and drop1's partial F."""
    from stardust.regression import _select as S

    p = REF["lm"][name]
    ref = p["R"]
    terms, names = _terms(p)
    kw = _kw(p)
    model = kw.get("model", "ols")
    opts = {"weights": kw["weights"]} if model == "wls" else {}
    cols, tl, inc, full, call = S._setup(
        "stepwise", p["y"], p["X"], model, opts, kw["intercept"], "propagate", terms, None
    )
    ev = S._evaluator("stepwise", model, full, kw["intercept"], None, None)
    j0 = 1 if kw["intercept"] else 0
    mid = {0, 1}
    cur, stats = ev.evaluate(
        S._design(kw["intercept"], tl, mid), [([j0 + c for c in tl[t]], False) for t in range(2, len(tl))]
    )
    got = [S._pvalue(ev, s, cur) for s in stats]
    assert np.allclose(got, ref["add1_p"], rtol=1e-11, atol=0)
    allt = set(range(len(tl)))
    cur, stats = ev.evaluate(
        S._design(kw["intercept"], tl, allt), [([j0 + c for c in tl[t]], True) for t in range(len(tl))]
    )
    got = [S._pvalue(ev, cur, s) for s in stats]
    assert np.allclose(got, ref["drop1_p"], rtol=1e-11, atol=0)


@pytest.mark.parametrize("name", sorted(n for n in REF["lm"] if "leaps_rss" in REF["lm"][n]["R"]))
def test_lm_best_subset_leaps(name):
    p = REF["lm"][name]
    ref = p["R"]
    kw = _kw(p)
    kw.pop("terms")
    k = len(p["X"][0])
    nforce = int(p["meta"][2])
    which = np.asarray(ref["leaps_which"]).reshape(-1, k)  # leaps' rows: one per size
    for crit, key in (("cp", "leaps_cp"), ("adjr2", "leaps_adjr2")):
        b = R.best_subset(p["y"], p["X"], criterion=crit, **kw)
        vals = b.values.tolist()
        for row in range(which.shape[0]):
            cols = [j for j in range(k) if which[row, j] and j >= nforce]
            assert b.subsets[len(cols)] == cols
            assert math.isclose(vals[len(cols)], ref[key][row], rel_tol=1e-10, abs_tol=1e-12)


@pytest.mark.parametrize("name", sorted(REF["glm"]))
@pytest.mark.parametrize("run", ["both", "both_bic", "forward"])
def test_glm_step_path(name, run):
    p = REF["glm"][name]
    ref = p["R"]
    kw = {"both": dict(), "both_bic": dict(criterion="bic"), "forward": dict(direction="forward")}[run]
    k = len(p["X"][0])
    terms, names = [[j] for j in range(k)], [f"x{j + 1}" for j in range(k)]
    r = R.stepwise(p["y"], p["X"], model="glm", family=p["family"], link=p["link"], **kw)
    assert _moves(r, names, terms) == ref[f"{run}_moves"]
    assert _final(r, names, terms) == ref[f"{run}_terms"]
    got = np.array([s.criterion for s in r.steps])
    diff = got - np.asarray(ref[f"{run}_aic"])
    if p["family"] == "gamma":
        # sd's llf takes the Pearson scale, R's the deviance / n: the AIC differences between
        # models are not R's (G174); here the same models still win
        assert np.max(np.abs(diff - diff[0])) > 1e-6
    else:
        assert np.max(np.abs(diff - diff[0])) <= 1e-8 * np.max(np.abs(got))


@pytest.mark.parametrize("name", sorted(REF["glm"]))
def test_glm_pvalues(name):
    """The GLM's 'pvalue' tests are likelihood ratios: R's add1 and drop1 (test = "Chisq") for the
    families with a fixed scale."""
    from stardust.regression import _select as S

    p = REF["glm"][name]
    if p["family"] == "gamma":
        pytest.skip("R's add1/drop1 Chisq of a gamma GLM is the scaled deviance, sd's the llf ratio (G170)")
    ref = p["R"]
    opts = {"family": p["family"], "link": p["link"]}
    cols, tl, inc, full, call = S._setup("stepwise", p["y"], p["X"], "glm", opts, True, "propagate", None, None)
    ev = S._evaluator("stepwise", "glm", full, True, None, None)
    cur, stats = ev.evaluate(S._design(True, tl, {0, 1}), [([1 + t], False) for t in range(2, len(tl))])
    assert np.allclose([S._pvalue(ev, s, cur) for s in stats], ref["add1_p"], rtol=1e-6, atol=0)
    cur, stats = ev.evaluate(S._design(True, tl, set(range(len(tl)))), [([1 + t], True) for t in range(len(tl))])
    assert np.allclose([S._pvalue(ev, cur, s) for s in stats], ref["drop1_p"], rtol=1e-6, atol=0)
