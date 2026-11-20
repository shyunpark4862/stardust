"""Arb references for the tests on fitted models (tools/regression_ref.py from tools/regref.py's
model_tests_linear and model_tests_glm; docs/design/regression.md §3.8, §6).

tests/regression/ref/model_tests.json holds, per problem, the inputs as doubles and the
references as [hi, lo] pairs: linear models (OLS, WLS with zero weights, no intercept, a
factor's dummies as one term, an ill-conditioned design with kappa near 1e8) and GLMs (binomial
logit and probit, poisson log, gamma log). The GLMs' Newton starts are sd's fits (the reference
certifies the root). Regenerate with

    .venv/bin/python tools/regression_ref.py model_tests
"""

import json
import pathlib

REF = pathlib.Path(__file__).parent / "ref" / "model_tests.json"


def load():
    """{problem: {...}} (the file's "_provenance" left out)."""
    with open(REF) as fh:
        data = json.load(fh)
    data.pop("_provenance", None)
    return data


# the problems whose White auxiliary design sd reduces further than the reference (above)
WHITE_OTHER = ("cond",)


def _array(x):
    import numpy as np

    return np.asarray(x.tolist() if hasattr(x, "tolist") else x, dtype=float)


def fit(p, cols=None):
    """sd's fit of a problem (ols, wls or glm), on the first `cols` columns of X when given."""
    import numpy as np

    import stardust as sd

    X = np.asarray(p["X"], dtype=float)
    if cols is not None:
        X = X[:, :cols]
    if p["kind"] == "glm":
        return sd.regression.glm(p["y"], X, family=p["family"], link=p["link"])
    if p["weights"] is None:
        return sd.regression.ols(p["y"], X, intercept=p["intercept"])
    return sd.regression.wls(p["y"], X, weights=p["weights"], intercept=p["intercept"])


def values(name, p) -> dict:
    """sd's value of each reference field of a problem, {field: array}, from the calls the
    reference defines (the problem `name`'s R, q, nested columns, terms and Goldfeld-Quandt split). A
    field sd computes as another quantity is None: White's tests on 'cond', whose auxiliary design
    loses a column to rounding (one degree of freedom fewer than the reference's)."""
    import stardust as sd

    R = sd.regression
    r, small = fit(p), fit(p, cols=p["small"])
    out = {}
    if p["kind"] == "glm":
        t = r.t_test(p["R"])
        w = r.wald_test(p["R"])
        c = r.compare_lr_test(small)
        t1 = R.anova(r, terms=p["terms"])
        t3 = R.anova(r, typ=3, terms=p["terms"])
        fixed = p["family"] in ("binomial", "poisson")
        out.update(t_tvalue=t.tvalue, t_pvalue=t.pvalue, wald_stat=w.statistic, wald_p=w.pvalue,
                   compare_lr=c.statistic, compare_lr_p=c.pvalue, anova1_dev=_array(t1.sum_sq)[:-1],
                   anova1_p=_array(t1.pvalue)[:-1],
                   anova3_wald=_array(t3.statistic)[:-1] * (1.0 if fixed else _array(t3.df)[:-1]),
                   hat=r.leverage, cooks=r.cooks_distance, dffits=r.dffits, dfbetas=r.dfbetas)  # fmt: skip
        if not fixed:
            c = r.compare_f_test(small)
            out.update(compare_F=c.fvalue, compare_F_p=c.pvalue)
        return {k: _array(v) for k, v in out.items()}
    t = r.t_test(p["R"], p["q"])
    w = r.wald_test(p["R"], p["q"])
    cf, cl = r.compare_f_test(small), r.compare_lr_test(small)
    t1 = R.anova(r, terms=p["terms"])
    t3 = R.anova(r, typ=3, terms=p["terms"])
    vif = _array(r.vif)
    out.update(t_effect=t.effect, t_sd=t.sd, t_tvalue=t.tvalue, t_pvalue=t.pvalue, wald_F=w.statistic,
               wald_p=w.pvalue, compare_F=cf.fvalue, compare_F_p=cf.pvalue, compare_lr=cl.statistic,
               compare_lr_p=cl.pvalue, anova1_ss=_array(t1.sum_sq)[:-1], anova1_F=_array(t1.statistic)[:-1],
               anova1_p=_array(t1.pvalue)[:-1], anova3_ss=_array(t3.sum_sq)[:-1],
               anova3_F=_array(t3.statistic)[:-1], anova3_p=_array(t3.pvalue)[:-1],
               vif=vif[1:] if p["intercept"] else vif, dffits=r.dffits, dfbetas=r.dfbetas)  # fmt: skip
    for tag, cross in (("white", True), ("white_nocross", False)):
        wh = r.white_test(cross_terms=cross)
        vals = (wh.lm, wh.lm_pvalue, wh.fvalue, wh.f_pvalue)
        for suffix, v in zip(("_lm", "_lm_p", "_F", "_F_p"), vals):
            out[tag + suffix] = None if name in WHITE_OTHER else v
    n1, drop, by = p["gq"]
    m = sum(1 for v in (p["weights"] or [1] * len(p["y"])) if v > 0)
    g = r.goldfeld_quandt(split=n1 if n1 is not None else m // 2, drop=drop, order_by=by)
    rs = r.reset_test()
    out.update(gq_F=g.fvalue, gq_p=g.pvalue, reset_F=rs.fvalue, reset_p=rs.pvalue)
    return {k: (None if v is None else _array(v)) for k, v in out.items()}
