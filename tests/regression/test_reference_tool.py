"""G0: the Arb reference tool for least squares (tools/regref.py) against the certified values
of the NIST StRD problems and against statsmodels (docs/design/regression.md §6)."""

import math
import os
import sys

import pytest

from . import nist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
regref = pytest.importorskip("regref")


def rel(hl, certified):
    v, c = hl[0] + hl[1], float(certified)
    return abs(v - c) / abs(c) if c != 0 else abs(v)


@pytest.mark.parametrize("name", nist.NAMES)
def test_certified_values(name):
    # NIST certifies 15 significant digits: the references must agree to that
    d = nist.load(name)
    y, X = nist.exact_design(d)
    ref, meta = regref.ols(y, X, intercept=d["intercept"])
    assert meta["exact_fit"] == (name in ("Wampler1", "Wampler2"))
    assert len(ref["params"]) == len(d["params"])
    for got, want in zip(ref["params"], d["params"]):
        assert rel(got, want) <= 1e-14
    for got, want in zip(ref["bse"], d["bse"]):
        assert rel(got, want) <= 1e-14
    assert rel(ref["rsquared"], d["rsquared"]) <= 1e-14
    rsd = math.sqrt(ref["scale"][0] + ref["scale"][1])
    assert (abs(rsd - float(d["rsd"])) <= 1e-14 * float(d["rsd"])) or float(d["rsd"]) == 0.0


def test_against_statsmodels():
    np = pytest.importorskip("numpy")
    sm = pytest.importorskip("statsmodels.api")
    st = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + rng.standard_normal(40)
    ref, meta = regref.ols(y.tolist(), X.tolist(), t_guess=float(st.t.ppf(0.975, 36)))
    r = sm.OLS(y, sm.add_constant(X)).fit()
    assert meta == {"nobs": 40, "df_model": 3, "df_resid": 36, "exact_fit": False, "maxlags": 3}
    close = lambda a, hl, tol=1e-13: abs(a - (hl[0] + hl[1])) <= tol * abs(hl[0])
    for name, theirs in (("params", r.params), ("bse", r.bse), ("pvalues", r.pvalues), ("bse_HC3", r.HC3_se)):
        assert all(close(a, b) for a, b in zip(theirs, ref[name])), name
    for name in ("rsquared", "rsquared_adj", "fvalue", "f_pvalue", "llf", "aic", "bic", "ssr"):
        assert close(getattr(r, name), ref[name]), name
    lo, hi = r.conf_int(alpha=0.05).T
    assert all(close(a, b) for a, b in zip(lo, ref["conf_int_lower"]))
    assert all(close(a, b) for a, b in zip(hi, ref["conf_int_upper"]))


def test_robust_and_diagnostics_against_statsmodels():
    # the G1b references: HAC, cluster, studentized residuals, Breusch-Pagan, Jarque-Bera
    np = pytest.importorskip("numpy")
    sm = pytest.importorskip("statsmodels.api")
    st = pytest.importorskip("scipy.stats")
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.stats.stattools import durbin_watson, jarque_bera

    rng = np.random.default_rng(2)
    X = rng.standard_normal((40, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + rng.standard_normal(40) * (1 + np.abs(X[:, 0]))
    g = [int(v) for v in rng.integers(0, 8, 40)]
    ref, meta = regref.ols(y.tolist(), X.tolist(), t_guess=float(st.t.ppf(0.975, 36)), groups=g, maxlags=4)
    Xc = sm.add_constant(X)
    close = lambda a, hl, tol=1e-12: abs(a - (hl[0] + hl[1])) <= tol * abs(hl[0])
    hac = sm.OLS(y, Xc).fit(cov_type="HAC", cov_kwds={"maxlags": 4}, use_t=True)
    assert all(close(a, b) for a, b in zip(hac.bse, ref["bse_HAC"]))
    cl = sm.OLS(y, Xc).fit(cov_type="cluster", cov_kwds={"groups": np.array(g)}, use_t=True)
    assert meta["n_groups"] == 8
    for name, theirs in (("bse_cluster", cl.bse), ("pvalues_cluster", cl.pvalues)):
        assert all(close(a, b) for a, b in zip(theirs, ref[name])), name
    assert close(cl.fvalue, ref["fvalue_cluster"], 1e-10) and close(cl.f_pvalue, ref["f_pvalue_cluster"], 1e-10)
    r = sm.OLS(y, Xc).fit()
    inf = r.get_influence()
    for name, theirs in (("resid_studentized_internal", inf.resid_studentized_internal),
                         ("resid_studentized_external", inf.resid_studentized_external)):
        assert all(close(a, b) for a, b in zip(theirs, ref[name])), name
    lm, lmp, f, fp = het_breuschpagan(r.resid, Xc)
    for a, name in ((lm, "bp_lm"), (lmp, "bp_lm_pvalue"), (f, "bp_fvalue"), (fp, "bp_f_pvalue")):
        assert close(a, ref[name]), name
    lm0, lmp0, _, _ = het_breuschpagan(r.resid, Xc, robust=False)
    assert close(lm0, ref["bp_lm_nonstudentized"]) and close(lmp0, ref["bp_lm_nonstudentized_pvalue"])
    jb, jbp, _, _ = jarque_bera(r.wresid)
    assert close(jb, ref["jarque_bera"]) and close(jbp, ref["jarque_bera_pvalue"])
    assert close(durbin_watson(r.wresid), ref["durbin_watson"])
