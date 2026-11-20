"""Comparators of the regression area: statsmodels, scikit-learn and R on the reference problems.

The references (tests/regression/ref, tools/regref.py) follow statsmodels' definitions, so
statsmodels' results are read field by field through the same mapping as sd's (sources._reg_value),
behind an adapter that gives them sd's few differing names. scikit-learn's ElasticNet has sd's
penalized objective (1/(2n) RSS + alpha (l1_ratio |b|_1 + (1 - l1_ratio)/2 |b|^2)) for the
Gaussian problems without standardization. R: lm and glm (stats), rlm (MASS) for the M problems,
lmrob (robustbase) for MM, glmnet for the penalized ones (its objective is the same, with
standardize). A library without the model, or without a field's quantity, is left out ("—").
"""

from __future__ import annotations

import numpy as np

from . import rsession

# ------------------------------------------------------------------------------ statsmodels


class _SM:
    """A statsmodels result with sd's names where they differ."""

    def __init__(self, res, kind):
        self._r, self._kind = res, kind

    def __getattr__(self, name):
        return getattr(self._r, name)

    @property
    def leverage(self):
        return self._r.get_influence().hat_matrix_diag

    @property
    def cooks_distance(self):
        return self._r.get_influence().cooks_distance[0]

    @property
    def resid_studentized_internal(self):
        return self._r.get_influence().resid_studentized_internal

    @property
    def resid_studentized_external(self):
        return self._r.get_influence().resid_studentized_external

    @property
    def durbin_watson(self):
        from statsmodels.stats.stattools import durbin_watson

        return durbin_watson(self._r.wresid)

    @property
    def llnull(self):  # statsmodels' null model keeps the full model's scale: another quantity
        raise AttributeError("llnull")

    @property
    def bic(self):
        return self._r.bic_llf if self._kind == "glm" else self._r.bic

    def jarque_bera(self):
        from types import SimpleNamespace

        from statsmodels.stats.stattools import jarque_bera

        s, p, _, _ = jarque_bera(self._r.wresid)
        return SimpleNamespace(statistic=s, pvalue=p)

    def breusch_pagan(self, studentize=True):
        from types import SimpleNamespace

        from statsmodels.stats.diagnostic import het_breuschpagan

        lm, lmp, f, fp = het_breuschpagan(self._r.resid, self._r.model.exog, robust=studentize)
        return SimpleNamespace(lm=lm, lm_pvalue=lmp, fvalue=f, f_pvalue=fp)


def _exog(p):
    import statsmodels.api as sm

    X = np.asarray(p["X"], dtype=float)
    return sm.add_constant(X, has_constant="add") if p["intercept"] else X


def sm_fit(sw, cov_type="nonrobust", groups=None):
    """statsmodels' fit of a problem as an adapter, or None where it has no such model."""
    import statsmodels.api as sm

    p, kind = sw.problem, sw.kind
    y, X = np.asarray(p["y"], dtype=float), _exog(p)
    kw = {}
    if cov_type == "HAC":
        kw = {"cov_type": "HAC", "cov_kwds": {"maxlags": p["meta"]["maxlags"]}}
    elif cov_type == "cluster":
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": np.asarray(groups)}}
    elif cov_type != "nonrobust":
        kw = {"cov_type": cov_type}
    if kind == "linear":
        m = sm.WLS(y, X, weights=p["weights"]) if p["weights"] is not None else sm.OLS(y, X)
        return _SM(m.fit(**kw), kind)
    if kind == "glm":
        fam = _sm_family(p["family"], p["link"])
        extra = {}
        if p.get("_exposure") is not None:
            extra = {"offset": p.get("_offset"), "exposure": p["_exposure"]}
        elif p["offset"] is not None:
            extra = {"offset": p["offset"]}
        if p["var_weights"] is not None:
            extra["var_weights"] = p["var_weights"]
        if p["freq_weights"] is not None:
            extra["freq_weights"] = p["freq_weights"]
        if p["trials"] is not None:  # sd's proportions with trials: statsmodels' (successes, failures)
            t = np.asarray(p["trials"], dtype=float)
            y = np.column_stack([y * t, t - y * t])
        scale = {"dev": "dev", None: None}.get(p["scale"], p["scale"])
        m = sm.GLM(y, X, family=fam, **extra)
        use_t = p["family"] in ("gaussian", "gamma", "inverse_gaussian")  # the references' t for an estimated scale
        return _SM(m.fit(scale=scale, tol=1e-14, maxiter=1000, use_t=use_t, **kw), kind)
    if kind == "robust" and p["method"] == "M":
        norm = {"huber": sm.robust.norms.HuberT(), "bisquare": sm.robust.norms.TukeyBiweight(),
                "hampel": sm.robust.norms.Hampel(), "andrew": sm.robust.norms.AndrewWave()}.get(p["norm"])  # fmt: skip
        if norm is None:
            return None
        scale_est = {"mad": "mad", "huber": sm.robust.scale.HuberScale()}.get(p.get("scale_est"))
        if scale_est is None:
            return None
        return _SM(sm.RLM(y, X, M=norm).fit(scale_est=scale_est, tol=1e-14, maxiter=1000), kind)
    return None


def _sm_family(family, link):
    import statsmodels.api as sm

    L = sm.families.links
    links = {"identity": L.Identity(), "log": L.Log(), "logit": L.Logit(), "probit": L.Probit(),
             "cloglog": L.CLogLog(), "inverse": L.InversePower(), "inverse_squared": L.InverseSquared(),
             "sqrt": L.Sqrt()}  # fmt: skip
    fams = {"gaussian": sm.families.Gaussian, "binomial": sm.families.Binomial, "poisson": sm.families.Poisson,
            "gamma": sm.families.Gamma, "inverse_gaussian": sm.families.InverseGaussian}  # fmt: skip
    return fams[family](link=links[link])


# ------------------------------------------------------------------------------ scikit-learn


def sklearn_values(sw) -> dict | None:
    p = sw.problem
    if sw.kind != "penalized" or p["family"] != "gaussian" or p["standardize"]:
        return None
    from sklearn.linear_model import ElasticNet

    y, X = np.asarray(p["y"], dtype=float), np.asarray(p["X"], dtype=float)
    m = ElasticNet(alpha=p["alpha"], l1_ratio=p["l1_ratio"], fit_intercept=p["intercept"], tol=1e-14,
                   max_iter=10**6, selection="cyclic")  # fmt: skip
    m.fit(X, y, sample_weight=p["weights"])
    params = np.r_[m.intercept_, m.coef_] if p["intercept"] else m.coef_
    r = y - m.predict(X)
    w = np.ones_like(y) if p["weights"] is None else np.asarray(p["weights"], dtype=float)
    return {"params": params, "ssr": float(np.sum(w * r * r))}


# ------------------------------------------------------------------------------ R


def r_values(sw) -> dict | None:
    """{field: values} from R, for the fields R reports the same way."""
    p, kind = sw.problem, sw.kind
    y, X = np.asarray(p["y"], dtype=float), np.asarray(p["X"], dtype=float)
    n, k = X.shape
    icpt = "" if p["intercept"] else " - 1"
    data = {"y": y, "X": X.ravel()}
    pre = f"X <- matrix(X, nrow = {n}, byrow = TRUE); "
    if kind == "linear":
        if p["weights"] is not None:
            data["w"] = np.asarray(p["weights"], dtype=float)
            fit = f"lm(y ~ X{icpt}, weights = w)"
        else:
            fit = f"lm(y ~ X{icpt})"
        # (R's AIC and BIC count sigma as a parameter: another quantity than statsmodels' and the references')
        expr = (pre + f"f <- {fit}; s <- summary(f); "
                "c(coef(f), s$coefficients[, 2], s$coefficients[, 3], s$coefficients[, 4], residuals(f), "
                "fitted(f), s$sigma^2, s$r.squared, s$adj.r.squared, as.numeric(logLik(f)))")  # fmt: skip
        m = k + (1 if p["intercept"] else 0)
        names = [("params", m), ("bse", m), ("tvalues", m), ("pvalues", m), ("resid", n), ("fittedvalues", n),
                 ("scale", 1), ("rsquared", 1), ("rsquared_adj", 1), ("llf", 1)]  # fmt: skip
        if p["weights"] is not None:  # R's weighted residuals and likelihood differ in what they include
            names = [t for t in names if t[0] not in ("resid", "llf")]
            expr = expr.replace("residuals(f), fitted(f), ", "fitted(f), ").replace(", as.numeric(logLik(f))", "")
        return _r_split(expr, data, names)
    if kind == "glm":
        if p["var_weights"] is not None or p["freq_weights"] is not None or p.get("_exposure") is not None:
            return None
        fam = {"gaussian": "gaussian", "binomial": "binomial", "poisson": "poisson", "gamma": "Gamma",
               "inverse_gaussian": "inverse.gaussian"}[p["family"]]  # fmt: skip
        link = {"inverse_squared": "1/mu^2"}.get(p["link"], p["link"])
        extra = ""
        if p["trials"] is not None:
            data["t"] = np.asarray(p["trials"], dtype=float)
            extra = ", weights = t"
        if p["offset"] is not None:
            data["o"] = np.asarray(p["offset"], dtype=float)
            extra += ", offset = o"
        m = k + (1 if p["intercept"] else 0)
        expr = (pre + f'f <- glm(y ~ X{icpt}, family = {fam}(link = "{link}"){extra}, '
                "control = glm.control(epsilon = 1e-14, maxit = 1000)); "
                "c(coef(f), deviance(f), f$null.deviance, fitted(f))")  # fmt: skip
        return _r_split(expr, data, [("params", m), ("deviance", 1), ("null_deviance", 1), ("fittedvalues", n)])
    if kind == "robust":
        m = k + (1 if p["intercept"] else 0)
        if p["method"] == "M" and p["norm"] == "huber" and p.get("scale_est") == "mad":
            expr = (pre + f"suppressWarnings(library(MASS)); f <- rlm(y ~ X{icpt}, psi = psi.huber, "
                    'scale.est = "MAD", acc = 1e-14, maxit = 1000); c(coef(f), f$s, f$w)')  # fmt: skip
            return _r_split(expr, data, [("params", m), ("scale", 1), ("weights", n)])
        if p["method"] == "MM":
            expr = (pre + f"suppressWarnings(library(robustbase)); f <- lmrob(y ~ X{icpt}); "
                    "c(coef(f), f$scale)")  # fmt: skip
            return _r_split(expr, data, [("params", m), ("scale", 1)])
        return None
    if kind == "penalized":
        fam = {"gaussian": "gaussian", "binomial": "binomial", "poisson": "poisson"}.get(p["family"])
        if fam is None:
            return None
        w = ""
        if p["weights"] is not None:
            data["w"] = np.asarray(p["weights"], dtype=float)
            w = ", weights = w"
        m = k + (1 if p["intercept"] else 0)
        sel = "" if p["intercept"] else "[-1]"
        expr = (pre + "suppressWarnings(library(glmnet)); "
                f'f <- glmnet(X, y, family = "{fam}", alpha = {p["l1_ratio"]!r}, lambda = {p["alpha"]!r}, '
                f'standardize = {"TRUE" if p["standardize"] else "FALSE"}, '
                f'intercept = {"TRUE" if p["intercept"] else "FALSE"}, thresh = 1e-20, maxit = 10^7{w}); '
                f"as.numeric(coef(f)){sel}")  # fmt: skip
        return _r_split(expr, data, [("params", m)])
    return None


def _r_split(expr, data, names):
    vals = rsession.evaluate_named(expr, data)
    out, i = {}, 0
    for name, size in names:
        out[name] = vals[i : i + size]
        i += size
    if i != vals.size:
        raise ValueError(f"R returned {vals.size} values, expected {i}")
    return out
