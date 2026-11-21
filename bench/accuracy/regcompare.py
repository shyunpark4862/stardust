"""Comparators of the regression area: statsmodels, scikit-learn and R on the reference problems.

The references (tests/regression/ref, tools/regref.py) follow statsmodels' definitions, so
statsmodels' results are read field by field through the same mapping as sd's (sources._reg_value),
behind an adapter that gives them sd's few differing names. scikit-learn's ElasticNet has sd's
penalized objective (1/(2n) RSS + alpha (l1_ratio |b|_1 + (1 - l1_ratio)/2 |b|^2)) for the
Gaussian problems without standardization. R: lm and glm (stats), rlm (MASS) for the M problems,
lmrob (robustbase) for MM, glmnet for the penalized ones (its objective is the same, with
standardize). The tests on fitted models: statsmodels' functions of the same names
(sm_model_tests_values) and R's lm / glm with car and lmtest (r_model_tests_values, the calls of
tools/gen_model_tests_r.py). Variable selection: statsmodels refitting each subset along the
reference's stepwise paths (sm_selection_values) and leaps' regsubsets for best_subset's Cp and
adjusted R^2 (r_selection_values). A library without the model, or without a field's quantity, is
left out ("—").
"""

from __future__ import annotations

import warnings

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
    if kind == "model_tests":
        return r_model_tests_values(sw)
    if kind == "selection":
        return r_selection_values(sw)
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
        # R drops a column it finds collinear (lm's tolerance 1e-7): its coefficient is NA, and
        # the measurement leaves those points out and counts them
        expr = (pre + f"f <- {fit}; s <- summary(f); b <- coef(f); se <- sqrt(diag(vcov(f, complete = TRUE))); "
                "t <- b / se; p <- 2 * pt(-abs(t), df.residual(f)); "
                "c(b, se, t, p, residuals(f), fitted(f), s$sigma^2, s$r.squared, s$adj.r.squared, "
                "as.numeric(logLik(f)))")  # fmt: skip
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


# ------------------------------------------------------------------------------ tests on fitted models


def sm_model_tests_values(sw) -> dict:
    """statsmodels' values of the model_tests fields it computes the same way: t_test, f_test /
    wald_test, compare_f_test, compare_lr_test, variance_inflation_factor, OLSInfluence /
    GLMInfluence(observed=False), het_white, het_goldfeldquandt and linear_reset. Left out (no
    such function, or another quantity): anova_lm (it needs a formula), White without cross
    terms (statsmodels keeps the squares only), and for weighted fits VIF, Goldfeld-Quandt and
    RESET (statsmodels' drop the weights); the GLM comparisons (GLMResults has no
    compare_lr_test)."""
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_goldfeldquandt, het_white, linear_reset
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    p = sw.problem
    y = np.asarray(p["y"], dtype=float)
    X = np.asarray(p["X"], dtype=float)
    Rm = np.asarray(p["R"], dtype=float)
    out = {}

    def put(fields, make):
        try:
            with warnings.catch_warnings():  # statsmodels' notes on conditioning and rank
                warnings.simplefilter("ignore")
                vals = make()
        except Exception:  # noqa: BLE001 -- a value statsmodels cannot give is left out
            return
        for f, v in zip(fields, vals):
            out[f] = np.ravel(np.asarray(v, dtype=float))

    if p["kind"] == "glm":
        fixed = p["family"] in ("binomial", "poisson")
        ex = sm.add_constant(X, has_constant="add")
        b = sm.GLM(y, ex, family=_sm_family(p["family"], p["link"])).fit(tol=1e-14, maxiter=1000, use_t=not fixed)
        put(("t_tvalue", "t_pvalue"), lambda: (lambda t: (t.tvalue, t.pvalue))(b.t_test(Rm)))
        put(("wald_stat", "wald_p"),
            lambda: (lambda w: (w.statistic, w.pvalue))(b.wald_test(Rm, use_f=not fixed, scalar=True)))  # fmt: skip
        inf = b.get_influence(observed=False)
        put(("hat",), lambda: (inf.hat_matrix_diag,))
        put(("cooks",), lambda: (inf.cooks_distance[0],))
        put(("dfbetas",), lambda: (inf.dfbetas,))
        return out
    ex = sm.add_constant(X, has_constant="add") if p["intercept"] else X
    j0 = 1 if p["intercept"] else 0
    weighted = p["weights"] is not None
    model = sm.WLS(y, ex, weights=p["weights"]) if weighted else sm.OLS(y, ex)
    s = model.fit()
    exs = ex[:, : j0 + p["small"]]
    ss = (sm.WLS(y, exs, weights=p["weights"]) if weighted else sm.OLS(y, exs)).fit()
    q = np.asarray(p["q"], dtype=float)
    put(("t_effect", "t_sd", "t_tvalue", "t_pvalue"),
        lambda: (lambda t: (t.effect, t.sd, t.tvalue, t.pvalue))(s.t_test((Rm, q))))  # fmt: skip
    put(("wald_F", "wald_p"), lambda: (lambda w: (w.fvalue, w.pvalue))(s.f_test((Rm, q))))
    put(("compare_F", "compare_F_p"), lambda: s.compare_f_test(ss)[:2])
    put(("compare_lr", "compare_lr_p"), lambda: s.compare_lr_test(ss)[:2])
    put(("dffits", "dfbetas"), lambda: (lambda inf: (inf.dffits[0], inf.dfbetas))(s.get_influence()))
    if sw.name.split("/", 1)[1] not in ("cond",):  # sd's White on cond is another quantity (refdata_model_tests)
        put(("white_lm", "white_lm_p", "white_F", "white_F_p"), lambda: het_white(s.wresid, s.model.wexog))
    if not weighted:
        put(("vif",), lambda: ([variance_inflation_factor(ex, j) for j in range(j0, ex.shape[1])],))
        n1, drop, by = p["gq"]
        idx = None if by is None else j0 + by
        put(("gq_F", "gq_p"), lambda: het_goldfeldquandt(y, ex, idx=idx, split=n1 if n1 is not None else len(y) // 2,
                                                        drop=drop or None, alternative="increasing",
                                                        result_object=False)[:2])  # fmt: skip
        put(("reset_F", "reset_p"), lambda: (lambda r: (r.fvalue, r.pvalue))(linear_reset(s, power=3, use_f=True)))
    return out


def _rmat(name, A):
    return f"{name} <- matrix({name}, nrow = {len(A)}, byrow = TRUE); "


def _rcols(cols):
    return "c(" + ", ".join(str(c + 1) for c in cols) + ")"


def r_model_tests_values(sw) -> dict | None:
    """R's values of the model_tests fields, by tools/gen_model_tests_r.py's calls (as
    tests/regression/test_model_tests_against_r.py pairs them with sd's): for lm, the t from
    vcov (R arithmetic), car::linearHypothesis, anova(small, big), lmtest::lrtest, anova (Type I)
    and drop1(test = "F") (Type III) of the terms as matrix terms, car::vif (with an intercept),
    dffits and dfbetas (cases of weight 0, which lm.influence leaves out, are NaN), and with unit
    weights gqtest, resettest(type = "fitted") and bptest with White's auxiliary formula (its LM
    and p only: bptest has no F). For glm: the t from vcov (z for a fixed dispersion),
    car::linearHypothesis (F for the gamma), lmtest::lrtest (not for the gamma: R's logLik takes
    the deviance / n scale), anova(small, big, test = "F") (gamma), anova (Type I),
    car::Anova(type = 3, test.statistic = "Wald"), hatvalues, cooks.distance, dffits, dfbetas;
    each glm is refit (epsilon = 1e-14) from its last coefficients until they stop changing: R's
    IRLS stops on the deviance's relative change, with the coefficients near sqrt(epsilon) of the
    root, and Fisher scoring (probit, the gamma's log) gains little per fit. A call R refuses
    gives NaN for its fields; so do all fields where R drops a column it finds collinear (lm's
    tolerance 1e-7, R's default kept as for the fits): its tests are of another model."""
    p = sw.problem
    y, X = np.asarray(p["y"], dtype=float), np.asarray(p["X"], dtype=float)
    Rm = np.asarray(p["R"], dtype=float)
    n, k = X.shape
    icpt = p.get("intercept", True)
    m = k + (1 if icpt else 0)
    terms = p["terms"]
    nt, r = len(terms), len(Rm)
    nm = [f"x{j + 1}" for j in range(k)]
    zero = "" if icpt else "0 + "
    data = {"y": y, "X": X.ravel(), "Rm": Rm.ravel()}
    pre = (_rmat("X", X) + _rmat("Rm", Rm) + "suppressMessages({library(car); library(lmtest)}); "
           "d <- data.frame(y = y); "
           + "".join(f"d${c} <- X[, {j + 1}]; " for j, c in enumerate(nm))
           + "".join(f"T{i} <- X[, {_rcols(t)}, drop = FALSE]; " for i, t in enumerate(terms))
           + ".tc <- function(e, len) { if (.aliased) return(rep(NaN, len)); "
           "v <- tryCatch(suppressWarnings(as.numeric(e)), error = function(err) NULL); "
           "if (length(v) != len) rep(NaN, len) else v }; ")  # fmt: skip
    fb = f"y ~ {zero}" + " + ".join(nm)
    fs = f"y ~ {zero}" + " + ".join(nm[: p["small"]])
    ft = f"y ~ {zero}" + " + ".join(f"T{i}" for i in range(nt))
    items = []  # (field, size, R expression): each evaluated alone, NaN where R refuses it

    def add(fields, sizes, expr):
        items.append((fields, sizes, expr))

    if p["kind"] == "glm":
        fam = {"binomial": "binomial", "poisson": "poisson", "gamma": "Gamma"}[p["family"]]
        fixed = p["family"] in ("binomial", "poisson")
        ctl = "control = glm.control(epsilon = 1e-14, maxit = 1000)"

        g = f'glm(f, family = {fam}(link = "{p["link"]}"), data = d, {ctl}'
        fits = (f".fit <- function(f) {{ m <- {g}); for (i in 1:100) {{ b0 <- coef(m); m <- {g}, start = b0); "
                "if (max(abs(coef(m) - b0)) <= 4 * .Machine$double.eps * max(1, abs(b0))) break }; m }; "
                f"big <- .fit({fb}); small <- .fit({fs}); bt <- .fit({ft}); .aliased <- anyNA(coef(big)); ")  # fmt: skip
        tvals = "{ est <- drop(Rm %*% coef(big)); se <- sqrt(diag(Rm %*% vcov(big) %*% t(Rm))); tv <- est / se; " + (
            "c(tv, 2 * pnorm(-abs(tv))) }" if fixed else "c(tv, 2 * pt(-abs(tv), df.residual(big))) }"
        )
        add(("t_tvalue", "t_pvalue"), (r, r), tvals)
        if fixed:
            add(
                ("wald_stat", "wald_p"),
                (1, 1),
                "{ lh <- linearHypothesis(big, Rm); c(lh$Chisq[2], lh[2, 'Pr(>Chisq)']) }",
            )
            add(
                ("compare_lr", "compare_lr_p"),
                (1, 1),
                "{ lr <- lrtest(small, big); c(lr$Chisq[2], lr[2, 'Pr(>Chisq)']) }",
            )
        else:
            add(("wald_stat", "wald_p"), (1, 1),
                "{ lh <- linearHypothesis(big, Rm, test = 'F'); c(lh$F[2], lh[2, 'Pr(>F)']) }")  # fmt: skip
            add(
                ("compare_F", "compare_F_p"),
                (1, 1),
                "{ a2 <- anova(small, big, test = 'F'); c(a2$F[2], a2[2, 'Pr(>F)']) }",
            )
        add(("anova1_dev", "anova1_p"), (nt, nt),
            f"{{ a1 <- anova(bt, test = '{'Chisq' if fixed else 'F'}'); c(a1$Deviance[-1], a1[-1, ncol(a1)]) }}")  # fmt: skip
        add(("anova3_wald",), (nt,), "Anova(bt, type = 3, test.statistic = 'Wald')$Chisq[-1]")
        add(("hat",), (n,), "hatvalues(big)")
        add(("cooks",), (n,), "cooks.distance(big)")
        add(("dffits",), (n,), "dffits(big)")
        add(("dfbetas",), (n * m,), "t(dfbetas(big))")
    else:
        w = np.ones(n) if p["weights"] is None else np.asarray(p["weights"], dtype=float)
        data["w"], data["q"] = w, np.asarray(p["q"], dtype=float)
        fits = (f"d$w <- w; big <- lm({fb}, data = d, weights = w); small <- lm({fs}, data = d, weights = w); "
                f"bt <- lm({ft}, weights = w); keep <- w > 0; .aliased <- anyNA(coef(big)); "
                f"full <- function(v, each = 1) {{ if (length(v) == {n} * each) return(v); "
                f"o <- matrix(NaN, each, {n}); o[, keep] <- matrix(v, nrow = each); as.numeric(o) }}; ")  # fmt: skip
        add(("t_effect", "t_sd", "t_tvalue", "t_pvalue"), (r, r, r, r),
            "{ est <- drop(Rm %*% coef(big)); se <- sqrt(diag(Rm %*% vcov(big) %*% t(Rm))); tv <- (est - q) / se; "
            "c(est, se, tv, 2 * pt(abs(tv), df.residual(big), lower.tail = FALSE)) }")  # fmt: skip
        add(("wald_F", "wald_p"), (1, 1), "{ lh <- linearHypothesis(big, Rm, q); c(lh$F[2], lh[2, 'Pr(>F)']) }")
        add(("compare_F", "compare_F_p"), (1, 1), "{ a2 <- anova(small, big); c(a2$F[2], a2[2, 'Pr(>F)']) }")
        add(("compare_lr", "compare_lr_p"), (1, 1), "{ lr <- lrtest(small, big); c(lr$Chisq[2], lr[2, 'Pr(>Chisq)']) }")
        add(("anova1_ss", "anova1_F", "anova1_p"), (nt, nt, nt),
            f"{{ a1 <- anova(bt); c(a1[['Sum Sq']][1:{nt}], a1[['F value']][1:{nt}], a1[['Pr(>F)']][1:{nt}]) }}")  # fmt: skip
        add(("anova3_ss", "anova3_F", "anova3_p"), (nt, nt, nt),
            "{ d1 <- drop1(bt, test = 'F'); c(d1[['Sum of Sq']][-1], d1[['F value']][-1], d1[['Pr(>F)']][-1]) }")  # fmt: skip
        if icpt:  # car's vif without an intercept is another quantity (it warns so)
            add(("vif",), (k,), "vif(big)")
        add(("dffits",), (n,), "full(dffits(big))")
        add(("dfbetas",), (n * m,), f"full(as.numeric(t(dfbetas(big))), {m})")
        if np.all(w == 1):  # lmtest's gqtest, resettest and bptest take no weights
            n1, drop, by = p["gq"]
            n1 = n // 2 if n1 is None else n1
            point = n1 + (drop + 1) // 2  # gqtest's point and fraction: the first n1 rows, then drop rows left out
            order = "" if by is None else f", order.by = X[, {by + 1}]"
            fits += (f"bu <- lm({fb}, data = d); Z <- X; aux <- list(Z); "
                     "for (i in seq_len(ncol(Z))) aux[[length(aux) + 1]] <- Z[, i] * Z[, i:ncol(Z), drop = FALSE]; "
                     "A <- do.call(cbind, aux); A0 <- cbind(Z, Z^2); ")  # fmt: skip
            add(("gq_F", "gq_p"), (1, 1),
                f"{{ g <- gqtest(bu, point = {point}, fraction = {drop}{order}, alternative = 'greater'); "
                "c(g$statistic, g$p.value) }")  # fmt: skip
            add(("reset_F", "reset_p"), (1, 1),
                "{ g <- resettest(bu, power = 2:3, type = 'fitted'); c(g$statistic, g$p.value) }")  # fmt: skip
            add(("white_lm", "white_lm_p"), (1, 1),
                "{ g <- bptest(bu, varformula = ~ A, data = data.frame(A = I(A))); c(g$statistic, g$p.value) }")  # fmt: skip
            add(("white_nocross_lm", "white_nocross_lm_p"), (1, 1),
                "{ g <- bptest(bu, varformula = ~ A0, data = data.frame(A0 = I(A0))); c(g$statistic, g$p.value) }")  # fmt: skip
    names = [(f, z) for fields, sizes, _ in items for f, z in zip(fields, sizes)]
    expr = pre + fits + "c(" + ", ".join(f".tc({e}, {sum(z)})" for _, z, e in items) + ")"
    vals = _r_split(expr, data, names)
    return {f: v for f, v in vals.items() if f in p["ref"]}


# ------------------------------------------------------------------------------ variable selection


def _positive_rows(p):
    """y, X of the rows with a positive weight (sd's weighted fits leave the others out), and those weights."""
    y, X = np.asarray(p["y"], dtype=float), np.asarray(p["X"], dtype=float)
    w = np.ones(len(y)) if p.get("weights") is None else np.asarray(p["weights"], dtype=float)
    keep = w > 0
    return y[keep], X[keep], w[keep]


def r_selection_values(sw) -> dict | None:
    """leaps' regsubsets (exhaustive) on a linear problem: summary's cp and adjr2 per size, the
    quantities of best_subset's 'cp' and 'adjr2' (each size's best subset by RSS); leaps has no
    size 0 (the intercept alone): NaN there. GLMs: none (leaps fits linear models)."""
    p = sw.problem
    if p["kind"] == "glm":
        return None
    y, X, w = _positive_rows(p)
    n, k = X.shape
    expr = (_rmat("X", X) + "suppressMessages(library(leaps)); "
            f"r <- regsubsets(X, y, weights = w, nvmax = {k}, intercept = {'TRUE' if p['intercept'] else 'FALSE'}, "
            "method = 'exhaustive'); s <- summary(r); c(s$cp, s$adjr2)")  # fmt: skip
    v = rsession.evaluate_named(expr, {"y": y, "X": X.ravel(), "w": w})
    out = {}
    for i, f in enumerate(("best_cp", "best_adjr2")):
        if f not in p["ref"]:
            continue
        vals = v[i * k : (i + 1) * k]
        if len(p["ref"][f]) == k + 1:
            vals = np.r_[np.nan, vals]
        if len(vals) == len(p["ref"][f]):
            out[f] = vals
    return out


def sm_selection_values(sw) -> dict:
    """statsmodels' AIC and BIC (GLM: bic_llf) of every model along the reference's stepwise
    paths (the moves of tests/regression/ref/selection.json from the full model, or for forward
    from no column, or the first without an intercept, as refdata_selection.stepwise starts),
    each subset refit on the rows of positive weight; best_subset's aic and bic for each size's
    reference subset. Cross-validation has no comparator (the folds and the score are sd's)."""
    import statsmodels.api as sm

    p = sw.problem
    y, X, w = _positive_rows(p)
    k = X.shape[1]
    glm = p["kind"] == "glm"
    icpt = p.get("intercept", True)

    def crit(cols, which):
        Z = X[:, sorted(cols)]
        if icpt:
            Z = sm.add_constant(Z, has_constant="add") if Z.size else np.ones((len(y), 1))
        if Z.shape[1] == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if glm:
                r = sm.GLM(y, Z, family=_sm_family(p["family"], p["link"])).fit(tol=1e-14, maxiter=1000)
                return float(r.aic if which == "aic" else r.bic_llf)
            r = sm.WLS(y, Z, weights=w).fit()
            return float(r.aic if which == "aic" else r.bic)

    out = {}
    for field, ref in p["ref"].items():
        kind, rest = field.split("_", 1)
        if kind == "step":
            direction, which = rest.split("_")
            if which not in ("aic", "bic"):
                continue
            cols = set(range(k)) if direction != "forward" else (set() if icpt else {0})
            vals = [crit(cols, which)]
            for action, j in p["runs"][f"{direction}/{which}"]["moves"]:
                cols = cols | {j} if action == "add" else cols - {j}
                vals.append(crit(cols, which))
        elif rest in ("aic", "bic"):
            sets = [s for s in p["best"][rest]["sets"] if s is not None]  # a size without a subset (no intercept, 0)
            vals = [crit({j for j in range(k) if s >> j & 1}, rest) for s in sets]
            vals = [v for v in vals if not np.isnan(v)]
        else:
            continue
        if len(vals) == len(ref):
            out[field] = np.asarray(vals, dtype=float)
    return out
