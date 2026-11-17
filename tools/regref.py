"""Rigorous reference values for linear least squares with Arb (python-flint): the quantities
of an OLS / WLS fit (docs/design/regression.md §6).

The inputs are doubles, so they are exact balls. Everything is computed at one working
precision from the normal equations X^T W X b = X^T W y: at high precision the squared
condition number costs bits, not correctness, and Arb's ball arithmetic carries every
rounding into the radius. When some quantity has not settled (arbref.settled) the whole
fit is redone at twice the precision, up to arbref.MAX_PREC; a fit that cannot settle
raises arbref.NotSettled. A singular X^T W X (rank deficiency) raises ValueError: those
references are built from the full-rank subproblem by the caller.

The result maps each name to a double-double (hi, lo) or a list of them (per coefficient
or per observation); see ols() for the names, which are statsmodels'. Besides the fit and
the classical inference: HC0-HC3, HAC (Newey-West, Bartlett) and one-way cluster standard
errors, hat values, studentized residuals, Cook's distance, Durbin-Watson, Jarque-Bera and
Breusch-Pagan (studentized and not). No mpmath (the
project uses Arb alone).
"""

import math

from flint import arb, arb_mat

import arbref as ar
from flint import ctx

EXACT_FIT_PREC = 1024  # bits from which residuals whose balls still hold 0 are taken as exact zeros


def _t_upper(t, df):
    """P(T > t) for t >= 0, Student t with df degrees of freedom: I_{df/(df+t^2)}(df/2, 1/2) / 2."""
    x = df / (df + t * t)
    y = t * t / (df + t * t)
    return _ibeta(x, df / 2, arb(0.5), y) / 2


def _f_upper(f, dfn, dfd):
    """P(F > f): I_{dfd/(dfd+dfn f)}(dfd/2, dfn/2)."""
    x = dfd / (dfd + dfn * f)
    y = dfn * f / (dfd + dfn * f)
    return _ibeta(x, dfd / 2, dfn / 2, y)


def _ibeta(x, a, b, y):
    """I_x(a, b), regularized, with y = 1 - x given (as tools/stats_ref.py)."""
    if x.is_zero():
        return arb(0)
    if y.is_zero():
        return arb(1)
    v = x.beta_lower(a, b, 1)
    if v.is_finite() and not v.is_nan():
        return v
    return 1 - y.beta_lower(b, a, 1)


def _t_pdf(t, df):
    return ((df + 1) / 2).lgamma().exp() / ((df * arb.pi()).sqrt() * (df / 2).lgamma().exp()) * \
        (1 + t * t / df) ** (-(df + 1) / 2)


def _t_quantile_upper(q, df, guess):
    """The t >= 0 with P(T > t) = q, as a ball that contains it (Newton from a double guess,
    then certified by arbref.certify_root's sign test)."""
    g = lambda x: q - _t_upper(x, df)  # increasing in x
    dg = lambda x: _t_pdf(x, df)
    return ar.solve_settled(g, dg, arb(guess), increasing=True)


def _chi2_upper(x, df):
    """P(X > x), chi-square with df degrees of freedom: Q(df/2, x/2)."""
    if x.is_zero():
        return arb(1)
    return (x / 2).gamma_upper(df / 2, regularized=1)


def _solve_ls(Z, v):
    """Least squares of v on the columns of Z (lists of arbs, rows): (coefficients, fitted)."""
    k = len(Z[0])
    G = arb_mat([[sum((r[a] * r[b] for r in Z), arb(0)) for b in range(k)] for a in range(k)])
    h = [sum((Z[i][a] * v[i] for i in range(len(Z))), arb(0)) for a in range(k)]
    try:
        Gi = G.inv()
    except ZeroDivisionError:
        raise ValueError("singular auxiliary regression")
    c = [sum((Gi[a, b] * h[b] for b in range(k)), arb(0)) for a in range(k)]
    return c, [sum((r[a] * c[a] for a in range(k)), arb(0)) for r in Z]


def _fit(y, X, w, intercept, alpha, t_guess, groups=None, maxlags=None):
    # y, X: doubles (exact balls), or callables giving arbs at the working precision (data
    # given in decimal, polynomial columns as exact powers: the NIST StRD problems)
    if callable(y):
        y = y()
    if callable(X):
        X = X()
    n = len(y)
    rows = [([arb(1)] if intercept else []) + list(r) for r in X]
    k = len(rows[0])
    A = [[arb(v) for v in r] for r in rows]
    Y = [arb(v) for v in y]
    W = [arb(v) for v in w] if w is not None else [arb(1)] * n
    used = [i for i in range(n) if not W[i].is_zero()]  # zero weights: out of the fit (R's lm)
    m = len(used)
    # normal equations
    G = [[sum((W[i] * A[i][a] * A[i][b] for i in used), arb(0)) for b in range(k)] for a in range(k)]
    h = [sum((W[i] * A[i][a] * Y[i] for i in used), arb(0)) for a in range(k)]
    Gm = arb_mat(G)
    try:
        Ginv = Gm.inv()
    except ZeroDivisionError:
        raise ValueError("X^T W X is singular (or not certified nonsingular at this precision)")
    beta = [sum((Ginv[a, b] * h[b] for b in range(k)), arb(0)) for a in range(k)]
    fitted = [sum((A[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(n)]
    resid = [Y[i] - fitted[i] for i in range(n)]
    ssr = sum((W[i] * resid[i] ** 2 for i in used), arb(0))
    df_resid = m - k
    df_model = k - 1 if intercept else k
    if ssr.contains(0) and ctx.prec >= EXACT_FIT_PREC:
        # y in the span of X (NIST Wampler1, Wampler2): residuals 0 to the working precision,
        # which no finite precision certifies; the fit is taken as exact, and the quantities
        # that divide by the residual scale (t, p, intervals, HC, diagnostics) do not exist
        zero = arb(0)
        return ({"params": beta, "fittedvalues": fitted, "resid": [zero] * n, "ssr": zero, "scale": zero,
                 "bse": [zero] * k, "rsquared": arb(1)},
                {"nobs": m, "df_model": df_model, "df_resid": df_resid, "exact_fit": True})
    scale = ssr / df_resid
    bse = [(scale * Ginv[a, a]).sqrt() for a in range(k)]
    tval = [beta[a] / bse[a] for a in range(k)]
    dfr = arb(df_resid)
    pval = [2 * _t_upper(abs(t), dfr) for t in tval]
    # centered (with an intercept) or uncentered total sum of squares, weighted
    sw = sum((W[i] for i in used), arb(0))
    ybar = sum((W[i] * Y[i] for i in used), arb(0)) / sw if intercept else arb(0)
    tss = sum((W[i] * (Y[i] - ybar) ** 2 for i in used), arb(0))
    ess = tss - ssr
    rsq = 1 - ssr / tss
    rsq_adj = 1 - (1 - rsq) * (m - (1 if intercept else 0)) / df_resid
    fval = (ess / df_model) / scale if df_model > 0 else None
    fp = _f_upper(fval, arb(df_model), dfr) if df_model > 0 else None
    # Gaussian log likelihood at the MLE of sigma (statsmodels' llf; weights enter as
    # sum log w / 2)
    llf = -arb(m) / 2 * ((2 * arb.pi()).log() + (ssr / m).log() + 1)
    if w is not None:
        llf = llf + sum((W[i].log() for i in used), arb(0)) / 2
    aic = -2 * llf + 2 * (k + 0)  # statsmodels: df_model + const, i.e. k parameters
    bic = -2 * llf + arb(m).log() * k
    # hat values and the HC covariances (sandwiches on G^-1)
    hat = [W[i] * sum((A[i][a] * Ginv[a, b] * A[i][b] for a in range(k) for b in range(k)), arb(0))
           if not W[i].is_zero() else arb(0) for i in range(n)]

    def sandwich(omega):
        M = [[sum((omega[i] * W[i] * W[i] * A[i][a] * A[i][b] for i in used), arb(0)) for b in range(k)]
             for a in range(k)]
        S = Ginv * arb_mat(M) * Ginv
        return [S[a, a].sqrt() for a in range(k)]

    e2 = [resid[i] ** 2 for i in range(n)]
    hc = {
        "bse_HC0": sandwich(e2),
        "bse_HC1": [v * (arb(m) / df_resid).sqrt() for v in sandwich(e2)],
        "bse_HC2": sandwich([e2[i] / (1 - hat[i]) for i in range(n)]),
        "bse_HC3": sandwich([e2[i] / (1 - hat[i]) ** 2 for i in range(n)]),
    }
    # the scores of the whitened problem, rows of the fit in order: w_i x_i e_i
    score = [[W[i] * A[i][a] * resid[i] for a in range(k)] for i in used]
    # HAC (Newey-West, Bartlett weights 1 - l/(L+1), no small-sample factor: statsmodels' default)
    L = maxlags if maxlags is not None else math.floor(4 * (m / 100) ** (2 / 9))
    S = [[sum((u[a] * u[b] for u in score), arb(0)) for b in range(k)] for a in range(k)]
    for lag in range(1, min(L, m - 1) + 1):
        wl = 1 - arb(lag) / (L + 1)
        for a in range(k):
            for b in range(k):
                S[a][b] += wl * sum((score[t][a] * score[t - lag][b] + score[t - lag][a] * score[t][b]
                                     for t in range(lag, m)), arb(0))
    V = Ginv * arb_mat(S) * Ginv
    hc["bse_HAC"] = [V[a, a].sqrt() for a in range(k)]
    hc_meta = {"maxlags": L}
    if groups is not None:
        # one-way cluster with G/(G-1) (m-1)/(m-k); t and the Wald F with G - 1 degrees of freedom
        labels = [groups[i] for i in used]
        order = list(dict.fromkeys(labels))
        G = len(order)
        sums = {g: [arb(0)] * k for g in order}
        for u, g in zip(score, labels):
            sums[g] = [sums[g][a] + u[a] for a in range(k)]
        S = [[sum((sums[g][a] * sums[g][b] for g in order), arb(0)) for b in range(k)] for a in range(k)]
        V = Ginv * arb_mat(S) * Ginv * (arb(G) / (G - 1) * arb(m - 1) / (m - k))
        bc = [V[a, a].sqrt() for a in range(k)]
        dfc = arb(G - 1)
        hc["bse_cluster"] = bc
        hc["pvalues_cluster"] = [2 * _t_upper(abs(beta[a] / bc[a]), dfc) for a in range(k)]
        q0 = 1 if intercept else 0
        q = k - q0
        if 0 < q <= G - 1:  # the meat has rank G - 1 at most (the scores sum to 0)
            V22 = arb_mat([[V[a, b] for b in range(q0, k)] for a in range(q0, k)])
            b2 = arb_mat([[beta[a]] for a in range(q0, k)])
            try:
                V22i = V22.inv()
            except ZeroDivisionError:
                raise ValueError("cluster covariance not certified nonsingular at this precision")
            fc = (b2.transpose() * V22i * b2)[0, 0] / q
            hc["fvalue_cluster"] = fc
            hc["f_pvalue_cluster"] = _f_upper(fc, arb(q), dfc)
        hc_meta["n_groups"] = G
    # studentized residuals (weighted: sqrt(w) e / (s sqrt(1 - h)); 0 at zero weights)
    st_int = [W[i].sqrt() * resid[i] / (scale * (1 - hat[i])).sqrt() if not W[i].is_zero() else arb(0)
              for i in range(n)]
    # external: the scale without row i, (ssr - w e^2 / (1 - h)) / (df - 1), from its definition
    st_ext = [W[i].sqrt() * resid[i] / ((ssr - W[i] * resid[i] ** 2 / (1 - hat[i])) / (dfr - 1) * (1 - hat[i])).sqrt()
              if not W[i].is_zero() else arb(0) for i in range(n)]
    # Breusch-Pagan: e~^2 on a constant and the whitened design's columns (the intercept
    # column sqrt(w) is the constant itself for OLS)
    ones_w = w is None or all(W[i] == 1 for i in used)
    Z = []
    for i in used:
        sw = W[i].sqrt()
        row = [arb(1)]
        if intercept and not ones_w:
            row.append(sw)
        row += [sw * v for v in A[i][(1 if intercept else 0):]]
        Z.append(row)
    v = [W[i] * resid[i] ** 2 for i in used]
    _, fz = _solve_ls(Z, v)
    vbar = sum(v, arb(0)) / m
    tss_a = sum(((x - vbar) ** 2 for x in v), arb(0))
    ssr_a = sum(((v[j] - fz[j]) ** 2 for j in range(m)), arb(0))
    ess_a = tss_a - ssr_a
    qa = len(Z[0]) - 1
    bp_lm = m * ess_a / tss_a
    s2 = ssr / m
    bp_lm0 = ess_a / (2 * s2 * s2)
    bp_f = (ess_a / qa) / (ssr_a / (m - qa - 1))
    bp = {"bp_lm": bp_lm, "bp_lm_pvalue": _chi2_upper(bp_lm, arb(qa)), "bp_fvalue": bp_f,
          "bp_f_pvalue": _f_upper(bp_f, arb(qa), arb(m - qa - 1)),
          "bp_lm_nonstudentized": bp_lm0, "bp_lm_nonstudentized_pvalue": _chi2_upper(bp_lm0, arb(qa))}
    # confidence intervals: beta -+ t_{1 - alpha/2} bse
    tq = _t_quantile_upper(arb(alpha) / 2, dfr, t_guess)
    ci_lo = [beta[a] - tq * bse[a] for a in range(k)]
    ci_hi = [beta[a] + tq * bse[a] for a in range(k)]
    # influence and residual diagnostics
    cooks = [W[i] * resid[i] ** 2 / (k * scale) * hat[i] / (1 - hat[i]) ** 2 if not W[i].is_zero() else arb(0)
             for i in range(n)]
    sw_resid = [W[i].sqrt() * resid[i] for i in used]
    dw = sum(((sw_resid[j] - sw_resid[j - 1]) ** 2 for j in range(1, m)), arb(0)) / ssr
    # central moments (statsmodels' jarque_bera centers; the mean is 0 with an intercept)
    rbar = sum(sw_resid, arb(0)) / m
    mu2 = sum(((r - rbar) ** 2 for r in sw_resid), arb(0)) / m
    mu3 = sum(((r - rbar) ** 3 for r in sw_resid), arb(0)) / m
    mu4 = sum(((r - rbar) ** 4 for r in sw_resid), arb(0)) / m
    skew = mu3 / mu2 ** arb(1.5)
    kurt = mu4 / mu2 ** 2
    jb = arb(m) / 6 * (skew ** 2 + (kurt - 3) ** 2 / 4)
    jb_p = (-jb / 2).exp()  # chi-square with 2 degrees of freedom
    out = {
        "params": beta, "bse": bse, "tvalues": tval, "pvalues": pval,
        "conf_int_lower": ci_lo, "conf_int_upper": ci_hi,
        "fittedvalues": fitted, "resid": resid, "hat": hat, "cooks_distance": cooks,
        "ssr": ssr, "ess": ess, "centered_tss": tss, "rsquared": rsq, "rsquared_adj": rsq_adj,
        "scale": scale, "llf": llf, "aic": aic, "bic": bic,
        "durbin_watson": dw, "jarque_bera": jb, "jarque_bera_pvalue": jb_p,
        "resid_studentized_internal": st_int, "resid_studentized_external": st_ext,
        **hc, **bp,
    }
    if fval is not None:
        out["fvalue"], out["f_pvalue"] = fval, fp
    meta = {"nobs": m, "df_model": df_model, "df_resid": df_resid, "exact_fit": False, **hc_meta}
    return out, meta


def _all_settled(d):
    for v in d.values():
        for x in (v if isinstance(v, list) else [v]):
            if not (x.is_exact() and x.is_zero()) and not ar.settled(x):
                return False
    return True


def ols(y, X, *, weights=None, intercept=True, alpha=0.05, t_guess=2.0, groups=None, maxlags=None):
    """References for the OLS (weights=None) or WLS fit of y on X (X: a list of rows, without
    the constant column, which intercept puts first; y and X may also be callables that give
    arbs at the working precision). t_guess: a double near the t quantile
    t_{1 - alpha/2} (scipy's value is fine; it only starts Newton). Returns (values, meta):
    values maps each name to (hi, lo) or a list of (hi, lo); meta holds nobs, df_model,
    df_resid, the HAC lags (maxlags: None for floor(4 (nobs/100)^(2/9))) and with groups (one
    label per row) the number of clusters."""
    p = ar.START_PREC
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals, meta = _fit(y, X, weights, intercept, alpha, t_guess, groups, maxlags)
            except (ar.NotSettled, ValueError):
                vals = None
            if vals is not None and _all_settled(vals):
                conv = {k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()}
                return conv, meta
        p *= 2
    raise ar.NotSettled("ols")


def dd_value(hl):
    """hi + lo as one float (for printing)."""
    return hl[0] + hl[1]


if __name__ == "__main__":  # a self-check against statsmodels on a random problem
    import numpy as np
    import scipy.stats as st
    import statsmodels.api as sm

    rng = np.random.default_rng(1)
    X = rng.standard_normal((40, 3))
    y = X @ [1.0, -2.0, 0.5] + 0.3 + rng.standard_normal(40)
    ref, meta = ols(y.tolist(), X.tolist(), t_guess=float(st.t.ppf(0.975, 36)))
    r = sm.OLS(y, sm.add_constant(X)).fit()
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-300)
    worst = {}
    for name, ours in (("params", r.params), ("bse", r.bse), ("pvalues", r.pvalues), ("hat", r.get_influence().hat_matrix_diag),
                       ("bse_HC3", r.HC3_se), ("cooks_distance", r.get_influence().cooks_distance[0])):
        worst[name] = max(rel(o, dd_value(t)) for o, t in zip(ours, ref[name]))
    for name, ours in (("rsquared", r.rsquared), ("fvalue", r.fvalue), ("llf", r.llf), ("aic", r.aic), ("bic", r.bic)):
        worst[name] = rel(ours, dd_value(ref[name]))
    ci = r.conf_int(0.05)
    worst["conf_int"] = max(max(rel(ci[i][0], dd_value(ref["conf_int_lower"][i])), rel(ci[i][1], dd_value(ref["conf_int_upper"][i])))
                            for i in range(4))
    for k, v in worst.items():
        print(f"{k:16s} statsmodels vs Arb: {v:.2e}")
    print(meta)
