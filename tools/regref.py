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
Breusch-Pagan (studentized and not). The tests on fitted models (linear hypotheses,
comparisons, the analysis of variance and the added diagnostics; model_tests_linear and
model_tests_glm) are at the end. No mpmath (the project uses Arb alone).
"""

import math

import arbref as ar
from flint import arb, arb_mat, ctx

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
    return (
        ((df + 1) / 2).lgamma().exp()
        / ((df * arb.pi()).sqrt() * (df / 2).lgamma().exp())
        * (1 + t * t / df) ** (-(df + 1) / 2)
    )


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
        raise ValueError("singular auxiliary regression") from None
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
        raise ValueError("X^T W X is singular (or not certified nonsingular at this precision)") from None
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
        return (
            {
                "params": beta,
                "fittedvalues": fitted,
                "resid": [zero] * n,
                "ssr": zero,
                "scale": zero,
                "bse": [zero] * k,
                "rsquared": arb(1),
            },
            {"nobs": m, "df_model": df_model, "df_resid": df_resid, "exact_fit": True},
        )
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
    hat = [
        W[i] * sum((A[i][a] * Ginv[a, b] * A[i][b] for a in range(k) for b in range(k)), arb(0))
        if not W[i].is_zero()
        else arb(0)
        for i in range(n)
    ]

    def sandwich(omega):
        M = [
            [sum((omega[i] * W[i] * W[i] * A[i][a] * A[i][b] for i in used), arb(0)) for b in range(k)]
            for a in range(k)
        ]
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
                S[a][b] += wl * sum(
                    (score[t][a] * score[t - lag][b] + score[t - lag][a] * score[t][b] for t in range(lag, m)), arb(0)
                )
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
                raise ValueError("cluster covariance not certified nonsingular at this precision") from None
            fc = (b2.transpose() * V22i * b2)[0, 0] / q
            hc["fvalue_cluster"] = fc
            hc["f_pvalue_cluster"] = _f_upper(fc, arb(q), dfc)
        hc_meta["n_groups"] = G
    # studentized residuals (weighted: sqrt(w) e / (s sqrt(1 - h)); 0 at zero weights)
    st_int = [
        W[i].sqrt() * resid[i] / (scale * (1 - hat[i])).sqrt() if not W[i].is_zero() else arb(0) for i in range(n)
    ]
    # external: the scale without row i, (ssr - w e^2 / (1 - h)) / (df - 1), from its definition
    st_ext = [
        W[i].sqrt() * resid[i] / ((ssr - W[i] * resid[i] ** 2 / (1 - hat[i])) / (dfr - 1) * (1 - hat[i])).sqrt()
        if not W[i].is_zero()
        else arb(0)
        for i in range(n)
    ]
    # Breusch-Pagan: e~^2 on a constant and the whitened design's columns (the intercept
    # column sqrt(w) is the constant itself for OLS)
    ones_w = w is None or all(W[i] == 1 for i in used)
    Z = []
    for i in used:
        sw = W[i].sqrt()
        row = [arb(1)]
        if intercept and not ones_w:
            row.append(sw)
        row += [sw * v for v in A[i][(1 if intercept else 0) :]]
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
    bp = {
        "bp_lm": bp_lm,
        "bp_lm_pvalue": _chi2_upper(bp_lm, arb(qa)),
        "bp_fvalue": bp_f,
        "bp_f_pvalue": _f_upper(bp_f, arb(qa), arb(m - qa - 1)),
        "bp_lm_nonstudentized": bp_lm0,
        "bp_lm_nonstudentized_pvalue": _chi2_upper(bp_lm0, arb(qa)),
    }
    # confidence intervals: beta -+ t_{1 - alpha/2} bse
    tq = _t_quantile_upper(arb(alpha) / 2, dfr, t_guess)
    ci_lo = [beta[a] - tq * bse[a] for a in range(k)]
    ci_hi = [beta[a] + tq * bse[a] for a in range(k)]
    # influence and residual diagnostics
    cooks = [
        W[i] * resid[i] ** 2 / (k * scale) * hat[i] / (1 - hat[i]) ** 2 if not W[i].is_zero() else arb(0)
        for i in range(n)
    ]
    sw_resid = [W[i].sqrt() * resid[i] for i in used]
    dw = sum(((sw_resid[j] - sw_resid[j - 1]) ** 2 for j in range(1, m)), arb(0)) / ssr
    # central moments (statsmodels' jarque_bera centers; the mean is 0 with an intercept)
    rbar = sum(sw_resid, arb(0)) / m
    mu2 = sum(((r - rbar) ** 2 for r in sw_resid), arb(0)) / m
    mu3 = sum(((r - rbar) ** 3 for r in sw_resid), arb(0)) / m
    mu4 = sum(((r - rbar) ** 4 for r in sw_resid), arb(0)) / m
    skew = mu3 / mu2 ** arb(1.5)
    kurt = mu4 / mu2**2
    jb = arb(m) / 6 * (skew**2 + (kurt - 3) ** 2 / 4)
    jb_p = (-jb / 2).exp()  # chi-square with 2 degrees of freedom
    out = {
        "params": beta,
        "bse": bse,
        "tvalues": tval,
        "pvalues": pval,
        "conf_int_lower": ci_lo,
        "conf_int_upper": ci_hi,
        "fittedvalues": fitted,
        "resid": resid,
        "hat": hat,
        "cooks_distance": cooks,
        "ssr": ssr,
        "ess": ess,
        "centered_tss": tss,
        "rsquared": rsq,
        "rsquared_adj": rsq_adj,
        "scale": scale,
        "llf": llf,
        "aic": aic,
        "bic": bic,
        "durbin_watson": dw,
        "jarque_bera": jb,
        "jarque_bera_pvalue": jb_p,
        "resid_studentized_internal": st_int,
        "resid_studentized_external": st_ext,
        **hc,
        **bp,
    }
    if fval is not None:
        out["fvalue"], out["f_pvalue"] = fval, fp
    meta = {"nobs": m, "df_model": df_model, "df_resid": df_resid, "exact_fit": False, **hc_meta}
    return out, meta


def _all_settled(d):
    for v in d.values():
        for x in v if isinstance(v, list) else [v]:
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
            except ar.NotSettled, ValueError:
                vals = None
            if vals is not None and _all_settled(vals):
                conv = {k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()}
                return conv, meta
        p *= 2
    raise ar.NotSettled("ols")


# ---- generalized linear models (docs/design/regression.md §3.4)


def _link(link, eta):
    """(mu, 1 - mu, mu', mu'') at eta (balls)."""
    if link == "identity":
        return eta, 1 - eta, arb(1), arb(0)
    if link == "log":
        mu = eta.exp()
        return mu, -eta.expm1(), mu, mu
    if link == "logit":
        mu = 1 / (1 + (-eta).exp())
        muc = 1 / (1 + eta.exp())
        d1 = mu * muc
        return mu, muc, d1, d1 * (muc - mu)
    if link == "probit":
        s2 = arb(2).sqrt()
        mu = (-eta / s2).erfc() / 2
        muc = (eta / s2).erfc() / 2
        d1 = (-eta * eta / 2).exp() / (2 * arb.pi()).sqrt()
        return mu, muc, d1, -eta * d1
    if link == "cloglog":
        t = eta.exp()
        muc = (-t).exp()
        mu = -(-t).expm1()
        d1 = t * muc
        return mu, muc, d1, d1 * (1 - t)
    if link == "sqrt":
        return eta * eta, 1 - eta * eta, 2 * eta, arb(2)
    if link == "inverse":
        mu = 1 / eta
        return mu, 1 - mu, -mu * mu, 2 * mu * mu * mu
    raise ValueError(link)


def _var(family, mu, muc):
    """(V(mu), V'(mu))."""
    if family == "binomial":
        return mu * muc, muc - mu
    if family == "poisson":
        return mu, arb(1)
    if family == "gamma":
        return mu * mu, 2 * mu
    return arb(1), arb(0)


def _xlogx_ratio(a, b):
    """a log(a/b), 0 for a = 0."""
    return arb(0) if a.is_zero() else a * (a / b).log()


def _unit_dev(family, y, mu, muc):
    if family == "gaussian":
        return (y - mu) ** 2
    if family == "poisson":
        return 2 * (_xlogx_ratio(y, mu) - (y - mu))
    if family == "gamma":
        return 2 * (-(y / mu).log() + (y - mu) / mu)
    return 2 * (_xlogx_ratio(y, mu) + _xlogx_ratio(1 - y, muc))


def _loglike_obs(family, y, mu, muc, v, n, phi):
    """statsmodels' loglike_obs (before the frequency weight)."""
    if family == "gaussian":
        return -(v * (y - mu) ** 2 / phi + (phi / v).log() + (2 * arb.pi()).log()) / 2
    if family == "poisson":
        ylogmu = arb(0) if y.is_zero() else y * mu.log()
        return v / phi * (ylogmu - mu - (y + 1).lgamma())
    if family == "gamma":
        ws = v / phi
        return ws * (ws * y / mu).log() - ws * y / mu - ws.lgamma() - y.log()
    s = n * y
    fl = n - s
    l = (n + 1).lgamma() - (s + 1).lgamma() - (fl + 1).lgamma()
    if not s.is_zero():
        l += s * mu.log()
    if not fl.is_zero():
        l += fl * muc.log()
    return v * l


def _normal_upper(z):
    """P(Z > z)."""
    return (z / arb(2).sqrt()).erfc() / 2


def _glm_fit(
    y, X, family, link, vw, fw, tr, off, intercept, scale, beta0, alpha, groups, maxlags, null_beta0, full=True
):
    n = len(y)
    A = [([arb(1)] if intercept else []) + [arb(v) for v in r] for r in X]
    k = len(A[0])
    Y = [arb(v) for v in y]
    V = [arb(v) for v in vw]
    F = [arb(v) for v in fw]
    T = [arb(v) for v in tr]
    O = [arb(v) for v in off]
    M = [V[i] * T[i] for i in range(n)]
    PW = [F[i] * M[i] for i in range(n)]
    used = [i for i in range(n) if not PW[i].is_zero()]
    nobs = sum((F[i] for i in used), arb(0))

    def at(beta):
        eta = [sum((A[i][a] * beta[a] for a in range(k)), arb(0)) + O[i] for i in range(n)]
        return eta, [_link(link, e) for e in eta]

    def grad_hess(beta):
        eta, L = at(beta)
        g = [arb(0)] * k
        H = [[arb(0)] * k for _ in range(k)]
        for i in used:
            mu, muc, d1, d2 = L[i]
            Vv, Vp = _var(family, mu, muc)
            r = Y[i] - mu
            gi = PW[i] * r * d1 / Vv
            hi = PW[i] * (-d1 * d1 / Vv + r * (d2 / Vv - d1 * d1 * Vp / (Vv * Vv)))
            for a in range(k):
                g[a] += A[i][a] * gi
                for b in range(k):
                    H[a][b] += A[i][a] * A[i][b] * hi
        return g, H

    # Newton on the midpoints, then Krawczyk's test on a box around the result
    beta = [arb(b) for b in beta0]
    for _ in range(200):
        g, H = grad_hess(beta)
        Hm = arb_mat([[H[a][b].mid() for b in range(k)] for a in range(k)])
        step = Hm.solve(arb_mat([[x.mid()] for x in g]))
        beta = [(beta[a] - step[a, 0]).mid() for a in range(k)]
        if all(abs(step[a, 0].mid()) <= (abs(beta[a]) + arb(2) ** -60) * arb(2) ** (-ctx.prec + 12) for a in range(k)):
            break
    g, H = grad_hess(beta)
    Ym = arb_mat([[H[a][b].mid() for b in range(k)] for a in range(k)]).inv()
    Ym = arb_mat([[Ym[a, b].mid() for b in range(k)] for a in range(k)])
    # the box: half the working precision around the point, relative to the largest
    # coefficient (Newton's point is good to about cond(H) 2^-prec in norm)
    bmax = max((abs(b.mid()) for b in beta), key=lambda x: float(x)) + arb(2) ** -60
    r = [bmax * arb(2) ** (-(ctx.prec // 2)) for a in range(k)]
    box = [arb(beta[a].mid(), r[a].upper()) for a in range(k)]
    _, HX = grad_hess(box)
    YH = Ym * arb_mat(HX)
    Yg = Ym * arb_mat([[x] for x in g])
    K = []
    for a in range(k):
        v = beta[a] - Yg[a, 0]
        for b in range(k):
            v += ((1 if a == b else 0) - YH[a, b]) * (box[b] - beta[b])
        if not (abs(v - beta[a]).upper() < r[a].lower()):
            raise ar.NotSettled("Krawczyk")
        K.append(v)
    beta = K
    eta, L = at(beta)
    mu = [l[0] for l in L]
    muc = [l[1] for l in L]
    if not full:
        return {"_mu": mu, "_muc": muc}, {}
    dev = sum((M[i] * F[i] * _unit_dev(family, Y[i], mu[i], muc[i]) for i in used), arb(0))
    pear = sum((PW[i] * (Y[i] - mu[i]) ** 2 / _var(family, mu[i], muc[i])[0] for i in used), arb(0))
    df_resid = nobs - k
    if scale is None:
        phi = pear / df_resid if family in ("gaussian", "gamma") else arb(1)
    elif scale == "X2":
        phi = pear / df_resid
    elif scale == "dev":
        phi = dev / df_resid
    else:
        phi = arb(scale)
    gid = family == "gaussian" and link == "identity"

    def loglike(mus, mucs):
        if gid:
            ssr = sum((PW[i] * (Y[i] - mus[i]) ** 2 for i in used), arb(0))
            return (
                -nobs / 2 * ((2 * arb.pi()).log() + (ssr / nobs).log() + 1)
                + sum((F[i] * V[i].log() for i in used), arb(0)) / 2
            )
        return sum((F[i] * _loglike_obs(family, Y[i], mus[i], mucs[i], V[i], T[i], phi) for i in used), arb(0))

    llf = loglike(mu, muc)
    # the null model
    if intercept and all(o == 0 for o in off):
        sw = sum((PW[i] for i in used), arb(0))
        m0 = sum((PW[i] * Y[i] for i in used), arb(0)) / sw
        mu0, muc0 = [m0] * n, [1 - m0] * n
    elif intercept:
        sub, _ = _glm_fit(
            y,
            [[] for _ in range(n)],
            family,
            link,
            vw,
            fw,
            tr,
            off,
            True,
            scale,
            [null_beta0],
            alpha,
            None,
            None,
            null_beta0,
            full=False,
        )
        mu0, muc0 = sub["_mu"], sub["_muc"]
    else:
        L0 = [_link(link, O[i]) for i in range(n)]
        mu0, muc0 = [l[0] for l in L0], [l[1] for l in L0]
    dev0 = sum((M[i] * F[i] * _unit_dev(family, Y[i], mu0[i], muc0[i]) for i in used), arb(0))
    llnull = loglike(mu0, muc0)
    # covariances: the expected information (nonrobust), the observed one as robust bread
    W = []
    for i in range(n):
        m_, mc_, d1, _ = L[i]
        W.append(PW[i] * d1 * d1 / _var(family, m_, mc_)[0] if i in used else arb(0))
    I = arb_mat([[sum((W[i] * A[i][a] * A[i][b] for i in used), arb(0)) for b in range(k)] for a in range(k)])
    Iinv = I.inv()
    cov = [[phi * Iinv[a, b] for b in range(k)] for a in range(k)]
    bse = [cov[a][a].sqrt() for a in range(k)]
    zval = [beta[a] / bse[a] for a in range(k)]
    fixed = (family in ("binomial", "poisson") and scale is None) or (scale is not None and not isinstance(scale, str))
    if fixed:
        pval = [2 * _normal_upper(abs(z)) for z in zval]
    else:
        pval = [2 * _t_upper(abs(z), df_resid) for z in zval]
    _, Hb = grad_hess(beta)
    Binv = (-arb_mat(Hb)).inv()
    u = []  # per-unit scores (no frequency weight)
    for i in used:
        m_, mc_, d1, _ = L[i]
        sc = M[i] * (Y[i] - m_) * d1 / _var(family, m_, mc_)[0]
        u.append([A[i][a] * sc for a in range(k)])
    fu = [F[i] for i in used]
    hat = [W[i] * sum((A[i][a] * Iinv[a, b] * A[i][b] for a in range(k) for b in range(k)), arb(0)) for i in used]

    def sand(rows, factor=1):
        S = arb_mat([[sum((r[a] * r[b] for r in rows), arb(0)) for b in range(k)] for a in range(k)])
        C = Binv * S * Binv
        return [(C[a, a] * factor).sqrt() for a in range(k)]

    sq = [[x * fu[j].sqrt() for x in u[j]] for j in range(len(used))]
    out = {
        "params": beta,
        "bse": bse,
        "tvalues": zval,
        "pvalues": pval,
        "deviance": dev,
        "pearson_chi2": pear,
        "scale": phi,
        "llf": llf,
        "null_deviance": dev0,
        "llnull": llnull,
        "aic": -2 * llf + 2 * k,
        "bic": -2 * llf + nobs.log() * k,
        "fittedvalues": mu,
        "_mu": mu,
        "_muc": muc,
    }
    out["bse_HC0"] = sand(sq)
    out["bse_HC1"] = sand(sq, nobs / (nobs - k))
    # a row of frequency f stands for f copies, each with the leverage h / f (decision 2026-11-19)
    hf = [hat[j] / fu[j] for j in range(len(used))]
    out["bse_HC2"] = sand([[x / (1 - hf[j]).sqrt() for x in sq[j]] for j in range(len(used))])
    out["bse_HC3"] = sand([[x / (1 - hf[j]) for x in sq[j]] for j in range(len(used))])
    # the default lags from nobs (the sum of the frequency weights)
    Lg = maxlags if maxlags is not None else math.floor(4 * (float(nobs.mid()) / 100) ** (2 / 9))
    S = [[sum((s[a] * s[b] for s in sq), arb(0)) for b in range(k)] for a in range(k)]
    for lag in range(1, min(Lg, len(used) - 1) + 1):
        wl = 1 - arb(lag) / (Lg + 1)
        for a in range(k):
            for b in range(k):
                S[a][b] += wl * sum(
                    (sq[t][a] * sq[t - lag][b] + sq[t - lag][a] * sq[t][b] for t in range(lag, len(used))), arb(0)
                )
    C = Binv * arb_mat(S) * Binv
    out["bse_HAC"] = [C[a, a].sqrt() for a in range(k)]
    meta = {"nobs": float(nobs.mid()), "df_resid": float(df_resid.mid()), "maxlags": Lg, "rank": k}
    if groups is not None:
        labels = [groups[i] for i in used]
        order = list(dict.fromkeys(labels))
        G = len(order)
        sums = {gg: [arb(0)] * k for gg in order}
        for j, gg in enumerate(labels):
            sums[gg] = [sums[gg][a] + fu[j] * u[j][a] for a in range(k)]
        S = arb_mat([[sum((sums[gg][a] * sums[gg][b] for gg in order), arb(0)) for b in range(k)] for a in range(k)])
        C = Binv * S * Binv * (arb(G) / (G - 1) * (nobs - 1) / (nobs - k))
        bc = [C[a, a].sqrt() for a in range(k)]
        out["bse_cluster"] = bc
        out["pvalues_cluster"] = [2 * _t_upper(abs(beta[a] / bc[a]), arb(G - 1)) for a in range(k)]
        meta["n_groups"] = G
    out["resid_pearson"] = [(M[i]).sqrt() * (Y[i] - mu[i]) / _var(family, mu[i], muc[i])[0].sqrt() for i in range(n)]
    return out, meta


def glm(
    y,
    X,
    *,
    family,
    link,
    var_weights=None,
    freq_weights=None,
    trials=None,
    offset=None,
    intercept=True,
    scale=None,
    beta0,
    alpha=0.05,
    groups=None,
    maxlags=None,
    null_beta0=0.0,
):
    """References for the GLM fit of y on X (rows without the constant column; intercept puts
    it first) with statsmodels' definitions: the MLE (Newton on the score equations from the
    double start beta0, then certified by Krawczyk's test: the interval Newton operator maps a
    box around it into itself, so the box holds the unique root), its standard errors
    (expected information x scale), z or t values and p-values (normal for a fixed scale,
    t(df_resid) for an estimated one), deviance, Pearson chi2, scale, log-likelihood (the
    gaussian identity's concentrated), null deviance and log-likelihood (the intercept-only
    fit, by the same solver when there is an offset; null_beta0 starts it), AIC, BIC (llf
    based), fitted values, Pearson residuals, and the robust standard errors HC0-HC3, HAC and
    cluster with the observed information as bread. Rows with a zero weight are out of the
    fit. Returns (values as (hi, lo) or lists of them, meta)."""
    n = len(y)
    vw = var_weights if var_weights is not None else [1.0] * n
    fw = freq_weights if freq_weights is not None else [1.0] * n
    tr = trials if trials is not None else [1.0] * n
    off = offset if offset is not None else [0.0] * n
    p = 256
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals, meta = _glm_fit(
                    y, X, family, link, vw, fw, tr, off, intercept, scale, beta0, alpha, groups, maxlags, null_beta0
                )
                vals = {key: v for key, v in vals.items() if not key.startswith("_")}
            except ar.NotSettled, ValueError, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                conv = {k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()}
                return conv, meta
        p *= 2
    raise ar.NotSettled("glm")


def _penalized_fit(y, X, w, alpha, l1_ratio, family, intercept, standardize, beta0):
    n, p = len(y), len(X[0]) if X else 0
    used = [i for i in range(n) if w[i] > 0]
    W = {i: arb(w[i]) for i in used}
    Y = {i: arb(y[i]) for i in used}
    Xa = {i: [arb(v) for v in X[i]] for i in used}
    sw = sum((W[i] for i in used), arb(0))
    # columns out of the fit: constant ones (zero ones without the intercept)
    cols = []
    for j in range(p):
        vals = {X[i][j] for i in used}
        if len(vals) == 1 and (intercept or 0.0 in vals):
            continue
        cols.append(j)
    xbar = {j: (sum((W[i] * Xa[i][j] for i in used), arb(0)) / sw if intercept else arb(0)) for j in cols}
    if standardize:
        sc = {j: (sum((W[i] * (Xa[i][j] - xbar[j]) ** 2 for i in used), arb(0)) / sw).sqrt() for j in cols}
    else:
        sc = {j: arb(1) for j in cols}
    l1 = arb(alpha) * arb(l1_ratio)
    l2 = arb(alpha) * (1 - arb(l1_ratio))
    off = 1 if intercept else 0
    act = [j for j in cols if beta0[off + j] != 0.0]
    sg = {j: (1 if beta0[off + j] > 0 else -1) for j in act}
    a = len(act)
    k = off + a

    def design(i):
        return ([arb(1)] if intercept else []) + [Xa[i][j] for j in act]

    D = {i: design(i) for i in used}

    # the penalty's gradient (beyond the data term) of each unknown: 0 for the intercept
    def pen_grad(theta):
        return [arb(0)] * off + [l2 * sc[j] ** 2 * theta[off + q] + l1 * sc[j] * sg[j] for q, j in enumerate(act)]

    def pen_hess_diag():
        return [arb(0)] * off + [l2 * sc[j] ** 2 for j in act]

    if family == "gaussian":
        M = [[sum((W[i] * D[i][u] * D[i][v] for i in used), arb(0)) / sw for v in range(k)] for u in range(k)]
        for u, h in enumerate(pen_hess_diag()):
            M[u][u] += h
        rhs = [sum((W[i] * D[i][u] * Y[i] for i in used), arb(0)) / sw for u in range(k)]
        for q, j in enumerate(act):
            rhs[off + q] -= l1 * sc[j] * sg[j]
        theta = [v for v in (arb_mat(M).solve(arb_mat([[r] for r in rhs])).entries())] if k else []

        def mean(eta):
            return eta
    else:
        link = "logit" if family == "binomial" else "log"

        def mean(eta):
            return _link(link, eta)[0]

        def grad_hess(theta):
            g = [arb(0)] * k
            H = [[arb(0)] * k for _ in range(k)]
            for i in used:
                eta = sum((D[i][u] * theta[u] for u in range(k)), arb(0))
                mu, muc, d1, _ = _link(link, eta)
                r = W[i] * (Y[i] - mu) / sw
                h = W[i] * d1 / sw
                for u in range(k):
                    g[u] += D[i][u] * r
                    for v in range(k):
                        H[u][v] += D[i][u] * D[i][v] * h
            pg = pen_grad(theta)
            for u in range(k):
                g[u] -= pg[u]
            for u, hd in enumerate(pen_hess_diag()):
                H[u][u] += hd
            return g, H  # g: minus the objective's gradient, H: its Hessian

        theta = [arb(beta0[0])] * off + [arb(beta0[off + j]) for j in act]
        for _ in range(200):
            g, H = grad_hess(theta)
            step = arb_mat([[H[u][v].mid() for v in range(k)] for u in range(k)]).solve(arb_mat([[x.mid()] for x in g]))
            theta = [(theta[u] + step[u, 0]).mid() for u in range(k)]
            if all(
                abs(step[u, 0].mid()) <= (abs(theta[u]) + arb(2) ** -60) * arb(2) ** (-ctx.prec + 12) for u in range(k)
            ):
                break
        # Krawczyk's test on a box around the point (as glm): the unique stationary point
        g, H = grad_hess(theta)
        Ym = arb_mat([[H[u][v].mid() for v in range(k)] for u in range(k)]).inv()
        Ym = arb_mat([[Ym[u, v].mid() for v in range(k)] for u in range(k)])
        tmax = max((abs(t.mid()) for t in theta), key=lambda x: float(x)) + arb(2) ** -60
        rad = tmax * arb(2) ** (-(ctx.prec // 2))
        box = [arb(theta[u].mid(), rad.upper()) for u in range(k)]
        _, HX = grad_hess(box)
        YH = Ym * arb_mat(HX)
        Yg = Ym * arb_mat([[x] for x in g])
        K = []
        for u in range(k):
            v = theta[u] + Yg[u, 0]
            for q in range(k):
                v += ((1 if u == q else 0) - YH[u, q]) * (box[q] - theta[q])
            if not (abs(v - theta[u]).upper() < rad.lower()):
                raise ar.NotSettled("Krawczyk")
            K.append(v)
        theta = K
    # the KKT certificate: the active coefficients keep their signs, every inactive column's
    # |x_j^T W r| / sw is strictly below l1 s_j
    for q, j in enumerate(act):
        b = theta[off + q]
        if not ((b > 0) if sg[j] > 0 else (b < 0)):
            raise ValueError(f"the sign of column {j} does not hold")
    eta = {i: sum((D[i][u] * theta[u] for u in range(k)), arb(0)) for i in used}
    mu = {i: mean(eta[i]) for i in used}
    for j in cols:
        if j in sg:
            continue
        gj = sum((W[i] * Xa[i][j] * (Y[i] - mu[i]) for i in used), arb(0)) / sw
        if not (abs(gj) < l1 * sc[j]):
            raise ValueError(f"column {j} violates the KKT conditions of the inactive set")
    params = [arb(0)] * (off + p)
    if intercept:
        params[0] = theta[0]
    for q, j in enumerate(act):
        params[off + j] = theta[off + q]
    out = {"params": params}
    if family == "gaussian":
        out["ssr"] = sum((W[i] * (Y[i] - mu[i]) ** 2 for i in used), arb(0))
    else:
        lk = "logit" if family == "binomial" else "log"
        out["deviance"] = sum((W[i] * _unit_dev(family, Y[i], mu[i], _link(lk, eta[i])[1]) for i in used), arb(0))
    return out


def penalized(y, X, *, alpha, l1_ratio, beta0, family="gaussian", weights=None, intercept=True, standardize=False):
    """The certified solution of the penalized regression (G45; the GLMs' deviance / (2 sw) +
    penalty, G47) whose active set and signs are those of the double solution beta0 (params,
    the intercept first): the stationarity conditions on that set are solved exactly (a
    linear system; Newton and Krawczyk's test for the GLMs), and the result is certified as
    the minimizer by its signs and the strict KKT inequalities of every inactive column
    (ValueError when they fail: then beta0's active set is not the solution's). Constant
    columns (zero ones without the intercept) are out of the fit. Returns params and ssr (or
    the deviance) as (hi, lo)."""
    n = len(y)
    w = weights if weights is not None else [1.0] * n
    p = 256
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals = _penalized_fit(y, X, w, alpha, l1_ratio, family, intercept, standardize, beta0)
            except ar.NotSettled, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                return {k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()}
        p *= 2
    raise ar.NotSettled("penalized")


# ---- robust regression (docs/design/regression.md §3.6)

_MAD_C = 0.6744897501960817  # the double the kernel divides |r| by (statsmodels' Gaussian.ppf(3/4))
_HUBER_D = 2.5  # statsmodels' HuberScale() d
_S_CHI = 1.54764  # robustbase's tuning.chi (bisquare, 50% breakdown)
_S_B = 0.5


def _region(norm, c, z):
    """The piece of the norm a double z falls in: Huber 0 (|z| <= t) / 1; bisquare 0 (|z| <=
    c) / 1; Hampel 0 (|z| <= a), 1 (<= b), 2 (<= c), 3."""
    az = abs(z)
    if norm == "huber":
        return 0 if az <= c[0] else 1
    if norm == "bisquare":
        return 0 if az <= c[0] else 1
    a, b, cc = c
    return 0 if az <= a else (1 if az <= b else (2 if az <= cc else 3))


def _in_region(norm, c, reg, z):
    """The ball z lies inside piece reg of the norm (strictly away from its ends)."""
    az = abs(z)
    if norm in ("huber", "bisquare"):
        t = arb(c[0])
        return bool(az < t) if reg == 0 else bool(az > t)
    a, b, cc = (arb(v) for v in c)
    return [bool(az < a), bool(az > a) and bool(az < b), bool(az > b) and bool(az < cc), bool(az > cc)][reg]


def _psi(norm, c, reg, z):
    """psi(z) and psi'(z) on piece reg (no branching on the value: balls stay one formula)."""
    if norm == "huber":
        t = arb(c[0])
        return (z, arb(1)) if reg == 0 else ((t if z > 0 else -t), arb(0))
    if norm == "bisquare":
        if reg == 1:
            return arb(0), arb(0)
        u = z / arb(c[0])
        t = 1 - u * u
        return z * t * t, t * (t - 4 * u * u)
    a, b, cc = (arb(v) for v in c)
    sg = arb(1) if z > 0 else arb(-1)  # certain off piece 0
    if reg == 0:
        return z, arb(1)
    if reg == 1:
        return a * sg, arb(0)
    if reg == 2:
        return a * (cc - abs(z)) / (cc - b) * sg, -a / (cc - b)
    return arb(0), arb(0)


def _rlm_fit(y, X, intercept, norm, c, scale_est, beta0, s_fixed, init_params, df_rank):
    n = len(y)
    A = [([arb(1)] if intercept else []) + [arb(v) for v in r] for r in X]
    k = len(A[0])
    Y = [arb(v) for v in y]
    b0 = [arb(v) for v in beta0]

    def resid(beta):
        return [Y[i] - sum((A[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(n)]

    r0 = [float(v.mid()) for v in resid(b0)]
    # the double solution's pattern: the scale rule's rows and each row's piece of the norm
    if scale_est == "mad":
        order = sorted(range(n), key=lambda i: abs(r0[i]))
        mids = [order[(n - 1) // 2]] if n % 2 else [order[n // 2 - 1], order[n // 2]]
        sg = {i: (1 if r0[i] > 0 else -1) for i in mids}
        s_d = (sum(abs(r0[i]) for i in mids) / len(mids)) / _MAD_C
    elif scale_est == "huber":
        d = arb(_HUBER_D)
        E = (
            d * d
            + (1 - d * d) * (arb(1) - (d / arb(2).sqrt()).erfc() / 2)
            - arb(0.5)
            - d / (2 * arb.pi()).sqrt() * (-d * d / 2).exp()
        )
        T = (n - df_rank) * E
        # the split |r| < d s from the double scale (the kernel's equation)
        Tm = float(T.mid())
        q = sorted(v * v for v in r0)
        s_d = None
        for j in range(n, -1, -1):  # j rows inside
            den = 2 * Tm - (n - j) * _HUBER_D**2
            if den <= 0:
                continue
            s2 = sum(q[:j]) / den
            if (j == 0 or q[j - 1] < _HUBER_D**2 * s2) and (j == n or q[j] >= _HUBER_D**2 * s2):
                s_d = s2**0.5
                break
        inside = [i for i in range(n) if r0[i] ** 2 < (_HUBER_D * s_d) ** 2]
        outside = [i for i in range(n) if i not in set(inside)]
    else:
        s_d = s_fixed
    reg = [_region(norm, c, r0[i] / s_d) for i in range(n)]

    def scale(r):
        if scale_est == "mad":
            return sum((sg[i] * r[i] for i in mids), arb(0)) / (len(mids) * arb(_MAD_C))
        if scale_est == "huber":
            Q = sum((r[i] * r[i] for i in inside), arb(0))
            return (Q / (2 * T - len(outside) * arb(_HUBER_D) ** 2)).sqrt()
        return arb(s_fixed)

    def grad_s(r, s):
        if scale_est == "mad":
            return [-sum((sg[i] * A[i][a] for i in mids), arb(0)) / (len(mids) * arb(_MAD_C)) for a in range(k)]
        if scale_est == "huber":
            den = 2 * T - len(outside) * arb(_HUBER_D) ** 2
            return [-2 * sum((r[i] * A[i][a] for i in inside), arb(0)) / (2 * s * den) for a in range(k)]
        return [arb(0)] * k

    def FJ(beta):
        r = resid(beta)
        s = scale(r)
        gs = grad_s(r, s)
        F = [arb(0)] * k
        J = [[arb(0)] * k for _ in range(k)]
        for i in range(n):
            z = r[i] / s
            ps, dp = _psi(norm, c, reg[i], z)
            for a in range(k):
                F[a] += ps * A[i][a]
            if dp == 0:
                continue
            dz = [(-A[i][b] - z * gs[b]) / s for b in range(k)]
            for a in range(k):
                for b in range(k):
                    J[a][b] += dp * A[i][a] * dz[b]
        return F, J, r, s

    beta = [arb(v) for v in beta0]
    for _ in range(100):
        F, J, _, _ = FJ(beta)
        Jm = arb_mat([[J[a][b].mid() for b in range(k)] for a in range(k)])
        step = Jm.solve(arb_mat([[x.mid()] for x in F]))
        beta = [(beta[a] - step[a, 0]).mid() for a in range(k)]
        if all(abs(step[a, 0].mid()) <= (abs(beta[a]) + arb(2) ** -60) * arb(2) ** (-ctx.prec + 12) for a in range(k)):
            break
    F, J, _, _ = FJ(beta)
    Ym = arb_mat([[J[a][b].mid() for b in range(k)] for a in range(k)]).inv()
    Ym = arb_mat([[Ym[a, b].mid() for b in range(k)] for a in range(k)])
    bmax = max((abs(b.mid()) for b in beta), key=lambda x: float(x)) + arb(2) ** -60
    rr = bmax * arb(2) ** (-(ctx.prec // 2))
    box = [arb(beta[a].mid(), rr.upper()) for a in range(k)]
    _, JX, rX, sX = FJ(box)
    # the pattern holds on the whole box: every row in its piece, the scale's rows the middle ones
    for i in range(n):
        if not _in_region(norm, c, reg[i], rX[i] / sX):
            raise ar.NotSettled("piece")
    if scale_est == "mad":
        lo = min((abs(rX[i]) for i in mids), key=lambda v: float(v.lower()))
        hi = max((abs(rX[i]) for i in mids), key=lambda v: float(v.upper()))
        below = sum(1 for i in range(n) if i not in mids and bool(abs(rX[i]) < lo))
        above = sum(1 for i in range(n) if i not in mids and bool(abs(rX[i]) > hi))
        if below + above != n - len(mids) or below != (n - len(mids)) // 2:
            raise ar.NotSettled("median")
        if not all(bool(sg[i] * rX[i] > 0) for i in mids):
            raise ar.NotSettled("median sign")
    if scale_est == "huber":
        if not all(bool(abs(rX[i]) < arb(_HUBER_D) * sX) for i in inside):
            raise ar.NotSettled("huber split")
        if not all(bool(abs(rX[i]) > arb(_HUBER_D) * sX) for i in outside):
            raise ar.NotSettled("huber split")
    YJ = Ym * arb_mat(JX)
    YF = Ym * arb_mat([[x] for x in F])
    K = []
    for a in range(k):
        v = beta[a] - YF[a, 0]
        for b in range(k):
            v += ((1 if a == b else 0) - YJ[a, b]) * (box[b] - beta[b])
        if not (abs(v - beta[a]).upper() < rr.lower()):
            raise ar.NotSettled("Krawczyk")
        K.append(v)
    beta = K
    r = resid(beta)
    s = scale(r)
    z = [r[i] / s for i in range(n)]
    pd = [_psi(norm, c, reg[i], z[i]) for i in range(n)]
    out = {"params": beta, "scale": s, "weights": [pd[i][0] / z[i] if z[i] != 0 else arb(1) for i in range(n)]}
    if init_params is None:
        # Huber's H1
        m = sum((p[1] for p in pd), arb(0)) / n
        var = sum(((p[1] - m) ** 2 for p in pd), arb(0)) / n
        ss = sum((p[0] ** 2 for p in pd), arb(0))
        kk = 1 + arb(df_rank) / n * var / (m * m)
        f = kk * kk * (ss / (n - df_rank) * s * s) / (m * m)
        G = arb_mat([[sum((A[i][a] * A[i][b] for i in range(n)), arb(0)) for b in range(k)] for a in range(k)]).inv()
        cov = [[f * G[a, b] for b in range(k)] for a in range(k)]
    else:
        # robustbase's .vcov.avar1 (lmrob's default for MM), s the S scale
        bS = [arb(v) for v in init_params]
        rS = resid(bS)
        cchi = arb(_S_CHI)

        def chi(u):
            if bool(abs(u) < cchi):
                v = u / cchi
                t = 1 - v * v
                return 1 - t * t * t, 6 * u * t * t / (cchi * cchi)
            if bool(abs(u) > cchi):
                return arb(1), arb(0)
            raise ar.NotSettled("chi piece")

        z0 = [rS[i] / s for i in range(n)]
        ch = [chi(u) for u in z0]
        G1 = arb_mat(
            [[sum((pd[i][1] * A[i][a] * A[i][b] for i in range(n)), arb(0)) for b in range(k)] for a in range(k)]
        )
        G2 = arb_mat(
            [[sum((pd[i][0] ** 2 * A[i][a] * A[i][b] for i in range(n)), arb(0)) for b in range(k)] for a in range(k)]
        )
        Am = G1.inv() * s
        v1 = arb_mat([[sum((A[i][a] * pd[i][1] * z[i] for i in range(n)), arb(0))] for a in range(k)])
        v2 = arb_mat([[sum((A[i][a] * pd[i][0] * ch[i][0] for i in range(n)), arb(0))] for a in range(k)])
        mw0 = sum((ch[i][1] * z0[i] for i in range(n)), arb(0)) / n
        mu4 = sum((ch[i][0] ** 2 - arb(_S_B) ** 2 for i in range(n)), arb(0)) / n
        av = Am * v1 * (1 / mw0)
        Av2 = Am * v2
        V = Am * G2 * Am
        cov = [
            [
                V[a, b] + (-(av[a, 0] * Av2[b, 0]) - Av2[a, 0] * av[b, 0] + mu4 * av[a, 0] * av[b, 0]) / n
                for b in range(k)
            ]
            for a in range(k)
        ]
    out["bse"] = [cov[a][a].sqrt() for a in range(k)]
    return out


def rlm(y, X, *, beta0, norm, c, scale_est="mad", intercept=True, scale=None, init_params=None):
    """The certified M estimate near the double solution beta0 (params, the intercept first):
    the estimating equations sum psi(r_i/s) x_i = 0 with the scale rule of the kernel --
    'mad' (s = median |r_i| / 0.6744897501960817, the double the kernel uses), 'huber'
    (Huber's proposal 2 with d = 2.5: sum chi(r_i/s) = (n - k) E chi(Z), E exact) or a fixed
    s (`scale`, MM's M step, with `init_params` the S estimate for the covariance) -- solved
    by Newton on the pieces of psi and of the median that the double solution has, and
    certified by Krawczyk's test on a box on which those pieces are certain. c: (t,) Huber,
    (c,) bisquare, (a, b, c) Hampel. Returns params, scale, weights and bse (H1 for M,
    robustbase's .vcov.avar1 with init_params) as (hi, lo)."""
    k = len(beta0)
    p = 256
    while p <= 2048:
        with ar.precision(p):
            try:
                vals = _rlm_fit(
                    y, X, intercept, norm, c, scale_est if scale is None else "fixed", beta0, scale, init_params, k
                )
            except ar.NotSettled, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                return {
                    kk: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for kk, v in vals.items()
                }
        p *= 2
    raise ar.NotSettled("rlm")


def _s_fit(y, X, intercept, beta0, s0):
    n = len(y)
    A = [([arb(1)] if intercept else []) + [arb(v) for v in r] for r in X]
    k = len(A[0])
    Y = [arb(v) for v in y]
    c = arb(_S_CHI)
    T = (n - k) * arb(_S_B)

    def resid(beta):
        return [Y[i] - sum((A[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(n)]

    r0 = [float(v.mid()) for v in resid([arb(v) for v in beta0])]
    inside = [abs(r0[i] / s0) < _S_CHI for i in range(n)]  # each row's piece, from the double solution

    def G(beta, s):
        # (sum psi(z_i) x_i, sum rho~(z_i) - T) and its Jacobian in (beta, s), z = r / s,
        # psi(z) = z t^2 (proportional to rho~'), rho~(z) = 1 - t^3, t = 1 - (z/c)^2 inside |z| < c
        r = resid(beta)
        F = [arb(0)] * (k + 1)
        J = [[arb(0)] * (k + 1) for _ in range(k + 1)]
        F[k] = -T
        for i in range(n):
            if not inside[i]:
                F[k] += 1
                continue
            z = r[i] / s
            v = z / c
            t = 1 - v * v
            ps, dps = z * t * t, t * (t - 4 * v * v)
            rh, drh = 1 - t * t * t, 6 * z * t * t / (c * c)
            dz = [-A[i][b] / s for b in range(k)] + [-z / s]
            for a in range(k):
                F[a] += ps * A[i][a]
                for b in range(k + 1):
                    J[a][b] += dps * A[i][a] * dz[b]
            F[k] += rh
            for b in range(k + 1):
                J[k][b] += drh * dz[b]
        return F, J, r

    u = [arb(v) for v in beta0] + [arb(s0)]
    for _ in range(100):
        F, J, _ = G(u[:k], u[k])
        Jm = arb_mat([[J[a][b].mid() for b in range(k + 1)] for a in range(k + 1)])
        step = Jm.solve(arb_mat([[x.mid()] for x in F]))
        u = [(u[a] - step[a, 0]).mid() for a in range(k + 1)]
        if all(abs(step[a, 0].mid()) <= (abs(u[a]) + arb(2) ** -60) * arb(2) ** (-ctx.prec + 12) for a in range(k + 1)):
            break
    F, J, _ = G(u[:k], u[k])
    Ym = arb_mat([[J[a][b].mid() for b in range(k + 1)] for a in range(k + 1)]).inv()
    Ym = arb_mat([[Ym[a, b].mid() for b in range(k + 1)] for a in range(k + 1)])
    umax = max((abs(x.mid()) for x in u), key=lambda x: float(x)) + arb(2) ** -60
    rr = umax * arb(2) ** (-(ctx.prec // 2))
    box = [arb(u[a].mid(), rr.upper()) for a in range(k + 1)]
    _, JX, rX = G(box[:k], box[k])
    # every row on its piece over the whole box
    for i in range(n):
        z = rX[i] / box[k]
        if inside[i] and not bool(abs(z) < c):
            raise ar.NotSettled("S piece")
        if not inside[i] and not bool(abs(z) > c):
            raise ar.NotSettled("S piece")
    YJ = Ym * arb_mat(JX)
    YF = Ym * arb_mat([[x] for x in F])
    K = []
    for a in range(k + 1):
        v = u[a] - YF[a, 0]
        for b in range(k + 1):
            v += ((1 if a == b else 0) - YJ[a, b]) * (box[b] - u[b])
        if not (abs(v - u[a]).upper() < rr.lower()):
            raise ar.NotSettled("Krawczyk")
        K.append(v)
    return {"params": K[:k], "scale": K[k]}


def s_estimate(y, X, *, beta0, scale0, intercept=True):
    """The certified S estimate (robustbase's lmrob S step, sd.regression's MM start, G53, G82)
    near the double solution (beta0, scale0): the fixed point of the S iteration, i.e. the
    root of sum psi(r_i/s) x_i = 0 together with the M-scale equation sum rho~(r_i/s) =
    (n - k)/2 (rho~ the normalized bisquare with c = 1.54764, psi proportional to rho~'),
    solved by Newton in (beta, s) on the pieces (|z| < c or not) of the double solution and
    certified by Krawczyk's test on a box on which those pieces are certain. Only the local
    root: which local minimum the subsampling reached is the double solution's. Returns
    params and scale as (hi, lo)."""
    p = 256
    while p <= 2048:
        with ar.precision(p):
            try:
                vals = _s_fit(y, X, intercept, beta0, scale0)
            except ar.NotSettled, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                return {
                    kk: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for kk, v in vals.items()
                }
        p *= 2
    raise ar.NotSettled("s_estimate")


# ---- quantile regression (G60-G63): the exact vertex of a basis and its certificate


def _q(v):
    """A double as an exact rational."""
    from flint import fmpq

    a, b = float(v).as_integer_ratio()
    return fmpq(a, b)


def _q_dd(v):
    with ar.precision(400):
        return ar.to_dd(arb(v))


def _exact_lp_max(A, b, c):
    """max c^T v s.t. A v = b, v >= 0 by the two-phase simplex in exact rationals (Bland's
    rule). A: list of rows (fmpq), b, c: lists. Returns the optimal value, or None when
    infeasible (the feasible sets here are bounded)."""
    from flint import fmpq

    m, nv = len(A), len(c)
    rows = []
    for i in range(m):
        sgn = -1 if b[i] < 0 else 1
        rows.append([sgn * a for a in A[i]] + [fmpq(1) if j == i else fmpq(0) for j in range(m)] + [sgn * b[i]])
    basis = [nv + i for i in range(m)]
    width = nv + m

    def run(cost, allowed):
        # tableau minimization of cost^T v (reduced costs recomputed from the basis)
        while True:
            red = [cost[j] - sum(cost[basis[i]] * rows[i][j] for i in range(m)) for j in range(width)]
            enter = next((j for j in range(width) if allowed(j) and j not in basis and red[j] < 0), None)
            if enter is None:
                return
            best = None
            for i in range(m):
                a = rows[i][enter]
                if a > 0:
                    t = rows[i][-1] / a
                    if best is None or t < best[0] or (t == best[0] and basis[i] < basis[best[1]]):
                        best = (t, i)
            if best is None:
                raise ValueError("unbounded")
            r = best[1]
            piv = rows[r][enter]
            rows[r] = [v / piv for v in rows[r]]
            for i in range(m):
                if i != r and rows[i][enter] != 0:
                    f = rows[i][enter]
                    rows[i] = [vi - f * vr for vi, vr in zip(rows[i], rows[r])]
            basis[r] = enter

    run([fmpq(0)] * nv + [fmpq(1)] * m, lambda j: True)
    if sum(rows[i][-1] for i in range(m) if basis[i] >= nv) != 0:
        return None
    # drive degenerate artificials out where possible, then optimize with them barred
    for i in range(m):
        if basis[i] >= nv:
            j = next((j for j in range(nv) if rows[i][j] != 0 and j not in basis), None)
            if j is not None:
                piv = rows[i][j]
                rows[i] = [v / piv for v in rows[i]]
                for r2 in range(m):
                    if r2 != i and rows[r2][j] != 0:
                        f = rows[r2][j]
                        rows[r2] = [vi - f * vr for vi, vr in zip(rows[r2], rows[i])]
                basis[i] = j
    run([-v for v in c] + [fmpq(0)] * m, lambda j: j < nv)
    val = fmpq(0)
    for i in range(m):
        if basis[i] < nv:
            val += c[basis[i]] * rows[i][-1]
    return val


def quantreg(y, X, basis, q, *, intercept=True):
    """The exact vertex of quantile regression's basis rows and its optimality certificate,
    in rationals (the doubles are exact rationals; no rounding anywhere).

    beta solves X_B beta = y_B; r = y - X beta exactly. The vertex is optimal iff some
    lambda with lambda_i = psi_q(r_i) off the zero residuals, lambda_i in [q - 1, q] on them
    and X^T lambda = 0 exists; it is the unique minimizer iff such a lambda exists strictly
    inside (q - 1, q) on the zero residuals (relative interiors commute with linear maps). On
    a nondegenerate vertex (zero residuals only on the basis) lambda_B is solved directly;
    otherwise an exact linear program gives the largest margin eps (rows with equal x merged:
    their multipliers' sum ranges over the box scaled by the count). Returns params and the
    objective sum rho_q(r_i) as double-doubles, optimal, unique, the number of zero residuals
    and the margin (min distance of lambda to the box's ends; a lower bound with ties)."""
    from flint import fmpq, fmpq_mat

    n = len(y)
    rows = [
        ([fmpq(1)] if intercept else []) + [_q(v) for v in (X[i] if hasattr(X[i], "__len__") else [X[i]])]
        for i in range(n)
    ]
    k = len(rows[0])
    yq = [_q(v) for v in y]
    tau = _q(q)
    B = list(basis)
    XB = fmpq_mat(k, k, [rows[i][c] for i in B for c in range(k)])
    beta = XB.solve(fmpq_mat(k, 1, [yq[i] for i in B]))
    beta = [beta[c, 0] for c in range(k)]
    r = [yq[i] - sum(rows[i][c] * beta[c] for c in range(k)) for i in range(n)]
    assert all(r[i] == 0 for i in B)
    zero = [i for i in range(n) if r[i] == 0]
    g = [fmpq(0)] * k
    obj = fmpq(0)
    for i in range(n):
        if r[i] == 0:
            continue
        psi = tau if r[i] > 0 else tau - 1
        obj += psi * r[i]
        for c in range(k):
            g[c] += psi * rows[i][c]
    h = [-v for v in g]
    if len(zero) == k:
        lam = XB.transpose().solve(fmpq_mat(k, 1, h))
        lam = [lam[a, 0] for a in range(k)]
        margin = min(min(v - (tau - 1), tau - v) for v in lam)
        optimal = margin >= 0
        unique = margin > 0
    else:
        groups = {}
        for i in zero:
            groups.setdefault(tuple(rows[i]), []).append(i)
        G = list(groups)
        mcount = [fmpq(len(groups[gk])) for gk in G]
        m = len(G)
        # variables: mu_g (m), eps, s_g (m), t_g (m); lambda_g = m_g (q - 1) + mu_g
        nv = 3 * m + 1
        A, bvec = [], []
        for c in range(k):
            row = [G[j][c] for j in range(m)] + [fmpq(0)] * (2 * m + 1)
            A.append(row)
            bvec.append(h[c] - sum(mcount[j] * (tau - 1) * G[j][c] for j in range(m)))
        for j in range(m):
            row = [fmpq(0)] * nv
            row[j], row[m], row[m + 1 + j] = fmpq(1), -mcount[j], fmpq(-1)
            A.append(row)
            bvec.append(fmpq(0))
            row = [fmpq(0)] * nv
            row[j], row[m], row[2 * m + 1 + j] = fmpq(1), mcount[j], fmpq(1)
            A.append(row)
            bvec.append(mcount[j])
        cvec = [fmpq(0)] * nv
        cvec[m] = fmpq(1)
        margin = _exact_lp_max(A, bvec, cvec)
        optimal = margin is not None
        unique = optimal and margin > 0
    return dict(
        params=[_q_dd(v) for v in beta],
        objective=_q_dd(obj),
        optimal=bool(optimal),
        unique=bool(unique),
        n_zero=len(zero),
        margin=float(margin) if margin is not None else None,
    )


# ---- nonlinear least squares (sd.regression.nls, docs/design/regression.md §3.7)


class ArbVec:
    """A vector of arbs with elementwise arithmetic, so model functions written for an array
    namespace (tests/regression/nist_nls.py's) run in Arb. A 2-column x is a list of rows;
    x[:, j] gives column j."""

    __slots__ = ("v",)

    def __init__(self, v):
        self.v = list(v)

    def _op(self, o, fn):
        if isinstance(o, ArbVec):
            return ArbVec(fn(a, b) for a, b in zip(self.v, o.v))
        o = o if isinstance(o, arb) else arb(o)
        return ArbVec(fn(a, o) for a in self.v)

    def _rop(self, o, fn):
        o = o if isinstance(o, arb) else arb(o)
        return ArbVec(fn(o, a) for a in self.v)

    __add__ = lambda s, o: s._op(o, lambda a, b: a + b)
    __radd__ = lambda s, o: s._rop(o, lambda a, b: a + b)
    __sub__ = lambda s, o: s._op(o, lambda a, b: a - b)
    __rsub__ = lambda s, o: s._rop(o, lambda a, b: a - b)
    __mul__ = lambda s, o: s._op(o, lambda a, b: a * b)
    __rmul__ = lambda s, o: s._rop(o, lambda a, b: a * b)
    __truediv__ = lambda s, o: s._op(o, lambda a, b: a / b)
    __rtruediv__ = lambda s, o: s._rop(o, lambda a, b: a / b)
    __pow__ = lambda s, o: s._op(o, lambda a, b: a**b)
    __neg__ = lambda s: ArbVec(-a for a in s.v)

    def __getitem__(self, key):
        _, j = key
        return ArbVec(row[j] for row in self.v)


class _ArbNamespace:
    """exp, log, cos, sin, atan and stack on ArbVecs (the subset nist_nls.model uses)."""

    @staticmethod
    def _map(x, fn):
        return ArbVec(fn(a) for a in x.v)

    exp = staticmethod(lambda x: _ArbNamespace._map(x, arb.exp))
    log = staticmethod(lambda x: _ArbNamespace._map(x, arb.log))
    cos = staticmethod(lambda x: _ArbNamespace._map(x, arb.cos))
    sin = staticmethod(lambda x: _ArbNamespace._map(x, arb.sin))
    atan = staticmethod(lambda x: _ArbNamespace._map(x, arb.atan))

    @staticmethod
    def stack(cols, axis=1):
        return [c.v for c in cols]  # columns

    @staticmethod
    def asarray(v):
        if v and isinstance(v[0], (list, tuple)):
            return ArbVec([[arb(t) for t in row] for row in v])
        return ArbVec(arb(t) for t in v)


ARB = _ArbNamespace()


def nls(f, jac, x, y, theta0, *, weights=None, prec=320, tol_bits=120, max_iter=3000):
    """The minimizer of sum w_i (y_i - f(x_i, theta))^2 near theta0 for double data x, y (and
    weights), with f and jac written for ARB: Gauss-Newton in Arb at `prec` bits (the exact
    Jacobian, the normal equations at that precision) until a step is below 2^-tol_bits of
    the parameters. Returns dict(params, bse, ssr, scale) as floats (the midpoints rounded:
    the remaining step and the balls' radii are far below a double's rounding), and the
    iterations."""
    with ar.precision(prec):
        X = ARB.asarray(x)
        Y = [arb(v) for v in y]
        sw = [arb(w).sqrt() for w in weights] if weights is not None else None
        used = [i for i in range(len(Y)) if weights is None or weights[i] > 0]
        th = [arb(t) for t in theta0]
        p = len(th)

        def system(th):
            fv = f(X, *th).v
            cols = jac(X, *th)
            r = [(Y[i] - fv[i]) * (sw[i] if sw else 1) for i in used]
            J = [[cols[j][i] * (sw[i] if sw else 1) for j in range(p)] for i in used]
            return r, J

        it = 0
        for it in range(1, max_iter + 1):  # noqa: B007  (it is read after the loop: iterations=it)
            r, J = system(th)
            A = arb_mat(J)
            At = A.transpose()
            d = (At * A).solve(At * arb_mat([[v] for v in r]))
            step = max(abs(float(d[j, 0].mid())) / max(abs(float(th[j].mid())), 1e-300) for j in range(p))
            th = [arb((th[j] + d[j, 0]).mid()) for j in range(p)]
            if step < 2.0**-tol_bits:
                break
        r, J = system(th)
        A = arb_mat(J)
        ssr = sum((v * v for v in r), arb(0))
        n = len(used)
        scale = ssr / (n - p)
        C = (A.transpose() * A).inv()
        bse = [(scale * C[j, j]).sqrt() for j in range(p)]
        return dict(
            params=[float(t.mid()) for t in th],
            bse=[float(b.mid()) for b in bse],
            ssr=float(ssr.mid()),
            scale=float(scale.mid()),
            iterations=it,
        )


# ---- tests on fitted models (docs/design/regression.md §3.8)
# Linear hypotheses, nested comparisons, the analysis of variance, VIF, DFFITS, DFBETAS, White,
# Goldfeld-Quandt and RESET of linear models, and the same tests and the influence measures of
# GLMs. Every value is computed from its definition, not from the formulas sd uses: the
# comparisons and the analysis of variance from the sums of squares (deviances) of the fits they
# compare, DFFITS and DFBETAS from the n fits with one row deleted, VIF from the auxiliary
# regressions, White, Goldfeld-Quandt and RESET from their auxiliary fits. The linear fits solve
# the normal equations in ball arithmetic; the GLM's maximum likelihood estimates are glm()'s
# (Newton, then Krawczyk's certificate). As above, the whole computation is redone at twice the
# precision until every value has settled. model_tests_linear(...) and model_tests_glm(...)
# return (values, meta): values map each name to (hi, lo) or a list of them.


def _mt_ls(D, Y, W, used):
    """The weighted least squares of Y on the rows D (lists of arbs) over `used`: (beta,
    (D^T W D)^-1, ssr, fitted values of every row)."""
    k = len(D[0]) if D else 0
    if k == 0:
        ssr = sum((W[i] * Y[i] ** 2 for i in used), arb(0))
        return [], None, ssr, [arb(0)] * len(Y)
    G = arb_mat([[sum((W[i] * D[i][a] * D[i][b] for i in used), arb(0)) for b in range(k)] for a in range(k)])
    h = [sum((W[i] * D[i][a] * Y[i] for i in used), arb(0)) for a in range(k)]
    try:
        Gi = G.inv()
    except ZeroDivisionError:
        raise ValueError("singular normal equations") from None
    beta = [sum((Gi[a, b] * h[b] for b in range(k)), arb(0)) for a in range(k)]
    fit = [sum((D[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(len(Y))]
    ssr = sum((W[i] * (Y[i] - fit[i]) ** 2 for i in used), arb(0))
    return beta, Gi, ssr, fit


def _mt_cols(D, keep):
    return [[r[j] for j in keep] for r in D]


def _mt_quad(V, d):
    """d^T V^-1 d."""
    m = len(d)
    Vi = arb_mat(V).inv()
    return sum((d[a] * Vi[a, b] * d[b] for a in range(m) for b in range(m)), arb(0))


def _model_tests_linear(y, X, w, intercept, R, q, small, terms, gq, power):
    n = len(y)
    D = [([arb(1)] if intercept else []) + [arb(v) for v in r] for r in X]
    k = len(D[0])
    j0 = 1 if intercept else 0
    Y = [arb(v) for v in y]
    W = [arb(v) for v in w] if w is not None else [arb(1)] * n
    used = [i for i in range(n) if not W[i].is_zero()]
    m = len(used)
    beta, Gi, ssr, fit = _mt_ls(D, Y, W, used)
    df = m - k
    s2 = ssr / df
    dfa = arb(df)
    out = {}
    # linear hypotheses
    Ra = [[arb(v) for v in r] for r in R]
    Qa = [arb(v) for v in q]
    eff = [sum((r[a] * beta[a] for a in range(k)), arb(0)) for r in Ra]
    V = [[s2 * sum((ri[a] * Gi[a, b] * rj[b] for a in range(k) for b in range(k)), arb(0)) for rj in Ra] for ri in Ra]
    sdv = [V[i][i].sqrt() for i in range(len(Ra))]
    tv = [(eff[i] - Qa[i]) / sdv[i] for i in range(len(Ra))]
    out["t_effect"], out["t_sd"], out["t_tvalue"] = eff, sdv, tv
    out["t_pvalue"] = [2 * _t_upper(abs(t), dfa) for t in tv]
    d = [eff[i] - Qa[i] for i in range(len(Ra))]
    Wd = _mt_quad(V, d)
    mR = len(Ra)
    out["wald_F"] = Wd / mR
    out["wald_p"] = _f_upper(Wd / mR, arb(mR), dfa)
    # the nested model of the first `small` columns of X (and the intercept)
    Ds = _mt_cols(D, list(range(j0 + small)))
    _, _, ssr_s, _ = _mt_ls(Ds, Y, W, used)
    dfn = k - (j0 + small)
    Fc = ((ssr_s - ssr) / dfn) / s2
    out["compare_F"], out["compare_F_p"] = Fc, _f_upper(Fc, arb(dfn), dfa)
    lr = arb(m) * (ssr_s / ssr).log()
    out["compare_lr"], out["compare_lr_p"] = lr, _chi2_upper(lr, arb(dfn))
    # Type I: the sums of squares of the fits with the terms up to each; Type III: without each
    prev = None
    if intercept:
        sw = sum((W[i] for i in used), arb(0))
        ybar = sum((W[i] * Y[i] for i in used), arb(0)) / sw
        prev = sum((W[i] * (Y[i] - ybar) ** 2 for i in used), arb(0))
    else:
        prev = sum((W[i] * Y[i] ** 2 for i in used), arb(0))
    cols = []
    ss1, f1, p1, ss3, f3, p3 = [], [], [], [], [], []
    for t in terms:
        cols = cols + t
        _, _, s_t, _ = _mt_ls(_mt_cols(D, list(range(j0)) + [j0 + c for c in cols]), Y, W, used)
        ss = prev - s_t
        ss1.append(ss)
        f = (ss / len(t)) / s2
        f1.append(f)
        p1.append(_f_upper(f, arb(len(t)), dfa))
        prev = s_t
        keep = [c for c in range(k) if c < j0 or (c - j0) not in t]
        _, _, s_w, _ = _mt_ls(_mt_cols(D, keep), Y, W, used)
        ss = s_w - ssr
        ss3.append(ss)
        f = (ss / len(t)) / s2
        f3.append(f)
        p3.append(_f_upper(f, arb(len(t)), dfa))
    out["anova1_ss"], out["anova1_F"], out["anova1_p"] = ss1, f1, p1
    out["anova3_ss"], out["anova3_F"], out["anova3_p"] = ss3, f3, p3
    # VIF: column j of the whitened design on the whitened intercept and the other columns
    sq = {i: W[i].sqrt() for i in used}
    vif = []
    for j in range(k - j0):
        x = [D[i][j0 + j] for i in range(n)]
        others = [[arb(1)] + [D[i][j0 + c] for c in range(k - j0) if c != j] for i in range(n)]
        _, _, sr, _ = _mt_ls(others, x, W, used)
        sw = sum((W[i] for i in used), arb(0))
        xb = sum((W[i] * x[i] for i in used), arb(0)) / sw
        tss = sum((W[i] * (x[i] - xb) ** 2 for i in used), arb(0))
        vif.append(tss / sr)
    out["vif"] = vif
    # DFFITS and DFBETAS from the n fits with row i deleted
    hat = {i: W[i] * sum((D[i][a] * Gi[a, b] * D[i][b] for a in range(k) for b in range(k)), arb(0)) for i in used}
    dff, dfb = [], []
    for i in range(n):
        if i not in hat:
            dff.append(arb(0))
            dfb.append([arb(0)] * k)
            continue
        rest = [t for t in used if t != i]
        bi, _, si, _ = _mt_ls(D, Y, W, rest)
        s_i = (si / (df - 1)).sqrt()
        db = [beta[a] - bi[a] for a in range(k)]
        dfb.append([db[a] / (s_i * Gi[a, a].sqrt()) for a in range(k)])
        dyi = sq[i] * sum((D[i][a] * db[a] for a in range(k)), arb(0))
        dff.append(dyi / (s_i * hat[i].sqrt()))
    out["dffits"] = dff
    out["dfbetas"] = [v for row in dfb for v in row]
    # White: e~^2 on a constant, the whitened design's columns, squares and cross products
    e2 = {i: W[i] * (Y[i] - fit[i]) ** 2 for i in used}
    Z = {i: ([sq[i]] if (intercept and w is not None) else []) + [sq[i] * D[i][c] for c in range(j0, k)] for i in used}
    pz = len(next(iter(Z.values())))
    # the auxiliary columns that are exactly 0 or exact copies of earlier ones (0/1 dummies:
    # d^2 = d, d1 d2 = 0) are dropped, as sd's QR drops them; the products of the doubles decide
    Zf = {
        i: ([w[i] ** 0.5] if (intercept and w is not None) else [])
        + [X[i][c - j0] * (1.0 if w is None else w[i] ** 0.5) for c in range(j0, k)]
        for i in used
    }
    for cross in (True, False):
        A, Af = {}, {}
        for i in used:
            row, rowf = [arb(1)] + list(Z[i]), [1.0] + list(Zf[i])
            for a in range(pz):
                for b in range(a, pz) if cross else (a,):
                    row.append(Z[i][a] * Z[i][b])
                    rowf.append(Zf[i][a] * Zf[i][b])
            A[i], Af[i] = row, rowf
        idx = list(used)
        keep_c, seen = [], set()
        for c in range(len(Af[idx[0]])):
            col = tuple(Af[i][c] for i in idx)
            if all(v == 0.0 for v in col) or col in seen:
                continue
            seen.add(col)
            keep_c.append(c)
        Al = [[A[i][c] for c in keep_c] for i in idx]
        ev = [e2[i] for i in idx]
        ones = [arb(1)] * len(idx)
        _, _, sa, _ = _mt_ls(Al, ev, ones, range(len(idx)))
        eb = sum(ev, arb(0)) / len(idx)
        ta = sum(((v - eb) ** 2 for v in ev), arb(0))
        qa = len(Al[0]) - 1
        lm = arb(len(idx)) * (1 - sa / ta)
        fa = ((ta - sa) / qa) / (sa / (len(idx) - qa - 1))
        tag = "white" if cross else "white_nocross"
        out[tag + "_lm"], out[tag + "_lm_p"] = lm, _chi2_upper(lm, arb(qa))
        out[tag + "_F"], out[tag + "_F_p"] = fa, _f_upper(fa, arb(qa), arb(len(idx) - qa - 1))
    # Goldfeld-Quandt: the fit's rows (stably sorted by a column) split at n1, drop d
    n1, dr, by = gq
    rows = list(used)
    if by is not None:
        rows = sorted(rows, key=lambda i: X[i][by])
    part1, part2 = rows[:n1], rows[n1 + dr :]
    _, _, g1, _ = _mt_ls(D, Y, W, part1)
    _, _, g2, _ = _mt_ls(D, Y, W, part2)
    d1, d2 = len(part1) - k, len(part2) - k
    fq = (g2 / d2) / (g1 / d1)
    out["gq_F"], out["gq_p"] = fq, _f_upper(fq, arb(d2), arb(d1))
    # RESET: the fitted values' powers added
    Dz = [list(D[i]) + [fit[i] ** p_ for p_ in power] for i in range(n)]
    _, _, se, _ = _mt_ls(Dz, Y, W, used)
    qd = len(power)
    dfe = df - qd
    fr = ((ssr - se) / qd) / (se / dfe)
    out["reset_F"], out["reset_p"] = fr, _f_upper(fr, arb(qd), arb(dfe))
    return out, {"df_resid": df}


def model_tests_linear(y, X, *, weights=None, intercept=True, R, q, small, terms, gq=(None, 0, None), power=(2, 3)):
    """References for a linear model's tests and diagnostics (X rows without the constant column; R over
    params, the intercept first; small: the nested model's columns of X, the first ones; terms:
    lists of column positions of X; gq = (n1, drop, column to sort by or None) with n1 None
    for half the fit's rows)."""
    if gq[0] is None:
        m = len(y) if weights is None else sum(1 for v in weights if v > 0)
        gq = (m // 2, gq[1], gq[2])
    p = ar.START_PREC
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals, meta = _model_tests_linear(y, X, weights, intercept, R, q, small, terms, gq, power)
            except ar.NotSettled, ValueError, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                return {
                    k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()
                }, meta
        p *= 2
    raise ar.NotSettled("model_tests_linear")


def _mt_glm_at(y, X, family, link, beta0):
    """glm()'s fit at the working precision: its values as arbs and the MLE's mu."""
    n = len(y)
    one = [1.0] * n
    vals, meta = _glm_fit(y, X, family, link, one, one, one, [0.0] * n, True, None, beta0, 0.05, None, None, 0.0)
    return vals, meta


def _model_tests_glm(y, X, family, link, R, small, terms, starts):
    n = len(y)
    big, meta = _mt_glm_at(y, X, family, link, starts["big"])
    beta = big["params"]
    k = len(beta)
    D = [[arb(1)] + [arb(v) for v in r] for r in X]
    Y = [arb(v) for v in y]
    eta = [sum((D[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(n)]
    Wt = []
    for i in range(n):
        mu, muc, d1, _ = _link(link, eta[i])
        Wt.append(d1 * d1 / _var(family, mu, muc)[0])
    I = arb_mat([[sum((Wt[i] * D[i][a] * D[i][b] for i in range(n)), arb(0)) for b in range(k)] for a in range(k)])
    Ii = I.inv()
    phi = big["scale"]
    fixed = family in ("binomial", "poisson")
    df = n - k
    out = {}
    Ra = [[arb(v) for v in r] for r in R]
    eff = [sum((r[a] * beta[a] for a in range(k)), arb(0)) for r in Ra]
    V = [[phi * sum((ri[a] * Ii[a, b] * rj[b] for a in range(k) for b in range(k)), arb(0)) for rj in Ra] for ri in Ra]
    sdv = [V[i][i].sqrt() for i in range(len(Ra))]
    tv = [eff[i] / sdv[i] for i in range(len(Ra))]
    out["t_tvalue"] = tv
    out["t_pvalue"] = [2 * (_normal_upper(abs(t)) if fixed else _t_upper(abs(t), arb(df))) for t in tv]
    Wd = _mt_quad(V, eff)
    mR = len(Ra)
    if fixed:
        out["wald_stat"], out["wald_p"] = Wd, _chi2_upper(Wd, arb(mR))
    else:
        out["wald_stat"], out["wald_p"] = Wd / mR, _f_upper(Wd / mR, arb(mR), arb(df))
    sm, _ = _mt_glm_at(y, [r[:small] for r in X], family, link, starts["small"])
    lr = 2 * (big["llf"] - sm["llf"])
    dfn = k - 1 - small
    out["compare_lr"], out["compare_lr_p"] = lr, _chi2_upper(lr, arb(dfn))
    if not fixed:
        fc = ((sm["deviance"] - big["deviance"]) / dfn) / phi
        out["compare_F"], out["compare_F_p"] = fc, _f_upper(fc, arb(dfn), arb(df))
    # Type I: deviances of the fits with the terms up to each (from the null model's)
    prev = big["null_deviance"]
    cols = []
    dev1, p1 = [], []
    for i, t in enumerate(terms):
        cols = cols + t
        if i == len(terms) - 1:
            dv = big["deviance"]
        else:
            sub, _ = _mt_glm_at(y, [[r[c] for c in cols] for r in X], family, link, starts["seq"][i])
            dv = sub["deviance"]
        dd = prev - dv
        dev1.append(dd)
        if fixed:
            p1.append(_chi2_upper(dd / phi, arb(len(t))))
        else:
            p1.append(_f_upper((dd / len(t)) / phi, arb(len(t)), arb(df)))
        prev = dv
    out["anova1_dev"], out["anova1_p"] = dev1, p1
    # Type III: the Wald test of each term
    w3 = []
    for t in terms:
        idx = [1 + c for c in t]
        Vt = [[phi * Ii[a, b] for b in idx] for a in idx]
        w3.append(_mt_quad(Vt, [beta[a] for a in idx]))
    out["anova3_wald"] = w3
    # influence: the final IRLS weighted least squares, Pearson residuals
    hat = [Wt[i] * sum((D[i][a] * Ii[a, b] * D[i][b] for a in range(k) for b in range(k)), arb(0)) for i in range(n)]
    e = []
    for i in range(n):
        mu, muc, _, _ = _link(link, eta[i])
        e.append((Y[i] - mu) / _var(family, mu, muc)[0].sqrt())
    X2 = sum((v * v for v in e), arb(0))
    cooks, dff, dfb = [], [], []
    for i in range(n):
        h = hat[i]
        cooks.append(e[i] ** 2 * h / (phi * k * (1 - h) ** 2))
        s_i = phi.sqrt() if fixed else ((X2 - e[i] ** 2 / (1 - h)) / (df - 1)).sqrt()
        dff.append(e[i] * h.sqrt() / (s_i * (1 - h)))
        xt = [Wt[i].sqrt() * D[i][a] for a in range(k)]
        db = [sum((Ii[a, b] * xt[b] for b in range(k)), arb(0)) * e[i] / (1 - h) for a in range(k)]
        dfb.extend(db[a] / (s_i * Ii[a, a].sqrt()) for a in range(k))
    out["hat"], out["cooks"], out["dffits"], out["dfbetas"] = hat, cooks, dff, dfb
    return out, {"df_resid": df}


def model_tests_glm(y, X, *, family, link, R, small, terms, starts):
    """References for a GLM's tests and influence measures (an intercept; X rows without it). starts: double
    starting points for glm()'s Newton ('big', 'small', 'seq': one per Type I submodel but the
    last)."""
    p = 256
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals, meta = _model_tests_glm(y, X, family, link, R, small, terms, starts)
            except ar.NotSettled, ValueError, ZeroDivisionError:
                vals = None
            if vals is not None and _all_settled(vals):
                return {
                    k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()
                }, meta
        p *= 2
    raise ar.NotSettled("model_tests_glm")


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
    for name, ours in (
        ("params", r.params),
        ("bse", r.bse),
        ("pvalues", r.pvalues),
        ("hat", r.get_influence().hat_matrix_diag),
        ("bse_HC3", r.HC3_se),
        ("cooks_distance", r.get_influence().cooks_distance[0]),
    ):
        worst[name] = max(rel(o, dd_value(t)) for o, t in zip(ours, ref[name]))
    for name, ours in (("rsquared", r.rsquared), ("fvalue", r.fvalue), ("llf", r.llf), ("aic", r.aic), ("bic", r.bic)):
        worst[name] = rel(ours, dd_value(ref[name]))
    ci = r.conf_int(0.05)
    worst["conf_int"] = max(
        max(rel(ci[i][0], dd_value(ref["conf_int_lower"][i])), rel(ci[i][1], dd_value(ref["conf_int_upper"][i])))
        for i in range(4)
    )
    for k, v in worst.items():
        print(f"{k:16s} statsmodels vs Arb: {v:.2e}")
    print(meta)
