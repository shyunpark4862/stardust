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


# ---- generalized linear models (docs/design/regression.md §3.4)

GLM_FAMILIES = ("gaussian", "binomial", "poisson", "gamma")


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


def _glm_fit(y, X, family, link, vw, fw, tr, off, intercept, scale, beta0, alpha, groups, maxlags, null_beta0,
             full=True):
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
            return -nobs / 2 * ((2 * arb.pi()).log() + (ssr / nobs).log() + 1) + \
                sum((F[i] * V[i].log() for i in used), arb(0)) / 2
        return sum((F[i] * _loglike_obs(family, Y[i], mus[i], mucs[i], V[i], T[i], phi) for i in used), arb(0))

    llf = loglike(mu, muc)
    # the null model
    if intercept and all(o == 0 for o in off):
        sw = sum((PW[i] for i in used), arb(0))
        m0 = sum((PW[i] * Y[i] for i in used), arb(0)) / sw
        mu0, muc0 = [m0] * n, [1 - m0] * n
    elif intercept:
        sub, _ = _glm_fit(y, [[] for _ in range(n)], family, link, vw, fw, tr, off, True, scale, [null_beta0], alpha,
                          None, None, null_beta0, full=False)
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
    out = {"params": beta, "bse": bse, "tvalues": zval, "pvalues": pval, "deviance": dev, "pearson_chi2": pear,
           "scale": phi, "llf": llf, "null_deviance": dev0, "llnull": llnull,
           "aic": -2 * llf + 2 * k, "bic": -2 * llf + nobs.log() * k,
           "fittedvalues": mu, "_mu": mu, "_muc": muc}
    out["bse_HC0"] = sand(sq)
    out["bse_HC1"] = sand(sq, nobs / (nobs - k))
    out["bse_HC2"] = sand([[x / (1 - hat[j]).sqrt() for x in sq[j]] for j in range(len(used))])
    out["bse_HC3"] = sand([[x / (1 - hat[j]) for x in sq[j]] for j in range(len(used))])
    Lg = maxlags if maxlags is not None else math.floor(4 * (len(used) / 100) ** (2 / 9))
    S = [[sum((s[a] * s[b] for s in sq), arb(0)) for b in range(k)] for a in range(k)]
    for lag in range(1, min(Lg, len(used) - 1) + 1):
        wl = 1 - arb(lag) / (Lg + 1)
        for a in range(k):
            for b in range(k):
                S[a][b] += wl * sum((sq[t][a] * sq[t - lag][b] + sq[t - lag][a] * sq[t][b] for t in range(lag, len(used))),
                                    arb(0))
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


def glm(y, X, *, family, link, var_weights=None, freq_weights=None, trials=None, offset=None, intercept=True,
        scale=None, beta0, alpha=0.05, groups=None, maxlags=None, null_beta0=0.0):
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
                vals, meta = _glm_fit(y, X, family, link, vw, fw, tr, off, intercept, scale, beta0, alpha, groups,
                                      maxlags, null_beta0)
                vals = {key: v for key, v in vals.items() if not key.startswith("_")}
            except (ar.NotSettled, ValueError, ZeroDivisionError):
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
            if all(abs(step[u, 0].mid()) <= (abs(theta[u]) + arb(2) ** -60) * arb(2) ** (-ctx.prec + 12)
                   for u in range(k)):
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
            except (ar.NotSettled, ZeroDivisionError):
                vals = None
            if vals is not None and _all_settled(vals):
                return {k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()}
        p *= 2
    raise ar.NotSettled("penalized")


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
