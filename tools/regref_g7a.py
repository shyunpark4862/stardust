"""Rigorous reference values for the G7a tests on fitted models with Arb (python-flint):
linear hypotheses, nested comparisons, the analysis of variance, VIF, DFFITS, DFBETAS, White,
Goldfeld-Quandt and RESET of linear models, and the same tests and the influence measures of
GLMs (docs/design/regression.md §3.8, §6).

Every value is computed from its definition, not from the formulas sd uses: the comparisons
and the analysis of variance from the sums of squares (deviances) of the fits they compare,
DFFITS and DFBETAS from the n fits with one row deleted, VIF from the auxiliary regressions,
White, Goldfeld-Quandt and RESET from their auxiliary fits. The linear fits solve the normal
equations in ball arithmetic (tools/regref.py); the GLM's maximum likelihood estimates are
regref's (Newton, then Krawczyk's certificate). As regref, the whole computation is redone at
twice the precision until every value has settled (arbref.settled). No mpmath.

lin(...) and glm(...) return (values, meta): values map each name to (hi, lo) or a list of them.
"""

import arbref as ar
import regref as rr
from flint import arb, arb_mat


def _ls(D, Y, W, used):
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


def _cols(D, keep):
    return [[r[j] for j in keep] for r in D]


def _quad(V, d):
    """d^T V^-1 d."""
    m = len(d)
    Vi = arb_mat(V).inv()
    return sum((d[a] * Vi[a, b] * d[b] for a in range(m) for b in range(m)), arb(0))


def _lin(y, X, w, intercept, R, q, small, terms, gq, power):
    n = len(y)
    D = [([arb(1)] if intercept else []) + [arb(v) for v in r] for r in X]
    k = len(D[0])
    j0 = 1 if intercept else 0
    Y = [arb(v) for v in y]
    W = [arb(v) for v in w] if w is not None else [arb(1)] * n
    used = [i for i in range(n) if not W[i].is_zero()]
    m = len(used)
    beta, Gi, ssr, fit = _ls(D, Y, W, used)
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
    out["t_pvalue"] = [2 * rr._t_upper(abs(t), dfa) for t in tv]
    d = [eff[i] - Qa[i] for i in range(len(Ra))]
    Wd = _quad(V, d)
    mR = len(Ra)
    out["wald_F"] = Wd / mR
    out["wald_p"] = rr._f_upper(Wd / mR, arb(mR), dfa)
    # the nested model of the first `small` columns of X (and the intercept)
    Ds = _cols(D, list(range(j0 + small)))
    _, _, ssr_s, _ = _ls(Ds, Y, W, used)
    dfn = k - (j0 + small)
    Fc = ((ssr_s - ssr) / dfn) / s2
    out["compare_F"], out["compare_F_p"] = Fc, rr._f_upper(Fc, arb(dfn), dfa)
    lr = arb(m) * (ssr_s / ssr).log()
    out["compare_lr"], out["compare_lr_p"] = lr, rr._chi2_upper(lr, arb(dfn))
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
        _, _, s_t, _ = _ls(_cols(D, list(range(j0)) + [j0 + c for c in cols]), Y, W, used)
        ss = prev - s_t
        ss1.append(ss)
        f = (ss / len(t)) / s2
        f1.append(f)
        p1.append(rr._f_upper(f, arb(len(t)), dfa))
        prev = s_t
        keep = [c for c in range(k) if c < j0 or (c - j0) not in t]
        _, _, s_w, _ = _ls(_cols(D, keep), Y, W, used)
        ss = s_w - ssr
        ss3.append(ss)
        f = (ss / len(t)) / s2
        f3.append(f)
        p3.append(rr._f_upper(f, arb(len(t)), dfa))
    out["anova1_ss"], out["anova1_F"], out["anova1_p"] = ss1, f1, p1
    out["anova3_ss"], out["anova3_F"], out["anova3_p"] = ss3, f3, p3
    # VIF: column j of the whitened design on the whitened intercept and the other columns
    sq = {i: W[i].sqrt() for i in used}
    vif = []
    for j in range(k - j0):
        x = [D[i][j0 + j] for i in range(n)]
        others = [[arb(1)] + [D[i][j0 + c] for c in range(k - j0) if c != j] for i in range(n)]
        _, _, sr, _ = _ls(others, x, W, used)
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
        bi, _, si, _ = _ls(D, Y, W, rest)
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
        _, _, sa, _ = _ls(Al, ev, ones, range(len(idx)))
        eb = sum(ev, arb(0)) / len(idx)
        ta = sum(((v - eb) ** 2 for v in ev), arb(0))
        qa = len(Al[0]) - 1
        lm = arb(len(idx)) * (1 - sa / ta)
        fa = ((ta - sa) / qa) / (sa / (len(idx) - qa - 1))
        tag = "white" if cross else "white_nocross"
        out[tag + "_lm"], out[tag + "_lm_p"] = lm, rr._chi2_upper(lm, arb(qa))
        out[tag + "_F"], out[tag + "_F_p"] = fa, rr._f_upper(fa, arb(qa), arb(len(idx) - qa - 1))
    # Goldfeld-Quandt: the fit's rows (stably sorted by a column) split at n1, drop d
    n1, dr, by = gq
    rows = list(used)
    if by is not None:
        rows = sorted(rows, key=lambda i: X[i][by])
    part1, part2 = rows[:n1], rows[n1 + dr :]
    _, _, g1, _ = _ls(D, Y, W, part1)
    _, _, g2, _ = _ls(D, Y, W, part2)
    d1, d2 = len(part1) - k, len(part2) - k
    fq = (g2 / d2) / (g1 / d1)
    out["gq_F"], out["gq_p"] = fq, rr._f_upper(fq, arb(d2), arb(d1))
    # RESET: the fitted values' powers added
    Dz = [list(D[i]) + [fit[i] ** p_ for p_ in power] for i in range(n)]
    _, _, se, _ = _ls(Dz, Y, W, used)
    qd = len(power)
    dfe = df - qd
    fr = ((ssr - se) / qd) / (se / dfe)
    out["reset_F"], out["reset_p"] = fr, rr._f_upper(fr, arb(qd), arb(dfe))
    return out, {"df_resid": df}


def lin(y, X, *, weights=None, intercept=True, R, q, small, terms, gq=(None, 0, None), power=(2, 3)):
    """References for a linear model's G7a values (X rows without the constant column; R over
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
                vals, meta = _lin(y, X, weights, intercept, R, q, small, terms, gq, power)
            except ar.NotSettled, ValueError, ZeroDivisionError:
                vals = None
            if vals is not None and rr._all_settled(vals):
                return {
                    k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()
                }, meta
        p *= 2
    raise ar.NotSettled("lin")


def _glm_one(y, X, family, link, beta0):
    """regref's GLM at the working precision: its values as arbs and the MLE's mu."""
    n = len(y)
    one = [1.0] * n
    vals, meta = rr._glm_fit(y, X, family, link, one, one, one, [0.0] * n, True, None, beta0, 0.05, None, None, 0.0)
    return vals, meta


def _glm(y, X, family, link, R, small, terms, starts):
    n = len(y)
    big, meta = _glm_one(y, X, family, link, starts["big"])
    beta = big["params"]
    k = len(beta)
    D = [[arb(1)] + [arb(v) for v in r] for r in X]
    Y = [arb(v) for v in y]
    eta = [sum((D[i][a] * beta[a] for a in range(k)), arb(0)) for i in range(n)]
    Wt = []
    for i in range(n):
        mu, muc, d1, _ = rr._link(link, eta[i])
        Wt.append(d1 * d1 / rr._var(family, mu, muc)[0])
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
    out["t_pvalue"] = [2 * (rr._normal_upper(abs(t)) if fixed else rr._t_upper(abs(t), arb(df))) for t in tv]
    Wd = _quad(V, eff)
    mR = len(Ra)
    if fixed:
        out["wald_stat"], out["wald_p"] = Wd, rr._chi2_upper(Wd, arb(mR))
    else:
        out["wald_stat"], out["wald_p"] = Wd / mR, rr._f_upper(Wd / mR, arb(mR), arb(df))
    sm, _ = _glm_one(y, [r[:small] for r in X], family, link, starts["small"])
    lr = 2 * (big["llf"] - sm["llf"])
    dfn = k - 1 - small
    out["compare_lr"], out["compare_lr_p"] = lr, rr._chi2_upper(lr, arb(dfn))
    if not fixed:
        fc = ((sm["deviance"] - big["deviance"]) / dfn) / phi
        out["compare_F"], out["compare_F_p"] = fc, rr._f_upper(fc, arb(dfn), arb(df))
    # Type I: deviances of the fits with the terms up to each (from the null model's)
    prev = big["null_deviance"]
    cols = []
    dev1, p1 = [], []
    for i, t in enumerate(terms):
        cols = cols + t
        if i == len(terms) - 1:
            dv = big["deviance"]
        else:
            sub, _ = _glm_one(y, [[r[c] for c in cols] for r in X], family, link, starts["seq"][i])
            dv = sub["deviance"]
        dd = prev - dv
        dev1.append(dd)
        if fixed:
            p1.append(rr._chi2_upper(dd / phi, arb(len(t))))
        else:
            p1.append(rr._f_upper((dd / len(t)) / phi, arb(len(t)), arb(df)))
        prev = dv
    out["anova1_dev"], out["anova1_p"] = dev1, p1
    # Type III: the Wald test of each term
    w3 = []
    for t in terms:
        idx = [1 + c for c in t]
        Vt = [[phi * Ii[a, b] for b in idx] for a in idx]
        w3.append(_quad(Vt, [beta[a] for a in idx]))
    out["anova3_wald"] = w3
    # influence: the final IRLS weighted least squares, Pearson residuals
    hat = [Wt[i] * sum((D[i][a] * Ii[a, b] * D[i][b] for a in range(k) for b in range(k)), arb(0)) for i in range(n)]
    e = []
    for i in range(n):
        mu, muc, _, _ = rr._link(link, eta[i])
        e.append((Y[i] - mu) / rr._var(family, mu, muc)[0].sqrt())
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


def glm(y, X, *, family, link, R, small, terms, starts):
    """References for a GLM's G7a values (an intercept; X rows without it). starts: double
    starting points for regref's Newton ('big', 'small', 'seq': one per Type I submodel but the
    last)."""
    p = 256
    while p <= ar.MAX_PREC:
        with ar.precision(p):
            try:
                vals, meta = _glm(y, X, family, link, R, small, terms, starts)
            except ar.NotSettled, ValueError, ZeroDivisionError:
                vals = None
            if vals is not None and rr._all_settled(vals):
                return {
                    k: ([ar.to_dd(x) for x in v] if isinstance(v, list) else ar.to_dd(v)) for k, v in vals.items()
                }, meta
        p *= 2
    raise ar.NotSettled("glm")
