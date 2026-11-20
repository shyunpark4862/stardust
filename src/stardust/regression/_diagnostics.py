"""The diagnostics added in G7a: VIF, DFBETAS, the White, Goldfeld-Quandt and RESET tests of the
linear models, and the GLM's influence measures (docs/api/regression.md §9.2,
docs/design/regression.md §3.8).

All are of the whitened problem the kernel fits (sqrt(w) for WLS, L^-1 for GLS; the final IRLS
weights for a GLM), as the existing diagnostics are, and are formed from the fit's own QR
(the kernel's thin-Q mode: its Q, R and kept columns) instead of refits: VIF from the rows of
R^-1, DFBETAS from R^-1 Q^T. Rows with weight 0 are out of every sum and their per-row values
are 0 (G30).
"""

import math
import numbers
import operator

from .. import _core as _sd
from .. import linalg as _linalg
from ..stats import ChiSquared as _ChiSquared
from ..stats import F as _F
from ..stats import Normal as _Normal
from ._linear import _REFINE_STEPS, _arg, _as_real, _check_bool, _check_choice, _div, _expand, _type_name
from ._tests import GoldfeldQuandtResult, ResetTestResult, WhiteTestResult, _unit_triangle

_k = _sd.regression


def _thin_q(r):
    """(hat, Q (..., n, k), R (..., k, k), kept (..., k)) of the fit's QR, in the fit's units
    (cached: one QR and one orgqr)."""

    def make():
        func, y, X, w, u, tol, _, _ = r._fitargs
        return tuple(_k.fit(func, y, X, w, u, r._intercept, tol, 0, 0, None, 0, -1, True, True))

    return r._cached("thin_q", make)


def _rinv(R, kept):
    """R^-1 (..., k, k) of the kept block in its places (0 on the aliased rows and columns)."""
    k = R.shape[-1]
    U = _unit_triangle(R, kept)
    Ui = _linalg.solve_triangular(U, _sd.broadcast_to(_sd.eye(k), tuple(U.shape)), upper=True)
    both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
    return _sd.where(both, Ui, 0.0)


# ---- the linear models


def vif(r):
    """VIF_j = 1 / (1 - R_j^2), R_j^2 the centered R^2 of the whitened column j on the whitened
    intercept and the other columns: [(X~^T X~)^-1]_jj ||x~_j - P_1 x~_j||^2 (G7a decision 12),
    with [(X~^T X~)^-1]_jj the squared norm of row j of R^-1. A model without an intercept is
    refactored with one (statsmodels' default and car's value). NaN at the intercept and the
    aliased columns."""
    func, y, X, w, u, tol, _, _ = r._fitargs
    kept_model = r._raw[5]
    if r._intercept:
        R = r._raw[2]
        fx = r._xfactor()
        if fx is not None:  # back to the fit's units (powers of two, exact)
            R = R / _sd.expand_dims(fx, axis=-2)
        kept = kept_model
    else:
        _, _, R, kept = _k.fit(func, y, X, w, u, True, tol, 0, 0, None, 0, -1, True, True)
    Ui = _rinv(R, kept)
    diag = _sd.sum(Ui * Ui, axis=-1)  # (..., k1): [(X~^T X~)^-1]_jj with the intercept first
    # the centered sums of squares of the whitened columns
    if u is not None:
        ones = u
    elif w is not None:
        ones = _sd.sqrt(w)
    else:
        ones = None
    if ones is None:
        Xc = X - _sd.mean(X, axis=-2, keepdims=True)
        tss = _sd.sum(Xc * Xc, axis=-2)
    else:
        Xw = X if u is not None else X * _sd.expand_dims(ones, axis=-1)
        o = _sd.expand_dims(ones, axis=-1)
        coef = _sd.sum(o * Xw, axis=-2, keepdims=True) / _sd.sum(o * o, axis=-2, keepdims=True)
        Xc = Xw - o * coef
        tss = _sd.sum(Xc * Xc, axis=-2)
    v = diag[..., 1:] * tss
    v = _sd.where(kept[..., 1:], v, math.nan)
    if r._intercept:
        v = _sd.concat([_sd.full(tuple(v.shape[:-1]) + (1,), math.nan), v], axis=-1)
    else:
        v = _sd.where(kept_model, v, math.nan)
    return _nan_rows(r, v, 1)


def _nan_rows(r, v, nd):
    """v with NaN for the regressions whose data had NaN (propagate)."""
    bad = r._isnan()
    if isinstance(bad, _sd.ndarray):
        return _sd.where(_expand(bad, nd), math.nan, v)
    return v * math.nan if bad else v


def dfbetas(r):
    """(beta - beta_(i))_j / (s_(i) sqrt([(X~^T X~)^-1]_jj)) for every row i and parameter j (R's
    dfbetas, statsmodels' dfbetas): beta - beta_(i) = R^-1 q_i^T e~_i / (1 - h_i), q_i the row i
    of the thin Q, so the n deletions are one triangular solve; with the external studentized
    residual t_i, the value is (R^-1 q_i^T)_j t_i / (sqrt(1 - h_i) ||(R^-1)_j||). NaN for the
    aliased parameters; 0 for the rows with weight 0."""
    hat, Q, R, kept = _thin_q(r)
    Ui = _rinv(R, kept)
    nrm = _sd.sqrt(_sd.sum(Ui * Ui, axis=-1))  # (..., k)
    M = _sd.matmul(Q, _sd.matrix_transpose(Ui))  # (..., n, k): row i is (R^-1 q_i^T)^T
    t = r.resid_studentized_external
    f = t / _sd.sqrt(1.0 - hat)
    v = M * _sd.expand_dims(f, axis=-1) / _sd.expand_dims(nrm, axis=-2)
    return _sd.where(_sd.expand_dims(kept, axis=-2), v, math.nan)


def white_test(r, cross_terms):
    """White's test: the auxiliary least squares of e~^2 on a constant, the whitened design's
    columns and their squares and (cross_terms) cross products; lm = n R^2 against chi-square
    with the auxiliary df_model, and the auxiliary F test (statsmodels' het_white on wresid and
    the whitened design with a constant). With cross_terms=False the levels stay (White's
    textbook form; statsmodels' interaction_terms=False keeps the squares alone)."""
    from ._linear import OLSResult, _batch_shape

    _check_bool(cross_terms, "white_test", "cross_terms")
    func, y, X, w, u, tol, _, _ = r._fitargs
    e = r._wresid()
    wa = None
    if u is not None:
        Z = (
            _sd.concat([_sd.expand_dims(u, axis=-1) * _sd.ones(tuple(X.shape[:-1]) + (1,)), X], axis=-1)
            if r._intercept
            else X
        )
    elif w is not None:
        sw = _sd.expand_dims(_sd.sqrt(w), axis=-1)
        Z = sw * X
        if r._intercept:
            Z = _sd.concat([_sd.broadcast_to(sw, tuple(Z.shape[:-1]) + (1,)), Z], axis=-1)
        wa = _sd.where(w > 0, 1.0, 0.0)  # the rows out of the fit stay out
    else:
        Z = X  # the constant is the auxiliary regression's intercept
    p = Z.shape[-1]
    cols = [Z]
    for i in range(p):
        zi = Z[..., i : i + 1]
        if cross_terms:
            cols.append(zi * Z[..., i:])
        else:
            cols.append(zi * zi)
    A = _sd.concat(cols, axis=-1)
    raw = _k.fit("white_test", e * e, A, wa, None, True, tol, 0, _REFINE_STEPS)
    a = OLSResult(raw, param_names=None, intercept=True, cov_type="nonrobust", batch=_batch_shape(raw))
    lm = a.nobs * a.rsquared
    df = a.df_model
    dfp = _sd.where(df > 0, df, math.nan) if isinstance(df, _sd.ndarray) else (df if df > 0 else math.nan)
    pv = r._scalar(_ChiSquared(df=_arg(dfp)).ccdf(_arg(lm)))
    return WhiteTestResult._make(
        batch=tuple(r._batch),
        lm=lm,
        lm_pvalue=pv,
        fvalue=a.fvalue,
        f_pvalue=a.f_pvalue,
        df=df,
        df_num=df,
        df_denom=a.df_resid,
    )


_ALTERNATIVES = ("greater", "less", "two-sided")


def _count_option(v, func, arg, lo, frac_ok):
    """An int >= lo (rows), or a float in (0, 1) (frac_ok: a fraction of the rows; [0, 1) for drop)."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(
            f"{func}(): '{arg}' must be an int (rows) or a float in (0, 1) (a fraction), not {_type_name(v)}"
        )
    if isinstance(v, numbers.Integral):
        v = operator.index(v)
        if v < lo:
            raise ValueError(f"{func}(): '{arg}' must be >= {lo} rows or a fraction in (0, 1), not {v}")
        return int(v), None
    x = float(v)
    if not (frac_ok[0] <= x < 1.0 and (x > 0.0 or frac_ok[0] == 0.0)):
        raise ValueError(f"{func}(): '{arg}' as a fraction must be in (0, 1), not {x!r}")
    return None, x


def goldfeld_quandt(r, split, drop, order_by, alternative):
    """The Goldfeld-Quandt test: the fit's rows (ordered by order_by, a stable sort) are cut into
    the first n1 and the last ones after `drop` more; both parts are fitted with the model's
    columns and F = (SSR_2 / df_2) / (SSR_1 / df_1) against F(df_2, df_1) (R's gqtest; statsmodels'
    het_goldfeldquandt swaps the degrees of freedom of the one-sided p-values). The parts are
    the model's own fit restricted by 0/1 weights (a row's place among the fit's rows decides its
    part), so no row is copied or sorted."""
    from ._linear import GLSResult, WLSResult, _batch_shape

    func = "goldfeld_quandt"
    if isinstance(r, GLSResult):
        raise ValueError(f"{func}(): takes an ols or wls result; the GLS whitening mixes the rows that the test splits")
    n1_rows, n1_frac = _count_option(split, func, "split", 1, (0.0, 1.0))
    if n1_frac is not None and n1_frac == 0.0:
        raise ValueError(f"{func}(): 'split' as a fraction must be in (0, 1), not 0.0")
    d_rows, d_frac = _count_option(drop, func, "drop", 0, (0.0, 1.0))
    _check_choice(alternative, func, "alternative", _ALTERNATIVES)
    fn, y, X, w, u, tol, _, _ = r._fitargs
    n = y.shape[-1]
    used = _sd.ones(tuple(X.shape[:-1])) > 0 if w is None else w > 0
    used = _sd.broadcast_to(used, tuple(r._batch) + (n,)) if r._batch else used
    key = _order_key(r, func, order_by, X, n)
    usedf = _sd.where(used, 1.0, 0.0)
    m = _sd.sum(usedf, axis=-1)  # the fit's rows per regression
    if key is None:
        pos = _sd.cumulative_sum(usedf, axis=-1) - 1.0
    else:
        # rows out of the fit go last; a stable sort keeps the order of equal keys (R's order)
        kk = _sd.where(used, key, math.inf)
        order = _sd.argsort(kk, axis=-1, stable=True)
        rank = _sd.argsort(order, axis=-1, stable=True)
        pos = rank.astype(_sd.float64)
    mm = _expand(m, 1) if isinstance(m, _sd.ndarray) else m
    n1 = (
        float(n1_rows)
        if n1_rows is not None
        else _sd.floor(mm * n1_frac)
        if isinstance(mm, _sd.ndarray)
        else float(math.floor(mm * n1_frac))
    )
    dd = (
        float(d_rows)
        if d_rows is not None
        else (_sd.floor(mm * d_frac) if isinstance(mm, _sd.ndarray) else float(math.floor(mm * d_frac)))
    )
    part1 = _sd.logical_and(used, pos < n1)
    part2 = _sd.logical_and(used, pos >= n1 + dd)
    base = _sd.ones((n,)) if w is None else w
    stats = []
    for part in (part1, part2):
        wp = _sd.where(part, base, 0.0)
        raw = _k.fit(fn, y, X, wp, None, r._intercept, tol, 0, _REFINE_STEPS)
        a = WLSResult(raw, param_names=None, intercept=r._intercept, cov_type="nonrobust", batch=_batch_shape(raw))
        stats.append((a._ssr_s(), a.df_resid))
    (s1, df1), (s2, df2) = stats
    Fv = _div(_div(s2, df2), _div(s1, df1))
    bad = _either_zero(s1, s2, df1, df2)
    Fv = _sd.where(bad, math.nan, Fv) if isinstance(Fv, _sd.ndarray) else (math.nan if bad else Fv)
    if isinstance(Fv, _sd.ndarray) and not r._batch:
        Fv = float(Fv)
    dist = _F(dfn=_arg(_posdf(df2)), dfd=_arg(_posdf(df1)))
    if alternative == "greater":
        p = dist.ccdf(_arg(Fv))
    elif alternative == "less":
        p = dist.cdf(_arg(Fv))
    else:
        lo, hi = dist.cdf(_arg(Fv)), dist.ccdf(_arg(Fv))
        p = 2.0 * (_sd.minimum(lo, hi) if isinstance(lo, _sd.ndarray) else min(lo, hi))
    p = r._scalar(p) if not r._batch else p
    return GoldfeldQuandtResult._make(
        batch=tuple(r._batch), fvalue=Fv, pvalue=p, alternative=alternative, df_num=df2, df_denom=df1
    )


def _posdf(v):
    if isinstance(v, _sd.ndarray):
        return _sd.where(v > 0.0, v, math.nan)
    return v if v > 0 else math.nan


def _either_zero(s1, s2, df1, df2):
    """A part with an exact fit (SSR 0) or no degree of freedom gives no statistic (G95, G118:
    a zero scale is the perfect-fit rule's, not a variance of 0)."""
    if isinstance(s1, _sd.ndarray) or isinstance(s2, _sd.ndarray):
        return _sd.logical_or(_sd.logical_or(s1 == 0.0, s2 == 0.0), _sd.logical_or(df1 <= 0.0, df2 <= 0.0))
    return s1 == 0.0 or s2 == 0.0 or not df1 > 0 or not df2 > 0


def _order_key(r, func, order_by, X, n):
    """None (the rows' own order), or the sort key (..., n): a column of X (its position or
    name) or the values per row (as many as the rows given to the fit; with nan_policy='omit'
    the rows it dropped are dropped from these too)."""
    if order_by is None:
        return None
    if isinstance(order_by, str):
        names = r._param_names[1:] if r._intercept else list(r._param_names)
        if order_by not in names:
            raise ValueError(f"{func}(): 'order_by' names a column {order_by!r} that 'X' does not have")
        return X[..., names.index(order_by)]
    if isinstance(order_by, numbers.Integral) and not isinstance(order_by, bool):
        p = X.shape[-1]
        j = operator.index(order_by)
        if not -p <= j < p:
            raise ValueError(f"{func}(): 'order_by' column {j} is out of range for {p} columns")
        return X[..., j % p]
    if isinstance(order_by, bool):
        raise TypeError(f"{func}(): 'order_by' must be a column (int or str) or values per row, not bool")
    a, _ = _as_real(order_by, func, "order_by")
    keep = r._extra.get("keep")
    if keep is not None and a.ndim >= 1 and a.shape[-1] == keep.shape[0]:
        a = a[..., keep]
    if a.ndim < 1 or a.shape[-1] != n:
        got = a.shape[-1] if a.ndim else "a scalar"
        raise ValueError(f"{func}(): 'order_by' must have {n} values like the rows of the fit, not {got}")
    if bool(_sd.any(_sd.logical_not(_sd.isfinite(a)))):
        raise ValueError(f"{func}(): 'order_by' must be finite")
    return a


def _int_powers(power, func):
    """power: a sequence of distinct ints >= 2 (R's power = 2:3)."""
    if isinstance(power, (numbers.Number, str, bytes)) or power is None:
        raise TypeError(f"{func}(): 'power' must be a sequence of integers >= 2 like (2, 3), not {_type_name(power)}")
    try:
        vals = list(power.tolist() if isinstance(power, _sd.ndarray) else power)
    except TypeError:
        raise TypeError(
            f"{func}(): 'power' must be a sequence of integers >= 2 like (2, 3), not {_type_name(power)}"
        ) from None
    out = []
    for v in vals:
        if isinstance(v, bool) or not isinstance(v, numbers.Integral):
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            else:
                raise TypeError(f"{func}(): 'power' must hold integers, not {_type_name(v)}")
        v = operator.index(v)
        if v < 2:
            raise ValueError(f"{func}(): each power must be an integer >= 2, not {v}")
        out.append(v)
    if not out:
        raise ValueError(f"{func}(): 'power' must not be empty")
    if len(set(out)) != len(out):
        raise ValueError(f"{func}(): the powers must be distinct, not {out}")
    if max(out) > 64:
        raise ValueError(f"{func}(): each power must be <= 64, not {max(out)}")
    return out


def _unit_scale(v, axis):
    """v divided by a power of two near its largest magnitude along `axis` (exact): the powers
    of the scaled values stay near 1, and the test does not change with a column's scale."""
    mx = _sd.max(_sd.abs(v), axis=axis, keepdims=True)
    e = _sd.where(mx > 0.0, _sd.ceil(_sd.log2(_sd.where(mx > 0.0, mx, 1.0))), 0.0)
    return v * (2.0 ** (0.0 - e))


def reset_test(r, power, kind):
    """Ramsey's RESET: the fitted values' powers (kind='fitted') or each column's (kind='regressor')
    added to the model, and the F test that their coefficients are all 0 (R's resettest). With
    a robust covariance the test is the Wald F of the extended model's same covariance (lmtest's
    vcov); classically the F of the two sums of squares. The added columns are scaled to a
    magnitude near 1 first (a power of two: the test is the same, the extended design better
    conditioned). An added column aliased in the extended model adds no degree of freedom."""
    from ._linear import GLSResult, _batch_shape

    func = "reset_test"
    pw = _int_powers(power, func)
    _check_choice(kind, func, "kind", ("fitted", "regressor"))
    fn, y, X, w, u, tol, codes, gtot = r._fitargs
    gls = isinstance(r, GLSResult)
    if kind == "fitted":
        if gls:
            base = r.fittedvalues  # the original scale's X beta
        else:
            fy = r._fy()
            base = r.fittedvalues / (_expand(fy, 1) if isinstance(fy, _sd.ndarray) else fy)
        base = _sd.expand_dims(base, axis=-1)
    else:
        base = r._extra["X0"] if gls else X
    used = None if w is None else _sd.expand_dims(w > 0, axis=-1)
    bz = base if used is None else _sd.where(used, base, 0.0)
    z = _unit_scale(bz, -2)
    if used is not None:
        z = _sd.where(used, z, 0.0)
    Z = _sd.concat([z**p if p != 2 else z * z for p in pw], axis=-1)
    if gls:
        Z = _linalg.solve_triangular(r._extra["L"], Z, upper=False)
    Xe = _sd.concat([_sd.broadcast_to(X, tuple(Z.shape[:-1]) + (X.shape[-1],)) if X.ndim < Z.ndim else X, Z], axis=-1)
    added = Z.shape[-1]
    cov = ("nonrobust", "HC0", "HC1", "HC2", "HC3", "HAC", "cluster").index(r.cov_type)
    lags = r._extra.get("maxlags", -1)
    raw = list(_k.fit(fn, y, Xe, w, u, r._intercept, tol, cov, _REFINE_STEPS, codes, gtot, lags, False))
    ext = type(r)(
        raw,
        param_names=None,
        intercept=r._intercept,
        cov_type=r.cov_type,
        batch=_batch_shape(raw),
        fitargs=(fn, y, Xe, w, u, tol, codes, gtot),
    )
    k0 = r.params.shape[-1]
    kept_add = raw[5][..., k0 : k0 + added]
    m = _sd.sum(_sd.where(kept_add, 1.0, 0.0), axis=-1)
    m = r._scalar(m) if not r._batch else m
    dfd = ext._df_inference()
    if r.cov_type == "nonrobust":
        s0, s1 = r._ssr_s(), ext._ssr_s()
        Fv = _div(_div(s0 - s1, m), _div(s1, ext.df_resid))
        Fv = _nan_where_any(Fv, m, ext)
    else:
        Fv = _wald_subset(ext, k0, added) / m
    p = _F(dfn=_arg(_posdf(m)), dfd=_arg(dfd)).ccdf(_arg(Fv))
    p = r._scalar(p) if not r._batch else p
    return ResetTestResult._make(batch=tuple(r._batch), fvalue=Fv, pvalue=p, df_num=m, df_denom=dfd)


def _nan_where_any(Fv, m, ext):
    bad = ext._perfect()
    if isinstance(Fv, _sd.ndarray) or isinstance(m, _sd.ndarray) or isinstance(bad, _sd.ndarray):
        return _sd.where(_sd.logical_or(bad, m <= 0.0), math.nan, Fv)
    return math.nan if (bad or not m > 0) else Fv


def _wald_subset(ext, k0, added):
    """W of the extended model's added coefficients (those kept) all 0, with its covariance: the
    other coordinates are padded with the identity and a 0 effect, so that regressions with
    different kept columns share one batched solve."""
    from ._tests import _quad_inv

    k = ext.params.shape[-1]
    kept = ext._raw[5]
    sel = _sd.logical_and(kept, _sd.arange(k) >= k0)
    C = ext._cov_s()
    both = _sd.logical_and(_sd.expand_dims(sel, axis=-1), _sd.expand_dims(sel, axis=-2))
    V = _sd.where(both, C, 0.0) + _sd.where(_sd.expand_dims(sel, axis=-1), 0.0, _sd.eye(k))
    V = 0.5 * (V + _sd.matrix_transpose(V))
    b = ext._extra.get("beff_params_s", ext.params)
    d = _sd.where(sel, b, 0.0)
    W = _quad_inv(V, d, k)
    W = _sd.where(_sd.any(_sd.isnan(_sd.where(both, C, 0.0)), axis=(-2, -1)), math.nan, W)
    bad = ext._perfect()
    if isinstance(bad, _sd.ndarray):
        W = _sd.where(bad, math.nan, W)
    elif bad:
        W = W * math.nan
    if not ext._batch:
        W = float(W)
    return W


# ---- the GLM's influence measures (G7a decision 14)


def _glm_weights(r):
    """The final IRLS weights m mu'(eta)^2 / V(mu) per row (the expected information's; m the
    var_weights x trials), times the frequency weights; 0 for the rows out of the fit."""
    fa = r._fitargs
    aux = fa["aux"]
    mvar = aux[..., 0, :] * aux[..., 2, :]
    f = aux[..., 1, :]
    mu, eta = r.fittedvalues, r.linear_predictor
    fam, link = r.family, r.link
    if link == "identity":
        W = mvar / (mu if fam == "poisson" else (mu * mu if fam == "gamma" else 1.0))
    elif link == "log":
        if fam == "gaussian":
            W = mvar * mu * mu
        elif fam == "poisson":
            W = mvar * mu
        elif fam == "gamma":
            W = mvar * _sd.ones(tuple(mu.shape))
        else:  # binomial log: mu / (1 - mu)
            W = mvar * mu / (1.0 - mu)
    elif link == "logit":
        from .. import special as _special

        W = mvar * _special.expit(eta) * _special.expit(0.0 - eta)
    elif link == "probit":
        from .. import special as _special

        ph = _Normal().pdf(eta)
        W = mvar * ph * ph / (_special.ndtr(eta) * _special.ndtr(0.0 - eta))
    elif link == "cloglog":
        ee = _sd.exp(eta)
        W = mvar * ee * ee * _sd.exp(0.0 - ee) / (0.0 - _sd.expm1(0.0 - ee))
    elif link == "sqrt":
        W = 4.0 * mvar * _sd.ones(tuple(mu.shape))
    else:  # inverse (gamma)
        W = mvar * mu * mu
    wt = W * f
    used = _sd.logical_and(mvar > 0.0, f > 0.0)
    wt = _sd.where(used, wt, 0.0)
    return _sd.where(_sd.isfinite(wt), wt, 0.0), f, used


def glm_influence(r):
    """(leverage, cooks_distance, dffits, dfbetas) of a GLM: the linear models' measures of the
    final IRLS's weighted least squares, the leverage of sqrt(W) X and the Pearson residuals as
    the whitened ones, with the deleted Pearson scale s_(i) where the scale is estimated and
    sqrt(scale) where it is fixed (R 4.6's influence.measures, dffits and dfbetas of a glm). A
    row with frequency weight f counts as f observations, all deleted together."""
    fa = r._fitargs
    wt, f, used = _glm_weights(r)
    X = fa["X"]
    n = X.shape[-2]
    y0 = _sd.zeros((n,))
    hat, Q, R, kept = _k.fit("glm", y0, X, wt, None, r._intercept, fa["rtol"], 0, 0, None, 0, -1, True, True)
    e = _sd.sqrt(f) * r.resid_pearson
    e = _sd.where(used, e, 0.0)
    scale = r.scale
    sc = _expand(scale, 1) if isinstance(scale, _sd.ndarray) else scale
    rank = r._stat(r._I_RANK)
    rk = _expand(rank, 1) if isinstance(rank, _sd.ndarray) else rank
    om = 1.0 - hat
    cooks = e * e * hat / (sc * rk * om * om)
    if r._scale_fixed:
        sig = _sd.sqrt(_sd.full(tuple(e.shape), 1.0) * sc)
    else:
        S = _sd.sum(e * e, axis=-1, keepdims=True)
        df = r.df_resid
        dfx = _expand(df, 1) if isinstance(df, _sd.ndarray) else df
        sig = _sd.sqrt((S - e * e / om) / (dfx - f))
    dffits = e * _sd.sqrt(hat) / (sig * om)
    Ui = _rinv(R, kept)
    nrm = _sd.sqrt(_sd.sum(Ui * Ui, axis=-1))
    M = _sd.matmul(Q, _sd.matrix_transpose(Ui))
    fct = e / (om * sig)
    dfb = M * _sd.expand_dims(fct, axis=-1) / _sd.expand_dims(nrm, axis=-2)
    dfb = _sd.where(_sd.expand_dims(_sd.logical_and(kept, r._raw[5]), axis=-2), dfb, math.nan)
    zero = _sd.logical_not(used)
    cooks = _sd.where(zero, 0.0, cooks)
    dffits = _sd.where(zero, 0.0, dffits)
    dfb = _sd.where(_sd.expand_dims(zero, axis=-1), _sd.where(_sd.isnan(dfb), dfb, 0.0), dfb)
    # NaN for the regressions with NaN data, separation, or an estimated scale of exactly 0 /
    # without a degree of freedom (0/0, G95, G118)
    bad = r._isnan()
    sep = r.separation
    perf = r._perfect()
    if isinstance(bad, _sd.ndarray) or isinstance(sep, _sd.ndarray) or isinstance(perf, _sd.ndarray):
        allbad = _sd.logical_or(_sd.logical_or(bad, sep), perf)
        b1 = _expand(allbad, 1)
        hat = _sd.where(_expand(_sd.logical_or(bad, sep), 1), math.nan, hat)
        cooks, dffits = _sd.where(b1, math.nan, cooks), _sd.where(b1, math.nan, dffits)
        dfb = _sd.where(_expand(allbad, 2), math.nan, dfb)
    else:
        if bad or sep:
            hat = hat * math.nan
        if bad or sep or perf:
            cooks, dffits, dfb = cooks * math.nan, dffits * math.nan, dfb * math.nan
    return hat, cooks, dffits, dfb
