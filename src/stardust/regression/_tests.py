"""Tests on fitted models: linear hypotheses (t_test, wald_test), the comparison of nested
models (compare_f_test, compare_lr_test) and the analysis of variance (anova)
(docs/api/regression.md §9.1, docs/design/regression.md §3.8).

The hypothesis is R beta = q (G126-G130 and the G7a decisions). The covariance is the result's
own (cov_params(), a robust one included) and the reference distribution is the one of the
result's coefficients: t and F with df_resid_inference, or the normal and chi-square for a GLM
whose scale is fixed (G23, G27, G38, G39). Everything is formed in the fit's internal units
(the result's beff_params_s, _cov_s() and _colfactor(), docs/api/regression.md §1's extreme
scales), so that a hypothesis on one coefficient gives that coefficient's tvalues and bse.

The test results are read-only classes on sd.htest's result base (G133, H6): fields are
read-only properties, array fields read-only arrays (G120, G136), a field the test does not
give for the call raises AttributeError, and a class unpacks like statsmodels' tuple where
statsmodels returns one.
"""

import math

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from .. import linalg as _linalg
from ..htest import _common as _hc
from ..htest._result import _TestResult
from ..stats import ChiSquared as _ChiSquared
from ..stats import F as _F
from ._linear import (
    _arg,
    _as_real,
    _check_choice,
    _div,
    _expand,
    _fresh,
    _ro,
    _type_name,
)

_k = _sd.regression
_EPS = 2.0**-52


# ---- the result classes (G133; public in sd.regression)


class TTestResult(_TestResult):
    """The result of t_test(): per row of R, the estimate effect = (R params)_i, its standard
    error sd, tvalue = (effect - q_i) / sd, the two-sided pvalue and the degrees of freedom df of
    the t (inf for a normal reference); conf_int(*, alpha=0.05) is the interval of effect."""

    __slots__ = ()
    _FIELDS = ("effect", "sd", "tvalue", "pvalue", "df")
    _CI = True
    _TITLE = "Test for Constraints"

    @_calls.method("summary")
    def summary(self, *, alpha=0.05):
        """statsmodels' t_test summary (G191): one row per row of R (c0, c1, ...) with the
        estimate, its standard error, t (z for a normal reference), the p-value and the
        1 - alpha interval; a batch one row per regression and row of R."""
        return _t_summary(self, _hc.alpha(alpha, "summary"))


class WaldTestResult(_TestResult):
    """The result of wald_test(): the joint test of R params = q. distribution 'F' (statistic
    W / df_num against F(df_num, df_denom)) or 'chi2' (statistic W against chi-square with df)."""

    __slots__ = ()
    _TITLE = "Wald test"
    _FIELDS = ("statistic", "pvalue", "distribution", "df_num", "df_denom", "df")


class CompareFTestResult(_TestResult):
    """The result of compare_f_test(): unpacks as statsmodels' (fvalue, pvalue, df_diff)."""

    __slots__ = ()
    _TITLE = "F test of nested models"
    _FIELDS = ("fvalue", "pvalue", "df_num", "df_denom")
    _UNPACK = ("fvalue", "pvalue", "df_num")


class CompareLRTestResult(_TestResult):
    """The result of compare_lr_test(): statsmodels' CompareLRTestResult (statistic, pvalue,
    df_diff), the degrees of freedom named df (G134); unpacks to the three."""

    __slots__ = ()
    _TITLE = "Likelihood ratio test of nested models"
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue", "df")


class WhiteTestResult(_TestResult):
    """The result of white_test(): unpacks as statsmodels' het_white (lm, lm_pvalue, fvalue,
    f_pvalue); df is the chi-square's, df_num and df_denom the F's."""

    __slots__ = ()
    _TITLE = "White test for heteroscedasticity"
    _FIELDS = ("lm", "lm_pvalue", "fvalue", "f_pvalue", "df", "df_num", "df_denom")
    _UNPACK = ("lm", "lm_pvalue", "fvalue", "f_pvalue")


class GoldfeldQuandtResult(_TestResult):
    """The result of goldfeld_quandt(): unpacks as statsmodels' het_goldfeldquandt (fvalue,
    pvalue, and the alternative where statsmodels gives its ordering)."""

    __slots__ = ()
    _TITLE = "Goldfeld-Quandt test"
    _FIELDS = ("fvalue", "pvalue", "alternative", "df_num", "df_denom")
    _UNPACK = ("fvalue", "pvalue", "alternative")


class ResetTestResult(_TestResult):
    """The result of reset_test(): the F test of the added powers (does not unpack, as
    statsmodels' linear_reset)."""

    __slots__ = ()
    _TITLE = "Ramsey RESET test"
    _FIELDS = ("fvalue", "pvalue", "df_num", "df_denom")


def _t_summary(r, alpha):
    v = r._values
    batch = r._batch
    n = _S.lanes(batch) if batch else 1
    eff = v["effect"]
    one = not isinstance(eff, _sd.ndarray) or eff.ndim == len(batch)  # R was one row
    m = 1 if one else eff.shape[-1]
    lo, hi = r._ci(alpha)
    fields = [eff, v["sd"], v["tvalue"], v["pvalue"], lo, hi]
    flat = [_sd.reshape(f, (n, m)).tolist() if isinstance(f, _sd.ndarray) else [[float(f)]] for f in fields]
    df = v["df"]
    dfs = _sd.reshape(df, (df.size,)).tolist() if isinstance(df, _sd.ndarray) else [float(df)]
    stat = "z" if all(math.isinf(d) for d in dfs) else "t"
    fmts = (lambda x: _S.forg(x, 4), _S.forg, _S.forg, _S.pvalue, _S.forg, _S.forg)
    rows = []
    for i in _S.shown(n * m):
        if i is None:
            rows.append(None)
            continue
        lane, j = divmod(i, m)
        stub = f"c{j}" if not batch else f"{_S.lane_label(lane, batch)} c{j}"
        rows.append((stub, [fmt(f[lane][j]) for f, fmt in zip(flat, fmts)]))
    table = _S.Table(["coef", "std err", stat, f"P>|{stat}|"] + _S.interval_columns(alpha), rows)
    left = [("Df:", _S.df(dfs[0]))] if len(set(dfs)) == 1 else []
    right = [("No. Constraints:", str(m))]
    if batch:
        right.insert(0, ("Batch shape:", _S.shape_text(batch)))
    if "nobs" in r._info:
        right.append(("No. Observations:", _S.sizes(r._info["nobs"])))
    return _S.render(r._TITLE, header=(left, right), tables=[table])


def _model_info(r, **extra):
    """summary()'s description of the model a test is on (G191): its number of observations
    (one regression) and extra keys."""
    d = dict(extra)
    if not r._batch:
        d["nobs"] = r.nobs
    return d


# ---- the hypothesis


def _hypothesis(func, r, R, q):
    """(R as an (m, k) float64 array, q as (m,), whether R was one row (k,))."""
    k = r.params.shape[-1]
    Ra, _ = _as_real(R, func, "R")
    if Ra.ndim not in (1, 2):
        raise ValueError(
            f"{func}(): 'R' must be 1-dimensional (one constraint) or 2-dimensional (one row per constraint), "
            f"not {Ra.ndim}-dimensional"
        )
    one = Ra.ndim == 1
    R2 = _sd.reshape(Ra, (1, Ra.shape[0])) if one else Ra
    if R2.shape[-1] != k:
        raise ValueError(f"{func}(): 'R' must have {k} column{'' if k == 1 else 's'} like 'params', not {R2.shape[-1]}")
    m = R2.shape[0]
    if m == 0:
        raise ValueError(f"{func}(): 'R' must have at least one row")
    if not bool(_sd.all(_sd.isfinite(R2))):
        raise ValueError(f"{func}(): 'R' must be finite")
    if q is None:
        qv = _sd.zeros((m,))
    else:
        if isinstance(q, bool) or isinstance(q, (str, bytes)):
            raise TypeError(f"{func}(): 'q' must be a number or a sequence of numbers, not {_type_name(q)}")
        if isinstance(q, (int, float)) or (hasattr(q, "__float__") and not hasattr(q, "__len__")):
            qv = _sd.full((m,), float(q))
        else:
            qa, _ = _as_real(q, func, "q")
            if qa.ndim != 1 or qa.shape[0] != m:
                got = qa.shape[0] if qa.ndim == 1 else f"shape {tuple(qa.shape)}"
                raise ValueError(
                    f"{func}(): 'q' must have {m} value{'' if m == 1 else 's'} like the rows of 'R', not {got}"
                )
            qv = qa
        if not bool(_sd.all(_sd.isfinite(qv))):
            raise ValueError(f"{func}(): 'q' must be finite")
    return R2, qv, one


def _independent_rows(R2):
    """The rows of R2 (m, k) that are not linear combinations of the rows before them: R's
    dqrdc2 rule of the aliased columns (G21) on the rows, |t_jj| <= rtol ||R_j|| with rtol =
    max(m, k) eps, the residual of row j against the earlier kept rows by Gram-Schmidt twice."""
    m, k = R2.shape
    tol = max(m, k) * _EPS
    rows, basis = [], None
    for i in range(m):
        v = R2[i]
        nv = float(_sd.sqrt(_sd.sum(v * v)))
        if nv == 0.0:
            continue
        res = v
        if basis is not None:
            for _ in range(2):
                res = res - _sd.matmul(_sd.matmul(basis, res), basis)
        nr = float(_sd.sqrt(_sd.sum(res * res)))
        if not nr > tol * nv:
            continue
        b = _sd.reshape(res / nr, (1, k))
        basis = b if basis is None else _sd.concat([basis, b], axis=0)
        rows.append(i)
    return rows


def _fit_units(r):
    """(params in the fit's units (..., k) with 0 at the aliased columns, the base-2 exponents
    of the factors from the fit's units to the data's (..., k) or None, kept (..., k)). The
    exponents, log2 fy - log2 fx_j, are formed apart: the factor itself may leave the doubles
    (fy / fx_j of data near 1e+-300)."""
    kept = r._raw[5]
    b = r._extra.get("beff_params_s")
    if b is None:
        b = r.params
    fx = r._xfactor()
    ge = None
    if fx is not None:
        fy = r._fy()
        lfy = _sd.log2(fy) if isinstance(fy, _sd.ndarray) else math.log2(fy)
        ge = _expand(lfy, 1) - _sd.log2(fx)
    return _sd.where(kept, b, 0.0), ge, kept


def _scale2(x, c):
    """x 2^c (c an integer-valued array or None), in two halves: 2^c alone may leave the doubles."""
    if c is None:
        return x
    h = _sd.floor(c / 2.0)
    return x * (2.0**h) * (2.0 ** (c - h))


def _unit_triangle(R, kept):
    """R (..., k, k) of the kept columns in their places, the identity on the aliased ones."""
    both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
    k = R.shape[-1]
    eye = _sd.eye(k)
    return _sd.where(both, R, 0.0) + _sd.where(_sd.expand_dims(kept, axis=-1), 0.0, eye)


def _cov_kept(r, kept):
    """(_cov_s() with 0 outside the kept block, whether the kept block has a NaN (..., ))."""
    C = r._cov_s()
    both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
    Ck = _sd.where(both, C, 0.0)
    return Ck, _sd.any(_sd.isnan(Ck), axis=(-2, -1))


def _lincomb(r, R2, q):
    """The rows of the hypothesis in the fit's units, each row scaled by a power of two 2^-c_i
    (its largest column factor; exact) so that neither it nor its estimate leaves the doubles:
    (e = R params 2^-c, d = (R params - q) 2^-c, the rows' variances times 4^-c, rows touching
    an aliased parameter (..., m) bool, the scaled rows (..., m, k), the covariance's NaN flag,
    c (..., m) or None). R params in double-double (the kernel's apply, rounded once). t and W
    do not change with a row's scale."""
    b, ge, kept = _fit_units(r)
    touches = _sd.any(_sd.logical_and(R2 != 0.0, _sd.logical_not(_sd.expand_dims(kept, axis=-2))), axis=-1)  # (..., m)
    c = None
    if ge is None:
        Rp, qs = R2, q
    else:
        E = _sd.where(R2 != 0.0, _sd.expand_dims(ge, axis=-2), -math.inf)  # (..., m, k)
        c = _sd.max(E, axis=-1)
        c = _sd.where(_sd.isfinite(c), c, 0.0)
        Rp = R2 * (2.0 ** _sd.where(R2 != 0.0, _sd.expand_dims(ge, axis=-2) - _sd.expand_dims(c, axis=-1), 0.0))
        qs = _scale2(q, 0.0 - c)
    Rp = _sd.where(_sd.expand_dims(kept, axis=-2), Rp, 0.0)
    effect, qd = _k.apply("t_test", Rp, b, False, qs)
    d = 0.0 - qd
    Ck, nanC = _cov_kept(r, kept)
    fac = r._cov_factor()
    if fac is None:
        var = _sd.sum(_sd.matmul(Rp, Ck) * Rp, axis=-1)
    else:
        U, s = fac
        U, badU = _safe_triangle(U)
        Z = _linalg.solve_triangular(_sd.matrix_transpose(U), _sd.matrix_transpose(Rp), upper=False)
        var = _sd.sum(Z * Z, axis=-2) * _expand(s * s, 1)
        var = _nan_mask(var, _expand(badU, 1) if isinstance(badU, _sd.ndarray) else badU)
    return effect, d, var, touches, Rp, nanC, c


def _nan_mask(v, mask):
    """v with NaN where mask (an array broadcasting against v, or a bool)."""
    if isinstance(mask, _sd.ndarray):
        return _sd.where(mask, math.nan, v)
    return v * math.nan if mask else v


def t_test(r, R, q):
    func = "t_test"
    R2, qv, one = _hypothesis(func, r, R, q)
    e, d, var, touches, _, nanC, c = _lincomb(r, R2, qv)
    lead = tuple(r.params.shape[:-1])
    # NaN: the estimate where a row touches an aliased parameter, its standard error also where
    # the covariance has none (no degree of freedom left, G95)
    e = _sd.where(touches, math.nan, e)
    sdv = _sd.where(_sd.logical_or(touches, _expand(nanC, 1)), math.nan, _sd.sqrt(var))
    t = d / sdv
    perfect = r._perfect()
    t = _nan_mask(t, _expand(perfect, 1) if isinstance(perfect, _sd.ndarray) else perfect)
    p = 2.0 * r._dist(1).ccdf(_sd.abs(t))
    effect, sd = _scale2(e, c), _scale2(sdv, c)
    df = r._df_inference()
    dist = r._dist(1)

    def pick(v):
        if not one:
            return v
        return v[..., 0] if lead else float(v[0])

    def ci(alpha):
        # the ends in the rows' own units, then scaled: each keeps its sign (D-1's conf_int)
        q = dist.iccdf(alpha / 2)
        return pick(_scale2(e - q * sdv, c)), pick(_scale2(e + q * sdv, c))

    return TTestResult._make(
        batch=lead, ci=ci, info=_model_info(r), effect=pick(effect), sd=pick(sd), tvalue=pick(t), pvalue=pick(p), df=df
    )


def _wald_statistic(r, R2, q):
    """(W, the rows kept) of the joint hypothesis: the rows that are linear combinations of
    earlier ones are dropped (G21's rule on R's rows); W is NaN where a row touches an aliased
    parameter or the covariance is NaN. Without a robust covariance W = ||T^-T d||^2 / s^2,
    T the triangle of the QR of U^-T R^T (no R C R^T is formed: kappa, not kappa^2, G7a)."""
    rows = _independent_rows(R2)
    lead = tuple(r.params.shape[:-1])
    if not rows:
        return _sd.full(lead, math.nan) if lead else math.nan, 0
    R3 = _sd.take(R2, _sd.asarray(rows, dtype=_sd.int64), axis=0)
    q3 = _sd.take(q, _sd.asarray(rows, dtype=_sd.int64), axis=0)
    _, d, _, touches, Rp, nanC, _ = _lincomb(r, R3, q3)
    m = len(rows)
    fac = r._cov_factor()
    if fac is not None:
        U, s = fac
        # a regression whose rows touch an aliased parameter is NaN: its rows are replaced by
        # independent ones so that the batch's solve stays nonsingular
        k = Rp.shape[-1]
        safe = _sd.where(_sd.arange(k) == _sd.reshape(_sd.arange(m), (m, 1)), 1.0, 0.0)
        Rs = _sd.where(_expand(_sd.any(touches, axis=-1), 2), safe, Rp)
        U, badU = _safe_triangle(U)
        Z = _linalg.solve_triangular(_sd.matrix_transpose(U), _sd.matrix_transpose(Rs), upper=False)
        T, badT = _safe_triangle(_linalg.qr(Z)[1])
        y = _linalg.solve_triangular(_sd.matrix_transpose(T), _sd.expand_dims(d, axis=-1), upper=False)[..., 0]
        W = _div(_sd.sum(y * y, axis=-1), s * s)
        W = _nan_mask(
            W,
            _sd.logical_or(badU, badT)
            if isinstance(badU, _sd.ndarray) or isinstance(badT, _sd.ndarray)
            else (badU or badT),
        )
    else:
        Ck, _ = _cov_kept(r, r._raw[5])
        V = _sd.matmul(_sd.matmul(Rp, Ck), _sd.matrix_transpose(Rp))
        V = 0.5 * (V + _sd.matrix_transpose(V))
        W = _quad_inv(V, d, m)
    W = _nan_mask(W, _sd.logical_or(_sd.any(touches, axis=-1), nanC))
    W = _nan_mask(W, r._perfect())
    return W, m


def _safe_triangle(T):
    """(T with the identity in place of the triangles that are singular or not finite (..., m, m),
    those triangles (...,) bool): their tests are NaN, and the batch's solve goes through."""
    dg = _linalg.diagonal(T)
    bad = _sd.logical_or(_sd.any(dg == 0.0, axis=-1), _sd.logical_not(_sd.all(_sd.isfinite(T), axis=(-2, -1))))
    if not isinstance(bad, _sd.ndarray):
        bad = bool(bad)
        return (_sd.eye(T.shape[-1]) if bad else T), bad
    if not bool(_sd.any(bad)):
        return T, bad
    m = T.shape[-1]
    return _sd.where(_expand(bad, 2), _sd.eye(m), T), bad


def _quad_inv(V, d, m):
    """d^T V^-1 d for symmetric V (..., m, m) through its eigenvalues; NaN where V is not
    positive definite to rounding (a singular robust covariance: the model's own fvalue is NaN
    there too)."""
    lam, vec = _linalg.eigh(V)
    z = _sd.sum(vec * _sd.expand_dims(d, axis=-1), axis=-2)
    lo = _sd.min(lam, axis=-1)
    hi = _sd.max(lam, axis=-1)
    W = _sd.sum(z * z / lam, axis=-1)
    ok = _sd.logical_and(hi > 0.0, lo > max(m, 1) * 8 * _EPS * hi)
    return _sd.where(ok, W, math.nan)


def _chi2_model(r):
    """True for a model whose coefficients are judged with the normal (a GLM with a fixed scale,
    G38, G39): its joint tests are chi-square."""
    f = getattr(r, "_normal", None)
    return bool(f()) if f is not None else False


def wald_test(r, R, q):
    func = "wald_test"
    R2, qv, _ = _hypothesis(func, r, R, q)
    W, m = _wald_statistic(r, R2, qv)
    lead = tuple(r.params.shape[:-1])
    df_inf = r._df_inference()
    if _chi2_model(r):
        p = _ChiSquared(df=float(m) if m else math.nan).ccdf(_arg(W))
        return WaldTestResult._make(
            batch=lead,
            info=_model_info(r),
            statistic=W,
            pvalue=r._scalar(p) if not lead else p,
            distribution="chi2",
            df=float(m),
        )
    Fv = W / m if m else W
    p = _F(dfn=float(m) if m else math.nan, dfd=_arg(df_inf)).ccdf(_arg(Fv))
    return WaldTestResult._make(
        batch=lead,
        info=_model_info(r),
        statistic=Fv,
        pvalue=r._scalar(p) if not lead else p,
        distribution="F",
        df_num=float(m),
        df_denom=df_inf,
    )


# ---- nested models (G7a: the same data, bit for bit, and nesting are checked)


def _nan_equal(a, b):
    if a is None or b is None:
        return a is None and b is None
    if tuple(a.shape) != tuple(b.shape):
        return False
    return bool(_sd.all(_sd.logical_or(a == b, _sd.logical_and(_sd.isnan(a), _sd.isnan(b)))))


def _check_pair(func, big, small, *, allow_robust=False):
    """The checks of a comparison of two fitted models (decisions of G7a): the same result type,
    the same batch shape, no robust covariance, the same data and the small model's columns in
    the span of the big model's."""
    from ._glm import GLMResult
    from ._linear import _LinearResult

    if not isinstance(small, (_LinearResult, GLMResult)) or type(small) is not type(big):
        raise TypeError(
            f"{func}(): the model compared must be the same kind of result as this one ({type(big).__name__}), "
            f"not {_type_name(small)}"
        )
    if tuple(big._batch) != tuple(small._batch):
        raise ValueError(
            f"{func}(): the models must have the same batch shape, not {tuple(big._batch)} and {tuple(small._batch)}"
        )
    if not allow_robust:
        for m in (big, small):
            if m.cov_type != "nonrobust":
                raise ValueError(
                    f"{func}(): a model has a robust covariance (cov_type='{m.cov_type}'), which the F and likelihood "
                    "ratio comparisons do not use; test the coefficients with wald_test()"
                )
    if not _same_data(big, small):
        raise ValueError(f"{func}(): the models must be fitted to the same data (y, the rows used and their weights)")
    _check_nested(func, big, small)


def _same_data(big, small):
    from ._glm import GLMResult

    if isinstance(big, GLMResult):
        a, b = big._fitargs, small._fitargs
        if (a["family"], a["link"], a["smode"], a["svalue"]) != (b["family"], b["link"], b["smode"], b["svalue"]):
            return False
        return _nan_equal(a["y"], b["y"]) and _nan_equal(a["aux"], b["aux"])
    fa, fb = big._fitargs, small._fitargs
    if fa[0] != fb[0]:
        return False
    if not (_nan_equal(fa[1], fb[1]) and _nan_equal(fa[3], fb[3]) and _nan_equal(fa[4], fb[4])):
        return False
    la, lb = big._extra.get("logdet"), small._extra.get("logdet")
    if la is None or lb is None:
        return la is None and lb is None
    if isinstance(la, _sd.ndarray) or isinstance(lb, _sd.ndarray):
        return _nan_equal(_sd.asarray(la) + 0.0, _sd.asarray(lb) + 0.0)
    return la == lb or (math.isnan(la) and math.isnan(lb))


def _design_columns(m):
    """[(the column's position in params, the column (..., n) in the fit's units, unwhitened
    for WLS and the GLM)] of a fitted linear model or GLM, the intercept's first."""
    from ._glm import GLMResult

    if isinstance(m, GLMResult):
        X = m._fitargs["X"]
        u = None
    else:
        X, u = m._fitargs[2], m._fitargs[4]
    cols = []
    j0 = 0
    if m._intercept:
        if u is None:
            cols.append((0, None))  # the column of ones
        else:
            cols.append((0, u))
        j0 = 1
    for j in range(X.shape[-1]):
        cols.append((j0 + j, X[..., j]))
    return cols


def _check_nested(func, big, small):
    """Every column of the small model (kept in some regression) is a column of the big model,
    or lies in the span of the big model's kept columns (the rows of the fit, whitened):
    ||(I - Q Q^T) x|| <= tol ||x|| with tol = max(n, k) eps or the big model's rtol."""
    bcols = _design_columns(big)
    scols = _design_columns(small)
    skept = small._raw[5]
    lead = tuple(skept.shape[:-1])
    pending = []
    for j, c in scols:
        used = bool(_sd.any(skept[..., j])) if lead else bool(skept[j])
        if not used:
            continue
        if any(_same_column(c, cb) for _, cb in bcols):
            continue
        pending.append((j, c))
    if not pending:
        return
    Q, w, tol = _span_basis(big)
    n = Q.shape[-2]
    for j, c in pending:
        x = _sd.ones(tuple(Q.shape[:-2]) + (n,)) if c is None else c
        if w is not None:
            x = x * w
        proj = _sd.matmul(Q, _sd.expand_dims(_sd.sum(Q * _sd.expand_dims(x, axis=-1), axis=-2), axis=-1))[..., 0]
        res = x - proj
        nr = _sd.sqrt(_sd.sum(res * res, axis=-1))
        nx = _sd.sqrt(_sd.sum(x * x, axis=-1))
        ok = nr <= tol * nx
        bad = _sd.logical_and(skept[..., j], _sd.logical_not(ok))
        if bool(_sd.any(bad)) if isinstance(bad, _sd.ndarray) else bool(bad):
            name = small._param_names[j] if small._param_names else f"column {j}"
            raise ValueError(
                f"{func}(): the models are not nested: '{name}' of the smaller model is not in the span of the "
                "larger model's columns (call the comparison on the larger model, with the smaller one as its argument)"
            )


def _same_column(a, b):
    if a is None or b is None:
        return a is None and b is None
    return _nan_equal(a, b)


def _span_basis(big):
    """(thin Q (..., n, k) of the big model's whitened design over the rows of its fit, the
    whitening of a column (..., n) or None, the tolerance)."""
    from ._glm import GLMResult

    if isinstance(big, GLMResult):
        fa = big._fitargs
        X = fa["X"]
        aux = fa["aux"]
        pw = aux[..., 0, :] * aux[..., 1, :] * aux[..., 2, :]
        w = _sd.where(pw > 0.0, 1.0, 0.0)
        n = X.shape[-2]
        y0 = _sd.zeros((n,))
        _, Q, _, _ = _k.fit("glm", y0, X, w, None, big._intercept, fa["rtol"], 0, 0, None, 0, -1, True, True)
        k = Q.shape[-1]
        rt = fa["rtol"]
        return Q, w, max(max(n, k) * _EPS, rt if rt >= 0 else 0.0)
    func, y, X, w, u, tol, _, _ = big._fitargs
    _, Q, _, _ = _k.fit(func, y, X, w, u, big._intercept, tol, 0, 0, None, 0, -1, True, True)
    n, k = Q.shape[-2], Q.shape[-1]
    ws = None if w is None else _sd.sqrt(w)
    return Q, ws, max(max(n, k) * _EPS, tol if tol >= 0 else 0.0)


def _ssr_s(m):
    """The residual sum of squares (deviance for a GLM) in the fit's units."""
    from ._glm import _G_DEV, GLMResult

    if isinstance(m, GLMResult):
        return m._stat(_G_DEV)
    return m._ssr_s()


def _scale_s(m):
    from ._glm import _G_SCALE, GLMResult

    if isinstance(m, GLMResult):
        return m._stat(_G_SCALE)
    return m._scale_s()


def _df_num(big, small):
    v = big._stat(big._I_RANK) - small._stat(small._I_RANK)
    return v


def compare_f_test(big, small):
    func = "compare_f_test"
    from ._glm import GLMResult

    _check_pair(func, big, small)
    if isinstance(big, GLMResult) and big._scale_fixed:
        raise ValueError(
            f"{func}(): the {big.family} model's scale is fixed, so the F test of the deviance is the likelihood "
            "ratio test: use compare_lr_test()"
        )
    dfn = _df_num(big, small)
    dfd = big.df_resid
    Ss, Sb = _ssr_s(small), _ssr_s(big)
    if isinstance(big, GLMResult):
        Fv = _div(_div(Ss - Sb, dfn), _scale_s(big))
    else:
        Fv = _div(_div(Ss - Sb, dfn), _div(Sb, dfd))
    Fv = _no_df(Fv, dfn)
    Fv = _nan_mask(Fv, big._perfect())
    p = _F(dfn=_arg(_pos(dfn)), dfd=_arg(dfd)).ccdf(_arg(Fv))
    return CompareFTestResult._make(
        batch=tuple(big._batch),
        info=_model_info(big),
        fvalue=Fv,
        pvalue=big._scalar(p) if not big._batch else p,
        df_num=dfn,
        df_denom=dfd,
    )


def _pos(v):
    """v where > 0, NaN elsewhere (a test with no degree of freedom has no distribution)."""
    if isinstance(v, _sd.ndarray):
        return _sd.where(v > 0.0, v, math.nan)
    return v if v > 0 else math.nan


def _no_df(v, df):
    if isinstance(df, _sd.ndarray) or isinstance(v, _sd.ndarray):
        return _sd.where(df > 0.0, v, math.nan)
    return v if df > 0 else math.nan


def compare_lr_test(big, small):
    func = "compare_lr_test"
    from ._glm import GLMResult

    _check_pair(func, big, small)
    df = _df_num(big, small)
    if isinstance(big, GLMResult):
        lr = -2.0 * (small.llf - big.llf)
    else:
        # -2 (llf_s - llf_b) = nobs log(ssr_s / ssr_b): the same rows, weights and sigma
        Ss, Sb = _ssr_s(small), _ssr_s(big)
        ratio = _div(Ss - Sb, Sb)
        lr = big.nobs * (_sd.log1p(ratio) if isinstance(ratio, _sd.ndarray) else _log1p(ratio))
    lr = _no_df(lr, df)
    lr = _nan_mask(lr, big._perfect())
    p = _ChiSquared(df=_arg(_pos(df))).ccdf(_arg(lr))
    return CompareLRTestResult._make(
        batch=tuple(big._batch),
        info=_model_info(big),
        statistic=lr,
        pvalue=big._scalar(p) if not big._batch else p,
        df=df,
    )


def _log1p(x):
    if math.isnan(x):
        return math.nan
    if x == -1.0:
        return -math.inf
    if x < -1.0:
        return math.nan
    if math.isinf(x):
        return math.inf
    return math.log1p(x)


# ---- the analysis of variance (anova)


class AnovaResult:
    """The table of anova() (G7a decision 9): read-only fields, one entry per row.

    Models (anova(r1, r2, ...)): df_resid, ssr (the deviance for a GLM), df_diff, ss_diff,
    statistic, pvalue (the first row's comparison fields NaN). Terms (anova(r, typ=...)):
    terms (the names, 'Residual' last), df, sum_sq, mean_sq, statistic, pvalue. test is 'F' or
    'Chisq', typ None (models), 1 or 3. A field the table does not have raises AttributeError."""

    __slots__ = ("_values", "_kind", "_info")
    _MODEL_FIELDS = ("df_resid", "ssr", "df_diff", "ss_diff", "statistic", "pvalue")
    _TERM_FIELDS = ("terms", "df", "sum_sq", "mean_sq", "statistic", "pvalue")

    def __init__(self, kind, values, info=None):
        self._kind = kind
        self._values = {k: _ro(v) for k, v in values.items()}
        self._info = info or {}  # summary()'s model, nobs and df_resid (G190)

    def _get(self, name):
        try:
            return _fresh(self._values[name])
        except KeyError:
            raise AttributeError(
                f"'AnovaResult' object has no attribute '{name}' (the {self._kind} table has no such column)"
            ) from None

    def summary(self):
        """The table in the regression box (G190): the model, the type (or the number of models),
        the test, the observations and residual degrees of freedom, then R's columns (df sum_sq
        mean_sq F Pr(>F) for one model's terms; df_resid ssr df_diff ss_diff F Pr(>F) for
        models; Chisq and Pr(>Chi) for a chi-square test). A cell that does not apply (the first
        model's comparison, the residual row's test) is blank, as R prints it."""
        v, info = self._values, self._info
        typ, test = v["typ"], v["test"]
        stat = "F" if test == "F" else "Chisq"
        pname = "Pr(>F)" if test == "F" else "Pr(>Chi)"
        if self._kind == "models":
            title = "Analysis of Variance Table (models)"
            n = len(v["df_resid"].tolist())
            names = [str(i + 1) for i in range(n)]
            heads = ["df_resid", "ssr", "df_diff", "ss_diff", stat, pname]
            fields = self._MODEL_FIELDS
            left = [("Model:", info.get("model", "")), ("No. Models:", str(n)), ("Test:", test)]
        else:
            title = f"Analysis of Variance Table (Type {'I' if typ == 1 else 'III'})"
            names = list(v["terms"])
            heads = ["df", "sum_sq", "mean_sq", stat, pname]
            fields = self._TERM_FIELDS[1:]
            left = [("Model:", info.get("model", "")), ("Type:", "I" if typ == 1 else "III"), ("Test:", test)]
        right = [("No. Observations:", _S.count(info["nobs"]))] if "nobs" in info else []
        if "df_resid" in info:
            right.append(("Df Residuals:", _S.count(info["df_resid"])))
        fmts = {
            "df": _S.df,
            "df_resid": _S.df,
            "df_diff": _S.df,
            "statistic": _S.forg,
            "pvalue": _S.pvalue,
        }
        cols = [v[c].tolist() for c in fields]
        rows = []
        for i, nm in enumerate(names):
            cells = []
            for c, vals in zip(fields, cols):
                x = vals[i]
                cells.append("" if math.isnan(x) else fmts.get(c, lambda t: _S.forg(t, 4))(x))
            rows.append((nm, cells))
        return _S.render(title, header=(left, right), tables=[_S.Table(heads, rows)])

    def __repr__(self):
        if self._kind == "models":
            return f"AnovaResult(models={len(self._values['df_resid'].tolist())}, test={self._values['test']!r})"
        v = self._values
        return f"AnovaResult(typ={v['typ']}, terms={len(v['terms']) - 1}, test={v['test']!r})"


for _name in (
    "df_resid",
    "ssr",
    "df_diff",
    "ss_diff",
    "terms",
    "df",
    "sum_sq",
    "mean_sq",
    "statistic",
    "pvalue",
    "test",
    "typ",
):
    setattr(
        AnovaResult,
        _name,
        property((lambda nm: lambda self: self._get(nm))(_name), doc=f"The table's column '{_name}' (read-only)."),
    )
_calls.methods(AnovaResult)


def _test_option(func, test, models):
    """The anova test: None is R's default ('F' for a linear model and for a GLM with an
    estimated scale, 'Chisq' for a GLM with a fixed one); 'F' needs an estimated scale."""
    from ._glm import GLMResult

    big = models[-1]
    if test is None:
        if isinstance(big, GLMResult) and big._scale_fixed:
            return "Chisq"
        return "F"
    _check_choice(test, func, "test", ("F", "Chisq"))
    if test == "F" and isinstance(big, GLMResult) and big._scale_fixed:
        raise ValueError(
            f"{func}(): the {big.family} model's scale is fixed: its deviance test is test='Chisq' (or None)"
        )
    return test


def anova(*results, typ=None, terms=None, test=None):
    func = "anova"
    from ._glm import GLMResult
    from ._linear import _LinearResult

    if not results:
        raise TypeError(f"{func}(): takes at least one fitted model")
    for r in results:
        if not isinstance(r, (_LinearResult, GLMResult)):
            raise TypeError(f"{func}(): takes the results of ols, wls, gls or glm, not {_type_name(r)}")
        if r._batch:
            raise ValueError(f"{func}(): takes one regression, not a batch of shape {tuple(r._batch)}")
    if len(results) == 1:
        if typ is None:
            typ = 1
        if isinstance(typ, bool) or not isinstance(typ, int):
            raise TypeError(f"{func}(): 'typ' must be 1 or 3, not {_type_name(typ)}")
        if typ not in (1, 3):
            if typ == 2:
                raise ValueError(f"{func}(): 'typ' must be 1 or 3, not 2 (Type II needs which terms contain which)")
            raise ValueError(f"{func}(): 'typ' must be 1 or 3, not {typ}")
        return _anova_terms(func, results[0], typ, terms, test)
    if typ is not None or terms is not None:
        raise ValueError(f"{func}(): 'typ' and 'terms' are for one model's table, not for {len(results)} models")
    return _anova_models(func, results, test)


def _anova_models(func, models, test):
    from ._glm import GLMResult

    first = models[0]
    for m in models[1:]:
        if type(m) is not type(first):
            raise TypeError(f"{func}(): the models must all be {type(first).__name__}, not {type(m).__name__}")
    for m in models:
        if m.cov_type != "nonrobust":
            raise ValueError(
                f"{func}(): a model has a robust covariance (cov_type='{m.cov_type}'), which the table's F and "
                "chi-square tests do not use; test the coefficients with wald_test() or anova(r, typ=3)"
            )
    for a, b in zip(models, models[1:]):
        # each neighbor pair nested, in either order (R's anova takes both)
        big, small = (a, b) if a._stat(a._I_RANK) >= b._stat(b._I_RANK) else (b, a)
        _check_pair(func, big, small)
    dfr = [m.df_resid for m in models]
    big_i = min(range(len(models)), key=lambda i: (dfr[i], i))
    big = models[big_i]
    test = _test_option(func, test, [big])
    ssr = [_ssr_s(m) for m in models]
    if isinstance(big, GLMResult):
        scale = _scale_s(big)
    else:
        scale = _div(ssr[big_i], dfr[big_i])
    df_scale = dfr[big_i]
    n = len(models)
    df_diff = [math.nan] + [dfr[i - 1] - dfr[i] for i in range(1, n)]
    ss_diff = [math.nan] + [ssr[i - 1] - ssr[i] for i in range(1, n)]
    stat, pv = [math.nan], [math.nan]
    perfect = big._perfect()
    for i in range(1, n):
        dfd, ssd = df_diff[i], ss_diff[i]
        if dfd == 0 or math.isnan(dfd) or perfect:
            stat.append(math.nan)
            pv.append(math.nan)
            continue
        if test == "F":
            v = _div(_div(ssd, dfd), scale)
            if v < 0:
                v = math.nan
            stat.append(v)
            pv.append(float(_F(dfn=abs(dfd), dfd=float(df_scale)).ccdf(v)) if not math.isnan(v) else math.nan)
        else:
            v = _div(ssd, scale) * math.copysign(1.0, dfd)
            if v < 0:
                v = math.nan
            stat.append(v)
            pv.append(float(_ChiSquared(df=abs(dfd)).ccdf(v)) if not math.isnan(v) else math.nan)
    out2 = models[0]._out2
    vals = {
        "df_resid": _sd.asarray([float(v) for v in dfr]),
        "ssr": _sd.asarray([float(out2(v)) for v in ssr]),
        "df_diff": _sd.asarray(df_diff),
        "ss_diff": _sd.asarray([math.nan] + [float(out2(v)) for v in ss_diff[1:]]),
        "statistic": _sd.asarray(stat),
        "pvalue": _sd.asarray(pv),
        "test": test,
        "typ": None,
    }
    info = {"model": type(big).__name__.removesuffix("Result"), "nobs": big.nobs, "df_resid": big.df_resid}
    return AnovaResult("models", vals, info)


def _parse_terms(func, r, terms):
    """[(name, [column positions in X, 0-based])] covering every column of X once."""
    names = r._param_names[1:] if r._intercept else list(r._param_names)
    p = len(names)
    if terms is None:
        return [(names[j], [j]) for j in range(p)]
    if isinstance(terms, (str, bytes)) or not hasattr(terms, "__iter__"):
        raise TypeError(
            f"{func}(): 'terms' must be a list of terms (each a column or a list of columns), not {_type_name(terms)}"
        )

    def col(c):
        if isinstance(c, str):
            if c not in names:
                raise ValueError(f"{func}(): 'terms' names a column {c!r} that 'X' does not have")
            return names.index(c)
        if isinstance(c, bool) or not hasattr(c, "__index__"):
            raise TypeError(f"{func}(): a term's columns must be positions (int) or names (str), not {_type_name(c)}")
        j = c.__index__()
        if not -p <= j < p:
            raise ValueError(f"{func}(): 'terms' has the column {j}, out of range for {p} columns")
        return j % p

    out = []
    seen = set()
    for t in terms:
        cols = [col(t)] if (isinstance(t, (str, int)) and not isinstance(t, bool)) else [col(c) for c in t]
        if not cols:
            raise ValueError(f"{func}(): a term must have at least one column")
        for c in cols:
            if c in seen:
                raise ValueError(f"{func}(): the column {names[c]!r} is in more than one term")
            seen.add(c)
        out.append((" + ".join(names[c] for c in cols), cols))
    if len(seen) != p:
        missing = [names[c] for c in range(p) if c not in seen]
        raise ValueError(f"{func}(): 'terms' must put every column of 'X' in a term; missing {missing}")
    return out


def _anova_terms(func, r, typ, terms, test):
    from ._glm import GLMResult

    parsed = _parse_terms(func, r, terms)
    test = _test_option(func, test, [r])
    if typ == 1 and r.cov_type != "nonrobust":
        raise ValueError(
            f"{func}(): Type I sums of squares do not use the robust covariance (cov_type='{r.cov_type}'); "
            "use typ=3 (Wald tests with that covariance)"
        )
    glm = isinstance(r, GLMResult)
    scale = _scale_s(r)
    df_res = r.df_resid
    j0 = 1 if r._intercept else 0
    rows_df, rows_ss, rows_stat, rows_p = [], [], [], []
    if typ == 1:
        if glm:
            dev = _glm_sequential(r, parsed)
            for df_t, ss in dev:
                rows_df.append(df_t)
                rows_ss.append(ss)
        else:
            for df_t, ss in _linear_sequential(r, parsed):
                rows_df.append(df_t)
                rows_ss.append(ss)
        for df_t, ss in zip(rows_df, rows_ss):
            st, p = _term_test(test, ss, df_t, scale, df_res, r._perfect())
            rows_stat.append(st)
            rows_p.append(p)
    else:
        kept = r._raw[5]
        k = r.params.shape[-1]
        chi2 = _chi2_model(r)
        for _, cols in parsed:
            pos = [j0 + c for c in cols if bool(kept[j0 + c])]
            if not pos:
                rows_df.append(0.0)
                rows_ss.append(math.nan)
                rows_stat.append(math.nan)
                rows_p.append(math.nan)
                continue
            R = _sd.zeros((len(pos), k))
            R = _sd.where(_sd.arange(k) == _sd.reshape(_sd.asarray(pos, dtype=_sd.int64), (len(pos), 1)), 1.0, 0.0)
            W, m = _wald_statistic(r, R, _sd.zeros((len(pos),)))
            W = float(W)
            dfm = float(m)
            Fv = W / dfm if dfm else math.nan
            # statsmodels' back-out of a sum of squares from the test, scale x df x F
            ss = Fv * dfm * scale if not chi2 else W * scale
            rows_df.append(dfm)
            rows_ss.append(ss)
            if test == "F" and not chi2:
                rows_stat.append(Fv)
                rows_p.append(float(_F(dfn=dfm, dfd=float(r._df_inference())).ccdf(Fv)) if dfm else math.nan)
            else:
                rows_stat.append(W)
                rows_p.append(float(_ChiSquared(df=dfm).ccdf(W)) if dfm else math.nan)
    out2 = r._out2
    ss_out = [float(out2(v)) for v in rows_ss] + [float(out2(_ssr_s(r)))]
    df_out = [float(v) for v in rows_df] + [float(df_res)]
    mean = [_div(s, d) if d else math.nan for s, d in zip(ss_out, df_out)]
    vals = {
        "terms": [nm for nm, _ in parsed] + ["Residual"],
        "df": _sd.asarray(df_out),
        "sum_sq": _sd.asarray(ss_out),
        "mean_sq": _sd.asarray(mean),
        "statistic": _sd.asarray([float(v) for v in rows_stat] + [math.nan]),
        "pvalue": _sd.asarray([float(v) for v in rows_p] + [math.nan]),
        "test": test,
        "typ": typ,
    }
    info = {"model": type(r).__name__.removesuffix("Result"), "nobs": r.nobs, "df_resid": df_res}
    return AnovaResult("terms", vals, info)


def _term_test(test, ss, df_t, scale, df_res, perfect):
    """(statistic, p) of a Type I term: F = (ss / df) / scale against F(df, df_resid), or the
    scaled chi-square ss / scale against chi-square(df)."""
    if not df_t or perfect:
        return math.nan, math.nan
    if test == "F":
        v = _div(_div(ss, df_t), scale)
        return v, float(_F(dfn=float(df_t), dfd=float(df_res)).ccdf(v))
    v = _div(ss, scale)
    return v, float(_ChiSquared(df=float(df_t)).ccdf(v))


def _linear_sequential(r, parsed):
    """[(df, sum of squares in the fit's units)] of the terms in order: the effects Q^T y~ of
    the QR of the whitened design with the columns in the terms' order (R's anova.lm), summed
    in double-double; a column aliased in that order adds no degree of freedom. In the fit's
    own column order the effects are R beta (the fit's triangle and refined coefficients; no
    refactoring), else the design is factored in the terms' order (the kernel's thin Q)."""
    func, y, X, w, u, tol, _, _ = r._fitargs
    order = [c for _, cols in parsed for c in cols]
    p = X.shape[-1]
    if order == list(range(p)):
        kept = r._raw[5]
        Rt = r._raw[2]
        fx = r._xfactor()
        if fx is not None:  # the fit's units (powers of two, exact)
            Rt = Rt / _sd.expand_dims(fx, axis=-2)
        b, _, _ = _fit_units(r)
        both = _sd.logical_and(_sd.expand_dims(kept, axis=-1), _sd.expand_dims(kept, axis=-2))
        c = _k.apply("anova", _sd.where(both, Rt, 0.0), b, False, None)[0]
    else:
        Xo = _sd.take(X, _sd.asarray(order, dtype=_sd.int64), axis=-1)
        _, Q, _, kept = _k.fit(func, y, Xo, w, u, r._intercept, tol, 0, 0, None, 0, -1, True, True)
        yt = y if w is None else _sd.sqrt(w) * y
        c = _k.apply("anova", _sd.matrix_transpose(Q), yt, False, None)[0]
    j0 = 1 if r._intercept else 0
    out = []
    at = j0
    for _, cols in parsed:
        sl = slice(at, at + len(cols))
        cc = c[sl]
        kk = kept[sl]
        out.append((float(_sd.sum(_sd.where(kk, 1.0, 0.0))), float(_sd.sum(cc * cc))))
        at += len(cols)
    return out


def _glm_sequential(r, parsed):
    """[(df, deviance reduction in the fit's units)] of the terms in order: the GLM refitted
    with the terms up to each (R's anova.glm), starting from the null model's deviance."""
    from ._glm import _G_DEV, _G_NULLDEV, _G_RANK

    fa = r._fitargs
    prev_dev = r._stat(_G_NULLDEV)
    prev_rank = 1.0 if r._intercept else 0.0
    out = []
    cols = []
    for i, (_, tcols) in enumerate(parsed):
        cols = cols + tcols
        if i == len(parsed) - 1:
            dev, rank = r._stat(_G_DEV), r._stat(_G_RANK)
        else:
            raw = _glm_refit(fa, cols, r._intercept)
            st = raw[6]
            dev, rank = float(st[_G_DEV]), float(st[_G_RANK])
        out.append((rank - prev_rank, prev_dev - dev))
        prev_dev, prev_rank = dev, rank
    return out


def _glm_refit(fa, cols, intercept):
    """The GLM's kernel fit on the columns `cols` of X (the fit's units), the same options, a
    classical covariance."""
    X = fa["X"]
    Xc = _sd.take(X, _sd.asarray(cols, dtype=_sd.int64), axis=-1) if cols else X[..., :0]
    return _k.glm_fit(
        "anova",
        fa["y"],
        Xc,
        fa["aux"],
        None,
        0,
        fa["family_code"],
        fa["link_code"],
        fa["has"][0],
        fa["has"][1],
        fa["has"][2],
        fa["has"][3],
        intercept,
        fa["rtol"],
        0,
        -1,
        fa["max_iter"],
        fa["tol"],
        fa["smode"],
        fa["svalue"],
    )
