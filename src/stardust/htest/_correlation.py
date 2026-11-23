"""The correlation tests (milestone H5; docs/api/htest.md §6, docs/design/htest.md §3.6).

pearsonr, spearmanrho, spearmanr, pointbiserialr, kendalltau, weightedtau, somersd and
chatterjeexi: scipy's tests with this library's conventions (positional data, keyword-only
options, axis=None, batches along the other axes, nan_policy, read-only results; decisions
H3-H21 and the H5 decisions H62-H66 in docs/design/htest.md §1.1).

The checks and the arrangement of the paired samples are here; the computation is one native
call (stardust._core.htest: htest/correlation.cpp). Pearson's r is computed from double-double
sums of the centred values and its p-value from 1 - r^2 formed without cancellation (the
projection's residual near |r| = 1), so the p-value keeps its digits where scipy's 1 - r^2 of a
rounded r loses them. spearmanrho is the
same on exact midranks, spearmanr its matrix of every pair of columns (H62); neither has an exact
distribution (H63). kendalltau counts the pairs by a merge sort and takes the exact distribution
of the inversions (Mahonian numbers) without ties; weightedtau is Vigna's algorithm with
double-double sums and, as scipy's, no p-value (H64). somersd and chatterjeexi form their
statistics and variances from exact integer sums. No result has scipy's alias `correlation`
(H65).
"""

import math
import operator

from .. import _calls
from .. import _core as _sd
from ..stats._descriptive import _as_array, _type_name
from . import _common as C
from . import _resampling as R
from ._gof import _nest
from ._result import _TestResult

_k = _sd.htest
_st = _sd.stats
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_REQ = {"auto": 0, "exact": 1, "asymptotic": 2}
_USED = ("exact", "asymptotic")
_VARIANT = {"b": 0, "c": 1}
_EXACT_CAP = 1e9  # kendalltau's exact work, n (c + 1) entry updates (H13; 1e7 entries held, htest/kendall.hpp)
_TABLE_CAP = 1e7  # somersd's cells from two samples (the table is the result's field)


# ---- the result classes (H6: not exported) ------------------------------------------------


class PearsonrResult(_TestResult):
    __slots__ = ()
    _TITLE = "Pearson correlation test"
    _CI_LABEL = "correlation"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")
    _CI = True

    @_calls.method("conf_int")
    def conf_int(self, *, alpha=0.05, method=None):
        """The 1 - alpha confidence interval of r: Fisher's z (method=None), or a BootstrapMethod's
        interval of r (H9, scipy's confidence_interval(method=)), within [-1, 1]; one-sided for a
        one-sided test. A new (..., 2) array, (2,) for one test."""
        a = C.alpha(alpha)
        if method is not None:
            R.check_method(method, "conf_int", R.BootstrapMethod, ("None",))
        low, high = self._ci(a, method)
        if isinstance(low, _sd.ndarray):
            return _sd.stack([low + 0.0, high + 0.0], axis=-1)
        return _sd.asarray([float(low), float(high)])


class SpearmanrhoResult(_TestResult):
    __slots__ = ()
    _TITLE = "Spearman rank correlation test"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


class SpearmanrResult(SpearmanrhoResult):
    __slots__ = ()
    _TITLE = "Spearman rank correlation test"


class PointbiserialrResult(SpearmanrhoResult):
    __slots__ = ()
    _TITLE = "Point-biserial correlation test"


class KendalltauResult(_TestResult):
    __slots__ = ()
    _TITLE = "Kendall rank correlation test"
    _FIELDS = ("statistic", "pvalue", "method")
    _UNPACK = ("statistic", "pvalue")


class WeightedtauResult(SpearmanrhoResult):
    __slots__ = ()
    _TITLE = "Weighted Kendall tau"


class SomersdResult(_TestResult):
    __slots__ = ()
    _TITLE = "Somers' D"
    _FIELDS = ("statistic", "pvalue", "table")
    _TABLES = ("table",)


class ChatterjeexiResult(SpearmanrhoResult):
    __slots__ = ()
    _TITLE = "Chatterjee's xi correlation test"


# ---- helpers ------------------------------------------------------------------------------


def _shape(s):
    return s.keep_shape if s.keepdims else s.batch


def _paired(name, x, y, axis, nan_policy, keepdims):
    return C.Samples(name, [x, y], ("x", "y"), axis, nan_policy, paired=True, keepdims=keepdims, scan=False)


def _resampling(method, name):
    """method=: None, a PermutationMethod or a MonteCarloMethod (H9)."""
    if method is not None:
        R.check_method(method, name, (R.PermutationMethod, R.MonteCarloMethod), ("None",))


def _fisher_interval(z, n, alt, s, boot=None):
    """pearsonr's conf_int (scipy's _pearsonr_fisher_ci): tanh(atanh r -+ q / sqrt(n - 3)), q the
    normal quantile; a one-sided test's other end is -1 or 1; n <= 3 gives (-1, 1)."""

    def ci(alpha, method=None):
        if method is not None:
            return boot(alpha, method)
        q = float(_st.normal_iccdf(0.0, 1.0, alpha / 2.0 if alt == "two-sided" else alpha))
        small = n <= 3.0
        se = 1.0 / _sd.sqrt(_sd.where(small, 4.0, n) - 3.0)
        low = _sd.tanh(z - q * se) if alt != "less" else _sd.full(z.shape, -1.0)
        high = _sd.tanh(z + q * se) if alt != "greater" else _sd.full(z.shape, 1.0)
        nan = _sd.isnan(z)
        low = _sd.where(nan, math.nan, _sd.where(small, -1.0, low))
        high = _sd.where(nan, math.nan, _sd.where(small, 1.0, high))
        return s.out(low), s.out(high)

    return ci


def _pearson(name, x, y, axis, alternative, nan_policy, keepdims, ranks):
    alt = C.alternative(alternative, name)
    s = _paired(name, x, y, axis, nan_policy, keepdims)
    r, p, z, n = _k.pearson(s.rows[0], s.rows[1], _ALT[alt], s.policy == "omit", ranks)
    return s, alt, r, p, z, n


# ---- Pearson and Spearman -----------------------------------------------------------------


def pearsonr(x, y, /, *, axis=None, alternative="two-sided", method=None, nan_policy="propagate", keepdims=False):
    """Pearson's correlation coefficient and the test of no correlation: statistic r, pvalue (the
    beta distribution of r under the normal theory, scipy's; from 1 - r^2 formed without
    cancellation) and conf_int(alpha=0.05, method=None) (Fisher's z, or a BootstrapMethod's). NaN
    where a sample is constant (H20) or holds an inf, or for fewer than 2 pairs; n = 2 gives r =
    +-1 and p = 1. method: None, a PermutationMethod (y's order permuted) or a MonteCarloMethod
    (both drawn by its rvs; H9). Unpacks as (statistic, pvalue)."""
    name = "pearsonr"
    _resampling(method, name)
    s, alt, r, p, z, n = _pearson(name, x, y, axis, alternative, nan_policy, keepdims, False)
    info = s.info(alt)

    def stat(a, b):  # (pearsonr(a, b, axis=-1).statistic from the kernel, no p-values: H119)
        rows, lead = R.kernel_rows([a, b])
        return R.from_lanes(_k.pearson(rows[0], rows[1], -1, False, False)[0], lead)

    if method is not None:  # H9: y's order permuted (x held), or both drawn by the method's rvs
        p, pairs = R.method_pvalue(name, method, s, stat, alt, ptype="pairings", held=1)
        info["pairs"] = pairs

    def boot(alpha, bm):
        lo, hi = R.method_interval("conf_int", bm, s, stat, alt, alpha, clip=(-1.0, 1.0))
        return s.out(lo), s.out(hi)

    return PearsonrResult._make(
        batch=s.batch, ci=_fisher_interval(z, n, alt, s, boot), info=info, statistic=s.out(r), pvalue=s.out(p)
    )


def pointbiserialr(x, y, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """The point-biserial correlation (x the binary variable): pearsonr's statistic and two-sided
    pvalue. Unpacks as (statistic, pvalue)."""
    name = "pointbiserialr"
    s, _, r, p, _, _ = _pearson(name, x, y, axis, "two-sided", nan_policy, keepdims, False)
    return PointbiserialrResult._make(batch=s.batch, info=s.info(), statistic=s.out(r), pvalue=s.out(p))


def spearmanrho(x, y, /, *, axis=None, alternative="two-sided", method=None, nan_policy="propagate", keepdims=False):
    """Spearman's rank correlation: statistic rho (Pearson's r of the exact midranks) and pvalue
    (Pearson's beta distribution of the ranks' r, the t approximation; no exact distribution,
    H63). scipy 1.18's new form: paired samples broadcast, batches along the other axes. method:
    None, a PermutationMethod or a MonteCarloMethod (H9; the Monte Carlo samples are ranked too).
    Unpacks as (statistic, pvalue)."""
    name = "spearmanrho"
    _resampling(method, name)
    s, alt, r, p, _, _ = _pearson(name, x, y, axis, alternative, nan_policy, keepdims, True)
    info = s.info(alt)
    if method is not None:  # H9: rho of y's permutations or of the method's draws (ranked: H126)

        def rho(a, b):  # (spearmanrho(a, b, axis=-1).statistic from the kernel, no p-values: H119)
            rows, lead = R.kernel_rows([a, b])
            return R.from_lanes(_k.pearson(rows[0], rows[1], -1, False, True)[0], lead)

        p, pairs = R.method_pvalue(name, method, s, rho, alt, ptype="pairings", held=1)
        info["pairs"] = pairs
    return SpearmanrhoResult._make(batch=s.batch, info=info, statistic=s.out(r), pvalue=s.out(p))


def _axis01(axis, name):
    """spearmanr's axis: None, 0 or 1 (scipy's)."""
    if axis is None:
        return None
    try:
        if isinstance(axis, bool):
            raise TypeError
        i = operator.index(axis)
    except TypeError:
        raise TypeError(f"{name}(): 'axis' must be None, 0 or 1, not {_type_name(axis)}") from None
    if i not in (0, 1):
        raise ValueError(f"{name}(): 'axis' must be None, 0 or 1, not {i}")
    return i


def _variables(v, axis, name, arg):
    """The variables of one input as 1-D arrays: a 1-D array is one variable, a 2-D one has a
    variable per column (axis=0) or per row (axis=1); axis=None flattens to one variable."""
    a = _as_array(v, name, arg)
    if a.ndim > 2:
        raise ValueError(f"{name}(): '{arg}' must have at most 2 dimensions, not {a.ndim}")
    if axis is None or a.ndim <= 1:
        return [_sd.reshape(a, (a.size,))]
    if axis == 0:
        return [a[:, i] for i in range(a.shape[1])]
    return [a[i, :] for i in range(a.shape[0])]


def spearmanr(a, b=None, /, *, axis=None, alternative="two-sided", nan_policy="propagate"):
    """Spearman's rank correlation of several variables (scipy's older form, H62): the variables are
    a's columns (axis=0), rows (axis=1) or a flattened (axis=None), and b's likewise when given.
    Two variables give statistic and pvalue as numbers, more give the (k, k) matrices (statistic
    1 and pvalue 0 on the diagonal). Entry (i, j) is spearmanrho(variable i, variable j) to the
    bit. nan_policy: 'propagate' (a variable holding a NaN has NaN entries), 'omit' (each pair
    drops the observations where either is NaN), 'raise'. Unpacks as (statistic, pvalue)."""
    name = "spearmanr"
    ax = _axis01(axis, name)
    alt = C.alternative(alternative, name)
    pol = C.nan_policy(nan_policy, name)
    vs = _variables(a, ax, name, "a")
    if b is not None:
        vs += _variables(b, ax, name, "b")
    if len(vs) < 2:
        raise ValueError(f"{name}(): needs at least 2 variables to compare, not {len(vs)}")
    sizes = sorted({v.size for v in vs})
    if len(sizes) > 1:
        listed = " and ".join(str(k) for k in sizes)
        raise ValueError(f"{name}(): the variables must have the same number of observations, not {listed}")
    cols = _sd.stack([v + 0.0 for v in vs], axis=0)
    if pol == "raise" and cols.size and bool(_sd.any(_sd.isnan(cols))):
        raise ValueError(f"{name}(): the input contains NaN (nan_policy='raise')")
    k = len(vs)
    r, p = _k.corr_matrix(cols, _ALT[alt], pol == "omit")
    info = {"alternative": alt}
    if pol != "omit" or not bool(_sd.any(_sd.isnan(cols))):
        info["nobs"] = vs[0].size
    if k == 2:
        return SpearmanrResult._make(info=info, statistic=float(r[1]), pvalue=float(p[1]))
    return SpearmanrResult._make(
        batch=(k, k), info=info, statistic=_sd.reshape(r, (k, k)), pvalue=_sd.reshape(p, (k, k))
    )


# ---- Kendall ------------------------------------------------------------------------------


def kendalltau(
    x, y, /, *, axis=None, method="auto", variant="b", alternative="two-sided", nan_policy="propagate", keepdims=False
):
    """Kendall's tau: statistic tau-b (variant='b', ties adjusted) or tau-c (variant='c', Stuart's),
    pvalue and method. method 'auto' (exact when there are no ties and n <= 33 or at most one
    pair is discordant, or concordant: scipy's rule), 'exact' (the distribution of the inversions;
    a ValueError with ties, as scipy's, and past its work cap) or 'asymptotic' (the normal
    approximation with scipy's tie-corrected variance). NaN where a sample is constant or for
    fewer than 2 pairs. Unpacks as (statistic, pvalue)."""
    name = "kendalltau"
    meth = C.method(method, name, ("exact", "asymptotic"))
    v = C.choice(variant, name, "variant", tuple(_VARIANT))
    alt = C.alternative(alternative, name)
    s = _paired(name, x, y, axis, nan_policy, keepdims)
    tau, p, codes, work, ties = _k.kendall(
        s.rows[0], s.rows[1], _VARIANT[v], _ALT[alt], _REQ[meth], _EXACT_CAP, s.policy == "omit"
    )
    if meth == "exact" and ties.size and bool(_sd.any(ties == 1.0)):
        raise ValueError(f"{name}(): method='exact' needs samples without ties; use method='asymptotic'")
    if meth == "exact" and work.size:
        w = float(_sd.max(work))
        if w > 0.0:
            C.check_exact_work(w, _EXACT_CAP, name)
    names = [_USED[int(c)] for c in codes.tolist()]
    one = not _shape(s) and not s.keepdims
    m = names[0] if one else _nest(names, _shape(s))
    info = s.info(alt, title=f"Kendall rank correlation test (tau-{v})")
    return KendalltauResult._make(batch=s.batch, info=info, statistic=s.out(tau), pvalue=s.out(p), method=m)


def _weigher_table(weigher, n, name):
    """The weights of the ranks 0..n-1: weigher(r) for each (scipy calls it on the ranks)."""
    out = []
    for r in range(n):
        w = weigher(r)
        try:
            out.append(float(w))
        except TypeError, ValueError:
            raise TypeError(f"{name}(): 'weigher' must return real numbers, not {_type_name(w)}") from None
    return _sd.asarray(out) if out else _sd.asarray([0.0])


def weightedtau(x, y, /, *, axis=None, rank=True, weigher=None, additive=True, nan_policy="propagate", keepdims=False):
    """Vigna's weighted tau: statistic and pvalue (always NaN: scipy defines none, H64). rank: True
    (the mean of the taus of the decreasing lexicographic ranks by (x, y) and by (y, x)), None (by
    (x, y)), False (the elements' positions), or a sample of ranks paired with x and y (their order
    only: scipy's dense codes). weigher: None (hyperbolic, 1/(r + 1)) or a callable of the rank r
    (0 the most important); additive: the weight of an exchange is the sum (True) or the product of
    the two weights. Unpacks as (statistic, pvalue)."""
    name = "weightedtau"
    add = C.flag(additive, name, "additive")
    if weigher is not None and not callable(weigher):
        raise TypeError(f"{name}(): 'weigher' must be None or a callable, not {_type_name(weigher)}")
    if rank is True:
        mode, extra = 0, None
    elif rank is None:
        mode, extra = 1, None
    elif rank is False:
        mode, extra = 2, None
    else:
        mode, extra = 2, rank
    if extra is None:
        s = _paired(name, x, y, axis, nan_policy, keepdims)
        ranks = None
        if mode == 2:
            n = s.n[0]
            ranks = _sd.broadcast_to(_sd.arange(float(n)), (s.lanes, n))
    else:
        s = C.Samples(name, [x, y, extra], ("x", "y", "rank"), axis, nan_policy, paired=True, keepdims=keepdims,
                      scan=False)  # fmt: skip
        ranks = s.rows[2]
        if ranks.size and not bool(_sd.all(_sd.isfinite(ranks))):
            raise ValueError(f"{name}(): 'rank' must hold finite numbers")
    table = None if weigher is None else _weigher_table(weigher, s.n[0], name)
    tau = _k.weightedtau(s.rows[0], s.rows[1], mode, table, ranks, add, s.policy == "omit")
    nan = _sd.full((s.lanes,), math.nan)
    return WeightedtauResult._make(batch=s.batch, info=s.info(), statistic=s.out(tau), pvalue=s.out(nan))


# ---- Somers' D ----------------------------------------------------------------------------


def somersd(x, y=None, /, *, alternative="two-sided"):
    """Somers' D(Y|X): statistic, pvalue (the normal approximation with the asymptotic standard
    error, scipy's) and table (int64: x's distinct values as rows, y's as columns, ascending).
    x and y are two rankings of the same length, or x is the contingency table itself (y None).
    NaN for a table of one row or column, or a zero standard error. Does not unpack (scipy's
    result does not)."""
    name = "somersd"
    alt = C.alternative(alternative, name)
    xa = _as_array(x, name, "x")
    if xa.ndim == 2:
        if y is not None:
            raise ValueError(f"{name}(): 'y' must be None when 'x' is a table")
        tab = xa + 0.0
        if tab.size:
            ok = _sd.isfinite(tab) & (tab >= 0.0) & (_sd.floor(tab) == tab)
            if not bool(_sd.all(ok)):
                raise ValueError(f"{name}(): 'x' must hold integer counts >= 0")
        if float(_sd.sum(_sd.where(tab != 0.0, 1.0, 0.0))) < 2.0:
            raise ValueError(f"{name}(): the table must hold at least two nonzero counts")
    elif xa.ndim == 1:
        if y is None:
            raise ValueError(f"{name}(): 'y' is required when 'x' is a 1-D sample (a 2-D 'x' is a table)")
        ya = _as_array(y, name, "y")
        if ya.ndim != 1:
            raise ValueError(f"{name}(): 'y' must be a 1-D sample as 'x', not {ya.ndim}-D")
        if xa.size != ya.size:
            raise ValueError(f"{name}(): the paired samples must agree in size, not {xa.size} and {ya.size}")
        u, v = xa + 0.0, ya + 0.0
        if bool(_sd.any(_sd.isnan(u))) or bool(_sd.any(_sd.isnan(v))):
            raise ValueError(f"{name}(): 'x' and 'y' must not contain NaN")
        tab, rows, cols = _k.crosstab(u, v, _TABLE_CAP)
        if tab is None:
            raise ValueError(
                f"{name}(): the table of {rows} x {cols} distinct values has more cells than its cap of "
                f"{_TABLE_CAP:.3g}"
            )
    else:
        raise ValueError(f"{name}(): 'x' must be a 1-D sample or a 2-D table, not {xa.ndim}-D")
    d, p = _k.somersd(tab, _ALT[alt])
    t = tab.astype(_sd.int64)
    return SomersdResult._make(info={"alternative": alt}, statistic=float(d), pvalue=float(p), table=t)


# ---- Chatterjee ---------------------------------------------------------------------------


def chatterjeexi(
    x, y, /, *, axis=None, y_continuous=False, method="asymptotic", nan_policy="propagate", keepdims=False
):
    """Chatterjee's xi correlation and the test of independence: statistic xi (y as a function of x;
    not symmetric) and pvalue (the normal approximation, one-sided: large xi). y_continuous=True
    takes Chatterjee's variance for a continuous y. Ties in x are broken by position (scipy breaks
    them by its sort, arbitrarily). method: 'asymptotic' or a PermutationMethod (H9: y's order
    permuted). NaN for fewer than 2 pairs or a constant y. Unpacks as (statistic, pvalue)."""
    name = "chatterjeexi"
    cont = C.flag(y_continuous, name, "y_continuous")
    perm = isinstance(method, R.PermutationMethod)
    if not perm:
        if not isinstance(method, str):
            R.check_method(method, name, R.PermutationMethod, ("asymptotic",))
        if method != "asymptotic":
            raise ValueError(f"{name}(): 'method' must be 'asymptotic' or a PermutationMethod, not {method!r}")
    s = _paired(name, x, y, axis, nan_policy, keepdims)
    xi, p = _k.chatterjee(s.rows[0], s.rows[1], cont, s.policy == "omit")
    info = s.info()
    if perm:  # H9: xi of y's permutations (x held), large xi extreme (scipy's)

        def stat(a, b):
            return chatterjeexi(a, b, axis=-1, y_continuous=cont).statistic

        p, pairs = R.method_pvalue(name, method, s, stat, "greater", ptype="pairings", held=1)
        info["pairs"] = pairs
    return ChatterjeexiResult._make(batch=s.batch, info=info, statistic=s.out(xi), pvalue=s.out(p))


# a wrong call shape in the library's form (API.md §2.1)
pearsonr = _calls.function(pearsonr)
spearmanrho = _calls.function(spearmanrho)
spearmanr = _calls.function(spearmanr)
pointbiserialr = _calls.function(pointbiserialr)
kendalltau = _calls.function(kendalltau)
weightedtau = _calls.function(weightedtau)
somersd = _calls.function(somersd)
chatterjeexi = _calls.function(chatterjeexi)
