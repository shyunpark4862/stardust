"""The tests of means and variances (milestone H1; docs/api/htest.md §2, docs/design/htest.md §3.2).

ttest_1samp, ttest_rel, ttest_ind (Student, Welch, Yuen's trimmed), ttest_ind_from_stats, f_oneway
(and Welch's one-way ANOVA, equal_var=False), alexandergovern, bartlett, levene and fligner:
scipy's tests with this library's conventions (positional data, keyword-only options,
axis=None, batches along the other axes, nan_policy, read-only results; decisions H3-H21 and
the H1 decisions in docs/design/htest.md §1.1).

The checks and the arrangement of the samples are here; the computation is one native call
(stardust._core.htest: htest/parametric.cpp) that returns every lane's results. The statistics
come from double-double means and sums of squares, so data with a large offset and nearly equal
variances keep their digits; the p-values are sd.stats' distributions (StudentT, F, ChiSquared)
called through their own methods. Degenerate lanes give NaN without warnings (H11).
"""

from .. import _calls
from .. import _core as _sd
from ..stats._descriptive import _as_array, _scalar_value, _type_name
from . import _common as C
from ._result import _TestResult

_k = _sd.htest
_st = _sd.stats
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_CENTERS = {"median": 0, "mean": 1, "trimmed": 2}
_INF = float("inf")


# ---- the result classes (H6: not exported) ------------------------------------------------


class _TtestBase(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")
    _CI = True


class Ttest1sampResult(_TtestBase):
    __slots__ = ()


class TtestRelResult(_TtestBase):
    __slots__ = ()


class TtestIndResult(_TtestBase):
    __slots__ = ()


class TtestIndFromStatsResult(_TtestBase):
    __slots__ = ()


class FOnewayResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df_num", "df_denom")
    _UNPACK = ("statistic", "pvalue")


class AlexandergovernResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")


class BartlettResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")


class LeveneResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df_num", "df_denom")
    _UNPACK = ("statistic", "pvalue")


class FlignerResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")


# ---- helpers ------------------------------------------------------------------------------


def _t_interval(est_hi, est_lo, se, df, alt, out):
    """The conf_int of a t test (scipy's: estimate -+ q se, q the t quantile; one-sided ends at
    -+inf): ci(alpha) gives (low, high) in the result's shape. A zero standard error gives the
    zero-width interval (est, est)."""

    def ci(alpha):
        p = alpha / 2.0 if alt == "two-sided" else alpha
        q = _st.studentt_iccdf(df, p)
        half = _sd.where(se == 0.0, 0.0, q * se)
        low = (est_hi - half) + est_lo
        high = (est_hi + half) + est_lo
        if alt == "less":
            low = _sd.full(low.shape, -_INF)
        elif alt == "greater":
            high = _sd.full(high.shape, _INF)
        return out(low), out(high)

    return ci


def _t_result(cls, s_out, batch, res, alt):
    stat, p, df, est_hi, est_lo, se = res
    return cls._make(
        batch=batch,
        ci=_t_interval(est_hi, est_lo, se, df, alt, s_out),
        statistic=s_out(stat),
        pvalue=s_out(p),
        df=s_out(df),
    )


def _per_lane(v, s, name, arg):
    """popmean as a (lanes,) float64 array: a real number, or an array that broadcasts to the
    result's shape (or, as scipy's, to it with the sample axes kept as length 1)."""
    sv = _scalar_value(v)
    if sv is not None:
        return _sd.full((s.lanes,), C.real(v, name, arg))
    a = _as_array(v, name, arg)
    for shape in (s.batch, s.keep_shape):
        try:
            b = _sd.broadcast_to(a, shape)
        except ValueError:
            continue
        return _sd.reshape(b + 0.0, (s.lanes,))
    raise ValueError(f"{name}(): '{arg}' cannot be broadcast to the result's shape {s.batch}")


# ---- the t tests --------------------------------------------------------------------------


def ttest_1samp(a, popmean, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """The one-sample t test of the mean of `a` against `popmean`: statistic, pvalue, df (n - 1)
    and conf_int(alpha=) (the mean's interval). Unpacks as (statistic, pvalue)."""
    name = "ttest_1samp"
    alt = C.alternative(alternative, name)
    s = C.Samples(name, [a], ("a",), axis, nan_policy, keepdims=keepdims, scan=False)
    mu = _per_lane(popmean, s, name, "popmean")
    res = _k.ttest(0, s.rows[0], None, mu, True, 0.0, _ALT[alt], s.policy == "omit")
    return _t_result(Ttest1sampResult, s.out, s.batch, res, alt)


def ttest_rel(a, b, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """The paired t test: the one-sample test of a - b (the differences as double-doubles: two
    large values' difference is exact). Fields as ttest_1samp's; conf_int is the mean difference's."""
    name = "ttest_rel"
    alt = C.alternative(alternative, name)
    s = C.Samples(name, [a, b], ("a", "b"), axis, nan_policy, paired=True, keepdims=keepdims, scan=False)
    res = _k.ttest(1, s.rows[0], s.rows[1], None, True, 0.0, _ALT[alt], s.policy == "omit")
    return _t_result(TtestRelResult, s.out, s.batch, res, alt)


def ttest_ind(
    a,
    b,
    /,
    *,
    axis=None,
    equal_var=True,
    alternative="two-sided",
    trim=0,
    method=None,
    nan_policy="propagate",
    keepdims=False,
):
    """The two-sample t test: Student's (equal_var=True), Welch's (False), with trim > 0 Yuen's
    trimmed test. statistic, pvalue, df and conf_int(alpha=) (the difference of the means, trimmed
    means for Yuen). method=None: the t distribution (resampling methods come with H9)."""
    name = "ttest_ind"
    C.flag(equal_var, name, "equal_var")
    alt = C.alternative(alternative, name)
    tr = C.real(trim, name, "trim")
    if not 0.0 <= tr < 0.5:
        raise ValueError(f"{name}(): 'trim' must be >= 0 and < 0.5")
    if method is not None:
        raise TypeError(f"{name}(): 'method' must be None, not {_type_name(method)}")
    s = C.Samples(name, [a, b], ("a", "b"), axis, nan_policy, keepdims=keepdims, scan=False)
    res = _k.ttest(2, s.rows[0], s.rows[1], None, equal_var, tr, _ALT[alt], s.policy == "omit")
    return _t_result(TtestIndResult, s.out, s.batch, res, alt)


def ttest_ind_from_stats(mean1, std1, nobs1, mean2, std2, nobs2, /, *, equal_var=True, alternative="two-sided"):
    """The two-sample t test from the samples' means, standard deviations (correction 1) and
    sizes: real numbers or arrays that broadcast together. statistic, pvalue, df and
    conf_int(alpha=) as ttest_ind's."""
    name = "ttest_ind_from_stats"
    C.flag(equal_var, name, "equal_var")
    alt = C.alternative(alternative, name)
    args = (mean1, std1, nobs1, mean2, std2, nobs2)
    names = ("mean1", "std1", "nobs1", "mean2", "std2", "nobs2")
    scalar = all(_scalar_value(v) is not None for v in args)
    arrays = []
    for v, arg in zip(args, names):
        if _scalar_value(v) is not None:
            arrays.append(_sd.asarray([C.real(v, name, arg)]))
        else:
            arrays.append(_as_array(v, name, arg))
    try:
        arrays = _sd.broadcast_arrays(*arrays)
    except ValueError:
        shapes = ", ".join(str(tuple(x.shape)) for x in arrays)
        raise ValueError(f"{name}(): the arguments could not be broadcast together: {shapes}") from None
    batch = () if scalar else tuple(arrays[0].shape)
    lanes = 1
    for d in batch:
        lanes *= d
    flat = [_sd.reshape(x + 0.0, (lanes,)) for x in arrays]

    def out(v):
        return float(v[0]) if scalar else _sd.reshape(v, batch)

    res = _k.ttest_stats(*flat, equal_var, _ALT[alt])
    return _t_result(TtestIndFromStatsResult, out, batch, res, alt)


# ---- k samples ----------------------------------------------------------------------------

_ANOVA, _WELCH, _AG, _BARTLETT, _LEVENE, _FLIGNER = range(6)


def _groups(name, samples, axis, nan_policy, keepdims, kind, center=0, ptc=0.0):
    if len(samples) < 2:
        raise TypeError(f"{name}(): at least two samples are required, not {len(samples)}")
    args = tuple(f"samples[{i}]" for i in range(len(samples)))
    s = C.Samples(name, list(samples), args, axis, nan_policy, keepdims=keepdims, scan=False)
    return s, _k.kgroups(kind, s.rows, s.policy == "omit", center, ptc)


def _center(center, proportiontocut, name):
    C.choice(center, name, "center", tuple(_CENTERS))
    ptc = C.real(proportiontocut, name, "proportiontocut")
    if not 0.0 <= ptc < 0.5:
        raise ValueError(f"{name}(): 'proportiontocut' must be >= 0 and < 0.5")
    return _CENTERS[center], ptc


def f_oneway(*samples, axis=None, equal_var=True, nan_policy="propagate", keepdims=False):
    """The one-way analysis of variance of two or more samples (equal_var=False: Welch's, R's
    oneway.test(var.equal=FALSE)): statistic (F), pvalue, df_num and df_denom. Unpacks as
    (statistic, pvalue)."""
    name = "f_oneway"
    C.flag(equal_var, name, "equal_var")
    s, (stat, p, dfn, dfd) = _groups(name, samples, axis, nan_policy, keepdims, _ANOVA if equal_var else _WELCH)
    return FOnewayResult._make(
        batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df_num=s.out(dfn), df_denom=s.out(dfd)
    )


def alexandergovern(*samples, axis=None, nan_policy="propagate", keepdims=False):
    """Alexander and Govern's test of equal means without equal variances: statistic (A), pvalue
    and df (chi^2 with k - 1 degrees of freedom, H29). Does not unpack (scipy's result does not)."""
    name = "alexandergovern"
    s, (stat, p, df, _) = _groups(name, samples, axis, nan_policy, keepdims, _AG)
    return AlexandergovernResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df=s.out(df))


def bartlett(*samples, axis=None, nan_policy="propagate", keepdims=False):
    """Bartlett's test of equal variances: statistic (T), pvalue and df (chi^2, k - 1; H29).
    Unpacks as (statistic, pvalue)."""
    name = "bartlett"
    s, (stat, p, df, _) = _groups(name, samples, axis, nan_policy, keepdims, _BARTLETT)
    return BartlettResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df=s.out(df))


def levene(*samples, axis=None, center="median", proportiontocut=0.05, nan_policy="propagate", keepdims=False):
    """Levene's test of equal variances (center='median': Brown and Forsythe's): the one-way
    ANOVA of |x - center|; statistic (W), pvalue, df_num and df_denom (F with k - 1 and N - k;
    H29). Unpacks as (statistic, pvalue)."""
    name = "levene"
    c, ptc = _center(center, proportiontocut, name)
    s, (stat, p, dfn, dfd) = _groups(name, samples, axis, nan_policy, keepdims, _LEVENE, c, ptc)
    return LeveneResult._make(
        batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df_num=s.out(dfn), df_denom=s.out(dfd)
    )


def fligner(*samples, axis=None, center="median", proportiontocut=0.05, nan_policy="propagate", keepdims=False):
    """The Fligner-Killeen test of equal variances: the normal scores of the ranks of
    |x - center|; statistic, pvalue and df (chi^2, k - 1; H29). Unpacks as (statistic, pvalue)."""
    name = "fligner"
    c, ptc = _center(center, proportiontocut, name)
    s, (stat, p, df, _) = _groups(name, samples, axis, nan_policy, keepdims, _FLIGNER, c, ptc)
    return FlignerResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df=s.out(df))


# a wrong call shape in the library's form, "ttest_ind(): unexpected keyword argument
# 'equal_variance'" (API.md §2.1)
ttest_1samp = _calls.function(ttest_1samp)
ttest_rel = _calls.function(ttest_rel)
ttest_ind = _calls.function(ttest_ind)
ttest_ind_from_stats = _calls.function(ttest_ind_from_stats)
f_oneway = _calls.function(f_oneway)
alexandergovern = _calls.function(alexandergovern)
bartlett = _calls.function(bartlett)
levene = _calls.function(levene)
fligner = _calls.function(fligner)
