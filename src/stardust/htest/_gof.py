"""The tests of goodness of fit to a given distribution (milestone H2; docs/api/htest.md §3,
docs/design/htest.md §3.3).

chisquare and power_divergence (the Cressie-Read family), D'Agostino's skewtest, kurtosistest
and normaltest, jarque_bera, shapiro (Royston's AS R94), anderson, ks_1samp and cramervonmises:
scipy's tests with this library's conventions (positional data, keyword-only options,
axis=None, batches along the other axes, nan_policy, read-only results; decisions H3-H21 and
the H2 decisions in docs/design/htest.md §1.1).

The checks and the arrangement of the samples are here; the computation is one native call
(stardust._core.htest: htest/gof.cpp) that returns every lane's results. ks_1samp and
cramervonmises call the given cdf once, on the sorted samples, here: an sd.stats distribution
object (its cdf) or a callable cdf(x); scipy's distribution names ('norm') and args=() are not
taken (H35). Degenerate lanes give NaN without warnings (H11).

Jarque-Bera's kernel is shared with sd.regression (H17): r.jarque_bera() takes its statistic from
_jb_of_sums on the sums of its whitened residuals, the same function this module's jarque_bera
uses on a sample's.
"""

import math

from .. import _calls
from .. import _core as _sd
from ..stats import _distribution as _dist
from ..stats._descriptive import _as_array, _scalar_value, _type_name
from . import _common as C
from ._result import _TestResult

_k = _sd.htest
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_LAMBDA = {
    "pearson": 1.0,
    "log-likelihood": 0.0,
    "freeman-tukey": -0.5,
    "mod-log-likelihood": -1.0,
    "neyman": -2.0,
    "cressie-read": 2.0 / 3.0,
}
_RTOL = 2.0**-26  # scipy's sum_check tolerance, sqrt(eps)
_KS_METHODS = {"exact": 0, "approx": 1, "asymp": 2}


# ---- the result classes (H6: not exported; JarqueBeraResult is the one H18 names) ---------


class _Pair(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


class ChisquareResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")


class PowerDivergenceResult(ChisquareResult):
    __slots__ = ()


class SkewtestResult(_Pair):
    __slots__ = ()


class KurtosistestResult(_Pair):
    __slots__ = ()


class NormaltestResult(_Pair):
    __slots__ = ()


class JarqueBeraResult(_Pair):
    __slots__ = ()


class ShapiroResult(_Pair):
    __slots__ = ()


class AndersonResult(_Pair):
    __slots__ = ()


class Ks1sampResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue", "statistic_location", "statistic_sign", "method")
    _UNPACK = ("statistic", "pvalue")


class CramervonmisesResult(_TestResult):
    __slots__ = ()
    _FIELDS = ("statistic", "pvalue")


# ---- chisquare and power_divergence -------------------------------------------------------


def _lambda(v, name):
    if v is None:
        return 1.0
    if isinstance(v, str):
        return _LAMBDA[C.choice(v, name, "lambda_", tuple(_LAMBDA))]
    if _scalar_value(v) is None or isinstance(v, bool):
        names = ", ".join(repr(k) for k in _LAMBDA)
        raise TypeError(f"{name}(): 'lambda_' must be None, a real number or one of {names}, not {_type_name(v)}")
    return C.real(v, name, "lambda_")


def _divergence(name, cls, f_obs, f_exp, axis, ddof, lam, sum_check, nan_policy, keepdims):
    ddof = C.integer(ddof, name, "ddof")
    C.flag(sum_check, name, "sum_check")
    if f_exp is None:
        s = C.Samples(name, [f_obs], ("f_obs",), axis, nan_policy, keepdims=keepdims, scan=False)
        exp_rows = None
    else:
        a, b = _as_array(f_obs, name, "f_obs"), _as_array(f_exp, name, "f_exp")
        try:
            a, b = _sd.broadcast_arrays(a, b)
        except ValueError:
            raise ValueError(
                f"{name}(): 'f_obs' and 'f_exp' could not be broadcast together: {tuple(a.shape)}, {tuple(b.shape)}"
            ) from None
        s = C.Samples(name, [a, b], ("f_obs", "f_exp"), axis, nan_policy, paired=True, keepdims=keepdims, scan=False)
        exp_rows = s.rows[1]
    stat, p, df, rel = _k.power_divergence(s.rows[0], exp_rows, lam, float(ddof), s.policy == "omit")
    if f_exp is not None and sum_check and rel.size:
        worst = float(_sd.max(rel))
        if worst > _RTOL:
            raise ValueError(
                f"{name}(): the sums of 'f_obs' and 'f_exp' must agree to a relative tolerance of {_RTOL:.3g} "
                f"in each sample, not {worst:.3g} (sum_check)"
            )
    return cls._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p), df=s.out(df))


def chisquare(f_obs, f_exp=None, /, *, axis=None, ddof=0, sum_check=True, nan_policy="propagate", keepdims=False):
    """Pearson's chi-squared test of observed frequencies against expected ones (f_exp=None: their
    mean): statistic sum (O - E)^2 / E, pvalue (chi^2 with k - 1 - ddof degrees of freedom) and
    df. sum_check: the sums of f_obs and f_exp agree to scipy's relative 2^-26. Unpacks as
    (statistic, pvalue)."""
    name = "chisquare"
    return _divergence(name, ChisquareResult, f_obs, f_exp, axis, ddof, 1.0, sum_check, nan_policy, keepdims)


def power_divergence(f_obs, f_exp=None, /, *, axis=None, ddof=0, lambda_=None, nan_policy="propagate", keepdims=False):
    """The Cressie-Read power divergence statistic 2/(lambda (lambda + 1)) sum O ((O/E)^lambda - 1)
    and its chi^2 test: lambda_ a real number or scipy's name ('pearson' 1, 'log-likelihood' 0
    (the G test), 'freeman-tukey' -1/2, 'mod-log-likelihood' -1, 'neyman' -2, 'cressie-read'
    2/3); None is Pearson's. statistic, pvalue, df; unpacks as (statistic, pvalue). The sums of
    f_obs and f_exp must agree as chisquare's sum_check (scipy's power_divergence always checks)."""
    name = "power_divergence"
    lam = _lambda(lambda_, name)
    return _divergence(name, PowerDivergenceResult, f_obs, f_exp, axis, ddof, lam, True, nan_policy, keepdims)


# ---- the moment tests ---------------------------------------------------------------------


def _one(name, x, arg, axis, nan_policy, keepdims):
    return C.Samples(name, [x], (arg,), axis, nan_policy, keepdims=keepdims, scan=False)


def _dagostino(kind, cls, name, a, axis, alternative, nan_policy, keepdims):
    alt = C.alternative(alternative, name)
    s = _one(name, a, "a", axis, nan_policy, keepdims)
    stat, p = _k.dagostino(kind, s.rows[0], _ALT[alt], s.policy == "omit")
    return cls._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


def skewtest(a, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """D'Agostino's test of zero skewness: statistic Z (the normal transform of the sample skewness)
    and pvalue (standard normal). n < 8 gives NaN. Unpacks as (statistic, pvalue)."""
    return _dagostino(0, SkewtestResult, "skewtest", a, axis, alternative, nan_policy, keepdims)


def kurtosistest(a, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """Anscombe and Glynn's test of the normal kurtosis: statistic Z and pvalue (standard normal).
    n < 5 gives NaN. Unpacks as (statistic, pvalue)."""
    return _dagostino(1, KurtosistestResult, "kurtosistest", a, axis, alternative, nan_policy, keepdims)


def normaltest(a, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """D'Agostino and Pearson's omnibus test of normality: statistic K^2 = Z_skew^2 + Z_kurt^2 and
    pvalue (chi^2 with 2 degrees of freedom). n < 8 gives NaN. Unpacks as (statistic, pvalue)."""
    name = "normaltest"
    s = _one(name, a, "a", axis, nan_policy, keepdims)
    stat, p = _k.dagostino(2, s.rows[0], 0, s.policy == "omit")
    return NormaltestResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


def jarque_bera(x, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """The Jarque-Bera test of normality: statistic n/6 (S^2 + (K - 3)^2 / 4) from the central
    moments (sums in double-double) and pvalue ChiSquared(df=2).ccdf (H17: sd.regression's
    r.jarque_bera() is this kernel on its whitened residuals). NaN for a constant sample (H20).
    Unpacks as (statistic, pvalue)."""
    name = "jarque_bera"
    s = _one(name, x, "x", axis, nan_policy, keepdims)
    stat, p = _k.jarque_bera(s.rows[0], s.policy == "omit")
    return JarqueBeraResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


def _jb_of_sums(n, s2, s3, s4, mean):
    """Jarque-Bera's (statistic, pvalue) from the count, the central sums of powers 2, 3, 4 and the
    mean (sd.regression's r.jarque_bera()): Python floats, or arrays of their broadcast shape."""
    vals = (n, s2, s3, s4, mean)
    if all(not isinstance(v, _sd.ndarray) for v in vals):
        stat, p = _k.jb_of_sums(*(_sd.asarray([float(v)]) for v in vals))
        return float(stat[0]), float(p[0])
    arrays = _sd.broadcast_arrays(*(_sd.asarray(v) + 0.0 for v in vals))
    shape = tuple(arrays[0].shape)
    size = 1
    for d in shape:
        size *= d
    stat, p = _k.jb_of_sums(*(_sd.reshape(a, (size,)) for a in arrays))
    return _sd.reshape(stat, shape), _sd.reshape(p, shape)


def shapiro(x, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """The Shapiro-Wilk test of normality by Royston's algorithm AS R94 (scipy's and R's): statistic
    W and pvalue (Royston's normalizing transform). n < 3 gives NaN. Unpacks as (statistic,
    pvalue)."""
    name = "shapiro"
    s = _one(name, x, "x", axis, nan_policy, keepdims)
    stat, p = _k.shapiro(s.rows[0], s.policy == "omit")
    return ShapiroResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


_ANDERSON = None


def _anderson_dists():
    global _ANDERSON
    if _ANDERSON is None:
        from .. import stats

        _ANDERSON = {stats.Normal: 0, stats.Exponential: 1, stats.Logistic: 2, stats.Gumbel: 3}
    return _ANDERSON


def anderson(x, /, *, axis=None, dist=None, method="interpolate", nan_policy="propagate", keepdims=False):
    """The Anderson-Darling test that x comes from a family of distributions with fitted parameters:
    dist the sd.stats class (None or sd.stats.Normal: mean and standard deviation; Exponential:
    the mean, location 0; Logistic and Gumbel (the maximum's): maximum likelihood). statistic A^2
    and pvalue interpolated in scipy's tables of critical values (method='interpolate'; Stephens),
    which lies in the table's range of levels. Unpacks as (statistic, pvalue)."""
    name = "anderson"
    dists = _anderson_dists()
    names = "sd.stats.Normal, sd.stats.Exponential, sd.stats.Logistic or sd.stats.Gumbel"
    if dist is None:
        code = 0
    elif isinstance(dist, type) and dist in dists:
        code = dists[dist]
    elif isinstance(dist, type) and issubclass(dist, _dist.ContinuousDistribution):
        raise ValueError(f"{name}(): 'dist' must be {names}, not sd.stats.{dist.__name__}")
    elif isinstance(dist, _dist.ContinuousDistribution):
        raise TypeError(
            f"{name}(): 'dist' must be a distribution class ({names}), not an instance of {type(dist).__name__}"
        )
    else:
        raise TypeError(f"{name}(): 'dist' must be a distribution class ({names}), not {_type_name(dist)}")
    if not isinstance(method, str):
        raise TypeError(f"{name}(): 'method' must be 'interpolate', not {_type_name(method)}")
    if method != "interpolate":
        raise ValueError(f"{name}(): 'method' must be 'interpolate', not {method!r}")
    s = _one(name, x, "x", axis, nan_policy, keepdims)
    stat, p = _k.anderson(s.rows[0], code, s.policy == "omit")
    return AndersonResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


# ---- the empirical distribution function tests -------------------------------------------


def _nest(flat, shape):
    """flat (a list) as nested lists of the shape."""
    if len(shape) <= 1 or shape[0] == 0:
        return list(flat)
    step = len(flat) // shape[0]
    return [_nest(flat[i * step : (i + 1) * step], shape[1:]) for i in range(shape[0])]


def _cdf_function(cdf, name):
    """The cdf to call: an sd.stats distribution object's cdf, or a callable (H35)."""
    if isinstance(cdf, (_dist.ContinuousDistribution, _dist.DiscreteDistribution)):
        return cdf.cdf
    if isinstance(cdf, type):
        raise TypeError(
            f"{name}(): 'cdf' must be an sd.stats distribution object (sd.stats.Normal(mu=0, sigma=1)) or a "
            f"callable cdf(x), not the class {cdf.__name__}"
        )
    if callable(cdf):
        return cdf
    raise TypeError(
        f"{name}(): 'cdf' must be an sd.stats distribution object or a callable cdf(x), not {_type_name(cdf)}"
    )


def _sorted_with_cdf(name, x, arg, cdf, axis, nan_policy, keepdims):
    """The samples sorted along the lanes (NaNs last) and the cdf at them, as (lanes, n) rows; the
    cdf is called once, on the sorted samples in the batch's shape."""
    f = _cdf_function(cdf, name)
    s = _one(name, x, arg, axis, nan_policy, keepdims)
    rows = s.rows[0]
    xs = _sd.sort(rows, axis=-1) if rows.size else rows + 0.0
    shape = s.batch + (s.n[0],)
    got = f(_sd.reshape(xs, shape))
    if _scalar_value(got) is not None:
        raise ValueError(f"{name}(): 'cdf' must return an array of the shape of its input {shape}, not ()")
    try:
        vals = _sd.asarray(got, dtype=_sd.float64)
    except TypeError, ValueError:
        raise TypeError(f"{name}(): 'cdf' must return an array of numbers, not {_type_name(got)}") from None
    if tuple(vals.shape) != shape:
        raise ValueError(
            f"{name}(): 'cdf' must return an array of the shape of its input {shape}, not {tuple(vals.shape)}"
        )
    return s, xs, _sd.reshape(vals, tuple(rows.shape))


def ks_1samp(x, cdf, /, *, axis=None, alternative="two-sided", method="auto", nan_policy="propagate", keepdims=False):
    """The one-sample Kolmogorov-Smirnov test of x against a continuous distribution: cdf an sd.stats
    distribution object or a callable cdf(x). statistic D (D+ for 'greater', D- for 'less'),
    pvalue, statistic_location (the observation where D is reached), statistic_sign (+1 for D+,
    -1 for D-) and method, the distribution used: the two-sided test's 'exact' (also 'auto'),
    'approx' (twice the one-sided tail) or 'asymp' (Kolmogorov's limit); the one-sided tests
    always use Smirnov's exact distribution ('exact'). A batch's method is the lanes' names in
    nested lists of its shape (H48). Unpacks as (statistic, pvalue)."""
    name = "ks_1samp"
    alt = C.alternative(alternative, name)
    meth = C.method(method, name, tuple(_KS_METHODS))
    used = "exact" if (alt != "two-sided" or meth == "auto") else meth
    s, xs, fs = _sorted_with_cdf(name, x, "x", cdf, axis, nan_policy, keepdims)
    stat, p, loc, sign = _k.ks(xs, fs, _ALT[alt], _KS_METHODS[used], s.policy == "omit")
    sign = s.out(sign)
    sign = int(sign) if isinstance(sign, float) else sign.astype(_sd.int64)
    shape = s.keep_shape if s.keepdims else s.batch
    if shape or s.keepdims:  # a batch: the lanes' names, nested in its shape (H48)
        used = _nest([used] * math.prod(shape), shape)
    return Ks1sampResult._make(
        batch=s.batch,
        statistic=s.out(stat),
        pvalue=s.out(p),
        statistic_location=s.out(loc),
        statistic_sign=sign,
        method=used,
    )


def cramervonmises(rvs, cdf, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """The one-sample Cramér-von Mises test of rvs against a continuous distribution (cdf as
    ks_1samp's): statistic W^2 = 1/(12n) + sum ((2i - 1)/(2n) - F(x_(i)))^2 and pvalue by Csörgő
    and Faraway's finite-sample distribution (at least 0, as scipy's). n < 2 gives NaN. Does not
    unpack (scipy's result does not)."""
    name = "cramervonmises"
    s, xs, fs = _sorted_with_cdf(name, rvs, "rvs", cdf, axis, nan_policy, keepdims)
    stat, p = _k.cvm(xs, fs, s.policy == "omit")
    return CramervonmisesResult._make(batch=s.batch, statistic=s.out(stat), pvalue=s.out(p))


# a wrong call shape in the library's form (API.md §2.1)
chisquare = _calls.function(chisquare)
power_divergence = _calls.function(power_divergence)
skewtest = _calls.function(skewtest)
kurtosistest = _calls.function(kurtosistest)
normaltest = _calls.function(normaltest)
jarque_bera = _calls.function(jarque_bera)
shapiro = _calls.function(shapiro)
anderson = _calls.function(anderson)
ks_1samp = _calls.function(ks_1samp)
cramervonmises = _calls.function(cramervonmises)
