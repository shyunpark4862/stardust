"""The proportions and the binomial (milestone H7; docs/api/htest.md §8, docs/design/htest.md §3.8).

binomtest, quantile_test and poisson_means_test (scipy's) and proportions_ztest,
proportions_chisquare, proportion_confint, test_proportions_2indep and confint_proportions_2indep
(statsmodels'), with this library's conventions (positional data, keyword-only options,
read-only results; decisions H3-H21 and the H7 decisions H85-H98 in docs/design/htest.md §1.1).

The checks are here; the computation is one native call (stardust._core.htest:
htest/proportion.cpp). The method names of proportion_confint are scipy's where scipy has the
method ('exact' for statsmodels' 'beta', 'wilson', scipy's 'wilsoncc') and statsmodels' elsewhere
(H85); the alternatives are 'less' and 'greater' (H86): for proportion_confint 'less' is the
interval [0, upper] and 'greater' [lower, 1], scipy's direction (statsmodels' 'larger' is [0,
upper], H95). The defaults are statsmodels 0.15.0's, fixed (H90). The counts are non-negative
integers; statsmodels' functions broadcast them (a batch), proportions_ztest and
proportions_chisquare reading the samples along the last axis.
"""

import math

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from ..stats._descriptive import _as_array, _scalar_value
from . import _common as C
from ._result import _TestResult

_k = _sd.htest
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_CI = {"normal": 0, "agresti_coull": 1, "exact": 2, "wilson": 3, "wilsoncc": 4, "jeffreys": 5, "binom_test": 6}
_BINOMTEST_CI = ("exact", "wilson", "wilsoncc")  # scipy's proportion_ci
_COMPARE = {"diff": 0, "ratio": 1, "odds-ratio": 2}
_M2 = {
    "wald": 0,
    "agresti-caffo": 1,
    "score": 2,
    "log": 3,
    "log-adjusted": 4,
    "logit": 5,
    "logit-adjusted": 6,
    "logit-smoothed": 7,
    "newcomb": 8,
}
_TEST_METHODS = {
    "diff": ("wald", "agresti-caffo", "score"),
    "ratio": ("log", "log-adjusted", "score"),
    "odds-ratio": ("logit", "logit-adjusted", "logit-smoothed", "score"),
}
_CI_METHODS = {
    "diff": ("wald", "agresti-caffo", "newcomb", "score"),
    "ratio": ("log", "log-adjusted", "score"),
    "odds-ratio": ("logit", "logit-adjusted", "logit-smoothed", "score"),
}
# statsmodels 0.15.0's defaults, fixed (H90)
_TEST_DEFAULT = {"diff": "agresti-caffo", "ratio": "log-adjusted", "odds-ratio": "logit-adjusted"}
_CI_DEFAULT = {"diff": "newcomb", "ratio": "log-adjusted", "odds-ratio": "logit-adjusted"}


# ---- the result classes (H6: not exported) ------------------------------------------------


def _box(r, alpha, *, drop=(), alternative=None, more=()):
    """summary() of a result whose alternative is a field (or is given) and whose fields `drop`
    are shown otherwise (as info's pairs, `more`)."""
    info = dict(r._info)
    if alternative is not None:
        info["alternative"] = alternative
    info["pairs"] = list(info.get("pairs", [])) + list(more)
    fields = [(k, r._values[k]) for k in r._FIELDS if k in r._values and k not in drop]
    interval = None
    if alpha is not None:
        lo, hi = r._ci(alpha)
        interval = (lo, hi, alpha, r._CI_LABEL)
    return _S.test_box(info.get("title") or r._TITLE, fields, r._batch, info, interval)


class BinomtestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Exact binomial test"
    _CI_LABEL = "proportion"
    _FIELDS = ("k", "n", "alternative", "statistic", "pvalue")
    _CI = True

    @_calls.method("conf_int")
    def conf_int(self, *, alpha=0.05, method="exact"):
        """The 1 - alpha confidence interval of the proportion (scipy's proportion_ci with alpha for
        1 - confidence_level, H16): method 'exact' (Clopper-Pearson), 'wilson' or 'wilsoncc'
        (Wilson's with the continuity correction), one-sided as the test ('less' [0, upper],
        'greater' [lower, 1]). A new (2,) array."""
        a = C.alpha(alpha)
        m = C.choice(method, "conf_int", "method", _BINOMTEST_CI)
        return _sd.asarray(list(self._ci(a, m)))

    @_calls.method("summary")
    def summary(self, *, alpha=0.05):
        """The test as a text table (H74): the header and the 1 - alpha exact interval."""
        a = C.alpha(alpha, "summary")
        v = self._values
        more = [("No. Successes:", str(v["k"]))]
        return _box(self, a, drop=("k", "n", "alternative"), alternative=v["alternative"], more=more)


class QuantileTestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Quantile test"
    _CI_LABEL = "quantile"
    _FIELDS = ("statistic", "statistic_type", "pvalue")
    _CI = True

    @_calls.method("summary")
    def summary(self, *, alpha=0.05):
        """The test as a text table (H74): the header and the 1 - alpha interval of the quantile;
        a batch one row per lane (H76)."""
        a = C.alpha(alpha, "summary")
        if self._batch:
            return _box(self, a)
        t = self._values["statistic_type"]
        kind = "nan" if t != t else ("1 (x <= q)" if t == 1.0 else "2 (x < q)")
        return _box(self, a, drop=("statistic_type",), more=[("Statistic Type:", kind)])


class PoissonMeansTestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Poisson means test (E-test)"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


class ProportionsZtestResult(PoissonMeansTestResult):
    __slots__ = ()
    _TITLE = "Z-test of proportions"


class ProportionsChisquareResult(_TestResult):
    """Unpacks as statsmodels' (chi2, p, (table, expected)) (H93)."""

    __slots__ = ()
    _TITLE = "Chi-squared test of proportions"
    _FIELDS = ("statistic", "pvalue", "df", "table", "expected_freq")
    _TABLES = ("table", "expected_freq")

    def _parts(self):
        return (self.statistic, self.pvalue, (self.table, self.expected_freq))

    def __iter__(self):
        return iter(self._parts())

    def __len__(self):
        return 3

    def __getitem__(self, i):
        return self._parts()[i]


class TestProportions2indepResult(_TestResult):
    __slots__ = ()
    __test__ = False  # (not a pytest test class)
    _TITLE = "Test of two independent proportions"
    _FIELDS = (
        "statistic",
        "pvalue",
        "compare",
        "method",
        "diff",
        "ratio",
        "odds_ratio",
        "variance",
        "alternative",
        "value",
        "prop1_null",
        "prop2_null",
    )
    _UNPACK = ("statistic", "pvalue")


# ---- helpers ------------------------------------------------------------------------------


def _counts(v, name, arg):
    """A count argument: an array of non-negative integer values (integer-valued floats too)."""
    a = _as_array(v, name, arg)
    if a.size:
        ok = _sd.isfinite(a) & (a >= 0.0) & (_sd.floor(a) == a)
        if not bool(_sd.all(ok)):
            raise ValueError(f"{name}(): '{arg}' must hold non-negative integer counts")
    return a


def _broadcast(name, arrays, args):
    try:
        return list(_sd.broadcast_arrays(*arrays))
    except ValueError:
        listed = ", ".join(f"'{a}'" for a in args)
        shapes = ", ".join(str(tuple(a.shape)) for a in arrays)
        raise ValueError(f"{name}(): {listed} could not be broadcast together: {shapes}") from None


def _scalars(*vs):
    """Every input a real scalar: one test (Python numbers out), not a batch of one."""
    return all(_scalar_value(v) is not None for v in vs)


_COUNT_LIMIT = 2**53  # the trials of binomtest and proportion_confint: exact float64 integers (H151)


def _below_limit(n, name, arg):
    if n.size and bool(_sd.any(n >= float(_COUNT_LIMIT))):
        raise ValueError(f"{name}(): '{arg}' must be below 2**53")


def _not_above(c, n, name, carg, narg):
    if c.size and bool(_sd.any(c > n)):
        raise ValueError(f"{name}(): '{carg}' must not exceed '{narg}'")


def _flat(a):
    return _sd.reshape(a, (a.size,))


def _out(v, shape):
    """A per-lane result: a Python float for one lane (shape ()), else the batch's shape."""
    if not shape:
        return float(v[0])
    return _sd.reshape(v, shape)


def _pair(lo, hi, shape):
    """An interval's (..., 2) array of the lower and upper ends ((2,) for one)."""
    if not shape:
        return _sd.asarray([float(lo[0]), float(hi[0])])
    return _sd.stack([_sd.reshape(lo, shape), _sd.reshape(hi, shape)], axis=-1)


def _proportion(v, name, arg, *, low_open=False, high_open=False):
    """A probability option in [0, 1] (open ends where asked)."""
    p = C.real(v, name, arg)
    ok = (p > 0.0 if low_open else p >= 0.0) and (p < 1.0 if high_open else p <= 1.0)
    if not ok:
        lo, hi = ("(" if low_open else "["), (")" if high_open else "]")
        raise ValueError(f"{name}(): '{arg}' must be in {lo}0, 1{hi}")
    return p


def _samples(name, count, nobs):
    """statsmodels' count and nobs of k samples: broadcast, the samples along the last axis (a
    scalar is one sample); (count, nobs) as (lanes, k) rows, the batch shape, k."""
    c, n = _broadcast(name, [_counts(count, name, "count"), _counts(nobs, name, "nobs")], ("count", "nobs"))
    _not_above(c, n, name, "count", "nobs")
    if c.ndim == 0:
        c, n = _sd.reshape(c, (1,)), _sd.reshape(n, (1,))
    k = int(c.shape[-1])
    if k == 0:
        raise ValueError(f"{name}(): 'count' must hold at least one sample")
    batch = tuple(c.shape[:-1])
    lanes = math.prod(batch)
    return _sd.reshape(c, (lanes, k)), _sd.reshape(n, (lanes, k)), batch, k


# ---- scipy's binomtest, quantile_test, poisson_means_test ---------------------------------


def binomtest(k, n, /, *, p=0.5, alternative="two-sided"):
    """The exact binomial test that the probability of success is p: k successes of n trials
    (ints, 0 <= k <= n, 1 <= n < 2**53: H151), p in [0, 1]. statistic k / n, pvalue the binomial tail
    ('less' P(X <= k), 'greater' P(X >= k), two-sided scipy's rule: the far side's outcomes at
    most (1 + 1e-7) times as probable as k, R's binom.test's too), and conf_int(*, alpha=0.05,
    method='exact') (scipy's proportion_ci: 'exact', 'wilson', 'wilsoncc'). Fields k n alternative
    statistic pvalue; does not unpack (scipy's result does not)."""
    name = "binomtest"
    kk = C.integer(k, name, "k", minimum=0)
    nn = C.integer(n, name, "n", minimum=1)
    if nn >= _COUNT_LIMIT:
        raise ValueError(f"{name}(): 'n' must be below 2**53")
    if kk > nn:
        raise ValueError(f"{name}(): 'k' must not exceed 'n'")
    pp = _proportion(p, name, "p")
    alt = C.alternative(alternative, name)
    stat, pv = _k.binomtest(float(kk), float(nn), pp, _ALT[alt])

    def ci(a, method="exact"):
        lo, hi = _k.binom_ci(_sd.asarray([float(kk)]), _sd.asarray([float(nn)]), a, _CI[method], _ALT[alt])
        return float(lo[0]), float(hi[0])

    info = {"nobs": nn, "pairs": [("Null Proportion:", f"{pp:g}")]}
    return BinomtestResult._make(ci=ci, info=info, k=kk, n=nn, alternative=alt, statistic=stat, pvalue=pv)


def quantile_test(x, /, *, axis=None, q=0.0, p=0.5, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """The test that q is the p-th quantile of x's population (scipy's): T1 = #(x <= q), T2 =
    #(x < q) of the n values and Y ~ Binomial(n, p); 'less' (the quantile below q) P(Y >= T2)
    with statistic T2 and statistic_type 2, 'greater' P(Y <= T1) with T1 and 1, two-sided twice
    the smaller of the two (at most 1) with its count. conf_int(*, alpha=0.05): the order
    statistics x[ppf(a) - 1] and x[isf(a)] of Y (a = alpha / 2, or alpha one-sided; NaN past the
    sample, -inf or inf on a one-sided test's open side). p in (0, 1). Does not unpack."""
    name = "quantile_test"
    qq = C.real(q, name, "q")
    pp = _proportion(p, name, "p", low_open=True, high_open=True)
    alt = C.alternative(alternative, name)
    s = C.Samples(name, [x], ("x",), axis, nan_policy, keepdims=keepdims, scan=False)
    omit = s.policy == "omit"
    stat, kind, pv = _k.quantile_test(s.rows[0], qq, pp, _ALT[alt], omit)
    rows = s.rows[0] + 0.0  # (the interval is the sample's: kept, not the caller's array)

    def ci(a):
        lo, hi = _k.quantile_ci(rows, pp, a, _ALT[alt], omit)
        return s.out(lo), s.out(hi)

    info = s.info(alt, pairs=[("Quantile:", f"{qq:g}"), ("Probability:", f"{pp:g}")])
    return QuantileTestResult._make(
        batch=s.batch,
        ci=ci,
        info=info,
        statistic=s.out(stat),
        statistic_type=s.out(kind),
        pvalue=s.out(pv),
    )


_POISSON_STEPS = 2e7  # poisson_means_test's x2 steps at most (csrc/htest/proportion.cpp kPoissonSteps, H154)


def _positive(v, name, arg):
    r = C.real(v, name, arg)
    if not (r > 0.0 and math.isfinite(r)):
        raise ValueError(f"{name}(): '{arg}' must be positive and finite")
    return r


def poisson_means_test(k1, n1, k2, n2, /, *, diff=0, alternative="two-sided"):
    """The E-test of Krishnamoorthy and Thomson (scipy's) that two Poisson rates differ by diff:
    k1 and k2 the counts (ints >= 0) in sizes n1 and n2 (> 0). statistic T = (k1/n1 - k2/n2 -
    diff) / sqrt(k1/n1^2 + k2/n2^2), pvalue the null probability (rates estimated under the null)
    of the outcomes with T at least as extreme, summed over all outcomes (scipy truncates the sums;
    the tail left out here is below 2^-60 of the sum; past 2e7 steps of it a ValueError, H154). A
    null estimate <= 0 gives (0, 1) (scipy's).
    Unpacks as (statistic, pvalue)."""
    name = "poisson_means_test"
    a1 = C.integer(k1, name, "k1", minimum=0)
    m1 = _positive(n1, name, "n1")
    a2 = C.integer(k2, name, "k2", minimum=0)
    m2 = _positive(n2, name, "n2")
    d = C.real(diff, name, "diff")
    if not (d >= 0.0 and math.isfinite(d)):
        raise ValueError(f"{name}(): 'diff' must be non-negative and finite")
    alt = C.alternative(alternative, name)
    stat, pv, capped = _k.poisson_means(float(a1), m1, float(a2), m2, d, _ALT[alt])
    if capped:
        raise ValueError(
            f"{name}(): the counts are too large for the exact sum (it needs more than {_POISSON_STEPS:.0e} steps)"
        )
    pairs = [("Counts:", f"{a1}, {a2}"), ("Sizes:", f"{m1:g}, {m2:g}")]
    if d:
        pairs.append(("Diff:", f"{d:g}"))
    return PoissonMeansTestResult._make(
        info={"alternative": alt, "pairs": pairs}, statistic=float(stat), pvalue=float(pv)
    )


# ---- statsmodels' proportions_ztest and proportions_chisquare -----------------------------


def proportions_ztest(count, nobs, /, *, value=None, alternative="two-sided", prop_var=None):
    """statsmodels' z test of one proportion or the difference of two: count and nobs (non-negative
    integers, count <= nobs) broadcast, the samples along the last axis (a scalar or a length-1
    axis: one sample; length 2: two) and a batch before it. One sample: (count/nobs - value) /
    sqrt(pp (1 - pp) / nobs), value required; two: (p1 - p2 - value) / sqrt(pp (1 - pp) (1/n1 +
    1/n2)), value 0 by default; pp the sample (pooled) proportion, or prop_var (in [0, 1]). A zero
    standard error gives NaN (H24). Unpacks as (statistic, pvalue)."""
    name = "proportions_ztest"
    alt = C.alternative(alternative, name)
    rc, rn, batch, k = _samples(name, count, nobs)
    if k > 2:
        raise ValueError(f"{name}(): 'count' must hold one or two samples along its last axis, not {k}")
    if value is None:
        if k == 1:
            raise ValueError(f"{name}(): 'value' must be given for a one-sample test")
        v = 0.0
    else:
        v = C.real(value, name, "value")
    pv = math.nan if prop_var is None else _proportion(prop_var, name, "prop_var")
    z, p = _k.prop_ztest(rc, rn, v, pv, _ALT[alt])
    pairs = [("Value:", f"{v:g}")]
    if prop_var is not None:
        pairs.append(("Prop Var:", f"{pv:g}"))
    info = {"alternative": alt, "pairs": pairs}
    if not batch:
        info["nobs"] = int(rn[0, 0]) if k == 1 else (int(rn[0, 0]), int(rn[0, 1]))
    title = "Z-test of one proportion" if k == 1 else "Z-test of two proportions"
    info["title"] = title
    return ProportionsZtestResult._make(batch=batch, info=info, statistic=_out(z, batch), pvalue=_out(p, batch))


def proportions_chisquare(count, nobs, /, *, value=None):
    """statsmodels' chi-squared test of k proportions: count and nobs broadcast, the samples along
    the last axis (a batch before it). table is the k x 2 table (count, nobs - count). value=None:
    the test of equal proportions, chi2_contingency(table, correction=False) (df = k - 1; no Yates
    correction, statsmodels', H92); a value in [0, 1]: the test against it, expected (nobs value,
    nobs (1 - value)), df = k. Fields statistic pvalue df table expected_freq; unpacks as
    statsmodels' (statistic, pvalue, (table, expected_freq)) (H93)."""
    name = "proportions_chisquare"
    rc, rn, batch, k = _samples(name, count, nobs)
    table = _sd.stack([rc, rn - rc], axis=-1)  # (lanes, k, 2)
    if value is None:
        stat, p, df, exp = _k.chi2_contingency(table, False, 1.0)
        pairs = []
    else:
        v = _proportion(value, name, "value")
        stat, p, df, exp = _k.prop_chisquare(rc, rn, v)
        pairs = [("Value:", f"{v:g}")]
    pairs.insert(0, ("No. Samples:", str(k)))
    return ProportionsChisquareResult._make(
        batch=batch,
        info={"pairs": pairs},
        statistic=_out(stat, batch),
        pvalue=_out(p, batch),
        df=_out(df, batch),
        table=_sd.reshape(table, batch + (k, 2)),
        expected_freq=_sd.reshape(exp, batch + (k, 2)),
    )


def proportion_confint(count, nobs, /, *, alpha=0.05, method="normal", alternative="two-sided"):
    """The 1 - alpha confidence interval of a binomial proportion (statsmodels'), count and nobs
    broadcast: a new (..., 2) array of the lower and upper ends ((2,) for one; it unpacks as
    statsmodels' (low, upp)). method (H85): 'normal' (Wald), 'agresti_coull', 'exact'
    (Clopper-Pearson; statsmodels' 'beta'), 'wilson', 'wilsoncc' (scipy's, with the continuity
    correction), 'jeffreys', 'binom_test' (the two-sided binomial test inverted: the ends of the
    acceptance region about count / nobs, alpha not halved). Two-sided intervals put alpha / 2 in
    each tail; 'less' gives [0, upper] and 'greater' [lower, 1] with alpha in the one tail (H95).
    'normal', 'agresti_coull' and 'wilson' are clipped to [0, 1]. nobs = 0 gives NaN; nobs must be
    below 2**53 (H151)."""
    name = "proportion_confint"
    a = C.alpha(alpha, name)
    meth = C.choice(method, name, "method", tuple(_CI))
    alt = C.alternative(alternative, name)
    c, n = _broadcast(name, [_counts(count, name, "count"), _counts(nobs, name, "nobs")], ("count", "nobs"))
    _not_above(c, n, name, "count", "nobs")
    _below_limit(n, name, "nobs")
    shape = () if _scalars(count, nobs) else tuple(c.shape)
    lo, hi = _k.binom_ci(_flat(c), _flat(n), a, _CI[meth], _ALT[alt])
    return _pair(lo, hi, shape)


# ---- two independent proportions ----------------------------------------------------------


def _two(name, count1, nobs1, count2, nobs2):
    args = ("count1", "nobs1", "count2", "nobs2")
    arrays = [_counts(v, name, a) for v, a in zip((count1, nobs1, count2, nobs2), args)]
    c1, n1, c2, n2 = _broadcast(name, arrays, args)
    _not_above(c1, n1, name, "count1", "nobs1")
    _not_above(c2, n2, name, "count2", "nobs2")
    shape = () if _scalars(count1, nobs1, count2, nobs2) else tuple(c1.shape)
    return shape, [_flat(a) for a in (c1, n1, c2, n2)]


def _compare_method(name, compare, method, methods, default):
    cmp = C.choice(compare, name, "compare", tuple(_COMPARE))
    meth = default[cmp] if method is None else C.choice(method, name, "method", methods[cmp])
    return cmp, meth


def test_proportions_2indep(
    count1,
    nobs1,
    count2,
    nobs2,
    /,
    *,
    value=None,
    method=None,
    compare="diff",
    alternative="two-sided",
    correction=True,
):
    """statsmodels' test of two independent proportions (count1 of nobs1 against count2 of nobs2,
    broadcast): compare 'diff' (p1 - p2, value 0 by default; method 'wald', 'agresti-caffo' the
    default, 'score'), 'ratio' (p1 / p2, value 1; 'log', 'log-adjusted' the default, 'score') or
    'odds-ratio' ('logit', 'logit-adjusted' the default, 'logit-smoothed', 'score'); the score
    tests (Miettinen-Nurminen) with the correction N / (N - 1) of the variance when correction.
    Fields statsmodels' (H88): statistic pvalue compare method diff ratio odds_ratio variance
    alternative value, and prop1_null prop2_null for 'score'; unpacks as (statistic, pvalue)."""
    name = "test_proportions_2indep"
    cmp, meth = _compare_method(name, compare, method, _TEST_METHODS, _TEST_DEFAULT)
    alt = C.alternative(alternative, name)
    corr = C.flag(correction, name, "correction")
    v = _value(value, cmp, name)
    shape, rows = _two(name, count1, nobs1, count2, nobs2)
    z, p, var, p1, p0, d, r, o = _k.prop2_test(*rows, _COMPARE[cmp], _M2[meth], v, _ALT[alt], corr)
    fields = {
        "statistic": _out(z, shape),
        "pvalue": _out(p, shape),
        "compare": cmp,
        "method": meth,
        "diff": _out(d, shape),
        "ratio": _out(r, shape),
        "odds_ratio": _out(o, shape),
        "variance": _out(var, shape),
        "alternative": alt,
        "value": v,
    }
    if meth == "score":
        fields["prop1_null"] = _out(p1, shape)
        fields["prop2_null"] = _out(p0, shape)
    info = {} if shape else {"nobs": (int(rows[1][0]), int(rows[3][0]))}
    if meth == "score":
        info["pairs"] = [("Correction:", str(corr))]
    return TestProportions2indepResult._make(batch=shape, info=info, **fields)


def _value(value, cmp, name):
    """The null value: a difference in (-1, 1) (0 by default), a ratio or an odds ratio in (0,
    inf) (1 by default) (H97)."""
    if value is None:
        return 0.0 if cmp == "diff" else 1.0
    v = C.real(value, name, "value")
    if cmp == "diff":
        if not -1.0 < v < 1.0:
            raise ValueError(f"{name}(): 'value' must be between -1 and 1 (exclusive) for compare='diff'")
    elif not (v > 0.0 and math.isfinite(v)):
        raise ValueError(f"{name}(): 'value' must be positive and finite for compare={cmp!r}")
    return v


def confint_proportions_2indep(
    count1, nobs1, count2, nobs2, /, *, method=None, compare="diff", alpha=0.05, correction=True
):
    """statsmodels' 1 - alpha confidence interval of the comparison of two independent proportions
    (broadcast): a new (..., 2) array ((2,) for one; unpacks as (low, upp)). compare 'diff' (method
    'wald', 'agresti-caffo', 'newcomb' the default (Wilson's intervals combined), 'score' (the
    Miettinen-Nurminen score test inverted)), 'ratio' ('log', 'log-adjusted' the default, 'score':
    Koopman's), 'odds-ratio' ('logit', 'logit-adjusted' the default, 'logit-smoothed', 'score');
    correction the score methods' N / (N - 1)."""
    name = "confint_proportions_2indep"
    cmp, meth = _compare_method(name, compare, method, _CI_METHODS, _CI_DEFAULT)
    a = C.alpha(alpha, name)
    corr = C.flag(correction, name, "correction")
    shape, rows = _two(name, count1, nobs1, count2, nobs2)
    lo, hi = _k.prop2_ci(*rows, _COMPARE[cmp], _M2[meth], a, corr)
    return _pair(lo, hi, shape)


# a wrong call shape in the library's form (API.md §2.1)
binomtest = _calls.function(binomtest)
quantile_test = _calls.function(quantile_test)
poisson_means_test = _calls.function(poisson_means_test)
proportions_ztest = _calls.function(proportions_ztest)
proportions_chisquare = _calls.function(proportions_chisquare)
proportion_confint = _calls.function(proportion_confint)
test_proportions_2indep = _calls.function(test_proportions_2indep)
confint_proportions_2indep = _calls.function(confint_proportions_2indep)
