"""The rank tests (milestone H3; docs/api/htest.md §4, docs/design/htest.md §3.4).

mannwhitneyu, ranksums, wilcoxon, kruskal, friedmanchisquare, brunnermunzel, ansari, mood,
median_test and page_trend_test: scipy's tests with this library's conventions (positional data,
keyword-only options, axis=None, batches along the other axes, nan_policy, read-only results;
decisions H3-H21 and the H3 decisions H45-H50 in docs/design/htest.md §1.1).

The checks and the arrangement of the samples are here; the computation is one native call
(stardust._core.htest: htest/rank.cpp) that returns every lane's results. The ranks are exact
midranks, so the rank sums and U, T+, AB, L are exact; with ties, method='exact' is the exact
distribution conditional on the observed midranks (H45; scipy's 'exact' ignores the ties).
'auto' picks the method by scipy's rules (H46), lane by lane; the result's method field says
which (H12), a nested list of the lanes' names for a batch (H48), and zstatistic is there
wherever the normal approximation was used (H47). method='exact' beyond its work cap (1e9
operations or 1e7 counts held) raises ValueError (H13); 'auto' takes the normal approximation
there. Degenerate lanes give NaN without warnings (H11).
"""

import math

from .. import _calls
from .. import _core as _sd
from ..stats._descriptive import _as_array, _scalar_value
from . import _common as C
from . import _resampling as R
from ._gof import _lambda, _nest
from ._result import _TestResult

_k = _sd.htest
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_REQ = {"auto": 0, "exact": 1, "asymptotic": 2}
_USED = ("exact", "asymptotic")
_RANK_METHODS = ("auto", "exact", "asymptotic")
_ZERO = {"wilcox": 0, "pratt": 1, "zsplit": 2}
_TIES = {"below": 0, "above": 1, "ignore": 2}
_DIST = {"t": 0, "normal": 1}
_EXACT_CAP = 1e9  # operations of method='exact' (H13; htest/rank.cpp also caps the counts it holds at 1e7)


# ---- the result classes (H6: not exported) ------------------------------------------------


class MannwhitneyuResult(_TestResult):
    __slots__ = ()
    _TITLE = "Mann-Whitney U test"
    _FIELDS = ("statistic", "pvalue", "zstatistic", "method")
    _UNPACK = ("statistic", "pvalue")


class RanksumsResult(_TestResult):
    __slots__ = ()
    _TITLE = "Wilcoxon rank-sum test"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


class WilcoxonResult(MannwhitneyuResult):
    __slots__ = ()
    _TITLE = "Wilcoxon signed-rank test"


class KruskalResult(_TestResult):
    __slots__ = ()
    _TITLE = "Kruskal-Wallis rank sum test"
    _FIELDS = ("statistic", "pvalue", "df")
    _UNPACK = ("statistic", "pvalue")


class FriedmanchisquareResult(KruskalResult):
    __slots__ = ()
    _TITLE = "Friedman rank sum test"


class BrunnermunzelResult(KruskalResult):
    __slots__ = ()
    _TITLE = "Brunner-Munzel test"


class AnsariResult(_TestResult):
    __slots__ = ()
    _TITLE = "Ansari-Bradley test"
    _FIELDS = ("statistic", "pvalue", "method")
    _UNPACK = ("statistic", "pvalue")


class MoodResult(RanksumsResult):
    __slots__ = ()
    _TITLE = "Mood two-sample test of scale"


class MedianTestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Mood's median test"
    _FIELDS = ("statistic", "pvalue", "median", "table", "df")
    _TABLES = ("table",)
    _UNPACK = ("statistic", "pvalue", "median", "table")


class PageTrendTestResult(_TestResult):
    __slots__ = ()
    _TITLE = "Page's trend test"
    _FIELDS = ("statistic", "pvalue", "method")


# ---- helpers ------------------------------------------------------------------------------


def _shape(s):
    return s.keep_shape if s.keepdims else s.batch


def _methods(codes, shape, one):
    """The lanes' methods (H12, H48): a name for one test, else nested lists in the batch's shape."""
    names = [_USED[int(v)] for v in codes.tolist()]
    return names[0] if one else _nest(names, shape)


def _check_work(work, method, name):
    """method='exact' beyond the cap in some lane: the lanes report their work (0 within it)."""
    if method == "exact" and work.size:
        w = float(_sd.max(work))
        if w > 0.0:
            C.check_exact_work(w, _EXACT_CAP, name)


def _two(name, x, y, axis, nan_policy, keepdims):
    return C.Samples(name, [x, y], ("x", "y"), axis, nan_policy, keepdims=keepdims, scan=False)


def _with_z(cls, s, stat, p, z, codes, method_out, info):
    """The result with zstatistic where the normal approximation was used (H47): a single test's
    when it was, a batch's always (H156; NaN in its exact lanes), so the field does not come and go
    with the data."""
    fields = {"statistic": s.out(stat), "pvalue": s.out(p), "method": method_out}
    if not _one(s):
        fields["zstatistic"] = s.out(_sd.where(codes == 1.0, z, math.nan))
    elif codes.size and bool(_sd.any(codes == 1.0)):
        fields["zstatistic"] = s.out(z)
    return cls._make(batch=s.batch, info=info, **fields)


def _one(s):
    return not _shape(s) and not s.keepdims


def _is_permutation(method, name):
    """method= of a rank test: a PermutationMethod (H9) or one of the strings; another resampling
    object is a TypeError naming them."""
    if isinstance(method, R.PermutationMethod):
        return True
    if isinstance(method, (R.MonteCarloMethod, R.BootstrapMethod)):
        R.check_method(method, name, R.PermutationMethod, _RANK_METHODS)
    return False


def _permutation_result(cls, s, stat, p, pairs, info, extra=None):
    """A rank test's result with its pvalue from the permutations: method 'permutation' (H12), no
    zstatistic (no normal approximation, H47)."""
    one = _one(s)
    m = "permutation" if one else R.nested_names("permutation", _shape(s))
    info["pairs"] = info.get("pairs", []) + pairs
    return cls._make(batch=s.batch, info=info, statistic=s.out(stat), pvalue=s.out(p), method=m, **(extra or {}))


# ---- two samples --------------------------------------------------------------------------


def mannwhitneyu(
    x, y, /, *, axis=None, use_continuity=True, alternative="two-sided", method="auto", nan_policy="propagate",
    keepdims=False,
):  # fmt: skip
    """The Mann-Whitney U test: statistic U1 (x's U: the pairs with x above y, ties counting
    half), pvalue, method and zstatistic (the normal approximation's, with the tie and continuity
    corrections). method 'auto' (exact when one sample has at most 8 values and there are no ties;
    scipy's rule), 'exact' (with ties, conditional on the midranks), 'asymptotic' or a
    PermutationMethod (H9: U under the permutations; method 'permutation'). Unpacks as (statistic,
    pvalue)."""
    name = "mannwhitneyu"
    cont = C.flag(use_continuity, name, "use_continuity")
    alt = C.alternative(alternative, name)
    perm = _is_permutation(method, name)
    meth = "asymptotic" if perm else C.method(method, name, ("exact", "asymptotic"))
    s = _two(name, x, y, axis, nan_policy, keepdims)
    stat, p, z, codes, work = _k.mwu(s.rows[0], s.rows[1], cont, _ALT[alt], _REQ[meth], _EXACT_CAP, s.policy == "omit")
    if perm:  # H9: U under the permutations of the pooled samples (scipy's)

        def u(a, b):
            return mannwhitneyu(a, b, axis=-1, method="asymptotic").statistic

        p, pairs = R.method_pvalue(name, method, s, u, alt)
        return _permutation_result(MannwhitneyuResult, s, stat, p, pairs, s.info(alt))
    _check_work(work, meth, name)
    return _with_z(MannwhitneyuResult, s, stat, p, z, codes, _methods(codes, _shape(s), _one(s)), s.info(alt))


def ranksums(x, y, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """The Wilcoxon rank-sum test: statistic z = (R1 - n1 (N + 1)/2) / sqrt(n1 n2 (N + 1)/12) (x's
    rank sum, without tie or continuity corrections: scipy's) and pvalue (standard normal). Kept
    though scipy marks it legacy (H50): mannwhitneyu has the corrections and the exact test.
    Unpacks as (statistic, pvalue)."""
    name = "ranksums"
    alt = C.alternative(alternative, name)
    s = _two(name, x, y, axis, nan_policy, keepdims)
    stat, p = _k.ranksums(s.rows[0], s.rows[1], _ALT[alt], s.policy == "omit")
    return RanksumsResult._make(batch=s.batch, info=s.info(alt), statistic=s.out(stat), pvalue=s.out(p))


def wilcoxon(
    x, y=None, /, *, axis=None, zero_method="wilcox", correction=False, alternative="two-sided", method="auto",
    nan_policy="propagate", keepdims=False,
):  # fmt: skip
    """The Wilcoxon signed-rank test of the differences d = x - y (x itself without y; exact
    differences: two close large values do not round): statistic min(T+, T-) (T+ for a one-sided
    test), pvalue, method and zstatistic. zero_method 'wilcox' (zeros dropped), 'pratt' (ranked,
    then left out) or 'zsplit' (their ranks split between T+ and T-). method 'auto' (exact up to
    50 differences without ties or zeros, exact by the signs' enumeration up to 13 with them:
    scipy's rule), 'exact' (with ties or zeros, conditional on the midranks), 'asymptotic' or a
    PermutationMethod (H9: T+ under the sign flips of the differences). Unpacks as (statistic,
    pvalue)."""
    name = "wilcoxon"
    zm = C.choice(zero_method, name, "zero_method", tuple(_ZERO))
    corr = C.flag(correction, name, "correction")
    alt = C.alternative(alternative, name)
    perm = _is_permutation(method, name)
    meth = "asymptotic" if perm else C.method(method, name, ("exact", "asymptotic"))
    if y is None:
        s = C.Samples(name, [x], ("x",), axis, nan_policy, keepdims=keepdims, scan=False)
        yrows = None
    else:
        s = C.Samples(name, [x, y], ("x", "y"), axis, nan_policy, paired=True, keepdims=keepdims, scan=False)
        yrows = s.rows[1]
    stat, p, z, codes, work = _k.wilcoxon(
        s.rows[0], yrows, _ZERO[zm], corr, _ALT[alt], _REQ[meth], _EXACT_CAP, s.policy == "omit"
    )
    info = s.info(alt, pairs=[("Zero Method:", zm)])
    if perm:  # H9: T+ under the sign flips of the differences (scipy's permutation_type='samples')
        d = s.rows[0] if yrows is None else s.rows[0] - yrows

        def tplus(v):
            return wilcoxon(v, axis=-1, zero_method=zm, alternative="greater", method="asymptotic").statistic

        p, pairs = R.method_pvalue(name, method, s, tplus, alt, ptype="samples", data=[d])
        return _permutation_result(WilcoxonResult, s, stat, p, pairs, info)
    _check_work(work, meth, name)
    return _with_z(WilcoxonResult, s, stat, p, z, codes, _methods(codes, _shape(s), _one(s)), info)


def brunnermunzel(
    x, y, /, *, axis=None, alternative="two-sided", distribution="t", nan_policy="propagate", keepdims=False
):
    """The Brunner-Munzel test: statistic W (scipy's: positive when y tends to be larger), pvalue
    (Student's t with Satterthwaite's df, or the standard normal) and df (distribution='t' only).
    NaN where both samples' placement variances are 0 (complete separation or every value tied)
    or a sample has fewer than 2 values. Unpacks as (statistic, pvalue)."""
    name = "brunnermunzel"
    alt = C.alternative(alternative, name)
    dist = C.choice(distribution, name, "distribution", tuple(_DIST))
    s = _two(name, x, y, axis, nan_policy, keepdims)
    stat, p, df = _k.brunnermunzel(s.rows[0], s.rows[1], _ALT[alt], _DIST[dist], s.policy == "omit")
    fields = {"statistic": s.out(stat), "pvalue": s.out(p)}
    if dist == "t":
        fields["df"] = s.out(df)
    info = s.info(alt, pairs=[("Distribution:", dist)])
    return BrunnermunzelResult._make(batch=s.batch, info=info, **fields)


def ansari(x, y, /, *, axis=None, alternative="two-sided", method="auto", nan_policy="propagate", keepdims=False):
    """The Ansari-Bradley test of equal scales: statistic AB (x's sum of min(r, N + 1 - r) over its
    midranks r), pvalue and method: 'auto' (exact when both samples have fewer than 55 values and
    there are no ties: scipy's rule), 'exact' (with ties, conditional on the midranks),
    'asymptotic' (the normal approximation, scipy's tie-corrected variance) or a PermutationMethod
    (H9). Unpacks as (statistic, pvalue)."""
    name = "ansari"
    alt = C.alternative(alternative, name)
    perm = _is_permutation(method, name)
    meth = "asymptotic" if perm else C.method(method, name, ("exact", "asymptotic"))
    s = _two(name, x, y, axis, nan_policy, keepdims)
    stat, p, codes, work = _k.ansari(s.rows[0], s.rows[1], _ALT[alt], _REQ[meth], _EXACT_CAP, s.policy == "omit")
    if perm:  # H9: AB under the permutations; small AB is a larger scale ratio (scipy flips the side)

        def ab(a, b):
            return ansari(a, b, axis=-1, method="asymptotic").statistic

        palt = {"less": "greater", "greater": "less"}.get(alt, alt)
        p, pairs = R.method_pvalue(name, method, s, ab, palt)
        return _permutation_result(AnsariResult, s, stat, p, pairs, s.info(alt))
    _check_work(work, meth, name)
    m = _methods(codes, _shape(s), _one(s))
    return AnsariResult._make(batch=s.batch, info=s.info(alt), statistic=s.out(stat), pvalue=s.out(p), method=m)


def mood(x, y, /, *, axis=None, alternative="two-sided", nan_policy="propagate", keepdims=False):
    """Mood's test of equal scales: statistic z (the squared-rank sum T of x standardized; with ties
    Mielke's T and variance, scipy's) and pvalue (standard normal). NaN for N < 3. Unpacks as
    (statistic, pvalue)."""
    name = "mood"
    alt = C.alternative(alternative, name)
    s = _two(name, x, y, axis, nan_policy, keepdims)
    stat, p = _k.mood(s.rows[0], s.rows[1], _ALT[alt], s.policy == "omit")
    return MoodResult._make(batch=s.batch, info=s.info(alt), statistic=s.out(stat), pvalue=s.out(p))


# ---- k samples ----------------------------------------------------------------------------


def _ksamples(name, samples, least, axis, nan_policy, keepdims, paired=False):
    if len(samples) < least:
        word = {2: "two", 3: "three"}[least]
        raise TypeError(f"{name}(): at least {word} samples are required, not {len(samples)}")
    args = tuple(f"samples[{i}]" for i in range(len(samples)))
    return C.Samples(name, list(samples), args, axis, nan_policy, paired=paired, keepdims=keepdims, scan=False)


def kruskal(*samples, axis=None, nan_policy="propagate", keepdims=False):
    """The Kruskal-Wallis H test: statistic H (with scipy's tie correction), pvalue (chi^2 with k - 1
    degrees of freedom) and df (k - 1). NaN where a sample is empty or every value ties. Unpacks as
    (statistic, pvalue)."""
    name = "kruskal"
    s = _ksamples(name, samples, 2, axis, nan_policy, keepdims)
    stat, p, df = _k.kruskal(s.rows, s.policy == "omit")
    return KruskalResult._make(batch=s.batch, info=s.info(), statistic=s.out(stat), pvalue=s.out(p), df=s.out(df))


def friedmanchisquare(*samples, axis=None, nan_policy="propagate", keepdims=False):
    """The Friedman test of k >= 3 paired samples (observation j of every sample is a block, ranked
    within the block): statistic (with scipy's tie correction), pvalue (chi^2 with k - 1 degrees of
    freedom) and df. nan_policy='omit' drops the blocks that hold a NaN. Unpacks as (statistic,
    pvalue)."""
    name = "friedmanchisquare"
    s = _ksamples(name, samples, 3, axis, nan_policy, keepdims, paired=True)
    stat, p, df = _k.friedman(s.rows, s.policy == "omit")
    return FriedmanchisquareResult._make(
        batch=s.batch, info=s.info(), statistic=s.out(stat), pvalue=s.out(p), df=s.out(df)
    )


def median_test(*samples, axis=None, ties="below", correction=True, lambda_=1, nan_policy="propagate", keepdims=False):
    """Mood's median test: the grand median of the pooled samples (exact: the mean of the two middle
    values is not rounded before the comparisons), the 2 x k table of the counts above and below it
    (ties 'below', 'above' or 'ignore': where the values equal to it go) and scipy's
    chi2_contingency of the table (Yates' correction for k = 2 when correction, the Cressie-Read
    lambda_ as power_divergence's). statistic, pvalue, median, table (int64, (..., 2, k)) and df
    (k - 1). axis and batches as the other tests (H49). Unpacks as (statistic, pvalue, median,
    table)."""
    name = "median_test"
    t = C.choice(ties, name, "ties", tuple(_TIES))
    corr = C.flag(correction, name, "correction")
    lam = _lambda(lambda_, name)
    s = _ksamples(name, samples, 2, axis, nan_policy, keepdims)
    stat, p, med, df, table = _k.median_test(s.rows, _TIES[t], corr, lam, s.policy == "omit")
    k = len(samples)
    tab = _sd.reshape(table, _shape(s) + (2, k)).astype(_sd.int64)
    return MedianTestResult._make(
        batch=s.batch,
        info=s.info(pairs=[("Ties:", t)]),
        statistic=s.out(stat),
        pvalue=s.out(p),
        median=s.out(med),
        table=tab,
        df=s.out(df),
    )


# ---- Page's trend test --------------------------------------------------------------------


def _predicted(v, n, name):
    msg = f"{name}(): 'predicted_ranks' must hold each integer from 1 to {n} once"
    if v is None:
        return _sd.arange(1.0, n + 1.0)
    if _scalar_value(v) is not None:
        raise ValueError(msg)
    a = _as_array(v, name, "predicted_ranks")
    vals = a.tolist() if a.ndim == 1 else None
    if vals is None or sorted(vals) != list(range(1, n + 1)):
        raise ValueError(msg)
    return a.astype(_sd.float64)


def page_trend_test(data, /, *, ranked=False, predicted_ranks=None, method="auto"):
    """Page's test of an ordered alternative: data (..., m, n), m blocks (rows) of n treatments,
    the leading axes a batch (H9). statistic L = sum_j p_j C_j (C_j the column sums of the rows'
    midranks, p the predicted ranks, default 1..n), pvalue (one-sided: L at least as large) and
    method: 'auto' (exact for n <= 8 and m <= 12, or n = 3 and m <= 20: scipy's rule), 'exact'
    (with ties, conditional on each row's midranks) or 'asymptotic' (normal). ranked=True takes
    data as ranks already. Does not unpack (scipy's result does not)."""
    name = "page_trend_test"
    rk = C.flag(ranked, name, "ranked")
    meth = C.method(method, name, ("exact", "asymptotic"))
    a = _as_array(data, name, "data")
    if a.ndim < 2:
        raise ValueError(f"{name}(): 'data' must have at least 2 dimensions (the table's), not {a.ndim}")
    m, n = int(a.shape[-2]), int(a.shape[-1])
    if m < 2 or n < 3:
        raise ValueError(f"{name}(): 'data' must be at least a 2 x 3 table (2 blocks, 3 treatments), not {m} x {n}")
    pred = _predicted(predicted_ranks, n, name)
    batch = tuple(a.shape[:-2])
    lanes = math.prod(batch)
    rows = _sd.reshape(a + 0.0, (lanes, m * n))
    if bool(_sd.any(_sd.isnan(rows))):
        raise ValueError(f"{name}(): 'data' must not contain NaN")
    if rk and not bool(_sd.all((rows >= 1.0) & (rows <= float(n)))):
        raise ValueError(f"{name}(): 'data' must hold ranks between 1 and {n} (ranked=True)")
    L, p, codes, work, odd = _k.page(rows, m, n, rk, pred, _REQ[meth], _EXACT_CAP)
    if meth == "exact" and bool(_sd.any(odd == 1.0)):
        raise ValueError(f"{name}(): method='exact' needs ranks that are integers or halves (ranked=True)")
    _check_work(work, meth, name)

    def out(v):
        return float(v[0]) if not batch else _sd.reshape(v, batch)

    info = {"pairs": [("No. Blocks:", str(m)), ("No. Treatments:", str(n))]}
    return PageTrendTestResult._make(
        batch=batch, info=info, statistic=out(L), pvalue=out(p), method=_methods(codes, batch, not batch)
    )


# a wrong call shape in the library's form (API.md §2.1)
mannwhitneyu = _calls.function(mannwhitneyu)
ranksums = _calls.function(ranksums)
wilcoxon = _calls.function(wilcoxon)
kruskal = _calls.function(kruskal)
friedmanchisquare = _calls.function(friedmanchisquare)
brunnermunzel = _calls.function(brunnermunzel)
ansari = _calls.function(ansari)
mood = _calls.function(mood)
median_test = _calls.function(median_test)
page_trend_test = _calls.function(page_trend_test)
