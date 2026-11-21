"""The contingency tables (milestone H6; docs/api/htest.md §7, docs/design/htest.md §3.7).

chi2_contingency, fisher_exact, barnard_exact, boschloo_exact, odds_ratio and relative_risk
(scipy's, scipy.stats.contingency's) and mcnemar and cochrans_q (statsmodels'), with this
library's conventions (positional data, keyword-only options, read-only results; decisions
H3-H21 and the H6 decisions H77-H82 in docs/design/htest.md §1.1). A table's last two axes are
the table and the leading axes a batch (H9).

The checks are here; the computation is one native call (stardust._core.htest:
htest/contingency.cpp). fisher_exact's r x c test is exact by the network algorithm (H77;
scipy's is an unseeded Monte Carlo), Barnard's and Boschloo's p-values are the certified
supremum over the nuisance parameter whatever n (H78: n is the first grid's size only),
odds_ratio's conditional estimate and interval come from Fisher's noncentral hypergeometric
(H79), chi2_contingency takes non-integer (weighted) counts (H80) and gives df as a float (H82),
mcnemar is exact by default (H81, statsmodels').
"""

import math
from fractions import Fraction

from .. import _calls
from .. import _core as _sd
from . import _common as C
from . import _resampling as R
from ._gof import _lambda
from ._result import _TestResult

_k = _sd.htest
_st = _sd.stats
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
_KIND = {"conditional": 0, "sample": 1}
_FISHER_CAP = 2e8  # fisher_exact's r x c work: compositions and (composition, past value) pairs (H13)
_FISHER_HOLD = 5e6  # its past values held at once (memory)


# ---- the result classes (H6: not exported) ------------------------------------------------


class Chi2ContingencyResult(_TestResult):
    __slots__ = ()
    _TITLE = "Chi-squared test of independence"
    _FIELDS = ("statistic", "pvalue", "df", "expected_freq")
    _UNPACK = ("statistic", "pvalue", "df", "expected_freq")


class FisherExactResult(_TestResult):
    __slots__ = ()
    _TITLE = "Fisher's exact test"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


class BarnardExactResult(_TestResult):
    __slots__ = ()
    _TITLE = "Barnard's exact test"
    _FIELDS = ("statistic", "pvalue")


class BoschlooExactResult(BarnardExactResult):
    __slots__ = ()
    _TITLE = "Boschloo's exact test"


class McnemarResult(BarnardExactResult):
    __slots__ = ()
    _TITLE = "McNemar's test"


class CochransQResult(_TestResult):
    __slots__ = ()
    _TITLE = "Cochran's Q test"
    _FIELDS = ("statistic", "pvalue", "df")


class OddsRatioResult(_TestResult):
    __slots__ = ()
    _TITLE = "Odds ratio"
    _CI_LABEL = "odds ratio"
    _FIELDS = ("statistic",)
    _CI = True

    def conf_int(self, *, alpha=0.05, alternative="two-sided"):
        """The 1 - alpha confidence interval of the odds ratio (scipy's confidence_interval with
        alpha for 1 - confidence_level, H16): the exact (conditional) or Woolf's (sample)
        interval, one-sided for alternative 'less' (0 below) or 'greater' (inf above). A new
        (..., 2) array, (2,) for one table."""
        alt = C.alternative(alternative, "conf_int")
        a = C.alpha(alpha)
        low, high = self._ci(a, alt)
        if isinstance(low, _sd.ndarray):
            return _sd.stack([low + 0.0, high + 0.0], axis=-1)
        return _sd.asarray([float(low), float(high)])


class RelativeRiskResult(_TestResult):
    __slots__ = ()
    _TITLE = "Relative risk"
    _CI_LABEL = "relative risk"
    _FIELDS = ("relative_risk", "exposed_cases", "exposed_total", "control_cases", "control_total")
    _CI = True


# ---- helpers ------------------------------------------------------------------------------


def _two_by_two(t, name, arg="table"):
    if t.table != (2, 2):
        raise ValueError(f"{name}(): '{arg}' must be a 2 x 2 table, not {t.table[0]} x {t.table[1]}")


def _shape_text(t):
    return f"{t.table[0]} x {t.table[1]}"


# ---- chi2_contingency ---------------------------------------------------------------------


def chi2_contingency(observed, /, *, correction=True, lambda_=None, method=None):
    """The chi-squared test of independence of a two-way table: statistic (power_divergence's of
    the observed counts against the expected R_i C_j / N; lambda_ its Cressie-Read power, None is
    Pearson's), pvalue (chi^2 with df = (r - 1)(c - 1)), df (a float) and expected_freq. With one
    degree of freedom and correction=True Yates' correction moves each count towards its
    expectation by min(1/2, |O - E|). The counts may be non-integer (a weighted table). df = 0
    (one row or column) gives statistic 0 and pvalue 1; a zero row or column (an expected count of
    0) gives NaN. Tables along the last two axes, a batch before them. method: None, or (H9,
    scipy's) a PermutationMethod or a MonteCarloMethod: the pvalue of Pearson's statistic among
    tables with the observed margins (integer counts, correction=False and lambda_=None; df is
    NaN). Unpacks as (statistic, pvalue, df, expected_freq)."""
    name = "chi2_contingency"
    corr = C.flag(correction, name, "correction")
    lam = _lambda(lambda_, name)
    if method is not None:  # H9 (scipy's): Pearson's statistic alone, of the counts
        R.check_method(method, name, (R.PermutationMethod, R.MonteCarloMethod), ("None",))
        if corr:
            raise ValueError(
                f"{name}(): correction=True is not compatible with a resampling method; pass correction=False"
            )
        if lambda_ is not None:
            raise ValueError(f"{name}(): lambda_ is not compatible with a resampling method; leave it None")
    t = C.Tables(observed, name, "observed", integer=method is not None, minimum=(1, 1))
    stat, p, df, exp = _k.chi2_contingency(t.rows, corr, lam)
    pairs = [("Table:", _shape_text(t))]
    if method is not None:
        _, pv, more = R.table_pvalues(name, method, t.rows, 0, "greater")
        p = _sd.asarray(pv)
        df = _sd.full((t.lanes,), math.nan)
        pairs += more
    if lambda_ is not None:
        pairs.append(("Lambda:", lambda_ if isinstance(lambda_, str) else f"{lam:g}"))
    if corr and t.table[0] > 1 and t.table[1] > 1 and (t.table[0] - 1) * (t.table[1] - 1) == 1:
        pairs.append(("Correction:", "Yates"))
    return Chi2ContingencyResult._make(
        batch=t.batch,
        info={"pairs": pairs},
        statistic=t.out(stat),
        pvalue=t.out(p),
        df=t.out(df),
        expected_freq=_sd.reshape(exp, t.batch + t.table),
    )


# ---- Fisher, Barnard, Boschloo ------------------------------------------------------------


def fisher_exact(table, /, *, alternative="two-sided", method=None):
    """Fisher's exact test. A 2 x 2 table: statistic the sample odds ratio ad / bc (inf for bc =
    0), pvalue from the hypergeometric distribution of the top-left cell given the margins
    (scipy's two-sided rule: the tables at most (1 + 1e-14) times as probable as the observed
    one; 1 when it is as probable as the mode). An r x c table: statistic the probability of the
    observed table, pvalue the exact sum over every table with its margins at most (1 + 1e-7)
    times as probable (R's fisher.test, the network algorithm; scipy's default is an unseeded Monte
    Carlo), two-sided only; past the documented work cap a ValueError. A zero row or column of a
    2 x 2 table gives NaN and pvalue 1. method: None, or (H9, scipy's) a PermutationMethod or a
    MonteCarloMethod: for any table (2 x 2 too) statistic the table's probability and pvalue the
    share of tables with its margins at most as probable (two-sided only). Unpacks as
    (statistic, pvalue)."""
    name = "fisher_exact"
    alt = C.alternative(alternative, name)
    if method is not None:
        R.check_method(method, name, (R.PermutationMethod, R.MonteCarloMethod), ("None",))
        if alt != "two-sided":
            raise ValueError(f"{name}(): a resampling method takes alternative='two-sided' only, not {alternative!r}")
    t = C.Tables(table, name, "table", minimum=(1, 1))
    if method is not None:  # H9 (scipy's r x c path, 2 x 2 too): the tables at most as probable
        stats, ps, pairs = R.table_pvalues(name, method, t.rows, 1, "less")
        for i in range(t.lanes):
            if t.table[0] == 1 or t.table[1] == 1 or float(_sd.sum(t.rows[i])) == 0.0:
                stats[i], ps[i] = 1.0, 1.0  # one table with these margins (scipy's)
        stat = _sd.minimum(_sd.asarray(stats), 1.0)
        info = {"pairs": [("Table:", _shape_text(t))] + pairs}
        return FisherExactResult._make(batch=t.batch, info=info, statistic=t.out(stat), pvalue=t.out(_sd.asarray(ps)))
    if t.table == (2, 2):
        stat, p = _k.fisher22(t.rows, _ALT[alt])
        info = {"alternative": alt}
    else:
        if alt != "two-sided":
            raise ValueError(
                f"{name}(): 'alternative' must be 'two-sided' for a table that is not 2 x 2, not {alternative!r}"
            )
        stat, p, capped = _k.fisher_rxc(t.rows, _FISHER_CAP, _FISHER_HOLD)
        if bool(_sd.any(capped != 0.0)):
            raise ValueError(
                f"{name}(): the exact r x c test needs more than its cap of {_FISHER_CAP:.3g} operations for this "
                "table; use chi2_contingency"
            )
        info = {"pairs": [("Table:", _shape_text(t))]}
    return FisherExactResult._make(batch=t.batch, info=info, statistic=t.out(stat), pvalue=t.out(p))


def barnard_exact(table, /, *, alternative="two-sided", pooled=True, n=32):
    """Barnard's exact test of a 2 x 2 table whose columns are the two binomial samples (scipy's
    orientation): statistic the Wald statistic of the difference of the proportions (the pooled
    variance, or the unpooled one), pvalue the supremum over the common proportion of the
    probability of the tables at least as extreme (|T| >= |T_obs|, or T <= T_obs, T >= T_obs). The
    supremum is certified to a relative 2^-48 whatever n, the size of the first grid of the search
    (scipy's n of shgo; H78). A zero column gives NaN and pvalue 1. Past (n1 + 1)(n2 + 1) = 1e7
    tables the pvalue is NaN. Does not unpack (scipy's result does not)."""
    name = "barnard_exact"
    alt = C.alternative(alternative, name)
    pool = C.flag(pooled, name, "pooled")
    grid = C.integer(n, name, "n", minimum=1)
    t = C.Tables(table, name, "table")
    _two_by_two(t, name)
    stat, p = _k.barnard(t.rows, _ALT[alt], pool, grid)
    title = f"Barnard's exact test ({'pooled' if pool else 'unpooled'})"
    return BarnardExactResult._make(
        batch=t.batch, info={"title": title, "alternative": alt}, statistic=t.out(stat), pvalue=t.out(p)
    )


def boschloo_exact(table, /, *, alternative="two-sided", n=32):
    """Boschloo's exact test of a 2 x 2 table whose columns are the two binomial samples: statistic
    the one-sided Fisher p-value of the observed table (two-sided: the smaller side's), pvalue the
    supremum over the common proportion of the probability of the tables at most (1 + 1e-13) times
    as significant (scipy's tolerance; two-sided: twice the smaller side's, at most 1). Certified
    whatever n, as barnard_exact (H78). A zero column gives NaN for both. Does not unpack."""
    name = "boschloo_exact"
    alt = C.alternative(alternative, name)
    grid = C.integer(n, name, "n", minimum=1)
    t = C.Tables(table, name, "table")
    _two_by_two(t, name)
    stat, p = _k.boschloo(t.rows, _ALT[alt], grid)
    return BoschlooExactResult._make(batch=t.batch, info={"alternative": alt}, statistic=t.out(stat), pvalue=t.out(p))


# ---- McNemar and Cochran's Q (statsmodels) ------------------------------------------------


def mcnemar(table, /, *, exact=True, correction=True):
    """McNemar's test of the symmetry of a paired 2 x 2 table (statsmodels'): from the off-diagonal
    counts b = table[0, 1] and c = table[1, 0], exact=True gives statistic min(b, c) and the exact
    binomial pvalue (twice the tail of Binomial(b + c, 1/2), at most 1); exact=False the chi-squared
    statistic (|b - c| - correction)^2 / (b + c) and its chi^2(1) pvalue (b + c = 0 gives NaN). Does
    not unpack (statsmodels' Bunch)."""
    name = "mcnemar"
    ex = C.flag(exact, name, "exact")
    corr = C.flag(correction, name, "correction")
    t = C.Tables(table, name, "table")
    _two_by_two(t, name)
    stat, p = _k.mcnemar(t.rows, ex, corr)
    title = "McNemar's test (exact)" if ex else "McNemar's test (chi-squared)"
    pairs = [] if ex else [("Correction:", str(corr))]
    info = {"title": title, "pairs": pairs}
    return McnemarResult._make(batch=t.batch, info=info, statistic=t.out(stat), pvalue=t.out(p))


def cochrans_q(x, /):
    """Cochran's Q test that k related binary samples have the same proportion of successes
    (statsmodels'): x is (N, k) of 0 and 1 (blocks by treatments); statistic Q, pvalue (chi^2 with
    k - 1 degrees of freedom) and df. NaN when every block is all 0 or all 1. Batches along the
    leading axes. Does not unpack."""
    name = "cochrans_q"
    t = C.Tables(x, name, "x", minimum=(1, 1))
    if t.rows.size and bool(_sd.any(t.rows > 1.0)):
        raise ValueError(f"{name}(): 'x' must hold 0 and 1 values (the successes)")
    q, p, df = _k.cochrans_q(t.rows)
    info = {"nobs": t.table[0], "pairs": [("No. Treatments:", str(t.table[1]))]}
    return CochransQResult._make(batch=t.batch, info=info, statistic=t.out(q), pvalue=t.out(p), df=t.out(df))


# ---- the estimates (scipy.stats.contingency) ----------------------------------------------


def odds_ratio(table, /, *, kind="conditional"):
    """The odds ratio of a 2 x 2 table (scipy.stats.contingency's): kind='conditional' its
    conditional maximum likelihood estimate (Fisher's noncentral hypergeometric of the top-left cell
    given the margins; 0 or inf at the ends of its support), 'sample' ad / bc. statistic, and
    conf_int(*, alpha=0.05, alternative='two-sided'): the exact interval (conditional) or Woolf's
    (sample). A zero row or column gives NaN and the interval (0, inf). Batches along the leading
    axes. Not a test: no pvalue, does not unpack."""
    name = "odds_ratio"
    kd = C.choice(kind, name, "kind", tuple(_KIND))
    t = C.Tables(table, name, "table")
    _two_by_two(t, name)
    (stat,) = _k.odds_ratio(t.rows, _KIND[kd])
    rows = t.rows + 0.0  # (the interval is computed when asked: the table kept, not the caller's array)

    def ci(alpha, alternative="two-sided"):
        lo, hi = _k.odds_ratio_ci(rows, _KIND[kd], alpha, _ALT[alternative])
        return t.out(lo), t.out(hi)

    title = "Odds ratio (conditional maximum likelihood)" if kd == "conditional" else "Odds ratio (sample)"
    return OddsRatioResult._make(batch=t.batch, ci=ci, info={"title": title}, statistic=t.out(stat))


def relative_risk(exposed_cases, exposed_total, control_cases, control_total, /):
    """The relative risk (scipy.stats.contingency's): (exposed_cases / exposed_total) /
    (control_cases / control_total), the counts ints (cases >= 0, totals >= 1, cases <= totals).
    Fields relative_risk and the four counts, and conf_int(*, alpha=0.05): Katz's interval rr
    exp(-+ z se), se^2 = 1/a - 1/n1 + 1/c - 1/n2 (scipy's ends where a case count is 0: (0, NaN),
    (NaN, inf), (NaN, NaN)). The estimate is the exact ratio rounded once. Not a test."""
    name = "relative_risk"
    a = C.integer(exposed_cases, name, "exposed_cases", minimum=0)
    n1 = C.integer(exposed_total, name, "exposed_total", minimum=1)
    c = C.integer(control_cases, name, "control_cases", minimum=0)
    n2 = C.integer(control_total, name, "control_total", minimum=1)
    if a > n1:
        raise ValueError(f"{name}(): 'exposed_cases' must not exceed 'exposed_total'")
    if c > n2:
        raise ValueError(f"{name}(): 'control_cases' must not exceed 'control_total'")
    if a == 0 and c == 0:
        rr = math.nan
    elif a == 0:
        rr = 0.0
    elif c == 0:
        rr = math.inf
    else:
        rr = float(Fraction(a * n2, n1 * c))

    def ci(alpha):
        if a == 0 and c == 0:
            return math.nan, math.nan
        if a == 0:
            return 0.0, math.nan
        if c == 0:
            return math.nan, math.inf
        z = float(_st.normal_iccdf(0.0, 1.0, alpha / 2.0))
        se = math.sqrt(float(Fraction(n1 - a, a * n1) + Fraction(n2 - c, c * n2)))
        return rr * math.exp(-z * se), rr * math.exp(z * se)

    return RelativeRiskResult._make(
        ci=ci,
        relative_risk=rr,
        exposed_cases=a,
        exposed_total=n1,
        control_cases=c,
        control_total=n2,
    )


# a wrong call shape in the library's form (API.md §2.1)
chi2_contingency = _calls.function(chi2_contingency)
fisher_exact = _calls.function(fisher_exact)
barnard_exact = _calls.function(barnard_exact)
boschloo_exact = _calls.function(boschloo_exact)
mcnemar = _calls.function(mcnemar)
cochrans_q = _calls.function(cochrans_q)
odds_ratio = _calls.function(odds_ratio)
relative_risk = _calls.function(relative_risk)
