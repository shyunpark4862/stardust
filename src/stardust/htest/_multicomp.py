"""Multiple comparisons and corrections (milestone H8; docs/api/htest.md §9, docs/design/htest.md §3.9).

tukey_hsd, dunnett, false_discovery_control and combine_pvalues (scipy's) and multipletests
(statsmodels'), with this library's conventions (positional data, keyword-only options,
read-only results; decisions H3-H21 and the H8 decisions H102-H116 in docs/design/htest.md §1.1).

The checks are here; the computation is native (stardust._core.htest: htest/multicomp.cpp).
tukey_hsd's p-values are the studentized range's upper tail and dunnett's the one-factor
multivariate t's, both two-level quadratures without randomness (H107, H108: dunnett takes no
rng). multipletests and false_discovery_control share one kernel (htest/multitest.hpp), so
false_discovery_control(method='bh') and multipletests(method='fdr_bh').pvalues_corrected are
the same bits (H102). multipletests takes statsmodels' ten long method names (no fdr_gbs, no
lfdr: H106, H109) and reads one family per lane along `axis` (H105); its reject is the
procedure's decision at alpha, the exact corrected p-value against alpha (H111). The p-values must lie in [0, 1] (H112).
"""

import math
import operator

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from ..stats._descriptive import _as_array, _Lanes, _scalar_value, _type_name
from . import _common as C
from ._result import _TestResult

_k = _sd.htest
_ALT = {"two-sided": 0, "less": 1, "greater": 2}
# statsmodels' long names (H102, H106); scipy's 'bh' and 'by' are fdr_bh and fdr_by
_METHODS = {
    "bonferroni": 0,
    "sidak": 1,
    "holm-sidak": 2,
    "holm": 3,
    "simes-hochberg": 4,
    "hommel": 5,
    "fdr_bh": 6,
    "fdr_by": 7,
    "fdr_tsbh": 8,
    "fdr_tsbky": 9,
}
_FDC = {"bh": 6, "by": 7}
_COMBINE = {"fisher": 0, "pearson": 1, "tippett": 2, "stouffer": 3, "mudholkar_george": 4}
_COMBINE_TITLES = {
    "fisher": "Fisher",
    "pearson": "Pearson",
    "tippett": "Tippett",
    "stouffer": "Stouffer",
    "mudholkar_george": "Mudholkar-George",
}


# ---- summaries of the comparisons ---------------------------------------------------------


def _comparisons(title, left, right, labels, columns, alpha, alternative, interval):
    """A box of one row per comparison: columns [(name, values, formatter)], the 1 - alpha
    interval's ends (lists) as the last two columns."""
    names = [c[0] for c in columns]
    if interval is not None:
        names += _S.interval_columns(alpha, alternative)
    rows = []
    for i in _S.shown(len(labels)):
        if i is None:
            rows.append(None)
            continue
        cells = [fmt(vals[i]) for _, vals, fmt in columns]
        if interval is not None:
            cells += [_S.forg(interval[0][i]), _S.forg(interval[1][i])]
        rows.append((labels[i], cells))
    return _S.render(title, header=(left, right), tables=[_S.Table(names, rows)])


def _sizes(info):
    return [("No. Observations:", _S.sizes(info["nobs"]))] if "nobs" in info else []


# ---- the result classes (H6: not exported) ------------------------------------------------


class TukeyHsdResult(_TestResult):
    """statistic[i, j] = mean_i - mean_j, pvalue[i, j] the studentized range's tail (k, k);
    df the degrees of freedom (N - k, or Games-Howell's per pair, (k, k); H113)."""

    __slots__ = ()
    _TITLE = "Tukey's HSD test"
    _FIELDS = ("statistic", "pvalue", "df")
    _CI = True

    @_calls.method("summary")
    def summary(self, *, alpha=0.05):
        """The comparisons i < j as a table (H74): mean difference, p-value (Games-Howell's df per
        pair) and the 1 - alpha interval."""
        a = C.alpha(alpha, "summary")
        info = self._info
        st, pv, df = self.statistic.tolist(), self.pvalue.tolist(), self.df
        lo, hi = self._ci(a)
        lo, hi = lo.tolist(), hi.tolist()
        k = len(st)
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        labels = [f"({i} - {j})" for i, j in pairs]
        columns = [
            ("statistic", [st[i][j] for i, j in pairs], _S.forg),
            ("pvalue", [pv[i][j] for i, j in pairs], _S.pvalue),
        ]
        left = []
        if isinstance(df, float):
            left.append(("Df:", _S.df(df)))
        else:
            dfl = df.tolist()
            columns.append(("df", [dfl[i][j] for i, j in pairs], _S.df))
        right = [("No. Groups:", str(k))] + _sizes(info)
        interval = ([lo[i][j] for i, j in pairs], [hi[i][j] for i, j in pairs])
        if not left:
            left, right = right, []
        return _comparisons(info.get("title") or self._TITLE, left, right, labels, columns, a, "two-sided", interval)


class DunnettResult(_TestResult):
    """statistic[i] and pvalue[i] of treatment i against the control; df the pooled degrees of
    freedom (H113)."""

    __slots__ = ()
    _TITLE = "Dunnett's test"
    _FIELDS = ("statistic", "pvalue", "df")
    _CI = True

    @_calls.method("summary")
    def summary(self, *, alpha=0.05):
        """The comparisons with the control as a table (H74): statistic, p-value and the 1 - alpha
        interval of the mean difference."""
        a = C.alpha(alpha, "summary")
        info = self._info
        alt = info.get("alternative", "two-sided")
        lo, hi = self._ci(a)
        st, pv = self.statistic.tolist(), self.pvalue.tolist()
        labels = [f"({i} - control)" for i in range(len(st))]
        columns = [("statistic", st, _S.forg), ("pvalue", pv, _S.pvalue)]
        left = [("Df:", _S.df(self.df))]
        right = [("Alternative:", alt)] + _sizes(info)
        if "control" in info:
            right.append(("Control Observations:", str(info["control"])))
        return _comparisons(self._TITLE, left, right, labels, columns, a, alt, (lo.tolist(), hi.tolist()))


class MultipletestsResult(_TestResult):
    """Unpacks as statsmodels' (reject, pvals_corrected, alphacSidak, alphacBonf) (H104)."""

    __slots__ = ()
    _TITLE = "Multiple testing correction"
    _FIELDS = ("reject", "pvalues_corrected", "alpha_sidak", "alpha_bonferroni")
    _UNPACK = ("reject", "pvalues_corrected", "alpha_sidak", "alpha_bonferroni")

    @_calls.method("summary")
    def summary(self):
        """The correction as a table (H74): the levels in the header, a row per p-value (its
        index in the input) of the corrected p-value and the rejection."""
        info = self._info
        v = self._values
        rej, pc = v["reject"], v["pvalues_corrected"]
        if isinstance(pc, float):
            shape, pcs, rjs = (1,), [pc], [rej]
        else:
            shape = tuple(pc.shape)
            pcs, rjs = _sd.reshape(pc, (pc.size,)).tolist(), _sd.reshape(rej, (rej.size,)).tolist()
        n = len(pcs)
        left = [
            ("Alpha:", f"{info['alpha']:g}"),
            ("Alpha Sidak:", _S.stat(v["alpha_sidak"])),
            ("Alpha Bonferroni:", _S.stat(v["alpha_bonferroni"])),
        ]
        right = [("Method:", info["method"]), ("No. Tests:", str(info["m"])), ("No. Rejected:", str(sum(rjs)))]
        if len(shape) > 1 or info.get("batch"):
            right.append(("Shape:", _S.shape_text(shape)))
        labels = [_S.lane_label(i, shape) for i in range(n)]
        columns = [("corrected", pcs, _S.pvalue), ("reject", rjs, _S.flag)]
        return _comparisons(self._TITLE, left, right, labels, columns, None, "two-sided", None)


class CombinePvaluesResult(_TestResult):
    __slots__ = ()
    _TITLE = "Combined p-values"
    _FIELDS = ("statistic", "pvalue")
    _UNPACK = ("statistic", "pvalue")


# ---- helpers ------------------------------------------------------------------------------


def _min_samples(samples, name, least, what):
    if len(samples) < least:
        raise TypeError(f"{name}(): at least {what} required, not {len(samples)}")


def _check_p(a, name, arg, nan=False):
    """p-values within [0, 1] (H112; NaN only where nan_policy decides)."""
    if not a.size:
        return
    ok = (a >= 0.0) & (a <= 1.0)
    if nan:
        ok = ok | _sd.isnan(a)
    if not bool(_sd.all(ok)):
        rule = "between 0 and 1" if nan else "between 0 and 1 (and no NaN)"
        raise ValueError(f"{name}(): '{arg}' must hold p-values {rule}")


def _one_axis(axis, name):
    """axis: an int or None (one family per lane, as sd.stats' rankdata: T37)."""
    if axis is None:
        return None
    try:
        if isinstance(axis, bool):
            raise TypeError
        return operator.index(axis)
    except TypeError:
        raise TypeError(f"{name}(): 'axis' must be an int or None, not {_type_name(axis)}") from None


def _families(x, axis, name, arg):
    """The p-values as (lanes, m) rows (axis=None: one family of every value, a 1-D result as
    scipy's and rankdata's) and the way back to the result's shape."""
    a = _as_array(x, name, arg)
    ax = _one_axis(axis, name)
    if ax is None:
        a = _sd.reshape(a, (a.size,))
        ax = 0
    _check_p(a, name, arg)
    return a, _Lanes(a, ax, name)


def _corrected(x, axis, name, arg, code, alpha, max_iter):
    """The corrected p-values and the rejections at alpha (the input's shape)."""
    a, lanes = _families(x, axis, name, arg)
    out, rej = _k.multitest(lanes.rows, code, alpha, max_iter)
    return lanes.back(out), lanes.back(rej) != 0.0, lanes, _scalar_value(x) is not None


# ---- tukey_hsd and dunnett ----------------------------------------------------------------


def tukey_hsd(*samples, equal_var=True):
    """Tukey's honestly significant difference test of every pair of k >= 2 samples (Tukey-Kramer
    for unequal sizes; equal_var=False: Games and Howell's): statistic (k, k) the mean
    differences mean_i - mean_j, pvalue (k, k) the studentized range's upper tail of
    |statistic| / standard error, df (N - k, or Games-Howell's Welch df per pair, (k, k)), and
    conf_int(alpha=0.05) (k, k, 2). Does not unpack (scipy's result does not)."""
    name = "tukey_hsd"
    _min_samples(samples, name, 2, "two samples are")
    C.flag(equal_var, name, "equal_var")
    k = len(samples)
    args = tuple(f"samples[{i}]" for i in range(k))
    s = C.Samples(name, list(samples), args, None, "propagate", scan=False)
    stat, p, se, df = _k.tukey(s.rows, equal_var)
    shape = (k, k)
    stat, p, se, df = (_sd.reshape(v, shape) for v in (stat, p, se, df))
    dfs = df.tolist()
    crits = {}

    def ci(alpha):
        if alpha not in crits:
            if equal_var:
                crits[alpha] = float(_k.srange_isf(alpha, float(k), _sd.asarray([dfs[0][0]]))[0])
            else:  # one critical value per pair (its Welch df), each pair once
                ij = [(i, j) for i in range(k) for j in range(i, k)]
                q = _k.srange_isf(alpha, float(k), _sd.asarray([dfs[i][j] for i, j in ij])).tolist()
                full = [[0.0] * k for _ in range(k)]
                for (i, j), v in zip(ij, q):
                    full[i][j] = full[j][i] = v
                crits[alpha] = _sd.asarray(full)
        r = crits[alpha] * se
        return stat - r, stat + r

    info = {"nobs": tuple(s.n), "title": None if equal_var else "Games-Howell test"}
    return TukeyHsdResult._make(ci=ci, info=info, statistic=stat, pvalue=p, df=float(dfs[0][0]) if equal_var else df)


def dunnett(*samples, control, alternative="two-sided"):
    """Dunnett's test of each sample's mean against the control's: statistic (the t statistics,
    one per sample), pvalue (the multivariate t's family-wise tail: P(max |T_j| > |t_i|),
    one-sided P(max T_j > t_i) or P(min T_j < t_i)), df (N - k - 1) and conf_int(alpha=0.05)
    (k, 2) of the mean differences. Deterministic: no rng (H108; scipy's is a randomized
    quasi-Monte Carlo). Does not unpack."""
    name = "dunnett"
    _min_samples(samples, name, 1, "one sample besides the control is")
    alt = C.alternative(alternative, name)
    d = len(samples)
    args = tuple(f"samples[{i}]" for i in range(d)) + ("control",)
    s = C.Samples(name, list(samples) + [control], args, None, "propagate", scan=False)
    stat, p, diff, scale, df = _k.dunnett(s.rows, _ALT[alt])
    ns = _sd.asarray([float(v) for v in s.n[:d]])
    n0 = float(s.n[d])
    crits = {}

    def ci(alpha):
        if alpha not in crits:
            crits[alpha] = _k.dunnett_isf(alpha, ns, n0, df, alt == "two-sided")
        r = crits[alpha] * scale
        lo, hi = diff - r, diff + r
        if alt == "greater":
            hi = _sd.full((d,), math.inf)
        elif alt == "less":
            lo = _sd.full((d,), -math.inf)
        return lo, hi

    info = {"alternative": alt, "nobs": tuple(s.n[:d]), "control": s.n[d]}
    return DunnettResult._make(ci=ci, info=info, statistic=stat, pvalue=p, df=df)


# ---- the corrections ----------------------------------------------------------------------


def multipletests(pvals, /, *, alpha=0.05, method="holm-sidak", max_iter=1, axis=None):
    """statsmodels' multiple testing corrections: reject (the procedure's decision at alpha), the
    corrected p-values (the input's shape; axis=None: one family of every value, 1-D), and the
    Sidak and Bonferroni per-test levels 1 - (1 - alpha)^(1/m) and alpha / m. method: one of
    statsmodels' long names ('bonferroni', 'sidak', 'holm-sidak', 'holm', 'simes-hochberg',
    'hommel', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'); max_iter the two-stage methods'
    stages (statsmodels' maxiter; H110). Unpacks as statsmodels' four."""
    name = "multipletests"
    a = C.alpha(alpha, name)
    meth = C.choice(method, name, "method", tuple(_METHODS))
    mi = C.integer(max_iter, name, "max_iter", minimum=0)
    out, reject, lanes, scalar = _corrected(pvals, axis, name, "pvals", _METHODS[meth], a, mi)
    m = lanes.n
    if m:
        a_sidak = -math.expm1(math.log1p(-a) / m)
        a_bonf = a / m
    else:
        a_sidak = a_bonf = math.nan
    if scalar:
        out, reject = float(out[0]), bool(reject[0])
    info = {"alpha": a, "method": meth, "m": m, "batch": len(lanes.kept_shape) > 0}
    return MultipletestsResult._make(
        info=info, reject=reject, pvalues_corrected=out, alpha_sidak=a_sidak, alpha_bonferroni=a_bonf
    )


def false_discovery_control(ps, /, *, axis=None, method="bh"):
    """scipy's false discovery rate adjustment: Benjamini-Hochberg's ('bh') or
    Benjamini-Yekutieli's ('by') adjusted p-values, an array of the input's shape (axis=None: of
    every value, 1-D, as scipy). The same bits as multipletests(method='fdr_bh' or 'fdr_by')."""
    name = "false_discovery_control"
    meth = C.choice(method, name, "method", tuple(_FDC))
    out, _, _, scalar = _corrected(ps, axis, name, "ps", _FDC[meth], 0.05, 1)
    return float(out[0]) if scalar else out


def combine_pvalues(pvalues, /, *, axis=None, method="fisher", weights=None, nan_policy="propagate", keepdims=False):
    """scipy's combination of independent p-values: statistic and pvalue of 'fisher'
    (-2 sum log p, chi^2 with 2n degrees of freedom), 'pearson' (2 sum log(1 - p), its lower
    tail), 'tippett' (min p), 'stouffer' (sum w_i isf(p_i) / |w|, the normal; weights paired
    with the p-values) or 'mudholkar_george' (-sum log(p / (1 - p)), scipy's t approximation).
    Unpacks as (statistic, pvalue)."""
    name = "combine_pvalues"
    meth = C.choice(method, name, "method", tuple(_COMBINE))
    xs = [pvalues] if weights is None else [pvalues, weights]
    s = C.Samples(name, xs, ("pvalues", "weights"), axis, nan_policy, paired=weights is not None,
                  keepdims=keepdims, scan=False)  # fmt: skip
    _check_p(s.rows[0], name, "pvalues", nan=True)
    stat, p = _k.combine(s.rows[0], s.rows[1] if weights is not None else None, _COMBINE[meth], s.policy == "omit")
    info = s.info(title=f"Combined p-values ({_COMBINE_TITLES[meth]})")
    n = info.pop("nobs", None)
    if n is not None:
        info["pairs"] = [("No. P-values:", str(n))]
    return CombinePvaluesResult._make(batch=s.batch, info=info, statistic=s.out(stat), pvalue=s.out(p))


# a wrong call shape in the library's form (API.md §2.1)
tukey_hsd = _calls.function(tukey_hsd)
dunnett = _calls.function(dunnett)
multipletests = _calls.function(multipletests)
false_discovery_control = _calls.function(false_discovery_control)
combine_pvalues = _calls.function(combine_pvalues)
