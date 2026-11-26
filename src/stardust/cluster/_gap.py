"""The gap statistic (docs/api/cluster.md §6; docs/design/cluster.md §2.6, §3.9, §4.6; decisions CL37, CL41 and those
of milestone CL5 in §1.1): R cluster's clusGap and maxSE.

For k = 1 .. k_max, log W_k of the data (W_k = 1/2 sum over the clusters of the sum of d^d_power over the cluster's
pairs divided by its size: R's W.k) and of n_reference reference sets drawn uniformly in the box of the data's
principal axes ('scaledpca', R's spaceH0 = "scaledPCA") or of its own axes ('original'); the gap is the reference
mean of log W less the data's, SE its standard deviation times sqrt(1 + 1/B), and k the choice of maxSE's rule.
The clusterings are sd.cluster.kmeans (k-means++, one start; method='kmeans') or kmedoids (PAM; method='pam') of
every set; the reference sets and their clusterings run side by side in C++ (csrc/cluster/metrics.cpp), each with
its own random elements of the call's stream (CL41), so the result is the same for any thread count.
"""

import math

from .. import _calls, _summary
from .. import _core as _sd
from .._calls import function
from ._common import check_rng, choice, contiguous, key, points, positive_int, real
from ._common import rows as _rows
from ._results import ClusterResult

_core = _sd.cluster
METHODS = ("kmeans", "pam")
REFERENCES = ("scaledpca", "original")
SELECTS = ("firstsemax", "tibs2001semax", "globalsemax", "firstmax", "globalmax")
_SUB_START = 2**23  # the clusterings' sub-keys: start 2^23 + s k_max + k (the reference sets use starts 0 .. B - 1)


def _lower_choice(v, func, arg, choices):
    """choice() that points a different case to the lowercase name (as optimize's methods: O15)."""
    if isinstance(v, str) and v not in choices and v.lower() in choices:
        raise ValueError(f"{func}(): '{arg}' must be lowercase: {v.lower()!r}, not {v!r}")
    return choice(v, func, arg, choices)


def max_se(f, se, method, factor):
    """R cluster's maxSE: the chosen k (1-based) of the gaps f and their standard errors se."""
    K = len(f)
    fse = [factor * s for s in se]

    def first_max():
        for j in range(K - 1):
            if f[j + 1] <= f[j]:
                return j + 1
        return K

    def global_max():
        best = 0
        for j in range(1, K):
            if f[j] > f[best]:
                best = j
        return best + 1

    def left_of(nc):
        for j in range(nc - 1):
            if f[j] >= f[nc - 1] - fse[nc - 1]:
                return j + 1
        return nc

    if method == "firstmax":
        return first_max()
    if method == "globalmax":
        return global_max()
    if method == "tibs2001semax":
        for j in range(K - 1):
            if f[j] >= f[j + 1] - fse[j + 1]:
                return j + 1
        return K
    if method == "firstsemax":
        return left_of(first_max())
    return left_of(global_max())


class GapResult(ClusterResult):
    """The result of gap_statistic (docs/api/cluster.md §6).

    Fields: k (the chosen number of clusters), k_max, gap, se, log_w, e_log_w ((k_max,) for k = 1 .. k_max: R's Tab
    columns gap, SE.sim, logW, E.logW), log_w_reference ((n_reference, k_max): each reference set's log W), nobs,
    omitted, method, reference, select, se_factor, n_reference and d_power. No labels: the statistic chooses k.
    """

    _TITLE = "Gap Statistic Results"
    _REPR = ("nobs", "k", "k_max", "method")

    def summary(self):
        """The box: the method, the sample, the reference sets and the rule on the left and right; one row per k
        (log W, E[log W], gap, SE), the chosen k marked."""
        left = [
            ("Method:", f"gap statistic ({self.method})"),
            ("No. Observations:", _summary.count(self.nobs)),
            ("Reference sets:", str(int(self.n_reference))),
            ("Chosen k:", str(int(self.k))),
        ]
        omitted = int(_sd.count_nonzero(self.omitted))
        if omitted:
            left.insert(2, ("Omitted rows (NaN):", str(omitted)))
        right = [
            ("Reference:", self.reference),
            ("Select:", self.select),
            ("SE factor:", _summary.stat(self.se_factor)),
            ("d power:", _summary.stat(self.d_power)),
        ]
        cols = [self.log_w.tolist(), self.e_log_w.tolist(), self.gap.tolist(), self.se.tolist()]
        rows = []
        for i in _summary.shown(int(self.k_max)):
            if i is None:
                rows.append(None)
                continue
            mark = " *" if i + 1 == self.k else ""
            rows.append((f"{i + 1}{mark}", [_summary.forg(c[i], 4) for c in cols]))
        table = _summary.Table(["log W", "E[log W]", "Gap", "SE"], rows, stub_head="k")
        return _summary.render(self._TITLE, header=(left, right), tables=(table,))

    def __repr__(self):
        return _summary.one_line(type(self).__name__, [(k, getattr(self, k)) for k in self._REPR])


_calls.methods(GapResult)


@function
def gap_statistic(
    X,
    k_max,
    /,
    *,
    rng,
    method="kmeans",
    n_reference=100,
    d_power=1,
    reference="scaledpca",
    select="firstsemax",
    se_factor=1.0,
    nan_policy="raise",
):
    """The gap statistic of Tibshirani, Walther and Hastie (2001) as R cluster's clusGap computes it, and the number
    of clusters its rule chooses.

    X holds the points ((n, p)); k_max >= 2 the largest k tried (at most the number of distinct points). rng (an
    sd.random.Generator, required) draws the reference sets and the k-means starts: two 64-bit words a call. method
    'kmeans' (sd.cluster.kmeans with its defaults) or 'pam' (sd.cluster.kmedoids); n_reference reference sets (R's
    B, at least 2) drawn uniformly in the box of the centred data's principal axes ('scaledpca') or of its columns
    ('original'); d_power the power of the Euclidean distances in W (R's d.power = 1; Tibshirani et al. use 2).
    select is R maxSE's rule in lowercase ('firstsemax', 'tibs2001semax', 'globalsemax', 'firstmax', 'globalmax')
    with se_factor (SE.factor). nan_policy 'raise' (the default) or 'omit'.
    """
    func = "gap_statistic"
    method = choice(method, func, "method", METHODS)
    reference = _lower_choice(reference, func, "reference", REFERENCES)
    select = _lower_choice(select, func, "select", SELECTS)
    Xp = points(X, func)
    n_input = Xp.shape[0]
    Xp, _, omitted = _rows(Xp, func, nan_policy)
    n, p = Xp.shape
    k_max = positive_int(k_max, func, "k_max")
    if k_max < 2:
        raise ValueError(f"{func}(): 'k_max' must be >= 2, not {k_max}")
    B = positive_int(n_reference, func, "n_reference")
    if B < 2:
        raise ValueError(f"{func}(): 'n_reference' must be >= 2 (the standard error needs two sets), not {B}")
    power = real(d_power, func, "d_power")
    if not (power > 0 and math.isfinite(power)):
        raise ValueError(f"{func}(): 'd_power' must be finite and > 0, not {power!r}")
    factor = real(se_factor, func, "se_factor")
    if not (factor >= 0 and math.isfinite(factor)):
        raise ValueError(f"{func}(): 'se_factor' must be finite and >= 0, not {factor!r}")
    rng = check_rng(rng, func)
    order = _core.lex_order(Xp)
    distinct = _core.kmeans_distinct(Xp, order, None)
    if k_max > distinct:
        raise ValueError(f"{func}(): 'k_max' ({k_max}) is larger than the number of distinct points ({distinct})")
    if (B + 2) * k_max >= _SUB_START or n * p >= 2**40:
        raise ValueError(f"{func}(): too many reference sets times k_max, or points times columns, for the stream")
    k0, k1 = key(rng)
    log_w, e_log_w, se, ref = _core.gap(Xp, order, k_max, B, power, REFERENCES.index(reference),
                                        METHODS.index(method), k0, k1)  # fmt: skip
    gap = contiguous(e_log_w - log_w)
    k = max_se(gap.tolist(), se.tolist(), select, factor)
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    return GapResult(
        k=k,
        k_max=k_max,
        gap=gap,
        se=se,
        log_w=log_w,
        e_log_w=e_log_w,
        log_w_reference=ref,
        nobs=n,
        omitted=om,
        method=method,
        reference=reference,
        select=select,
        se_factor=factor,
        n_reference=B,
        d_power=power,
    )
