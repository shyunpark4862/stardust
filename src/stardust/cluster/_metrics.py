"""The clustering indices (docs/api/cluster.md §6; docs/design/cluster.md §2.6, §3.8, §4.6; decisions CL13, CL14,
CL36 and those of milestone CL5 in §1.1): scikit-learn 1.9.1's sklearn.metrics.cluster under its names.

External indices compare two labelings through their contingency table (csrc/cluster/metrics.cpp): the pair counts
are exact integers, so ARI, RI and the pair confusion matrix are the exact rationals rounded once (Python's int
division is correctly rounded) and Fowlkes-Mallows the correctly rounded square root of an exact rational; the
mutual information and the entropies are sums of terms whose logarithm has a relative error of a few u, and AMI's
expected mutual information is the hypergeometric sum by a ratio recurrence (no lgamma differences). Indices whose
exact value lies in [0, 1] are clamped there (scikit-learn's homogeneity can read 1.0000000000000002).

Internal indices (silhouette, Calinski-Harabasz, Davies-Bouldin) take the points in key order (CL7) and compute
every distance by sd.spatial's kernels, so the same points in another order give the same values bit for bit.

Labels: 1-D integers, bools or strings (a float array whose values are whole numbers too); the clusters are the
distinct values, ordered as numpy's np.unique orders them (the contingency matrix's rows and columns).
"""

import collections
import math
import numbers

from .. import _core as _sd
from .._calls import function
from ._common import check_alloc, choice, contiguous, distances, metric_columns, points, real, type_name
from ._common import metric as _metric

_core = _sd.cluster
_AVERAGES = ("min", "geometric", "arithmetic", "max")
_EPS = 2.0**-52
_FAST = {"euclidean": 0, "sqeuclidean": 1, "cityblock": 2, "chebyshev": 3}
_MAX_N = 3037000499  # n (n - 1) < 2^63: the pair counts (pair_confusion_matrix's entries) fit in int64


# homogeneity_completeness_v_measure's result
HomogeneityResult = collections.namedtuple("HomogeneityResult", ["homogeneity", "completeness", "v_measure"])


# ---- labels ------------------------------------------------------------------------------------------------------


def _strings(values, func, arg):
    # the distinct values in first-seen order (a str never equals another type, so the first value that is not a
    # str is the first such key), then their sorted numbering mapped over the values at C speed
    first = dict.fromkeys(values)
    for v in first:
        if not isinstance(v, str):
            raise TypeError(f"{func}(): '{arg}' must hold integers, bools or strings, not {type_name(v)}")
    uniq = sorted(first)
    code = {u: i for i, u in enumerate(uniq)}
    return _sd.asarray(list(map(code.__getitem__, values)), dtype=_sd.int64), len(uniq), uniq


def _labels(v, func, arg):
    """(codes: a C (n,) int64 array of 0 .. K - 1 in the sorted order of the distinct labels, K, the labels)."""
    if isinstance(v, numbers.Number) or isinstance(v, (str, bytes)) or v is None:
        raise TypeError(f"{func}(): '{arg}' must be a 1-D array of labels, not {type_name(v)}")
    mod = type(v).__module__
    if mod.startswith("stardust.frame"):
        from .. import frame as _frame

        if isinstance(v, _frame.Series):
            if bool(_sd.any(_sd.asarray(v.isna().tolist()))):
                raise ValueError(f"{func}(): '{arg}' contains missing labels")
            if str(v.dtype) == "string":
                return _strings(v.tolist(), func, arg)
            v = v.to_array()
        else:
            raise TypeError(f"{func}(): '{arg}' must be a 1-D array of labels, not {type_name(v)}")
    elif mod == "numpy" and getattr(v, "dtype", None) is not None and v.dtype.kind in "USO":
        if v.ndim != 1:
            raise ValueError(f"{func}(): '{arg}' must be 1-D, not {v.ndim}-D")
        vals = v.tolist()
        if v.dtype.kind != "O" or all(isinstance(x, str) for x in vals):
            return _strings(vals, func, arg)
        v = vals  # an object array of numbers: as a list
    if not isinstance(v, _sd.ndarray):
        try:
            v = _sd.asarray(v)
        except TypeError:
            vals = list(v)
            if vals and all(isinstance(x, str) for x in vals):
                return _strings(vals, func, arg)
            raise TypeError(f"{func}(): '{arg}' must hold integers, bools or strings") from None
        except ValueError as e:
            raise ValueError(f"{func}(): '{arg}': {e}") from None
    if v.ndim != 1:
        raise ValueError(f"{func}(): '{arg}' must be 1-D, not {v.ndim}-D")
    if v.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must hold integers, bools or strings, not complex")
    if v.dtype == _sd.float64:
        if not bool(_sd.all(_sd.isfinite(v))) or not bool(_sd.all(_sd.floor(v) == v)):
            raise ValueError(f"{func}(): float labels in '{arg}' must be whole numbers")
    vals, inv = _sd.unique_inverse(v)
    return contiguous(inv.astype(_sd.int64)), int(vals.shape[0]), vals


def _two(labels_true, labels_pred, func):
    a, ka, ua = _labels(labels_true, func, "labels_true")
    b, kb, ub = _labels(labels_pred, func, "labels_pred")
    n, m = int(a.shape[0]), int(b.shape[0])
    if n != m:
        raise ValueError(f"{func}(): 'labels_true' and 'labels_pred' must have the same length, not {n} and {m}")
    if n == 0:
        raise ValueError(f"{func}(): the labelings must have at least 1 point")
    if n > _MAX_N:
        raise ValueError(f"{func}(): at most {_MAX_N} points (the pair counts must fit in int64), not {n}")
    return a, ka, ua, b, kb, ub, n


def _counts(labels_true, labels_pred, func, emi=False):
    """(n, ka, kb, sc, sa, sb, mi, h_true, h_pred, emi): the pair counts are Python ints."""
    a, ka, _, b, kb, _, n = _two(labels_true, labels_pred, func)
    sc, sa, sb, mi, ha, hb, e = _core.external(a, b, ka, kb, emi)
    return n, ka, kb, int(sc), int(sa), int(sb), float(mi), float(ha), float(hb), float(e)


def _pairs(n, sc, sa, sb):
    """(tn, fp, fn, tp): the unordered pairs (scikit-learn's pair confusion matrix halved)."""
    tp, fp, fn = sc, sb - sc, sa - sc
    return n * (n - 1) // 2 - tp - fp - fn, fp, fn, tp


def _sqrt_ratio(num, den):
    """sqrt(num / den) correctly rounded, for integers num, den > 0 (an integer root with a sticky bit)."""
    s = 60 - (num.bit_length() - den.bit_length()) // 2
    q, rem = divmod(num << (2 * s), den) if s >= 0 else divmod(num, den << (-2 * s))
    r = math.isqrt(q)
    m = (r << 1) | (0 if rem == 0 and r * r == q else 1)
    return math.ldexp(float(m), -s - 1)


def _average(h_true, h_pred, method):
    if method == "min":
        return min(h_true, h_pred)
    if method == "max":
        return max(h_true, h_pred)
    if method == "geometric":
        return math.sqrt(h_true * h_pred)
    return (h_true + h_pred) / 2.0


def _beta(v, func):
    b = real(v, func, "beta")
    if not (b >= 0 and math.isfinite(b)):
        raise ValueError(f"{func}(): 'beta' must be finite and >= 0, not {b!r}")
    return b


def _v(mi, h_true, h_pred, beta):
    """V = (1 + beta) MI / (H_true + beta H_pred) (= the harmonic form of homogeneity and completeness)."""
    if beta == 0.0:
        return 1.0 if h_true == 0.0 else min(1.0, mi / h_true)
    if h_true == 0.0 and h_pred == 0.0:
        return 1.0
    return min(1.0, ((1.0 + beta) * mi) / (h_true + beta * h_pred))


# ---- external indices --------------------------------------------------------------------------------------------


@function
def adjusted_rand_score(labels_true, labels_pred, /):
    """The adjusted Rand index of two labelings: (RI - E[RI]) / (max RI - E[RI]) over the pairs of points, 1 for equal
    partitions, about 0 for independent ones. The exact rational of the pair counts, correctly rounded."""
    n, _, _, sc, sa, sb, *_ = _counts(labels_true, labels_pred, "adjusted_rand_score")
    tn, fp, fn, tp = _pairs(n, sc, sa, sb)
    if fn == 0 and fp == 0:
        return 1.0
    return (2 * (tp * tn - fn * fp)) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))


@function
def rand_score(labels_true, labels_pred, /):
    """The Rand index: the share of the pairs of points that the two labelings treat alike (both together or both
    apart). The exact rational, correctly rounded; 1 without pairs."""
    n, _, _, sc, sa, sb, *_ = _counts(labels_true, labels_pred, "rand_score")
    tn, fp, fn, tp = _pairs(n, sc, sa, sb)
    pairs = tn + fp + fn + tp
    if pairs == 0 or tp + tn == pairs:
        return 1.0
    return (tp + tn) / pairs


@function
def pair_confusion_matrix(labels_true, labels_pred, /):
    """scikit-learn's 2 x 2 pair confusion matrix (int64): the ordered pairs of distinct points [[apart in both,
    together only in labels_pred], [together only in labels_true, together in both]]."""
    n, _, _, sc, sa, sb, *_ = _counts(labels_true, labels_pred, "pair_confusion_matrix")
    tn, fp, fn, tp = _pairs(n, sc, sa, sb)
    return _sd.asarray([[2 * tn, 2 * fp], [2 * fn, 2 * tp]], dtype=_sd.int64)


@function
def fowlkes_mallows_score(labels_true, labels_pred, /):
    """The Fowlkes-Mallows index TP / sqrt((TP + FP)(TP + FN)) over the pairs of points (0 without a pair together
    in both): the correctly rounded square root of the exact rational."""
    n, _, _, sc, sa, sb, *_ = _counts(labels_true, labels_pred, "fowlkes_mallows_score")
    if sc == 0:
        return 0.0
    return _sqrt_ratio(sc * sc, sa * sb)


@function
def mutual_info_score(labels_true, labels_pred, /):
    """The mutual information sum_ij (n_ij / n) log(n n_ij / (a_i b_j)) of two labelings (natural log), >= 0."""
    return _counts(labels_true, labels_pred, "mutual_info_score")[6]


@function
def normalized_mutual_info_score(labels_true, labels_pred, /, *, average_method="arithmetic"):
    """MI / mean(H_true, H_pred), the mean by average_method ('min', 'geometric', 'arithmetic' (the default) or
    'max'); 1 when both labelings are one cluster, 0 when MI is 0. In [0, 1]."""
    func = "normalized_mutual_info_score"
    average_method = choice(average_method, func, "average_method", _AVERAGES)
    n, ka, kb, _, _, _, mi, ha, hb, _ = _counts(labels_true, labels_pred, func)
    if ka == kb == 1:
        return 1.0
    if mi == 0.0:
        return 0.0
    return min(1.0, mi / _average(ha, hb, average_method))


@function
def adjusted_mutual_info_score(labels_true, labels_pred, /, *, average_method="arithmetic"):
    """(MI - E[MI]) / (mean(H_true, H_pred) - E[MI]), E[MI] the mutual information expected of two random labelings
    with the same cluster sizes (Vinh et al. 2010). 1 for equal partitions that are one cluster or all singletons; 0
    when one labeling is one cluster and the other is not. When one side is all singletons every relabeling has the
    same MI (= E[MI] = the other side's entropy): 0, but 1 for 'min', whose normalizer is that entropy too (a
    refinement's AMI is 1 with 'min'). Else scikit-learn's form, its denominator kept at least 2^-52 away from 0."""
    func = "adjusted_mutual_info_score"
    average_method = choice(average_method, func, "average_method", _AVERAGES)
    a, ka, _, b, kb, _, n = _two(labels_true, labels_pred, func)
    if ka == kb == 1 or (ka == kb == n):
        return 1.0
    if ka == 1 or kb == 1:
        return 0.0
    if ka == n or kb == n:  # every relabeling has MI = E[MI] = the other side's entropy
        return 1.0 if average_method == "min" else 0.0
    _, _, _, mi, ha, hb, emi = _core.external(a, b, ka, kb, True)
    den = _average(ha, hb, average_method) - emi
    den = min(den, -_EPS) if den < 0 else max(den, _EPS)
    return min(1.0, (mi - emi) / den)


@function
def homogeneity_score(labels_true, labels_pred, /):
    """MI / H_true: 1 when every cluster of labels_pred holds points of one class only (1 when H_true = 0)."""
    _, _, _, _, _, _, mi, ha, _, _ = _counts(labels_true, labels_pred, "homogeneity_score")
    return 1.0 if ha == 0.0 else min(1.0, mi / ha)


@function
def completeness_score(labels_true, labels_pred, /):
    """MI / H_pred: 1 when every class of labels_true lies in one cluster (1 when H_pred = 0)."""
    _, _, _, _, _, _, mi, _, hb, _ = _counts(labels_true, labels_pred, "completeness_score")
    return 1.0 if hb == 0.0 else min(1.0, mi / hb)


@function
def v_measure_score(labels_true, labels_pred, /, *, beta=1.0):
    """The V-measure (1 + beta) h c / (beta h + c) of homogeneity h and completeness c, computed as
    (1 + beta) MI / (H_true + beta H_pred); with beta = 1 it is normalized_mutual_info_score (arithmetic), bit for
    bit."""
    func = "v_measure_score"
    beta = _beta(beta, func)
    _, _, _, _, _, _, mi, ha, hb, _ = _counts(labels_true, labels_pred, func)
    return _v(mi, ha, hb, beta)


@function
def homogeneity_completeness_v_measure(labels_true, labels_pred, /, *, beta=1.0):
    """HomogeneityResult(homogeneity, completeness, v_measure) from one contingency table."""
    func = "homogeneity_completeness_v_measure"
    beta = _beta(beta, func)
    _, _, _, _, _, _, mi, ha, hb, _ = _counts(labels_true, labels_pred, func)
    h = 1.0 if ha == 0.0 else min(1.0, mi / ha)
    c = 1.0 if hb == 0.0 else min(1.0, mi / hb)
    return HomogeneityResult(h, c, _v(mi, ha, hb, beta))


@function
def contingency_matrix(labels_true, labels_pred, /, *, sparse=False):
    """The (classes, clusters) int64 contingency table: entry (i, j) counts the points of the i-th label of
    labels_true and the j-th of labels_pred (labels in sorted order). sparse=True gives an sd.sparse.csr_array."""
    func = "contingency_matrix"
    if not isinstance(sparse, bool):
        raise TypeError(f"{func}(): 'sparse' must be a bool, not {type_name(sparse)}")
    a, ka, _, b, kb, _, _ = _two(labels_true, labels_pred, func)
    rows, cols, cnt, _, _ = _core.contingency(a, b, ka, kb)
    if sparse:
        from .. import sparse as _sparse

        return _sparse.csr_array((cnt, (rows, cols)), shape=(ka, kb))
    check_alloc(8 * ka * kb, func)
    M = _sd.zeros((ka * kb,), dtype=_sd.int64)
    M[rows * kb + cols] = cnt
    return _sd.reshape(M, (ka, kb))


# ---- internal indices --------------------------------------------------------------------------------------------


def _internal(X, labels, func, metric=None):
    """(the data for the core: (X in key order, None) or (None, condensed D), the codes in key order, K, order)."""
    codes, K, _ = _labels(labels, func, "labels")
    if metric == "precomputed":
        D, n = distances(X, func)
        if codes.shape[0] != n:
            raise ValueError(f"{func}(): 'labels' must have one label per point ({n}), not {codes.shape[0]}")
        data, order = (None, D), None
    else:
        Xp = points(X, func)
        n = Xp.shape[0]
        if codes.shape[0] != n:
            raise ValueError(f"{func}(): 'labels' must have one label per point ({n}), not {codes.shape[0]}")
        if bool(_sd.any(_sd.isnan(Xp))):
            raise ValueError(f"{func}(): 'X' contains NaN")
        if bool(_sd.any(_sd.isinf(Xp))):
            raise ValueError(f"{func}(): 'X' contains infinity")
        order = _core.lex_order(Xp)
        Xk = contiguous(Xp[order])
        codes = contiguous(codes[order])
        if metric is None or metric in _FAST:
            data = (Xk, None)
        else:
            from .. import spatial as _spatial

            metric_columns(metric, Xk.shape[1], func)
            check_alloc(8 * (n * (n - 1) // 2), func)
            data = (None, contiguous(_spatial.pdist(Xk, metric=metric)))
    if not 2 <= K <= n - 1:
        raise ValueError(f"{func}(): the number of clusters must be between 2 and n - 1 ({n - 1}), not {K}")
    return data, codes, K, order


def _back(values, order):
    if order is None:
        return values
    out = _sd.empty(tuple(values.shape), dtype=_sd.float64)
    out[order] = values
    return out


def _silhouette(X, labels, metric, func):
    metric = _metric(metric, func)
    (Xk, D), codes, K, order = _internal(X, labels, func, metric)
    vals, mean = _core.silhouette(Xk, D, codes, K, _FAST.get(metric, 0) if Xk is not None else 0)
    return vals, float(mean), order


@function
def silhouette_samples(X, labels, /, *, metric="euclidean"):
    """Each point's silhouette (b - a) / max(a, b): a its mean distance to the other points of its cluster, b the
    smallest mean distance to another cluster; 0 for a point alone in its cluster. X: observations ((n, p), metric
    one of sd.spatial's names) or distances (metric='precomputed': an (n, n) matrix or a condensed vector). Every
    pair, without an n x n matrix for the euclidean, sqeuclidean, cityblock and chebyshev observations."""
    vals, _, order = _silhouette(X, labels, metric, "silhouette_samples")
    return _back(vals, order)


@function
def silhouette_score(X, labels, /, *, metric="euclidean"):
    """The mean of silhouette_samples (over every point: no sampling)."""
    return _silhouette(X, labels, metric, "silhouette_score")[1]


@function
def calinski_harabasz_score(X, labels, /):
    """The variance ratio B (n - K) / (W (K - 1)): B the between-cluster sum of squares sum_c n_c ||m_c - m||^2, W
    the within sum_i ||x_i - m_c(i)||^2 (1 when W = 0)."""
    (Xk, _), codes, K, _ = _internal(X, labels, "calinski_harabasz_score")
    return float(_core.ch_db(Xk, codes, K)[0])


@function
def davies_bouldin_score(X, labels, /):
    """The mean over the clusters of max_other (s_c + s_e) / d(m_c, m_e), s_c the mean distance of a cluster's points
    to its mean (Euclidean); clusters with the same mean are left out of each other's maximum, and the index is 0
    when every s_c or every distance between the means is 0."""
    (Xk, _), codes, K, _ = _internal(X, labels, "davies_bouldin_score")
    return float(_core.ch_db(Xk, codes, K)[1])
