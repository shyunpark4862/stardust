"""Fuzzy c-means (docs/api/cluster.md §3; docs/design/cluster.md §2.3, §3.5, §4.3; decision CL27 and those of
milestone CL2 in §1.1): Bezdek's alternating updates of the memberships u_ij = 1 / sum_l (d_ij / d_il)^(2/(m-1))
and the centres (the u^m-weighted means), on the squared distance of kmeans (sd.spatial's sqeuclidean of each pair),
until the largest change of a membership is below tol. R's fanny (another objective, on a distance matrix) is not
this function.
"""

import math

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import check_rng, contiguous, key, labels_back, points, positive_int, real
from ._common import rows as _rows
from ._kmeans import _distinct_check, _init, _k, _tol
from ._results import ClusterResult

_core = _sd.cluster


class FuzzyCMeansResult(ClusterResult):
    """The result of fuzzy_cmeans (docs/api/cluster.md §3).

    Fields: labels ((n,) int64: each point's cluster of largest membership, the smaller on a tie; first-seen
    numbering, -1 for the rows nan_policy='omit' left out), n_clusters (k), nobs, omitted, memberships ((nobs, k):
    the kept rows' memberships, column j that of label j; rows sum to 1), centers ((k, p)), m, objective
    (sum u^m d^2), partition_coefficient (the mean of sum_j u_ij^2: 1 for a hard partition, 1/k for equal shares),
    n_iter and converged.
    """

    _TITLE = "Fuzzy C-Means Results"
    _METHOD = "fuzzy c-means"
    _OBJECTIVE = ("Objective:", "objective")
    _TUNING = (("m:", "m"), ("Partition coef.:", "partition_coefficient"))
    _COLUMNS = (("Membership", "cluster_membership"),)
    _REPR = ("nobs", "n_clusters", "objective", "n_iter")

    def predict(self, X_new, /):
        """The cluster of largest membership of each new point ((m,) int64; the smaller label on a tie)."""
        return _sd.argmax(self.predict_proba(X_new), axis=1)

    def predict_proba(self, X_new, /):
        """The memberships of new points to the centres ((m, k), rows summing to 1)."""
        func = "predict_proba"
        Xn = points(X_new, func, "X_new")
        p = self.centers.shape[1]
        if Xn.shape[1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns (the fit's), not {Xn.shape[1]}")
        if not bool(_sd.all(_sd.isfinite(Xn))):
            raise ValueError(f"{func}(): 'X_new' must be finite (no NaN or infinity)")
        return _core.fuzzy_memberships(Xn, self.centers, self.m)


_calls.methods(FuzzyCMeansResult)


@function
def fuzzy_cmeans(X, k, /, *, rng, m=2.0, init="random", max_iter=100, tol=1e-05, nan_policy="raise"):
    """Fuzzy c-means (Bezdek): k centres and each point's memberships u_ij in [0, 1] (summing to 1 over j) that
    make sum u^m ||x - c||^2 small, by alternating the memberships and the centres from k random points (init
    'random', drawn with rng) or a (k, p) array, until the largest change of a membership is below tol (at most
    max_iter times). m > 1 is the fuzziness (m -> 1: k-means' hard partition). A point on a centre belongs to it
    alone (to the centres it lies on equally). nan_policy 'raise' (the default) or 'omit'.
    """
    func = "fuzzy_cmeans"
    X = points(X, func)
    n_input = X.shape[0]
    X, _, omitted = _rows(X, func, nan_policy)
    n, p = X.shape
    k = _k(k, func, n)
    m = real(m, func, "m")
    if not (m > 1 and math.isfinite(m)):
        raise ValueError(f"{func}(): 'm' must be finite and > 1, not {m!r}")
    if isinstance(init, str) and init != "random":
        raise ValueError(f"{func}(): 'init' must be 'random' or a (k, p) array, not {init!r}")
    kind, C0 = _init(init, func, k, p)
    max_iter = positive_int(max_iter, func, "max_iter")
    tol = _tol(tol, func)
    order = _core.lex_order(X)
    _distinct_check(X, order, None, k, func)
    rng = check_rng(rng, func)
    k0, k1 = key(rng)
    U, C, obj, n_iter, converged = _core.fuzzy_cmeans(
        X, order, k, m, 1 if kind == "random" else 2, C0, max_iter, tol, k0, k1
    )
    lab_key = _sd.argmax(U, axis=1)
    labels, number = _core.first_seen(contiguous(lab_key.astype(_sd.int64)), order, k)
    perm = _sd.empty((k,), dtype=_sd.int64)
    perm[number] = _sd.arange(k, dtype=_sd.int64)
    inv = _sd.empty((n,), dtype=_sd.int64)
    inv[order] = _sd.arange(n, dtype=_sd.int64)
    Uin = contiguous(contiguous(U[inv])[:, perm])  # the kept rows in the input's order, columns by label
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    return FuzzyCMeansResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=om,
        memberships=Uin,
        centers=contiguous(C[perm]),
        m=m,
        objective=float(obj),
        partition_coefficient=float(_sd.sum(Uin * Uin)) / n,
        cluster_membership=contiguous(_sd.sum(Uin, axis=0)),
        n_iter=int(n_iter),
        converged=bool(converged),
    )
