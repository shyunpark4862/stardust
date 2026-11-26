"""Affinity propagation (docs/api/cluster.md §7; docs/design/cluster.md §2.9, §3.12; decisions CL40 and those of
milestone CL6 in §1.1): scikit-learn 1.9.1's AffinityPropagation without the noise it adds to the similarities.

Messages pass between the points until the exemplars settle: responsibilities R and availabilities A, each step
damped (R = damping R + (1 - damping) R_new), stopping when every point's exemplar status stayed the same for
convergence_iter steps (at most max_iter). The similarity is the negative squared Euclidean distance
(affinity='euclidean'; sd.spatial's difference form) or a given matrix (affinity='precomputed'); the preference
(the diagonal: how much a point wants to be an exemplar) defaults to the median similarity. scikit-learn breaks
ties with random noise; here the points go in key order (CL7) and every tie goes to the smaller key, so the same
points in another order give the same partition, and the result is scikit-learn's (with zero noise, on points in
key order) bit for bit (csrc/cluster/affinity.cpp).
"""

import math
import numbers

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._common import _array, check_alloc, choice, contiguous, labels_back, points, positive_int, real
from ._common import rows as _rows
from ._results import ClusterResult

_core = _sd.cluster
AFFINITIES = ("euclidean", "precomputed")


class AffinityPropagationResult(ClusterResult):
    """The result of affinity_propagation (docs/api/cluster.md §7).

    Fields: labels ((n,) int64, first-seen numbering; -1 for the rows nan_policy='omit' left out, and for every
    point when no exemplar emerged), n_clusters, nobs, omitted, exemplars ((K,) int64: the input rows of the
    exemplars, exemplars[j] that of label j), exemplar_points ((K, p) observations; None for affinity='precomputed'),
    n_iter, converged, damping, preference (the one used: a float, or the (n,) array given), affinity.
    """

    _TITLE = "Affinity Propagation Results"
    _METHOD = "affinity propagation"
    _TUNING = (("Affinity:", "affinity"), ("Damping:", "damping"), ("Preference:", "preference"))
    _COLUMNS = (("Exemplar", "exemplars"),)
    _REPR = ("nobs", "n_clusters", "n_iter", "converged")

    def predict(self, X_new, /):
        """The nearest exemplar of each new point ((m,) int64; the smallest squared Euclidean distance, on a tie the
        exemplar the fit lists first, as the fit assigns its own points). A fit on similarities cannot place new
        points (ValueError); a fit without exemplars labels every point -1."""
        func = "predict"
        if self.exemplar_points is None:
            raise ValueError(f"{func}(): a fit on precomputed similarities cannot place new points")
        Xn = points(X_new, func, "X_new")
        p = self.exemplar_points.shape[1]
        if Xn.shape[1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns (the fit's), not {Xn.shape[1]}")
        if not bool(_sd.all(_sd.isfinite(Xn))):
            raise ValueError(f"{func}(): 'X_new' must be finite (no NaN or infinity)")
        m = Xn.shape[0]
        if self.n_clusters == 0:
            return _sd.full((m,), -1, dtype=_sd.int64)
        from .. import spatial as _spatial

        P = contiguous(self.exemplar_points[self._fit_labels])  # the exemplars in the fit's order
        lab = _sd.argmin(_spatial.cdist(Xn, P, metric="sqeuclidean"), axis=1)
        return contiguous(self._fit_labels[lab])


_calls.methods(AffinityPropagationResult)


def _preference(v, func, n):
    """None, a float, or a C (n,) float64 array (finite)."""
    if v is None:
        return None
    if isinstance(v, numbers.Number) or (type(v).__module__ == "numpy" and getattr(v, "ndim", None) == 0):
        x = real(v, func, "preference")
        if not math.isfinite(x):
            raise ValueError(f"{func}(): 'preference' must be finite, not {x!r}")
        return x
    a = _array(v, func, "preference")
    if a.ndim != 1 or a.shape[0] != n:
        raise ValueError(f"{func}(): an array 'preference' must have one value per point ({n})")
    if not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{func}(): 'preference' must be finite")
    return contiguous(a)


@function
def affinity_propagation(
    X,
    /,
    *,
    damping=0.5,
    max_iter=200,
    convergence_iter=15,
    preference=None,
    affinity="euclidean",
    nan_policy="raise",
):
    """Affinity propagation (Frey and Dueck 2007; scikit-learn's AffinityPropagation): exemplars chosen among the
    points by passing messages, the number of clusters following from the preferences.

    X holds observations ((n, p); affinity='euclidean': the similarity is the negative squared Euclidean distance) or
    an (n, n) similarity matrix (affinity='precomputed'; larger is more similar, need not be symmetric). damping in
    [0.5, 1); at most max_iter steps, stopping once the exemplars stayed the same for convergence_iter steps.
    preference: a float, one value per point, or None (the median of the similarities). nan_policy 'raise' (the
    default) or 'omit' (observations only). Ties go to the point of smaller key (scikit-learn adds random noise
    instead), so there is no rng.
    """
    func = "affinity_propagation"
    affinity = choice(affinity, func, "affinity", AFFINITIES)
    d = real(damping, func, "damping")
    if not 0.5 <= d < 1.0:
        raise ValueError(f"{func}(): 'damping' must be in [0.5, 1), not {d!r}")
    max_iter = positive_int(max_iter, func, "max_iter")
    conv = positive_int(convergence_iter, func, "convergence_iter")
    omitted = None
    if affinity == "precomputed":
        S = _array(X, func, "X")
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError(f"{func}(): with affinity='precomputed', 'X' must be a square (n, n) similarity matrix")
        n = S.shape[0]
        if n == 0:
            raise ValueError(f"{func}(): 'X' must have at least 1 point")
        if not bool(_sd.all(_sd.isfinite(S))):
            raise ValueError(f"{func}(): the similarities 'X' must be finite")
        S = contiguous(S)
        n_input = n
        order = _sd.arange(n, dtype=_sd.int64)
        Xk = None
        Xpts = None
    else:
        Xpts = points(X, func)
        n_input = Xpts.shape[0]
        Xpts, _, omitted = _rows(Xpts, func, nan_policy)
        n = Xpts.shape[0]
        order = _core.lex_order(Xpts)
        Xk = contiguous(Xpts[order])
        S = None
    pref = _preference(preference, func, n_input)
    pref_given = pref
    if isinstance(pref, _sd.ndarray):
        if omitted is not None:
            pref = contiguous(pref[_sd.logical_not(omitted)])
        pref_k = contiguous(pref[order])
    elif pref is not None:
        pref_k = _sd.full((n,), pref, dtype=_sd.float64)
    else:
        pref_k = None
    check_alloc(4 * 8 * n * n, func)
    I, c, n_iter, converged, used = _core.affinity_propagation(Xk, S, pref_k, d, max_iter, conv)
    K = int(I.shape[0])
    if K == 0:
        labels = _sd.full((n,), -1, dtype=_sd.int64)
        exemplars = _sd.empty((0,), dtype=_sd.int64)
        fit_labels = _sd.empty((0,), dtype=_sd.int64)
        rows = exemplars
    else:
        labels, number = _core.first_seen(c, order, K)  # number[q]: the label of exemplar I[q]
        perm = _sd.empty((K,), dtype=_sd.int64)
        perm[number] = _sd.arange(K, dtype=_sd.int64)  # perm[label] = q
        rows = contiguous(order[I[perm]])  # each label's exemplar among the kept rows
        fit_labels = contiguous(number)
    if omitted is not None:
        kept = _sd.nonzero(_sd.logical_not(omitted))[0]
        exemplars = contiguous(kept[rows]) if K else rows
    else:
        exemplars = rows
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    if pref_given is None:
        pref_out = float(used)
    else:
        pref_out = pref_given
    return AffinityPropagationResult(
        labels=labels_back(labels, omitted),
        n_clusters=K,
        nobs=n,
        omitted=om,
        exemplars=exemplars,
        exemplar_points=contiguous(Xpts[rows])
        if (Xpts is not None and K)
        else (_sd.empty((0, Xpts.shape[1]), dtype=_sd.float64) if Xpts is not None else None),  # fmt: skip
        n_iter=int(n_iter),
        converged=bool(converged),
        damping=d,
        preference=pref_out,
        affinity=affinity,
        _fit_labels=fit_labels,
    )
