"""Reference values for sd.cluster (docs/design/cluster.md §4, §7.2) on the test data of tests/cluster/samples.py:
exact rationals (python-flint's fmpq, fmpq_mat, Python's Fraction) and Arb.

    .venv/bin/python tools/cluster_ref.py                     # every family, on 3 processes
    .venv/bin/python tools/cluster_ref.py hierarchy kmeans    # only these families
    .venv/bin/python tools/cluster_ref.py --list              # the families and their data

writes tests/cluster/ref/<family>.npz (layout: tests/reffile.py), read by tests/cluster/refdata.py. A point has
the inputs "set" (the id in samples.py), "kind" (what the point is, per family below), "index" (its number in
its kind) and "digest" (the data's digest); a group is one set (one labeling, one mixture) and holds the tuning
the references were computed with. The data are doubles taken as exact rationals; a reference is the true value
for those doubles, rounded once (a double-double hi + lo). Squared Euclidean distances of doubles are exact
rationals, so every choice the definitions make (a nearest centre, a pair to merge, a core point) is made on
exact values; a choice between values equal in exact arithmetic follows the key of decision CL7 (the points'
lexicographic ranks, samples.lex_rank) and "gap" records how close the runner-up was (0: an exact tie).

    hierarchy   the primitive definition of docs/design/cluster.md §3.2 (CL17): n - 1 times, the pair of
                clusters of smallest (linkage value, key) merges, the key (min rank in A, min rank in B) of the
                pair's two smallest ranks in increasing order. Methods single, complete, average, weighted,
                centroid, median, ward with scipy's heights: single and complete the extreme distances,
                average and weighted the Lance-Williams values of the Euclidean distances (exact sums of square
                roots, compared in Arb and by their exact form), centroid, median and ward the square roots of
                exact rational Lance-Williams values of the squared distances (ward: R's ward.D2). One point per
                merge (kind 0, index the merge's number): the height; fields a, b (scipy's cluster numbers:
                the points 0 .. n - 1 by input index, merge i is n + i; a < b), size and gap (the runner-up's
                relative distance from the chosen value at that step). Sets of at most 60 points
    kmeans      Lloyd's iteration in exact arithmetic from the group input init (k distinct points of the
                set, by a fixed permutation) until no label changes (at most 300 steps): every point to the
                centre of smallest exact squared distance (the smaller centre on a tie), the exact weighted
                means; an emptied cluster takes the point farthest from its centre (largest exact squared
                distance, the smaller key on a tie; scikit-learn's relocation). Methods lloyd_label (kind 0,
                field gap), lloyd_center (kind 1, index j p + d) and over the groups lloyd_inertia (field
                n_iter, relocated); then with the centres rounded to doubles (group input centers_f):
                label_f (kind 0, field gap) and inertia_f; and the exact weighted means of the true labels
                (truth_mean, kind 2) where a set has them
    kmedoids    the medoids of the Euclidean distances (Arb): pam (R's BUILD and SWAP, best swap first, the
                swap of smallest (change, key of the medoid, key of the point)), with fields medoid (the input
                index) per medoid (kind 0, index the medoid's number, hi the BUILD objective's... see below) and
                over the groups pam_build and pam_swap (the mean distance after each phase, R's objective), and
                for at most 20 points the best k-set of all (optimum; field ties = how many k-sets share it)
    fuzzy       fuzzy c-means (decision CL27) at given centres (group input centers: midpoints of the k-means
                starts and their neighbours, rounded): the memberships of every point (kind 0, index i k + j) for
                m = 2 (u_m2) and m = 1.5 (u_m15) as exact rationals and m = 3 (u_m3, square roots in Arb); one
                step at m = 2: the centres from the exact memberships (centre1, kind 1, index j p + d) and the
                objective sum u^2 d^2 at the given centres (objective0, over the groups)
    density     DBSCAN at the group's eps and min_samples (exact: d^2 <= eps^2, the point itself counted;
                weights summed): label (kind 0; the first-seen numbering, -1 noise; fields core, near =
                some pair of the point within 2^-40 eps^2 of eps^2); HDBSCAN's pieces at min_samples 5:
                core_distance (kind 0, the 5th smallest distance with the point itself) and the minimum spanning
                tree of the mutual reachability max(core_a, core_b, d(a, b)) (kind 1, one point per edge in
                (weight, min key, max key) order; fields a, b the input indices, a < b)
    mixture     for each samples.mixtures set: score (kind 0: the log density of the mixture at the point,
                logsumexp of log w_j + log N(x; mu_j, Sigma_j)), logdens_<j> and resp_<j> (the component's log
                density and responsibility) in Arb from the exact inverse and determinant (fmpq_mat); mstep
                (kind 1): sklearn's M step from the rounded responsibilities (weights, means, covariances in
                sklearn's shapes, reg_covar 1e-6 added), exact rationals, index in that order
    metrics     per labeling (one group each): ari, ri, fm, mi, nmi_<avg> and ami_<avg> for the four
                averages, homogeneity, completeness, v (beta 1) over the groups (Fraction for ari, ri and the
                counts; Arb otherwise, AMI's expectation the exact hypergeometric sum with lgamma, left out
                above 10^5 points; sklearn's values in its special cases, field special 1); per point set with
                true labels: silhouette (kind 0; Arb sums of square roots), and over the groups
                silhouette_score, calinski_harabasz (exact rational) and davies_bouldin

Settling: an exact rational is exact; an Arb value is settled to 2^-110 relative. Workers are forked, replaced every
8 tasks (maxtasksperchild), each task under a time limit (SIGALRM). No mpmath (the project's rule). The console
output also goes to build/logs/cluster_ref/.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import signal
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import arbref  # noqa: E402
from arbref import NotSettled  # noqa: E402
from flint import arb, fmpq, fmpq_mat  # noqa: E402

OUT = ROOT / "tests" / "cluster" / "ref"
TOOL = "tools/cluster_ref.py"
SECONDS = 1800  # a task's time limit
METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
HIER_MAX_N = 160
PAM_MAX_N = 160
OPT_MAX_N = 20
AMI_MAX_N = 10**5
TIE_BITS = (256, 1024, 4096)  # the precisions an Arb comparison is tried at before it counts as a tie


class _Timeout(Exception):
    pass


def _alarm(*_):
    raise _Timeout


def S():
    from tests.cluster import samples

    return samples


# ---------------------------------------------------------------- exact helpers


def Q(v) -> fmpq:
    return fmpq(*float(v).as_integer_ratio())


def fr(q) -> Fraction:
    return q if isinstance(q, Fraction) else Fraction(int(q.p), int(q.q))


def dd_of(q):
    """(hi, lo) of an exact rational: hi rounded to nearest (ties to even), lo the rest rounded."""
    q = fr(q)
    try:
        hi = float(q)
    except OverflowError:
        return (math.inf if q > 0 else -math.inf), 0.0
    if math.isinf(hi):
        return hi, 0.0
    return hi, float(q - Fraction(hi))


def arb_q(q):
    q = fmpq(q.numerator, q.denominator) if isinstance(q, Fraction) else q
    return arb(q.p) / arb(q.q)


def settle(fn):
    return arbref.to_dd(arbref.rigorous(fn))


def safe(fn, **fields):
    """(hi, lo, fields) of settle(fn), or None (the value is dropped) when it does not settle."""
    try:
        return (*settle(fn), fields)
    except NotSettled:
        return None


def sqrt_dd(q):
    """(hi, lo) of sqrt(q) for an exact rational q >= 0."""
    if q == 0:
        return 0.0, 0.0
    return settle(lambda: arb_q(q).sqrt())


def sqdist(a, b):
    s = fmpq(0)
    for x, y in zip(a, b):
        d = x - y
        s += d * d
    return s


def exact_points(X):
    return [[Q(v) for v in row] for row in np.asarray(X, float)]


def sqdist_matrix(P):
    n = len(P)
    D = [[fmpq(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            D[i][j] = D[j][i] = sqdist(P[i], P[j])
    return D


class Radicals:
    """An exact sum of c_q sqrt(q) over rationals q >= 0 (the average and weighted linkage values): the exact
    form for equality, Arb for order."""

    __slots__ = ("terms", "_arb")

    def __init__(self, terms):
        self.terms = terms  # {fmpq q: fmpq c}
        self._arb = {}

    @staticmethod
    def sqrt(q):
        return Radicals({q: fmpq(1)})

    def scaled(self, c):
        return Radicals({q: v * c for q, v in self.terms.items()})

    def plus(self, other):
        t = dict(self.terms)
        for q, v in other.terms.items():
            t[q] = t.get(q, fmpq(0)) + v
        return Radicals({q: v for q, v in t.items() if v != 0})

    def value(self, prec):
        v = self._arb.get(prec)
        if v is None:
            with arbref.precision(prec):
                v = arb(0)
                for q, c in self.terms.items():
                    v += arb_q(c) * arb_q(q).sqrt()
            self._arb[prec] = v
        return v

    def float(self):
        return float(self.value(256).mid())


def cmp_values(a, b):
    """-1, 0 or 1: the order of two linkage values (fmpq or Radicals) in exact arithmetic; 0 for exact ties
    (Radicals: an equal exact form, or not separated at 4096 bits)."""
    if isinstance(a, Radicals):
        if a.terms == b.terms:
            return 0
        for prec in TIE_BITS:
            with arbref.precision(prec):
                d = a.value(prec) - b.value(prec)
                if d > 0:
                    return 1
                if d < 0:
                    return -1
        return 0
    return (a > b) - (a < b)


def to_float(v):
    return v.float() if isinstance(v, Radicals) else float(fr(v))


# ---------------------------------------------------------------- hierarchy


def hierarchy_set(pid, method):
    """The primitive tree: [(a, b, height (hi, lo), size, gap)] in merge order."""
    s = S().points(pid)
    n = s.n
    P = exact_points(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_matrix(P)
    squared = method in ("centroid", "median", "ward", "single", "complete")
    # cluster state: id -> (members' min rank, size); values[(i, j)] for live ids i < j
    live = {i: (rank[i], 1) for i in range(n)}
    values = {}
    for i in range(n):
        for j in range(i + 1, n):
            values[(i, j)] = D2[i][j] if squared else Radicals.sqrt(D2[i][j])
    out = []
    for step in range(n - 1):
        best = None
        second = None
        for (i, j), v in values.items():
            key = tuple(sorted((live[i][0], live[j][0])))
            if best is None:
                best = (v, key, i, j)
                continue
            c = cmp_values(v, best[0])
            if c < 0 or (c == 0 and key < best[1]):
                second = best
                best = (v, key, i, j)
            elif second is None or cmp_values(v, second[0]) < 0 or (cmp_values(v, second[0]) == 0 and key < second[1]):
                second = (v, key, i, j)
        v, _, i, j = best
        if second is None:
            gap = math.inf
        else:
            c = cmp_values(second[0], v)
            fb, fs = to_float(v), to_float(second[0])
            gap = 0.0 if c == 0 else (fs - fb) / fb if fb > 0 else math.inf
        if squared:
            height = sqrt_dd(v)
        else:
            height = settle(lambda v=v: _radicals_arb(v))
        new = n + step
        ni, nj = live[i][1], live[j][1]
        out.append((min(i, j), max(i, j), height, ni + nj, gap))
        # Lance-Williams updates (exact)
        dij = values[(i, j)]
        newvals = {}
        for k in live:
            if k in (i, j):
                continue
            dik = values[(min(i, k), max(i, k))]
            djk = values[(min(j, k), max(j, k))]
            nk = live[k][1]
            if method == "single":
                u = dik if dik < djk else djk
            elif method == "complete":
                u = dik if dik > djk else djk
            elif method == "average":
                u = dik.scaled(fmpq(ni, ni + nj)).plus(djk.scaled(fmpq(nj, ni + nj)))
            elif method == "weighted":
                u = dik.scaled(fmpq(1, 2)).plus(djk.scaled(fmpq(1, 2)))
            elif method == "centroid":
                u = (ni * dik + nj * djk) / (ni + nj) - fmpq(ni * nj, (ni + nj) ** 2) * dij
            elif method == "median":
                u = dik / 2 + djk / 2 - dij / 4
            else:  # ward on squared values
                u = ((ni + nk) * dik + (nj + nk) * djk - nk * dij) / (ni + nj + nk)
            newvals[k] = u
        for key in [kk for kk in values if i in kk or j in kk]:
            del values[key]
        live[new] = (min(live[i][0], live[j][0]), ni + nj)
        del live[i], live[j]
        for k, u in newvals.items():
            values[(min(k, new), max(k, new))] = u
    return out


def _radicals_arb(v):
    s = arb(0)
    for q, c in v.terms.items():
        s += arb_q(c) * arb_q(q).sqrt()
    return s


def hierarchy_task(pid):
    s = S().points(pid)
    pts = {}
    for m in METHODS:
        for k, (a, b, h, size, gap) in enumerate(hierarchy_set(pid, m)):
            pts.setdefault(k, {})[m] = (h[0], h[1], {"a": a, "b": b, "size": size, "gap": gap})
    return {
        "set": pid,
        "digest": s.digest,
        "ginputs": {},
        "points": [(0, k, ms) for k, ms in sorted(pts.items())],
        "gmethods": {},
    }


def hierarchy_sets():
    return [pid for pid in S().point_ids(max_n=HIER_MAX_N) if S().points(pid).n >= 2]


# ---------------------------------------------------------------- k-means


def _init_centres(s):
    """k distinct points of the set by a fixed permutation (samples.rng("init", id)); None when there are fewer
    distinct points than k."""
    g = S().rng("init", s.id)
    seen, idx = set(), []
    for i in g.permutation(s.n).tolist():
        t = tuple(s.X[i].tolist())
        if t in seen:
            continue
        seen.add(t)
        idx.append(i)
        if len(idx) == s.k:
            return np.array(idx)
    return None


def _assign(P, C, rank):
    """(labels, gaps, d2 to the own centre) by exact squared distance; the smaller centre on a tie."""
    labs, gaps, own = [], [], []
    for x in P:
        ds = [sqdist(x, c) for c in C]
        j = min(range(len(C)), key=lambda t: (ds[t], t))
        rest = [ds[t] for t in range(len(C)) if t != j]
        second = min(rest) if rest else None
        if second is None:
            gap = math.inf
        elif second == ds[j]:
            gap = 0.0
        else:
            b = float(fr(ds[j]))
            gap = float(fr(second - ds[j])) / b if b > 0 else math.inf
        labs.append(j)
        gaps.append(gap)
        own.append(ds[j])
    return labs, gaps, own


def _means(P, W, labs, k, p, old):
    sums = [[fmpq(0)] * p for _ in range(k)]
    tot = [fmpq(0)] * k
    for x, w, j in zip(P, W, labs):
        tot[j] += w
        for d in range(p):
            sums[j][d] += w * x[d]
    return [[sums[j][d] / tot[j] for d in range(p)] if tot[j] != 0 else list(old[j]) for j in range(k)]


def kmeans_task(pid):
    s = S().points(pid)
    n, p, k = s.n, s.p, s.k
    P = exact_points(s.X)
    W = [Q(v) for v in s.w] if s.w is not None else [fmpq(1)] * n
    rank = S().lex_rank(s.X).tolist()
    pts = {}
    gm = {}
    gin = {}
    idx = _init_centres(s)
    if idx is not None and k <= n:
        C = [list(P[i]) for i in idx.tolist()]
        gin["init"] = s.X[idx]
        labs = None
        relocated = 0
        it = 0
        path_gap = math.inf
        for step in range(1, 301):
            it = step
            new, gaps_it, own = _assign(P, C, rank)
            path_gap = min([path_gap] + gaps_it)
            # empty clusters take the farthest points (largest exact d^2 to their centre, the smaller key first)
            counts = [0] * k
            for j, w in zip(new, W):
                counts[j] += 1 if w != 0 else 0
            empty = [j for j in range(k) if counts[j] == 0]
            if empty:
                far = sorted(range(n), key=lambda i: (-fr(own[i]), rank[i]))
                for j, i in zip(empty, far):
                    new[i] = j
                    relocated += 1
            if labs is not None and new == labs:
                break
            labs = new
            C = _means(P, W, labs, k, p, C)
        labs, gaps, _ = _assign(P, C, rank)
        inertia = sum((W[i] * sqdist(P[i], C[labs[i]]) for i in range(n)), fmpq(0))
        for i in range(n):
            pts.setdefault((0, i), {})["lloyd_label"] = (float(labs[i]), 0.0, {"gap": gaps[i]})
        for j in range(k):
            for d in range(p):
                h, lo = dd_of(C[j][d])
                pts.setdefault((1, j * p + d), {})["lloyd_center"] = (h, lo, {})
        h, lo = dd_of(inertia)
        gm["lloyd_inertia"] = (h, lo, {"n_iter": float(it), "relocated": float(relocated), "path_gap": path_gap})
        Cf = np.array([[dd_of(c)[0] for c in row] for row in C])
        gin["centers_f"] = Cf
        CfQ = exact_points(Cf)
        lf, gf, _ = _assign(P, CfQ, rank)
        for i in range(n):
            pts.setdefault((0, i), {})["label_f"] = (float(lf[i]), 0.0, {"gap": gf[i]})
        inf_ = sum((W[i] * sqdist(P[i], CfQ[lf[i]]) for i in range(n)), fmpq(0))
        h, lo = dd_of(inf_)
        gm["inertia_f"] = (h, lo, {})
    if s.truth is not None:
        kt = int(s.truth.max()) + 1
        lab = [int(v) for v in s.truth.tolist()]
        keep = [i for i in range(n) if lab[i] >= 0]
        M = _means([P[i] for i in keep], [W[i] for i in keep], [lab[i] for i in keep], kt, p, [[fmpq(0)] * p] * kt)
        for j in range(kt):
            for d in range(p):
                h, lo = dd_of(M[j][d])
                pts.setdefault((2, j * p + d), {})["truth_mean"] = (h, lo, {})
    return {
        "set": pid,
        "digest": s.digest,
        "ginputs": gin,
        "points": [(kind, idx_, ms) for (kind, idx_), ms in sorted(pts.items())],
        "gmethods": gm,
    }


def kmeans_sets():
    return [i for i in S().point_ids() if S().points(i).k is not None and S().points(i).n >= S().points(i).k]


# ---------------------------------------------------------------- fuzzy c-means


def _fuzzy_centres(s):
    """The given centres of a set (decision CL27's references): midpoints of the k start points of _init_centres
    and their neighbours in that list, rounded to doubles (k = 1: the first point plus 0.25); None without a start."""
    idx = _init_centres(s)
    if idx is None:
        return None
    X0 = s.X[idx]
    if s.k == 1:
        return X0 + 0.25
    return (X0 + s.X[np.roll(idx, 1)]) / 2.0


def _memberships(Drow, q):
    """A point's exact memberships for the exponent q = 1 / (m - 1) (an int: rationals), from its exact squared
    distances; the centres it lies on share 1 equally."""
    zeros = [j for j, d in enumerate(Drow) if d == 0]
    if zeros:
        return [fmpq(1, len(zeros)) if d == 0 else fmpq(0) for d in Drow]
    mn = min(Drow)
    r = [(mn / d) ** q for d in Drow]
    t = sum(r, fmpq(0))
    return [v / t for v in r]


def fuzzy_task(pid):
    s = S().points(pid)
    n, p, k = s.n, s.p, s.k
    C0 = _fuzzy_centres(s)
    pts, gm, gin = {}, {}, {}
    if C0 is not None:
        gin["centers"] = C0
        P, Cq = exact_points(s.X), exact_points(C0)
        D = [[sqdist(x, c) for c in Cq] for x in P]
        U2 = [_memberships(row, 1) for row in D]
        U15 = [_memberships(row, 2) for row in D]
        for i in range(n):
            for j in range(k):
                pts.setdefault((0, i * k + j), {})["u_m2"] = (*dd_of(U2[i][j]), {})
                pts.setdefault((0, i * k + j), {})["u_m15"] = (*dd_of(U15[i][j]), {})
            row = D[i]
            if any(d == 0 for d in row):
                U3 = _memberships(row, 1)  # the same shares for every m
                for j in range(k):
                    pts[(0, i * k + j)]["u_m3"] = (*dd_of(U3[j]), {})
                continue
            mn = min(row)
            for j in range(k):

                def u3(j=j, row=row, mn=mn):
                    r = [arb_q(mn / d).sqrt() for d in row]
                    return r[j] / sum(r[1:], r[0])

                e = safe(u3)
                if e is not None:
                    pts[(0, i * k + j)]["u_m3"] = e
        # one step at m = 2: the centres from the exact memberships U2, and the objective at (U2, C0)
        for j in range(k):
            wsum = sum((U2[i][j] ** 2 for i in range(n)), fmpq(0))
            for d in range(p):
                c = sum((U2[i][j] ** 2 * P[i][d] for i in range(n)), fmpq(0)) / wsum
                pts.setdefault((1, j * p + d), {})["centre1"] = (*dd_of(c), {})
        obj = sum((U2[i][j] ** 2 * D[i][j] for i in range(n) for j in range(k)), fmpq(0))
        gm["objective0"] = (*dd_of(obj), {})
    return {
        "set": pid,
        "digest": s.digest,
        "ginputs": gin,
        "points": [(kind, idx_, ms) for (kind, idx_), ms in sorted(pts.items())],
        "gmethods": gm,
    }


def fuzzy_sets():
    return [i for i in kmeans_sets() if S().points(i).n <= 300]


# ---------------------------------------------------------------- k-medoids


def _dist_arb(D2, prec=256):
    with arbref.precision(prec):
        return [[arb_q(q).sqrt() for q in row] for row in D2]


def kmedoids_task(item):
    pid, k = item
    s = S().points(pid)
    n = s.n
    P = exact_points(s.X)
    rank = S().lex_rank(s.X).tolist()
    by_rank = sorted(range(n), key=lambda i: rank[i])
    D2 = sqdist_matrix(P)
    with arbref.precision(256):
        D = _dist_arb(D2)

        def total(meds):
            t = arb(0)
            for j in range(n):
                m = D[j][meds[0]]
                for q in meds[1:]:
                    if D[j][q] < m:
                        m = D[j][q]
                t += m
            return t

        # BUILD (R's pam): the point of smallest distance sum, then the point of largest gain; the smaller key
        # on a tie (the points in key order, replaced only by a strictly better one)
        sums = [sum(D[i], arb(0)) for i in range(n)]
        first = by_rank[0]
        for i in by_rank[1:]:
            if sums[i] < sums[first]:
                first = i
        meds = [first]
        Dmin = [D[j][first] for j in range(n)]
        while len(meds) < k:
            best, bg = None, None
            for i in by_rank:
                if i in meds:
                    continue
                g = arb(0)
                for j in range(n):
                    d = Dmin[j] - D[j][i]
                    if d > 0:
                        g += d
                if best is None or g > bg:
                    best, bg = i, g
            meds.append(best)
            Dmin = [D[j][best] if D[j][best] < Dmin[j] else Dmin[j] for j in range(n)]
        build = total(meds) / n
        # SWAP: the best (medoid, point) swap while it lowers the total; the smaller (key of the medoid, key
        # of the point) on a tie
        cur = total(meds)
        swaps = 0
        while swaps < 1000:
            best = None
            for m in sorted(meds, key=lambda i: rank[i]):
                for o in by_rank:
                    if o in meds:
                        continue
                    trial = [o if q == m else q for q in meds]
                    t = total(trial)
                    if best is None or t < best[0]:
                        best = (t, m, o)
            if best is None or not best[0] < cur:
                break
            meds = [best[2] if q == best[1] else q for q in meds]
            cur = best[0]
            swaps += 1
        swap = cur / n
        out_pts = [(0, q, {"pam": (float(m), 0.0, {})}) for q, m in enumerate(sorted(meds))]
        gm = {"pam_build": (*arbref.to_dd(build), {}), "pam_swap": (*arbref.to_dd(swap), {"swaps": float(swaps)})}
        if n <= OPT_MAX_N:
            import itertools

            best, ties = None, 0
            for comb in itertools.combinations(range(n), k):
                t = total(list(comb))
                if best is None or t < best[0]:
                    best, ties = (t, comb), 1
                elif not best[0] < t:
                    ties += 1
            out_pts += [(1, q, {"optimum": (float(m), 0.0, {})}) for q, m in enumerate(best[1])]
            gm["optimum_objective"] = (*arbref.to_dd(best[0] / n), {"ties": float(ties)})
    return {
        "set": f"{pid}|k{k}",
        "digest": s.digest,
        "ginputs": {"k": np.array(k)},
        "points": out_pts,
        "gmethods": gm,
    }


def kmedoids_sets():
    out = []
    for pid in S().point_ids(max_n=PAM_MAX_N):
        s = S().points(pid)
        distinct = len({tuple(r) for r in s.X.tolist()})
        for k in sorted({2, 3, s.k or 2}):
            if k <= distinct and k < s.n:
                out.append((pid, k))
    return out


# ---------------------------------------------------------------- density

DBSCAN_PARAMS = {  # set -> [(eps, min_samples)], eps dyadic doubles
    "eps/6": [(0.5, 2), (0.5, 3), (0.25, 2)],
    "blobs/k3p2": [(1.0, 5), (2.0, 5)],
    "density/k3p2": [(0.25, 5), (0.5, 5), (1.0, 5)],
    "rings/p2": [(0.5, 5), (0.25, 4)],
    "moons/p2": [(0.25, 5)],
    "lattice/4x4": [(1.0, 3), (1.0, 5)],
    "lattice/4x4-shuffled": [(1.0, 3), (1.0, 5)],
    "lattice/3x3x3": [(1.0, 4)],
    "dups/4x5": [(0.5, 5), (3.0, 6)],
    "allsame/8": [(0.5, 5)],
    "hexagon/7": [(1.0, 3), (1.0, 7)],
    "weights/k2p2": [(1.0, 5)],
    "blobs/k2p1": [(1.0, 5)],
    "sizes/k3p2": [(0.5, 5)],
}
HDBSCAN_MIN_SAMPLES = 5


def density_items():
    return [(pid, eps, ms) for pid, ps in DBSCAN_PARAMS.items() for eps, ms in ps] + [
        (pid, None, None) for pid in DBSCAN_PARAMS if S().points(pid).n >= HDBSCAN_MIN_SAMPLES
    ]


def density_task(item):
    pid, eps, ms = item
    s = S().points(pid)
    n = s.n
    P = exact_points(s.X)
    rank = S().lex_rank(s.X).tolist()
    D2 = sqdist_matrix(P)
    pts = []
    gm = {}
    if eps is not None:
        e2 = Q(eps) * Q(eps)
        W = [Q(v) for v in s.w] if s.w is not None else [fmpq(1)] * n
        nb = [[j for j in range(n) if D2[i][j] <= e2] for i in range(n)]
        core = [sum((W[j] for j in nb[i]), fmpq(0)) >= ms for i in range(n)]
        near = [any(abs(D2[i][j] - e2) <= e2 / 2**40 for j in range(n) if j != i) for i in range(n)]
        comp = [-1] * n
        c = 0
        for i in sorted(range(n), key=lambda t: rank[t]):
            if not core[i] or comp[i] >= 0:
                continue
            stack = [i]
            comp[i] = c
            while stack:
                u = stack.pop()
                for v in nb[u]:
                    if core[v] and comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
        lab = list(comp)
        for i in range(n):
            if core[i]:
                continue
            cands = [j for j in nb[i] if core[j]]
            if cands:
                j = min(cands, key=lambda t: (D2[i][t], rank[t]))
                lab[i] = comp[j]
        lab = S().first_seen(lab).tolist()
        for i in range(n):
            pts.append((0, i, {"label": (float(lab[i]), 0.0, {"core": float(core[i]), "near": float(near[i])})}))
        tag = f"|eps{eps:g}|ms{ms}"
        ginputs = {"eps": np.array(eps), "min_samples": np.array(ms)}
    else:
        m = HDBSCAN_MIN_SAMPLES
        core2 = [sorted(D2[i])[m - 1] for i in range(n)]
        for i in range(n):
            h, lo = sqrt_dd(core2[i])
            pts.append((0, i, {"core_distance": (h, lo, {})}))

        def w2(a, b):
            v = D2[a][b]
            return max(v, core2[a], core2[b])

        # Prim on exact squared values, the total order (w2, min key, max key)
        inside = [False] * n
        best = [None] * n
        inside[0] = True
        for v in range(1, n):
            best[v] = (w2(0, v), tuple(sorted((rank[0], rank[v]))), 0)
        edges = []
        for _ in range(n - 1):
            v = min((u for u in range(n) if not inside[u]), key=lambda u: (best[u][0], best[u][1]))
            wv, keyv, fv = best[v]
            edges.append((wv, keyv, min(v, fv), max(v, fv)))
            inside[v] = True
            for u in range(n):
                if not inside[u]:
                    cand = (w2(v, u), tuple(sorted((rank[v], rank[u]))), v)
                    if (cand[0], cand[1]) < (best[u][0], best[u][1]):
                        best[u] = cand
        edges.sort(key=lambda e: (e[0], e[1]))
        for q, (wv, _, a, b) in enumerate(edges):
            h, lo = sqrt_dd(wv)
            pts.append((1, q, {"mst": (h, lo, {"a": float(a), "b": float(b)})}))
        tag = f"|hdbscan{m}"
        ginputs = {"eps": np.array(math.nan), "min_samples": np.array(m)}
    return {"set": pid + tag, "digest": s.digest, "ginputs": ginputs, "points": pts, "gmethods": gm}


# ---------------------------------------------------------------- mixtures


def _covs(mx):
    """The full covariance matrix of each component as exact rationals."""
    k, p = mx.k, mx.p
    c = mx.covariances
    out = []
    for j in range(k):
        if mx.covariance_type == "full":
            M = c[j]
        elif mx.covariance_type == "tied":
            M = c
        elif mx.covariance_type == "diag":
            M = np.diag(c[j])
        else:
            M = np.eye(p) * c[j]
        out.append(fmpq_mat([[Q(v) for v in row] for row in np.asarray(M)]))
    return out


def mixture_task(mid):
    mx = S().mixture(mid)
    k, p = mx.k, mx.p
    covs = _covs(mx)
    invs = [C.inv() for C in covs]
    dets = [C.det() for C in covs]
    mu = [[Q(v) for v in row] for row in mx.means]
    lw = [Q(v) for v in mx.weights]
    pts = []
    resp_f = []
    for i, x in enumerate(mx.X):
        xq = [Q(v) for v in x]
        quads = []
        for j in range(k):
            dvec = fmpq_mat([[xq[t] - mu[j][t]] for t in range(p)])
            quads.append((dvec.transpose() * invs[j] * dvec)[0, 0])

        def logdens(j):
            return arb_q(lw[j]).log() - arb_q(quads[j]) / 2 - arb_q(dets[j]).log() / 2 - p * (2 * arb.pi()).log() / 2

        def score():
            ls = [logdens(j) for j in range(k)]
            m = max(ls, key=lambda v: v.mid())
            return m + sum(((v - m).exp() for v in ls), arb(0)).log()

        ms = {"score": (*settle(score), {})}
        rf = []
        for j in range(k):
            ms[f"logdens_{j}"] = (*settle(lambda j=j: logdens(j) - arb_q(lw[j]).log()), {})
            hj = settle(lambda j=j: (logdens(j) - score()).exp())
            ms[f"resp_{j}"] = (*hj, {})
            rf.append(hj[0])
        resp_f.append(rf)
        pts.append((0, i, ms))
    # M step from the rounded responsibilities (sklearn's _estimate_gaussian_parameters, reg_covar 1e-6)
    R = [[Q(v) for v in row] for row in resp_f]
    Xq = [[Q(v) for v in row] for row in mx.X]
    m = len(Xq)
    reg = Q(1e-6)
    nk = [sum((R[i][j] for i in range(m)), fmpq(0)) for j in range(k)]
    means = [[sum((R[i][j] * Xq[i][t] for i in range(m)), fmpq(0)) / nk[j] for t in range(p)] for j in range(k)]
    vals = [nk[j] / m for j in range(k)]
    vals += [means[j][t] for j in range(k) for t in range(p)]
    if mx.covariance_type == "full":
        for j in range(k):
            for a in range(p):
                for b in range(p):
                    v = sum((R[i][j] * (Xq[i][a] - means[j][a]) * (Xq[i][b] - means[j][b]) for i in range(m)), fmpq(0))
                    vals.append(v / nk[j] + (reg if a == b else 0))
    elif mx.covariance_type == "tied":
        # sklearn: (X'X - sum_j nk_j mu_j mu_j') / n + reg, written as the centred sum
        for a in range(p):
            for b in range(p):
                v = sum(
                    (R[i][j] * (Xq[i][a] - means[j][a]) * (Xq[i][b] - means[j][b]) for i in range(m) for j in range(k)),
                    fmpq(0),
                )
                vals.append(v / sum(nk, fmpq(0)) + (reg if a == b else 0))
    elif mx.covariance_type == "diag":
        for j in range(k):
            for a in range(p):
                v = sum((R[i][j] * (Xq[i][a] - means[j][a]) ** 2 for i in range(m)), fmpq(0))
                vals.append(v / nk[j] + reg)
    else:
        for j in range(k):
            v = sum((R[i][j] * (Xq[i][a] - means[j][a]) ** 2 for i in range(m) for a in range(p)), fmpq(0))
            vals.append(v / nk[j] / p + reg)
    for q, v in enumerate(vals):
        h, lo = dd_of(v)
        pts.append((1, q, {"mstep": (h, lo, {})}))
    return {
        "set": mid,
        "digest": mx.digest,
        "ginputs": {"covariance_type": np.array(mx.covariance_type), "k": np.array(k)},
        "points": pts,
        "gmethods": {},
    }


# ---------------------------------------------------------------- metrics


def _contingency(a, b):
    ua, ia = np.unique(np.asarray(a), return_inverse=True)
    ub, ib = np.unique(np.asarray(b), return_inverse=True)
    M = np.zeros((len(ua), len(ub)), np.int64)
    np.add.at(M, (ia, ib), 1)
    return M


def _c2(v):
    return v * (v - 1) // 2


def external_task(lid):
    lb = S().labeling(lid)
    M = _contingency(lb.a, lb.b)
    n = int(M.sum())
    av = [int(v) for v in M.sum(axis=1)]
    bv = [int(v) for v in M.sum(axis=0)]
    cells = [int(v) for v in M[M > 0]]
    sum_c = sum(_c2(v) for v in cells)
    sum_a = sum(_c2(v) for v in av)
    sum_b = sum(_c2(v) for v in bv)
    pairs = _c2(n)
    gm = {}
    # the pair counts (scikit-learn's pair_confusion_matrix halved) and its special cases (docs/design/cluster.md
    # §2.6): ARI 1 when no pair disagrees, RI 1 when every pair agrees or there is no pair, FM 0 without a
    # same-same pair, NMI 1 for one cluster on both sides and 0 for MI = 0, AMI 1 for one cluster on both sides
    # and 0 for one on either side
    tp = sum_c
    fp = sum_b - sum_c
    fn = sum_a - sum_c
    tn = pairs - tp - fp - fn
    ri_special = tp + tn == pairs or pairs == 0
    gm["ri"] = (*dd_of(Fraction(1) if ri_special else Fraction(tp + tn, pairs)), {"special": float(ri_special)})
    ari_special = fn == 0 and fp == 0
    ari = (
        Fraction(1) if ari_special else Fraction(2 * (tp * tn - fn * fp), (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    )
    gm["ari"] = (*dd_of(ari), {"special": float(ari_special)})
    if tp == 0:
        gm["fm"] = (0.0, 0.0, {"special": 1.0})
    else:
        gm["fm"] = safe(lambda: arb(tp) / (arb(sum_b) * arb(sum_a)).sqrt(), special=0.0)

    def entropy(v):
        s = arb(0)
        for c in v:
            if c:
                s -= arb(c) / n * (arb(c) / n).log()
        return s

    def mi():
        s = arb(0)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                c = int(M[i, j])
                if c:
                    s += arb(c) / n * (arb(n) * c / (arb(av[i]) * bv[j])).log()
        return s

    both_one = len(av) == len(bv) == 1
    one_side = len(av) == 1 or len(bv) == 1
    gm["mi"] = safe(mi, special=0.0)

    def mean(ha, hb, avg):
        return {
            "min": lambda: ha if ha < hb else hb,
            "max": lambda: ha if ha > hb else hb,
            "geometric": lambda: (ha * hb).sqrt(),
            "arithmetic": lambda: (ha + hb) / 2,
        }[avg]()

    for avg in ("min", "geometric", "arithmetic", "max"):
        if both_one:
            gm[f"nmi_{avg}"] = (1.0, 0.0, {"special": 1.0})
            gm[f"ami_{avg}"] = (1.0, 0.0, {"special": 1.0})
            continue
        if one_side:  # MI = 0
            gm[f"nmi_{avg}"] = (0.0, 0.0, {"special": 1.0})
            gm[f"ami_{avg}"] = (0.0, 0.0, {"special": 1.0})
            continue
        gm[f"nmi_{avg}"] = safe(lambda avg=avg: mi() / mean(entropy(av), entropy(bv), avg), special=0.0)
        if len(av) == len(bv) == n:  # all singletons on both sides: MI = E[MI] = the normalizer, 1 (sklearn's eps)
            gm[f"ami_{avg}"] = (1.0, 0.0, {"special": 1.0})
        elif n <= AMI_MAX_N:
            gm[f"ami_{avg}"] = safe(
                lambda avg=avg: _ami(M, av, bv, n, mi(), entropy(av), entropy(bv), mean, avg), special=0.0
            )

    def hcv():
        m = mi()
        h = arb(1) if len(av) == 1 else m / entropy(av)
        c = arb(1) if len(bv) == 1 else m / entropy(bv)
        return h, c

    gm["homogeneity"] = safe(lambda: hcv()[0])
    gm["completeness"] = safe(lambda: hcv()[1])

    def v():
        h, c = hcv()
        if h + c == 0:
            return arb(0)
        return 2 * h * c / (h + c)

    gm["v"] = safe(v)
    return {"set": lid, "digest": lb.digest, "ginputs": {"n": np.array(n)}, "points": [], "gmethods": gm}


def _ami(M, av, bv, n, mi, ha, hb, mean, avg):
    """(MI - E[MI]) / (mean(H_a, H_b) - E[MI]), E[MI] the exact hypergeometric expectation (Vinh et al. 2010)."""
    lgn = arb(n + 1).lgamma()
    emi = arb(0)
    for a in av:
        for b in bv:
            lo, hi = max(1, a + b - n), min(a, b)
            base = arb(a + 1).lgamma() + arb(b + 1).lgamma() + arb(n - a + 1).lgamma() + arb(n - b + 1).lgamma() - lgn
            for c in range(lo, hi + 1):
                lt = (
                    base
                    - arb(c + 1).lgamma()
                    - arb(a - c + 1).lgamma()
                    - arb(b - c + 1).lgamma()
                    - arb(n - a - b + c + 1).lgamma()
                )
                emi += arb(c) / n * (arb(n) * c / (arb(a) * b)).log() * lt.exp()
    den = mean(ha, hb, avg) - emi
    return (mi - emi) / den


def internal_task(pid):
    s = S().points(pid)
    n, p = s.n, s.p
    lab = s.truth.tolist()  # noise -1 counts as one cluster (sklearn)
    P = exact_points(s.X)
    D2 = sqdist_matrix(P)
    ks = sorted(set(lab))
    idx = {c: [i for i in range(n) if lab[i] == c] for c in ks}
    pts = []
    sil = []
    for i in range(n):
        own = idx[lab[i]]

        def si(i=i, own=own):
            Dr = [arb_q(q).sqrt() for q in D2[i]]
            if len(own) == 1:
                return arb(0)
            a = sum((Dr[j] for j in own), arb(0)) / (len(own) - 1)
            b = None
            for c in ks:
                if c == lab[i]:
                    continue
                m = sum((Dr[j] for j in idx[c]), arb(0)) / len(idx[c])
                b = m if b is None or m < b else b
            mx = a if a > b else b
            return (b - a) / mx

        sil.append(si)
        pts.append((0, i, {"silhouette": (*settle(si), {})}))
    gm = {}
    gm["silhouette_score"] = (*settle(lambda: sum((f() for f in sil), arb(0)) / n), {})
    # Calinski-Harabasz: exact rational
    mean_all = [sum((P[i][d] for i in range(n)), fmpq(0)) / n for d in range(p)]
    cents = {c: [sum((P[i][d] for i in idx[c]), fmpq(0)) / len(idx[c]) for d in range(p)] for c in ks}
    B = sum((len(idx[c]) * sqdist(cents[c], mean_all) for c in ks), fmpq(0))
    Wd = sum((sqdist(P[i], cents[lab[i]]) for i in range(n)), fmpq(0))
    K = len(ks)
    ch = Fraction(1) if Wd == 0 else fr(B * (n - K) / (Wd * (K - 1)))
    gm["calinski_harabasz"] = (*dd_of(ch), {})

    def db():
        sc = {c: sum((arb_q(sqdist(P[i], cents[c])).sqrt() for i in idx[c]), arb(0)) / len(idx[c]) for c in ks}
        tot = arb(0)
        for c in ks:
            best = None
            for e in ks:
                if e == c:
                    continue
                r = (sc[c] + sc[e]) / arb_q(sqdist(cents[c], cents[e])).sqrt()
                best = r if best is None or r > best else best
            tot += best
        return tot / K

    gm["davies_bouldin"] = (*settle(db), {})
    return {"set": pid, "digest": s.digest, "ginputs": {}, "points": pts, "gmethods": gm}


def metrics_items():
    return [("ext", lid) for lid in S().labeling_ids()] + [
        ("int", pid) for pid in S().point_ids() if S().points(pid).truth is not None and S().points(pid).n <= 300
    ]


def metrics_task(item):
    kind, ident = item
    return external_task(ident) if kind == "ext" else internal_task(ident)


# ---------------------------------------------------------------- driver


def _items(fam):
    return {
        "hierarchy": hierarchy_sets,
        "kmeans": kmeans_sets,
        "kmedoids": kmedoids_sets,
        "fuzzy": fuzzy_sets,
        "density": density_items,
        "mixture": lambda: list(S().mixture_ids()),
        "metrics": metrics_items,
    }[fam]()


TASKS = {
    "hierarchy": hierarchy_task,
    "kmeans": kmeans_task,
    "kmedoids": kmedoids_task,
    "fuzzy": fuzzy_task,
    "density": density_task,
    "mixture": mixture_task,
    "metrics": metrics_task,
}
FAMILIES = tuple(TASKS)
EXTRA = {
    "hierarchy": "exact rationals (fmpq); sums of square roots compared in Arb to 4096 bits",
    "kmeans": "exact rationals (fmpq)",
    "fuzzy": "exact rationals (fmpq); m = 3's square roots in Arb",
    "density": "exact rationals (fmpq)",
    "mixture": "exact rationals (fmpq, fmpq_mat)",
    "metrics": "exact rationals (Fraction)",
}


def _run(task):
    fam, k, item = task
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(SECONDS)
    t0 = time.monotonic()
    try:
        res, err = TASKS[fam](item), None
    except (_Timeout, NotSettled) as e:
        res, err = None, type(e).__name__
    finally:
        signal.alarm(0)
    return fam, k, res, time.monotonic() - t0, err


def assemble(name, results):
    """Write one file from the tasks' results (in item order): one group per task."""
    from tests import reffile

    inputs = {"set": [], "kind": [], "index": [], "digest": []}
    group, gin, methods = [], {}, {}

    def put(m, over, at, entry):
        e = methods.setdefault(m, {"over": over, "index": [], "hi": [], "lo": [], "dropped": []})
        if entry is None:
            e["dropped"].append(at)
            return
        hi, lo, fields = entry
        e["index"].append(at)
        e["hi"].append(hi)
        e["lo"].append(lo)
        for key, v in fields.items():
            e.setdefault(key, []).append(v)

    gkeys = sorted({key for r in results for key in r["ginputs"]})
    for g, r in enumerate(results):
        for key in gkeys:
            gin.setdefault(key, []).append(r["ginputs"].get(key))
        for kind, idx, ms in r["points"]:
            at = len(group)
            group.append(g)
            for key, v in (("set", r["set"]), ("kind", kind), ("index", idx), ("digest", r["digest"])):
                inputs[key].append(v)
            for m, entry in ms.items():
                put(m, "point", at, entry)
        for m, entry in r["gmethods"].items():
            put(m, "group", g, entry)
    for e in methods.values():
        width = len(e["index"])
        for key in list(e):
            if key in ("over", "dropped"):
                continue
            if len(e[key]) != width:  # a field some entries lack: NaN for them
                raise ValueError(f"{name}: field {key} has {len(e[key])} of {width} values")
        for key in e:
            if key != "over":
                e[key] = np.asarray(e[key], dtype=np.int64 if key in ("index", "dropped") else np.float64)
    inputs = {
        "set": np.array(inputs["set"], dtype=str),
        "kind": np.array(inputs["kind"], np.int64),
        "index": np.array(inputs["index"], np.int64),
        "digest": np.array(inputs["digest"], dtype=str),
    }
    ginputs = {"set": np.array([r["set"] for r in results], dtype=str)}
    for key, vs in gin.items():
        arrs = [np.full((0,), np.nan) if v is None else np.asarray(v) for v in vs]
        if all(a.dtype.kind in "US" for a in arrs if a.size):
            ginputs[key] = np.array([str(a) if a.size else "" for a in arrs], dtype=str)
        elif len({a.shape for a in arrs}) == 1:
            ginputs[key] = np.stack(arrs).astype(float)
        else:
            ginputs[key] = [a.astype(float) for a in arrs]
    reffile.write(
        OUT / f"{name}.npz",
        **arbref.provenance(extra=EXTRA.get(name)),
        tool=TOOL,
        inputs=inputs,
        methods=methods,
        group=np.array(group, np.int64),
        ngroups=len(results),
        group_inputs=ginputs,
        n_requested=len(group),
    )
    print(
        f"{name}: {len(group)} points, {len(methods)} methods -> {(OUT / f'{name}.npz').relative_to(ROOT)}", flush=True
    )


def build(names, jobs):
    import resource

    start = time.monotonic()
    plans = {f: _items(f) for f in names}
    tasks = [(f, k, it) for f in names for k, it in enumerate(plans[f])]
    tasks.sort(key=lambda t: -_cost(t))  # the long ones first
    results = {f: {} for f in names}
    dropped = []
    with mp.get_context("fork").Pool(jobs, maxtasksperchild=8) as pool:
        for fam, k, res, secs, err in pool.imap_unordered(_run, tasks, chunksize=1):
            if res is None:
                dropped.append((fam, plans[fam][k], err))
                print(f"  dropped {fam} {plans[fam][k]}: {err}", flush=True)
            else:
                results[fam][k] = res
            if secs > 30:
                print(f"  {fam} {plans[fam][k]}: {secs:.0f} s", flush=True)
    for f in names:
        assemble(f, [results[f][k] for k in sorted(results[f])])
    rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 2**30
    print(
        f"{len(tasks)} tasks, {len(dropped)} dropped, {(time.monotonic() - start):.0f} s, "
        f"largest worker RSS {rss:.2f} GB",
        flush=True,
    )


def _cost(task):
    fam, _, item = task
    if fam == "metrics" and item[0] == "ext":
        return S().labeling(item[1]).n
    pid = item[0] if isinstance(item, tuple) else item
    try:
        s = S().points(pid)
    except KeyError:
        return 0
    return s.n**3 if fam in ("hierarchy", "kmedoids") else s.n * s.p * (s.k or 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog=TOOL, description=__doc__.split("\n\n")[0])
    ap.add_argument("names", nargs="*", metavar="FAMILY", help=", ".join(FAMILIES))
    ap.add_argument("--list", action="store_true", help="list the families and their data")
    from bench import cli

    cli.add_jobs(ap, default=3)
    a = ap.parse_args(argv)
    bad = [f for f in a.names if f not in FAMILIES]
    if bad:
        ap.error(f"unknown family {bad[0]!r} (known: {', '.join(FAMILIES)})")
    names = [f for f in FAMILIES if not a.names or f in a.names]
    if a.list:
        for f in names:
            items = _items(f)
            print(f"{f} ({len(items)}): " + ", ".join(map(str, items)))
        return 0
    with arbref.logged("cluster_ref"):
        build(names, a.jobs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
