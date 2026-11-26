"""sd.cluster's references and their contract (docs/design/cluster.md §4, §7), as the tests and the accuracy
measurement (bench/accuracy) both judge them.

tests/cluster/ref/<family>.npz (tools/cluster_ref.py; layout: tests/reffile.py) holds points with the inputs "set"
(the id in tests/cluster/samples.py, with "|..." for the tuning: "|k3", "|eps0.5|ms5", "|hdbscan5"), "kind",
"index" and "digest" (the data's digest); one group per set, whose inputs are the tuning the references were
computed with; and per method the reference hi + lo:

    hierarchy  kind 0, one point per merge (the primitive definition, key CL7): single, complete, average,
               weighted, centroid, median, ward; fields a, b, size (scipy's Z row) and gap (the runner-up's
               relative distance; 0 an exact tie)
    kmeans     exact Lloyd from the group input init: lloyd_label (kind 0, field gap), lloyd_center (kind 1,
               index j p + d), lloyd_inertia (groups; fields n_iter, relocated, path_gap: the smallest gap of
               any step, 0 when a step met an exact tie); at the rounded centres
               (group input centers_f): label_f (kind 0, field gap), inertia_f; truth_mean (kind 2)
    fuzzy      at the group input centers (given centres): u_m2, u_m15, u_m3 (kind 0, index i k + j: the
               memberships for m = 2, 1.5, 3), centre1 (kind 1, index j p + d: the centres one step on at m = 2)
               and over the groups objective0
    kmedoids   pam (kind 0: the medoids' input indices, sorted), optimum (kind 1, sets of <= 20 points);
               over the groups pam_build, pam_swap (field swaps), optimum_objective (field ties); group input k
    density    label (kind 0, DBSCAN at the group's eps and min_samples; fields core, near), core_distance (kind
               0) and mst (kind 1; fields a, b) of HDBSCAN at min_samples 5; on the same "|hdbscan5" sets HDBSCAN's
               definition at min_cluster_size 5 (CL67-CL69): hdbscan_eom and hdbscan_leaf (kind 0, the labels),
               probability and outlier (kind 0, 'eom'), stability (kind 2, index the condensed cluster - n; fields
               plus = S+ and rows), selection_gap (over the groups: the smallest relative gap of an 'eom'
               comparison, 0 an exact tie); OPTICS at min_samples 5, max_eps inf: optics_reach (kind 0; field pred)
               and optics_order (kind 3, index the position, hi the input index; field near: a choice within a
               near tie)
    mixture    score, logdens_<j>, resp_<j> (kind 0, the mixture's points, at the exact covariances); score_u,
               logdens_u_<j>, resp_u_<j> (kind 0, at the given factors: group input prec_chol, scikit-learn's
               precisions_cholesky_ shape flattened); prec_chol (kind 2: the exact factor of the exact covariance,
               flattened); mstep (kind 1: weights, means, covariances in sklearn's shapes; decision CL75)
    metrics    over the labelings' groups: ari, ri, fm, mi, nmi_<avg>, ami_<avg>, homogeneity, completeness, v,
               h_true, h_pred, emi (field plus: E+ of §4.6) (field special: sklearn's special case and AMI's
               degenerate ones); per point set with true labels: silhouette (kind 0) and over the groups
               silhouette_score, calinski_harabasz, davies_bouldin, logw_1, logw_2 (the gap's log W of the true
               partition for d_power 1 and 2)

tests/cluster/data holds the other libraries' records (tools/cluster_probe.py: scipy.json, sklearn.json, r.json)
and the prototype measurements of milestone CL0 (prototype.json), each with its own "_provenance".

The contracts (docs/design/cluster.md §4; the constants are the drafts milestone CL0 measured with its prototypes,
prototype.cpp, which milestone CL2 removed (tests/cluster/data/prototype.json; tests/cluster/test_reference_data.py
holds them to these constants), and each milestone settles its own):

    the discrete results   equal to the reference's choice where its gap is above the rounding (gap > NEAR_TIE);
                           at or below it either choice
    single, complete       the height is spatial's computed distance: |h - h*| <= gamma(p / 2 + 2) h*      (§4.1)
    the other linkages     the Lance-Williams recurrence on the final tree (decisions CL45, CL55), proved in
                           §4.1: |H - H*| <= gamma(c mu + p + 2) V+, H the recurrence's scale (the squared height
                           for ward, centroid and median), V+ the recurrence with every subtraction made an
                           addition (V+ = H* for average and weighted), mu the merge's chain of roundings
                           (recurrence_chain) and c the roundings of one step (STEP_ROUNDINGS); on the output
                           heights (height_bound): squared heights gain gamma(3) for the root, and a monotone
                           linkage's height carried up from a child (CL51) is held to the child's bound
    k-means centres        |c - c*| <= C_CENTER gamma(ceil(log2 n) + 1) max |x|                         (§4.2)
    inertia                |I - I*| <= C_INERTIA gamma(p + ceil(log2 n)) I*                             (§4.2)
    PAM's objective        |O - O*| <= C_PAM gamma(ceil(log2 n) + p + 2) O*                              (§4.3)
    fuzzy memberships      |u - u*| <= 2 u (max(q, 1) (2p + 7) + k + 2), q = 1 / (m - 1)                  (§4.3)
    fuzzy centres          |c - c*| <= 4 gamma(ceil(log2 n) + 2p + k + 8) max |x| (one step from given
                           centres)                                                                    (§4.3)
    the density distances  a reachability, core distance or mutual reachability: spatial's computed distance,
                           |d - d*| <= gamma(p / 2 + 2) d* (as single linkage's heights)                (§4.4)
    HDBSCAN's stability    |S - S*| <= gamma(p / 2 + 6 + rows) S+, S+ = sum of (lambda + birth) size over the
                           cluster's rows                                                              (§4.4)
    probability, GLOSH     |v - v*| <= 2 gamma(p / 2 + 5) (absolute; both in [0, 1])                   (§4.4)
    mixture log densities  at given factors U (theorem 7, first order, doubled): a component's
                           |l - l*| <= gamma(3p + 7) (q+ + p log 2 pi) / 2 + gamma(p + 2) Lambda+ + u |l*|, q+ the
                           quadratic form with |U| and |z|, Lambda+ = sum |log U_dd|; the mixture's and the
                           responsibilities from those (mixture_bounds)                                (§4.5)
    the factor of S        full, tied: max |U - U*| <= C_PREC p gamma(p + 1) kappa_2(S) max |U*| (normwise,
                           first order; C_PREC measured);
                           diag, spherical: gamma(2) |U*| (one square root, one division)              (§4.5)
    the M step             at given responsibilities (theorem 9, first order, doubled; mstep_bound)       (§4.5)
    ARI, RI, FM            correctly rounded (the reference's hi); the counts and matrices exact        (§4.6)
    MI, entropies          gamma(ceil(log2 m) + 16) T+, T+ the sum of the terms' absolute values (mi_bound,
                           entropy_bound)                                                               (§4.6)
    the expected MI        u E+ (first order; E+ the reference's field plus: emi_bound)                (§4.6)
    NMI, AMI, h, c, V      the quotients of those (ratio_bound, first order)                          (§4.6)
    silhouette             4 gamma(ceil(n_max / 4) + p/2 + 8) + 2u, absolute (silhouette_bound)         (§4.6)
    Calinski-Harabasz,     relative, with the means' error delta = gamma(ceil(log2 n) + 13) M of the points less
    Davies-Bouldin         the first (ch_bound, db_bound; first order)                                 (§4.6)
    the gap's log W        gamma(G) / (1 - gamma(G)) + 2u |log W| for a given partition (logw_bound)      (§4.6)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("density", "fuzzy", "hierarchy", "kmeans", "kmedoids", "metrics", "mixture")
U = 2.0**-53
NEAR_TIE = 2.0**-40  # a choice whose runner-up is within this relative gap may go either way (§4)
C_DIST = 1.0  # CL0's draft (the measured largest error); the contract is height_bound's
C_HEIGHT = 2.0  # CL0's draft, superseded by height_bound (CL55)
STEP_ROUNDINGS = {"average": 3, "weighted": 1, "centroid": 5, "median": 2, "ward": 5}
SQUARED = ("centroid", "median", "ward")
C_CENTER = 2.0
C_INERTIA = 2.0
C_PAM = 2.0
C_PREC = 2.0  # milestone CL4's draft for the factor of a full or tied covariance (§4.5)
C_FIRST = 2.0  # the first-order bounds of §4.5 doubled: the products of gammas they drop
LN2PI = math.log(2 * math.pi)


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k) -> float:
    return k * U / (1 - k * U)


def at(r: reffile.Ref, method: str, sid: str, kind: int | None = None):
    """(positions in the method's arrays, the points' index) of one set's points (one kind)."""
    m = r.methods[method]
    sets = r.inputs["set"][m.index]
    keep = sets == sid
    if kind is not None:
        keep &= r.inputs["kind"][m.index] == kind
    pos = np.nonzero(keep)[0]
    return pos, r.inputs["index"][m.index][pos]


def group_of(r: reffile.Ref, sid: str) -> int:
    return int(np.nonzero(r.group_inputs["set"] == sid)[0][0])


def tree(r: reffile.Ref, method: str, sid: str):
    """The reference Z ((n - 1, 4): a, b, height, size) and the gaps of one set, in merge order."""
    pos, idx = at(r, method, sid)
    order = np.argsort(idx, kind="stable")
    pos = pos[order]
    m = r.methods[method]
    Z = np.column_stack([m.fields["a"][pos], m.fields["b"][pos], m.hi[pos], m.fields["size"][pos]])
    return Z, m.fields["gap"][pos], m.lo[pos]


def merges(Z, n):
    """The clusters of a tree as frozensets of points: the order-free comparison of two trees."""
    members = {i: frozenset([i]) for i in range(n)}
    out = set()
    for i, row in enumerate(np.asarray(Z)):
        members[n + i] = members[int(row[0])] | members[int(row[1])]
        out.add(members[n + i])
    return out


def depths(Z, n):
    """The depth of every merge: the most merges on a path below it (a merge of two points has depth 1)."""
    d = {i: 0 for i in range(n)}
    out = np.empty(len(Z), np.int64)
    for i, row in enumerate(np.asarray(Z)):
        d[n + i] = 1 + max(d[int(row[0])], d[int(row[1])])
        out[i] = d[n + i]
    return out


def center_bound(n, maxabs):
    return C_CENTER * gamma(math.ceil(math.log2(max(n, 2))) + 1) * maxabs


def inertia_bound(n, p, value):
    return C_INERTIA * gamma(p + math.ceil(math.log2(max(n, 2)))) * abs(value)


def pam_bound(n, p, value):
    return C_PAM * gamma(math.ceil(math.log2(max(n, 2))) + p + 2) * abs(value)


def membership_bound(p, k, m):
    return 2 * U * (max(1.0 / (m - 1.0), 1.0) * (2 * p + 7) + k + 2)


def fuzzy_centre_bound(n, p, k, maxabs):
    return 4 * gamma(math.ceil(math.log2(max(n, 2))) + 2 * p + k + 8) * maxabs


def distance_bound(p, d):
    return gamma(p / 2 + 2) * abs(d) + (p + 2) * 2.0**-1074


def stability_bound(p, plus, rows):
    return gamma(p / 2 + 6 + rows) * plus


def membership_strength_bound(p):
    return 2 * gamma(p / 2 + 5)


def tree_shape(Z, n, keys):
    """(children (outer, inner) per merge, sizes): the outer child is the one with the smaller key (the
    recurrence expands it first), keys the points' keys (their lexicographic ranks, or the input index)."""
    size = {i: 1 for i in range(n)}
    key = {i: int(keys[i]) for i in range(n)}
    kids = {}
    for i, row in enumerate(np.asarray(Z)):
        a, b = int(row[0]), int(row[1])
        v = n + i
        size[v] = size[a] + size[b]
        key[v] = min(key[a], key[b])
        kids[v] = (a, b) if key[a] < key[b] else (b, a)
    return kids, size


def recurrence_chain(Z, n, keys):
    """mu of every merge (§4.1): the longest chain of recurrence steps behind its value. The value of a merge (A, B)
    expands A over its subtree down to A's points, each point's value against B over B's subtree, and every
    step also uses the height of the merge it stands at: g(B) = 0 for a point, 1 + max(g(B1), g(B2), mu(B))
    otherwise; mu(v) = G(A) with G = g(B) at A's points and 1 + max(G(A1), G(A2), mu(A')) above."""
    kids, _ = tree_shape(Z, n, keys)
    mu = {}

    def up(T, leaf):
        if T < n:
            return leaf
        a, b = kids[T]
        return 1 + max(up(a, leaf), up(b, leaf), mu[T])

    out = []
    for i in range(len(Z)):
        v = n + i
        A, B = kids[v]
        mu[v] = up(A, up(B, 0))
        out.append(mu[v])
    return np.asarray(out, np.int64)


def positive_values(Z, X, method, keys):
    """V+ of every merge: the recurrence on the tree with every subtraction made an addition (the heights it
    uses are V+ too), from the exact squared (or plain) distances of the points, in float64 (all terms positive:
    a relative error of a few u, far below the bound's slack)."""
    n = X.shape[0]
    m = float(np.max(np.abs(X)))
    e = -math.frexp(m)[1] if m > 0 else 0
    if e:  # away from overflow and underflow (exact; V+ scales by 2^(2e) or 2^e)
        sq = method in SQUARED
        return np.ldexp(positive_values(Z, np.ldexp(X, e), method, keys), -(2 * e if sq else e))
    kids, size = tree_shape(Z, n, keys)
    sq = method in SQUARED
    H = {}

    def base(x, y):
        d = X[x] - X[y]
        s2 = float(np.dot(d, d))
        return s2 if sq else math.sqrt(s2)

    def comb(v1, v2, n1, n2, nq, h):
        if method == "average":
            return (n1 * v1 + n2 * v2) / (n1 + n2)
        if method == "weighted":
            return 0.5 * (v1 + v2)
        if method == "centroid":
            t = n1 + n2
            return (n1 * v1 + n2 * v2) / t + (n1 * n2) * h / (t * t)
        if method == "median":
            return 0.5 * v1 + 0.5 * v2 + 0.25 * h
        t = n1 + n2 + nq
        return ((n1 + nq) * v1 + (n2 + nq) * v2 + nq * h) / t

    def val(a, b):
        if a < n and b < n:
            return base(a, b)
        if a < n:
            a, b = b, a
        a1, a2 = kids[a]
        return comb(val(a1, b), val(a2, b), size[a1], size[a2], size[b], H[a])

    out = []
    for i in range(len(Z)):
        v = n + i
        A, B = kids[v]
        H[v] = val(A, B)
        out.append(H[v])
    return np.asarray(out)


def height_bound(Z, X, method, keys, h_ref):
    """The bound on |h - h*| of every height of the tree Z of points X (the contract of §4.1 on the output
    heights; h_ref the exact heights)."""
    n, p = X.shape
    h_ref = np.abs(np.asarray(h_ref, float))
    tiny = 2.0**-1074  # each rounding of a subnormal result adds at most half of it
    if method in ("single", "complete"):
        return gamma(p / 2 + 2) * h_ref + (p + 2) * tiny
    mu = recurrence_chain(Z, n, keys)
    c = STEP_ROUNDINGS[method]
    vp = positive_values(Z, X, method, keys)
    g = np.array([gamma(c * m + p + 5) for m in mu])
    if method in SQUARED:  # on H = h^2: |h - h*| <= |H - H*| / h* (and the root's rounding)
        b = g * vp / np.maximum(h_ref, 2.0**-1074) + 2 * U * h_ref + tiny
        b = np.where(h_ref > 0, b, np.sqrt(g * vp) + tiny)
    else:
        b = g * vp + np.array([(c * m + p + 5) * tiny for m in mu])
    if method in ("average", "weighted", "ward"):  # CL51: a height carried up is held to its child's bound
        kids, _ = tree_shape(Z, n, keys)
        for i in range(len(Z)):
            for c_ in kids[n + i]:
                if c_ >= n:
                    b[i] = max(b[i], b[c_ - n])
    return b


# ---- Gaussian mixtures (CL4) -----------------------------------------------------------------------------------


def _blocks(ctype, U, k, p):
    """The upper triangular p x p factor of each component from an array in scikit-learn's shape."""
    U = np.asarray(U, float)
    if ctype == "full":
        return [U[j] for j in range(k)]
    if ctype == "tied":
        return [U] * k
    if ctype == "diag":
        return [np.diag(U[j]) for j in range(k)]
    return [np.eye(p) * U[j] for j in range(k)]


def mixture_bounds(ctype, X, weights, means, factors, wlp):
    """Bounds of §4.5's theorem 7 at given factors (scikit-learn's precisions_cholesky_ shape): (the weighted log
    densities' (m, k), the mixture log densities' (m,), the responsibilities' (m, k)); wlp the exact weighted log
    densities (m, k). First order, doubled (C_FIRST)."""
    X = np.asarray(X, float)
    m, p = X.shape
    k = len(weights)
    B = _blocks(ctype, factors, k, p)
    E = np.empty((m, k))
    for j in range(k):
        Z = np.abs(X - means[j])
        a = Z @ np.abs(B[j])  # a_e = sum_d |U_de| |z_d|
        qplus = np.sum(a * a, axis=1)
        lam = float(np.sum(np.abs(np.log(np.diag(B[j])))))
        lw = math.log(weights[j]) if weights[j] > 0 else 0.0
        ell = wlp[:, j] - lw
        E[:, j] = gamma(3 * p + 7) * (qplus + p * LN2PI) / 2 + gamma(p + 2) * lam + U * np.abs(ell)
        E[:, j] += 2 * U * abs(lw) + U * np.abs(wlp[:, j])
    top = np.max(wlp, axis=1)
    amax = np.max(np.abs(wlp - top[:, None]), axis=1)
    lpn = top + np.log(np.sum(np.exp(wlp - top[:, None]), axis=1))
    emax = np.max(E, axis=1)
    score = 2 * emax + U * amax + gamma(k + 2) + U * (math.log(k) + np.abs(lpn))
    resp = np.exp(wlp - lpn[:, None]) * (4 * emax + 2 * U * amax + gamma(k + 4))[:, None] + 2.0**-1074
    return C_FIRST * E, C_FIRST * score, C_FIRST * resp


def precision_bound(ctype, cov, Uref, k, p):
    """The bound on |U - U*| of the factor of each covariance (scikit-learn's shapes; Uref the exact factor)."""
    Uref = np.abs(np.asarray(Uref, float))
    if ctype in ("diag", "spherical"):
        return gamma(2) * Uref
    C = np.asarray(cov, float).reshape(-1, p, p)
    R = Uref.reshape(-1, p, p)
    out = np.empty_like(R)
    for j in range(C.shape[0]):
        ev = np.linalg.eigvalsh(C[j])
        out[j] = C_PREC * p * gamma(p + 1) * (ev[-1] / ev[0]) * np.max(R[j])
    return out.reshape(Uref.shape)


def mstep_bound(ctype, X, R, ref):
    """The bound of §4.5's theorem 9 on the M step at given responsibilities R (m, k): one value per entry of
    (weights, means, covariances) in scikit-learn's shapes, as the reference's mstep; ref those exact values.
    First order, doubled."""
    X = np.asarray(X, float)
    R = np.asarray(R, float)
    m, p = X.shape
    k = R.shape[1]
    w, mu = ref[:k], ref[k : k + k * p].reshape(k, p)
    cov = ref[k + k * p :]
    nk = R.sum(axis=0)
    terms = m * k if ctype == "tied" else m
    bw = gamma(2 * m + k + 2) * np.abs(w)
    A = (R.T @ np.abs(X)) / nk[:, None]  # sum r |x_d| / nk
    dmu = gamma(2 * m + 3) * A
    bmu = dmu.ravel()
    Zs = [np.abs(X - mu[j]) for j in range(k)]
    if ctype in ("full", "tied"):
        blocks = []
        for j in range(k):
            Bj = (Zs[j] * R[:, j : j + 1]).T @ Zs[j]
            Cj = R[:, j] @ Zs[j]
            Dj = np.outer(dmu[j], Cj) + np.outer(Cj, dmu[j])
            blocks.append((Bj, Dj))
        if ctype == "full":
            b = [(gamma(2 * terms + 5) * B + D) / nk[j] for j, (B, D) in enumerate(blocks)]
            bc = np.concatenate([x.ravel() for x in b])
        else:
            B = sum(x[0] for x in blocks)
            D = sum(x[1] for x in blocks)
            bc = ((gamma(2 * terms + k + 5) * B + D) / nk.sum()).ravel()
    else:
        per = np.empty((k, p))
        for j in range(k):
            Bj = R[:, j] @ (Zs[j] ** 2)
            Dj = 2 * dmu[j] * (R[:, j] @ Zs[j])
            per[j] = (gamma(2 * terms + 5) * Bj + Dj) / nk[j]
        if ctype == "diag":
            bc = per.ravel()
        else:
            bc = per.sum(axis=1) / p + gamma(p + 1) * np.abs(cov)
    bc = bc + U * np.abs(cov)
    return C_FIRST * np.concatenate([bw, bmu, bc])


# ---- the indices (§4.6; milestone CL5) ---------------------------------------------------------------------------


def contingency(a, b):
    """(cells (m,) of the nonzero counts, row sums, column sums, n) of two labelings, as numpy int64."""
    _, ia = np.unique(np.asarray(a), return_inverse=True)
    _, ib = np.unique(np.asarray(b), return_inverse=True)
    M = np.zeros((ia.max() + 1, ib.max() + 1), np.int64)
    np.add.at(M, (ia, ib), 1)
    return M, M.sum(axis=1), M.sum(axis=0), int(M.sum())


def _log_depth(m):
    return math.ceil(math.log2(max(m, 2)))


def mi_terms(M, n):
    """|t_ij| = (n_ij / n) |log(n n_ij / (a_i b_j))| of the nonzero cells (doubles: a bound's input)."""
    a, b = M.sum(axis=1), M.sum(axis=0)
    i, j = np.nonzero(M)
    c = M[i, j].astype(float)
    return c / n * np.abs(np.log(n * c / (a[i].astype(float) * b[j])))


def mi_bound(M, n):
    """|MI^ - MI| <= gamma(ceil(log2 m) + 16) T+, T+ = sum |t_ij| (m nonzero cells): each term's logarithm within 9u
    relatively (log1p near 1), two more roundings, the pairwise sum (§4.6)."""
    t = mi_terms(M, n)
    return gamma(_log_depth(t.size) + 16) * float(t.sum())


def entropy_bound(counts, value):
    """|H^ - H| <= gamma(ceil(log2 k) + 16) H (positive terms)."""
    return gamma(_log_depth(int(np.count_nonzero(counts))) + 16) * abs(value)


def emi_bound(plus):
    """|E^ - E| <= u E+ (first order; E+ the reference's field: the recurrence's steps, the sums' depths, §4.6)."""
    return U * plus


def ratio_bound(num, dnum, den, dden):
    """|q^ - q| for q = num / den from the errors of both, first order, plus the division's rounding."""
    q = abs(num / den) if den else 0.0
    return (dnum + q * dden) / abs(den) + U * q if den else math.inf


def silhouette_bound(n_max, p):
    """|s^ - s| <= 4 gamma(ceil(n_max / 4) + p / 2 + 8) + 2u (absolute; (a + b) / max(a, b) <= 2): the distances
    within gamma(p/2 + 2), the four interleaved sums, the means and the quotient (§4.6)."""
    return 4 * gamma(math.ceil(n_max / 4) + p / 2 + 8) + 2 * U


def logw_bound(n, p, k, power, value):
    """|log W^ - log W| <= gamma(max(1, power)(p/2 + 2) + ceil(n/4) + ceil(log2 n) + k + 12) + u |log W| (§4.6)."""
    g = gamma(max(1.0, power) * (p / 2 + 2) + math.ceil(n / 4) + _log_depth(n) + k + 12)
    return g / (1 - g) + 2 * U * abs(value)


def _translated(X):
    Y = np.asarray(X, float) - np.asarray(X, float)[0]
    return Y, float(np.max(np.abs(Y))) if Y.size else 0.0


def ch_bound(X, labels, value):
    """|CH^ - CH| <= CH gamma(A)(3 + 4 M sqrt(n p / B)) + CH n p delta^2 / W, A = p + ceil(log2 n) + K + 16, M the
    largest |coordinate| of the points less the first, delta = gamma(ceil(log2 n) + 13) M the means' error, B and
    W the between and within sums (§4.6: the means' error enters B to first order, W to second)."""
    Y, Mx = _translated(X)
    n, p = Y.shape
    ks = np.unique(labels)
    K = len(ks)
    mu = Y.mean(axis=0)
    B = sum(np.sum(labels == c) * np.sum((Y[labels == c].mean(axis=0) - mu) ** 2) for c in ks)
    W = sum(np.sum((Y[labels == c] - Y[labels == c].mean(axis=0)) ** 2) for c in ks)
    A = p + _log_depth(n) + K + 16
    delta = gamma(_log_depth(n) + 13) * Mx
    cond = 4 * Mx * math.sqrt(n * p / B) if B > 0 else (0.0 if Mx == 0 else math.inf)
    rel = gamma(A) * (3 + cond) + (n * p * delta**2 / W if W > 0 else 0.0)
    return abs(value) * rel


def db_bound(X, labels, value):
    """|DB^ - DB| <= DB (gamma(A) + 4 sqrt(p) delta / m) with m the smallest positive s_c + s_e or distance between
    means, delta = gamma(ceil(log2 n) + 13) M as in ch_bound (§4.6: each ratio's parts move by sqrt(p) delta)."""
    Y, Mx = _translated(X)
    n, p = Y.shape
    ks = np.unique(labels)
    K = len(ks)
    mus = np.array([Y[labels == c].mean(axis=0) for c in ks])
    s = np.array([np.mean(np.sqrt(np.sum((Y[labels == c] - mus[q]) ** 2, axis=1))) for q, c in enumerate(ks)])
    d = np.sqrt(((mus[:, None, :] - mus[None, :, :]) ** 2).sum(axis=2))
    cand = [v for v in list((s[:, None] + s[None, :])[np.triu_indices(K, 1)]) + list(d[np.triu_indices(K, 1)]) if v > 0]
    m = min(cand) if cand else math.inf
    A = p + _log_depth(n) + K + 16
    delta = gamma(_log_depth(n) + 13) * Mx
    return abs(value) * (gamma(A) + 4 * math.sqrt(p) * delta / m) + gamma(A) * abs(value)
