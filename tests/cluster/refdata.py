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
               0) and mst (kind 1; fields a, b) of HDBSCAN at min_samples 5
    mixture    score, logdens_<j>, resp_<j> (kind 0, the mixture's points), mstep (kind 1: weights, means,
               covariances in sklearn's shapes)
    metrics    over the labelings' groups: ari, ri, fm, mi, nmi_<avg>, ami_<avg>, homogeneity, completeness, v
               (field special: sklearn's special case); per point set with true labels: silhouette (kind 0) and
               over the groups silhouette_score, calinski_harabasz, davies_bouldin

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
