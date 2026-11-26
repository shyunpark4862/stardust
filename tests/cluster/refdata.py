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

The contracts (docs/design/cluster.md §4; the constants are the drafts milestone CL0 measured with the prototypes
of csrc/cluster/prototype.cpp (tests/cluster/data/prototype.json; tests/cluster/test_reference_data.py holds them
to these constants), and each milestone settles its own):

    the discrete results   equal to the reference's choice where its gap is above the rounding (gap > NEAR_TIE);
                           at or below it either choice
    single, complete       the height is the computed distance: |h - h*| <= C_DIST u h*                    (§4.1)
    the other linkages     the Lance-Williams recurrence on the final tree (decision CL45):
                           |h - h*| <= C_HEIGHT u (p + d) h*, d the merge's depth (merges below it)      (§4.1)
    k-means centres        |c - c*| <= C_CENTER gamma(ceil(log2 n) + 1) max |x|                         (§4.2)
    inertia                |I - I*| <= C_INERTIA gamma(p + ceil(log2 n)) I*                             (§4.2)
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("density", "hierarchy", "kmeans", "kmedoids", "metrics", "mixture")
U = 2.0**-53
NEAR_TIE = 2.0**-40  # a choice whose runner-up is within this relative gap may go either way (§4)
C_DIST = 1.0
C_HEIGHT = 2.0
C_CENTER = 2.0
C_INERTIA = 2.0


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


def height_bound(h_ref, p, depth):
    """|h - h*| allowed for a height of the tree recurrence (average, weighted, centroid, median, ward)."""
    return C_HEIGHT * U * (p + depth) * np.abs(h_ref)


def center_bound(n, maxabs):
    return C_CENTER * gamma(math.ceil(math.log2(max(n, 2))) + 1) * maxabs


def inertia_bound(n, p, value):
    return C_INERTIA * gamma(p + math.ceil(math.log2(max(n, 2)))) * abs(value)
