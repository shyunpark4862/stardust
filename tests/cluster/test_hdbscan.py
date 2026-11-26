"""HDBSCAN (src/stardust/cluster/_density.py, csrc/cluster/density.cpp, csrc/cluster/mst.cpp; docs/api/cluster.md §4;
docs/design/cluster.md §2.4, §3.6, §4.4; decisions CL29, CL67-CL69, CL72): the core distances, the minimum
spanning tree of the mutual reachability, the labels ('eom' and 'leaf'), probabilities, GLOSH scores and stabilities
against the exact definition of tools/cluster_ref.py; scikit-learn's HDBSCAN where its tree has no equal heights
(min_samples = 1) with every option; the order-free result where scikit-learn's depends on the input order; the
paths (KD tree and Boruvka, scan and Prim, the matrix) and the thread counts; the lazy centres; the errors."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S
from .conftest import LIVE

CL = sd.cluster
C = sd._core.cluster
REF = D.ref("density")
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _hd_sets():
    return [sid for sid in REF.group_inputs["set"].tolist() if "|hdbscan" in sid]


def _points(method, sid, kind=0):
    m = REF.methods[method]
    pos, idx = D.at(REF, method, sid, kind)
    order = np.argsort(idx, kind="stable")
    pos = pos[order]
    return m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def _partition(lab):
    groups = {}
    for i, v in enumerate(np.asarray(lab).tolist()):
        groups.setdefault(v, []).append(i)
    return {frozenset(g) for v, g in groups.items() if v >= 0}


def test_core_distances_and_the_spanning_tree():
    # §4.4: the core distances and the tree's weights are spatial's distances (gamma(p/2 + 2)); the tree's edges are
    # the exact one's (no near ties among these sets' weights)
    for sid in _hd_sets():
        s = S.points(sid.split("|")[0])
        order = _np(C.lex_order(A(s.X)))
        Xs = A(np.ascontiguousarray(s.X[order]))
        core_ref, _, _ = _points("core_distance", sid)
        for path in (1, 2):
            core = np.empty(s.n)
            core[order] = _np(C.core_distances(Xs, None, 5, 0, path))
            assert np.all(np.abs(core - core_ref) <= D.distance_bound(s.p, core_ref)), (sid, path)
        r = CL.hdbscan(s.X, min_samples=5)
        Z = _np(r.single_linkage)
        w_ref, _, f = _points("mst", sid, 1)
        assert np.all(np.abs(np.sort(Z[:, 2]) - w_ref) <= D.distance_bound(s.p, w_ref)), sid
        assert np.all(np.diff(Z[:, 2]) >= 0) and Z[-1, 3] == s.n


def test_against_the_exact_definition():
    # the labels of 'eom' and 'leaf' are the definition's (the merges of one height taken together, scikit-learn's
    # selection; selection gaps above the rounding on these sets), the probabilities and GLOSH scores and the
    # stabilities within the contract of §4.4
    for sid in _hd_sets():
        s = S.points(sid.split("|")[0])
        r = CL.hdbscan(s.X, min_samples=5)
        rl = CL.hdbscan(s.X, min_samples=5, cluster_selection_method="leaf")
        eom, _, _ = _points("hdbscan_eom", sid)
        leaf, _, _ = _points("hdbscan_leaf", sid)
        assert _np(r.labels).tolist() == eom.astype(int).tolist(), sid
        assert _np(rl.labels).tolist() == leaf.astype(int).tolist(), sid
        b = D.membership_strength_bound(s.p)
        for field, name in (("probabilities", "probability"), ("outlier_scores", "outlier")):
            hi, lo, _ = _points(name, sid)
            assert np.all(np.abs(_np(getattr(r, field)) - hi - lo) <= b), (sid, field)
        st_hi, st_lo, f = _points("stability", sid, 2)
        got = _np(r._stability)
        assert len(got) == len(st_hi), sid
        fin = np.isfinite(st_hi)
        bound = D.stability_bound(s.p, f["plus"][fin], f["rows"][fin])
        assert np.all(np.abs(got[fin] - st_hi[fin] - st_lo[fin]) <= bound), sid
        assert np.array_equal(np.isnan(got), np.isnan(st_hi)) and np.array_equal(np.isinf(got), np.isinf(st_hi))
        assert np.all((_np(r.probabilities) > 0) == (_np(r.labels) >= 0))


@LIVE
def test_sklearn_without_equal_heights():
    # min_samples = 1: the mutual reachability is the distance, a random tree has no equal heights, and scikit-learn's
    # labels and probabilities are sd's, option by option
    from sklearn.cluster import HDBSCAN

    g = np.random.default_rng(5)
    checked = 0
    for t in range(16):
        k = 2 + t % 4
        X = np.vstack([g.normal(g.uniform(-8, 8, 2), g.uniform(0.3, 1.2), (g.integers(15, 90), 2)) for _ in range(k)])
        X = np.vstack([X, g.uniform(-10, 10, (15, 2))])
        options = [
            {},
            {"cluster_selection_method": "leaf"},
            {"min_cluster_size": 12},
            {"allow_single_cluster": True},
            {"max_cluster_size": 40},
            {"alpha": 0.7},
        ]
        for kw in options:
            r = CL.hdbscan(X, min_samples=1, **kw)
            sk = HDBSCAN(min_samples=1, copy=True, **kw).fit(X)
            assert _partition(r.labels) == _partition(sk.labels_), (t, kw)
            assert np.allclose(_np(r.probabilities), sk.probabilities_, rtol=1e-13, atol=1e-15), (t, kw)
            checked += 1
        sk = HDBSCAN(min_samples=1, copy=True, store_centers="both").fit(X)
        r = CL.hdbscan(X, min_samples=1)
        num = {}
        for a, b in zip(_np(r.labels).tolist(), sk.labels_.tolist()):
            if a >= 0:
                num[a] = b
        perm = [num[j] for j in range(r.n_clusters)]
        assert np.allclose(_np(r.centroids), sk.centroids_[perm], rtol=1e-13, atol=1e-13), t
        assert np.array_equal(_np(r.medoids), sk.medoids_[perm]), t
    assert checked == 96


@LIVE
def test_equal_heights_are_order_free():
    # min_samples > 1: equal mutual reachabilities are the rule (a point's core distance to all its near
    # neighbours), scikit-learn's tree then follows its Prim's order and its labels follow the input order; sd's
    # take the merges of one height together, so the same points in any order give the same labels and bits
    from sklearn.cluster import HDBSCAN

    sk_changes = 0
    for t in range(6):
        g = np.random.default_rng(t)
        k = g.integers(2, 6)
        X = np.vstack([g.normal(g.uniform(-8, 8, 2), g.uniform(0.3, 1.5), (g.integers(10, 150), 2)) for _ in range(k)])
        X = np.vstack([X, g.uniform(-10, 10, (20, 2))])
        perm = g.permutation(len(X))
        a = CL.hdbscan(X)
        b = CL.hdbscan(X[perm])
        assert _partition(_np(a.labels)[perm]) == _partition(_np(b.labels)), t
        assert bits_equal(_np(a.probabilities)[perm], _np(b.probabilities)), t
        assert bits_equal(_np(a.outlier_scores)[perm], _np(b.outlier_scores)), t
        s1 = HDBSCAN(copy=True).fit(X).labels_
        s2 = HDBSCAN(copy=True).fit(X[perm]).labels_
        sk_changes += _partition(s1[perm]) != _partition(s2)
        # the flat clustering at a cut depends on the tree's components alone: scikit-learn's equals sd's
        for cut in (0.2, 0.4):
            ref = HDBSCAN(copy=True).fit(X).dbscan_clustering(cut, min_cluster_size=5)
            assert _partition(a.dbscan_clustering(cut)) == _partition(ref), (t, cut)
    assert sk_changes >= 2  # measured: 2 of 6


@LIVE
def test_cluster_selection_epsilon():
    # scikit-learn 1.9.1 with numpy 2.5 raises in traverse_upwards as soon as a leaf is born below the epsilon (its
    # one-element array taken as an index), so the rule is checked on a case worked by hand: two blobs 2 apart and a
    # third 20 away; with epsilon 3 the close pair is one cluster (their parent, born at a larger distance), with
    # epsilon 1 they stay apart; 'leaf' and 'eom' alike, and allow_single_cluster lets the root be chosen
    from sklearn.cluster import HDBSCAN

    g = np.random.default_rng(8)
    X = np.vstack([g.normal(c, 0.15, (40, 2)) for c in ([0.0, 0.0], [2.0, 0.0], [20.0, 0.0])])
    with pytest.raises(TypeError):
        HDBSCAN(copy=True, cluster_selection_epsilon=3.0, cluster_selection_method="leaf").fit(X)
    for method in ("eom", "leaf"):
        apart = CL.hdbscan(X, cluster_selection_method=method, cluster_selection_epsilon=1.0)
        joined = CL.hdbscan(X, cluster_selection_method=method, cluster_selection_epsilon=3.0)
        lab = _np(joined.labels)
        assert apart.n_clusters == 3 and joined.n_clusters == 2, method
        assert len(set(lab[:80].tolist())) == 1 and lab[80] != lab[0]
        root = CL.hdbscan(X, cluster_selection_method=method, cluster_selection_epsilon=30.0, allow_single_cluster=True)
        assert root.n_clusters == 1, method


def test_paths_and_thread_counts():
    g = np.random.default_rng(7)
    X = np.vstack([g.normal(c, 0.5, (1500, 2)) for c in (0.0, 2.0, 4.0)] + [g.uniform(-2, 6, (300, 2))])
    order = _np(C.lex_order(A(X)))
    Xs = A(np.ascontiguousarray(X[order]))
    o = sd.arange(len(X), dtype=sd.int64)
    for alpha in (1.0, 0.8):
        outs = [C.hdbscan(Xs, None, o, 5, alpha, 0, path, 10, 0, False, 0.0, -1) for path in (0, 1, 2)]
        assert bits_equal(outs[0], outs[1]) and bits_equal(outs[0], outs[2]), alpha
    for metric, code in (("cityblock", 1), ("chebyshev", 2)):
        a = C.hdbscan(Xs[:900], None, o[:900], 5, 1.0, code, 1, 10, 0, False, 0.0, -1)
        b = C.hdbscan(Xs[:900], None, o[:900], 5, 1.0, code, 2, 10, 0, False, 0.0, -1)
        assert bits_equal(a, b), metric
        r = CL.hdbscan(np.asarray(Xs[:900]), metric=metric)
        q = CL.hdbscan(sd.spatial.pdist(Xs[:900], metric=metric), metric="precomputed")
        assert bits_equal(r.labels, q.labels) and bits_equal(r.probabilities, q.probabilities), metric
        assert bits_equal(r.single_linkage, q.single_linkage)

    def results():
        r = CL.hdbscan(X)
        q = CL.hdbscan(X, cluster_selection_method="leaf", min_samples=3, alpha=1.5)
        return [r.labels, r.probabilities, r.outlier_scores, r.single_linkage, r.persistence, q.labels, r.centroids]

    sd.set_num_threads(1)
    base = results()
    try:
        for t in (3, 10):
            sd.set_num_threads(t)
            for k, (a, b) in enumerate(zip(base, results())):
                assert bits_equal(a, b), (t, k)
    finally:
        sd.set_num_threads(10)


def test_result_fields():
    s = S.points("density/k3p2")
    r = CL.hdbscan(s.X)
    n = s.n
    assert repr(r).startswith("HDBSCANResult(nobs=132, n_clusters=") and "HDBSCAN Results" in r.summary()
    Z = _np(r.single_linkage)
    assert Z.shape == (n - 1, 4) and bool(CL.is_valid_linkage(r.single_linkage))
    ct = r.condensed_tree
    assert ct._fields == ("parent", "child", "lambda_val", "child_size")
    assert int(_np(ct.parent).min()) == n and np.all(_np(ct.child_size) >= 1)
    pts = _np(ct.child) < n
    assert sorted(_np(ct.child)[pts].tolist()) == list(range(n))  # every point leaves once
    assert _np(r.persistence).shape == (r.n_clusters,) and np.all(_np(r.persistence) >= 0)
    sc = _np(r.outlier_scores)
    assert np.all((sc >= 0) & (sc <= 1))
    assert _np(r.centroids).shape == (r.n_clusters, 2) and _np(r.medoids).shape == (r.n_clusters, 2)
    for j in range(r.n_clusters):  # a medoid is a point of its cluster
        assert any(np.array_equal(_np(r.medoids)[j], x) for x in s.X[_np(r.labels) == j])
    rp = CL.hdbscan(sd.spatial.pdist(A(s.X)), metric="precomputed")
    assert rp.centroids is None and rp.medoids is None
    rc = CL.hdbscan(s.X, metric="cosine")
    assert _np(rc.medoids).shape == (rc.n_clusters, 2)
    # the duplicates of scikit-learn's probe (tests/cluster/data/sklearn.json rule/hdbscan_duplicates): 30 copies of
    # one point make a cluster of their own (core distance 0)
    b = S.points("blobs/k3p2").X
    Xd = np.vstack([b, np.repeat(b[:1], 30, axis=0)])
    assert sorted(set(_np(CL.hdbscan(Xd).labels).tolist())) == [0, 1, 2]
    X = s.X.copy()
    X[[2, 9], 0] = np.nan
    ro = CL.hdbscan(X, nan_policy="omit")
    assert np.isnan(_np(ro.probabilities)[[2, 9]]).all() and _np(ro.labels)[[2, 9]].tolist() == [-1, -1]
    assert _np(ro.single_linkage).shape == (n - 3, 4)
    assert _np(ro.dbscan_clustering(0.5))[[2, 9]].tolist() == [-1, -1]
    one = CL.hdbscan(S.points("blobs/k2p1").X, allow_single_cluster=True, min_cluster_size=30)
    assert one.n_clusters >= 1


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.hdbscan(np.eye(3), min_cluster_size=1), ValueError,
         r"^hdbscan\(\): 'min_cluster_size' must be >= 2"),
        (lambda: CL.hdbscan(np.eye(3), min_samples=4), ValueError, r"'min_samples' \(4\) must be at most the number"),
        (lambda: CL.hdbscan([[0.0]]), ValueError, r"at least 2 points"),
        (lambda: CL.hdbscan(np.eye(9), alpha=0.0), ValueError, r"'alpha' must be finite and > 0"),
        (lambda: CL.hdbscan(np.eye(9), cluster_selection_epsilon=-1.0), ValueError, r"'cluster_selection_epsilon'"),
        (lambda: CL.hdbscan(np.eye(9), cluster_selection_method="EOM"), ValueError, r"'cluster_selection_method'"),
        (lambda: CL.hdbscan(np.eye(9), allow_single_cluster=1), TypeError, r"'allow_single_cluster' must be a bool"),
        (lambda: CL.hdbscan(np.eye(9), max_cluster_size=0), ValueError, r"'max_cluster_size' must be >= 1"),
        (lambda: CL.hdbscan(np.eye(9), copy=True), TypeError, r"^hdbscan\(\): unexpected keyword argument 'copy'"),
        (lambda: CL.hdbscan(np.eye(9)).dbscan_clustering(), TypeError, r"^dbscan_clustering\(\)"),
        (lambda: CL.hdbscan(np.eye(9)).dbscan_clustering(np.nan), ValueError, r"'cut_distance' must not be NaN"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
