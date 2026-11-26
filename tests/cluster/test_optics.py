"""OPTICS (src/stardust/cluster/_density.py, csrc/cluster/density.cpp; docs/api/cluster.md §4; docs/design/cluster.md
§2.4, §3.6, §4.4; decisions CL30, CL70-CL73): the order, reachabilities and predecessors against the exact order of
tools/cluster_ref.py; scikit-learn's OPTICS on the points in key order (its ties go to the smaller index, sd's to the
smaller key; its values rounded to 15 decimals); the xi and dbscan extractions against scikit-learn's functions run on
sd's own reachability plot; the paths, the thread counts, the input order; the errors."""

import math

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


def _sets():
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


def _blobs(seed, n=(150, 150, 150), p=2):
    g = np.random.default_rng(seed)
    X = np.vstack([g.normal(c, s, (m, p)) for c, s, m in zip((0.0, 3.0, 6.0), (0.4, 0.8, 0.3), n)])
    return np.vstack([X, g.uniform(-2, 8, (40, p))])


def test_against_the_exact_order():
    # the order is the exact one (the near ties of these sets break alike: equal computed values from one core
    # distance), the reachabilities spatial's distances (gamma(p/2 + 2)), the predecessors the exact ones
    for sid in _sets():
        s = S.points(sid.split("|")[0])
        r = CL.optics(s.X, min_samples=5)
        order, _, _ = _points("optics_order", sid, 3)
        assert _np(r.ordering).tolist() == order.astype(int).tolist(), sid
        hi, lo, f = _points("optics_reach", sid)
        got = _np(r.reachability)
        fin = np.isfinite(hi)
        assert np.array_equal(np.isfinite(got), fin), sid
        assert np.all(np.abs(got[fin] - hi[fin]) <= D.distance_bound(s.p, hi[fin])), sid
        assert _np(r.predecessor).tolist() == f["pred"].astype(int).tolist(), sid
        core, _, _ = _points("core_distance", sid)
        assert np.all(np.abs(_np(r.core_distances) - core) <= D.distance_bound(s.p, core)), sid


@LIVE
def test_sklearn_optics_on_points_in_key_order():
    from sklearn.cluster import OPTICS

    for seed in range(6):
        X = _blobs(seed, p=2 + seed % 2)
        X = X[_np(C.lex_order(A(X)))]  # the keys are the input indices: both break ties alike
        kw = {"min_samples": 5 + seed % 3}
        if seed % 2:
            kw["max_eps"] = 1.5
        r = CL.optics(X, **kw)
        sk = OPTICS(**kw).fit(X)
        assert np.array_equal(_np(r.ordering), sk.ordering_), seed
        assert np.array_equal(_np(r.predecessor), sk.predecessor_), seed
        # scikit-learn rounds its distances to 15 decimals
        fin = np.isfinite(sk.reachability_)
        assert np.array_equal(np.isfinite(_np(r.reachability)), fin)
        assert np.allclose(_np(r.reachability)[fin], sk.reachability_[fin], rtol=4e-16, atol=6e-16), seed
        assert np.allclose(_np(r.core_distances), sk.core_distances_, rtol=4e-16, atol=6e-16), seed
        assert _partition(r.labels) == _partition(sk.labels_), seed
        assert np.array_equal(_np(r.cluster_hierarchy), np.asarray(sk.cluster_hierarchy_).reshape(-1, 2)), seed


@LIVE
def test_extractions_are_scikit_learns_on_the_same_plot():
    # extract_xi and extract_dbscan are scikit-learn's cluster_optics_xi and cluster_optics_dbscan run on sd's own
    # ordering, reachabilities and predecessors: the same floating-point operations, so the same clusters
    from sklearn.cluster import cluster_optics_dbscan, cluster_optics_xi

    for seed in range(4):
        X = _blobs(10 + seed)
        r = CL.optics(X, min_samples=6)
        reach, pred, order, core = (_np(r.reachability), _np(r.predecessor), _np(r.ordering), _np(r.core_distances))
        for xi in (0.01, 0.05, 0.2, 0.5):
            for ms, mcs, corr in ((6, None, True), (4, 15, False), (10, 0.05, True)):
                e = r.extract_xi(xi=xi, min_samples=ms, min_cluster_size=mcs, predecessor_correction=corr)
                lab, hier = cluster_optics_xi(
                    reachability=reach, predecessor=pred, ordering=order, min_samples=ms,
                    min_cluster_size=mcs, xi=xi, predecessor_correction=corr,
                )  # fmt: skip
                assert _np(e.labels).tolist() == S.first_seen(lab).tolist(), (seed, xi, ms)
                assert np.array_equal(_np(e.cluster_hierarchy), np.asarray(hier).reshape(-1, 2)), (seed, xi, ms)
                assert e.cluster_method == "xi" and e.xi == xi and bits_equal(e.ordering, r.ordering)
        for eps in (0.1, 0.3, 0.8, math.inf):
            e = r.extract_dbscan(eps)
            ref = cluster_optics_dbscan(reachability=reach, core_distances=core, ordering=order, eps=eps)
            assert _np(e.labels).tolist() == S.first_seen(ref).tolist(), (seed, eps)
            assert e.cluster_hierarchy is None and e.eps == eps
        d = CL.optics(X, min_samples=6, cluster_method="dbscan", eps=0.3)
        assert bits_equal(d.labels, r.extract_dbscan(0.3).labels)


def test_paths_thread_counts_and_input_order():
    g = np.random.default_rng(3)
    X = _blobs(20, n=(1500, 1200, 800))
    order = _np(C.lex_order(A(X)))
    Xs = A(np.ascontiguousarray(X[order]))
    for code in (0, 1, 2):
        for max_eps in (math.inf, 0.5):
            a = C.optics(Xs, None, 7, max_eps, code, 1)
            b = C.optics(Xs, None, 7, max_eps, code, 2)
            assert bits_equal(a, b), (code, max_eps)
    r = CL.optics(np.asarray(Xs[:800]), metric="cityblock")
    q = CL.optics(sd.spatial.pdist(Xs[:800], metric="cityblock"), metric="precomputed")
    assert bits_equal(r.ordering, q.ordering) and bits_equal(r.reachability, q.reachability)

    def results():
        r = CL.optics(X, min_samples=8)
        q = CL.optics(X, min_samples=5, max_eps=0.4, cluster_method="dbscan", eps=0.2)
        return [r.ordering, r.reachability, r.core_distances, r.predecessor, r.labels, q.labels]

    sd.set_num_threads(1)
    base = results()
    try:
        for t in (3, 10):
            sd.set_num_threads(t)
            for k, (a, b) in enumerate(zip(base, results())):
                assert bits_equal(a, b), (t, k)
    finally:
        sd.set_num_threads(10)
    perm = g.permutation(len(X))
    a = CL.optics(X[:900], min_samples=8)
    b = CL.optics(X[:900][perm[perm < 900]], min_samples=8)
    p9 = perm[perm < 900]
    assert _np(b.ordering).tolist() == [int(np.nonzero(p9 == i)[0][0]) for i in _np(a.ordering)]
    assert bits_equal(_np(a.reachability)[p9], _np(b.reachability))


def test_result_and_nan_policy():
    X = _blobs(4)
    r = CL.optics(X)
    assert repr(r).startswith("OPTICSResult(nobs=490, n_clusters=") and "OPTICS Results" in r.summary()
    assert sorted(_np(r.ordering).tolist()) == list(range(len(X)))
    assert _np(r.predecessor)[_np(r.ordering)[0]] == -1 and math.isinf(_np(r.reachability)[_np(r.ordering)[0]])
    h = _np(r.cluster_hierarchy)
    assert h.ndim == 2 and h.shape[1] == 2 and np.all(h[:, 0] <= h[:, 1])
    f = CL.optics(X, min_samples=0.02, min_cluster_size=0.05)
    assert f.min_samples == max(2, int(0.02 * len(X)))
    Xn = X.copy()
    Xn[[1, 30], 0] = np.nan
    ro = CL.optics(Xn, nan_policy="omit")
    assert np.isnan(_np(ro.reachability)[[1, 30]]).all() and _np(ro.predecessor)[[1, 30]].tolist() == [-1, -1]
    assert sorted(_np(ro.ordering).tolist()) == sorted(set(range(len(X))) - {1, 30})
    keep = np.ones(len(X), bool)
    keep[[1, 30]] = False
    rk = CL.optics(X[keep])
    assert bits_equal(_np(ro.reachability)[keep], rk.reachability)
    assert _np(ro.extract_dbscan(0.3).labels)[[1, 30]].tolist() == [-1, -1]
    e = r.extract_xi(xi=0.1)
    assert e.n_clusters >= 1 and bits_equal(e.reachability, r.reachability)


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.optics(np.eye(9), min_samples=1), ValueError, r"^optics\(\): 'min_samples' must be >= 2, not 1"),
        (lambda: CL.optics(np.eye(9), min_samples=1.5), ValueError, r"a float 'min_samples' must be in \(0, 1\]"),
        (lambda: CL.optics(np.eye(9), min_samples=10), ValueError, r"'min_samples' \(10\) must be at most"),
        (lambda: CL.optics(np.eye(9), min_samples="5"), TypeError, r"'min_samples' must be an int or a float"),
        (lambda: CL.optics(np.eye(9), max_eps=-1.0), ValueError, r"'max_eps' must be >= 0"),
        (lambda: CL.optics(np.eye(9), cluster_method="DBSCAN"), ValueError, r"'cluster_method' must be"),
        (lambda: CL.optics(np.eye(9), cluster_method="dbscan", eps=2.0, max_eps=1.0), ValueError,
         r"'eps' \(2.0\) must be at most 'max_eps' \(1.0\)"),
        (lambda: CL.optics(np.eye(9), xi=1.5), ValueError, r"'xi' must be in \[0, 1\]"),
        (lambda: CL.optics(np.eye(9), predecessor_correction=0), TypeError, r"must be a bool"),
        (lambda: CL.optics(np.eye(9), min_cluster_size=1), ValueError, r"'min_cluster_size' must be >= 2"),
        (lambda: CL.optics(np.eye(9)).extract_dbscan(eps=0.5), TypeError, r"^extract_dbscan\(\)"),
        (lambda: CL.optics(np.eye(9), max_eps=1.0).extract_dbscan(2.0), ValueError, r"^extract_dbscan\(\): 'eps'"),
        (lambda: CL.optics(np.eye(9)).extract_xi(xi=-0.1), ValueError, r"^extract_xi\(\): 'xi' must be in"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
