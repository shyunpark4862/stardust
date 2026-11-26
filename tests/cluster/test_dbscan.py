"""DBSCAN (src/stardust/cluster/_density.py, csrc/cluster/density.cpp; docs/api/cluster.md §4; docs/design/cluster.md
§2.4, §3.6, §4.4; decisions CL12, CL28, CL72): the partition against the exact one of tools/cluster_ref.py
(d^2 <= eps^2 in exact arithmetic, the border points to their nearest core point), scikit-learn's DBSCAN everywhere
but at the border points it gives to the first cluster that reaches them, the KD tree and the scan (and every
metric's paths) giving the same bits, the thread counts, the input order, the weights and the errors."""

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


def _eps_sets():
    out = []
    for g, sid in enumerate(REF.group_inputs["set"].tolist()):
        if "|eps" in sid:
            out.append((sid, float(REF.group_inputs["eps"][g]), int(REF.group_inputs["min_samples"][g])))
    return out


def _ref_labels(sid):
    m = REF.methods["label"]
    pos, idx = D.at(REF, "label", sid, 0)
    order = np.argsort(idx, kind="stable")
    pos = pos[order]
    return m.hi[pos].astype(int), m.fields["core"][pos] == 1


def _partition(lab):
    """The partition of a labeling as a set of frozensets (noise apart)."""
    groups = {}
    for i, v in enumerate(np.asarray(lab).tolist()):
        groups.setdefault(v, []).append(i)
    return {frozenset(g) for v, g in groups.items() if v >= 0}, frozenset(groups.get(-1, ()))


def test_against_the_exact_partition():
    # §4.4: the computed distances against eps decide; on these sets (dyadic eps, exact pairs on eps/6 and the
    # lattices) that is the exact partition, border points to their nearest core point by (distance, key)
    for sid, eps, ms in _eps_sets():
        s = S.points(sid.split("|")[0])
        lab, core = _ref_labels(sid)
        r = CL.dbscan(s.X, eps=eps, min_samples=ms, weights=s.w)
        assert _np(r.labels).tolist() == lab.tolist(), sid
        assert _np(r.core_mask).tolist() == core.tolist(), sid
        assert r.n_clusters == len(set(lab.tolist()) - {-1}) and r.nobs == s.n


@LIVE
def test_sklearn_dbscan_apart_from_the_border_points():
    from sklearn.cluster import DBSCAN

    g = np.random.default_rng(1)
    for t in range(12):
        p = 2 + t % 3
        X = np.vstack([g.normal(c, 0.6, (120, p)) for c in (0.0, 2.5, 5.0)] + [g.uniform(-2, 7, (40, p))])
        w = g.uniform(0.0, 2.0, len(X)) if t % 4 == 3 else None
        eps, ms = 0.35 + 0.05 * (t % 3), 4 + t % 4
        r = CL.dbscan(X, eps=eps, min_samples=ms, weights=w)
        sk = DBSCAN(eps=eps, min_samples=ms).fit(X, sample_weight=w)
        core = _np(r.core_mask)
        skc = np.zeros(len(X), bool)
        skc[sk.core_sample_indices_] = True
        assert np.array_equal(core, skc), t
        lab = _np(r.labels)
        assert np.array_equal(lab < 0, sk.labels_ < 0), t
        assert _partition(np.where(core, lab, -1)) == _partition(np.where(core, sk.labels_, -1)), t
        # a border point joins its nearest core point (the smaller key on a tie)
        keys = S.lex_rank(X)
        d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
        for i in np.nonzero(~core & (lab >= 0))[0]:
            cand = np.nonzero(core & (d[i] <= eps))[0]
            j = min(cand, key=lambda c: (d[i, c], keys[c]))
            assert lab[i] == lab[j], (t, i)


def test_border_point_and_eps_rules():
    # scikit-learn gives the middle point to the first cluster it expands (tests/cluster/data/sklearn.json
    # rule/dbscan_border: left when the left points come first, right otherwise); sd to its nearest core point,
    # here the two at exactly 1, the smaller key: the left one, in any order
    left = [[-1.0 - 0.05 * i, 0.0] for i in range(4)]
    right = [[1.0 + 0.05 * i, 0.0] for i in range(4)]
    X = np.array(left + right + [[0.0, 0.0]])
    for idx in (np.arange(9), np.r_[4:8, 0:4, 8]):
        lab = np.empty(9, int)
        lab[idx] = _np(CL.dbscan(X[idx], eps=1.0, min_samples=4).labels)
        assert lab[8] == lab[0] != lab[4]
    assert _np(CL.dbscan(np.array([[0.0], [1.0]]), eps=1.0, min_samples=2).labels).tolist() == [0, 0]  # d <= eps
    assert _np(CL.dbscan(np.array([[0.0], [1.0]]), eps=0.999, min_samples=2).labels).tolist() == [-1, -1]
    r = CL.dbscan(np.array([[0.0], [5.0], [9.0]]), eps=1.0, min_samples=1)  # every point its own core
    assert _np(r.labels).tolist() == [0, 1, 2] and bool(_np(r.core_mask).all())


def test_paths_give_the_same_bits():
    # the KD tree and the scan (each metric of the tree), the precomputed matrix with the same keys
    g = np.random.default_rng(2)
    for p, metric in ((2, "euclidean"), (3, "cityblock"), (2, "chebyshev"), (5, "euclidean")):
        X = np.vstack([g.normal(c, 0.5, (300, p)) for c in (0.0, 2.0, 4.0)])
        order = _np(C.lex_order(A(X)))
        Xs = np.ascontiguousarray(X[order])  # key order = the input order: the matrix numbers the points alike
        code = {"euclidean": 0, "cityblock": 1, "chebyshev": 2}[metric]
        w = A(g.uniform(0.5, 1.5, len(X)))
        for wk in (None, w):
            a = C.dbscan(A(Xs), None, 0.4, 5.0, wk, code, 1)
            b = C.dbscan(A(Xs), None, 0.4, 5.0, wk, code, 2)
            assert bits_equal(a, b), (metric, wk is None)
        r = CL.dbscan(Xs, eps=0.4, metric=metric)
        q = CL.dbscan(sd.spatial.pdist(A(Xs), metric=metric), eps=0.4, metric="precomputed")
        sq = CL.dbscan(sd.spatial.squareform(sd.spatial.pdist(A(Xs), metric=metric)), eps=0.4, metric="precomputed")
        assert bits_equal(r.labels, q.labels) and bits_equal(r.core_mask, q.core_mask), metric
        assert bits_equal(q.labels, sq.labels)
    X = g.normal(0.0, 1.0, (200, 3))
    a = CL.dbscan(X, eps=0.6, metric="cosine")  # through sd.spatial.pdist
    b = CL.dbscan(sd.spatial.pdist(A(X[_np(C.lex_order(A(X)))]), metric="cosine"), eps=0.6, metric="precomputed")
    assert a.n_clusters == b.n_clusters and a.metric == "cosine"


def test_thread_counts_and_input_order():
    g = np.random.default_rng(3)
    X = np.vstack([g.normal(c, 0.5, (3000, 2)) for c in (0.0, 2.0, 4.0)] + [g.uniform(-2, 6, (500, 2))])
    w = g.uniform(0.0, 2.0, len(X))

    def results():
        a = CL.dbscan(X, eps=0.08, min_samples=6)
        b = CL.dbscan(X, eps=0.1, min_samples=5, weights=w)
        return [a.labels, a.core_mask, b.labels, b.core_mask]

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
    a = CL.dbscan(X, eps=0.08, min_samples=6)
    b = CL.dbscan(X[perm], eps=0.08, min_samples=6)
    back = np.empty(len(X), int)
    back[perm] = _np(b.labels)
    assert _partition(back) == _partition(_np(a.labels))


@LIVE
def test_weights():
    from sklearn.cluster import DBSCAN

    s = S.points("weights/k2p2")
    r = CL.dbscan(s.X, eps=1.0, min_samples=5, weights=s.w)
    sk = DBSCAN(eps=1.0, min_samples=5).fit(s.X, sample_weight=s.w)
    assert _np(r.core_mask).sum() == len(sk.core_sample_indices_)
    heavy = CL.dbscan(np.array([[0.0], [10.0], [20.0]]), eps=1.0, min_samples=5, weights=[5.0, 1.0, 0.0])
    assert _np(heavy.labels).tolist() == [0, -1, -1]  # a weight of min_samples makes a point core alone


def test_result_and_nan_policy():
    s = S.points("blobs/k3p2")
    r = CL.dbscan(s.X, eps=1.0)
    assert "DBSCAN Results" in r.summary() and repr(r).startswith("DBSCANResult(nobs=90, n_clusters=")
    assert r.eps == 1.0 and r.min_samples == 5 and r.metric == "euclidean"
    with pytest.raises(AttributeError):
        r.eps = 2.0
    X = s.X.copy()
    X[[4, 7], 1] = np.nan
    ro = CL.dbscan(X, eps=1.0, nan_policy="omit")
    assert _np(ro.labels)[[4, 7]].tolist() == [-1, -1] and not _np(ro.core_mask)[[4, 7]].any()
    assert ro.nobs == 88 and _np(ro.omitted).sum() == 2
    keep = np.ones(len(X), bool)
    keep[[4, 7]] = False
    rk = CL.dbscan(s.X[keep], eps=1.0)
    assert _partition(_np(ro.labels)[keep]) == _partition(_np(rk.labels))


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.dbscan(np.eye(3), eps=0.0), ValueError, r"^dbscan\(\): 'eps' must be finite and > 0"),
        (lambda: CL.dbscan(np.eye(3), eps=np.inf), ValueError, r"'eps' must be finite and > 0"),
        (lambda: CL.dbscan(np.eye(3), min_samples=0), ValueError, r"'min_samples' must be >= 1"),
        (lambda: CL.dbscan(np.eye(3), min_samples=2.0), TypeError, r"'min_samples' must be an int"),
        (lambda: CL.dbscan(np.eye(3), metric="manhattan"), ValueError, r"'metric' must be .* not 'manhattan'"),
        (lambda: CL.dbscan(np.eye(3), weights=[1.0, -1.0, 1.0]), ValueError, r"'weights' must be >= 0"),
        (lambda: CL.dbscan(np.eye(3), weights=[1.0, 1.0]), ValueError, r"one value per point \(3\)"),
        (lambda: CL.dbscan([[0.0], [np.nan]]), ValueError, r"contains NaN \(nan_policy='raise'\)"),
        (lambda: CL.dbscan(np.eye(3), nan_policy="drop"), ValueError, r"'nan_policy' must be"),
        (lambda: CL.dbscan(np.eye(3), 0.5), TypeError, r"^dbscan\(\)"),
        (lambda: CL.dbscan(np.ones((3, 3)), metric="precomputed"), ValueError, r"zero diagonal"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
