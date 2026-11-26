"""k-means, mini-batch and bisecting k-means (src/stardust/cluster/_kmeans.py, csrc/cluster/kmeans.cpp;
docs/api/cluster.md §3; docs/design/cluster.md §2.3, §3.3, §4.2; decisions CL10, CL23-CL25, CL41-CL43, CL48, CL49,
CL59-CL66): the exact Lloyd fixed point of tools/cluster_ref.py from the same start, the contract, the same bits on
every path (Lloyd, Hamerly's bounds, the expanded form's proven filter) and every thread count, the input order,
rng's two words, scikit-learn and R from the same start, and the errors."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S
from .conftest import LIVE

CL = sd.cluster
CORE = sd._core.cluster
REF = D.ref("kmeans")
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _rng(seed=0):
    return sd.random.default_rng(seed)


def _group(sid):
    m = REF.methods["lloyd_inertia"]
    at = np.nonzero(m.index == D.group_of(REF, sid))[0]
    if not len(at):
        return None
    return m.hi[at[0]] + m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}


def _ref_sets():
    return [s for s in REF.group_inputs["set"].tolist() if _group(s) is not None]


def _per_point(method, sid, kind):
    pos, idx = D.at(REF, method, sid, kind)
    m = REF.methods[method]
    out = np.empty(len(idx))
    out[idx] = m.hi[pos] + m.lo[pos]
    gap = None
    if "gap" in m.fields:
        gap = np.empty(len(idx))
        gap[idx] = m.fields["gap"][pos]
    return out, gap


def _init(sid):
    s = S.points(sid)
    return np.asarray(REF.group_inputs["init"][D.group_of(REF, sid)], float)[: s.k]


def _core_fit(X, init, path, w=None, tol=0.0, max_iter=300):
    X = np.asarray(X, float)
    order = CORE.lex_order(A(X))
    return CORE.kmeans(A(X), order, None if w is None else A(w), len(init), 2, A(init), 1, max_iter, tol, 1, 2, path)


# ---- the definition and the contract ---------------------------------------------------------------------------


def test_lloyd_fixed_point_of_the_references():
    # §4.2: from the references' start (tol 0: until no label changes) the exact Lloyd fixed point's labels where no
    # step met a near tie, the centres and the inertia within the contract
    for sid in _ref_sets():
        s = S.points(sid)
        r = CL.kmeans(s.X, s.k, rng=_rng(), init=_init(sid), tol=0.0, weights=s.w)
        I, f = _group(sid)
        lab, _ = _per_point("lloyd_label", sid, 0)
        got = _np(r.labels)
        if f["path_gap"] > D.NEAR_TIE:
            assert np.array_equal(got, S.first_seen(lab.astype(int))), sid
        c, _ = _per_point("lloyd_center", sid, 1)
        C = c.reshape(-1, s.p)
        # the result's centres follow its first-seen labels: map them back through the reference's labels
        num = {}
        for a, b in zip(lab.astype(int).tolist(), got.tolist()):
            num.setdefault(b, a)
        Cr = np.array([C[num[j]] for j in range(s.k)])
        bound = D.center_bound(s.n, float(np.max(np.abs(s.X))))
        assert np.max(np.abs(_np(r.centers) - Cr)) <= bound, sid
        assert abs(r.inertia - I) <= D.inertia_bound(s.n, s.p, I) + 1e-300, sid
        assert r.converged


def test_labels_at_the_rounded_centres_are_exact():
    # the scans (both forms) give the exact nearest centre wherever the reference's choice is not a near tie, the
    # scaled and shifted sets included (CL10, CL43)
    for sid in _ref_sets():
        s = S.points(sid)
        Cf = np.asarray(REF.group_inputs["centers_f"][D.group_of(REF, sid)], float)
        lab, gap = _per_point("label_f", sid, 0)
        far = gap > D.NEAR_TIE
        for path in (1, 2):
            got = _np(CORE.kmeans_predict(A(s.X), A(Cf), A(Cf), 99999, path)[0])
            assert np.array_equal(got[far], lab[far].astype(int)), (sid, path)


@pytest.mark.parametrize("path", [2, 3, 4])
def test_every_path_gives_lloyds_bits(path):
    # CL42, CL49: the bounds and the filter skip only what they prove, so labels, centres, inertia and iterations
    # have Lloyd's bits
    g = np.random.default_rng(1)
    datas = [(S.points(sid).X, _init(sid), S.points(sid).w) for sid in _ref_sets()]
    for shift in (0.0, 1e4, 1e8, 1e12):
        X = np.vstack([g.normal(c, 1.0, (300, 6)) for c in (0.0, 3.0, 7.0)]) + shift
        datas.append((X, X[:20], None))
    lat = np.array([[i, j] for i in range(30) for j in range(30)], float)  # many exact ties
    datas.append((lat, lat[[0, 5, 300, 450, 899]], None))
    sub = np.round(g.standard_normal((400, 3)) * 64) * 2.0**-1068
    datas.append((sub, sub[:4], None))
    big = g.standard_normal((5000, 40))
    datas.append((big, big[:37], None))
    for X, init, w in datas:
        base = _core_fit(X, init, 1, w)
        got = _core_fit(X, init, path, w)
        assert all(bits_equal(a, b) for a, b in zip(base[0], got[0])) and base[1:] == got[1:], (X.shape, path)


def test_thread_counts_give_the_same_bytes():
    # CL106: the path (filter, bounds) is chosen by shape and thread count, and every path has Lloyd's bits, so the
    # bytes do not depend on the thread count; X12 takes the filter at 1 and 3 threads and the bounds with the direct
    # scan at 10, X30 with k = 30 the filter with bounds at 1 and 3 and the direct scan at 10
    g = np.random.default_rng(5)
    X = np.vstack([g.normal(c, 1.0, (6000, 5)) for c in (0.0, 4.0, 9.0)])
    Xi = np.round(2.0 * g.standard_normal((9000, 2)))  # exact ties
    W = g.random(len(X))
    X12 = np.vstack([g.normal(c, 1.0, (1500, 12)) for c in g.normal(0.0, 3.0, (10, 12))])

    def results():
        out = []
        for data in (X, Xi):
            for init in ("k-means++", "random"):
                r = CL.kmeans(data, 7, rng=_rng(3), init=init)
                out += [r.labels, r.centers, r.inertia, r.n_iter]
        r = CL.kmeans(X, 30, rng=_rng(4), weights=W)
        out += [r.labels, r.centers, r.inertia]
        r = CL.kmeans(X12, 10, rng=_rng(7))
        out += [r.labels, r.centers, r.inertia, r.n_iter, r.predict(X12[::7])]
        r = CL.minibatch_kmeans(X, 5, rng=_rng(5), batch_size=500)
        out += [r.labels, r.centers, r.n_steps]
        r = CL.bisecting_kmeans(X, 6, rng=_rng(6))
        out += [r.labels, r.centers, r.tree]
        return out

    sd.set_num_threads(1)
    base = results()
    try:
        for t in (3, 10):
            sd.set_num_threads(t)
            for k, (a, b) in enumerate(zip(base, results())):
                assert bits_equal(a, b), (t, k)
    finally:
        sd.set_num_threads(10)


@pytest.mark.parametrize("fn", ["kmeans", "minibatch_kmeans", "bisecting_kmeans"])
def test_the_input_order_does_not_change_the_result(fn):
    # CL7: the draws are made in key order, so the same points in another order give the same centres and the
    # labels of the reordered points
    g = np.random.default_rng(7)
    X = np.vstack([g.normal(c, 1.0, (200, 3)) for c in (0.0, 5.0, 10.0)])
    perm = g.permutation(len(X))
    f = getattr(CL, fn)
    a = f(X, 4, rng=_rng(8))
    b = f(X[perm], 4, rng=_rng(8))
    assert np.array_equal(S.first_seen(_np(a.labels)[perm]), _np(b.labels))
    la, lb = _np(a.labels)[perm], _np(b.labels)
    for j in range(4):
        assert bits_equal(_np(a.centers)[la[lb == j][0]], _np(b.centers)[j])
    assert a.inertia == b.inertia


def test_rng_moves_two_words_and_decides():
    X = S.points("probe/300").X
    for fn in (CL.kmeans, CL.minibatch_kmeans, CL.bisecting_kmeans, CL.fuzzy_cmeans):
        g1, g2 = _rng(11), _rng(11)
        fn(X, 3, rng=g1)
        g2.integers(0, 10, shape=(2,))
        assert g1.integers(0, 2**62) == g2.integers(0, 2**62), fn
    r1 = CL.kmeans(X, 3, rng=_rng(1), init="random", n_init=4)
    r2 = CL.kmeans(X, 3, rng=_rng(1), init="random", n_init=4)
    assert bits_equal(r1.centers, r2.centers) and r1.best_init == r2.best_init
    g = _rng(1)
    CL.kmeans(X, 3, rng=g, init=X[:3])  # an array start still takes its two words
    assert g.integers(0, 2**62) != _rng(1).integers(0, 2**62)


def test_powers_of_two_scale_exactly():
    # CL43: the data times 2^e give the same labels and the centres times 2^e
    s = S.points("blobs/k3p2")
    a = CL.kmeans(s.X, 3, rng=_rng(2))
    for e in (-1060, -300, 300, 1000):
        b = CL.kmeans(s.X * 2.0**e, 3, rng=_rng(2))
        assert np.array_equal(_np(a.labels), _np(b.labels)), e
        assert np.array_equal(_np(a.centers) * 2.0**e, _np(b.centers)), e


# ---- scikit-learn, R and the behaviours of the design ----------------------------------------------------------


@LIVE
def test_scikit_learn_from_the_same_start():
    from sklearn.cluster import KMeans

    agree = 0
    for sid in _ref_sets():
        s = S.points(sid)
        I, f = _group(sid)
        if f["path_gap"] <= D.NEAR_TIE or f["relocated"] or "subnormal" in sid:
            continue
        init = _init(sid)
        sk = KMeans(s.k, init=init, n_init=1, tol=0.0, max_iter=300).fit(s.X, sample_weight=s.w)
        r = CL.kmeans(s.X, s.k, rng=_rng(), init=init, tol=0.0, weights=s.w)
        assert np.array_equal(S.first_seen(sk.labels_), _np(r.labels)), sid
        assert r.inertia == pytest.approx(sk.inertia_, rel=1e-10, abs=1e-300), sid
        agree += 1
    assert agree > 20


def test_r_lloyd_from_the_same_start():
    rec = D.data("r")
    agree = 0
    for sid in _ref_sets():
        v = rec.get(f"kmeans_lloyd/{sid}")
        if v is None or "value" not in v:
            continue
        s = S.points(sid)
        I, f = _group(sid)
        if f["path_gap"] <= D.NEAR_TIE or f["relocated"] or "subnormal" in sid:
            continue
        r = CL.kmeans(s.X, s.k, rng=_rng(), init=_init(sid), tol=0.0)
        vals = np.asarray(v["value"])
        assert np.array_equal(S.first_seen(vals[: s.n].astype(int)), _np(r.labels)), sid
        assert vals[s.n + s.k * s.p] == pytest.approx(r.inertia, rel=1e-12, abs=1e-300), sid
        agree += 1
    assert agree > 15


def test_empty_cluster_takes_the_farthest_point():
    # scikit-learn's relocation on exact distances: 0.1 is farther from 0 than 10.1 from 10 (sklearn's centred
    # data rounds the other way and moves 10.1: [0, 0, 1, 2])
    E = np.array([[0.0], [0.1], [10.0], [10.1]])
    r = CL.kmeans(E, 3, rng=_rng(), init=np.array([[0.0], [10.0], [100.0]]))
    assert _np(r.labels).tolist() == [0, 1, 2, 2] and _np(r.centers).ravel().tolist() == [0.0, 0.1, 10.05]


def test_predict_transform_and_the_result():
    s = S.points("probe/300")
    r = CL.kmeans(s.X, 3, rng=_rng(4))
    assert np.array_equal(_np(r.predict(s.X)), _np(r.labels))
    assert bits_equal(r.transform(s.Xe), sd.spatial.cdist(A(s.Xe), r.centers))
    d = _np(r.transform(s.Xe))
    assert np.array_equal(_np(r.predict(s.Xe)), np.argmin(d, axis=1))
    assert r.objective == r.inertia and math.isclose(float(np.sum(_np(r.cluster_inertia))), r.inertia, rel_tol=1e-14)
    assert _np(r.labels)[0] == 0 and sorted(set(_np(r.labels).tolist())) == [0, 1, 2]
    assert repr(r).startswith("KMeansResult(nobs=300, n_clusters=3, inertia=")
    text = r.summary()
    assert "K-Means Results" in text and "Inertia" in text and "k-means++" in text
    with pytest.raises(AttributeError):
        r.labels = None
    with pytest.raises(ValueError, match=r"^predict\(\): 'X_new' must have 2 columns"):
        r.predict(np.zeros((2, 3)))
    with pytest.raises(TypeError, match=r"^predict\(\): "):
        r.predict()


def test_weights_and_nan_policy():
    s = S.points("weights/k2p2")
    r = CL.kmeans(s.X, 2, rng=_rng(), weights=s.w)
    w = s.w
    lab = _np(r.labels)
    for j in range(2):
        sel = lab == j
        assert np.allclose(_np(r.centers)[j], np.average(s.X[sel], axis=0, weights=w[sel]), rtol=1e-14)
    X = s.X.copy()
    X[[2, 7], 1] = np.nan
    r = CL.kmeans(X, 2, rng=_rng(), nan_policy="omit")
    assert _np(r.labels)[[2, 7]].tolist() == [-1, -1] and r.nobs == len(X) - 2
    assert _np(r.omitted).sum() == 2
    with pytest.raises(ValueError, match=r"^kmeans\(\): the input contains NaN"):
        CL.kmeans(X, 2, rng=_rng())


def test_minibatch_and_bisecting():
    g = np.random.default_rng(9)
    X = np.vstack([g.normal(c, 0.5, (2000, 2)) for c in ([0, 0], [6, 0], [0, 6], [6, 6])])
    full = CL.kmeans(X, 4, rng=_rng(1))
    mb = CL.minibatch_kmeans(X, 4, rng=_rng(1))
    assert mb.inertia <= 1.01 * full.inertia and mb.n_steps >= 1 and mb.n_iter == math.ceil(mb.n_steps * 1024 / len(X))
    assert np.array_equal(_np(mb.predict(X)), _np(mb.labels))
    bk = CL.bisecting_kmeans(X, 4, rng=_rng(1))
    assert bk.inertia <= 1.01 * full.inertia
    Z = bk.tree
    assert CL.is_valid_linkage(Z) and CL.is_monotonic(Z) and _np(Z).shape == (3, 4)
    assert len(_np(bk.best_init)) == 3
    lc = CL.bisecting_kmeans(X, 6, rng=_rng(1), strategy="largest_cluster")
    assert CL.is_monotonic(lc.tree) and lc.n_clusters == 6
    assert "Bisecting K-Means Results" in bk.summary()
    # predict goes down the tree: the fit's points land in their own leaf here (well separated)
    assert np.array_equal(_np(bk.predict(X)), _np(bk.labels))


@LIVE
def test_minibatch_and_bisecting_scikit_learn_behaviour():
    # the same rules as scikit-learn's: the n_init defaults, steps and passes
    from sklearn.cluster import BisectingKMeans, MiniBatchKMeans

    g = np.random.default_rng(3)
    X = np.vstack([g.normal(c, 0.5, (3000, 3)) for c in ([0, 0, 0], [6, 0, 0], [0, 6, 0])])
    sk = MiniBatchKMeans(3, random_state=0).fit(X)
    r = CL.minibatch_kmeans(X, 3, rng=_rng(0))
    assert r.inertia == pytest.approx(sk.inertia_, rel=0.02)
    skb = BisectingKMeans(3, random_state=0).fit(X)
    b = CL.bisecting_kmeans(X, 3, rng=_rng(0))
    assert b.inertia == pytest.approx(skb.inertia_, rel=0.02)


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.kmeans(np.zeros((5, 2)), 3, rng=_rng()), ValueError,
         r"^kmeans\(\): 'k' \(3\) is larger than the number of distinct points \(1\)"),
        (lambda: CL.kmeans(np.eye(3), 3), TypeError, r"^kmeans\(\): missing 1 required keyword-only argument: 'rng'"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=1), TypeError, r"^kmeans\(\): 'rng' must be an sd.random.Generator"),
        (lambda: CL.kmeans(np.eye(3), 0, rng=_rng()), ValueError, r"^kmeans\(\): 'k' must be >= 1, not 0"),
        (lambda: CL.kmeans(np.eye(3), 5, rng=_rng()), ValueError, r"^kmeans\(\): 'k' \(5\) is larger than the number"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), init="kmeans++"), ValueError, r"^kmeans\(\): 'init' must be"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), init=np.eye(3)), ValueError, r"^kmeans\(\): an array 'init'"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), init=np.eye(3)[:2], n_init=3), ValueError, r"'n_init' must be 1"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), n_init="all"), ValueError, r"'n_init' must be 'auto'"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), tol=-1.0), ValueError, r"'tol' must be finite and >= 0"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), max_iter=0), ValueError, r"'max_iter' must be >= 1"),
        (lambda: CL.kmeans(np.eye(3), 2, rng=_rng(), n_clusters=2), TypeError, r"unexpected keyword argument"),
        (lambda: CL.kmeans([[0.0], [np.inf]], 1, rng=_rng()), ValueError, r"^kmeans\(\): 'X' contains infinity"),
        (lambda: CL.minibatch_kmeans(np.eye(5), 3, rng=_rng(), init_size=2), ValueError, r"'init_size' \(2\)"),
        (lambda: CL.minibatch_kmeans(np.eye(5), 2, rng=_rng(), max_no_improvement=-1), ValueError,
         r"'max_no_improvement' must be >= 0 or None"),
        (lambda: CL.bisecting_kmeans(np.eye(5), 2, rng=_rng(), strategy="big"), ValueError, r"'strategy' must be"),
        (lambda: CL.bisecting_kmeans(np.eye(5), 2, rng=_rng(), init=np.eye(5)[:2]), TypeError, r"'init' must be"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
