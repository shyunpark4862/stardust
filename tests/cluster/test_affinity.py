"""Affinity propagation (src/stardust/cluster/_affinity.py, csrc/cluster/affinity.cpp; docs/api/cluster.md §7;
docs/design/cluster.md §2.9, §3.12; decisions CL40, CL91-CL94): scikit-learn 1.9.1's _affinity_propagation with zero
noise on the points in key order (the same exemplars, labels and iterations), the two inputs, the input order, the
thread counts, the equal case, the stop, predict, nan_policy, the result and the errors."""

import warnings

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import samples as S

CL = sd.cluster
CORE = sd._core.cluster
A = sd.asarray


def _np(a):
    return np.asarray(a)


class _ZeroNoise:
    """A random_state whose standard normals are all 0: scikit-learn's _affinity_propagation without its noise."""

    def standard_normal(self, size):
        return np.zeros(size)


def _sklearn(Sm, pref, damping=0.5, max_iter=200, conv=15):
    from sklearn.cluster._affinity_propagation import _affinity_propagation

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _affinity_propagation(
            Sm.copy(), preference=np.asarray(pref), convergence_iter=conv, max_iter=max_iter, damping=damping,
            verbose=False, return_n_iter=True, random_state=_ZeroNoise(),
        )  # fmt: skip


def _similarity(X):
    """-sqeuclidean (sd.spatial's) of the points in key order, and the order."""
    o = _np(CORE.lex_order(A(X)))
    Xk = np.ascontiguousarray(X[o])
    return -_np(sd.spatial.cdist(A(Xk), A(Xk), metric="sqeuclidean")) + 0.0, o


def _same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return len(np.unique(a)) == len(np.unique(b)) == len({(x, y) for x, y in zip(a.tolist(), b.tolist())})


def test_scikit_learn_without_noise():
    # CL40, CL91: the same exemplars, partition and iterations as scikit-learn's algorithm with zero noise, on every
    # test set (precomputed similarities in key order: the keys are the input index)
    count = 0
    for sid in S.point_ids(max_n=400):
        Sm, o = _similarity(S.points(sid).X)
        for damping, mi, conv in ((0.5, 200, 15), (0.9, 200, 15), (0.7, 30, 5)):
            centres, lab, it = _sklearn(Sm, np.median(Sm), damping, mi, conv)
            r = CL.affinity_propagation(A(Sm), affinity="precomputed", damping=damping, max_iter=mi,
                                        convergence_iter=conv)  # fmt: skip
            assert r.n_iter == it, (sid, damping)
            assert sorted(_np(r.exemplars).tolist()) == sorted(np.asarray(centres).tolist()), (sid, damping)
            assert _same_partition(r.labels, lab), (sid, damping)
            count += 1
    assert count >= 100
    # a given preference, and one per point
    Sm, _ = _similarity(S.points("blobs/k3p2").X)
    for pref in (-50.0, np.linspace(-80.0, -5.0, len(Sm))):
        centres, lab, it = _sklearn(Sm, pref)
        r = CL.affinity_propagation(A(Sm), affinity="precomputed", preference=pref)
        assert r.n_iter == it and _same_partition(r.labels, lab)
        assert sorted(_np(r.exemplars).tolist()) == sorted(np.asarray(centres).tolist())


def test_observations_and_input_order():
    for sid in ("blobs/k3p2", "lattice/4x4-shuffled", "dups/4x5", "probe/300", "huge/k3p2"):
        X = S.points(sid).X
        Sm, o = _similarity(X)
        a = CL.affinity_propagation(X)
        b = CL.affinity_propagation(A(Sm), affinity="precomputed")
        lab = np.empty(len(X), np.int64)
        lab[o] = _np(b.labels)
        assert _same_partition(a.labels, lab) and a.n_iter == b.n_iter, sid
        assert sorted(o[_np(b.exemplars)].tolist()) == sorted(_np(a.exemplars).tolist())
        perm = np.random.default_rng(1).permutation(len(X))
        c = CL.affinity_propagation(X[perm])
        assert _same_partition(_np(c.labels), _np(a.labels)[perm]) and c.n_iter == a.n_iter
        assert sorted(perm[_np(c.exemplars)].tolist()) == sorted(_np(a.exemplars).tolist())
        assert bits_equal(_np(c.exemplar_points), X[perm][_np(c.exemplars)])


def test_threads():
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(c, 1.0, (300, 2)) for c in (0.0, 5.0, 10.0)])
    out = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        r = CL.affinity_propagation(X, damping=0.8)
        out.append((_np(r.labels), _np(r.exemplars), r.n_iter, r.converged))
    sd.set_num_threads(10)
    for o in out[1:]:
        assert bits_equal(o[0], out[0][0]) and bits_equal(o[1], out[0][1]) and o[2:] == out[0][2:]


def test_equal_case_stop_and_no_exemplar():
    # scikit-learn's equal case: identical points (every similarity 0, the median preference 0): one cluster
    r = CL.affinity_propagation(np.ones((5, 2)))
    assert r.n_clusters == 1 and r.n_iter == 0 and _np(r.labels).tolist() == [0] * 5 and r.converged
    # equal similarities with a larger preference: every point its own exemplar
    Sm = np.full((4, 4), -1.0)
    r = CL.affinity_propagation(A(Sm), affinity="precomputed", preference=0.0)
    assert r.n_clusters == 4 and _np(r.labels).tolist() == [0, 1, 2, 3]
    r = CL.affinity_propagation(np.array([[2.0, 3.0]]))
    assert r.n_clusters == 1 and _np(r.exemplars).tolist() == [0]
    # the stop: max_iter reached is not an error
    X = S.points("blobs/k3p2").X
    r = CL.affinity_propagation(X, max_iter=3)
    assert r.n_iter == 3 and not r.converged
    # no exemplar emerged: every label -1, no cluster, predict gives -1
    Sm, _ = _similarity(X)
    centres, lab, it = _sklearn(Sm, np.median(Sm), 0.5, 1, 15)
    r = CL.affinity_propagation(A(Sm), affinity="precomputed", max_iter=1)
    assert len(centres) == 0 and r.n_clusters == 0 and np.all(_np(r.labels) == -1) and not r.converged


def test_predict():
    X = S.points("blobs/k3p2").X
    r = CL.affinity_propagation(X, preference=-60.0)
    assert bits_equal(_np(r.predict(X)), _np(r.labels))  # the fit's points get the fit's labels
    assert bits_equal(_np(r.exemplar_points), X[_np(r.exemplars)])
    new = np.array([[0.0, 0.0], [6.0, 0.0], [100.0, 100.0]])
    d = ((new[:, None, :] - _np(r.exemplar_points)[None]) ** 2).sum(axis=2)
    assert _np(r.predict(new)).tolist() == np.argmin(d, axis=1).tolist()
    rp = CL.affinity_propagation(A(_similarity(X)[0]), affinity="precomputed")
    with pytest.raises(ValueError, match=r"^predict\(\): a fit on precomputed similarities cannot place new points$"):
        rp.predict(new)
    with pytest.raises(ValueError, match=r"^predict\(\): 'X_new' must have 2 columns like the fit's 'X', not 3$"):
        r.predict(np.zeros((2, 3)))


def test_nan_policy_and_result():
    X = S.points("blobs/k3p2").X.copy()
    X[[4, 50]] = np.nan
    with pytest.raises(ValueError, match=r"^affinity_propagation\(\): the input contains NaN"):
        CL.affinity_propagation(X)
    r = CL.affinity_propagation(X, nan_policy="omit")
    k = CL.affinity_propagation(np.delete(X, [4, 50], axis=0))
    lab = _np(r.labels)
    assert lab[4] == lab[50] == -1 and bits_equal(np.delete(lab, [4, 50]), _np(k.labels))
    kept = np.delete(np.arange(len(X)), [4, 50])
    assert bits_equal(_np(r.exemplars), kept[_np(k.exemplars)])
    assert r.nobs == len(X) - 2 and int(np.count_nonzero(_np(r.omitted))) == 2
    assert repr(r).startswith("AffinityPropagationResult(nobs=88, n_clusters=")
    text = r.summary()
    assert "Affinity Propagation Results" in text and "Exemplar" in text and "Omitted rows (NaN):" in text
    assert r.affinity == "euclidean" and r.damping == 0.5 and isinstance(r.preference, float)
    # labels in first-seen order; exemplars[j] the exemplar of label j
    labs = _np(k.labels)
    assert labs[0] == 0 and np.all(labs[_np(k.exemplars)] == np.arange(k.n_clusters))
    with pytest.raises(AttributeError):
        r.labels = None


def test_errors():
    X = S.points("blobs/k3p2").X
    with pytest.raises(ValueError, match=r"^affinity_propagation\(\): 'damping' must be in \[0.5, 1\), not 1.0$"):
        CL.affinity_propagation(X, damping=1.0)
    with pytest.raises(ValueError, match=r"^affinity_propagation\(\): 'max_iter' must be >= 1, not 0$"):
        CL.affinity_propagation(X, max_iter=0)
    with pytest.raises(
        ValueError, match=r"^affinity_propagation\(\): 'affinity' must be 'euclidean' or 'precomputed', not 'cosine'$"
    ):
        CL.affinity_propagation(X, affinity="cosine")
    with pytest.raises(
        ValueError, match=r"^affinity_propagation\(\): with affinity='precomputed', 'X' must be a square"
    ):
        CL.affinity_propagation(X, affinity="precomputed")
    with pytest.raises(
        ValueError,
        match=r"^affinity_propagation\(\): an array 'preference' must have one value per point \(90\), not 2$",
    ):
        CL.affinity_propagation(X, preference=[1.0, 2.0])
    with pytest.raises(TypeError, match=r"^affinity_propagation\(\): unexpected keyword argument 'metric'$"):
        CL.affinity_propagation(X, metric="precomputed")
    with pytest.raises(TypeError, match=r"^affinity_propagation\(\): unexpected keyword argument 'random_state'$"):
        CL.affinity_propagation(X, random_state=0)
