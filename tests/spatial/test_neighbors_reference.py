"""kneighbors against the Arb references and the other libraries' records (docs/design/spatial.md §4
neighbours 2 and 3, §7.4; tests/spatial/ref/neighbors.npz, data/scipy.json, data/sklearn.json).

- §4 neighbours 2: on the lattice sets (every difference and sum exact, the square root correctly rounded)
  the distances are the references' RN(d) and the answer is the Arb order (RN(d), index), ties included;
- §4 neighbours 3: on general data the k nearest are the true k nearest within the distance contract:
  every returned point's true distance is at most d_(k)(1 + eta) and every other point's at least
  d_(k)(1 - eta), d_(k) the returned k-th distance;
- scipy 1.18.1's KDTree records (k = 5): the same sets where scipy's own order has no ties;
- sklearn 1.9.1's pairwise_distances_argmin_min records: the same answer on nn/normal3, and the true one on
  far/1e8 where the expanded form found none."""

import numpy as np
import pytest

import stardust as sd

from . import points as P
from . import refdata

S = sd.spatial
REF = refdata.ref("neighbors")
SCIPY = refdata.data("scipy")
SKLEARN = refdata.data("sklearn")
METHODS = (("p2", "euclidean"), ("p1", "cityblock"), ("pinf", "chebyshev"))
K = 9


def _rows(sid):
    return [k for k in range(REF.points) if REF.inputs["set"][k] == sid]


@pytest.mark.parametrize("sid", [s for s in P.ids("neighbors") if "lattice" in P.get(s).tags])
@pytest.mark.parametrize("method,metric", METHODS)
def test_lattice_answer_is_the_arb_order(sid, method, metric):
    s = P.get(sid)
    rows = _rows(sid)
    Q = np.array([REF.inputs["query"][k] for k in rows])
    for path in (1, 2):
        got = S._neighbors._knn("kneighbors", Q, s.x, K, metric, None, None, None, None, np.inf, path)
        d, i = np.asarray(got.distances), np.asarray(got.indices)
        for q, k in enumerate(rows):
            hi = REF.methods[method].hi[k]
            want = np.lexsort((np.arange(hi.size), hi))[:K]
            assert i[q].tolist() == want.tolist(), (sid, q, path)
            assert np.array_equal(d[q], hi[want]), (sid, q, path)


@pytest.mark.parametrize("sid", [s for s in P.ids("neighbors") if "lattice" not in P.get(s).tags])
@pytest.mark.parametrize("method,metric", METHODS)
def test_general_data_is_the_true_knn_within_the_contract(sid, method, metric):
    s = P.get(sid)
    rows = _rows(sid)
    Q = np.array([REF.inputs["query"][k] for k in rows])
    n = s.n
    eta = refdata.bound(metric, 1.0, n=n) - refdata.ETA  # relative (chebyshev: half an ULP of 1)
    eta = max(eta, 2.0**-52)
    for k_ in (1, K):
        d, i = S.kneighbors(Q, s.x, k=k_, metric=metric)
        d, i = np.asarray(d).reshape(len(rows), -1), np.asarray(i).reshape(len(rows), -1)
        for q, k in enumerate(rows):
            true = REF.methods[method].hi[k] + REF.methods[method].lo[k]
            dk = d[q, -1]
            mask = np.zeros(true.size, bool)
            mask[i[q]] = True
            assert np.all(true[mask] <= dk * (1 + eta) + refdata.ETA), (sid, q)
            assert np.all(true[~mask] >= dk * (1 - eta) - refdata.ETA), (sid, q)
            for j, v in zip(i[q], d[q]):  # each distance within its contract
                assert abs(v - true[j]) <= refdata.bound(metric, true[j], n=n), (sid, q, j)


def test_scipy_kdtree_records():
    for sid in P.ids("neighbors"):
        rec = SCIPY[f"knn/{sid}"]
        s = P.get(sid)
        assert rec["digest"] == P.set_digest(s)
        d, i = S.kneighbors(s.queries, s.x, k=rec["k"])
        d, i = np.asarray(d), np.asarray(i)
        for q in range(len(rec["indices"])):
            sd_ = np.asarray(rec["distances"][q])
            # sets agree where scipy's k-th distance is clear of the (k + 1)-th: no tie at the boundary
            D = np.asarray(S.cdist(s.queries[q : q + 1], s.x))[0]
            srt = np.sort(D)
            if srt[rec["k"] - 1] < srt[min(rec["k"], len(srt) - 1)] * (1 - 1e-12):
                assert sorted(i[q].tolist()) == sorted(rec["indices"][q]), (sid, q)
            assert np.allclose(d[q], np.sort(sd_), rtol=1e-14, atol=0), (sid, q)


def test_sklearn_argmin_min_records():
    s = P.get("nn/normal3")
    rec = SKLEARN["argmin_min/nn/normal3"]
    assert rec["digest"] == P.set_digest(s)
    d, i = S.kneighbors(s.queries, s.x)
    assert np.asarray(i).tolist() == rec["indices"]
    assert np.allclose(np.asarray(d), rec["distances"], rtol=1e-12)
    far = P.get("far/1e8")
    rec = SKLEARN["argmin_min/far/1e8"]
    d, i = S.kneighbors(far.queries, far.x)
    rows = _rows("far/1e8")
    true = [int(np.argmin(REF.methods["p2"].hi[k])) for k in rows]
    assert np.asarray(i).tolist() == true  # sklearn's expanded form: none right, every distance 0
    assert sum(a == b for a, b in zip(rec["indices"], true)) == 0
    assert np.all(np.asarray(d) > 0)
    # six copies of one point: sklearn's order [3, 4, 1, 5, 2, 0], sd's (d, j)
    assert SKLEARN["nn/tie_order"]["value"]["array"] == [3, 4, 1, 5, 2, 0]
    assert np.asarray(S.kneighbors(np.zeros((1, 3)), np.zeros((6, 3)), k=6).indices).tolist() == [[0, 1, 2, 3, 4, 5]]
