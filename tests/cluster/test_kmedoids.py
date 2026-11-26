"""k-medoids (src/stardust/cluster/_kmedoids.py, csrc/cluster/kmedoids.cpp; docs/api/cluster.md §3;
docs/design/cluster.md §2.3, §3.4, §4.3; decisions CL12, CL26, CL41, CL59-CL66): PAM against the exact BUILD and SWAP
of tools/cluster_ref.py and R's pam, CLARA against R's clara with one sample of every point, the distances' paths,
the thread counts, the input order and the errors."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S

CL = sd.cluster
REF = D.ref("kmedoids")
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _group(name, sid):
    m = REF.methods[name]
    at = np.nonzero(m.index == D.group_of(REF, sid))[0]
    return (m.hi[at[0]] + m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}) if len(at) else None


def _sets():
    out = []
    for sid in REF.group_inputs["set"].tolist():
        pid, k = sid.split("|")
        out.append((sid, pid, int(k[1:])))
    return out


def test_pam_against_the_exact_build_and_swap():
    # §4.3: the objectives within the contract; the medoids those of the exact PAM unless equal objectives tie
    # (1-D medians, lattices: the computed sums decide), and never above a lower objective
    same = 0
    for sid, pid, k in _sets():
        s = S.points(pid)
        r = CL.kmedoids(s.X, k)
        swap, _ = _group("pam_swap", sid)
        build, _ = _group("pam_build", sid)
        assert abs(r.objective - swap) <= D.pam_bound(s.n, s.p, swap) + 1e-300, sid
        # BUILD at an exact tie (the two middle points of an even 1-D cluster) follows the computed sums
        assert r.build_objective >= r.objective
        pos, _ = D.at(REF, "pam", sid, 0)
        meds = sorted(REF.methods["pam"].hi[pos].astype(int).tolist())
        same += sorted(_np(r.medoids).tolist()) == meds
        opt = _group("optimum_objective", sid)
        if opt is not None:
            assert r.objective >= opt[0] * (1 - 1e-14), sid
        assert r.converged
    assert same >= 50  # measured 53 of 63; the others tie exactly (equal objectives)


def test_r_pam_and_clara():
    rec = D.data("r")
    agree = 0
    for sid, pid, k in _sets():
        v = rec.get(f"pam/{sid}")
        if v is None or "value" not in v or "subnormal" in pid or sid == "lattice/4x4-shuffled|k4":
            continue
        s = S.points(pid)
        r = CL.kmedoids(s.X, k)
        vals = v["value"]
        if sorted(_np(r.medoids).tolist()) == sorted(int(x) for x in vals[:k]):
            assert r.objective == pytest.approx(vals[k + 1], rel=1e-13), sid
            agree += 1
    assert agree >= 35  # measured 38: R's choices at the ties differ
    for sid in ("blobs/k3p2", "probe/300", "sizes/k3p2", "overlap/k3p2"):
        vals = rec[f"clara_pamlike/{sid}"]["value"]
        X = S.points(sid).X
        c = CL.kmedoids(X, 3, method="clara", rng=sd.random.default_rng(0), sample_size=len(X), samples=1)
        p = CL.kmedoids(X, 3)
        assert sorted(_np(c.medoids).tolist()) == sorted(_np(p.medoids).tolist()) == [int(x) for x in vals[:3]]
        assert c.objective == p.objective == pytest.approx(vals[3], rel=1e-14)


def test_distance_paths_give_the_same_bits():
    g = np.random.default_rng(2)
    X = np.vstack([g.normal(c, 1.0, (60, 3)) for c in (0.0, 4.0, 8.0)])
    order = _np(sd._core.cluster.lex_order(A(X)))
    Xs = X[order]  # in key order the condensed matrix and the observations number the points alike
    for metric in ("euclidean", "cityblock"):
        a = CL.kmedoids(Xs, 4, metric=metric)
        b = CL.kmedoids(sd.spatial.pdist(A(Xs), metric=metric), 4, metric="precomputed")
        c = CL.kmedoids(sd.spatial.squareform(sd.spatial.pdist(A(Xs), metric=metric)), 4, metric="precomputed")
        assert bits_equal(a.medoids, b.medoids) and a.objective == b.objective == c.objective, metric
        assert bits_equal(a.labels, b.labels)
    a = CL.kmedoids(X, 4, metric="chebyshev")  # through sd.spatial.pdist
    assert a.metric == "chebyshev" and a.n_clusters == 4
    big = g.standard_normal((6000, 2))  # past the cached matrix's size: distances when needed
    r1 = CL.kmedoids(big[:3000], 3)
    assert r1.converged


def test_thread_counts_and_input_order():
    g = np.random.default_rng(4)
    X = np.vstack([g.normal(c, 1.0, (500, 2)) for c in (0.0, 5.0, 10.0)])

    def results():
        r = CL.kmedoids(X, 5)
        c = CL.kmedoids(X, 5, method="clara", rng=sd.random.default_rng(1), samples=7)
        return [r.labels, r.medoids, r.objective, c.labels, c.medoids, c.objective]

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
    a = CL.kmedoids(X, 5)
    b = CL.kmedoids(X[perm], 5)
    assert sorted(perm[_np(b.medoids)].tolist()) == sorted(_np(a.medoids).tolist()) and a.objective == b.objective
    c1 = CL.kmedoids(X, 5, method="clara", rng=sd.random.default_rng(1))
    c2 = CL.kmedoids(X[perm], 5, method="clara", rng=sd.random.default_rng(1))
    assert sorted(perm[_np(c2.medoids)].tolist()) == sorted(_np(c1.medoids).tolist())


def test_result_predict_and_nan_policy():
    s = S.points("probe/300")
    r = CL.kmedoids(s.X, 3)
    assert np.array_equal(_np(r.predict(s.X)), _np(r.labels))
    assert bits_equal(r.medoid_points, A(s.X[_np(r.medoids)]))
    assert _np(r.labels)[_np(r.medoids)].tolist() == [0, 1, 2]
    assert "K-Medoids Results" in r.summary() and repr(r).startswith("KMedoidsResult(method='pam', nobs=300")
    X = s.X.copy()
    X[[5, 9], 0] = np.nan
    ro = CL.kmedoids(X, 3, nan_policy="omit")
    assert _np(ro.labels)[[5, 9]].tolist() == [-1, -1]
    assert not np.isnan(X[_np(ro.medoids)]).any()
    rp = CL.kmedoids(sd.spatial.pdist(A(s.X)), 3, metric="precomputed")
    assert rp.medoid_points is None
    with pytest.raises(ValueError, match=r"^predict\(\): a fit on precomputed distances"):
        rp.predict(s.X)
    m0 = CL.kmedoids(s.X, 3, max_iter=0)
    assert m0.n_swaps == 0 and m0.objective == m0.build_objective


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.kmedoids(np.eye(4), 2, rng=sd.random.default_rng()), ValueError,
         r"^kmedoids\(\): 'rng' is only used with method='clara'"),
        (lambda: CL.kmedoids(np.eye(4), 2, method="clara"), TypeError, r"^kmedoids\(\): 'rng' must be an sd.random"),
        (lambda: CL.kmedoids(np.eye(4), 2, method="fasterpam"), ValueError, r"^kmedoids\(\): 'method' must be"),
        (lambda: CL.kmedoids(np.eye(4), 5), ValueError, r"'k' \(5\) is larger than the number of points \(4\)"),
        (lambda: CL.kmedoids(np.eye(4), 2, max_iter=-1), ValueError, r"'max_iter' must be >= 0"),
        (lambda: CL.kmedoids(np.eye(8), 2, method="clara", rng=sd.random.default_rng(), sample_size=2), ValueError,
         r"'sample_size' must be between k \+ 1 \(3\) and n \(8\)"),
        (lambda: CL.kmedoids(np.eye(4), 2, metric="manhattan"), ValueError, r"'metric' must be .* not 'manhattan'"),
        (lambda: CL.kmedoids(np.ones((3, 3)), 2, metric="precomputed"), ValueError, r"zero diagonal"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()
