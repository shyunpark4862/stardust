"""The reference and comparison data of sd.cluster (docs/design/cluster.md §4, §7): every file belongs to the test
data as they are (digests), the references pass checks of their own (valid trees, monotone heights where the
linkage is, single linkage as the minimum spanning tree, Lloyd's fixed point, PAM no worse than its BUILD and no
better than the optimum, DBSCAN's core points, responsibilities summing to 1, the indices' identities), the recorded
scipy 1.18.1, scikit-learn 1.9.1 and R 4.6.2 results agree with the references where they compute the same thing
and show the behaviours the design departs from (§2.8), and milestone CL0's prototype measurements stay within the
contract constants of refdata.py. The recorded files are compared, not the installed libraries, so the tests do
not change with them."""

import math

import numpy as np
import pytest

from . import refdata as D
from . import samples as S

REFS = {f: D.ref(f) for f in D.FAMILIES}
SCIPY = D.data("scipy")
SK = D.data("sklearn")
R = D.data("r")
PROTO = D.data("prototype")


def value(rec):
    v = rec["value"]
    if isinstance(v, dict) and "array" in v:
        return np.array(v["array"])
    return v


def obj(sid):
    return S.points(sid.split("|")[0])


def per_set(family, method, sid, kind=None):
    """(index, hi, lo, fields) of a set's points with that method, ordered by the points' index."""
    r = REFS[family]
    pos, idx = D.at(r, method, sid, kind)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def group_value(family, method, sid):
    r = REFS[family]
    m = r.methods[method]
    g = D.group_of(r, sid)
    at = np.nonzero(m.index == g)[0]
    if not len(at):
        return None
    return m.hi[at[0]], m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}


# ---- the files belong to the data -----------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert D.names() == sorted(D.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/cluster_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f
        assert all(not len(m.dropped) for m in r.methods.values()), f
    for name, rec in (("scipy", SCIPY), ("sklearn", SK), ("r", R), ("prototype", PROTO)):
        assert rec["_provenance"]["tool"] == "tools/cluster_probe.py", name
    assert SCIPY["_provenance"]["scipy"] == "1.18.1"
    assert SK["_provenance"]["scikit-learn"] == "1.9.1"
    assert R["_provenance"]["R"] == "R version 4.6.2"
    assert value(R["_packages"]) == [4.0, 2.0]


def test_points_are_the_generators_points():
    for f, r in REFS.items():
        for sid, dig in set(zip(r.inputs["set"].tolist(), r.inputs["digest"].tolist())):
            base = sid.split("|")[0]
            if f == "mixture":
                assert S.mixture(base).digest == dig, sid
            elif base in S.labeling_ids():
                assert S.labeling(base).digest == dig, sid
            else:
                assert S.points(base).digest == dig, sid


def test_the_key_is_lexicographic_and_order_free():
    # CL7: the rank of a point is its place under (coordinates, index); shuffled points keep their ranks' order
    a, b = S.points("lattice/4x4"), S.points("lattice/4x4-shuffled")
    ra, rb = S.lex_rank(a.X), S.lex_rank(b.X)
    assert sorted(ra.tolist()) == list(range(16))
    for i in range(16):
        j = int(np.nonzero((a.X == b.X[i]).all(axis=1))[0][0])
        assert rb[i] == ra[j]
    assert S.first_seen([5, 5, -1, 2, 5, 2]).tolist() == [0, 0, -1, 1, 0, 1]


# ---- hierarchy --------------------------------------------------------------------------------


def hier_sets():
    return REFS["hierarchy"].group_inputs["set"].tolist()


def test_reference_trees_are_valid_and_monotone_where_the_linkage_is():
    for sid in hier_sets():
        n = obj(sid).n
        for m in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
            Z, gaps, _ = D.tree(REFS["hierarchy"], m, sid)
            assert Z.shape == (n - 1, 4), (sid, m)
            size = {i: 1 for i in range(n)}
            used = set()
            for i, (a, b, h, c) in enumerate(Z):
                a, b = int(a), int(b)
                assert a < b < n + i and a not in used and b not in used, (sid, m, i)
                used |= {a, b}
                size[n + i] = size[a] + size[b]
                assert c == size[n + i] and h >= 0
            assert len(D.merges(Z, n)) == n - 1
            if m not in ("centroid", "median"):
                assert np.all(np.diff(Z[:, 2]) >= 0), (sid, m)


def _mst_weights(X):
    """Prim's tree on the float64 distances (a different algorithm from the reference's)."""
    n = len(X)
    best = np.full(n, np.inf)
    inside = np.zeros(n, bool)
    cur, out = 0, []
    inside[0] = True
    for _ in range(n - 1):
        best = np.minimum(best, np.sqrt(np.sum((X - X[cur]) ** 2, axis=1)))
        best[inside] = np.inf
        cur = int(np.argmin(best))
        out.append(best[cur])
        inside[cur] = True
    return np.sort(out)


def test_single_linkage_is_the_minimum_spanning_tree():
    for sid in hier_sets():
        s = obj(sid)
        if "subnormal" in s.tags:
            continue  # float64 squares of subnormal coordinates underflow
        Z, _, lo = D.tree(REFS["hierarchy"], "single", sid)
        assert np.allclose(_mst_weights(s.X), Z[:, 2], rtol=4e-16, atol=0), sid


def test_inversions_and_the_order_free_tree():
    Z, _, _ = D.tree(REFS["hierarchy"], "centroid", "inversion/3")
    assert Z[:, 2].tolist() == [2.0, pytest.approx(1.8, abs=1e-15)]  # the design's triangle: 2.0 then 1.8
    # the primitive tree with the CL7 key is the same tree for the lattice in any order (scipy's is not)
    a, b = S.points("lattice/4x4"), S.points("lattice/4x4-shuffled")
    perm = [int(np.nonzero((a.X == row).all(axis=1))[0][0]) for row in b.X]
    for m in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
        Za, _, _ = D.tree(REFS["hierarchy"], m, "lattice/4x4")
        Zb, _, _ = D.tree(REFS["hierarchy"], m, "lattice/4x4-shuffled")
        mapped = {frozenset(perm[i] for i in c) for c in D.merges(Zb, 16)}
        assert mapped == D.merges(Za, 16), m
        assert np.array_equal(np.sort(Za[:, 2]), np.sort(Zb[:, 2])), m
    assert not all(
        SCIPY[f"rule/linkage_input_order/{m}"]["same_tree"] for m in ("single", "complete", "average", "ward")
    )


def test_scipy_and_r_give_the_reference_trees_without_ties():
    """Where no step has a near tie, scipy's tree is the primitive tree and its heights are within 1e-13; R's hclust
    (ward.D2, mcquitty; centroid and median on squared distances, square roots taken) gives the same."""
    checked = 0
    for sid in hier_sets():
        s = obj(sid)
        for m in ("single", "complete", "average", "weighted", "centroid", "median", "ward"):
            Z, gaps, lo = D.tree(REFS["hierarchy"], m, sid)
            if np.min(gaps) <= 1e-12:
                continue
            Zs = value(SCIPY[f"linkage/{sid}/{m}"])
            if "subnormal" in s.tags:  # scipy's (and R's) distances of subnormal points underflow to 0
                assert np.max(Zs[:, 2]) < np.min(Z[:, 2]) / 2, (sid, m)
                continue
            assert D.merges(Zs, s.n) == D.merges(Z, s.n), (sid, m)
            assert np.allclose(np.sort(Zs[:, 2]), np.sort(Z[:, 2] + lo), rtol=1e-13, atol=0), (sid, m)
            if sid == "huge/k3p2" and m in ("centroid", "median", "ward"):
                continue  # R's squared distances near 1e300 overflow in its updates: another tree
            rv = np.array(value(R[f"hclust/{sid}/{m}"]))
            k = s.n - 1
            merge = rv[: 2 * k].reshape(2, k).T.astype(int)
            heights = rv[2 * k : 3 * k]
            members, out = {}, set()
            for i, (x, y) in enumerate(merge):
                mx = frozenset([-x - 1]) if x < 0 else members[x - 1]
                my = frozenset([-y - 1]) if y < 0 else members[y - 1]
                members[i] = mx | my
                out.add(members[i])
            assert out == D.merges(Z, s.n), (sid, m)
            assert np.allclose(np.sort(heights), np.sort(Z[:, 2] + lo), rtol=1e-13, atol=1e-300), (sid, m)
            checked += 1
    assert checked > 60


def test_scipy_cuts_the_design_departs_from():
    rec = SCIPY["rule/fcluster_maxclust_ties"]
    assert len(set(rec["fcluster_maxclust_2"])) == 1 and len(set(rec["cut_tree_2"])) == 2  # "t or fewer" at a tie
    assert value(R["rule/cutree_ties_k2"]) == [1.0] * 6 + [2.0]
    inv = SCIPY["rule/inversion_cuts"]
    assert inv["is_monotonic"] is False
    assert len(set(inv["fcluster_maxclust_3"])) == 2 and inv["cut_tree_3"] == [0, 0, 0, 2]
    assert len(set(inv["fcluster_distance_1.9"])) == 4 and inv["cut_tree_height_1.9"] == [0, 0, 0, 2]
    assert value(R["rule/cutree_inversion_k3"]) == [1.0, 1.0, 2.0, 3.0]
    assert "not sorted" in R["rule/cutree_inversion_h"]["error"]
    cuts = SCIPY["rule/cut_tree_vs_maxclust"]["differ"]
    assert set(cuts) == {"centroid", "median"} and min(cuts.values()) > 50
    assert "value" in SCIPY["rule/precomputed_ward_cityblock"]  # any condensed distance is taken as Euclidean
    assert "error" in SCIPY["rule/observations_noneuclidean_ward"]
    assert any("ClusterWarning" in w for w in SCIPY["rule/linkage_square_input"]["warnings"])


def test_r_linkage_names_and_scales():
    ward = np.array(value(R["rule/ward_D_vs_D2"]))
    k = len(ward) // 2
    Z, _, lo = D.tree(REFS["hierarchy"], "ward", "inversion/random30")
    assert np.allclose(np.sort(ward[k:]), np.sort(Z[:, 2] + lo), rtol=1e-13)  # ward = R's ward.D2
    assert np.max(np.abs(np.sort(ward[:k]) - np.sort(ward[k:]))) > 0.1  # and not ward.D
    Zc, _, loc = D.tree(REFS["hierarchy"], "centroid", "inversion/random30")
    plain = np.sort(np.array(value(R["rule/centroid_plain_distances"])))
    assert np.max(np.abs(plain - np.sort(Zc[:, 2] + loc))) > 0.1  # R's centroid wants squared distances


# ---- k-means ------------------------------------------------------------------------------------


def km_sets():
    r = REFS["kmeans"]
    return [s for s in r.group_inputs["set"].tolist() if group_value("kmeans", "lloyd_inertia", s) is not None]


def test_lloyd_reference_is_a_fixed_point():
    for sid in km_sets():
        s = obj(sid)
        _, lab, _, f = per_set("kmeans", "lloyd_label", sid, 0)
        _, labf, _, ff = per_set("kmeans", "label_f", sid, 0)
        far = ff["gap"] > D.NEAR_TIE
        assert np.array_equal(lab[far], labf[far]), sid
        _, c, clo, _ = per_set("kmeans", "lloyd_center", sid, 1)
        k = int(lab.max()) + 1
        C = (c + clo).reshape(-1, s.p)
        w = np.ones(s.n) if s.w is None else s.w
        for j in range(k):
            sel = (lab == j) & (w > 0)
            if sel.any():
                assert np.allclose(C[j], np.average(s.X[sel], axis=0, weights=w[sel]), rtol=1e-13, atol=1e-300), (
                    sid,
                    j,
                )
        inertia = group_value("kmeans", "lloyd_inertia", sid)
        got = float(np.sum(w * np.sum((s.X - C[lab.astype(int)]) ** 2, axis=1)))
        assert got == pytest.approx(inertia[0], rel=1e-12, abs=1e-300), sid


def test_sklearn_reaches_the_same_fixed_point_from_the_same_start():
    agree = 0
    for sid in km_sets():
        rec = SK.get(f"kmeans/{sid}")
        if rec is None or "value" not in rec:
            continue
        v = rec["value"]
        _, lab, _, f = per_set("kmeans", "lloyd_label", sid, 0)
        _, c, clo, _ = per_set("kmeans", "lloyd_center", sid, 1)
        fields = group_value("kmeans", "lloyd_inertia", sid)[2]
        relocated = fields["relocated"]
        if "subnormal" in sid:  # scikit-learn's expanded squared distances are all 0 there: one cluster
            assert len(set(v["labels"]["array"])) == 1 and len(set(lab.tolist())) == 3
            continue
        if relocated or fields["path_gap"] <= D.NEAR_TIE:
            continue  # an emptied cluster, or a step with a (near) tie: scikit-learn's rounding may choose otherwise
        assert np.array_equal(np.array(v["labels"]["array"]), lab), sid
        s = obj(sid)
        bound = D.center_bound(s.n, float(np.max(np.abs(s.X))))
        err = np.max(np.abs(np.array(v["centers"]["array"]).ravel() - (c + clo)))
        assert err <= 64 * bound, (sid, err / bound)
        agree += 1
    assert agree > 20


def test_kmeans_behaviours_the_design_departs_from():
    assert SK["rule/kmeans_n_init_auto"] == {"k-means++": 1, "random": 10}
    assert SK["rule/kmeans_tie"] == {"predict": [0, 0], "predict_swapped": [0, 0]}
    assert SCIPY["rule/vq_tie"] == {"vq": [0, 0], "vq_swapped": [0, 0]}
    few = SK["rule/kmeans_fewdistinct"]
    assert len(set(value(few).tolist())) == 2 and "ConvergenceWarning" in few["warnings"][0]
    assert "more cluster centers than distinct data points" in R["rule/kmeans_fewdistinct"]["error"]
    assert value(SK["rule/kmeans_empty"]).tolist() == [0, 0, 1, 2]  # the far point takes the empty cluster
    assert "empty cluster" in R["rule/kmeans_empty_lloyd"]["warnings"] and "error" in R["rule/kmeans_empty_hw"]
    assert len(SCIPY["rule/kmeans2_empty"]["warnings"]) >= 1
    assert SK["rule/kmeans_nan"]["error"] == "ValueError"
    th = SK["rule/kmeans_threads"]
    assert th["labels_equal"] and not th["centers_bit_equal"] and 0 < th["centers_max_diff"] < 1e-13
    assert SK["rule/kmeans_lloyd_elkan"]["labels_equal"]
    hw, lloyd = value(R["rule/kmeans_hw_vs_lloyd"])
    _, lab, _, _ = per_set("kmeans", "lloyd_label", "probe/300", 0)
    assert lloyd == pytest.approx(group_value("kmeans", "lloyd_inertia", "probe/300")[0], rel=1e-12)
    assert hw <= lloyd  # Hartigan-Wong moves single points: here a lower local minimum


# ---- k-medoids ------------------------------------------------------------------------------------


def test_pam_references_and_r():
    r = REFS["kmedoids"]
    same = 0
    for sid in r.group_inputs["set"].tolist():
        build = group_value("kmedoids", "pam_build", sid)[0]
        swap = group_value("kmedoids", "pam_swap", sid)[0]
        assert swap <= build * (1 + 1e-15)
        opt = group_value("kmedoids", "optimum_objective", sid)
        if opt is not None:
            assert opt[0] <= swap * (1 + 1e-15)
        k = int(r.group_inputs["k"][D.group_of(r, sid)])
        rv = value(R[f"pam/{sid}"])
        if "subnormal" in sid:
            assert rv[k] == rv[k + 1] == 0.0  # R's distances of subnormal points underflow to 0
            continue
        if sid == "lattice/4x4-shuffled|k4":
            assert rv[k + 1] < swap  # R's input order reaches a lower local minimum on the shuffled lattice
            continue
        assert rv[k] == pytest.approx(build, rel=1e-12) and rv[k + 1] == pytest.approx(swap, rel=1e-12), sid
        _, meds, _, _ = per_set("kmedoids", "pam", sid, 0)
        same += sorted(int(v) for v in rv[:k]) == sorted(meds.astype(int).tolist())
    assert same > 20  # the medoids differ only where equal objectives tie (1-D medians, lattices)


# ---- density --------------------------------------------------------------------------------------


def test_dbscan_reference_and_sklearn():
    r = REFS["density"]
    for sid in r.group_inputs["set"].tolist():
        if "|eps" not in sid:
            continue
        _, lab, _, f = per_set("density", "label", sid, 0)
        core = f["core"] == 1
        assert np.all(lab[core] >= 0)
        assert S.first_seen(lab).tolist() == lab.astype(int).tolist()
        sk = value(SK[f"dbscan/{sid}"])
        assert np.array_equal(sk < 0, lab < 0), sid  # noise is noise in both
        sel = core | (lab < 0)  # border points may join another cluster
        assert len(set(zip(sk[sel].tolist(), lab[sel].tolist()))) == len(set(sk[sel].tolist())), sid
    assert SK["rule/dbscan_border"] == {"left_first": "left", "right_first": "right"}
    assert SK["rule/dbscan_eps_inclusive"]["array"] == [0, 0]


def test_hdbscan_pieces():
    r = REFS["density"]
    for sid in r.group_inputs["set"].tolist():
        if "|hdbscan" not in sid:
            continue
        s = obj(sid)
        _, core, _, _ = per_set("density", "core_distance", sid, 0)
        idx, w, _, f = per_set("density", "mst", sid, 1)
        assert len(w) == s.n - 1 and np.all(np.diff(w) >= 0)
        a, b = f["a"].astype(int), f["b"].astype(int)
        d = np.linalg.norm(s.X[a] - s.X[b], axis=1)
        assert np.allclose(w, np.maximum(np.maximum(core[a], core[b]), d), rtol=1e-15, atol=0), sid
    assert "FutureWarning" in SK["rule/hdbscan_copy_warning"]["warnings"][0]
    assert value(SK["rule/hdbscan_duplicates"]) == [0, 1, 2]


# ---- mixtures -------------------------------------------------------------------------------------


def test_mixture_references_and_sklearn():
    for mid in S.mixture_ids():
        mx = S.mixture(mid)
        _, score, _, _ = per_set("mixture", "score", mid, 0)
        resp = np.column_stack([per_set("mixture", f"resp_{j}", mid, 0)[1] for j in range(mx.k)])
        assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-15)
        rec = SK[f"gmm_given/{mid}"]["value"]
        got = np.array(rec["score_samples"]["array"])
        assert np.allclose(got, score, rtol=1e-12, atol=0), mid
        assert np.allclose(np.array(rec["predict_proba"]["array"]), resp, atol=1e-12), mid
    gm = SK["rule/gmm_lower_bound_vs_score"]
    assert gm["converged"] and 0 < abs(gm["lower_bound"] - gm["score"]) < 1e-4  # two names (CL33)


# ---- indices --------------------------------------------------------------------------------------


def test_external_indices_and_sklearn():
    keys = ("ari", "ri", "fm", "mi", "homogeneity", "completeness", "v") + tuple(
        f"{p}_{a}" for p in ("nmi", "ami") for a in ("min", "geometric", "arithmetic", "max")
    )
    for lid in S.labeling_ids():
        rec = SK[f"metrics/{lid}"]
        for k in keys:
            ref = group_value("metrics", k, lid)
            if ref is None:
                assert k.startswith("ami") and S.labeling(lid).n > 10**5
                continue
            got = rec[k]["value"] if k in rec else None
            if got is None:
                continue
            tol = 1e-12 if not k.startswith("ami") else 1e-9
            assert got == pytest.approx(ref[0], rel=tol, abs=1e-15), (lid, k, got, ref[0])
    assert group_value("metrics", "nmi_arithmetic", "small/6")[0] == group_value("metrics", "v", "small/6")[0]


def test_internal_indices_and_sklearn():
    for sid in S.point_ids():
        s = S.points(sid)
        if s.truth is None or s.n > 300:
            continue
        rec = SK[f"internal/{sid}"]
        _, sil, _, _ = per_set("metrics", "silhouette", sid, 0)
        if sid in ("shifted/k3p2", "subnormal/k3p2"):  # scikit-learn's expanded distances lose these digits
            got = value(rec["silhouette_samples"])
            assert not np.all(np.abs(got - sil) <= 1e-3), sid
            continue
        assert np.allclose(value(rec["silhouette_samples"]), sil, rtol=0, atol=1e-11), sid
        for k in ("silhouette_score", "calinski_harabasz", "davies_bouldin"):
            assert value(rec[k]) == pytest.approx(group_value("metrics", k, sid)[0], rel=1e-10), (sid, k)
    sing = np.array(SK["rule/silhouette_singleton"]["array"])
    assert sing[2] == 0.0 and np.allclose(sing, value(R["cluster/silhouette_singleton"]), atol=1e-15)
    assert all("Number of labels is 1" in v["msg"] for v in SK["rule/internal_one_cluster"].values())


def test_r_cluster_records():
    assert value(R["cluster/agnes_ac"])[1:] == pytest.approx(value(R["cluster/hclust_average_heights"]), rel=1e-14)
    gap = value(R["cluster/clusgap"])
    assert gap[-1] == 3  # clusGap's firstSEmax picks the three blobs
    assert "asymm" in R["cluster/daisy_gower"]["warnings"]


# ---- the prototypes of milestone CL0 ---------------------------------------------------------------


def test_distance_forms():
    t = PROTO["distance/time"]
    assert all(v["tile_equals_scalar"] and v["filtered_equals_diff"] for v in t.values())
    acc = PROTO["distance/labels_at_reference_centres"]
    assert acc["diff"] == acc["filtered"]  # the filter keeps the difference form's labels
    for e, row in PROTO["distance/shifted_blobs"].items():
        assert row["diff"] == row["filtered"] == 0, e  # exact labels far from the origin
    assert PROTO["distance/shifted_blobs"]["1e8"]["expanded"] > 100  # the expanded form's cancellation
    assert PROTO["distance/shifted_blobs"]["1e8"]["sklearn_predict"] > 100


def test_bounds_keep_lloyds_bits():
    for c, v in PROTO["bounds/reference_sets"].items():
        if int(c) >= 1:
            assert v["equal"] == v["sets"], c
    for v in PROTO["bounds/time"].values():
        assert v["equal"]
    for v in PROTO["bounds/near_ties"].values():
        assert v["c1"]["equal"]


def test_spanning_trees_agree():
    assert all(PROTO["mst/same_tree"].values())


def test_chain_ties_are_detected():
    t = PROTO["nnchain/against_primitive"]["totals"]
    assert t["differ_unflagged"] == 0  # every tree the chain gets wrong met an exact tie in a scan
    assert t["cases"] == t["same"] + t["differ_flagged"]


def test_heights_and_centres_within_the_draft_constants():
    # CL45: the recurrence on the final tree is within 2 ULP everywhere; the definition's sums (the design draft)
    # lose the ward heights of shifted points (the centroids' cancellation)
    for m, v in PROTO["heights/tree_recurrence"].items():
        assert v["tree_ulp"] <= 2.0 * D.C_HEIGHT, m  # an ULP is at most 2u relative; p + d >= 2
    h = PROTO["heights"]
    assert h["ward"]["recomputed_ulp"] > 1e6 and h["ward"]["lance_williams_ulp"] < 4
    assert PROTO["centers/chunk_sums"]["worst"] <= D.C_CENTER


def test_a_far_point_in_every_mixture():
    for mid in S.mixture_ids():
        assert S.mixture(mid).X.shape[0] == 2 * S.mixture(mid).k + 3


@pytest.mark.parametrize("bad", [math.nan])
def test_nan_is_not_in_the_data(bad):
    for sid in S.point_ids():
        assert not np.isnan(S.points(sid).X).any(), sid
