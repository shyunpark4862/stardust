"""The clustering indices (src/stardust/cluster/_metrics.py, csrc/cluster/metrics.cpp; docs/api/cluster.md §6;
docs/design/cluster.md §2.6, §3.8, §4.6; decisions CL13, CL14, CL36, CL82-CL86): the exact indices against the
exact rationals of tools/cluster_ref.py bit for bit, the others within the contracts of §4.6, scikit-learn 1.9.1's
values and special cases, R cluster's silhouette, the label types, the paths, the thread counts, the input order
and the errors."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S

CL = sd.cluster
REF = D.ref("metrics")
CORE = sd._core.cluster
A = sd.asarray
AVERAGES = ("min", "geometric", "arithmetic", "max")


def _np(a):
    return np.asarray(a)


def _group(name, sid):
    m = REF.methods[name]
    at = np.nonzero(m.index == D.group_of(REF, sid))[0]
    return (m.hi[at[0]], m.lo[at[0]], {k: v[at[0]] for k, v in m.fields.items()}) if len(at) else None


def _norm(avg, ha, hb):
    return {"min": min(ha, hb), "max": max(ha, hb), "geometric": math.sqrt(ha * hb), "arithmetic": (ha + hb) / 2}[avg]


def _internal_sets():
    sets = set(REF.group_inputs["set"].tolist())
    for sid in S.point_ids():
        s = S.points(sid)
        if sid in sets and s.truth is not None and 2 <= len(set(s.truth.tolist())) <= s.n - 1:
            yield sid, s


# ---- external indices ----------------------------------------------------------------------------------------------


def test_exact_indices_are_the_exact_rationals_rounded_once():
    # CL36: ARI, RI and FM correctly rounded (the reference's hi is the nearest double of the exact value)
    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        for name, f in (("ari", CL.adjusted_rand_score), ("ri", CL.rand_score), ("fm", CL.fowlkes_mallows_score)):
            hi, _, _ = _group(name, lid)
            assert f(lb.a, lb.b) == hi, (lid, name)


def test_pair_counts_and_contingency_are_exact():
    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        M, a, b, n = D.contingency(lb.a, lb.b)
        assert np.array_equal(_np(CL.contingency_matrix(lb.a, lb.b)), M), lid
        sp = CL.contingency_matrix(lb.a, lb.b, sparse=True)
        assert isinstance(sp, sd.sparse.csr_array) and np.array_equal(_np(sp.toarray()), M)
        tp = int((M * (M - 1) // 2).sum())
        sa, sb = int((a * (a - 1) // 2).sum()), int((b * (b - 1) // 2).sum())
        tn = n * (n - 1) // 2 - tp - (sa - tp) - (sb - tp)
        want = [[2 * tn, 2 * (sb - tp)], [2 * (sa - tp), 2 * tp]]
        assert _np(CL.pair_confusion_matrix(lb.a, lb.b)).tolist() == want, lid


def test_information_indices_within_the_contract():
    # §4.6: MI and the entropies within gamma(ceil(log2 m) + 16) of their absolute terms, E[MI] within u E+, and
    # NMI, AMI, homogeneity, completeness and V by the propagation of those
    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        M, ra, rb, n = D.contingency(lb.a, lb.b)
        codes_a = A(np.unique(lb.a, return_inverse=True)[1].astype(np.int64))
        codes_b = A(np.unique(lb.b, return_inverse=True)[1].astype(np.int64))
        _, _, _, mi, ha, hb, emi = CORE.external(codes_a, codes_b, M.shape[0], M.shape[1], True)
        mi_ref, ha_ref, hb_ref = (_group(k, lid) for k in ("mi", "h_true", "h_pred"))
        dmi = D.mi_bound(M, n)
        dha, dhb = D.entropy_bound(ra, ha_ref[0]), D.entropy_bound(rb, hb_ref[0])
        assert abs(mi - (mi_ref[0] + mi_ref[1])) <= dmi + 1e-300, lid
        assert abs(ha - (ha_ref[0] + ha_ref[1])) <= dha + 1e-300, lid
        assert abs(hb - (hb_ref[0] + hb_ref[1])) <= dhb + 1e-300, lid
        e_ref = _group("emi", lid)
        if e_ref is not None:
            de = D.emi_bound(e_ref[2]["plus"])
            assert abs(emi - (e_ref[0] + e_ref[1])) <= de, lid
            assert abs(emi - (e_ref[0] + e_ref[1])) <= 0.1 * de, lid  # measured below 0.04 of the bound
        for avg in AVERAGES:
            nmi = CL.normalized_mutual_info_score(lb.a, lb.b, average_method=avg)
            ref = _group(f"nmi_{avg}", lid)
            if ref[2]["special"]:
                assert nmi == ref[0], (lid, avg)
            else:
                N = _norm(avg, ha_ref[0], hb_ref[0])
                assert abs(nmi - ref[0]) <= D.ratio_bound(mi_ref[0], dmi, N, dha + dhb + 2 * D.U * N), (lid, avg)
            ref = _group(f"ami_{avg}", lid)
            if ref is None:
                continue
            ami = CL.adjusted_mutual_info_score(lb.a, lb.b, average_method=avg)
            if ref[2]["special"]:
                assert ami == ref[0], (lid, avg)
                continue
            de = D.emi_bound(e_ref[2]["plus"])
            N = _norm(avg, ha_ref[0], hb_ref[0])
            bound = D.ratio_bound(mi_ref[0] - e_ref[0], dmi + de, N - e_ref[0], dha + dhb + de + 2 * D.U * N)
            assert abs(ami - ref[0]) <= bound, (lid, avg, ami, ref[0], bound)
        for name, f, num, den in (
            ("homogeneity", CL.homogeneity_score, dmi, dha),
            ("completeness", CL.completeness_score, dmi, dhb),
        ):
            ref = _group(name, lid)
            h = ha_ref[0] if name == "homogeneity" else hb_ref[0]
            got = f(lb.a, lb.b)
            assert 0.0 <= got <= 1.0
            assert abs(got - ref[0]) <= (D.ratio_bound(mi_ref[0], num, h, den) if h else 0.0), (lid, name)
        ref = _group("v", lid)
        got = CL.v_measure_score(lb.a, lb.b)
        s_ = ha_ref[0] + hb_ref[0]
        assert abs(got - ref[0]) <= (D.ratio_bound(2 * mi_ref[0], 2 * dmi, s_, dha + dhb + D.U * s_) if s_ else 0.0)


def test_scikit_learns_values_and_special_cases():
    import sklearn.metrics as M

    pairs = [("adjusted_rand_score", 1e-15), ("rand_score", 1e-15), ("fowlkes_mallows_score", 1e-15),
             ("mutual_info_score", 1e-13), ("homogeneity_score", 1e-13), ("completeness_score", 1e-13),
             ("v_measure_score", 1e-13)]  # fmt: skip
    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        for name, tol in pairs:
            assert getattr(CL, name)(lb.a, lb.b) == pytest.approx(getattr(M, name)(lb.a, lb.b), rel=tol, abs=1e-15), (
                lid,
                name,
            )
        for avg in AVERAGES:
            got = CL.normalized_mutual_info_score(lb.a, lb.b, average_method=avg)
            assert got == pytest.approx(M.normalized_mutual_info_score(lb.a, lb.b, average_method=avg), rel=1e-13)
            if lb.n <= 10**5:
                got = CL.adjusted_mutual_info_score(lb.a, lb.b, average_method=avg)
                want = M.adjusted_mutual_info_score(lb.a, lb.b, average_method=avg)
                assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (lid, avg)
    # the design's special cases (§2.6)
    one, uniq = [0] * 6, list(range(6))
    assert CL.adjusted_rand_score(one, one) == CL.adjusted_rand_score(uniq, uniq) == 1.0
    assert CL.adjusted_mutual_info_score(one, one) == 1.0
    assert CL.normalized_mutual_info_score(one, uniq) == 0.0
    # V with beta = 1 is NMI with the arithmetic mean, bit for bit; scikit-learn's homogeneity can exceed 1
    for lid in S.labeling_ids():
        lb = S.labeling(lid)
        assert CL.v_measure_score(lb.a, lb.b) == CL.normalized_mutual_info_score(lb.a, lb.b), lid
    lb = S.labeling("refine/120")
    assert M.homogeneity_score(lb.a, lb.b) > 1.0 and 1.0 - 4 * D.U <= CL.homogeneity_score(lb.a, lb.b) <= 1.0
    h = CL.homogeneity_completeness_v_measure(lb.a, lb.b, beta=2.0)
    assert isinstance(h, CL.HomogeneityResult) and h._fields == ("homogeneity", "completeness", "v_measure")
    assert h.homogeneity == CL.homogeneity_score(lb.a, lb.b)
    assert h.v_measure == CL.v_measure_score(lb.a, lb.b, beta=2.0)
    assert h.v_measure == pytest.approx(M.v_measure_score(lb.a, lb.b, beta=2.0), rel=1e-13)
    assert CL.v_measure_score(lb.a, lb.b, beta=0.0) == CL.homogeneity_score(lb.a, lb.b)


def test_ami_when_one_side_is_all_singletons():
    # CL84: every relabeling of an all-singletons side gives the same MI, so MI = E[MI] = the other side's entropy
    # exactly: AMI is 0, and 1 with 'min' (its normalizer is that entropy too; a refinement's AMI_min is 1).
    # scikit-learn reaches the same by its rounding: 1.0 for 'min', noise around 0 for the others.
    import sklearn.metrics as M

    a, b = list(range(8)), [0, 0, 1, 1, 2, 2, 3, 3]
    for avg in AVERAGES:
        want = 1.0 if avg == "min" else 0.0
        assert CL.adjusted_mutual_info_score(a, b, average_method=avg) == want
        assert CL.adjusted_mutual_info_score(b, a, average_method=avg) == want
        assert M.adjusted_mutual_info_score(a, b, average_method=avg) == pytest.approx(want, abs=1e-12)
    assert CL.adjusted_mutual_info_score(a, a) == 1.0
    # a refinement that is not all singletons: AMI_min is 1 up to rounding
    a2, b2 = [0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 2, 2, 3, 3]
    assert CL.adjusted_mutual_info_score(a2, b2, average_method="min") == pytest.approx(1.0, abs=4e-16)


def test_large_labelings():
    import sklearn.metrics as M

    lb = S.labeling("large/1e6")
    for name in ("adjusted_rand_score", "rand_score", "fowlkes_mallows_score", "v_measure_score"):
        assert getattr(CL, name)(lb.a, lb.b) == pytest.approx(getattr(M, name)(lb.a, lb.b), rel=1e-12), name
    # AMI's expectation by the ratio recurrence takes milliseconds at 10^6 points (scikit-learn's lgamma loop takes
    # seconds); the same value as the subsample's scale-free check: it lies between 0 and NMI
    ami = CL.adjusted_mutual_info_score(lb.a, lb.b)
    assert 0.0 < ami < CL.normalized_mutual_info_score(lb.a, lb.b)
    assert ami == pytest.approx(M.adjusted_mutual_info_score(lb.a[:100000], lb.b[:100000]), abs=5e-3)


def test_label_types():
    a, b = [0, 0, 1, 1, 2], [1, 1, 0, 0, 0]
    want = CL.adjusted_rand_score(a, b)
    names = np.array(["x", "x", "y", "y", "z"])
    assert CL.adjusted_rand_score(names, b) == want
    assert CL.adjusted_rand_score(names.tolist(), b) == want
    assert CL.adjusted_rand_score(sd.frame.Series(names.tolist()), b) == want
    assert CL.adjusted_rand_score(A(np.array(a, float)), A(np.array(b))) == want
    assert CL.adjusted_rand_score(a, [True, True, False, False, False]) == want
    assert CL.adjusted_rand_score(np.array([7, 7, -3, -3, 100]), b) == want
    # the contingency matrix's rows and columns: the labels in sorted order
    assert _np(CL.contingency_matrix(["b", "a", "b"], [2, 2, 1])).tolist() == [[0, 1], [1, 1]]
    with pytest.raises(ValueError, match=r"^adjusted_rand_score\(\): float labels in 'labels_true' must be whole"):
        CL.adjusted_rand_score([0.5, 0.5, 1.0], [0, 1, 1])
    with pytest.raises(ValueError, match=r"^rand_score\(\): 'labels_true' and 'labels_pred' must have the same length"):
        CL.rand_score([0, 0, 1], [0, 1])
    with pytest.raises(ValueError, match=r"^mutual_info_score\(\): the labelings must have at least 1 point"):
        CL.mutual_info_score([], [])
    with pytest.raises(ValueError, match=r"^v_measure_score\(\): 'labels_pred' must be 1-D, not 2-D"):
        CL.v_measure_score([0, 1], [[0], [1]])
    with pytest.raises(TypeError, match=r"^fowlkes_mallows_score\(\): 'labels_true' must be a 1-D array of labels"):
        CL.fowlkes_mallows_score(3, [0])
    with pytest.raises(ValueError, match=r"^normalized_mutual_info_score\(\): 'average_method' must be 'min'"):
        CL.normalized_mutual_info_score(a, b, average_method="mean")
    with pytest.raises(ValueError, match=r"^v_measure_score\(\): 'beta' must be finite and >= 0"):
        CL.v_measure_score(a, b, beta=-1.0)
    with pytest.raises(TypeError, match=r"^contingency_matrix\(\): 'sparse' must be a bool"):
        CL.contingency_matrix(a, b, sparse=1)


# ---- internal indices ----------------------------------------------------------------------------------------------


def test_silhouettes_within_the_contract():
    for sid, s in _internal_sets():
        pos, idx = D.at(REF, "silhouette", sid, 0)
        m = REF.methods["silhouette"]
        o = np.argsort(idx)
        ref = m.hi[pos[o]] + m.lo[pos[o]]
        got = _np(CL.silhouette_samples(s.X, s.truth))
        n_max = int(np.max(np.unique(s.truth, return_counts=True)[1]))
        bound = D.silhouette_bound(n_max, s.p)
        assert np.all(np.abs(got - ref) <= bound), (sid, float(np.max(np.abs(got - ref))), bound)
        assert np.max(np.abs(got - ref)) <= 32 * D.U, sid  # measured within 2u
        hi, lo, _ = _group("silhouette_score", sid)
        assert abs(CL.silhouette_score(s.X, s.truth) - (hi + lo)) <= bound, sid


def test_calinski_harabasz_and_davies_bouldin_within_the_contract():
    for sid, s in _internal_sets():
        for name, f, bound in (
            ("calinski_harabasz", CL.calinski_harabasz_score, D.ch_bound),
            ("davies_bouldin", CL.davies_bouldin_score, D.db_bound),
        ):
            hi, lo, _ = _group(name, sid)
            got = f(s.X, s.truth)
            assert abs(got - (hi + lo)) <= bound(s.X, s.truth, hi), (sid, name, got, hi)


def test_internal_indices_against_scikit_learn_and_r():
    import sklearn.metrics as M

    rec = D.data("r")
    for sid, s in _internal_sets():
        got = _np(CL.silhouette_samples(s.X, s.truth))
        if sid not in ("shifted/k3p2", "subnormal/k3p2"):  # scikit-learn's expanded distances lose these (§2.8)
            assert np.allclose(got, M.silhouette_samples(s.X, s.truth), rtol=0, atol=1e-11), sid
            assert CL.calinski_harabasz_score(s.X, s.truth) == pytest.approx(
                M.calinski_harabasz_score(s.X, s.truth), rel=1e-10
            )
            assert CL.davies_bouldin_score(s.X, s.truth) == pytest.approx(
                M.davies_bouldin_score(s.X, s.truth), rel=1e-10
            )
        r = rec.get(f"silhouette/{sid}")
        if r is not None and "value" in r and sid != "subnormal/k3p2":  # R's dist underflows there (all 0)
            assert np.allclose(got, r["value"], rtol=0, atol=1e-13), sid
    Xs = np.array([[0.0], [0.1], [5.0], [9.0], [9.2]])
    got = _np(CL.silhouette_samples(Xs, [0, 0, 1, 2, 2]))
    assert got[2] == 0.0 and np.allclose(got, rec["cluster/silhouette_singleton"]["value"], rtol=0, atol=1e-15)


def test_silhouette_metrics_and_paths():
    s = S.points("blobs/k3p2")
    X, lab = s.X, s.truth
    order = _np(CORE.lex_order(A(X)))
    Xk, labk = X[order], lab[order]
    for metric in ("euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "cosine"):
        got = _np(CL.silhouette_samples(X, lab, metric=metric))
        # the matrix-free path and the precomputed path on the key-ordered points: the same distances in the same
        # order, so the same bits
        D_ = sd.spatial.pdist(A(np.ascontiguousarray(Xk)), metric=metric)
        pre = _np(CL.silhouette_samples(D_, labk, metric="precomputed"))
        assert bits_equal(got[order], pre), metric
        sq = sd.spatial.squareform(D_)
        assert bits_equal(_np(CL.silhouette_samples(sq, labk, metric="precomputed")), pre)
        import sklearn.metrics as M

        assert np.allclose(got, M.silhouette_samples(X, lab, metric=metric), atol=1e-12), metric


def test_threads_and_input_order():
    rng = np.random.default_rng(5)
    X = np.vstack([rng.normal(c, 1.0, (700, 6)) for c in (0.0, 3.0, 6.0)])
    lab = np.repeat([0, 1, 2], 700)
    out = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        out.append((_np(CL.silhouette_samples(X, lab)), CL.silhouette_score(X, lab), CL.calinski_harabasz_score(X, lab),
                    CL.davies_bouldin_score(X, lab)))  # fmt: skip
    sd.set_num_threads(10)
    for o in out[1:]:
        assert bits_equal(o[0], out[0][0]) and o[1:] == out[0][1:]
    perm = rng.permutation(len(X))
    assert bits_equal(_np(CL.silhouette_samples(X[perm], lab[perm])), out[0][0][perm])
    assert CL.silhouette_score(X[perm], lab[perm]) == out[0][1]
    assert CL.calinski_harabasz_score(X[perm], lab[perm]) == out[0][2]
    # relabeling the clusters does not change the values
    assert bits_equal(_np(CL.silhouette_samples(X, (lab + 1) % 3)), out[0][0])
    # one thread takes each pair once (CL98): the same bits as the scan of every pair, for every metric and for
    # clusters of sizes 1-3 (the chains a cluster does not fill) and of uneven sizes
    Y = rng.standard_normal((211, 5)) * 1e3
    labs = rng.integers(0, 7, 211)
    labs[:3] = [7, 8, 8]  # a cluster of one and one of two
    for metric in ("euclidean", "sqeuclidean", "cityblock", "chebyshev"):
        got = []
        for t in (1, 3):
            sd.set_num_threads(t)
            got.append(_np(CL.silhouette_samples(Y, labs, metric=metric)))
        sd.set_num_threads(10)
        assert bits_equal(got[0], got[1]), metric
    a = rng.integers(0, 30, 200000)
    b = np.where(rng.random(200000) < 0.7, a, rng.integers(0, 40, 200000))
    vals = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        vals.append([CL.adjusted_mutual_info_score(a, b), CL.mutual_info_score(a, b), CL.adjusted_rand_score(a, b)])
    sd.set_num_threads(10)
    assert vals[0] == vals[1] == vals[2]


def test_scale_and_far_from_the_origin():
    # CH and DB on the points less the first (translation-free) and CL43's scale: shifted and subnormal sets keep
    # their digits; the silhouette is the same under exact scaling
    s = S.points("blobs/k3p2")
    X, lab = s.X, s.truth
    for f in (CL.silhouette_score, CL.calinski_harabasz_score, CL.davies_bouldin_score):
        v = f(X, lab)
        assert f(X * 2.0**-1000, lab) == v and f(X * 2.0**900, lab) == v, f.__name__
        assert f(X + 1e8, lab) == pytest.approx(v, rel=1e-6), f.__name__


def test_internal_errors():
    X = np.array([[0.0], [1.0], [5.0], [6.0]])
    with pytest.raises(
        ValueError, match=r"^silhouette_score\(\): the number of clusters must be between 2 and n - 1 \(3\), not 1$"
    ):
        CL.silhouette_score(X, [0, 0, 0, 0])
    with pytest.raises(ValueError, match=r"between 2 and n - 1 \(3\), not 4"):
        CL.calinski_harabasz_score(X, [0, 1, 2, 3])
    with pytest.raises(
        ValueError, match=r"^davies_bouldin_score\(\): 'labels' must have one label per point \(4\), not 3$"
    ):
        CL.davies_bouldin_score(X, [0, 1, 1])
    with pytest.raises(ValueError, match=r"^silhouette_samples\(\): 'X' contains NaN$"):
        CL.silhouette_samples(np.array([[0.0], [np.nan], [5.0]]), [0, 0, 1])
    with pytest.raises(ValueError, match=r"^silhouette_samples\(\): 'metric' must be 'euclidean', .* not 'foo'"):
        CL.silhouette_samples(X, [0, 0, 1, 1], metric="foo")
    with pytest.raises(ValueError, match=r"^silhouette_samples\(\): the distance matrix 'X' must be symmetric$"):
        CL.silhouette_samples(
            np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.5, 1.0, 0.0]]), [0, 0, 1], metric="precomputed"
        )
    with pytest.raises(TypeError, match=r"^calinski_harabasz_score\(\): unexpected keyword argument 'metric'$"):
        CL.calinski_harabasz_score(X, [0, 0, 1, 1], metric="euclidean")
