"""Hierarchical clustering (src/stardust/cluster/_hierarchy.py, csrc/cluster/hierarchy.cpp, mst.cpp; docs/api/cluster.md
§2; docs/design/cluster.md §2.2, §3.2, §4.1; decisions CL4, CL12, CL15-CL22, CL43, CL45-CL47, CL50-CL57): the trees
of the exact primitive definition (tools/cluster_ref.py), the heights' contract, the same bits on every path and
every thread count, the input order, the scipy tools against scipy 1.18.1, DIANA against R's cluster 2.1.8.3, and
the errors."""

import itertools
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S
from .conftest import LIVE

CL = sd.cluster
CORE = sd._core.cluster
METHODS = ("single", "complete", "average", "weighted", "centroid", "median", "ward")
CODE = {m: i for i, m in enumerate(METHODS)}
REF = D.ref("hierarchy")
SETS = REF.group_inputs["set"].tolist()
A = sd.asarray


def _np(a):
    return np.asarray(a)


def _keyed(X):
    order = CORE.lex_order(A(X))
    return order, A(np.ascontiguousarray(X[_np(order)]))


def _core(X, method, path=0, matrix=False):
    """linkage's core on one path: (Z, flags)."""
    order, Xr = _keyed(X)
    if method in ("complete", "average", "weighted") or matrix:
        Dm = sd.spatial.pdist(Xr)
        Xa = Xr if (method in D.SQUARED or path == 1) else None
        return CORE.linkage(Xa, Dm, order, CODE[method], path)
    return CORE.linkage(Xr, None, order, CODE[method], path)


def _no_near_tie(sid, method):
    _, gaps, _ = D.tree(REF, method, sid)
    return len(gaps) == 0 or float(np.min(gaps)) > D.NEAR_TIE


# ---- the definition and the contract ----------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_trees_are_the_primitive_definition(method):
    # CL17: rows (the two clusters and the size, in merge order) of the exact primitive tree with the key of CL7
    differ = []
    for sid in SETS:
        s = S.points(sid)
        Zr, gaps, _ = D.tree(REF, method, sid)
        Z = _np(CL.linkage(s.X, method=method))
        if not np.array_equal(Z[:, [0, 1, 3]], Zr[:, [0, 1, 3]]):
            differ.append(sid)
            assert float(np.min(gaps)) <= D.NEAR_TIE, (sid, method)  # only where the reference met a near tie
    assert len(differ) <= 2, differ  # measured: none (the ties of the lattices included)


@pytest.mark.parametrize("method", METHODS)
def test_heights_within_the_contract(method):
    # §4.1 (CL55): single and complete the computed distance; the recurrence's proved bound
    worst = 0.0
    for sid in SETS:
        s = S.points(sid)
        Zr, gaps, lo = D.tree(REF, method, sid)
        Z = _np(CL.linkage(s.X, method=method))
        if not np.array_equal(Z[:, [0, 1, 3]], Zr[:, [0, 1, 3]]):
            continue
        h_ref = Zr[:, 2] + lo
        bound = D.height_bound(Zr, s.X, method, S.lex_rank(s.X), h_ref)
        err = np.array([abs(Fraction(float(a)) - Fraction(float(b)) - Fraction(float(c))) for a, b, c in
                        zip(Z[:, 2], Zr[:, 2], lo)], dtype=object)  # fmt: skip
        for e, b in zip(err, bound):
            assert e <= Fraction(float(b)), (sid, method, float(e), b)
            if b > 0:
                worst = max(worst, float(e) / b)
    assert worst <= 1.0


def test_contract_helpers_on_a_cancelling_ward_tree():
    # V+ is at least the height; mu grows with the depth; ward's cancellation shows in V+ / H*
    s = S.points("shifted/k3p2")
    Z = _np(CL.linkage(s.X, method="ward"))
    keys = S.lex_rank(s.X)
    vp = D.positive_values(Z, s.X, "ward", keys)
    assert np.all(vp >= Z[:, 2] ** 2 * (1 - 1e-12))
    mu = D.recurrence_chain(Z, s.n, keys)
    assert mu[-1] >= max(mu[:-1]) and mu[0] == 0


# ---- paths, threads and the input order -------------------------------------------------------------------------


def _blob_data():
    g = np.random.default_rng(20261126)
    return [g.standard_normal((300, 3)), g.standard_normal((257, 7)) * 10.0 + 3.0, S.points("probe/300").X]


@pytest.mark.parametrize("method", METHODS)
def test_every_path_gives_the_same_bits(method):
    # CL18, CL45-CL47, CL96: matrix and matrix-free, the chain (square and condensed copies) and the generic
    # algorithm, Prim and Boruvka, and the precomputed distances of sd.spatial (single and complete: the heights are
    # those distances)
    for X in _blob_data():
        Z, _ = _core(X, method)
        paths = [_core(X, method, 3)]
        if method in D.SQUARED:
            paths.append(_core(X, method, 1, matrix=True))
        if method == "single":
            paths += [_core(X, method, 4), _core(X, method, 5), _core(X, method, matrix=True)]
        if method in ("complete", "average", "weighted", "ward"):  # the chain's square and condensed copies (CL96)
            paths += [_core(X, method, matrix=True), _core(X, method, 6, matrix=True)]
        if method == "single":  # sd.spatial's distances given: the same tree and heights
            Zp = _np(CL.linkage(sd.spatial.pdist(A(X))))
            assert D.merges(Zp, len(X)) == D.merges(_np(Z), len(X))
        for Zq, _ in paths:
            assert bits_equal(Zq, Z), method


@pytest.mark.parametrize("method", METHODS)
def test_ties_flag_the_chain_and_follow_the_key(method):
    # CL46: an exact tie in a scan sends the input to the generic algorithm, whose tree is the definition's
    s = S.points("lattice/4x4")
    Z, flags = _core(s.X, method)
    if method in ("complete", "average", "weighted", "ward"):
        assert flags == 1
    Zg, _ = _core(s.X, method, 3)
    assert bits_equal(Z, Zg)


@pytest.mark.usefixtures("restore_threads")
def test_thread_counts_give_the_same_bytes():
    g = np.random.default_rng(5)
    X = g.standard_normal((3000, 4))
    Xi = np.round(3.0 * g.standard_normal((1500, 2)))  # many exact ties
    big = g.standard_normal((5000, 3))

    def results():
        out = []
        for data in (X, Xi):
            for m in METHODS:
                out.append(CL.linkage(data, method=m))
        out.append(CL.linkage(big, method="single"))
        out.append(CL.divisive(X[:400]).linkage)
        Z = CL.linkage(X[:300], method="average")
        out.append(CL.cophenet(Z))
        out.append(CL.optimal_leaf_ordering(Z, sd.spatial.pdist(A(X[:300]))))
        return out

    sd.set_num_threads(1)
    base = results()
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        for k, (a, b) in enumerate(zip(base, results())):
            assert bits_equal(a, b), (t, k)


@pytest.mark.parametrize("method", METHODS)
def test_the_input_order_does_not_change_the_tree(method):
    # CL7: the same points in another order give the same Z up to the points' numbers, bit for bit
    g = np.random.default_rng(9)
    for X in (S.points("blobs/k4p3").X, S.points("lattice/4x4").X, g.standard_normal((200, 2))):
        n = len(X)
        perm = g.permutation(n)
        Z = _np(CL.linkage(X, method=method))
        Zp = _np(CL.linkage(X[perm], method=method))
        ids = Zp[:, :2].astype(int)
        mapped = np.where(ids < n, perm[np.minimum(ids, n - 1)], ids)
        Zp[:, :2] = np.sort(mapped, axis=1)
        assert np.array_equal(Zp, Z), method
    # repeated points are interchangeable: the clusters as multisets of coordinates and the heights are the same
    X = S.points("dups/4x5").X
    perm = g.permutation(len(X))

    def clusters(Z, X):
        return sorted(tuple(sorted(map(tuple, X[sorted(c)].tolist()))) for c in D.merges(Z, len(X)))

    Z, Zp = _np(CL.linkage(X, method=method)), _np(CL.linkage(X[perm], method=method))
    assert clusters(Z, X) == clusters(Zp, X[perm]) and np.array_equal(Z[:, 2:], Zp[:, 2:])


@pytest.mark.parametrize("method", ["ward", "centroid", "median", "single", "average"])
def test_powers_of_two_scale_the_heights_exactly(method):
    # CL43: the squared methods scale far data by a power of two; the tree is the same and the heights scale
    X = S.points("blobs/k3p2").X
    Z = _np(CL.linkage(X, method=method))
    for e in (-1000, -600, 300, 900):  # the data and the heights stay normal
        Ze = _np(CL.linkage(np.ldexp(X, e), method=method))
        assert np.array_equal(Ze[:, [0, 1, 3]], Z[:, [0, 1, 3]])
        if method in D.SQUARED:
            assert np.array_equal(Ze[:, 2], np.ldexp(Z[:, 2], e)), e


def test_matrix_free_paths_take_large_n():
    g = np.random.default_rng(3)
    X = g.standard_normal((40000, 2))
    Z = _np(CL.linkage(X, method="single"))  # 6.4 GB as a matrix
    assert Z.shape == (39999, 4) and CL.is_monotonic(Z)
    Zw = _np(CL.linkage(X[:6000], method="ward"))
    assert CL.is_valid_linkage(Zw)


# ---- against scipy and R -------------------------------------------------------------------------------------


@LIVE
@pytest.mark.parametrize("method", METHODS)
def test_scipy_gives_the_same_trees_without_ties(method):
    for sid in SETS:
        if not _no_near_tie(sid, method) or "subnormal" in sid or "huge" in sid:
            continue
        s = S.points(sid)
        Z = _np(CL.linkage(s.X, method=method))
        Zs = sch.linkage(s.X, method)
        assert np.array_equal(Z[:, [0, 1, 3]], Zs[:, [0, 1, 3]]), (sid, method)
        h = np.maximum(np.abs(Zs[:, 2]), 2.0**-1074)
        assert np.max(np.abs(Z[:, 2] - Zs[:, 2]) / h) < 64 * D.U * (s.p + 8), (sid, method)


def test_r_names_and_scales():
    # §2.2: ward = R's ward.D2, centroid and median = R on squared distances (rooted), weighted = mcquitty
    s = S.points("blobs/k3p2")
    rec = D.data("r")
    Z = _np(CL.linkage(s.X, method="average"))
    assert np.allclose(np.sort(Z[:, 2]), rec["cluster/hclust_average_heights"]["value"], rtol=4e-15, atol=0)
    r = CL.agglomerative(S.points("blobs/k3p2").X, method="average")
    assert abs(r.coefficient - rec["cluster/agnes_ac"]["value"][0]) < 1e-14


def test_divisive_is_r_diana():
    rec = D.data("r")["cluster/diana"]["value"]
    X = S.points("blobs/k3p2").X
    n = len(X)
    r = CL.divisive(X)
    assert r.method == "diana" and abs(r.coefficient - rec[0]) < 1e-14
    assert np.array_equal(np.sort(_np(r.heights)), np.sort(rec[1:n]))
    assert CL.is_monotonic(r.linkage)


def test_divisive_splits_by_the_mean_distances():
    # the first split of four points on a line: 10 is far from the rest; then 6 is closer to {10} on average
    X = np.array([[0.0], [1.0], [6.0], [10.0]])
    r = CL.divisive(X)
    Z = _np(r.linkage)
    assert Z[-1, 2] == 10.0 and Z[-1, 3] == 4
    assert set(D.merges(Z, 4)) == {frozenset({0, 1}), frozenset({2, 3}), frozenset({0, 1, 2, 3})}
    assert _np(r.cut(k=2)).tolist() == [0, 0, 1, 1]


# ---- the scipy tools ---------------------------------------------------------------------------------------------


def _trees():
    g = np.random.default_rng(11)
    out = []
    for i in range(60):
        n = int(g.integers(2, 22))
        X = g.standard_normal((n, 3))
        out.append((sch.linkage(X, METHODS[i % 7]), X))
    return out


@LIVE
def test_tools_match_scipy():
    for Z, X in _trees():
        n = len(X)
        sZ = A(Z)
        assert np.array_equal(_np(CL.leaves_list(sZ)), sch.leaves_list(Z))
        assert CL.is_monotonic(sZ) == sch.is_monotonic(Z)
        assert CL.num_obs_linkage(sZ) == n and CL.is_valid_linkage(sZ)
        assert np.array_equal(_np(CL.maxdists(sZ)), sch.maxdists(Z))
        assert np.array_equal(_np(CL.cophenet(sZ)), sch.cophenet(Z))
        R = sch.inconsistent(Z)
        Rs = _np(CL.inconsistent(sZ))
        assert np.allclose(Rs[:, [0, 2]], R[:, [0, 2]], rtol=1e-14, atol=0)
        assert np.allclose(Rs[:, 1], R[:, 1], rtol=1e-8, atol=1e-7)  # scipy's one-pass deviation cancels
        assert np.allclose(_np(CL.maxinconsts(sZ, A(R))), sch.maxinconsts(Z, R))
        for i in range(4):
            assert np.array_equal(_np(CL.maxRstat(sZ, A(R), i)), sch.maxRstat(Z, R, i))
        for t in (0.0, 0.5, 1.0, 2.0):
            assert np.array_equal(_np(CL.fcluster(sZ, t)), sch.fcluster(Z, t))
            md = sch.maxdists(Z)
            assert np.array_equal(
                _np(CL.fcluster(sZ, t, criterion="monocrit", monocrit=A(md))),
                sch.fcluster(Z, t, "monocrit", monocrit=md),
            )
        for k in range(1, n + 2):
            T = sch.fcluster(Z, k, "maxclust")
            L, M = CL.leaders(sZ, A(T))
            Ls, Ms = sch.leaders(Z, T)
            assert np.array_equal(_np(L), Ls) and np.array_equal(_np(M), Ms)
            if sch.is_monotonic(Z):
                assert np.array_equal(_np(CL.fcluster(sZ, k, criterion="maxclust")), T)
                assert np.array_equal(_np(CL.cut_tree(sZ, n_clusters=min(k, n))), sch.cut_tree(Z, n_clusters=min(k, n)))
        if n > 2:
            y = ssd.pdist(X)
            c = CL.cophenet_correlation(sZ, A(y))
            assert abs(c.correlation - sch.cophenet(Z, y)[0]) < 1e-13
            assert bits_equal(c.distances, CL.cophenet(sZ))


@LIVE
def test_dendrogram_matches_scipy():
    options = [
        {},
        {"truncate_mode": "lastp", "p": 4},
        {"truncate_mode": "level", "p": 2},
        {"count_sort": True},
        {"distance_sort": "descending"},
        {"color_threshold": 0.0},
        {"count_sort": "descending", "show_leaf_counts": False, "truncate_mode": "lastp", "p": 3},
    ]
    for Z, _ in _trees():
        for kw in options:
            a = CL.dendrogram(A(Z), **kw)
            b = sch.dendrogram(Z, no_plot=True, **kw)
            assert np.array_equal(_np(a.icoord).reshape(-1, 4), np.array(b["icoord"]).reshape(-1, 4))
            assert np.array_equal(_np(a.dcoord).reshape(-1, 4), np.array(b["dcoord"]).reshape(-1, 4))
            assert list(a.ivl) == list(b["ivl"]) and _np(a.leaves).tolist() == b["leaves"]
            names = {}  # one group number per scipy colour, -1 for the links above the threshold
            for x, y in zip(_np(a.color_list).tolist() + _np(a.leaves_color_list).tolist(),
                            b["color_list"] + b["leaves_color_list"]):  # fmt: skip
                assert names.setdefault(x, y) == y
            assert names.get(-1, "C0") == "C0"


def test_dendrogram_labels_and_the_result_method():
    r = CL.agglomerative(S.points("blobs/k2p1").X[:6], method="complete")
    d = r.dendrogram(labels=list("abcdef"))
    assert sorted(d.ivl) == list("abcdef") and d.icoord.shape == (5, 4)
    assert _np(d.color_list).dtype == np.int64


def _flips(Z, n):
    """Every leaf order the tree allows."""

    def rec(v):
        if v < n:
            return [[v]]
        a, b = int(Z[v - n, 0]), int(Z[v - n, 1])
        return [x + y for x, y in itertools.product(rec(a), rec(b))] + [
            y + x for x, y in itertools.product(rec(a), rec(b))
        ]

    return rec(2 * n - 2)


def test_optimal_leaf_ordering_is_optimal():
    # CL53: the smallest sum of neighbouring distances over every order the tree allows (scipy 1.18.1 misses it
    # at times: the n = 4 case below)
    g = np.random.default_rng(3)
    scipy_worse = 0
    for t in range(12):
        n = int(g.integers(3, 8))
        X = g.standard_normal((n, 3))
        y = ssd.pdist(X)
        Dm = ssd.squareform(y)
        Z = sch.linkage(X, METHODS[t % 7])
        Zo = _np(CL.optimal_leaf_ordering(A(Z), A(y)))
        assert D.merges(Zo, n) == D.merges(Z, n)

        def cost(order):
            return sum(Dm[order[i], order[i + 1]] for i in range(n - 1))

        best = min(cost(o) for o in _flips(Z, n))
        assert cost(_np(CL.leaves_list(A(Zo)))) <= best * (1 + 1e-12)
        scipy_worse += cost(sch.leaves_list(sch.optimal_leaf_ordering(Z, y))) > best * (1 + 1e-12)
    assert scipy_worse > 0
    Zl = _np(CL.linkage(S.points("blobs/k3p2").X, method="average", optimal_ordering=True))
    assert CL.is_valid_linkage(Zl)


# ---- cutting ---------------------------------------------------------------------------------------------------


def test_cuts_of_an_inverted_tree():
    # CL19: k cuts by merge order (R's cutree), heights are refused; scipy's cuts disagree with each other
    X = S.points("inversion/4").X
    r = CL.agglomerative(X, method="centroid")
    assert not CL.is_monotonic(r.linkage)
    rec = D.data("r")["rule/cutree_inversion_k3"]["value"]
    assert _np(r.cut(k=3)).tolist() == [int(v) - 1 for v in rec]
    assert len(set(_np(CL.fcluster(r.linkage, 3, criterion="maxclust")).tolist())) == 3
    assert _np(CL.cut_tree(r.linkage, n_clusters=3))[:, 0].tolist() == [0, 0, 1, 2]
    for f in (lambda: r.cut(height=1.9), lambda: CL.cut_tree(r.linkage, height=1.9),
              lambda: CL.fcluster(r.linkage, 1.9, criterion="distance")):  # fmt: skip
        with pytest.raises(ValueError, match="non-monotonic"):
            f()


def test_cuts_at_tied_heights():
    # the merge order (R's cutree): line/7 single at k = 2 is R's two clusters; fcluster's maxclust is scipy's
    # "at most t" (one cluster: every merge is at height 1)
    X = S.points("line/7").X
    r = CL.agglomerative(X)
    rec = D.data("r")["rule/cutree_ties_k2"]["value"]
    assert _np(r.cut(k=2)).tolist() == [int(v) - 1 for v in rec]
    assert _np(CL.fcluster(r.linkage, 2, criterion="maxclust")).tolist() == [1] * 7
    assert _np(r.cut(height=1.0)).tolist() == [0] * 7  # at or below the height (R; scipy's cut_tree: below)
    assert _np(r.cut(height=0.5)).tolist() == list(range(7))


def test_cut_tree_shapes_and_labels():
    Z = CL.linkage(S.points("blobs/k3p2").X, method="ward")
    full = _np(CL.cut_tree(Z))
    assert full.shape == (90, 90) and full[:, 0].tolist() == list(range(90)) and set(full[:, -1]) == {0}
    two = _np(CL.cut_tree(Z, n_clusters=[2, 3]))
    assert two.shape == (90, 2) and two[0, 0] == 0
    assert _np(CL.cut_tree(Z, height=1e9)).shape == (90, 1)


# ---- the result object --------------------------------------------------------------------------------------


def test_hierarchy_result_fields_and_text():
    X = S.points("blobs/k3p2").X
    r = CL.agglomerative(X, method="ward")
    assert bits_equal(r.linkage, CL.linkage(X, method="ward"))
    assert bits_equal(r.heights, _np(r.linkage)[:, 2].copy())
    assert bits_equal(r.order, CL.leaves_list(r.linkage))
    assert r.nobs == 90 and r.method == "ward" and r.metric == "euclidean" and 0 < r.coefficient < 1
    with pytest.raises(AttributeError):
        r.method = "single"
    with pytest.raises(ValueError):
        r.heights[0] = 1.0
    assert repr(r).startswith("HierarchyResult(method='ward', nobs=90") and "\n" not in repr(r)
    text = r.summary()
    assert "agglomerative (ward)" in text and "Coefficient" in text and "clusters" in text
    assert _np(r.cut(k=3)).max() == 2
    assert bits_equal(r.cophenet(), CL.cophenet(r.linkage))
    with pytest.raises(TypeError, match=r"^cut\(\): give exactly one of 'k' and 'height'$"):
        r.cut()
    with pytest.raises(TypeError, match=r"^cut\(\): unexpected keyword argument 'n'$"):
        r.cut(n=2)


def test_precomputed_and_nan_policy():
    X = S.points("blobs/k3p2").X.copy()
    Dm = ssd.squareform(ssd.pdist(X))
    a = CL.agglomerative(Dm, method="average", metric="precomputed")
    b = CL.agglomerative(ssd.pdist(X), method="average", metric="precomputed")
    assert bits_equal(a.linkage, b.linkage) and a.nobs == 90
    X[[4, 7], 1] = np.nan
    r = CL.agglomerative(X, method="complete", nan_policy="omit")
    assert r.nobs == 88 and int(np.sum(_np(r.omitted))) == 2
    lab = _np(r.cut(k=3))
    assert lab[4] == -1 and lab[7] == -1 and lab.shape == (90,)
    with pytest.raises(ValueError, match="NaN"):
        CL.agglomerative(X)
    with pytest.raises(ValueError, match="NaN"):
        CL.linkage(X)
    d = CL.divisive(X, nan_policy="omit")
    assert d.nobs == 88


def test_metrics():
    X = S.points("blobs/k3p2").X
    Z = _np(CL.linkage(X, method="average", metric="cityblock"))
    assert np.allclose(Z[:, 2], sch.linkage(X, "average", "cityblock")[:, 2], rtol=1e-14)
    with pytest.raises(ValueError, match=r"^linkage\(\): method='ward' requires metric='euclidean'$"):
        CL.linkage(X, method="ward", metric="cityblock")
    with pytest.raises(ValueError, match="condensed distances"):
        CL.linkage(ssd.pdist(X), metric="cityblock")
    assert CL.divisive(X, metric="chebyshev").metric == "chebyshev"


@pytest.mark.parametrize(
    ("call", "err", "msg"),
    [
        (lambda: CL.linkage([[1.0, 2.0]]), ValueError, "at least 2 points"),
        (lambda: CL.linkage([1.0, 2.0]), ValueError, "condensed distance vector"),
        (lambda: CL.linkage([[0.0], [np.inf]]), ValueError, "infinity"),
        (lambda: CL.linkage([[0.0], [1.0]], method="Ward"), ValueError, "'method' must be"),
        (lambda: CL.linkage([[0.0], [1.0]], optimal_ordering=1), TypeError, "'optimal_ordering' must be a bool"),
        (lambda: CL.linkage([[0.0], [1.0]], "single"), TypeError, "takes 1 positional argument"),
        (lambda: CL.fcluster([[0.0, 1.0, 1.0, 2.0]], 0, criterion="maxclust"), ValueError, "'t' must be >= 1"),
        (lambda: CL.fcluster([[0.0, 1.0, 1.0, 2.0]], 1, criterion="dist"), ValueError, "'criterion' must be"),
        (lambda: CL.fcluster([[0.0, 1.0, 1.0, 2.0]], 1, criterion="monocrit"), ValueError, "needs 'monocrit'"),
        (lambda: CL.cophenet([[0.0, 1.0, 1.0, 3.0]]), ValueError, "has the count 3.0, not 2"),
        (lambda: CL.cophenet([[0.0, 0.0, 1.0, 2.0]]), ValueError, "a second time"),
        (lambda: CL.leaves_list([[0.0, 2.0, 1.0, 2.0]]), ValueError, "before it is formed"),
        (lambda: CL.maxdists([[0.0, 1.0, -1.0, 2.0]]), ValueError, "negative height"),
        (lambda: CL.maxdists([[0.0, 1.0, np.nan, 2.0]]), ValueError, "not finite"),
        (lambda: CL.maxdists(np.zeros((0, 4))), ValueError, "no row"),
        (lambda: CL.cut_tree([[0.0, 1.0, 1.0, 2.0]], n_clusters=3), ValueError, "between 1 and the number"),
        (lambda: CL.inconsistent([[0.0, 1.0, 1.0, 2.0]], d=-1), ValueError, "'d' must be >= 0"),
        (lambda: CL.dendrogram([[0.0, 1.0, 1.0, 2.0]], truncate_mode="mtica"), ValueError, "'truncate_mode'"),
        (lambda: CL.dendrogram([[0.0, 1.0, 1.0, 2.0]], labels=["a"]), ValueError, "one label per point"),
        (lambda: CL.leaders([[0.0, 1.0, 1.0, 2.0], [2.0, 3.0, 2.0, 3.0]], [1, 2, 1]), ValueError, "not one subtree"),
        (lambda: CL.is_isomorphic([1, 2], [1]), ValueError, "the same length"),
    ],
)
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()


def test_validity_and_small_tools():
    assert not CL.is_valid_linkage([[0.0, 1.0, 1.0, 3.0]])
    assert not CL.is_valid_linkage("Z")
    assert CL.is_valid_linkage([[0.0, 1.0, 1.0, 2.0]])
    assert CL.is_isomorphic([1, 1, 2], ["a", "a", "b"]) and not CL.is_isomorphic([1, 1, 2], [1, 2, 2])
    assert CL.num_obs_linkage([[0.0, 1.0, 0.5, 2.0], [2.0, 3.0, 1.0, 3.0]]) == 3
    # a tree of two points
    Z = CL.linkage([[0.0], [3.0]], method="ward")
    assert _np(Z).tolist() == [[0.0, 1.0, 3.0, 2.0]]
    assert CL.agglomerative([[0.0], [3.0]]).coefficient == 0.0
    assert math.isnan(CL.agglomerative([[1.0], [1.0], [1.0]]).coefficient)  # every height 0
