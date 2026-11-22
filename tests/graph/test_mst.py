"""minimum_spanning_tree (docs/api/graph.md §4, docs/design/graph.md §2.4, §3.4; decision GR17; milestone GR3):
the unique forest of the key (w, min(i, j), max(i, j)) on every test graph and input form (the definition in
definitions.py), the places of the chosen edges, negative weights, the dense rule (no 1e-8 cut), the
errors, and scipy's tree wherever the weights have no ties."""

import math

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS

import stardust as sd

from . import definitions as D
from . import graphs as G
from .conftest import LIVE
from .forms import csr, forms, square_ids

GR = sd.graph


def tree_dict(T):
    """{(i, j): w} of a csr_array's stored entries."""
    T = T.tocsr()
    ip, ix, w = np.asarray(T.indptr), np.asarray(T.indices), np.asarray(T.data)
    return {(i, int(ix[k])): float(w[k]) for i in range(T.shape[0]) for k in range(ip[i], ip[i + 1])}


def finite(g):
    return all(math.isfinite(float(w)) for _, _, w in g.edges())


@pytest.mark.parametrize("gid", square_ids())
def test_the_unique_forest_on_every_test_graph(gid):
    g = G.get(gid)
    for form, A in forms(g).items():
        if not finite(g) and form in ("csr", "csc", "coo", "dok"):
            with pytest.raises(ValueError, match="stored edge weights must be finite"):
                GR.minimum_spanning_tree(A)
            continue
        T = GR.minimum_spanning_tree(A)
        assert T.dtype == sd.float64 and T.shape == g.shape
        edges = g.edges() if form in ("csr", "csc", "coo", "dok") else [e for e in g.edges() if float(e[2]) != 0]
        want = D.spanning_forest(g.n, edges)
        got = tree_dict(T)
        assert got.keys() == want.keys(), (gid, form)
        for k, w in want.items():  # the stored value as it is (-0.0 stays -0.0)
            assert math.copysign(1.0, got[k]) == math.copysign(1.0, w) and got[k] == w, (gid, form, k)
        # canonical CSR
        ip, ix = np.asarray(T.indptr), np.asarray(T.indices)
        for i in range(g.n):
            assert np.all(np.diff(ix[ip[i] : ip[i + 1]]) > 0)


def test_ties_are_decided_by_the_smaller_pair():
    # a 4-cycle of equal weights: in (min, max) order the last pair, (2, 3), is the edge left out
    A = sd.sparse.csr_array(np.array([[0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0.0]]))
    assert sorted(tree_dict(GR.minimum_spanning_tree(A))) == [(0, 1), (0, 3), (1, 2)]
    # the transpose gives the same edges, at the transposed places
    assert sorted(tree_dict(GR.minimum_spanning_tree(A.T.tocsr()))) == [(1, 0), (2, 1), (3, 0)]


def test_places_of_two_way_edges():
    M = np.zeros((4, 4))
    M[0, 1], M[1, 0] = 5.0, 2.0  # the smaller weight's place: (1, 0)
    M[1, 2], M[2, 1] = 3.0, 3.0  # the same weight: the upper triangle
    M[3, 2] = 1.0  # one way
    T = tree_dict(GR.minimum_spanning_tree(M))
    assert T == {(1, 0): 2.0, (1, 2): 3.0, (3, 2): 1.0}
    # 0.0 and -0.0 (stored zeros): -0.0 is the smaller weight and keeps its place
    Z = sd.sparse.coo_array((np.array([0.0, -0.0, 1.0]), (np.array([0, 1, 1]), np.array([1, 0, 2]))), shape=(3, 3))
    T = tree_dict(GR.minimum_spanning_tree(Z))
    assert list(T) == [(1, 0), (1, 2)] and math.copysign(1.0, T[(1, 0)]) == -1.0


def test_negative_weights_self_loops_and_empty():
    M = np.array([[-4.0, -1, 0], [0, 7, -2], [3, 0, 0]])
    assert tree_dict(GR.minimum_spanning_tree(M)) == {(0, 1): -1.0, (1, 2): -2.0}
    for n in (0, 1):
        T = GR.minimum_spanning_tree(np.zeros((n, n)))
        assert T.shape == (n, n) and T.nnz == 0
    # a forest: one tree per component
    A = sd.sparse.csr_array(np.kron(np.eye(3), np.array([[0, 1.0], [1.0, 0]])))
    assert GR.minimum_spanning_tree(A).nnz == 3


def test_errors():
    with pytest.raises(ValueError, match=r"minimum_spanning_tree\(\): stored edge weights must be finite"):
        GR.minimum_spanning_tree(
            sd.sparse.csr_array((np.array([np.nan]), np.array([1]), np.array([0, 1, 1])), shape=(2, 2))
        )
    with pytest.raises(ValueError, match="square"):
        GR.minimum_spanning_tree(np.zeros((2, 3)))
    with pytest.raises(TypeError, match="sd.sparse.csr_array"):
        GR.minimum_spanning_tree(S.eye_array(3))
    with pytest.raises(TypeError, match="unexpected keyword argument 'overwrite'"):
        GR.minimum_spanning_tree(np.eye(2), overwrite=True)


def test_large_random_against_the_definition():
    r = np.random.Generator(np.random.PCG64(77))
    n = 300
    M = np.round(r.uniform(-3, 3, (n, n)) * 2) / 2 * (r.random((n, n)) < 0.05)  # many ties
    g_edges = [(i, j, M[i, j]) for i, j in zip(*np.nonzero(M))]
    assert tree_dict(GR.minimum_spanning_tree(M)) == D.spanning_forest(n, g_edges)


@LIVE
def test_scipy_gives_the_same_tree_without_ties():
    # distinct weights: the minimum spanning forest is unique for scipy too; scipy also keeps the stored places
    for gid in square_ids():
        g = G.get(gid)
        if g.matrix is not None or not finite(g) or g.nnz == 0:
            continue
        w = [float(x) for x in g.data]
        if len(set(w)) != len(w) or any(x == 0 for x in w):
            continue
        A = csr(g)
        a = S.csr_array((g.data.astype(np.float64), g.indices, g.indptr), shape=g.shape)
        mine = tree_dict(GR.minimum_spanning_tree(A))
        theirs = {k: v for k, v in S.dok_array(CS.minimum_spanning_tree(a)).items()}
        theirs = {(int(i), int(j)): float(v) for (i, j), v in theirs.items()}
        assert sorted(mine.values()) == sorted(theirs.values()), gid
        assert {tuple(sorted(k)) for k in mine} == {tuple(sorted(k)) for k in theirs}, gid


@LIVE
def test_scipy_drops_small_dense_entries():
    # [scipy differs]: scipy's dense input drops |w| < 1e-8; sd keeps every entry that is not 0, NaN or inf
    M = np.array([[0, 1e-9, 0], [0, 0, 2.0], [0, 0, 0]])
    assert tree_dict(GR.minimum_spanning_tree(M)) == {(0, 1): 1e-9, (1, 2): 2.0}
    assert CS.minimum_spanning_tree(M).nnz == 1
