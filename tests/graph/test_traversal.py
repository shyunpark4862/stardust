"""The traversals of sd.graph (docs/api/graph.md §2, docs/design/graph.md §2.2, §3.3; milestone GR1):
breadth_first_order, depth_first_order, breadth_first_tree and depth_first_tree equal their definitions
(tests/graph/definitions.py) on every square test graph, from every start, directed and undirected, in every
input form; the result types; the undirected reading of G and of its transpose is one answer; scipy 1.18.1
agrees where the design says it does (directed orders and trees, symmetric undirected graphs) and differs
where it says it differs (the undirected neighbour order, zero weights in undirected trees)."""

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS

import stardust as sd

from . import definitions as D
from . import graphs as G
from .conftest import LIVE
from .forms import csr, forms, square_ids

IDS = square_ids()


def starts(g):
    return sorted({0, g.n // 2, g.n - 1}) if g.n else []


def tree_dict(T):
    """{(i, j): w} of a csr_array's stored entries."""
    out = {}
    ip, ix, dt = (np.asarray(a) for a in (T.indptr, T.indices, T.data))
    for i in range(T.shape[0]):
        for k in range(ip[i], ip[i + 1]):
            out[(i, int(ix[k]))] = float(dt[k])
    return out


def same_weights(a: dict, b: dict) -> bool:
    if a.keys() != b.keys():
        return False
    return all(np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() or (a[k] != a[k] and b[k] != b[k]) for k in a)


@pytest.mark.parametrize("gid", IDS)
def test_orders_equal_the_definitions(gid):
    g = G.get(gid)
    edges = g.edges()
    for form, A in forms(g).items():
        for s in starts(g):
            for directed in (True, False):
                for f, ref in ((sd.graph.breadth_first_order, D.bfs), (sd.graph.depth_first_order, D.dfs)):
                    order, pred = f(A, s, directed=directed)
                    want_order, want_pred = ref(g.n, edges, s, directed)
                    assert np.asarray(order).tolist() == want_order, (gid, form, s, directed, f.__name__)
                    assert np.asarray(pred).tolist() == want_pred, (gid, form, s, directed, f.__name__)


@pytest.mark.parametrize("gid", IDS)
def test_trees_equal_the_definitions(gid):
    g = G.get(gid)
    edges = g.edges()
    for form, A in forms(g).items():
        for s in starts(g):
            for directed in (True, False):
                for f, depth in ((sd.graph.breadth_first_tree, False), (sd.graph.depth_first_tree, True)):
                    T = f(A, s, directed=directed)
                    assert isinstance(T, sd.sparse.csr_array) and T.shape == g.shape and T.dtype == sd.float64
                    want = D.tree(g.n, edges, s, directed, depth)
                    assert same_weights(tree_dict(T), want), (gid, form, s, directed, f.__name__)


def test_result_types():
    A = csr(G.get("grid2d/5x5"))
    r = sd.graph.breadth_first_order(A, 3)
    assert type(r).__name__ == "TraversalResult" and r._fields == ("node_array", "predecessors")
    order, pred = r
    for a in r:
        assert isinstance(a, sd.ndarray) and a.dtype == sd.int64 and not a._flags()["writeable"]
    assert order.shape == (25,) and pred.shape == (25,)
    alone = sd.graph.depth_first_order(A, 3, return_predecessors=False)
    assert isinstance(alone, sd.ndarray) and alone.dtype == sd.int64 and alone._flags()["writeable"]
    assert np.asarray(alone).tolist() == np.asarray(sd.graph.depth_first_order(A, 3).node_array).tolist()
    T = sd.graph.breadth_first_tree(A, 0)
    T.data[0] = 7.0  # an ordinary new array
    assert sd.graph.breadth_first_tree(A, 0).data[0] == 1.0


def test_the_start_alone_and_unreached_nodes():
    A = sd.sparse.csr_array((5, 5))
    order, pred = sd.graph.breadth_first_order(A, 2)
    assert np.asarray(order).tolist() == [2] and np.asarray(pred).tolist() == [-9999] * 5
    T = sd.graph.depth_first_tree(A, 2)
    assert T.nnz == 0 and T.shape == (5, 5)
    one = sd.graph.breadth_first_order([[0.0]], 0)
    assert np.asarray(one.node_array).tolist() == [0]


@pytest.mark.parametrize("gid", IDS)
def test_undirected_reading_of_g_and_its_transpose_is_one_answer(gid):
    g = G.get(gid)
    A = csr(g)
    At = A.T.tocsr()
    for s in starts(g):
        for f in (sd.graph.breadth_first_order, sd.graph.depth_first_order):
            a, b = f(A, s, directed=False), f(At, s, directed=False)
            assert np.asarray(a.node_array).tolist() == np.asarray(b.node_array).tolist()
            assert np.asarray(a.predecessors).tolist() == np.asarray(b.predecessors).tolist()
        for f in (sd.graph.breadth_first_tree, sd.graph.depth_first_tree):
            assert same_weights(tree_dict(f(A, s, directed=False)), tree_dict(f(At, s, directed=False)))


def test_undirected_weights_take_the_smaller_direction():
    # 0 -> 1 weight 5, 1 -> 0 weight 2; 1 -> 2 weight 0 stored; 2 -> 3 NaN, 3 -> 2 1; 3 -> 4 -0.0, 4 -> 3 0.0
    A = sd.sparse.coo_array(
        (
            np.array([5.0, 2.0, 0.0, np.nan, 1.0, -0.0, 0.0]),
            (np.array([0, 1, 1, 2, 3, 3, 4]), np.array([1, 0, 2, 3, 2, 4, 3])),
        ),
        shape=(5, 5),
    )
    T = tree_dict(sd.graph.breadth_first_tree(A, 0, directed=False))
    assert T[(0, 1)] == 2.0 and T[(1, 2)] == 0.0 and np.isnan(T[(2, 3)])
    assert T[(3, 4)] == 0.0 and np.signbit(T[(3, 4)])
    assert tree_dict(sd.graph.breadth_first_tree(A, 0))[(0, 1)] == 5.0  # directed: the edge as stored


def test_random_graphs_equal_the_definitions():
    for seed in range(40):
        r = np.random.Generator(np.random.PCG64([20261123, seed]))
        n = int(r.integers(1, 40))
        p = float(r.uniform(0.0, 0.3))
        M = (r.random((n, n)) < p) * r.integers(1, 5, (n, n)).astype(np.float64)
        edges = [(i, j, M[i, j]) for i in range(n) for j in range(n) if M[i, j] != 0]
        s = int(r.integers(0, n))
        for directed in (True, False):
            for f, ref in ((sd.graph.breadth_first_order, D.bfs), (sd.graph.depth_first_order, D.dfs)):
                order, pred = f(M, s, directed=directed)
                assert (np.asarray(order).tolist(), np.asarray(pred).tolist()) == tuple(ref(n, edges, s, directed))


# ---- scipy 1.18.1 ----


@LIVE
@pytest.mark.parametrize("gid", IDS)
def test_scipy_directed_orders_and_trees_are_the_same(gid):
    g = G.get(gid)
    if g.n == 0:
        return
    A, B = csr(g), g.scipy()
    for s in starts(g):
        for name in ("breadth_first_order", "depth_first_order"):
            got, want = getattr(sd.graph, name)(A, s), getattr(CS, name)(B, s)
            assert np.asarray(got.node_array).tolist() == want[0].tolist(), (gid, name, s)
            assert np.asarray(got.predecessors).tolist() == want[1].tolist(), (gid, name, s)
        for name in ("breadth_first_tree", "depth_first_tree"):
            got, want = getattr(sd.graph, name)(A, s), getattr(CS, name)(B, s)
            assert same_weights(tree_dict(got), tree_dict(want)), (gid, name, s)


@LIVE
@pytest.mark.parametrize("gid", [i for i in IDS if "undirected" in G.get(i).tags])
def test_scipy_undirected_orders_are_the_same_on_symmetric_graphs(gid):
    g = G.get(gid)
    A, B = csr(g), g.scipy()
    for s in starts(g):
        for name in ("breadth_first_order", "depth_first_order"):
            got, want = getattr(sd.graph, name)(A, s, directed=False), getattr(CS, name)(B, s, directed=False)
            assert np.asarray(got.node_array).tolist() == want[0].tolist(), (gid, name, s)


@LIVE
def test_scipy_differs_where_documented():
    # the undirected neighbour order (decision GR13): scipy looks at the out-neighbours first
    M = np.zeros((5, 5))
    M[0, 3] = M[0, 4] = M[1, 0] = M[2, 0] = 1.0
    assert CS.breadth_first_order(S.csr_array(M), 0, directed=False)[0].tolist() == [0, 3, 4, 1, 2]
    assert np.asarray(sd.graph.breadth_first_order(M, 0, directed=False).node_array).tolist() == [0, 1, 2, 3, 4]
    # zero weights in an undirected tree (decision GR29): scipy reads a stored 0 as no edge there
    Z = S.csr_array((np.array([0.0, 3.0]), np.array([1, 0]), np.array([0, 1, 2])), shape=(2, 2))
    assert CS.breadth_first_tree(Z, 0, directed=False).data.tolist() == [3.0]
    assert tree_dict(sd.graph.breadth_first_tree(sd.sparse.csr_array(Z), 0, directed=False)) == {(0, 1): 0.0}
    one = S.csr_array((np.array([0.0]), np.array([1]), np.array([0, 1, 1])), shape=(2, 2))
    assert CS.breadth_first_tree(one, 0, directed=False).data.tolist() == [np.inf]
    assert tree_dict(sd.graph.breadth_first_tree(sd.sparse.csr_array(one), 0, directed=False)) == {(0, 1): 0.0}
    # int32 against int64 (GR10)
    assert CS.breadth_first_order(S.csr_array(M), 0)[0].dtype == np.int32
    assert sd.graph.breadth_first_order(M, 0).node_array.dtype == sd.int64
