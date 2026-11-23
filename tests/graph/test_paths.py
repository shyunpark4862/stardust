"""The shortest-path functions' behaviour (docs/api/graph.md §3, docs/design/graph.md §2.3, §2.9; milestone
GR2): the result forms, indices, method, limit, min_only, unweighted, the undirected reading, negative
weights and cycles (the rule and the message), stored NaN and inf, construct_dist_matrix and
reconstruct_path with their errors, the MemoryError before an n^2 output, the call shape, and the
differences from scipy 1.18.1 (its records in tests/graph/data/scipy.json and live where it is the
comparator)."""

import math

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CG

import stardust as sd

from . import graphs as G
from . import refdata
from .conftest import LIVE
from .forms import csr
from .test_reference_data import arr

GR = sd.graph
NONE = refdata.NONE
SCIPY = refdata.data("scipy")
INF = math.inf


def small():
    return csr(G.get("er/float100"))


def scipy_of(gid):
    g = G.get(gid)
    if g.matrix is not None:
        return S.csr_array(CG.csgraph_from_dense(g.matrix))
    return S.csr_array((g.data.astype(np.float64), g.indices, g.indptr), shape=g.shape)


# ---- forms ------------------------------------------------------------------------------------


def test_result_shapes_and_types():
    A = small()
    n = A.shape[0]
    for f in (GR.shortest_path, GR.dijkstra, GR.bellman_ford, GR.johnson):
        assert f(A).shape == (n, n)
        assert f(A, indices=3).shape == (n,)
        assert f(A, indices=[3, 1, 3]).shape == (3, n)
        assert f(A, indices=np.int32([4])).shape == (1, n)
        assert f(A, indices=[]).shape == (0, n)
        d = f(A, indices=2)
        assert isinstance(d, sd.ndarray) and d.dtype == sd.float64 and d._flags()["writeable"]
        r = f(A, indices=[0, 5], return_predecessors=True)
        assert type(r).__name__ == "ShortestPathResult" and r._fields == ("dist_matrix", "predecessors")
        assert r.predecessors.dtype == sd.int64 and r.predecessors.shape == (2, n)
        assert not r.dist_matrix._flags()["writeable"] and not r.predecessors._flags()["writeable"]
        dd, pp = r
        assert np.asarray(dd).tobytes() == np.asarray(f(A, indices=[0, 5])).tobytes()
        # duplicated sources give equal rows
        three = np.asarray(f(A, indices=[3, 1, 3]))
        assert three[0].tobytes() == three[2].tobytes()
    assert GR.floyd_warshall(A).shape == (n, n)
    for f in (GR.shortest_path, GR.dijkstra, GR.floyd_warshall):
        assert f(sd.sparse.csr_array((0, 0))).shape == (0, 0)


def test_dense_and_sparse_forms_agree():
    g = G.get("er/int60")
    A = csr(g)
    D = g.dense()
    for f in (GR.shortest_path, GR.bellman_ford, GR.floyd_warshall):
        want = np.asarray(f(A)).tobytes()
        for x in (D, sd.asarray(D), D.tolist(), A.tocsc(), A.tocoo(), A.todok()):
            assert np.asarray(f(x)).tobytes() == want
    # the dense rule: 0, NaN and inf are no edge; a stored zero is a zero-weight edge
    M = np.array([[0, np.nan, np.inf, 2.0], [0, 0, 1.0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    assert np.asarray(GR.dijkstra(M, indices=0)).tolist() == [0.0, INF, INF, 2.0]
    Z = sd.sparse.coo_array((np.array([0.0, 0.0]), (np.array([0, 1]), np.array([1, 2]))), shape=(3, 3))
    assert np.asarray(GR.dijkstra(Z, indices=0)).tolist() == [0.0, 0.0, 0.0]
    for x in (np.array([[0, 1], [0, 0]], dtype=bool), np.array([[0, 3], [0, 0]])):
        assert np.asarray(GR.dijkstra(x, indices=0)).tolist() == [0.0, float(x[0, 1])]


def test_indices_rules():
    A = small()
    for bad, err, msg in (
        (1.5, TypeError, r"^dijkstra\(\): 'indices' must be an int, not float$"),
        (True, TypeError, r"^dijkstra\(\): 'indices' must be an int, not bool$"),
        (-1, ValueError, r"^dijkstra\(\): 'indices' must be a node of the graph \(0 \.\. 99\), not -1$"),
        (100, ValueError, r"not 100$"),
        ([[0, 1]], ValueError, r"^dijkstra\(\): 'indices' must be an int or a 1-D array of ints, not a 2-D array$"),
        ([0, 100], ValueError, r"^dijkstra\(\): 'indices' must hold nodes of the graph \(0 \.\. 99\), not 100$"),
        ([0.0, 1.0], TypeError, r"^dijkstra\(\): 'indices' must hold ints, not float64$"),
        ([True], TypeError, r"must hold ints, not bool$"),
        ("0", TypeError, r"^dijkstra\(\): 'indices' must be an int, not str$"),
    ):
        with pytest.raises(err, match=msg):
            GR.dijkstra(A, indices=bad)
    assert GR.dijkstra(A, indices=np.int64(7)).shape == (100,)


def test_method_option():
    A = small()
    with pytest.raises(
        ValueError, match=r"^shortest_path\(\): 'method' must be 'auto', 'D', 'BF', 'J' or 'FW', not 'X'$"
    ):
        GR.shortest_path(A, method="X")
    with pytest.raises(TypeError, match=r"^shortest_path\(\): 'method' must be .*, not NoneType$"):
        GR.shortest_path(A, method=None)
    with pytest.raises(ValueError, match=r"^shortest_path\(\): 'indices' cannot be given with method='FW'$"):
        GR.shortest_path(A, method="FW", indices=0)
    assert np.asarray(GR.shortest_path(A, method="FW")).tobytes() == np.asarray(GR.floyd_warshall(A)).tobytes()


def test_call_shape_and_flags():
    A = small()
    with pytest.raises(TypeError, match=r"^dijkstra\(\): takes 1 positional argument"):
        GR.dijkstra(A, False)
    with pytest.raises(TypeError, match=r"^shortest_path\(\): unexpected keyword argument 'overwrite'"):
        GR.shortest_path(A, overwrite=True)
    with pytest.raises(TypeError, match=r"^johnson\(\): 'directed' must be True or False, not int$"):
        GR.johnson(A, directed=1)
    with pytest.raises(TypeError, match=r"^dijkstra\(\): 'min_only' must be True or False"):
        GR.dijkstra(A, min_only=None)


# ---- options ------------------------------------------------------------------------------------


def test_limit():
    unit = sd.sparse.csr_array((np.ones(4), (np.array([0, 1, 2, 3]), np.array([1, 2, 3, 4]))), shape=(5, 5))
    rec = SCIPY["rule/limit"]
    assert np.asarray(GR.dijkstra(unit, indices=0, limit=1)).tolist() == arr(rec["one"]["value"]).tolist()
    r = GR.dijkstra(unit, indices=0, limit=1.0, return_predecessors=True)
    assert np.asarray(r.predecessors).tolist() == [NONE, 0, NONE, NONE, NONE]
    for bad in (np.nan, -1, -1e-300):
        with pytest.raises(ValueError, match=r"^dijkstra\(\): 'limit' must be a number >= 0 or inf, not"):
            GR.dijkstra(unit, limit=bad)
    assert rec["nan"]["value"]["array"][1] == "inf"  # scipy: every node unreachable under a NaN limit
    with pytest.raises(TypeError, match=r"^dijkstra\(\): 'limit' must be an int or a float, not str$"):
        GR.dijkstra(unit, limit="1")
    assert np.asarray(GR.dijkstra(unit, indices=0, limit=0)).tolist() == [0.0, INF, INF, INF, INF]
    assert np.asarray(GR.dijkstra(unit, indices=0, limit=2, unweighted=True)).tolist() == [0, 1, 2, INF, INF]


def test_min_only():
    g5 = csr(G.get("grid2d/5x5"))
    rec = SCIPY["rule/min_only"]["value"]
    r = GR.dijkstra(g5, indices=[0, 24], min_only=True, return_predecessors=True)
    assert type(r).__name__ == "MinOnlyResult" and r._fields == ("dist_matrix", "predecessors", "sources")
    assert np.asarray(r.dist_matrix).tolist() == arr(rec[0]).tolist()
    assert r.predecessors.dtype == sd.int64 and r.sources.dtype == sd.int64
    srcs, pred = np.asarray(r.sources), np.asarray(r.predecessors)
    for v in range(25):  # sources[v] ends v's chain
        x = v
        while pred[x] != NONE:
            x = pred[x]
        assert srcs[v] == x
    assert GR.dijkstra(g5, indices=[0, 24], min_only=True).shape == (25,)
    a = GR.dijkstra(g5, min_only=True, return_predecessors=True)
    assert np.asarray(a.dist_matrix).tolist() == arr(SCIPY["rule/min_only_all"]["value"]).tolist()
    assert np.asarray(a.sources).tolist() == list(range(25)) and set(np.asarray(a.predecessors).tolist()) == {NONE}


def test_unweighted_counts_edges():
    A = csr(G.get("zero/15"))
    d = np.asarray(GR.shortest_path(A, unweighted=True))
    b = np.asarray(GR.shortest_path(A.astype(sd.bool), unweighted=True))
    assert d.tobytes() == b.tobytes()
    for v in range(15):
        order = np.asarray(GR.breadth_first_order(A, v).node_array)
        assert np.isfinite(d[v]).sum() == order.size
    # negative and NaN-free weights are not read: the pattern only (GR34)
    N = sd.sparse.csr_array(np.array([[0, -5.0], [-1.0, 0]]))
    assert np.asarray(GR.dijkstra(N, unweighted=True)).tolist() == [[0, 1], [1, 0]]


def test_undirected_reading():
    # i - j is the stored direction, the smaller weight when both are (scipy: rule/undirected_min)
    A = sd.sparse.csr_array(np.array([[0, 5.0, 0], [2.0, 0, 0], [0, 7.0, 0]]))
    for m in ("D", "BF", "J", "FW"):
        assert np.asarray(GR.shortest_path(A, method=m, directed=False)).tolist() == [
            [0, 2, 9],
            [2, 0, 7],
            [9, 7, 0],
        ]
    assert all(SCIPY["rule/undirected_min"].values())
    # G and its transpose read the same undirected graph
    g = csr(G.get("oneway/12"))
    assert (
        np.asarray(GR.shortest_path(g, directed=False, return_predecessors=True).predecessors).tobytes()
        == np.asarray(GR.shortest_path(g.T.tocsr(), directed=False, return_predecessors=True).predecessors).tobytes()
    )


# ---- negative weights and cycles ------------------------------------------------------------------


def test_negative_weights():
    A = csr(G.get("negative/dag30"))
    with pytest.raises(
        ValueError, match=r"^dijkstra\(\): negative edge weights are not allowed; use bellman_ford or johnson$"
    ):
        GR.dijkstra(A)
    with pytest.raises(ValueError, match=r"^shortest_path\(\): negative edge weights are not allowed; use method="):
        GR.shortest_path(A, method="D")
    assert "UserWarning" in SCIPY["rule/negative_dijkstra"]["warnings"]  # scipy warns and returns values
    want = np.asarray(GR.bellman_ford(A)).tobytes()
    for f in (lambda: GR.johnson(A), lambda: GR.shortest_path(A), lambda: GR.shortest_path(A, method="J")):
        assert np.asarray(f()).tobytes() == want
    # a negative zero is no negative weight
    Z = sd.sparse.csr_array((np.array([-0.0]), (np.array([0]), np.array([1]))), shape=(2, 2))
    assert np.asarray(GR.dijkstra(Z, indices=0)).tolist() == [0.0, 0.0]


def test_negative_cycle_error_class_and_message():
    assert issubclass(GR.NegativeCycleError, ValueError)
    assert GR.NegativeCycleError.__module__ == "stardust.graph"
    assert SCIPY["rule/negative_cycle_class"][1] == "Exception"  # scipy's derives from Exception
    # the cycle 3 -> 4 -> 2 -> 3 reachable from 0: the message names its smallest node
    M = np.zeros((5, 5))
    M[0, 1], M[1, 3], M[3, 4], M[4, 2], M[2, 3] = 1, 1, 1, -1, -1
    for f in (GR.bellman_ford, GR.johnson, GR.shortest_path, GR.floyd_warshall):
        with pytest.raises(GR.NegativeCycleError, match=rf"^{f.__name__}\(\): negative cycle through node 2$"):
            f(M)
    # a negative self loop is a cycle by itself
    L = sd.sparse.csr_array(np.array([[0, 1.0, 0], [0, -0.5, 0], [0, 0, 0]]))
    for f in (GR.bellman_ford, GR.johnson, GR.floyd_warshall):
        with pytest.raises(GR.NegativeCycleError, match=r"through node 1$"):
            f(L)
    assert np.asarray(GR.bellman_ford(L, indices=2)).tolist() == [INF, INF, 0.0]


def test_only_cycles_reachable_from_the_sources():
    un = csr(G.get("negative/unreach12"))
    rec = SCIPY["rule/unreachable_cycle"]
    want = arr(rec["bellman_ford"]["value"])
    for f in (GR.bellman_ford, GR.johnson):
        assert np.asarray(f(un, indices=0)).tolist() == want.tolist()
        with pytest.raises(GR.NegativeCycleError):
            f(un)
        with pytest.raises(GR.NegativeCycleError):
            f(un, indices=[0, 9])
    assert rec["johnson"]["error"] == "NegativeCycleError"  # scipy's johnson(indices=0) raises
    with pytest.raises(GR.NegativeCycleError):
        GR.floyd_warshall(un)
    # an undirected negative edge is a cycle (scipy too)
    u6 = csr(G.get("negative/undirected6"))
    assert SCIPY["rule/undirected_negative_edge"]["error"] == "NegativeCycleError"
    for f in (GR.bellman_ford, GR.johnson, GR.floyd_warshall):
        with pytest.raises(GR.NegativeCycleError):
            f(u6, directed=False)
    # a cycle of exact sum 0 is no negative cycle
    z = csr(G.get("negative/zerocycle10"))
    r = GR.bellman_ford(z, indices=0, return_predecessors=True)
    rec = SCIPY["rule/zero_cycle"]["value"]
    assert np.asarray(r.dist_matrix).tolist() == arr(rec[0]).tolist()
    assert np.asarray(r.predecessors).tolist() == arr(rec[1]).tolist()


def test_stored_nan_and_inf_are_errors():
    for gid in ("nonfinite/nan", "nonfinite/inf", "nonfinite/neginf"):
        A = csr(G.get(gid))
        for f in (
            lambda: GR.shortest_path(A),
            lambda: GR.dijkstra(A, indices=0),
            lambda: GR.bellman_ford(A, indices=0),
            lambda: GR.johnson(A),
            lambda: GR.floyd_warshall(A),
            lambda: GR.yen(A, 0, 3, 2),
            lambda: GR.construct_dist_matrix(A, np.full((4, 4), NONE)),
            lambda: GR.reconstruct_path(A, np.full(4, NONE)),
        ):
            with pytest.raises(ValueError, match=r"\(\): stored edge weights must be finite$"):
                f()
        assert np.asarray(GR.shortest_path(A, unweighted=True)).shape == (4, 4)  # the pattern only
    # scipy skips a stored NaN and lets -inf through
    assert SCIPY["rule/nan_dijkstra"]["value"]["array"] == [0.0, 1.0, "inf", 5.0]
    assert SCIPY["rule/neginf_bellman_ford"]["value"]["array"][2] == "-inf"


def test_memory_error_before_an_n_squared_output():
    big = sd.sparse.csr_array((10**6, 10**6))
    for f in (GR.shortest_path, GR.floyd_warshall, GR.johnson):
        with pytest.raises(
            MemoryError, match=r"\(\): cannot allocate \d+ bytes: more than this machine's physical memory$"
        ):
            f(big)
    with pytest.raises(MemoryError):
        GR.yen(big, 0, 1, 10**9, return_predecessors=True)
    assert GR.dijkstra(big, indices=0).shape == (10**6,)


# ---- construct_dist_matrix, reconstruct_path -------------------------------------------------------


def test_construct_dist_matrix():
    A = small()
    r = GR.shortest_path(A, return_predecessors=True)
    c = GR.construct_dist_matrix(A, r.predecessors)
    assert np.asarray(c).tobytes() == np.asarray(r.dist_matrix).tobytes()
    # scipy adds from the end of the path: its construct differs from its own distances (rule/sums)
    assert not SCIPY["rule/sums/er/float100"]["construct_equals_shortest_path"]
    p32 = np.asarray(r.predecessors).astype(np.int32)  # scipy's int32 predecessors are taken too
    assert np.asarray(GR.construct_dist_matrix(A, p32)).tobytes() == np.asarray(c).tobytes()
    G3 = sd.sparse.csr_array(np.array([[0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]]))
    P = [[NONE, 0, 1], [NONE, NONE, 1], [NONE, NONE, NONE]]
    assert np.asarray(GR.construct_dist_matrix(G3, P, null_value=-1)).tolist() == [[0, 1, 3], [-1, 0, 2], [-1, -1, 0]]
    U = sd.sparse.csr_array(np.array([[0, 3.0, 0], [1.0, 0, 2.0], [0, 0, 0]]))
    Pu = [[NONE, 0, 1], [1, NONE, 1], [1, 2, NONE]]
    assert np.asarray(GR.construct_dist_matrix(U, Pu, directed=False)).tolist() == [[0, 1, 3], [1, 0, 2], [3, 2, 0]]


def test_construct_dist_matrix_errors():
    G3 = sd.sparse.csr_array(np.array([[0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]]))
    name = "construct_dist_matrix"
    for P, err, msg in (
        ([[NONE, 0, 1]], ValueError, r"'predecessors' must have the shape \(3, 3\), not \(1, 3\)$"),
        (np.zeros((3, 3)), TypeError, r"'predecessors' must hold ints, not float64$"),
        (
            [[NONE, 0, 1], [NONE, NONE, 7], [NONE] * 3],
            ValueError,
            r"predecessors\[1, 2\] = 7: a predecessor is a node \(0 \.\. 2\) or -9999$",
        ),
        (
            [[NONE, 0, 0], [NONE, NONE, 1], [NONE] * 3],
            ValueError,
            r"predecessors\[0, 2\] = 0, but the graph has no edge 0 -> 2$",
        ),
        (
            [[NONE, 2, 1], [NONE, NONE, 1], [NONE] * 3],
            ValueError,
            r"the predecessors of row 0 go round a cycle through node \d$",
        ),
        (
            [[NONE, NONE, 1], [NONE, NONE, 1], [NONE] * 3],
            ValueError,
            r"the predecessors of row 0 lead from node 2 to node 1, which has no predecessor, not to node 0$",
        ),
        (sd.sparse.csr_array(np.eye(3)), TypeError, r"'predecessors' must be an array of ints, not"),
    ):
        with pytest.raises(err, match=rf"^{name}\(\): " + msg.lstrip("^")):
            GR.construct_dist_matrix(G3, P)


def test_construct_dist_matrix_checks_only_the_range_of_the_diagonal():
    """The path from a node to itself is empty, so predecessors[i, i] is range-checked and not followed (scipy
    reads it neither): a node number there, the row's own one included, changes nothing; off the diagonal a
    node pointing at itself is a cycle."""
    G3 = sd.sparse.csr_array(np.array([[0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]]))
    P = [[NONE, 0, 1], [NONE, NONE, 1], [NONE] * 3]
    want = np.asarray(GR.construct_dist_matrix(G3, P)).tolist()
    for diag in ([0, 1, 2], [2, 0, 1]):
        Q = [row[:] for row in P]
        for i, v in enumerate(diag):
            Q[i][i] = v
        assert np.asarray(GR.construct_dist_matrix(G3, Q)).tolist() == want
    Q = [row[:] for row in P]
    Q[0][0] = 3
    with pytest.raises(ValueError, match=r"predecessors\[0, 0\] = 3: a predecessor is a node \(0 \.\. 2\) or -9999$"):
        GR.construct_dist_matrix(G3, Q)
    with pytest.raises(ValueError, match=r"the predecessors of row 0 go round a cycle through node 2$"):
        GR.construct_dist_matrix(G3, [[NONE, 0, 2], [NONE, NONE, 1], [NONE] * 3])


def test_reconstruct_path():
    A = small()
    r = GR.shortest_path(A, indices=4, return_predecessors=True)
    T = GR.reconstruct_path(A, r.predecessors)
    assert isinstance(T, sd.sparse.csr_array) and T.shape == (100, 100) and T.dtype == sd.float64
    p = np.asarray(r.predecessors)
    assert T.nnz == int((p != NONE).sum())
    dense = np.asarray(T.toarray())
    Ad = np.asarray(A.toarray())
    for v in range(100):
        if p[v] != NONE:
            assert dense[p[v], v] == Ad[p[v], v]
    Z = sd.sparse.csr_array((np.array([0.0, 1.0]), (np.array([0, 1]), np.array([1, 2]))), shape=(3, 3))
    assert GR.reconstruct_path(Z, [NONE, 0, 1]).nnz == 2  # a zero-weight edge stays a stored 0
    U = sd.sparse.csr_array(np.array([[0, 3.0, 0], [1.0, 0, 2.0], [0, 0, 0]]))
    assert np.asarray(GR.reconstruct_path(U, [NONE, 0, 1], directed=False).toarray()).tolist() == [
        [0, 1, 0],
        [0, 0, 2],
        [0, 0, 0],
    ]
    name = "reconstruct_path"
    G3 = sd.sparse.csr_array(np.array([[0, 1.0, 0], [0, 0, 2.0], [0, 0, 0]]))
    for P, err, msg in (
        ([NONE, 0, 0], ValueError, r"predecessors\[2\] = 0, but the graph has no edge 0 -> 2$"),
        ([NONE, 0, 1, 2], ValueError, r"'predecessors' must have the shape \(3,\), not \(4,\)$"),
        ([NONE, 0, 9], ValueError, r"predecessors\[2\] = 9: a predecessor is a node"),
        ([NONE, 1, 1], ValueError, r"predecessors\[1\] = 1: a node is not its own predecessor$"),
        ([0.0, 0.0, 1.0], TypeError, r"'predecessors' must hold ints, not float64$"),
    ):
        with pytest.raises(err, match=rf"^{name}\(\): " + msg):
            GR.reconstruct_path(G3, P)


# ---- scipy -------------------------------------------------------------------------------------------


def test_scipy_records_distances():
    """scipy's D (and J for the negative graphs) sums from the source as sd does: the same distances, bit
    for bit; its FW's last bits are its own."""
    seen = 0
    for key, rec in SCIPY.items():
        if not (key.startswith("graph/") and key.endswith("/shortest_path")):
            continue
        gid = key[len("graph/") : -len("/shortest_path")]
        m = "D" if "D" in rec else "J"
        want = arr(rec[m]["value"][0])
        got = np.asarray(GR.shortest_path(csr(G.get(gid))))
        assert got.tobytes() == want.tobytes(), gid
        seen += 1
    assert seen > 20


def test_scipy_predecessor_ties_differ_by_method():
    rec = SCIPY["rule/diamond_predecessors"]
    assert rec == {"D": 2, "BF": 1, "J": 2, "FW": 1}
    A = csr(G.get("diamond/4"))
    for m in ("D", "BF", "J", "auto"):
        assert int(np.asarray(GR.shortest_path(A, method=m, return_predecessors=True).predecessors)[0, 3]) == 1


def test_scipy_auto_takes_fw_on_dense_graphs():
    assert SCIPY["rule/auto"]["0.3"]["equals_FW"] and not SCIPY["rule/auto"]["0.3"]["equals_D"]
    g = np.random.Generator(np.random.PCG64([7, 30]))
    mask = g.random((200, 200)) < 0.3
    np.fill_diagonal(mask, False)
    A = np.where(mask, g.uniform(0.01, 1.01, (200, 200)), 0.0)
    auto = GR.shortest_path(A, return_predecessors=True)
    d = GR.shortest_path(A, method="D", return_predecessors=True)
    assert np.asarray(auto.dist_matrix).tobytes() == np.asarray(d.dist_matrix).tobytes()
    assert np.asarray(auto.predecessors).tobytes() == np.asarray(d.predecessors).tobytes()


def test_scipy_indices_2d_and_overwrite():
    assert SCIPY["rule/indices_2d"]["value"] == [2, 2, 10]  # scipy: a 3-D result
    with pytest.raises(ValueError):
        GR.dijkstra(small(), indices=[[0, 1], [2, 3]])


@LIVE
def test_scipy_live_same_distances_and_predecessors_without_ties():
    from .graphs import random_graph

    ip, ix, w = random_graph(500, 6, seed=11)
    a = S.csr_array((w, ix, ip), shape=(500, 500))
    A = sd.sparse.csr_array((w, ix, ip), shape=(500, 500))
    for method in ("D", "BF", "J"):
        d, p = CG.shortest_path(a, method=method, indices=np.arange(0, 500, 7), return_predecessors=True)
        r = GR.shortest_path(A, method=method, indices=np.arange(0, 500, 7), return_predecessors=True)
        assert np.asarray(r.dist_matrix).tobytes() == d.tobytes(), method
        assert np.asarray(r.predecessors).tolist() == p.tolist(), method  # random weights: no ties
    fw = CG.floyd_warshall(a)
    ours = np.asarray(GR.floyd_warshall(A))
    assert np.allclose(ours, fw, rtol=1e-14, atol=0)
    lim = CG.dijkstra(a, indices=3, limit=1.5)
    assert np.asarray(GR.dijkstra(A, indices=3, limit=1.5)).tobytes() == lim.tobytes()
    mo, mp, ms = CG.dijkstra(a, indices=[1, 2, 3], min_only=True, return_predecessors=True)
    r = GR.dijkstra(A, indices=[1, 2, 3], min_only=True, return_predecessors=True)
    assert np.asarray(r.dist_matrix).tobytes() == mo.tobytes()
    assert np.asarray(r.predecessors).tolist() == mp.tolist() and np.asarray(r.sources).tolist() == ms.tolist()
    y = CG.yen(a, 0, 499, 4)
    assert np.asarray(GR.yen(A, 0, 499, 4)).tolist() == y.tolist()
