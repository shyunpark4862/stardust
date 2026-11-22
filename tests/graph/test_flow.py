"""maximum_flow (docs/api/graph.md §4, docs/design/graph.md §2.4, §3.5; decisions GR15, GR16; milestone GR3):
the flow value equals the definition's and the capacity of the minimum cut the returned flow leaves, the flow
is antisymmetric, within the capacities and conserved, Dinic and Edmonds-Karp give the same value, the
int64 capacities are exact (scipy reads them as int32), the result object, and the errors."""

import numpy as np
import pytest
import scipy.sparse as S
import scipy.sparse.csgraph as CS

import stardust as sd

from . import definitions as D
from . import graphs as G
from .conftest import LIVE
from .forms import csr

GR = sd.graph
METHODS = ("dinic", "edmonds_karp")


def flow_dict(F):
    ip, ix, w = np.asarray(F.indptr), np.asarray(F.indices), np.asarray(F.data)
    return {(i, int(ix[k])): int(w[k]) for i in range(F.shape[0]) for k in range(ip[i], ip[i + 1])}


def check_flow(n, caps, s, t, r):
    """The returned flow is a maximum flow: antisymmetric on the pattern of G and its transpose, within the
    capacities, conserved, of value flow_value, and the nodes the residual graph reaches from s make a cut
    of capacity flow_value (so no flow is larger)."""
    f = flow_dict(r.flow)
    pattern = set(caps) | {(v, u) for u, v in caps}
    assert set(f) == pattern
    for (u, v), x in f.items():
        assert x == -f[(v, u)]
        if u == v:
            assert x == 0
        else:
            assert x <= caps.get((u, v), 0)
    net = [0] * n
    for (u, _), x in f.items():
        net[u] += x
    assert net[s] == r.flow_value and net[t] == -r.flow_value
    assert all(net[v] == 0 for v in range(n) if v not in (s, t))
    reach, stack = {s}, [s]
    while stack:
        u = stack.pop()
        for (a, b), x in f.items():
            if a == u and b not in reach and caps.get((a, b), 0) - x > 0:
                reach.add(b)
                stack.append(b)
    assert t not in reach
    cut = sum(c for (u, v), c in caps.items() if u in reach and v not in reach)
    assert cut == r.flow_value


def capacities(g):
    return {(i, j): int(w) for i, j, w in g.edges()}


@pytest.mark.parametrize("gid", G.ids("flow"))
@pytest.mark.parametrize("method", METHODS)
def test_flow_graphs(gid, method):
    g = G.get(gid)
    A = csr(g)
    caps = capacities(g)
    s, t = 0, g.n - 1
    if gid == "flow/overflow3":
        with pytest.raises(OverflowError, match=r"maximum_flow\(\): the capacities out of the source sum past"):
            GR.maximum_flow(A, s, t, method=method)
        return
    r = GR.maximum_flow(A, s, t, method=method)
    assert type(r.flow_value) is int and r.flow_value == D.max_flow_value(g.n, caps, s, t)
    assert r.flow.dtype == sd.int64
    check_flow(g.n, caps, s, t, r)


def random_network(n, p, seed, hi=20, loops=True):
    rng = np.random.Generator(np.random.PCG64(seed))
    M = rng.integers(1, hi, (n, n)) * (rng.random((n, n)) < p)
    if not loops:
        np.fill_diagonal(M, 0)
    return M.astype(np.int64)


@pytest.mark.parametrize("seed", range(12))
def test_random_networks_both_methods(seed):
    n = 6 + 3 * seed
    M = random_network(n, 0.3, seed)
    caps = {(int(i), int(j)): int(M[i, j]) for i, j in zip(*np.nonzero(M))}
    want = D.max_flow_value(n, caps, 0, n - 1)
    values = []
    for method in METHODS:
        for A in (M, sd.asarray(M), sd.sparse.csr_array(M), sd.sparse.coo_array(M)):
            r = GR.maximum_flow(A, 0, n - 1, method=method)
            check_flow(n, caps, 0, n - 1, r)
            values.append(r.flow_value)
    assert set(values) == {want}


def test_stored_zero_capacities_and_bool():
    # a stored 0 is an edge of capacity 0: in the flow's pattern, carrying nothing
    A = sd.sparse.coo_array((np.array([0, 3, 2], np.int64), (np.array([0, 1, 0]), np.array([1, 2, 2]))), shape=(3, 3))
    r = GR.maximum_flow(A, 0, 2)
    assert r.flow_value == 2 and flow_dict(r.flow) == {
        (0, 1): 0,
        (0, 2): 2,
        (1, 0): 0,
        (1, 2): 0,
        (2, 0): -2,
        (2, 1): 0,
    }
    B = np.array([[False, True, True], [False, False, True], [False, False, False]])
    assert GR.maximum_flow(B, 0, 2).flow_value == 2
    assert GR.maximum_flow(sd.sparse.csr_array(B), 0, 2, method="edmonds_karp").flow_value == 2


def test_self_loops_and_two_way_edges():
    M = np.array([[5, 4, 0], [2, 7, 3], [0, 0, 9]], np.int64)
    r = GR.maximum_flow(M, 0, 2)
    assert r.flow_value == 3
    f = flow_dict(r.flow)
    assert f[(0, 0)] == f[(1, 1)] == f[(2, 2)] == 0  # the diagonal stays in the pattern (scipy's too)
    assert f[(0, 1)] == 3 and f[(1, 0)] == -3


def test_int64_capacities_are_exact():
    big = G.get("flow/big6")
    assert GR.maximum_flow(csr_int(big), 0, 5).flow_value == 2**62
    past = G.get("flow/past_int32")
    for method in METHODS:
        assert GR.maximum_flow(csr_int(past), 0, 3, method=method).flow_value == 2**40 + 2**32 + 5
    # residual capacities past int64: c(u, v) + c(v, u) > 2^63 - 1
    M = np.zeros((4, 4), np.int64)
    M[0, 1] = M[1, 3] = 2**62
    M[1, 2] = M[2, 1] = 2**63 - 1
    M[2, 3] = 2**62
    for method in METHODS:
        r = GR.maximum_flow(M, 0, 3, method=method)
        assert r.flow_value == 2**62
        check_flow(4, {(int(i), int(j)): int(M[i, j]) for i, j in zip(*np.nonzero(M))}, 0, 3, r)


def csr_int(g):
    return sd.sparse.csr_array((sd.asarray(g.data), sd.asarray(g.indices), sd.asarray(g.indptr)), shape=g.shape)


def test_result_object():
    r = GR.maximum_flow(np.array([[0, 5, 0], [0, 0, 7], [0, 0, 0]], np.int64), 0, 2)
    assert repr(r) == "MaximumFlowResult(flow_value=5, flow=<csr_array (3, 3), nnz 4>)"
    a, b = r.flow, r.flow
    assert a is not b
    a.data[0] = 99  # a new array each time: writing one leaves the result as it was
    assert int(r.flow.data[0]) == 5
    assert type(r).__name__ == "MaximumFlowResult" and type(r).__module__ == "stardust.graph"


def test_unreachable_sink_and_empty_rows():
    r = GR.maximum_flow(np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]], np.int64), 0, 2)
    assert r.flow_value == 0 and flow_dict(r.flow) == {(0, 1): 0, (1, 0): 0}


def test_errors():
    M = np.array([[0, 3], [0, 0]], np.int64)
    with pytest.raises(TypeError, match=r"maximum_flow\(\): the capacities must be bool or int64, not float64"):
        GR.maximum_flow(M.astype(np.float64), 0, 1)
    with pytest.raises(TypeError, match="bool or int64, not float64"):
        GR.maximum_flow(sd.sparse.csr_array(M.astype(np.float64)), 0, 1)
    with pytest.raises(ValueError, match=r"maximum_flow\(\): the capacities must be nonnegative"):
        GR.maximum_flow(np.array([[0, -3], [0, 0]]), 0, 1)
    with pytest.raises(ValueError, match="'source' and 'sink' must be different nodes, not both 1"):
        GR.maximum_flow(M, 1, 1)
    with pytest.raises(ValueError, match=r"'sink' must be a node of the graph \(0 .. 1\), not 2"):
        GR.maximum_flow(M, 0, 2)
    with pytest.raises(TypeError, match="'source' must be an int, not float"):
        GR.maximum_flow(M, 0.0, 1)
    with pytest.raises(ValueError, match="'method' must be"):
        GR.maximum_flow(M, 0, 1, method="push_relabel")
    with pytest.raises(TypeError, match="positional"):
        GR.maximum_flow(M, 0, 1, "dinic")
    with pytest.raises(ValueError, match="square"):
        GR.maximum_flow(np.zeros((2, 3), np.int64), 0, 1)
    with pytest.raises(TypeError, match="sd.sparse.csr_array"):
        GR.maximum_flow(S.csr_array(M), 0, 1)
    with pytest.raises(TypeError, match="real"):
        GR.maximum_flow(np.eye(2) * 1j, 0, 1)
    with pytest.raises(ValueError, match="no nodes"):
        GR.maximum_flow(np.zeros((0, 0), np.int64), 0, 0)


def test_large_network():
    rng = np.random.Generator(np.random.PCG64(9))
    n, d = 3000, 6
    rows = np.repeat(np.arange(n), d)
    cols = rng.integers(0, n, n * d)
    A = sd.sparse.coo_array((rng.integers(1, 50, n * d), (rows, cols)), shape=(n, n)).tocsr()
    values = {GR.maximum_flow(A, 0, n - 1, method=m).flow_value for m in METHODS}
    a = S.coo_array(
        (np.asarray(A.data), (np.repeat(np.arange(n), np.diff(np.asarray(A.indptr))), np.asarray(A.indices))),
        shape=(n, n),
    )
    caps = {(int(i), int(j)): int(c) for i, j, c in zip(a.row, a.col, a.data)}
    assert values == {D.max_flow_value(n, caps, 0, n - 1)}


@LIVE
@pytest.mark.parametrize("method", METHODS)
def test_scipy_gives_the_same_flow(method):
    # the same augmenting order (neighbours ascending on the residual pattern): scipy's flow is sd's,
    # wherever the capacities fit in int32 and none is negative
    for seed in range(8):
        n = 10 + 7 * seed
        M = random_network(n, 0.25, 100 + seed, hi=60)
        mine = GR.maximum_flow(sd.sparse.csr_array(M), 0, n - 1, method=method)
        theirs = CS.maximum_flow(S.csr_array(M.astype(np.int32)), 0, n - 1, method=method)
        assert mine.flow_value == int(theirs.flow_value)
        F = theirs.flow
        assert np.array_equal(np.asarray(mine.flow.indptr), F.indptr)
        assert np.array_equal(np.asarray(mine.flow.indices), F.indices)
        assert np.array_equal(np.asarray(mine.flow.data), F.data)


@LIVE
def test_scipy_truncates_int64_capacities():
    # [scipy differs]: scipy reads the capacities as int32: 2^40 becomes 0 and 2^32 + 5 becomes 5
    g = G.get("flow/past_int32")
    assert int(CS.maximum_flow(g.scipy(), 0, 3).flow_value) == 5
    assert GR.maximum_flow(csr_int(g), 0, 3).flow_value == 2**40 + 2**32 + 5
    # and reads a negative capacity as none, where sd raises
    assert int(CS.maximum_flow(S.csr_array(np.array([[0, -3], [0, 0]])), 0, 1).flow_value) == 0
