"""topological_sort of sd.graph (docs/api/graph.md §2, docs/design/graph.md §2.2, §3.3; milestone GR1): the
lexicographically smallest topological order, by its definition (the smallest of all topological orders,
enumerated on small DAGs; the greedy rule on the test graphs and random DAGs), a ValueError for a cycle and
a self loop, and the same answer as networkx 3.7's lexicographical_topological_sort."""

import itertools

import networkx as nx
import numpy as np
import pytest

import stardust as sd

from . import definitions as D
from . import graphs as G
from .conftest import LIVE
from .forms import csr, forms, square_ids

IDS = square_ids()


def topo(A):
    return np.asarray(sd.graph.topological_sort(A)).tolist()


@pytest.mark.parametrize("gid", IDS)
def test_test_graphs(gid):
    g = G.get(gid)
    edges = g.edges()
    want = D.topological_sort(g.n, edges)
    for form, A in forms(g).items():
        if want is None:
            with pytest.raises(ValueError, match=r"^topological_sort\(\): the graph has a cycle$"):
                sd.graph.topological_sort(A)
        else:
            got = topo(A)
            assert got == want and D.is_topological(g.n, edges, got), (gid, form)


def test_the_smallest_of_all_orders_on_small_dags():
    for seed in range(80):
        r = np.random.Generator(np.random.PCG64([20261125, seed]))
        n = int(r.integers(1, 7))
        perm = r.permutation(n)
        M = np.triu(r.random((n, n)) < 0.4, 1)[np.ix_(perm, perm)]  # a DAG in a random node order
        edges = [(i, j, 1.0) for i in range(n) for j in range(n) if M[i, j]]
        orders = [list(p) for p in itertools.permutations(range(n)) if D.is_topological(n, edges, p)]
        assert topo(M) == min(orders)


def test_random_dags():
    for seed in range(30):
        r = np.random.Generator(np.random.PCG64([20261126, seed]))
        n = int(r.integers(1, 300))
        perm = r.permutation(n)
        M = np.triu(r.random((n, n)) < 3.0 / max(n, 1), 1)[np.ix_(perm, perm)]
        edges = [(i, j, 1.0) for i in range(n) for j in range(n) if M[i, j]]
        assert topo(sd.sparse.csr_array(M)) == D.topological_sort(n, edges)


def test_cycles_and_self_loops():
    with pytest.raises(ValueError, match="has a cycle"):
        sd.graph.topological_sort([[1, 0], [0, 0]])  # a self loop
    with pytest.raises(ValueError, match="has a cycle"):
        sd.graph.topological_sort([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    # a stored zero and a stored NaN are edges (the pattern; docs/design/graph.md §2.1)
    Z = sd.sparse.coo_array((np.array([0.0, np.nan]), (np.array([1, 2]), np.array([0, 1]))), shape=(3, 3))
    assert topo(Z) == [2, 1, 0]
    assert topo(sd.sparse.csr_array((0, 0))) == [] and topo([[0]]) == [0]


def test_result_type():
    r = sd.graph.topological_sort(csr(G.get("dag/30")))
    assert isinstance(r, sd.ndarray) and r.dtype == sd.int64 and r.shape == (30,) and r._flags()["writeable"]


@LIVE
@pytest.mark.parametrize("gid", [i for i in IDS if D.topological_sort(G.get(i).n, G.get(i).edges()) is not None])
def test_networkx_lexicographical_topological_sort_is_the_same(gid):
    g = G.get(gid)
    H = nx.DiGraph()
    H.add_nodes_from(range(g.n))
    H.add_edges_from((i, j) for i, j, _ in g.edges())
    assert topo(csr(g)) == list(nx.lexicographical_topological_sort(H))
