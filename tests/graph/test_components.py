"""connected_components of sd.graph (docs/api/graph.md §2, docs/design/graph.md §2.2, §3.3; milestone GR1):
weak, strong and undirected components equal their definitions (tests/graph/definitions.py: undirected
paths, mutual reachability) on every square test graph and on random graphs, numbered by the order of each
component's smallest node (GR12) whichever union order or path (the serial union-find, the parallel one
from 2^18 stored entries) found them; the result types; scipy 1.18.1 gives the same weak labels and the
same strong partition under its own numbering."""

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


def labels(r):
    return int(r.n_components), np.asarray(r.labels).tolist()


@pytest.mark.parametrize("gid", IDS)
def test_components_equal_the_definitions(gid):
    g = G.get(gid)
    edges = g.edges()
    weak, strong = D.components(g.n, edges, False), D.components(g.n, edges, True)
    for form, A in forms(g).items():
        assert labels(sd.graph.connected_components(A)) == weak, (gid, form)
        assert labels(sd.graph.connected_components(A, connection="strong")) == strong, (gid, form)
        for connection in ("weak", "strong"):  # undirected: connection is not read
            assert labels(sd.graph.connected_components(A, directed=False, connection=connection)) == weak


def test_random_graphs_equal_the_definitions():
    for seed in range(60):
        r = np.random.Generator(np.random.PCG64([20261124, seed]))
        n = int(r.integers(1, 50))
        M = r.random((n, n)) < float(r.uniform(0.0, 0.12))
        edges = [(i, j, 1.0) for i in range(n) for j in range(n) if M[i, j]]
        assert labels(sd.graph.connected_components(M)) == D.components(n, edges, False)
        assert labels(sd.graph.connected_components(M, connection="strong")) == D.components(n, edges, True)


def test_labels_follow_the_smallest_node():
    # components {0, 3}, {1}, {2, 4}: numbered 0, 1, 2 by their smallest nodes 0, 1, 2
    M = np.zeros((5, 5))
    M[3, 0] = M[0, 3] = M[4, 2] = 1.0
    assert labels(sd.graph.connected_components(M)) == (3, [0, 1, 2, 0, 2])
    assert labels(sd.graph.connected_components(M, connection="strong")) == (4, [0, 1, 2, 0, 3])


@LIVE
@pytest.mark.usefixtures("restore_threads")
def test_large_graphs_take_the_parallel_union_find_and_agree():
    # 2^18 stored entries and more split the unions (csrc/graph/traversal.cpp) when the pool has threads; one
    # thread runs the serial union-find. Both give scipy's labels (the smallest-node order).
    r = np.random.Generator(np.random.PCG64(77))
    n, d = 150000, 6
    rows = np.repeat(np.arange(n), d)
    cols = (rows * 7 + r.integers(0, 300, n * d)) % n  # many components of local edges
    keep = r.random(n * d) < 0.3
    B = S.csr_array((np.ones(int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n))
    assert B.nnz >= 2**18
    A = sd.sparse.csr_array(B)
    want = CS.connected_components(B)
    for threads in (1, 10):
        sd.set_num_threads(threads)
        assert labels(sd.graph.connected_components(A)) == (want[0], want[1].tolist()), threads
        assert labels(sd.graph.connected_components(A, directed=False)) == (want[0], want[1].tolist()), threads


def test_result_types():
    A = csr(G.get("components/40"))
    r = sd.graph.connected_components(A)
    assert type(r).__name__ == "ConnectedComponentsResult" and r._fields == ("n_components", "labels")
    assert type(r.n_components) is int and r.labels.dtype == sd.int64 and not r.labels._flags()["writeable"]
    count = sd.graph.connected_components(A, return_labels=False)
    assert type(count) is int and count == r.n_components == 4
    e = sd.graph.connected_components(sd.sparse.csr_array((0, 0)))
    assert e.n_components == 0 and e.labels.shape == (0,)


# ---- scipy 1.18.1 ----


def partition(lab):
    groups = {}
    for v, c in enumerate(lab):
        groups.setdefault(c, []).append(v)
    return sorted(groups.values())


@LIVE
@pytest.mark.parametrize("gid", IDS)
def test_scipy_weak_labels_are_the_same_and_strong_partitions_too(gid):
    g = G.get(gid)
    A, B = csr(g), g.scipy()
    for directed in (True, False):
        got, want = sd.graph.connected_components(A, directed=directed), CS.connected_components(B, directed=directed)
        assert labels(got) == (want[0], want[1].tolist()), gid
    got = sd.graph.connected_components(A, connection="strong")
    want = CS.connected_components(B, connection="strong")
    assert got.n_components == want[0] and partition(labels(got)[1]) == partition(want[1].tolist()), gid


@LIVE
def test_scipy_numbers_strong_components_otherwise():
    # decision GR12: scipy numbers the strong components in its algorithm's order
    M = np.zeros((4, 4))
    M[0, 1] = M[1, 2] = M[2, 3] = 1.0
    assert CS.connected_components(S.csr_array(M), connection="strong")[1].tolist() == [3, 2, 1, 0]
    assert labels(sd.graph.connected_components(M, connection="strong")) == (4, [0, 1, 2, 3])
    assert CS.connected_components(S.csr_array(M))[1].dtype == np.int32
