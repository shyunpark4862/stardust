"""sd.graph gives the same bytes for 1, 3 and 10 threads (docs/design/graph.md §1.3, §3.8; milestone GR1):
inputs large enough for every parallel path (the parallel union-find from 2^18 stored entries, the
row-block passes of the dense conversion and of csgraph_to_dense) and for the calls that drop the GIL."""

import hashlib

import numpy as np
import pytest

import stardust as sd

from .graphs import random_graph

pytestmark = pytest.mark.usefixtures("restore_threads")


def outputs(A, D, S):
    """Every GR1 function's results as bytes."""
    g = sd.graph
    out = []

    def add(x):
        if isinstance(x, sd.sparse.sparray):
            out.extend(np.asarray(a).tobytes() for a in (x.indptr, x.indices, x.data))
        elif isinstance(x, tuple):
            for y in x:
                add(y)
        elif isinstance(x, int):
            out.append(str(x).encode())
        else:
            out.append(np.asarray(x).tobytes())

    for G in (A, S):
        add(g.connected_components(G))
        add(g.connected_components(G, connection="strong"))
        add(g.connected_components(G, directed=False))
        for directed in (True, False):
            add(g.breadth_first_order(G, 0, directed=directed))
            add(g.depth_first_order(G, 7, directed=directed))
            add(g.breadth_first_tree(G, 3, directed=directed))
            add(g.depth_first_tree(G, 3, directed=directed))
    add(g.topological_sort(sd.sparse.csr_array(sd.graph.csgraph_from_dense(np.triu(np.asarray(D)[:600, :600], 1)))))
    add(g.csgraph_from_dense(D))
    add(g.csgraph_from_dense(D, null_value=0.5, infinity_null=False))
    add(g.csgraph_to_dense(g.csgraph_from_dense(D), null_value=np.inf))
    add(g.connected_components(D))
    return out


def test_one_three_ten_threads():
    n = 80000
    ip, ix, w = random_graph(n, 4, seed=3)  # 3.2 * 10^5 stored entries: the parallel union-find
    A = sd.sparse.csr_array((w, ix, ip), shape=(n, n))
    sym_ip, sym_ix, sym_w = random_graph(20000, 8, seed=4, symmetric=True)
    S = sd.sparse.csr_array((sym_w, sym_ix, sym_ip), shape=(20000, 20000))
    r = np.random.Generator(np.random.PCG64(5))
    D = r.uniform(0, 1, (1200, 1200))
    D[D < 0.7] = 0.0
    D[r.random(D.shape) < 0.01] = np.nan
    D[r.random(D.shape) < 0.01] = np.inf
    runs = []
    for threads in (1, 3, 10):
        sd.set_num_threads(threads)
        runs.append(outputs(A, D, S))
    assert runs[0] == runs[1] == runs[2]


def path_outputs(A, N, D):
    """Every GR2 function's results as bytes: the multi-source tasks (indices, all pairs), the blocked
    Floyd-Warshall's block tasks, the dense Dijkstra of auto, Johnson's potentials, the rows of
    construct_dist_matrix (milestone GR2)."""
    g = sd.graph
    out = []

    def add(x):  # digests: the results are tens of MB
        if isinstance(x, sd.sparse.sparray):
            out.extend(hashlib.sha256(np.asarray(a).tobytes()).hexdigest() for a in (x.indptr, x.indices, x.data))
        elif isinstance(x, tuple):
            for y in x:
                add(y)
        else:
            out.append(hashlib.sha256(np.asarray(x).tobytes()).hexdigest())

    for method in ("D", "BF", "J", "auto"):
        add(g.shortest_path(A, method=method, indices=np.arange(0, 600, 3), return_predecessors=True))
    add(g.dijkstra(A, indices=np.arange(50), min_only=True, return_predecessors=True))
    add(g.dijkstra(A, indices=np.arange(40), limit=2.0, return_predecessors=True, directed=False))
    add(g.johnson(N, indices=np.arange(0, 600, 5), return_predecessors=True))
    add(g.bellman_ford(N, indices=np.arange(0, 600, 7), return_predecessors=True, directed=True))
    r = g.shortest_path(A[:700, :700], return_predecessors=True)
    add(r)
    add(g.construct_dist_matrix(A[:700, :700], r.predecessors))
    add(g.floyd_warshall(D, return_predecessors=True))
    add(g.floyd_warshall(D[:300, :300], directed=False))
    add(g.shortest_path(D, return_predecessors=True))  # auto: the dense Dijkstra
    add(g.yen(A, 0, 5000, 6, return_predecessors=True))
    return out


def test_paths_one_three_ten_threads():
    n = 20000
    ip, ix, w = random_graph(n, 6, seed=7)
    A = sd.sparse.csr_array((w, ix, ip), shape=(n, n))
    r = np.random.Generator(np.random.PCG64(8))
    # w(u, v) + p(u) - p(v): many negative weights, every cycle sum still positive
    p = r.uniform(0, 2, n)
    rows = np.repeat(np.arange(n), np.diff(ip))
    N = sd.sparse.csr_array((w + p[rows] - p[ix], ix, ip), shape=(n, n))
    D = r.uniform(0.01, 1.01, (450, 450))
    D[r.random(D.shape) < 0.3] = 0.0
    runs = []
    for threads in (1, 3, 10):
        sd.set_num_threads(threads)
        runs.append(path_outputs(A, N, D))
    assert runs[0] == runs[1] == runs[2]
