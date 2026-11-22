"""Golden cases for sd.graph (tools/golden.py has the contract).

The traversals and components (milestone GR1, docs/design/graph.md §8): connected_components (weak,
strong, undirected; with and without labels), breadth_first_order, depth_first_order, breadth_first_tree,
depth_first_tree (directed and undirected, from several starts), topological_sort and the conversion tools
csgraph_from_dense and csgraph_to_dense, on small graphs given every way (the four sparse formats, dense
sd, numpy and nested lists, bool, int64 and float64, stored zeros and NaN, self loops, one-way and two-way
edges of different weights, n = 0 and 1) and on graphs large enough for the parallel union-find (2^18
stored entries and more) and the row-block passes of the dense conversion at 3 and 10 threads; the wrong
calls (their errors are results). The graphs come from fixed literals and L.rng(seed); the largest result
is 1.2 * 10^5 int64s.

The shortest paths (milestone GR2): shortest_path with every method ('auto', 'D', 'BF', 'J', 'FW'), dijkstra
(indices of every form, limit, min_only), bellman_ford, johnson, floyd_warshall, yen, construct_dist_matrix
and reconstruct_path, directed and undirected, weighted and unweighted, on the small graphs above and on
graphs with ties, zero-weight edges, negative weights, reachable and unreachable negative cycles
(NegativeCycleError is a result); and large enough for the parallel paths at 3 and 10 threads: the
source tasks (indices, all pairs), the blocked Floyd-Warshall's block tasks, auto's dense Dijkstra and the
rows of construct_dist_matrix (the largest result 4.9 * 10^5 float64s).

The spanning tree, flow, matchings and assignment (milestone GR3): minimum_spanning_tree on the small graphs
above and on ties, two-way edges, stored -0.0 and negative weights; maximum_flow by both methods on int64 and
bool capacities (stored zero capacities, self loops, two-way edges, capacities near 2^62 and past int32);
maximum_bipartite_matching (both perm_types, rectangular); min_weight_full_bipartite_matching and
linear_sum_assignment (ties, rectangular both ways, +inf and -inf, maximize, stored zeros, infeasible) and
the errors; and large inputs past the GIL threshold at 3 and 10 threads (the largest result 1.6 * 10^5
int64s). Which of several optima the flow, the matchings and the assignments return is fixed by their scan
order (decision GR16): these cases guard it.

The ordering, structural rank and laplacian (milestone GR4): reverse_cuthill_mckee (both modes) and
structural_rank on the small graphs above, on rectangular matrices and on graphs of many ties; laplacian on
the small graphs with every option (normed, use_out_degree, symmetrized, return_diag, dtype), on signed
values, stored zeros (-0.0), NaN and inf, cancelling pairs, int64 and bool, and form='lo' through its
products; and large inputs for the row and column blocks of the laplacian at 3 and 10 threads (the largest
result 4.9 * 10^5 float64s).

The algorithms scipy lacks (milestone GR5): articulation_points, bridges, biconnected_components and
core_number on the small graphs above and on cycles, bowties, self loops and one-way edges; pagerank (alpha,
personalization, dangling, directed, unweighted, tol, zero-weight rows, self loops) and
betweenness_centrality (directed and undirected, weighted and unweighted, normalized, endpoints, equal
distances and absorbed steps) and their errors; and large inputs past the GIL threshold at 3 and 10 threads:
the structure searches, PageRank's row blocks and split sums (more than 2^16 nodes) and the betweenness's
source tasks and waves (the largest result 7 * 10^4 float64s).
"""

import math

NAN, INF = math.nan, math.inf


def encode(x, L, enc):
    """A csr_array as its CSR arrays (sd.sparse's own golden form)."""
    S = L.sd.sparse
    if isinstance(x, S.sparray):
        return {"sparse": type(x).__name__, "shape": list(x.shape), "dtype": str(x.dtype), "nnz": x.nnz,
                "indptr": enc(x.indptr), "indices": enc(x.indices), "data": enc(x.data)}  # fmt: skip
    return None


def _graphs(L):
    np = L.np
    S = L.sd.sparse
    r = L.rng(2301)
    small = np.round(r.uniform(0, 1, (12, 12)) * (r.random((12, 12)) < 0.25) * 8) / 2  # dyadic weights, ties
    two_way = np.zeros((6, 6))
    two_way[0, 1], two_way[1, 0], two_way[1, 2], two_way[3, 2], two_way[4, 4], two_way[5, 4] = 5, 2, 1, 1, 3, 1
    special = np.array([[0, 1, NAN, INF, 0], [-INF, 0, 2, 0, 0], [0, 0, 0, 1, -1], [3, 0, 0, 0, 0], [0, 0, 0, 0, 0.0]])
    zeros = S.coo_array((np.array([0.0, 0.0, NAN, 1.0, -0.0, 0.0, 2.0]),
                         (np.array([0, 1, 2, 3, 3, 4, 4]), np.array([1, 2, 3, 2, 4, 3, 0]))), shape=(6, 6))  # fmt: skip
    dag = np.triu(r.random((15, 15)) < 0.3, 1)
    perm = r.permutation(15)
    out = {
        "small": S.csr_array(small),
        "small dense": L.arr(small),
        "small list": small.tolist(),
        "small csc": S.csc_array(small),
        "small coo": S.coo_array(small),
        "small dok": S.dok_array(small),
        "small bool": L.arr(small != 0),
        "small int64": S.csr_array((small * 2).astype(np.int64)),
        "two way": S.csr_array(two_way),
        "special dense": special,
        "zeros": zeros,
        "dag": L.arr(dag[np.ix_(perm, perm)]),
        "empty0": S.csr_array((0, 0)),
        "one": np.zeros((1, 1)),
    }
    return out


def _large(L):
    """About 2.7 * 10^5 stored entries over 4 * 10^4 nodes (local edges: many components): the parallel
    union-find; and a symmetric graph read undirected."""
    np = L.np
    r = L.rng(2302)
    n, d = 40000, 8
    rows = np.repeat(np.arange(n), d)
    cols = (rows + r.integers(-60, 61, n * d)) % n
    keep = r.random(n * d) < 0.85
    A = L.sd.sparse.coo_array((r.uniform(0.5, 1.5, int(keep.sum())), (rows[keep], cols[keep])), shape=(n, n)).tocsr()
    return A


def cases(L):
    sd, np = L.sd, L.np
    GR = sd.graph
    graphs = _graphs(L)
    for name, A in graphs.items():
        yield f"connected_components {name}", lambda A=A: GR.connected_components(A)
        yield f"connected_components strong {name}", lambda A=A: GR.connected_components(A, connection="strong")
        yield f"connected_components undirected {name}", lambda A=A: GR.connected_components(A, directed=False)
        yield f"topological_sort {name}", lambda A=A: GR.topological_sort(A)
        yield f"csgraph_to_dense {name}", lambda A=A: GR.csgraph_to_dense(A, null_value=-1.5)
        n = 0 if name == "empty0" else (len(A) if isinstance(A, list) else A.shape[0])
        for s in sorted({0, n // 2, n - 1}) if n else []:
            for directed in (True, False):
                tag = f"{name} {s}{'' if directed else ' undirected'}"
                for f in ("breadth_first_order", "depth_first_order", "breadth_first_tree", "depth_first_tree"):
                    yield f"{f} {tag}", lambda A=A, s=s, d=directed, f=f: getattr(GR, f)(A, s, directed=d)
    yield "connected_components no labels", lambda: GR.connected_components(graphs["small"], return_labels=False)
    yield (
        "breadth_first_order no predecessors",
        lambda: GR.breadth_first_order(graphs["small"], 1, return_predecessors=False),
    )

    special = graphs["special dense"]
    for null in (0, -1, 2.0, INF, NAN, None):
        for nan_null in (True, False):
            for inf_null in (True, False):
                yield (
                    f"csgraph_from_dense {null} {nan_null} {inf_null}",
                    lambda null=null, a=nan_null, b=inf_null: GR.csgraph_from_dense(
                        special, null_value=null, nan_null=a, infinity_null=b
                    ),
                )

    # large: the parallel union-find, the dense conversion's row blocks, the calls without the GIL
    big = _large(L)
    yield "large connected_components", lambda: GR.connected_components(big)
    yield "large connected_components strong", lambda: GR.connected_components(big, connection="strong")
    yield "large connected_components undirected", lambda: GR.connected_components(big, directed=False)
    yield "large breadth_first_order undirected", lambda: GR.breadth_first_order(big, 5, directed=False)
    yield "large depth_first_order", lambda: GR.depth_first_order(big, 5)
    yield "large depth_first_tree undirected", lambda: GR.depth_first_tree(big, 5, directed=False)
    r = L.rng(2303)
    D = r.uniform(0, 1, (700, 700))
    D[D < 0.6] = 0.0
    D[r.random(D.shape) < 0.01] = NAN
    yield "large csgraph_from_dense", lambda: GR.csgraph_from_dense(D)
    yield "large csgraph_to_dense", lambda: GR.csgraph_to_dense(D, null_value=INF)
    yield "large connected_components dense", lambda: GR.connected_components(D)
    yield "large topological_sort", lambda: GR.topological_sort(np.triu(D, 1) != 0)

    yield from _paths_cases(L, graphs)
    yield from _tree_flow_cases(L, graphs)
    yield from _ordering_cases(L, graphs)
    yield from _extra_cases(L, graphs)

    # wrong calls
    small = graphs["small"]
    yield "error complex", lambda: GR.connected_components(np.eye(2) * 1j)
    yield "error not square", lambda: GR.breadth_first_order(np.zeros((2, 3)), 0)
    yield "error 1-D", lambda: GR.topological_sort([1.0, 2.0])
    yield "error scipy sparse", lambda: GR.connected_components(__import__("scipy.sparse", fromlist=["x"]).eye_array(3))
    yield "error i_start float", lambda: GR.breadth_first_order(small, 1.0)
    yield "error i_start bool", lambda: GR.depth_first_tree(small, True)
    yield "error i_start range", lambda: GR.depth_first_order(small, 12)
    yield "error i_start negative", lambda: GR.breadth_first_tree(small, -1)
    yield "error connection", lambda: GR.connected_components(small, connection="both")
    yield "error directed", lambda: GR.connected_components(small, directed=1)
    yield "error positional option", lambda: GR.connected_components(small, False)
    yield "error cycle", lambda: GR.topological_sort(small)
    yield "error null_value", lambda: GR.csgraph_to_dense(small, null_value="x")
    yield "error from_dense sparse", lambda: GR.csgraph_from_dense(small)


def _paths_cases(L, graphs):
    """The shortest paths (milestone GR2)."""
    sd, np = L.sd, L.np
    GR = sd.graph
    S = sd.sparse
    r = L.rng(2304)
    neg = np.round(r.uniform(-1, 3, (10, 10)) * (r.random((10, 10)) < 0.3) * 4) / 4
    order = r.permutation(10)
    neg = neg[np.ix_(order, order)] * np.triu(np.ones((10, 10)), 1)[np.ix_(order, order)]  # a DAG: no cycle
    cyc = np.zeros((8, 8))
    cyc[0, 1], cyc[1, 2], cyc[2, 3], cyc[3, 1], cyc[5, 6], cyc[6, 5], cyc[4, 5] = 1, 1, -3, 1, -1, 0.5, 1
    zero = S.coo_array((np.array([0.0, 0.0, 1.0, 0.0, 2.0, -0.0]),
                        (np.array([0, 1, 2, 2, 3, 4]), np.array([1, 0, 3, 4, 4, 5]))), shape=(6, 6))  # fmt: skip
    grid = S.csr_array(
        L.np.kron(np.eye(4), np.eye(4, k=1) + np.eye(4, k=-1)) + np.kron(np.eye(4, k=1) + np.eye(4, k=-1), np.eye(4))
    )
    pg = {
        "small": graphs["small"],
        "small dense": graphs["small dense"],
        "small list": graphs["small list"],
        "small bool": graphs["small bool"],
        "small int64": graphs["small int64"],
        "two way": graphs["two way"],
        "special dense": graphs["special dense"],
        "zeros": zero,
        "grid": grid,
        "negative dag": S.csr_array(neg),
        "empty0": graphs["empty0"],
        "one": graphs["one"],
    }
    for name, A in pg.items():
        n = 0 if name == "empty0" else (len(A) if isinstance(A, list) else A.shape[0])
        for directed in (True, False):
            for unweighted in (False, True):
                tag = f"{name}{'' if directed else ' undirected'}{' unweighted' if unweighted else ''}"
                kw = {"directed": directed, "unweighted": unweighted}
                for m in ("auto", "D", "BF", "J", "FW"):
                    yield (
                        f"shortest_path {m} {tag}",
                        lambda A=A, m=m, kw=kw: GR.shortest_path(A, method=m, return_predecessors=True, **kw),
                    )
                yield f"floyd_warshall {tag}", lambda A=A, kw=kw: GR.floyd_warshall(A, **kw)
                if n:
                    yield f"dijkstra index {tag}", lambda A=A, kw=kw, n=n: GR.dijkstra(A, indices=n - 1, **kw)
                    yield (
                        f"bellman_ford indices {tag}",
                        lambda A=A, kw=kw, n=n: GR.bellman_ford(
                            A, indices=[0, n // 2, 0], return_predecessors=True, **kw
                        ),
                    )
                    yield (
                        f"johnson indices {tag}",
                        lambda A=A, kw=kw, n=n: GR.johnson(
                            A, indices=np.arange(n)[::2], return_predecessors=True, **kw
                        ),
                    )
                    yield f"yen {tag}", lambda A=A, kw=kw, n=n: GR.yen(A, 0, n - 1, 5, return_predecessors=True, **kw)
                    yield (
                        f"dijkstra min_only {tag}",
                        lambda A=A, kw=kw, n=n: GR.dijkstra(
                            A, indices=[0, n - 1], min_only=True, return_predecessors=True, **kw
                        ),
                    )
                    for limit in (0, 1, 1.5, 2.5):
                        yield (
                            f"dijkstra limit {limit} {tag}",
                            lambda A=A, kw=kw, limit=limit: GR.dijkstra(
                                A, indices=0, limit=limit, return_predecessors=True, **kw
                            ),
                        )
            if n:
                yield (
                    f"construct_dist_matrix {name}{'' if directed else ' undirected'}",
                    lambda A=A, d=directed: GR.construct_dist_matrix(
                        A,
                        GR.shortest_path(A, method="BF", directed=d, return_predecessors=True).predecessors,
                        directed=d,
                    ),
                )
                yield (
                    f"reconstruct_path {name}{'' if directed else ' undirected'}",
                    lambda A=A, d=directed: GR.reconstruct_path(
                        A,
                        GR.shortest_path(A, method="BF", indices=0, directed=d, return_predecessors=True).predecessors,
                        directed=d,
                    ),
                )
    yield "dijkstra indices empty", lambda: GR.dijkstra(pg["small"], indices=[])
    yield "dijkstra min_only all", lambda: GR.dijkstra(pg["small"], min_only=True, return_predecessors=True)
    yield "yen source is sink", lambda: GR.yen(pg["small"], 3, 3, 4, return_predecessors=True)
    yield "yen K 0", lambda: GR.yen(pg["small"], 0, 5, 0, return_predecessors=True)
    yield (
        "construct_dist_matrix null_value",
        lambda: GR.construct_dist_matrix(
            pg["small"], GR.shortest_path(pg["small"], return_predecessors=True).predecessors, null_value=-2.5
        ),
    )

    # negative cycles: NegativeCycleError when the sources reach one, not otherwise
    for m in ("auto", "BF", "J", "FW"):
        yield f"NegativeCycleError shortest_path {m}", lambda m=m: GR.shortest_path(cyc, method=m)
    yield "NegativeCycleError bellman_ford reachable", lambda: GR.bellman_ford(cyc, indices=0)
    yield "NegativeCycleError johnson reachable", lambda: GR.johnson(cyc, indices=[7, 4])
    yield "bellman_ford unreachable cycle", lambda: GR.bellman_ford(cyc, indices=7, return_predecessors=True)
    yield "johnson unreachable cycle", lambda: GR.johnson(cyc, indices=7, return_predecessors=True)
    yield "NegativeCycleError yen", lambda: GR.yen(cyc, 0, 3, 2)
    yield "NegativeCycleError undirected edge", lambda: GR.bellman_ford(pg["negative dag"], directed=False)
    loop = S.csr_array(np.array([[0, 1.0, 0], [0, -0.5, 0], [0, 0, 0]]))
    yield "NegativeCycleError self loop", lambda: GR.johnson(loop, indices=0)

    # large: the source tasks, the Floyd-Warshall blocks, the dense Dijkstra, construct's rows
    r = L.rng(2305)
    n = 6000
    rows = np.repeat(np.arange(n), 8)
    cols = (rows + r.integers(1, 400, n * 8)) % n
    w = r.uniform(0.01, 1.01, n * 8)
    big = S.coo_array((w, (rows, cols)), shape=(n, n)).tocsr()
    pot = r.uniform(0, 2, n)
    bigneg = S.coo_array((w + pot[rows] - pot[cols], (rows, cols)), shape=(n, n)).tocsr()
    srcs = np.arange(0, n, 75)
    for m in ("D", "BF", "J", "auto"):
        yield (
            f"large shortest_path {m}",
            lambda m=m: GR.shortest_path(big, method=m, indices=srcs, return_predecessors=True),
        )
    yield "large johnson negative", lambda: GR.johnson(bigneg, indices=srcs, return_predecessors=True)
    yield "large bellman_ford negative undirected", lambda: GR.bellman_ford(bigneg[:300, :300], directed=False)
    yield "large dijkstra min_only", lambda: GR.dijkstra(big, indices=srcs, min_only=True, return_predecessors=True)
    yield "large dijkstra limit", lambda: GR.dijkstra(big, indices=srcs, limit=1.0)
    yield "large yen", lambda: GR.yen(big, 0, n // 2, 8, return_predecessors=True)
    sub = big[:700, :700]
    yield "large all pairs", lambda: GR.shortest_path(sub, return_predecessors=True)
    yield (
        "large construct_dist_matrix",
        lambda: GR.construct_dist_matrix(sub, GR.shortest_path(sub, method="J", return_predecessors=True).predecessors),
    )
    D = r.uniform(0.01, 1.01, (330, 330))
    D[r.random(D.shape) < 0.4] = 0.0
    yield "large floyd_warshall", lambda: GR.floyd_warshall(D, return_predecessors=True)
    yield "large floyd_warshall undirected", lambda: GR.floyd_warshall(D, directed=False)
    yield "large shortest_path auto dense", lambda: GR.shortest_path(D, return_predecessors=True)

    # wrong calls
    small = pg["small"]
    yield "error dijkstra negative", lambda: GR.dijkstra(pg["negative dag"])
    yield "error shortest_path D negative", lambda: GR.shortest_path(pg["negative dag"], method="D")
    yield "error method", lambda: GR.shortest_path(small, method="BFS")
    yield "error method type", lambda: GR.shortest_path(small, method=1)
    yield "error FW indices", lambda: GR.shortest_path(small, method="FW", indices=0)
    yield "error indices float", lambda: GR.dijkstra(small, indices=1.0)
    yield "error indices 2-D", lambda: GR.dijkstra(small, indices=[[0, 1]])
    yield "error indices range", lambda: GR.johnson(small, indices=[0, 12])
    yield "error limit nan", lambda: GR.dijkstra(small, limit=float("nan"))
    yield "error limit negative", lambda: GR.dijkstra(small, limit=-1)
    yield "error stored nan", lambda: GR.bellman_ford(graphs["zeros"])
    yield "error yen K", lambda: GR.yen(small, 0, 1, -2)
    yield "error yen sink", lambda: GR.yen(small, 0, 12, 1)
    yield "error construct shape", lambda: GR.construct_dist_matrix(small, [[0]])
    yield (
        "error construct edge",
        lambda: GR.construct_dist_matrix(S.csr_array(np.eye(3, k=1)), [[-9999, 0, 0], [-9999] * 3, [-9999] * 3]),
    )
    yield (
        "error construct cycle",
        lambda: GR.construct_dist_matrix(
            S.csr_array(np.eye(3, k=1) + np.eye(3, k=-1)), [[-9999, 2, 1], [-9999] * 3, [-9999] * 3]
        ),
    )
    yield "error reconstruct self", lambda: GR.reconstruct_path(small, [0] + [-9999] * 11)
    yield "error positional option dijkstra", lambda: GR.dijkstra(small, False)


def _tree_flow_cases(L, graphs):
    """The spanning tree, flow, matchings and assignment (milestone GR3)."""
    sd, np = L.sd, L.np
    GR = sd.graph
    S = sd.sparse
    for name, A in graphs.items():
        yield f"minimum_spanning_tree {name}", lambda A=A: GR.minimum_spanning_tree(A)
        yield f"maximum_bipartite_matching {name}", lambda A=A: GR.maximum_bipartite_matching(A)
    r = L.rng(2306)
    ties = np.round(r.uniform(-2, 2, (14, 14)) * 2) / 2 * (r.random((14, 14)) < 0.4)
    two = np.zeros((5, 5))
    two[0, 1], two[1, 0], two[1, 2], two[2, 1], two[3, 4], two[2, 3] = 4, 2, 3, 3, -1, 1
    negz = S.coo_array(
        (np.array([0.0, -0.0, -1.0, 2.0]), (np.array([0, 1, 1, 2]), np.array([1, 0, 2, 0]))), shape=(3, 3)
    )
    for name, A in (("ties", ties), ("two way 5", two), ("signed zeros", negz), ("ties csr", S.csr_array(ties))):
        yield f"minimum_spanning_tree {name}", lambda A=A: GR.minimum_spanning_tree(A)

    caps = (r.integers(1, 9, (12, 12)) * (r.random((12, 12)) < 0.35)).astype(np.int64)
    caps[3, 3] = 4  # a self loop
    zero_caps = S.coo_array(
        (np.array([0, 3, 2, 0], np.int64), (np.array([0, 1, 0, 2]), np.array([1, 2, 2, 0]))), shape=(3, 3)
    )
    big = np.zeros((6, 6), np.int64)
    big[0, 1] = big[0, 2] = 2**61
    big[1, 3] = big[2, 4] = big[3, 5] = big[4, 5] = 2**62
    big[1, 2] = big[2, 1] = 2**63 - 1
    past = np.zeros((4, 4), np.int64)
    past[0, 1], past[1, 3], past[0, 2], past[2, 3] = 2**40, 2**40, 2**32 + 5, 2**32 + 5
    complete = np.ones((6, 6), np.int64) - np.eye(6, dtype=np.int64)
    nets = {
        "random": caps, "random csr": S.csr_array(caps), "bool": caps > 4, "stored zeros": zero_caps,
        "near 2^62": big, "past int32": S.csr_array(past), "ties": complete,
    }  # fmt: skip
    for name, A in nets.items():
        n = A.shape[0]
        for method in ("dinic", "edmonds_karp"):
            yield f"maximum_flow {method} {name}", lambda A=A, m=method, n=n: GR.maximum_flow(A, 0, n - 1, method=m)
    yield "maximum_flow middle", lambda: GR.maximum_flow(caps, 5, 2)

    stored0 = S.coo_array(
        (np.r_[np.zeros(4), np.ones(3)], (np.r_[0, 1, 2, 3, 0, 1, 2], np.r_[0, 1, 2, 3, 1, 2, 3])), shape=(4, 4)
    )  # a perfect matching of stored zeros
    bip = (r.random((9, 13)) < 0.3) * r.integers(1, 5, (9, 13)).astype(np.float64)
    for name, A in (("9x13", bip), ("13x9", bip.T.copy()), ("9x13 csr", S.csr_array(bip))):
        for perm in ("row", "column"):
            yield (
                f"maximum_bipartite_matching {name} {perm}",
                lambda A=A, p=perm: GR.maximum_bipartite_matching(A, perm_type=p),
            )
    for name, A in (
        ("9x13", bip),
        ("13x9", bip.T.copy()),
        ("ties", np.ones((6, 6))),
        ("zeros", stored0),
    ):
        for maximize in (False, True):
            yield (
                f"min_weight_full_bipartite_matching {name} {maximize}",
                lambda A=A, m=maximize: GR.min_weight_full_bipartite_matching(A, maximize=m),
            )
    costs = {
        "int": r.integers(0, 9, (10, 10)).astype(np.float64),
        "rect wide": r.integers(0, 5, (6, 11)),
        "rect tall": r.integers(0, 5, (11, 6)),
        "float": r.uniform(-1, 1, (9, 9)),
        "ties": np.ones((7, 7)),
        "inf": np.where(r.random((8, 8)) < 0.4, np.inf, r.integers(0, 9, (8, 8))),
        "bool": r.random((5, 5)) < 0.5,
        "empty": np.zeros((0, 4)),
    }
    for name, C in costs.items():
        for maximize in (False, True):
            yield (
                f"linear_sum_assignment {name} {maximize}",
                lambda C=C, m=maximize: GR.linear_sum_assignment(C, maximize=m),
            )
    yield (
        "linear_sum_assignment -inf maximize",
        lambda: GR.linear_sum_assignment(np.where(np.isinf(costs["inf"]), -np.inf, costs["inf"]), maximize=True),
    )

    # large: past the GIL threshold (serial algorithms; the same bytes at 3 and 10 threads)
    n = 40000
    rows = np.repeat(np.arange(n), 4)
    cols = (rows + r.integers(1, 2000, n * 4)) % n
    w = np.round(r.uniform(0, 4, n * 4) * 4) / 4
    A = S.coo_array((w, (rows, cols)), shape=(n, n)).tocsr()
    yield "large minimum_spanning_tree", lambda: GR.minimum_spanning_tree(A)
    yield "large maximum_bipartite_matching", lambda: GR.maximum_bipartite_matching(A)
    F = S.coo_array((r.integers(1, 60, n * 4), (rows, cols)), shape=(n, n)).tocsr()
    for method in ("dinic", "edmonds_karp"):
        yield f"large maximum_flow {method}", lambda m=method: GR.maximum_flow(F[:6000, :6000], 0, 5999, method=m)
    C = r.integers(0, 40, (300, 300)).astype(np.float64)
    yield "large linear_sum_assignment", lambda: GR.linear_sum_assignment(C)
    m = 2500
    M = S.coo_array(
        (r.uniform(0, 1, m * 8), (np.repeat(np.arange(m), 8), r.integers(0, m, m * 8))), shape=(m, m)
    ).tocsr()
    M = M + S.eye_array(m) * 3.0
    yield "large min_weight_full_bipartite_matching", lambda: GR.min_weight_full_bipartite_matching(M)

    # wrong calls
    yield "error mst stored nan", lambda: GR.minimum_spanning_tree(graphs["zeros"])
    yield "error flow float", lambda: GR.maximum_flow(caps.astype(np.float64), 0, 1)
    yield "error flow negative", lambda: GR.maximum_flow(-caps, 0, 1)
    yield "error flow overflow", lambda: GR.maximum_flow(np.array([[0, 2**62, 2**62], [0, 0, 1], [0, 0, 0]]), 0, 2)
    yield "error flow source is sink", lambda: GR.maximum_flow(caps, 1, 1)
    yield "error flow method", lambda: GR.maximum_flow(caps, 0, 1, method="preflow")
    yield "error matching perm_type", lambda: GR.maximum_bipartite_matching(bip, perm_type="rows")
    yield "error no full matching", lambda: GR.min_weight_full_bipartite_matching(np.array([[1.0, 0], [2.0, 0]]))
    yield "error lsa infeasible", lambda: GR.linear_sum_assignment(np.array([[1.0, np.inf], [2.0, np.inf]]))
    yield "error lsa nan", lambda: GR.linear_sum_assignment(np.array([[np.nan]]))
    yield "error lsa inf maximize", lambda: GR.linear_sum_assignment(np.array([[np.inf, 1.0]]), maximize=True)
    yield "error lsa sparse", lambda: GR.linear_sum_assignment(S.csr_array(np.eye(2)))


def _ordering_cases(L, graphs):
    """The ordering, structural rank and laplacian (milestone GR4)."""
    sd, np = L.sd, L.np
    GR = sd.graph
    S = sd.sparse
    for name, A in graphs.items():
        for mode in (False, True):
            yield (
                f"reverse_cuthill_mckee {name} {mode}",
                lambda A=A, m=mode: GR.reverse_cuthill_mckee(A, symmetric_mode=m),
            )
        yield f"structural_rank {name}", lambda A=A: GR.structural_rank(A)
        for normed in (False, True):
            for out in (False, True):
                for sym in (False, True):
                    yield (
                        f"laplacian {name} {normed} {out} {sym}",
                        lambda A=A, a=normed, b=out, c=sym: GR.laplacian(
                            A, normed=a, use_out_degree=b, symmetrized=c, return_diag=True
                        ),
                    )
    r = L.rng(2307)
    ties = (r.random((30, 30)) < 0.12) | (r.random((30, 30)) < 0.12).T  # many equal degrees
    rect = (r.random((7, 12)) < 0.3) * 1.0
    yield "reverse_cuthill_mckee ties", lambda: GR.reverse_cuthill_mckee(ties)
    yield "reverse_cuthill_mckee ties csr", lambda: GR.reverse_cuthill_mckee(S.csr_array(ties), symmetric_mode=True)
    yield "structural_rank 7x12", lambda: GR.structural_rank(rect)
    yield "structural_rank 12x7 csr", lambda: GR.structural_rank(S.csr_array(rect.T.copy()))
    signed = np.round(r.uniform(-3, 3, (9, 9)) * 2) / 2
    signed[r.random((9, 9)) < 0.4] = 0.0
    cancel = S.coo_array(
        (np.array([1.0, -1.0, 0.0, 2.5, NAN, INF]), (np.array([0, 1, 2, 3, 4, 5]), np.array([1, 0, 3, 2, 5, 4]))),
        shape=(6, 6),
    )
    ints = r.integers(-5, 6, (8, 8))
    for name, A in (("signed", signed), ("cancel", cancel), ("int", ints), ("int csr", S.csr_array(ints))):
        for normed in (False, True):
            for sym in (False, True):
                yield (
                    f"laplacian {name} {normed} {sym}",
                    lambda A=A, a=normed, c=sym: GR.laplacian(A, normed=a, symmetrized=c, return_diag=True),
                )
    yield "laplacian dtype float64", lambda: GR.laplacian(ints, dtype="float64", return_diag=True)
    yield "laplacian dtype int64 normed", lambda: GR.laplacian(signed, normed=True, dtype=sd.int64)
    x = r.normal(size=9)
    for name, A in (("dense", signed), ("csr", S.csr_array(signed))):
        for normed in (False, True):
            yield (
                f"laplacian lo {name} {normed}",
                lambda A=A, a=normed: GR.laplacian(A, normed=a, symmetrized=True, form="lo") @ L.arr(x),
            )

    # large: the row blocks and column blocks past the GIL threshold
    n = 40000
    rows = np.repeat(np.arange(n), 5)
    cols = (rows + r.integers(1, 3000, n * 5)) % n
    A = S.coo_array((r.uniform(0.1, 2, n * 5), (rows, cols)), shape=(n, n)).tocsr()
    yield "large reverse_cuthill_mckee", lambda: GR.reverse_cuthill_mckee(A)
    yield "large structural_rank", lambda: GR.structural_rank(A)
    for normed in (False, True):
        for out in (False, True):
            for sym in (False, True):
                yield (
                    f"large laplacian {normed} {out} {sym}",
                    lambda a=normed, b=out, c=sym: GR.laplacian(
                        A, normed=a, use_out_degree=b, symmetrized=c, return_diag=True
                    ),
                )
    D = r.uniform(-1, 1, (700, 700))
    D[r.random(D.shape) < 0.5] = 0.0
    for normed in (False, True):
        for out in (False, True):
            yield (
                f"large laplacian dense {normed} {out}",
                lambda a=normed, b=out: GR.laplacian(np.abs(D) if a else D, normed=a, use_out_degree=b),
            )
    yield "large laplacian dense symmetrized", lambda: GR.laplacian(D, symmetrized=True, return_diag=True)

    # wrong calls
    yield "error rcm not square", lambda: GR.reverse_cuthill_mckee(np.zeros((2, 3)))
    yield "error rcm symmetric_mode", lambda: GR.reverse_cuthill_mckee(np.eye(2), symmetric_mode=None)
    yield "error structural_rank 1-D", lambda: GR.structural_rank(np.zeros(3))
    yield "error laplacian form", lambda: GR.laplacian(np.eye(2), form="function")
    yield "error laplacian copy", lambda: GR.laplacian(np.eye(2), copy=True)
    yield "error laplacian dtype", lambda: GR.laplacian(np.eye(2), dtype="bool")
    yield "error laplacian not square", lambda: GR.laplacian(np.zeros((3, 2)))


def _extra_cases(L, graphs):
    """The articulation points, bridges, biconnected components, core numbers, PageRank and betweenness
    (milestone GR5)."""
    sd, np = L.sd, L.np
    GR = sd.graph
    S = sd.sparse
    for name, A in graphs.items():
        for f in ("articulation_points", "bridges", "biconnected_components", "core_number"):
            yield f"{f} {name}", lambda A=A, f=f: getattr(GR, f)(A)
        for directed in (True, False):
            for unweighted in (True, False):
                tag = f"{name} {directed} {unweighted}"
                yield f"pagerank {tag}", lambda A=A, a=directed, b=unweighted: GR.pagerank(A, directed=a, unweighted=b)
                yield (
                    f"betweenness_centrality {tag}",
                    lambda A=A, a=directed, b=unweighted: GR.betweenness_centrality(A, directed=a, unweighted=b),
                )
    r = L.rng(2311)
    cycle = np.roll(np.eye(9), 1, axis=1)
    bow = np.zeros((7, 7))
    for i, j in ((0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (2, 4), (4, 5), (5, 6)):
        bow[i, j] = 1.0
    loops = np.eye(5) + np.diag(np.ones(4), 1)
    for name, A in (("cycle", cycle), ("bow", bow), ("loops", loops), ("bow csr", S.csr_array(bow.T.copy()))):
        for f in ("articulation_points", "bridges", "biconnected_components", "core_number"):
            yield f"{f} {name}", lambda A=A, f=f: getattr(GR, f)(A)
    w = np.round(r.uniform(0, 3, (14, 14)) * (r.random((14, 14)) < 0.3) * 2) / 2  # dyadic: many ties
    p = r.uniform(0, 1, 14)
    p[[2, 5, 7]] = 0.0
    for alpha in (0.0, 0.5, 0.85, 0.99):
        yield f"pagerank alpha {alpha}", lambda a=alpha: GR.pagerank(w, alpha=a)
    yield "pagerank personalization", lambda: GR.pagerank(w, personalization=p)
    yield "pagerank dangling", lambda: GR.pagerank(w, dangling=p[::-1].copy())
    yield "pagerank both ints", lambda: GR.pagerank(w, personalization=[1] * 7 + [0] * 7, dangling=list(range(14)))
    yield "pagerank tol", lambda: GR.pagerank(w, tol=1e-4)
    zero_row = S.csr_array((np.array([0.0, 2.0, 1.0]), np.array([1, 2, 0]), np.array([0, 1, 2, 3])), shape=(3, 3))
    yield "pagerank zero row", lambda: GR.pagerank(zero_row)
    yield "pagerank self loop", lambda: GR.pagerank(np.array([[1.0, 0], [1.0, 0]]))
    yield "pagerank empty", lambda: GR.pagerank(np.zeros((0, 0)))
    for normalized in (False, True):
        for endpoints in (False, True):
            for directed in (True, False):
                yield (
                    f"betweenness_centrality ties {normalized} {endpoints} {directed}",
                    lambda a=normalized, b=endpoints, c=directed: GR.betweenness_centrality(
                        w, normalized=a, endpoints=b, directed=c
                    ),
                )
    absorbed = np.zeros((5, 5))
    absorbed[0, 3], absorbed[3, 1], absorbed[1, 2], absorbed[2, 4], absorbed[0, 4] = 1e17, 1.0, 1.0, 1.0, 1e17
    yield "betweenness_centrality absorbed", lambda: GR.betweenness_centrality(absorbed, normalized=False)
    yield "betweenness_centrality two", lambda: GR.betweenness_centrality(np.ones((2, 2)), endpoints=True)

    # large: the structure past the GIL threshold, PageRank's row blocks and split sums, the betweenness's
    # source tasks and waves
    n = 70000
    rows = np.repeat(np.arange(n), 3)
    cols = (rows + r.integers(1, 5000, n * 3)) % n
    A = S.coo_array((r.uniform(0.1, 2, n * 3), (rows, cols)), shape=(n, n)).tocsr()
    for f in ("articulation_points", "bridges", "biconnected_components", "core_number"):
        yield f"large {f}", lambda f=f: getattr(GR, f)(A)
    for directed in (True, False):
        yield f"large pagerank {directed}", lambda a=directed: GR.pagerank(A, directed=a)
    pl = r.uniform(0, 1, n)
    yield "large pagerank unweighted personalization", lambda: GR.pagerank(A, unweighted=True, personalization=pl)
    m = 1500
    rows = np.repeat(np.arange(m), 4)
    cols = (rows + r.integers(1, 200, m * 4)) % m
    B = S.coo_array((np.round(r.uniform(0.5, 2, m * 4) * 4) / 4, (rows, cols)), shape=(m, m)).tocsr()
    for directed in (True, False):
        for unweighted in (True, False):
            yield (
                f"large betweenness_centrality {directed} {unweighted}",
                lambda a=directed, b=unweighted: GR.betweenness_centrality(B, directed=a, unweighted=b, endpoints=a),
            )

    # wrong calls
    yield "error pagerank alpha", lambda: GR.pagerank(w, alpha=1.0)
    yield "error pagerank tol", lambda: GR.pagerank(w, tol=0)
    yield "error pagerank personalization length", lambda: GR.pagerank(w, personalization=[1.0, 2.0])
    yield "error pagerank personalization sum", lambda: GR.pagerank(w, personalization=np.zeros(14))
    yield "error pagerank negative", lambda: GR.pagerank(-w)
    yield "error pagerank max_iter", lambda: GR.pagerank(w, max_iter=100)
    yield "error betweenness zero weight", lambda: GR.betweenness_centrality(zero_row)
    yield "error betweenness k", lambda: GR.betweenness_centrality(w, k=3)
    yield "error betweenness normalized", lambda: GR.betweenness_centrality(w, normalized=1)
    yield "error bridges not square", lambda: GR.bridges(np.zeros((2, 3)))
    yield "error core_number complex", lambda: GR.core_number(np.eye(2) * 1j)
