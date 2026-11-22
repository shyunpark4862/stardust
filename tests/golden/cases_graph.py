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
