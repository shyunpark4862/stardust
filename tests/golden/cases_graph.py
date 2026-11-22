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
