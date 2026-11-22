"""A dense input of any memory layout reads as the same graph (docs/api/graph.md §1.2): a column slice, an
F-ordered array, reversed strides and a transposed view give the bytes of their C-ordered copy for every
function that takes a dense matrix (GR1, GR2, GR3), sd arrays and numpy arrays alike. A strided view once
reached the native conversion as it was (a reshape that returned a view instead of a copy): an internal
TypeError."""

import numpy as np
import pytest

import stardust as sd

GR = sd.graph


def _base(dtype):
    r = np.random.Generator(np.random.PCG64(41))
    w = np.round(r.uniform(0.5, 4, (12, 24)) * 2) / 2 * (r.random((12, 24)) < 0.45)
    if dtype == "bool":
        return w != 0
    if dtype == "int64":
        return (w * 2).astype(np.int64)
    return w


def _views(dtype):
    """{name: (strided view, its C-ordered copy)} of 12 x 12 square matrices, sd and numpy."""
    w = _base(dtype)
    out = {}
    for lib, wrap in (("sd", sd.asarray), ("numpy", lambda a: a)):
        b = wrap(w)
        out[f"{lib} columns"] = b[:, ::2]
        out[f"{lib} reversed"] = b[::-1, ::-2]
        out[f"{lib} transposed"] = b[:, 5:17].T
        out[f"{lib} fortran"] = wrap(np.asfortranarray(w[:, 3:15]))
    return {k: (v, np.ascontiguousarray(np.asarray(v))) for k, v in out.items()}


def _out(f, A):
    """f(A) as bytes, or its ValueError (a graph with no full matching)."""
    try:
        return _bytes(f(A))
    except ValueError as e:
        return [repr(e).encode()]


def _bytes(x):
    if isinstance(x, sd.sparse.sparray):
        return [np.asarray(a).tobytes() for a in (x.indptr, x.indices, x.data)]
    if isinstance(x, tuple):
        return [b for y in x for b in _bytes(y)]
    if isinstance(x, int):
        return [str(x).encode()]
    return [np.asarray(x).tobytes()]


CALLS = {
    "connected_components": lambda A: GR.connected_components(A),
    "connected_components strong": lambda A: GR.connected_components(A, connection="strong"),
    "breadth_first_order": lambda A: GR.breadth_first_order(A, 0, directed=False),
    "depth_first_order": lambda A: GR.depth_first_order(A, 3),
    "breadth_first_tree": lambda A: GR.breadth_first_tree(A, 1),
    "depth_first_tree": lambda A: GR.depth_first_tree(A, 2, directed=False),
    "csgraph_from_dense": lambda A: GR.csgraph_from_dense(A),
    "csgraph_to_dense": lambda A: GR.csgraph_to_dense(A),
    "shortest_path": lambda A: GR.shortest_path(A, return_predecessors=True),
    "dijkstra": lambda A: GR.dijkstra(A, indices=[0, 4], return_predecessors=True),
    "bellman_ford": lambda A: GR.bellman_ford(A, indices=2, directed=False),
    "johnson": lambda A: GR.johnson(A, return_predecessors=True),
    "floyd_warshall": lambda A: GR.floyd_warshall(A, return_predecessors=True),
    "yen": lambda A: GR.yen(A, 0, 7, 4, return_predecessors=True),
    "minimum_spanning_tree": lambda A: GR.minimum_spanning_tree(A),
    "maximum_bipartite_matching": lambda A: GR.maximum_bipartite_matching(A),
    "min_weight_full_bipartite_matching": lambda A: GR.min_weight_full_bipartite_matching(A),
    "linear_sum_assignment": lambda A: GR.linear_sum_assignment(A),
}
FLOW = {"maximum_flow": lambda A: GR.maximum_flow(A, 0, 11)}


@pytest.mark.parametrize("dtype", ["float64", "int64", "bool"])
@pytest.mark.parametrize("call", sorted(CALLS))
def test_dense_layouts_read_the_same(call, dtype):
    if call == "csgraph_from_dense" and dtype == "bool":
        pytest.skip("csgraph_from_dense compares with a float null_value: bool is covered by float64's pattern")
    f = CALLS[call]
    for name, (view, copy) in _views(dtype).items():
        assert _out(f, view) == _out(f, copy), name


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_dense_layouts_maximum_flow(dtype):
    for name, (view, copy) in _views(dtype).items():
        a, b = FLOW["maximum_flow"](view), FLOW["maximum_flow"](copy)
        assert a.flow_value == b.flow_value, name
        assert _bytes(a.flow) == _bytes(b.flow), name


def test_predecessor_layouts():
    w = _base("float64")[:, :12]
    p = np.asarray(GR.shortest_path(w, method="BF", return_predecessors=True).predecessors)
    want = _bytes(GR.construct_dist_matrix(w, p))
    for q in (np.asfortranarray(p), sd.asarray(np.asfortranarray(p)), sd.asarray(np.repeat(p, 2, axis=1))[:, ::2]):
        assert _bytes(GR.construct_dist_matrix(w, q)) == want
    row = sd.asarray(np.vstack([p[3], p[3]]).T.copy())[:, 0]  # a strided 1-D view
    assert _bytes(GR.reconstruct_path(w, row)) == _bytes(GR.reconstruct_path(w, p[3]))
    idx = sd.arange(12)[::-3]
    assert _bytes(GR.dijkstra(w, indices=idx)) == _bytes(GR.dijkstra(w, indices=[11, 8, 5, 2]))
