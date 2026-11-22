"""The shortest paths of sd.graph (docs/api/graph.md §3, docs/design/graph.md §2.3, §3.2; milestone GR2).

The arguments are checked here; the searches are stardust._core.graph's (csrc/graph/paths.cpp, yen.cpp).
A distance is the smallest left-to-right sum of a path's weights (each step rounded, from the source on),
the least fixed point of the relaxation: 'D', 'BF', 'J', 'auto' and dijkstra, bellman_ford and johnson
reach it and give the same bits, and the predecessors follow from the distances alone (the tight edges, the
fewest edges, the smallest node; GR5, GR6). 'FW' and floyd_warshall are the blocked Floyd-Warshall's
rounding (GR25, GR27). 'auto' never takes FW (GR26). A negative cycle reachable from the sources is a
NegativeCycleError for every method, through the smallest node of the cycle one shared check finds.
"""

import math

from .. import _core as _sd
from .. import sparse as _sparse
from .._calls import function
from .._results import MinOnlyResult, ShortestPathResult, YenResult
from ..stats._descriptive import _choice
from . import _common as C
from ._common import NegativeCycleError

_core = _sd.graph
_METHODS = ("auto", "D", "BF", "J", "FW")
_BLOCK = 64  # the blocked Floyd-Warshall's block (csrc/graph/paths.cpp)


def _weights(name, csgraph, unweighted):
    """(n, indptr, indices, data, negative) of a graph whose weights a search reads: data None when
    unweighted (the pattern only; GR34); a stored NaN or +-inf is a ValueError (GR8)."""
    n, indptr, indices, data = C.graph(csgraph, name, values=not unweighted)
    negative = False
    if not unweighted:
        nonfinite, negative = _core._scan(data)
        if nonfinite:
            raise ValueError(f"{name}(): stored edge weights must be finite")
    return n, indptr, indices, data, negative


def _sources(indices, n, name):
    """(int64 sd array of the sources or None for every node, flat): an int (flat) or a 1-D array of ints,
    each a node (GR22); duplicates allowed."""
    if indices is None:
        return None, False
    if isinstance(indices, (list, tuple, range, _sd.ndarray)) or getattr(indices, "ndim", 0) >= 1:
        what = f"{name}(): 'indices' must be an int or a 1-D array of ints"
        if isinstance(indices, _sparse.sparray):
            raise TypeError(f"{what}, not {C.type_name(indices)}")
        try:
            a = _sd.asarray(indices)
        except TypeError, ValueError:
            raise TypeError(f"{what}, not {C.type_name(indices)}") from None
        if a.ndim != 1:
            raise ValueError(f"{what}, not a {a.ndim}-D array")
        if a.size == 0:
            return _sd.zeros(0, dtype=_sd.int64), False
        if a.dtype != _sd.int64:
            raise TypeError(f"{name}(): 'indices' must hold ints, not {a.dtype}")
        bad = (a < 0) | (a >= n)
        if bool(_sd.any(bad)):
            v = int(a[int(_sd.argmax(bad))])
            raise ValueError(f"{name}(): 'indices' must hold nodes of the graph (0 .. {n - 1}), not {v}")
        return C.contiguous(a), False
    return _sd.asarray([C.node(indices, n, name, "indices")]), True


def _paths(name, n, indptr, indices, data, src, flat, code, directed, unweighted, preds, limit, min_only, negative):
    if unweighted:
        code = "U"
    rows = n if src is None else src.size
    out = n if (min_only or flat) else rows * n
    C.room(name, 8 * out * (2 if preds else 1) + (8 * n * n if code == "dense" else 0))
    dist, pred, srcs, cycle = _core._paths(
        indptr, indices, data, n, src, code, directed, preds, limit, min_only, flat, negative
    )
    if cycle >= 0:
        raise NegativeCycleError(f"{name}(): negative cycle through node {cycle}")
    if not preds:
        return dist
    if min_only:
        return MinOnlyResult(C.readonly(dist), C.readonly(pred), C.readonly(srcs))
    return ShortestPathResult(C.readonly(dist), C.readonly(pred))


def _floyd_warshall(name, n, indptr, indices, data, directed, preds):
    padded = -(-n // _BLOCK) * _BLOCK
    per = 2 if preds else 1
    C.room(name, 8 * per * (n * n + (padded * padded if padded != n else 0)))
    dist, pred, cycle = _core._floyd_warshall(indptr, indices, data, n, directed, preds)
    if cycle >= 0:
        raise NegativeCycleError(f"{name}(): negative cycle through node {cycle}")
    if preds:
        return ShortestPathResult(C.readonly(dist), C.readonly(pred))
    return dist


def _flags(name, directed, return_predecessors, unweighted):
    return (
        C.flag(directed, name, "directed"),
        C.flag(return_predecessors, name, "return_predecessors"),
        C.flag(unweighted, name, "unweighted"),
    )


def _no_negative(name, negative, advice):
    if negative:
        raise ValueError(f"{name}(): negative edge weights are not allowed; {advice}")


@function
def shortest_path(
    csgraph, /, *, method="auto", directed=True, return_predecessors=False, unweighted=False, indices=None
):
    """The shortest-path distances from the nodes in indices (None: every node) by method 'auto', 'D',
    'BF', 'J' or 'FW': (n, n), (n,) or (len(indices), n) float64 (inf where unreachable), with
    ShortestPathResult(dist_matrix, predecessors) when return_predecessors (int64; -9999 for the sources
    and the unreachable). Every method but 'FW' gives the same bits; 'auto' is 'J' with a negative weight
    and 'D' otherwise."""
    name = "shortest_path"
    method = _choice(method, name, "method", _METHODS)
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    n, indptr, ix, data, negative = _weights(name, csgraph, unweighted)
    if method == "FW":
        if indices is not None:
            raise ValueError(f"{name}(): 'indices' cannot be given with method='FW'")
        return _floyd_warshall(name, n, indptr, ix, data, directed, preds)
    if method == "D":
        _no_negative(name, negative, "use method='BF', 'J' or 'auto'")
    src, flat = _sources(indices, n, name)
    code = method
    if method == "auto":
        dense = src is None and n > 0 and 4 * ix.size >= n * n
        code = "J" if negative else ("dense" if dense else "D")
    return _paths(name, n, indptr, ix, data, src, flat, code, directed, unweighted, preds, math.inf, False, negative)


@function
def dijkstra(
    csgraph,
    /,
    *,
    directed=True,
    indices=None,
    return_predecessors=False,
    unweighted=False,
    limit=math.inf,
    min_only=False,
):
    """Dijkstra's shortest-path distances (non-negative weights), as shortest_path's; limit: the nodes
    farther than it are unreachable (inf, -9999); min_only: one (n,) distance from all of indices together,
    with MinOnlyResult(dist_matrix, predecessors, sources) when return_predecessors."""
    name = "dijkstra"
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    limit = C.real(limit, name, "limit")
    if math.isnan(limit) or limit < 0:
        raise ValueError(f"{name}(): 'limit' must be a nonnegative number or inf, not {limit!r}")
    min_only = C.flag(min_only, name, "min_only")
    n, indptr, ix, data, negative = _weights(name, csgraph, unweighted)
    _no_negative(name, negative, "use bellman_ford or johnson")
    src, flat = _sources(indices, n, name)
    return _paths(name, n, indptr, ix, data, src, flat, "D", directed, unweighted, preds, limit, min_only, False)


@function
def bellman_ford(csgraph, /, *, directed=True, indices=None, return_predecessors=False, unweighted=False):
    """The shortest-path distances by a queue Bellman-Ford (negative weights allowed), as shortest_path's;
    NegativeCycleError for a negative cycle reachable from indices."""
    name = "bellman_ford"
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    n, indptr, ix, data, negative = _weights(name, csgraph, unweighted)
    src, flat = _sources(indices, n, name)
    return _paths(name, n, indptr, ix, data, src, flat, "BF", directed, unweighted, preds, math.inf, False, negative)


@function
def johnson(csgraph, /, *, directed=True, indices=None, return_predecessors=False, unweighted=False):
    """The shortest-path distances by Johnson's algorithm (negative weights allowed), as shortest_path's;
    NegativeCycleError for a negative cycle reachable from indices."""
    name = "johnson"
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    n, indptr, ix, data, negative = _weights(name, csgraph, unweighted)
    src, flat = _sources(indices, n, name)
    return _paths(name, n, indptr, ix, data, src, flat, "J", directed, unweighted, preds, math.inf, False, negative)


@function
def floyd_warshall(csgraph, /, *, directed=True, return_predecessors=False, unweighted=False):
    """All-pairs distances by the blocked Floyd-Warshall (negative weights allowed): (n, n) float64, with
    ShortestPathResult when return_predecessors (Floyd-Warshall's update rule). Its sums join paths in
    the middle, so the last bits may differ from the other methods' (within the accuracy contract)."""
    name = "floyd_warshall"
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    n, indptr, ix, data, _ = _weights(name, csgraph, unweighted)
    return _floyd_warshall(name, n, indptr, ix, data, directed, preds)


@function
def yen(csgraph, source, sink, K, /, *, directed=True, return_predecessors=False, unweighted=False):
    """The first K loopless source-sink paths in the order (left-to-right sum, edge count, node
    sequence): their lengths (float64 (M,), M <= K), with YenResult(dist_array, predecessors) when
    return_predecessors ((M, n) int64, a path per row)."""
    name = "yen"
    directed, preds, unweighted = _flags(name, directed, return_predecessors, unweighted)
    n, indptr, ix, data, negative = _weights(name, csgraph, unweighted)
    s = C.node(source, n, name, "source")
    t = C.node(sink, n, name, "sink")
    K = C.count(K, name, "K")
    if preds:
        C.room(name, 8 * K * n)
    dist, pred, cycle = _core._yen(indptr, ix, data, n, s, t, min(K, 2**62), directed, preds, negative)
    if cycle >= 0:
        raise NegativeCycleError(f"{name}(): negative cycle through node {cycle}")
    if preds:
        return YenResult(C.readonly(dist), C.readonly(pred))
    return dist


def _predecessors(x, shape, name):
    """The predecessors argument as a C-contiguous int64 sd array of the given shape."""
    what = f"{name}(): 'predecessors' must be an array of ints"
    if isinstance(x, _sparse.sparray) or C._is_scipy_sparse(x):
        raise TypeError(f"{what}, not {C.type_name(x)}")
    try:
        a = _sd.asarray(x)
    except TypeError, ValueError:
        raise TypeError(f"{what}, not {C.type_name(x)}") from None
    if tuple(a.shape) != shape:
        raise ValueError(f"{name}(): 'predecessors' must have the shape {shape}, not {tuple(a.shape)}")
    if a.dtype != _sd.int64:
        if a.size:
            raise TypeError(f"{name}(): 'predecessors' must hold ints, not {a.dtype}")
        a = _sd.zeros(shape, dtype=_sd.int64)
    return C.contiguous(a)


def _pred_error(name, n, err, row=True):
    if row:
        kind, i, j, v = err
        at, path = f"[{i}, {j}]", f"the predecessors of row {i}"
    else:
        kind, j, v = err
        at, path = f"[{j}]", "the predecessors"
    if kind == 1:
        span = f"0 .. {n - 1}" if n else "none"
        return ValueError(f"{name}(): predecessors{at} = {v}: a predecessor is a node ({span}) or -9999")
    if kind == 2:
        return ValueError(f"{name}(): predecessors{at} = {v}, but the graph has no edge {v} -> {j}")
    if kind == 3:
        return ValueError(f"{name}(): {path} go round a cycle through node {j}")
    if kind == 4:
        return ValueError(f"{name}(): {path} lead from node {j} to node {v}, which has no predecessor, not to node {i}")
    return ValueError(f"{name}(): predecessors{at} = {v}: a node is not its own predecessor")


@function
def construct_dist_matrix(csgraph, predecessors, /, *, directed=True, null_value=math.inf):
    """The (n, n) float64 distances along the paths an (n, n) predecessor matrix draws (row i: from i),
    each the left-to-right sum of its edges' weights; null_value where there is no path, 0 on the
    diagonal."""
    name = "construct_dist_matrix"
    directed = C.flag(directed, name, "directed")
    null = C.real(null_value, name, "null_value")
    n, indptr, ix, data, _ = _weights(name, csgraph, False)
    P = _predecessors(predecessors, (n, n), name)
    dist, err = _core._construct(indptr, ix, data, n, P, directed, null)
    if err is not None:
        raise _pred_error(name, n, err)
    return dist


@function
def reconstruct_path(csgraph, predecessors, /, *, directed=True):
    """The tree a predecessor array (n,) draws: an (n, n) float64 csr_array with an edge pred[v] -> v of the
    graph's weight (undirected: the smaller of the two directions) for every v that has one."""
    name = "reconstruct_path"
    directed = C.flag(directed, name, "directed")
    n, indptr, ix, data, _ = _weights(name, csgraph, False)
    P = _predecessors(predecessors, (n,), name)
    out = _core._reconstruct(indptr, ix, data, n, P, directed)
    if out[0] is None:
        raise _pred_error(name, n, out[1], row=False)
    return C.csr(n, *out)
