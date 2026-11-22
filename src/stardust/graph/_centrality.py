"""PageRank and betweenness centrality of sd.graph (docs/api/graph.md §6, docs/design/graph.md §2.6, §3.7;
milestone GR5).

The arguments are checked here; the iterations and the searches are stardust._core.graph's
(csrc/graph/centrality.cpp).

- pagerank is the solution x* of x = alpha G^T x + (1 - alpha) v, G the out-weight normalised transition
  matrix (a self loop is an edge, decision GR28) with the dangling rows replaced by the dangling
  distribution. It iterates from v and stops on a proven l1 bound: the result is within tol + F of x*, F
  the rounding of one step over 1 - alpha (decisions GR19, GR56, GR57). No max_iter: the count is bounded
  by alpha and tol.
- betweenness_centrality is Brandes's definition over the shortest-path DAG of the shortest paths'
  definition (left-to-right sums, docs/design/graph.md §3.2), networkx's endpoints and scaling (decisions
  GR20, GR58). Weights must be positive.
"""

import math

from .. import _core as _sd
from .._calls import function
from . import _common as C

_core = _sd.graph


def _weights(name, csgraph, unweighted, positive):
    """(n, indptr, indices, data or None): the pattern only when unweighted; else the float64 weights,
    finite, and positive (positive) or 0 or more."""
    n, indptr, indices, data = C.graph(csgraph, name, values=not unweighted)
    if not unweighted:
        nonfinite, negative = _core._scan(data)
        if nonfinite:
            raise ValueError(f"{name}(): stored edge weights must be finite")
        if positive and (negative or (data.size and bool(_sd.any(data == 0.0)))):
            raise ValueError(f"{name}(): edge weights must be positive; unweighted=True counts the edges")
        if negative:
            raise ValueError(f"{name}(): negative edge weights are not allowed")
    return n, indptr, indices, data


def _distribution(x, n, name, arg):
    """A personalization or dangling argument: n non-negative finite numbers with a positive sum, as a
    C-contiguous float64 sd array (None: the uniform one)."""
    if x is None:
        return _sd.ones(n)
    try:
        a = _sd.asarray(x)
    except TypeError, ValueError:
        raise TypeError(f"{name}(): '{arg}' must be a 1-D array of numbers, not {C.type_name(x)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{name}(): '{arg}' must be real, not a complex128 array")
    if a.dtype not in (_sd.bool, _sd.int64, _sd.float64):
        raise TypeError(f"{name}(): '{arg}' must be a 1-D array of numbers, not {a.dtype}")
    if a.ndim != 1:
        raise ValueError(f"{name}(): '{arg}' must be 1-D, not {a.ndim}-D")
    if a.shape[0] != n:
        raise ValueError(f"{name}(): '{arg}' must have one entry per node ({n}), not {a.shape[0]}")
    a = C.contiguous(a.astype(_sd.float64) if a.dtype != _sd.float64 else a)
    if n:
        if not bool(_sd.all(_sd.isfinite(a))):
            raise ValueError(f"{name}(): '{arg}' must be finite")
        if bool(_sd.any(a < 0.0)):
            raise ValueError(f"{name}(): '{arg}' must be 0 or more")
        if not bool(_sd.any(a > 0.0)):
            raise ValueError(f"{name}(): '{arg}' must have a positive sum")
    return a


@function
def pagerank(
    csgraph, /, *, alpha=0.85, personalization=None, dangling=None, directed=True, unweighted=False, tol=1e-12
):
    """The PageRank vector (float64 (n,), sum 1): the solution of x = alpha P^T x + (1 - alpha) v, P the
    transition matrix of the out-weights (unweighted: every edge weighs 1; a self loop is an edge) whose rows
    of out-weight 0 are the dangling distribution (default: personalization), v the personalization
    (default: uniform), both divided by their sums. directed=False reads the graph undirected. Within tol
    of the solution in l1 (plus one step's rounding over 1 - alpha)."""
    name = "pagerank"
    directed = C.flag(directed, name, "directed")
    unweighted = C.flag(unweighted, name, "unweighted")
    a = C.real(alpha, name, "alpha")
    if not 0.0 <= a < 1.0:
        raise ValueError(f"{name}(): 'alpha' must be 0 or more and less than 1, not {alpha!r}")
    t = C.real(tol, name, "tol")
    if not (t > 0.0 and math.isfinite(t)):
        raise ValueError(f"{name}(): 'tol' must be positive and finite, not {tol!r}")
    n, indptr, indices, data = _weights(name, csgraph, unweighted, False)
    p = _distribution(personalization, n, name, "personalization")
    q = p if dangling is None else _distribution(dangling, n, name, "dangling")
    x, _ = _core._pagerank(indptr, indices, data, n, directed, a, p, q, t)
    return x


@function
def betweenness_centrality(csgraph, /, *, directed=True, unweighted=False, normalized=True, endpoints=False):
    """Brandes's betweenness centrality (float64 (n,)): for each node v, the sum over the pairs (s, t) of
    other nodes of the fraction of the shortest s-t paths through v. Undirected: each pair counted once.
    normalized divides by the number of ordered pairs (n - 1)(n - 2) (endpoints: n(n - 1)); endpoints counts
    the paths' ends too (networkx's meaning and scaling). The weights must be positive;
    unweighted counts the edges."""
    name = "betweenness_centrality"
    directed = C.flag(directed, name, "directed")
    unweighted = C.flag(unweighted, name, "unweighted")
    normalized = C.flag(normalized, name, "normalized")
    endpoints = C.flag(endpoints, name, "endpoints")
    n, indptr, indices, data = _weights(name, csgraph, unweighted, True)
    return _core._betweenness(indptr, indices, data, n, directed, normalized, endpoints)
