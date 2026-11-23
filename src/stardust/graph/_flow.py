"""The maximum flow of sd.graph (docs/api/graph.md §4, docs/design/graph.md §2.4, §3.5; milestone GR3).

The arguments are checked here; Dinic's and Edmonds-Karp's algorithms are stardust._core.graph's
(csrc/graph/flow.cpp). The capacities are exact integers: bool or int64 only (float64 is a TypeError),
none negative, and the capacities out of the source must sum within int64, which bounds every flow (GR15).
The flow value is unique; the flow returned is the one the fixed scan order (neighbours ascending) gives
(GR16), the same for any thread count.
"""

from .. import _core as _sd
from .. import sparse as _sparse
from .._calls import function
from ..stats._descriptive import _choice
from . import _common as C

_core = _sd.graph
_METHODS = ("dinic", "edmonds_karp")


class MaximumFlowResult:
    """The result of maximum_flow: flow_value (a Python int) and flow, the antisymmetric (n, n) int64
    csr_array of the flow on the pattern of the graph and its transpose (a new array on every access)."""

    __slots__ = ("_n", "_indptr", "_indices", "_flow", "flow_value")

    def __init__(self, flow_value, n, indptr, indices, flow):
        self.flow_value = flow_value
        self._n, self._indptr, self._indices, self._flow = n, indptr, indices, flow

    @property
    def flow(self):
        """The flow as a new (n, n) int64 csr_array: f(u, v) = -f(v, u), stored on every entry of the
        graph or its transpose (the diagonal entries of the graph too, with 0)."""
        return _sparse.csr_array(
            (self._flow.copy(), self._indices.copy(), self._indptr.copy()), shape=(self._n, self._n)
        )

    def __repr__(self):
        n, nnz = self._n, int(self._indices.shape[0])
        return f"MaximumFlowResult(flow_value={self.flow_value}, flow=<csr_array ({n}, {n}), nnz {nnz}>)"


MaximumFlowResult.__module__ = "stardust.graph"


@function
def maximum_flow(csgraph, source, sink, /, *, method="dinic"):
    """The maximum flow from source to sink of the network whose capacities are csgraph's (bool or int64;
    a stored entry i -> j is an edge of that capacity), by method 'dinic' or 'edmonds_karp', as
    MaximumFlowResult(flow_value, flow). Every method gives the same flow value."""
    name = "maximum_flow"
    method = _choice(method, name, "method", _METHODS)
    n, indptr, indices, caps = C.capacities(csgraph, name)
    s = C.node(source, n, name, "source")
    t = C.node(sink, n, name, "sink")
    if s == t:
        raise ValueError(f"{name}(): 'source' and 'sink' must be different nodes, not both {s}")
    if caps.size and bool(_sd.any(caps < 0)):
        raise ValueError(f"{name}(): the capacities must be >= 0")
    out = _core._max_flow(indptr, indices, caps, n, s, t, method == "dinic")
    if out is None:
        raise OverflowError(f"{name}(): the capacities out of the source sum past the int64 range")
    value, fp, fi, fd = out
    return MaximumFlowResult(int(value), n, fp, fi, fd)
