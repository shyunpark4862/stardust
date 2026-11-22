"""csgraph_from_dense and csgraph_to_dense of sd.graph (docs/api/graph.md §2, docs/design/graph.md §2.7;
milestone GR1). scipy's names and arguments; the masked-array tools have no counterpart (GR14).
"""

import math

from .. import _core as _sd
from .._calls import function
from . import _common as C

_core = _sd.graph


@function
def csgraph_from_dense(graph, /, *, null_value=0, nan_null=True, infinity_null=True):
    """The float64 csr_array of a square dense matrix's entries, leaving out those equal to null_value
    (None: no value is), NaN with nan_null and +-inf with infinity_null (scipy's rule: a NaN null_value
    sets nan_null, an infinite one infinity_null; null_value=inf keeps the zeros as zero-weight edges)."""
    name = "csgraph_from_dense"
    use_null = null_value is not None
    null = C.real(null_value, name, "null_value") if use_null else 0.0
    nan_null = C.flag(nan_null, name, "nan_null")
    infinity_null = C.flag(infinity_null, name, "infinity_null")
    if use_null and math.isnan(null):  # scipy's rule: a NaN or infinite null_value means nan_null or infinity_null
        use_null, nan_null = False, True
    elif use_null and math.isinf(null):
        use_null, infinity_null = False, True
    a = C.dense(graph, name, "graph")
    n = C._square(a.shape, name, "graph")
    indptr, indices, data = _core._dense_csr(a, null, use_null, nan_null, infinity_null, True)
    return C.csr(n, indptr, indices, data)


@function
def csgraph_to_dense(csgraph, /, *, null_value=0):
    """The (n, n) float64 sd.ndarray of a graph: its stored entries at their places (a dense input read by
    the graph rule: 0, NaN and +-inf are no edge), null_value everywhere else."""
    name = "csgraph_to_dense"
    null = C.real(null_value, name, "null_value")
    n, indptr, indices, data = C.graph(csgraph, name, values=True)
    C.room(name, 8 * n * n)
    return _core._to_dense(indptr, indices, data, n, null)
