"""The matchings and the assignment of sd.graph (docs/api/graph.md §4, docs/design/graph.md §2.4, §3.5;
milestone GR3).

The arguments are checked here; Hopcroft-Karp and the shortest augmenting path solver are
stardust._core.graph's (csrc/graph/matching.cpp). min_weight_full_bipartite_matching (the stored pairs of a
biadjacency matrix) and linear_sum_assignment (every entry of a dense cost matrix a pair, +inf none) share
the solver: the smaller side is assigned row by row in ascending order, each row's search taking the
columns in the order (distance, free first, index), so ties among optimal assignments are decided by that
fixed order (GR16) and integer costs give the exact optimum.
"""

from .. import _core as _sd
from .. import sparse as _sparse
from .._calls import function
from .._results import AssignmentResult
from ..stats._descriptive import _choice
from . import _common as C

_core = _sd.graph


def _empty():
    return AssignmentResult(C.readonly(_sd.zeros(0, dtype=_sd.int64)), C.readonly(_sd.zeros(0, dtype=_sd.int64)))


@function
def maximum_bipartite_matching(graph, /, *, perm_type="row"):
    """A largest matching of the bipartite graph whose rows and columns are graph's (any shape; a stored
    entry, a stored 0 too, is an edge): perm_type='row' gives each column its row (int64 (cols,)),
    'column' each row its column (int64 (rows,)), -1 where unmatched."""
    name = "maximum_bipartite_matching"
    perm_type = _choice(perm_type, name, "perm_type", ("row", "column"))
    rows, cols, indptr, indices, _ = C.biadjacency(graph, name)
    col_of_row = _core._bipartite_matching(indptr, indices, rows, cols)
    if perm_type == "column":
        return col_of_row
    out = _sd.full(cols, -1, dtype=_sd.int64)
    matched = col_of_row >= 0
    if bool(_sd.any(matched)):
        out[col_of_row[matched]] = _sd.arange(rows, dtype=_sd.int64)[matched]
    return out


@function
def min_weight_full_bipartite_matching(biadjacency, /, *, maximize=False):
    """The matching of size min(rows, cols) of least total weight (maximize: greatest) of the bipartite
    graph whose stored entries are the pairs (a stored 0 a pair of weight 0; dense: the entries that are not
    0, NaN or +-inf), as AssignmentResult(row_ind, col_ind) (int64, row_ind ascending). ValueError when
    there is no such matching."""
    name = "min_weight_full_bipartite_matching"
    maximize = C.flag(maximize, name, "maximize")
    rows, cols, indptr, indices, data = C.biadjacency(biadjacency, name, values=True, arg="biadjacency")
    nonfinite, _ = _core._scan(data)
    if nonfinite:
        raise ValueError(f"{name}(): stored edge weights must be finite")
    if rows == 0 or cols == 0:
        return _empty()
    if data.size == rows * cols:  # every pair stored: the dense scan, the same assignment (GR45) and faster (GR62)
        status, ri, ci, _, _ = _core._assign_dense(data.reshape((rows, cols)), maximize)
    else:
        status, ri, ci, _, _ = _core._assign_sparse(indptr, indices, data, rows, cols, maximize)
    if status:
        raise ValueError(f"{name}(): no full matching exists")
    return AssignmentResult(C.readonly(ri), C.readonly(ci))


@function
def linear_sum_assignment(cost_matrix, /, *, maximize=False):
    """The assignment of the smaller side of a dense (rows, cols) cost matrix of least total cost
    (maximize: greatest), as AssignmentResult(row_ind, col_ind) (int64, row_ind ascending): every entry is a
    pair, +inf one that cannot be chosen (-inf with maximize). ValueError for NaN, -inf (+inf with maximize)
    and when no assignment exists."""
    name = "linear_sum_assignment"
    maximize = C.flag(maximize, name, "maximize")
    if isinstance(cost_matrix, _sparse.sparray):
        raise TypeError(
            f"{name}(): 'cost_matrix' must be a dense matrix, not {C.type_name(cost_matrix)}; the assignment of "
            "a sparse matrix's stored pairs is min_weight_full_bipartite_matching"
        )
    a = C.dense(cost_matrix, name, "cost_matrix")
    if a.dtype != _sd.float64:
        a = a.astype(_sd.float64)
    if a.shape[0] == 0 or a.shape[1] == 0:
        return _empty()
    status, ri, ci, _, _ = _core._assign_dense(a, maximize)
    if status == 2:
        bad = "NaN or +inf" if maximize else "NaN or -inf"
        raise ValueError(f"{name}(): 'cost_matrix' must not hold {bad}")
    if status:
        raise ValueError(f"{name}(): cost matrix is infeasible")
    return AssignmentResult(C.readonly(ri), C.readonly(ci))
