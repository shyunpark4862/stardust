"""The ordering, the structural rank and the laplacian of sd.graph (docs/api/graph.md §5,
docs/design/graph.md §2.5, §3.6; milestone GR4).

The arguments are checked here; the reverse Cuthill-McKee search and the laplacian's degrees and entries
are stardust._core.graph's (csrc/graph/ordering.cpp), the structural rank is the size of the largest
matching of the matchings' Hopcroft-Karp (csrc/graph/matching.cpp).

- reverse_cuthill_mckee reads the pattern only (a stored 0 or NaN is an edge; a dense matrix by the graph
  rule); its ties are decided by (degree, index), the seeds included (decision GR52).
- laplacian is a matrix function: a dense matrix is read as its values (0 is 0, NaN and inf propagate),
  not by the graph rule. L = D - A with D the column sums (use_out_degree: the row sums), the diagonal of
  A left out; normed: D^-1/2 L D^-1/2 with 1 on the diagonal (decision GR50); symmetrized: A + A^T
  first, by sd.sparse's sum rule (decision GR53). form='lo' wraps the same matrix (decision GR51).
"""

from .. import _core as _sd
from .. import sparse as _sparse
from .._calls import function
from .._results import LaplacianResult
from ..sparse import _common as _spc
from ..stats._descriptive import _choice
from . import _common as C

_core = _sd.graph


@function
def reverse_cuthill_mckee(graph, /, *, symmetric_mode=False):
    """The reverse Cuthill-McKee permutation of a square graph's pattern (int64 (n,)): a breadth-first
    order from a node of least degree per component, neighbours by (degree, index), reversed. Without
    symmetric_mode the pattern of A and of its transpose together is read."""
    name = "reverse_cuthill_mckee"
    symmetric_mode = C.flag(symmetric_mode, name, "symmetric_mode")
    n, indptr, indices, _ = C.graph(graph, name, arg="graph")
    return _core._rcm(indptr, indices, n, symmetric_mode)


@function
def structural_rank(graph, /):
    """The structural rank of a matrix's pattern (any shape; a stored 0 is an entry): the size of the
    largest matching of its rows and columns, a Python int."""
    name = "structural_rank"
    rows, cols, indptr, indices, _ = C.biadjacency(graph, name)
    if rows == 0 or cols == 0:
        return 0
    col_of_row = _core._bipartite_matching(indptr, indices, rows, cols)
    return int(_sd.count_nonzero(col_of_row >= 0))


def _dtype(dtype, name):
    """The dtype option: None, int64 or float64 (an object or its name)."""
    if dtype is None:
        return None
    t = _spc.dtype_arg(dtype, name)
    if t not in (_sd.int64, _sd.float64):
        raise TypeError(f"{name}(): 'dtype' must be None, int64 or float64, not {t}")
    return t


@function
def laplacian(
    csgraph, /, *, normed=False, return_diag=False, use_out_degree=False, form="array", dtype=None, symmetrized=False
):
    """The laplacian L = D - A of a square matrix (its values as they are; the diagonal left out): D the
    column sums (use_out_degree: the row sums), normed D^-1/2 L D^-1/2 (a node of degree 0 divided by 1),
    symmetrized A + A^T first. A csr_array for a sparse input, an sd.ndarray for a dense one,
    sd.sparse.LinearOperator for form='lo'; return_diag adds the degrees (normed: their square roots, 1
    for degree 0) as LaplacianResult(laplacian, diag). dtype None: the input's (bool: int64; normed:
    float64)."""
    name = "laplacian"
    normed = C.flag(normed, name, "normed")
    return_diag = C.flag(return_diag, name, "return_diag")
    use_out_degree = C.flag(use_out_degree, name, "use_out_degree")
    symmetrized = C.flag(symmetrized, name, "symmetrized")
    form = _choice(form, name, "form", ("array", "lo"))
    dtype = _dtype(dtype, name)
    a = C._matrix(csgraph, name, "csgraph")
    n = C._square(a.shape, name, "csgraph")
    work = _sd.float64 if normed or a.dtype == _sd.float64 else _sd.int64
    if isinstance(a, _sparse.sparray):
        A = a.tocsr()
        data = A.data if A.dtype == work else A.data.astype(work)
        ip, ix, v, diag = _core._laplacian_csr(A.indptr, A.indices, data, n, normed, use_out_degree, symmetrized)
        L = C.csr(n, ip, ix, v)
    else:
        if a.dtype != work:
            a = a.astype(work)
        L, diag = _core._laplacian_dense(a, normed, use_out_degree, symmetrized)
    if dtype is not None and dtype != L.dtype:
        L = L.astype(dtype)
        diag = diag.astype(dtype)
    if form == "lo":
        L = _sparse.aslinearoperator(L)
    if not return_diag:
        return L
    if isinstance(L, _sd.ndarray):
        L = C.readonly(L)
    return LaplacianResult(L, C.readonly(diag))
