"""The input conversion and the argument checks every sd.graph function shares (docs/api/graph.md §1,
docs/design/graph.md §2.1, §3.1; decisions GR7, GR8, GR21, GR22).

A graph reaches the native core as one canonical CSR (int64 indptr and indices, float64 values or only
the pattern; csrc/graph/graph.hpp):

- an sd.sparse array (csr, csc, coo, dok): its CSR as it is (a csr_array is not copied; another format
  is converted once, COO's duplicates summed). Every stored entry is an edge, a stored 0 one of weight 0.
- a dense matrix (an sd.ndarray, a numpy array, a nested sequence of numbers): the entries that are not
  0, NaN or +-inf are the edges (scipy's rule), found by the native _dense_csr.
- scipy.sparse objects and frame DataFrames are TypeErrors (the message points at sd.sparse.csr_array(A)).
- bool, int64 and float64; the values a function uses are float64. complex128 is a TypeError.
- 2-D and square (ValueError otherwise); n = 0 is a graph too.

A node argument is an int from 0 to n - 1 (operator.index's rule, bool excluded): another type is a
TypeError, a value out of range a ValueError (GR22). Options are True or False where they are flags.
"""

import operator
import sys

from .. import _core as _sd
from .. import sparse as _sparse
from ..stats._distribution import _real_option

_core = _sd.graph

_TAKES = "an sd.sparse array or a dense matrix (an sd.ndarray, a numpy array or a sequence of numbers)"


class NegativeCycleError(ValueError):
    """A cycle of negative weight that a shortest-path search can reach from the nodes it starts at
    (docs/api/graph.md §3): the distances have no least value then."""


# Shown as stardust.graph.NegativeCycleError in tracebacks: the public path, not this module's.
NegativeCycleError.__module__ = "stardust.graph"


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def _is_scipy_sparse(x):
    sp = sys.modules.get("scipy.sparse")
    return sp is not None and sp.issparse(x)


def _square(shape, name, arg):
    if len(shape) != 2:
        raise ValueError(f"{name}(): '{arg}' must have 2 dimensions, not {len(shape)}")
    if shape[0] != shape[1]:
        raise ValueError(f"{name}(): '{arg}' must be a square matrix, not of shape {tuple(shape)}")
    return int(shape[0])


def _real_dtype(dtype, name, arg):
    if dtype == _sd.complex128:
        raise TypeError(f"{name}(): '{arg}' must be real, not a complex128 array")


def dense(x, name, arg):
    """A dense matrix argument as a C-contiguous 2-D sd.ndarray of bool, int64 or float64 (not checked for
    squareness)."""
    sd = _sd
    if isinstance(x, sd.ndarray):
        a = x
    elif isinstance(x, _sparse.sparray) or _is_scipy_sparse(x) or type(x).__module__.startswith("stardust.frame"):
        raise TypeError(f"{name}(): '{arg}' must be a dense matrix, not {type_name(x)}")
    else:
        try:
            a = sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" not in msg:  # a ragged sequence, ...
                raise ValueError(f"{name}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}") from None
    _real_dtype(a.dtype, name, arg)
    if a.ndim != 2:
        raise ValueError(f"{name}(): '{arg}' must have 2 dimensions, not {a.ndim}")
    return contiguous(a)


def contiguous(a):
    """a as a C-contiguous array: a itself when it is one, else a new C-ordered copy (any strides: a column
    slice, an F-ordered or a reversed view)."""
    if a._flags()["c_contiguous"]:
        return a
    out = _sd.empty(tuple(a.shape), dtype=a.dtype)
    out[...] = a
    return out


def _matrix(x, name, arg):
    """An accepted matrix argument: the sd.sparse array as it is, or a dense one as a C-contiguous sd.ndarray
    (TypeError for scipy.sparse, a DataFrame, complex and other objects; ValueError unless 2-D)."""
    if isinstance(x, _sparse.sparray):
        _real_dtype(x.dtype, name, arg)
        return x
    if _is_scipy_sparse(x):
        raise TypeError(
            f"{name}(): '{arg}' must be {_TAKES}, not a scipy.sparse {type(x).__name__}; convert it with "
            "sd.sparse.csr_array(A)"
        )
    if type(x).__module__.startswith("stardust.frame"):  # an adjacency matrix is not a table
        raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
    return dense(x, name, arg)


def _csr(a, values):
    """(indptr, indices, float64 data or None) of an accepted matrix (_matrix's result)."""
    if isinstance(a, _sparse.sparray):
        A = a.tocsr()
        data = None
        if values:
            data = A.data if A.dtype == _sd.float64 else A.data.astype(_sd.float64)
        return A.indptr, A.indices, data
    return _core._dense_csr(a, 0.0, True, True, True, values)


def graph(csgraph, name, *, values=False, arg="csgraph"):
    """(n, indptr, indices, data) of a square graph argument: canonical CSR arrays and, with `values`, the
    float64 weights (else None: the functions that read only the pattern)."""
    a = _matrix(csgraph, name, arg)
    n = _square(a.shape, name, arg)
    return (n, *_csr(a, values))


def biadjacency(x, name, *, values=False, arg="graph"):
    """(rows, cols, indptr, indices, data) of a bipartite graph's (rows, cols) biadjacency matrix (the
    matchings): any 2-D shape, read by the graph rules (stored entries; the dense rule)."""
    a = _matrix(x, name, arg)
    if len(a.shape) != 2:
        raise ValueError(f"{name}(): '{arg}' must have 2 dimensions, not {len(a.shape)}")
    return (int(a.shape[0]), int(a.shape[1]), *_csr(a, values))


def capacities(x, name, arg="csgraph"):
    """(n, indptr, indices, caps) of a flow network: the capacities as int64 (bool read as 0 and 1). Only
    bool and int64 matrices: float64 is a TypeError even with integer values (GR15)."""
    a = _matrix(x, name, arg)
    n = _square(a.shape, name, arg)
    if a.dtype not in (_sd.bool, _sd.int64):
        raise TypeError(f"{name}(): the capacities must be bool or int64, not {a.dtype}")
    if isinstance(a, _sparse.sparray):
        A = a.tocsr()
        caps = A.data if A.dtype == _sd.int64 else A.data.astype(_sd.int64)
        return n, A.indptr, A.indices, caps
    return (n, *_core._dense_csr(a, 0.0, True, True, True, True, as_int64=True))


def node(v, n, name, arg):
    """A node argument: an int in 0 .. n - 1 (GR22)."""
    t = type(v)
    if t is not int and (
        t is bool
        or t is float
        or isinstance(v, (bool, _sd.ndarray))
        or (t.__name__ == "bool" and t.__module__ == "numpy")
    ):
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        i = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}") from None
    if not 0 <= i < n:
        if n == 0:
            raise ValueError(f"{name}(): '{arg}' must be a node of the graph, which has no nodes")
        raise ValueError(f"{name}(): '{arg}' must be a node of the graph (0 .. {n - 1}), not {i}")
    return i


def flag(v, name, arg):
    if v is not True and v is not False:
        raise TypeError(f"{name}(): '{arg}' must be True or False, not {type_name(v)}")
    return v


def real(v, name, arg):
    """An int-or-float option as a float (numpy's floats too); TypeError for anything else."""
    r = _real_option(v, name, arg)
    if r is None and type(v).__module__ == "numpy" and getattr(getattr(v, "dtype", None), "kind", "") == "f":
        r = float(v)
    if r is None:
        raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {type_name(v)}")
    return r


def count(v, name, arg):
    """A count argument: an int of 0 or more (bool and float are TypeErrors)."""
    t = type(v)
    if t is bool or t is float or isinstance(v, _sd.ndarray) or (t.__name__ == "bool" and t.__module__ == "numpy"):
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        k = operator.index(v)
    except TypeError:
        raise TypeError(f"{name}(): '{arg}' must be an int, not {type_name(v)}") from None
    if k < 0:
        raise ValueError(f"{name}(): '{arg}' must be >= 0, not {k}")
    return k


def room(name, nbytes):
    """MemoryError before an output larger than this machine's physical memory (API.md §2)."""
    if nbytes > _sd._physical_memory():
        raise MemoryError(f"{name}(): cannot allocate {nbytes} bytes: more than this machine's physical memory")


def readonly(a):
    a._set_readonly()
    return a


def csr(n, indptr, indices, data):
    """The (n, n) sd.sparse.csr_array of canonical arrays made here."""
    return _sparse.csr_array((data, indices, indptr), shape=(n, n))
