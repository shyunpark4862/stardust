"""stardust.sparse: sparse arrays and sparse linear algebra (docs/api/sparse.md,
docs/design/sparse.md).

Usage: ``import stardust as sd`` then ``A = sd.sparse.csr_array((data, indices, indptr), shape=(m, n))``,
``A.tocsc()`` and ``A.toarray()``. The formats are scipy.sparse's sparse **array** forms (no
spmatrix classes), and scipy.sparse.linalg's functions will sit here directly (no
sd.sparse.linalg); inputs and options follow this library's rules (decisions SP1-SP34 in
docs/design/sparse.md §1.1).

The public names come milestone by milestone (docs/design/sparse.md §8): the formats, their
conversions and operations, indexing and the builders, the direct solves and LinearOperator, the
iterative solvers and the eigensolvers so far.
The native core is stardust._core.sparse.
"""

from .. import _calls
from .. import _core as _sd_core
from ._base import sparray
from ._compressed import csc_array, csr_array
from ._construct import (
    block_array,
    block_diag,
    diags_array,
    eye_array,
    find,
    hstack,
    kron,
    kronsum,
    random_array,
    tril,
    triu,
    vstack,
)
from ._coo import coo_array
from ._direct import factorized, inv, is_sptriangular, spbandwidth, splu, spsolve, spsolve_triangular
from ._dok import dok_array
from ._eigen import eigs, eigsh, svds
from ._interface import LinearOperator, aslinearoperator
from ._iterative import bicgstab, cg, gmres, lsmr, lsqr, minres

_core = _sd_core.sparse

eye_array = _calls.function(eye_array)
diags_array = _calls.function(diags_array)
random_array = _calls.function(random_array)
kron = _calls.function(kron)
kronsum = _calls.function(kronsum)
hstack = _calls.function(hstack)
vstack = _calls.function(vstack)
block_array = _calls.function(block_array)
block_diag = _calls.function(block_diag)
tril = _calls.function(tril)
triu = _calls.function(triu)
find = _calls.function(find)
spsolve = _calls.function(spsolve)
splu = _calls.function(splu)
factorized = _calls.function(factorized)
inv = _calls.function(inv)
spsolve_triangular = _calls.function(spsolve_triangular)
is_sptriangular = _calls.function(is_sptriangular)
spbandwidth = _calls.function(spbandwidth)
aslinearoperator = _calls.function(aslinearoperator)
cg = _calls.function(cg)
gmres = _calls.function(gmres)
minres = _calls.function(minres)
bicgstab = _calls.function(bicgstab)
lsqr = _calls.function(lsqr)
lsmr = _calls.function(lsmr)
eigsh = _calls.function(eigsh)
eigs = _calls.function(eigs)
svds = _calls.function(svds)


def issparse(x, /):
    """Whether x is an sd sparse array (a scipy.sparse object is not)."""
    return isinstance(x, sparray)


def matrix_transpose(A, /):
    """The transpose of a 2-D sparse array (A.T); a ValueError for 1-D, as for an sd.ndarray."""
    if not isinstance(A, sparray):
        raise TypeError(f"matrix_transpose(): 'A' must be a sparse array, not {_sd_core._type_name(A)}")
    if A.ndim != 2:
        raise ValueError(f"matrix_transpose(): the array must have at least 2 dimensions, got {A.ndim}")
    return A._transpose()


__all__ = [
    "sparray", "coo_array", "csr_array", "csc_array", "dok_array", "issparse", "matrix_transpose",
    "eye_array", "diags_array", "random_array", "kron", "kronsum", "hstack", "vstack", "block_array", "block_diag",
    "tril", "triu", "find",
    "spsolve", "splu", "factorized", "inv", "spsolve_triangular", "is_sptriangular", "spbandwidth",
    "LinearOperator", "aslinearoperator",
    "cg", "gmres", "minres", "bicgstab", "lsqr", "lsmr",
    "eigsh", "eigs", "svds",
]  # fmt: skip
