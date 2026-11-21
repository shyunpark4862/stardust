"""stardust.sparse: sparse arrays and sparse linear algebra (docs/api/sparse.md,
docs/design/sparse.md).

Usage: ``import stardust as sd`` then ``A = sd.sparse.csr_array((data, indices, indptr), shape=(m, n))``,
``A.tocsc()`` and ``A.toarray()``. The formats are scipy.sparse's sparse **array** forms (no
spmatrix classes), and scipy.sparse.linalg's functions will sit here directly (no
sd.sparse.linalg); inputs and options follow this library's rules (decisions SP1-SP34 in
docs/design/sparse.md §1.1).

The public names come milestone by milestone (docs/design/sparse.md §8): the formats and their
conversions so far. The native core is stardust._core.sparse.
"""

from .. import _core as _sd_core
from ._base import sparray
from ._compressed import csc_array, csr_array
from ._coo import coo_array
from ._dok import dok_array

_core = _sd_core.sparse


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


__all__ = ["sparray", "coo_array", "csr_array", "csc_array", "dok_array", "issparse", "matrix_transpose"]
