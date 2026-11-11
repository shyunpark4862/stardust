"""stardust.linalg: linear algebra on stardust arrays, wrapping Accelerate LAPACK.

Usage: ``import stardust as sd`` then ``sd.linalg.solve(a, b)``. The public API
is docs/api/linalg.md.
"""

from .. import _core as _sd_core
from . import _results
from ._errors import LinAlgError

# The functions live in the submodule stardust._core.linalg (DESIGN.md §3.2).
_core = _sd_core.linalg

# The C++ core builds these Python objects itself.
_core._setup(LinAlgError, _results.ALL)

__version__ = "0.0.1"

__all__ = [
    "LinAlgError",
    "cholesky",
    "cholesky_solve",
    "cond",
    "cross",
    "det",
    "diagonal",
    "eig",
    "eigh",
    "eigvals",
    "eigvalsh",
    "expm",
    "hessenberg",
    "inv",
    "kron",
    "ldl",
    "lstsq",
    "lu",
    "lu_solve",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "outer",
    "pinv",
    "qr",
    "qr_pivoted",
    "schur",
    "slogdet",
    "solve",
    "solve_banded",
    "solve_sylvester",
    "solve_triangular",
    "solve_tridiagonal",
    "svd",
    "svdvals",
    "trace",
    "vecdot",
    "vector_norm",
]

globals().update({_name: getattr(_core, _name) for _name in __all__ if _name != "LinAlgError"})
