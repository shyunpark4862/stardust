"""stardust-linalg: an educational linear algebra library on top of stardust-numpy.

The functions are listed in API.md and arrive milestone by milestone (ROADMAP.md).
"""

from . import _core, _results
from ._errors import LinAlgError

# The C++ core builds these Python objects itself (ARCHITECTURE.md §1).
_core._setup(LinAlgError, _results.ALL)

from ._core import (  # noqa: E402
    cholesky,
    cholesky_solve,
    cond,
    cross,
    det,
    diagonal,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    expm,
    hessenberg,
    inv,
    kron,
    ldl,
    lstsq,
    lu,
    lu_solve,
    matrix_norm,
    matrix_power,
    matrix_rank,
    outer,
    pinv,
    qr,
    qr_pivoted,
    schur,
    slogdet,
    solve,
    solve_banded,
    solve_sylvester,
    solve_triangular,
    solve_tridiagonal,
    svd,
    svdvals,
    trace,
    vecdot,
    vector_norm,
)

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
