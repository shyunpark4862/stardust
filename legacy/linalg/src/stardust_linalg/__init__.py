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
    det,
    diagonal,
    inv,
    lu,
    lstsq,
    lu_solve,
    matrix_norm,
    matrix_power,
    outer,
    qr,
    qr_pivoted,
    slogdet,
    solve,
    solve_triangular,
    trace,
    vecdot,
    vector_norm,
)

__version__ = "0.0.1"

__all__ = [
    "LinAlgError",
    "cholesky",
    "cholesky_solve",
    "det",
    "diagonal",
    "inv",
    "lu",
    "lstsq",
    "lu_solve",
    "matrix_norm",
    "matrix_power",
    "outer",
    "qr",
    "qr_pivoted",
    "slogdet",
    "solve",
    "solve_triangular",
    "trace",
    "vecdot",
    "vector_norm",
]
