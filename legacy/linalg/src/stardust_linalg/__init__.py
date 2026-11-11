"""stardust-linalg: an educational linear algebra library on top of stardust-numpy.

The functions are listed in API.md and arrive milestone by milestone (ROADMAP.md).
"""

from . import _core
from ._errors import LinAlgError
from ._results import SlogdetResult

# The C++ core builds these Python objects itself (ARCHITECTURE.md §1).
_core._setup(LinAlgError, SlogdetResult)

from ._core import det, inv, slogdet, solve  # noqa: E402

__version__ = "0.0.1"

__all__ = ["LinAlgError", "det", "inv", "slogdet", "solve"]
