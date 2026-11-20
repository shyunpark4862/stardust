"""stardust.sparse: sparse arrays and sparse linear algebra (docs/api/sparse.md,
docs/design/sparse.md).

Usage: ``import stardust as sd`` then ``A = sd.sparse.csr_array((data, indices, indptr), shape=(m, n))``,
``A @ x`` and ``sd.sparse.spsolve(A, b)``. The formats and functions are scipy.sparse's sparse
**array** forms (no spmatrix classes), and scipy.sparse.linalg's functions sit here directly (no
sd.sparse.linalg); inputs and options follow this library's rules (decisions SP1-SP18 in
docs/design/sparse.md §1.1).

The public names come milestone by milestone (docs/design/sparse.md §8): none yet. The native
core is stardust._core.sparse (the Accelerate LU wrapper, csrc/accelerate/sparse.hpp).
"""

from .. import _core as _sd_core

_core = _sd_core.sparse

__all__ = []
