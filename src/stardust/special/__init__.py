"""stardust.special: special functions on stardust arrays (scipy.special's names).

Usage: ``import stardust as sd`` then ``sd.special.erf(x)``. The public API is
docs/api/special.md; the functions arrive milestone by milestone (ROADMAP.md,
special-function area, S1-S8).
"""

from .. import _core as _sd_core

# The functions live in the submodule stardust._core.special (docs/design/special.md §1.4).
_core = _sd_core.special

__version__ = "0.0.1"

__all__ = []

globals().update({_name: getattr(_core, _name) for _name in __all__})
