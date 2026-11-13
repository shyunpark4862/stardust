"""stardust.special: special functions on stardust arrays (scipy.special's names).

Usage: ``import stardust as sd`` then ``sd.special.erf(x)``. The public API is
docs/api/special.md; the functions arrive milestone by milestone (ROADMAP.md,
special-function area, S1-S8).
"""

from .. import _core as _sd_core

# The functions live in the submodule stardust._core.special (docs/design/special.md §1.4).
_core = _sd_core.special

__version__ = "0.0.1"

__all__ = [
    # S1: logistic and information (docs/api/special.md §5)
    "expit", "logit", "log_expit", "xlogy", "xlog1py", "entr", "rel_entr",
    # S2: the error function and the normal distribution (§2)
    "erf", "erfc", "erfcx", "erfinv", "erfcinv", "ndtr", "log_ndtr", "ndtri",
]

globals().update({_name: getattr(_core, _name) for _name in __all__})
