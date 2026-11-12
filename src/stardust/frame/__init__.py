"""The frame area: Series, DataFrame and Index (``sd.frame``).

The public API is specified in docs/api/frame.md. The thread count is the
library-wide ``sd.set_num_threads`` (API.md §3).
"""

from .. import _core as _sd_core

_core = _sd_core.frame  # first: every module below takes its native functions from here

from ._na import NA  # noqa: E402
from .frame import DataFrame  # noqa: E402
from .index import Index, RangeIndex  # noqa: E402
from .io import read_csv  # noqa: E402
from .merge import concat, merge  # noqa: E402
from .series import Series  # noqa: E402

__all__ = ["NA", "DataFrame", "Index", "RangeIndex", "Series", "concat", "merge", "read_csv"]
