"""The frame area: Series, DataFrame and Index (``sd.frame``).

The public API is specified in docs/api/frame.md. The thread count is the
library-wide ``sd.set_num_threads`` (API.md §3).
"""

from .. import _core as _sd_core

_core = _sd_core.frame  # first: every module below takes its native functions from here

from ._na import NA  # noqa: E402
from ._frame import DataFrame  # noqa: E402
from ._index import Index, RangeIndex  # noqa: E402
from ._io import read_csv  # noqa: E402
from ._merge import concat, merge  # noqa: E402
from ._series import Series  # noqa: E402

__all__ = ["NA", "DataFrame", "Index", "RangeIndex", "Series", "concat", "merge", "read_csv"]
