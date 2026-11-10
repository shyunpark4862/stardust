"""stardust-pandas: an educational rebuild of a pandas subset.

Usage: ``import stardust_pandas as sdpd``. The public API is specified in API.md.
"""

from . import _core
from ._core import get_num_threads, set_num_threads
from ._na import NA
from .frame import DataFrame
from .index import RangeIndex
from .series import Series

__all__ = ["NA", "DataFrame", "RangeIndex", "Series", "get_num_threads", "set_num_threads"]
