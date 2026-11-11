"""stardust-pandas: an educational rebuild of a pandas subset.

Usage: ``import stardust_pandas as sdpd``. The public API is specified in API.md.
"""

from . import _core
from ._core import get_num_threads, set_num_threads
from ._na import NA
from .frame import DataFrame
from .index import Index, RangeIndex
from .io import read_csv
from .merge import concat, merge
from .series import Series

__all__ = ["NA", "DataFrame", "Index", "RangeIndex", "Series", "concat", "get_num_threads", "merge", "read_csv", "set_num_threads"]
