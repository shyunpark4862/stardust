"""stardust-pandas: an educational rebuild of a pandas subset.

Usage: ``import stardust_pandas as sdpd``. The public API is specified in API.md.
"""

from . import _core
from ._core import get_num_threads, set_num_threads

__all__ = ["get_num_threads", "set_num_threads"]
