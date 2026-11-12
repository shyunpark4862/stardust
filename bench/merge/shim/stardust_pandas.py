"""`import stardust_pandas` gets stardust.frame (see stardust_numpy.py here),
plus the thread functions, which are library-wide now (sd.set_num_threads)."""

from stardust import get_num_threads, set_num_threads  # noqa: F401
from stardust.frame import *  # noqa: F401,F403
