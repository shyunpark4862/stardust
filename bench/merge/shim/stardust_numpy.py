"""Lets the old benchmark scripts (bench/{array,linalg,frame}) time stardust: `import
stardust_numpy` gets stardust's array namespace. Put this directory first on
PYTHONPATH (bench/merge/rerun.sh)."""

from stardust import *  # noqa: F401,F403
from stardust import __all__, get_num_threads, set_num_threads  # noqa: F401
