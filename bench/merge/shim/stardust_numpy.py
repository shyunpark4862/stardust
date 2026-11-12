"""Lets the old benchmark scripts (legacy/*/bench) time stardust: `import
stardust_numpy` gets stardust's array namespace. Put this directory first on
PYTHONPATH (bench/merge/rerun.sh)."""

from stardust import *  # noqa: F401,F403
from stardust import __all__, set_num_threads, get_num_threads  # noqa: F401
