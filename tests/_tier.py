"""The test tier (tests/conftest.py sets it from --tier or SD_TIER) and what depends on it.

quick   the default: every test without a deep or external mark, hypothesis derandomized
deep    quick and the deep tests, hypothesis random with ten times the examples
all     deep and the external tests (they run R or a Fortran compiler)
"""

import os

TIERS = ("quick", "deep", "all")
DEEP_FACTOR = 10

tier = os.environ.get("SD_TIER", "quick")  # replaced by tests/conftest.py at configure time


def examples(n):
    """A test's hypothesis example count: n in quick, DEEP_FACTOR * n in deep and all. Each
    test keeps its own n (the cost of one example differs); the tier is the one knob."""
    return n if tier == "quick" else DEEP_FACTOR * n
