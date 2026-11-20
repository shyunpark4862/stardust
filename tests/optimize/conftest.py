"""Shared helpers of the optimize tests: the scipy the live comparisons are for, and the split of
the scipy sweeps between the quick and the deep tier."""

import pytest
import scipy

# The live comparisons follow scipy's steps call by call and the recorded runs
# (tests/optimize/data/scipy_*.json) were made with this version: another scipy is a failure
# to fix (record again or pin the version), never a reason to skip.
SCIPY_VERSION = "1.18.1"


@pytest.fixture
def reference_scipy():
    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the live scipy comparisons are for scipy {SCIPY_VERSION}, not {scipy.__version__}")


# On a test that runs scipy alongside sd and compares their steps.
LIVE = pytest.mark.usefixtures("reference_scipy")


def quick_first(cases, first):
    """The cases as parameters: those of problem `first` (a case's id starts with it) stay in the
    quick tier, the others are deep."""
    return [c if c[0].startswith(first + " ") else pytest.param(c, marks=pytest.mark.deep) for c in cases]
