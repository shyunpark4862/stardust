"""Shared helpers of the sd.interpolate tests: the scipy and numpy the live comparisons are for."""

import numpy
import pytest
import scipy

# The live comparisons hold the non-rounding behavior to these versions (API.md §4, rule 2): another
# version is a failure to look at, never a reason to skip.
SCIPY_VERSION = "1.18.1"
NUMPY_VERSION = "2.5.4"


@pytest.fixture
def reference_libraries():
    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the live scipy comparisons are for scipy {SCIPY_VERSION}, not {scipy.__version__}")
    if numpy.__version__ != NUMPY_VERSION:
        pytest.fail(f"the live numpy comparisons are for numpy {NUMPY_VERSION}, not {numpy.__version__}")


LIVE = pytest.mark.usefixtures("reference_libraries")
