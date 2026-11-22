"""Shared helpers of the sd.graph tests: the scipy and networkx the live comparisons are for."""

import networkx
import pytest
import scipy

# The live comparisons hold the non-rounding behavior to these versions (API.md §4, rule 2): another
# version is a failure to look at, never a reason to skip.
SCIPY_VERSION = "1.18.1"
NETWORKX_VERSION = "3.7"


@pytest.fixture
def reference_libraries():
    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the live scipy comparisons are for scipy {SCIPY_VERSION}, not {scipy.__version__}")
    if networkx.__version__ != NETWORKX_VERSION:
        pytest.fail(f"the live networkx comparisons are for {NETWORKX_VERSION}, not {networkx.__version__}")


LIVE = pytest.mark.usefixtures("reference_libraries")
