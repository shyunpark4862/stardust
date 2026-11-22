"""Shared helpers of the sd.spatial tests: the scipy and scikit-learn the live comparisons are for."""

import pytest
import scipy
import sklearn

# The live comparisons hold the non-rounding behavior to these versions (API.md §4, rule 2): another
# version is a failure to look at, never a reason to skip.
SCIPY_VERSION = "1.18.1"
SKLEARN_VERSION = "1.9.1"


@pytest.fixture
def reference_libraries():
    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the live scipy comparisons are for scipy {SCIPY_VERSION}, not {scipy.__version__}")
    if sklearn.__version__ != SKLEARN_VERSION:
        pytest.fail(f"the live scikit-learn comparisons are for {SKLEARN_VERSION}, not {sklearn.__version__}")


LIVE = pytest.mark.usefixtures("reference_libraries")
