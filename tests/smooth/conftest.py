"""Shared helpers of the sd.smooth tests: the comparison libraries the live comparisons are for."""

import numpy
import pytest
import scipy
import sklearn
import statsmodels

# The live comparisons hold the non-rounding behavior to these versions (API.md §4, rule 2): another
# version is a failure to look at, never a reason to skip.
VERSIONS = {"scipy": "1.18.1", "numpy": "2.5.4", "statsmodels": "0.15.0", "scikit-learn": "1.9.1"}


@pytest.fixture
def reference_libraries():
    for name, mod in (("scipy", scipy), ("numpy", numpy), ("statsmodels", statsmodels), ("scikit-learn", sklearn)):
        if mod.__version__ != VERSIONS[name]:
            pytest.fail(f"the live {name} comparisons are for {name} {VERSIONS[name]}, not {mod.__version__}")


LIVE = pytest.mark.usefixtures("reference_libraries")
