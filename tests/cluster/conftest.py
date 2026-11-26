"""Shared helpers of the sd.cluster tests: the comparison libraries the live comparisons are for."""

import numpy
import pytest
import scipy
import sklearn

# The live comparisons hold the non-rounding behavior to these versions (API.md §4, rule 2): another version is a
# failure to look at, never a reason to skip.
VERSIONS = {"scipy": "1.18.1", "numpy": "2.5.4", "scikit-learn": "1.9.1"}


@pytest.fixture
def reference_libraries():
    for name, mod in (("scipy", scipy), ("numpy", numpy), ("scikit-learn", sklearn)):
        if mod.__version__ != VERSIONS[name]:
            pytest.fail(f"the live {name} comparisons are for {name} {VERSIONS[name]}, not {mod.__version__}")


LIVE = pytest.mark.usefixtures("reference_libraries")
