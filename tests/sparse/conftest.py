"""Shared helpers of the sd.sparse tests: the scipy the live comparisons are for, and the test
matrices (tests/sparse/matrices.py) as sd sparse arrays."""

import numpy as np
import pytest
import scipy

import stardust as sd

# The live comparisons hold the non-rounding behavior to this scipy (API.md §4, rule 2): another
# scipy is a failure to look at, never a reason to skip.
SCIPY_VERSION = "1.18.1"


@pytest.fixture
def reference_scipy():
    if scipy.__version__ != SCIPY_VERSION:
        pytest.fail(f"the live scipy comparisons are for scipy {SCIPY_VERSION}, not {scipy.__version__}")


LIVE = pytest.mark.usefixtures("reference_scipy")


def coo(m):
    """Matrix m as an sd coo_array of its coordinates as given (unsorted, duplicates kept)."""
    return sd.sparse.coo_array((sd.asarray(m.data), (sd.asarray(m.row), sd.asarray(m.col))), shape=m.shape)


def scipy_coo(m):
    import scipy.sparse as sp

    return sp.coo_array((m.data, (m.row, m.col)), shape=m.shape)


def npy(x):
    """An sd.ndarray (or a tuple of them) as numpy."""
    if isinstance(x, tuple):
        return tuple(np.asarray(v) for v in x)
    return np.asarray(x)


def i64(a):
    return np.asarray(a).astype(np.int64)
