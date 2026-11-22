"""csgraph_from_dense and csgraph_to_dense of sd.graph (docs/api/graph.md §2, docs/design/graph.md §2.7;
milestone GR1): the definitions (every entry kept or left out by null_value, nan_null and infinity_null;
the stored entries at their places), dtypes, round trips, the argument errors, and scipy 1.18.1's results
on the same matrices."""

import itertools
import math

import numpy as np
import pytest
import scipy.sparse.csgraph as CS

import stardust as sd

from . import graphs as G
from .conftest import LIVE
from .forms import csr, square_ids

GR = sd.graph
SPECIAL = np.array(
    [[0.0, 1.0, np.nan, np.inf], [-np.inf, 0.0, -0.0, 2.5], [3.0, 0.0, 0.0, -1.0], [np.nan, 4.0, np.inf, 0.0]]
)


def from_dense_definition(M, null_value, nan_null, infinity_null):
    """{(i, j): value} of the entries kept (scipy's rule, read on the float64 matrix: a NaN null_value means
    nan_null, an infinite one infinity_null)."""
    if null_value is not None and math.isnan(null_value):
        null_value, nan_null = None, True
    elif null_value is not None and math.isinf(null_value):
        null_value, infinity_null = None, True
    out = {}
    for (i, j), x in np.ndenumerate(np.asarray(M, dtype=np.float64)):
        if null_value is not None and x == null_value:
            continue
        if nan_null and math.isnan(x):
            continue
        if infinity_null and math.isinf(x):
            continue
        out[(i, j)] = x
    return out


def stored(A):
    ip, ix, dt = (np.asarray(a) for a in (A.indptr, A.indices, A.data))
    return {(i, int(ix[k])): float(dt[k]) for i in range(A.shape[0]) for k in range(ip[i], ip[i + 1])}


def same(a: dict, b: dict):
    return a.keys() == b.keys() and all(
        np.float64(a[k]).tobytes() == np.float64(b[k]).tobytes() or (math.isnan(a[k]) and math.isnan(b[k])) for k in a
    )


def test_from_dense_equals_the_definition():
    for null_value, nan_null, infinity_null in itertools.product(
        (0, 0.0, -1, 2.5, np.inf, -np.inf, np.nan, None), (True, False), (True, False)
    ):
        A = GR.csgraph_from_dense(SPECIAL, null_value=null_value, nan_null=nan_null, infinity_null=infinity_null)
        assert isinstance(A, sd.sparse.csr_array) and A.dtype == sd.float64 and A.shape == (4, 4)
        assert same(stored(A), from_dense_definition(SPECIAL, null_value, nan_null, infinity_null)), (
            null_value,
            nan_null,
            infinity_null,
        )


def test_from_dense_dtypes_and_forms():
    B = np.array([[True, False], [False, True]])
    I = np.array([[0, -3], [2**53 + 1, 0]])
    assert stored(GR.csgraph_from_dense(B)) == {(0, 0): 1.0, (1, 1): 1.0}
    assert stored(GR.csgraph_from_dense(I)) == {(0, 1): -3.0, (1, 0): float(2**53 + 1)}
    assert stored(GR.csgraph_from_dense(sd.asarray(I).T)) == {(1, 0): -3.0, (0, 1): float(2**53 + 1)}
    assert stored(GR.csgraph_from_dense(SPECIAL.tolist())) == stored(GR.csgraph_from_dense(SPECIAL))
    assert GR.csgraph_from_dense(np.zeros((0, 0))).shape == (0, 0)
    # null_value=inf keeps the zeros: zero-weight edges (scipy's way to give them densely)
    assert stored(GR.csgraph_from_dense([[0.0, np.inf], [1.0, 0.0]], null_value=np.inf)) == {
        (0, 0): 0.0,
        (1, 0): 1.0,
        (1, 1): 0.0,
    }


@pytest.mark.parametrize("gid", square_ids())
def test_to_dense_and_back(gid):
    g = G.get(gid)
    A = csr(g)
    for null in (0.0, np.inf, -7.0, np.nan):
        M = GR.csgraph_to_dense(A, null_value=null)
        assert isinstance(M, sd.ndarray) and M.dtype == sd.float64 and M.shape == g.shape and M._flags()["writeable"]
        want = np.full(g.shape, null)
        for i, j, w in g.edges():
            want[i, j] = w
        assert (
            np.array_equal(np.asarray(M), want, equal_nan=True)
            and np.asarray(M).tobytes() == want.tobytes()
            or (np.isnan(null) and np.array_equal(np.asarray(M), want, equal_nan=True))
        ), (gid, null)
    if not np.any(np.isnan(g.data.astype(float))) and not np.any(g.data.astype(float) == 0):
        back = GR.csgraph_from_dense(GR.csgraph_to_dense(A, null_value=np.nan), null_value=None, infinity_null=False)
        assert same(stored(back), stored(A.astype(sd.float64))), gid


def test_to_dense_of_a_dense_input_reads_the_graph_rule():
    M = GR.csgraph_to_dense(SPECIAL, null_value=-5.0)
    want = np.where((SPECIAL != 0) & np.isfinite(SPECIAL), SPECIAL, -5.0)
    assert np.array_equal(np.asarray(M), want)


def test_errors():
    for bad in (None, "x", 1j, True, [1.0]):
        with pytest.raises(TypeError, match=r"^csgraph_to_dense\(\): 'null_value' must be an int or a float, not "):
            GR.csgraph_to_dense(SPECIAL, null_value=bad)
    for bad in ("x", 1j, True):
        with pytest.raises(TypeError, match=r"^csgraph_from_dense\(\): 'null_value' must be an int or a float, not "):
            GR.csgraph_from_dense(SPECIAL, null_value=bad)
    for arg in ("nan_null", "infinity_null"):
        with pytest.raises(TypeError, match=rf"^csgraph_from_dense\(\): '{arg}' must be True or False, not int$"):
            GR.csgraph_from_dense(SPECIAL, **{arg: 1})
    with pytest.raises(TypeError, match=r"^csgraph_from_dense\(\): 'graph' must be a dense matrix, not csr_array$"):
        GR.csgraph_from_dense(sd.sparse.csr_array(SPECIAL))
    with pytest.raises(TypeError, match=r"^csgraph_from_dense\(\): 'graph' must be real, not a complex128 array$"):
        GR.csgraph_from_dense(np.eye(2) * 1j)
    with pytest.raises(
        ValueError, match=r"^csgraph_from_dense\(\): 'graph' must be a square matrix, not of shape \(2, 3\)$"
    ):
        GR.csgraph_from_dense(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"^csgraph_from_dense\(\): 'graph' must have 2 dimensions, not 1$"):
        GR.csgraph_from_dense([1.0, 2.0])
    with pytest.raises(TypeError, match=r"^csgraph_from_dense\(\): "):
        GR.csgraph_from_dense(SPECIAL, 0)
    with pytest.raises(OverflowError, match=r"^csgraph_to_dense\(\): 'null_value' is out of range$"):
        GR.csgraph_to_dense(SPECIAL, null_value=10**400)


@LIVE
def test_scipy_gives_the_same():
    for null_value, nan_null, infinity_null in itertools.product((0, -1, np.inf, None), (True, False), (True, False)):
        got = GR.csgraph_from_dense(SPECIAL, null_value=null_value, nan_null=nan_null, infinity_null=infinity_null)
        want = CS.csgraph_from_dense(SPECIAL, null_value=null_value, nan_null=nan_null, infinity_null=infinity_null)
        assert same(stored(got), stored(want)), (null_value, nan_null, infinity_null)
    for gid in square_ids():
        g = G.get(gid)
        if np.any(np.isnan(g.data.astype(np.float64))):
            continue  # below
        for null in (0.0, np.inf):
            got = np.asarray(GR.csgraph_to_dense(csr(g), null_value=null))
            assert np.array_equal(got, CS.csgraph_to_dense(g.scipy(), null_value=null), equal_nan=True), gid


@LIVE
def test_scipy_turns_a_stored_nan_into_inf():
    # decision GR30: csgraph_to_dense writes the stored values as they are; scipy's fill (the minimum of
    # duplicates over a +inf start) loses a stored NaN
    g = G.get("nonfinite/nan")
    assert CS.csgraph_to_dense(g.scipy())[1, 2] == np.inf
    assert np.isnan(np.asarray(GR.csgraph_to_dense(csr(g)))[1, 2])
