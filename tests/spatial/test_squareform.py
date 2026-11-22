"""squareform (docs/api/spatial.md §2; decision SPT16): scipy's behaviour, its records in
tests/spatial/data/scipy.json, the dtype kept, the checks and their messages."""

import math

import numpy as np
import pytest
import scipy.spatial.distance as D

import stardust as sd

from .._bits import bits_equal
from . import refdata
from .conftest import LIVE

S = sd.spatial
SCIPY = refdata.data("scipy")


def _rec(key):
    v = SCIPY[f"squareform/{key}"]["value"]
    return np.asarray(v["array"], dtype=v["dtype"]).reshape(v["shape"])


def test_records():
    assert bits_equal(S.squareform(np.zeros(0)), _rec("empty"))
    assert bits_equal(S.squareform(np.zeros((1, 1))), _rec("1x1"))
    assert bits_equal(S.squareform(np.zeros((0, 0))), _rec("0x0"))
    assert bits_equal(S.squareform(np.array([True, False, True])), _rec("bool"))
    assert bits_equal(S.squareform(np.array([1, 2, 3], dtype=np.int64)), _rec("int"))
    assert bits_equal(S.squareform(np.array([[0.0, 1.0], [1.0, 0.0]]), force="tovector"), _rec("tovector"))
    for key, x in (("asym", [[0.0, 1.0], [2.0, 0.0]]), ("diag", [[1.0, 1.0], [1.0, 0.0]]),
                   ("nan", [[0.0, math.nan], [math.nan, 0.0]]), ("bad_length", np.zeros(4)),
                   ("3d", np.zeros((2, 2, 2)))):  # fmt: skip
        assert SCIPY[f"squareform/{key}"]["error"] == "ValueError"
        with pytest.raises(ValueError):
            S.squareform(x)


@LIVE
@pytest.mark.parametrize("d", [2, 3, 7, 40])
def test_round_trip_and_scipy(d):
    g = np.random.default_rng(d)
    v = g.standard_normal(d * (d - 1) // 2)
    m = S.squareform(v)
    assert bits_equal(m, D.squareform(v))
    assert bits_equal(S.squareform(m), v)
    x = g.standard_normal((d, 2))
    p = S.pdist(x)
    assert bits_equal(S.squareform(p), D.squareform(np.asarray(p)))
    full = np.asarray(S.cdist(x, x))
    assert bits_equal(S.squareform(full, checks=False), p)


def test_dtypes_kept():
    for dt in (np.bool_, np.int64, np.float64):
        v = np.array([1, 0, 1], dtype=dt)
        m = S.squareform(v)
        assert str(m.dtype) == np.dtype(dt).name
        assert str(S.squareform(m).dtype) == np.dtype(dt).name
    with pytest.raises(TypeError, match=r"^squareform\(\): 'X' must be real, not a complex128 array$"):
        S.squareform(np.array([1j, 0, 0]))


def test_force_and_checks():
    m = np.array([[0.0, 1.0], [1.5, 0.0]])
    assert bits_equal(S.squareform(m, checks=False), np.array([1.0]))
    assert bits_equal(S.squareform(np.array([2.0]), force="tomatrix"), np.array([[0.0, 2.0], [2.0, 0.0]]))
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' must be symmetric$"):
        S.squareform(m)
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' has NaN off the diagonal, so it is not symmetric$"):
        S.squareform([[0.0, math.nan], [math.nan, 0.0]])
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' must have a zero diagonal$"):
        S.squareform([[1.0, 2.0], [2.0, 0.0]])
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' must have a zero diagonal$"):
        S.squareform([[math.nan]])
    with pytest.raises(ValueError, match=r"^squareform\(\): force='tovector' needs a square matrix, not a 1-D 'X'$"):
        S.squareform([1.0], force="tovector")
    with pytest.raises(ValueError, match=r"^squareform\(\): force='tomatrix' needs a condensed vector, not a 2-D"):
        S.squareform([[0.0]], force="tomatrix")
    with pytest.raises(ValueError, match=r"^squareform\(\): the length of 'X' \(4\) is not d \(d - 1\) / 2 for any d$"):
        S.squareform(np.zeros(4))
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' must be square, not of shape \(2, 3\)$"):
        S.squareform(np.zeros((2, 3)))
    with pytest.raises(ValueError, match=r"^squareform\(\): 'X' must have 1 or 2 dimensions, not 3$"):
        S.squareform(np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match=r"^squareform\(\): 'force' must be 'no', 'tomatrix' or 'tovector', not 'x'$"):
        S.squareform([1.0], force="x")
    with pytest.raises(TypeError, match=r"^squareform\(\): 'checks' must be True or False, not int$"):
        S.squareform([1.0], checks=1)


def test_results_are_new_arrays():
    v = sd.asarray([1.0, 2.0, 3.0])
    m = S.squareform(v)
    m[0, 1] = 9.0
    assert v[0] == 1.0
