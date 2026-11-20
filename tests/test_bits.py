"""The comparison helpers of tests/_bits.py keep their three meanings apart; max_rel_error
measures element by element."""

import math

import numpy as np
import pytest

import stardust as sd
from tests._bits import bits_equal, bits_equal_any_nan, max_rel_error, value_equal

QNAN = math.nan
NAN_PAYLOAD = np.array([0x7FF8000000000001], dtype=np.uint64).view(np.float64)[0]
NEG_NAN = -math.nan


def test_the_three_meanings():
    # signed zeros: only value_equal takes -0.0 for 0.0
    assert not bits_equal(0.0, -0.0) and not bits_equal_any_nan(0.0, -0.0) and value_equal(0.0, -0.0)
    # NaNs: bits_equal sees the payload and the sign, the other two take any NaN for any NaN
    assert bits_equal(QNAN, QNAN) and not bits_equal(QNAN, NAN_PAYLOAD) and not bits_equal(QNAN, NEG_NAN)
    assert bits_equal_any_nan(QNAN, NAN_PAYLOAD) and bits_equal_any_nan(QNAN, NEG_NAN)
    assert value_equal(QNAN, NAN_PAYLOAD) and not value_equal(QNAN, 1.0)
    # complex: part by part
    assert bits_equal_any_nan(complex(QNAN, 0.0), complex(NAN_PAYLOAD, 0.0))
    assert not bits_equal_any_nan(complex(1.0, 0.0), complex(1.0, -0.0)) and value_equal(1 + 0j, complex(1, -0.0))
    # dtypes: the bit comparisons want the same dtype, value_equal compares values
    assert not bits_equal(1, 1.0) and not bits_equal_any_nan(1, 1.0) and value_equal(1, 1.0)
    assert not bits_equal(np.array([True]), np.array([1]))


def test_arrays_sequences_and_shapes():
    x = np.array([1.0, -0.0, QNAN])
    assert bits_equal(sd.asarray(x), x) and bits_equal(x[::-1], x[::-1].copy())  # any layout
    assert bits_equal_any_nan(x, [1.0, -0.0, NAN_PAYLOAD]) and not bits_equal_any_nan(x, [1.0, 0.0, QNAN])
    assert not value_equal(x, x[:2]) and not bits_equal(x.reshape(3, 1), x)  # shapes never broadcast
    # a result tuple field by field, fields of different shapes and types
    r = (np.eye(2), np.array([1, 0]), 3.0)
    assert bits_equal(r, (np.eye(2), np.array([1, 0]), 3.0))
    assert not bits_equal(r, (np.eye(2), np.array([0, 1]), 3.0)) and not bits_equal(r, r[:2])
    assert bits_equal((1.0, None, "a"), [1.0, None, "a"]) and not bits_equal((None,), (0,))


def test_max_rel_error():
    # the largest element-wise error, not a norm-wise one: the small element counts
    assert max_rel_error([1.0, 1e-10 * (1 + 1e-6)], [1.0, 1e-10]) == pytest.approx(1e-6)
    assert max_rel_error(sd.asarray([2.0, 4.0]), np.array([2.0, 4.0])) == 0.0
    assert max_rel_error(2.0, 4.0) == 0.5
    # a NaN, or a zero reference that the value misses, fails any bound
    assert math.isnan(max_rel_error([QNAN], [1.0])) and math.isnan(max_rel_error([None], [1.0]))
    assert max_rel_error([1e-300], [0.0]) == math.inf
    with pytest.raises(ValueError, match="shapes"):
        max_rel_error([1.0, 2.0], [1.0])
