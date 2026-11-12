"""asarray from nested sequences, ndarrays and Python scalars."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust as sd
from .conftest import assert_same

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [[1.5, 2], [3, 4]],
        [True, False],
        [True, 2],  # bool + int -> int64
        [1, 2.5],
        [1, 2j],
        (1, 2),
        [(1, 2), [3, 4]],
        [[[1], [2]], [[3], [4]]],
        [],
        [[]],
        [[], []],
        [-0.0, math.inf, -math.inf],
        [INT64_MIN, INT64_MAX],
        [2**63, 0.5],  # a big int is fine when the target is float64 (docs/api/array.md §1.5)
    ],
)
def test_nested_sequences_match_numpy(data):
    assert_same(sd.asarray(data), np.asarray(data))


def test_nan_survives():
    a = sd.asarray([math.nan, 1.0])
    assert math.isnan(a.tolist()[0])


@pytest.mark.parametrize("data", [[[1, 2], [3]], [1, [2]], [[1], 2], [[1, 2], [3, 4, 5]]])
def test_ragged_is_value_error(data):
    with pytest.raises(ValueError):
        sd.asarray(data)


@pytest.mark.parametrize("data", [["a"], [None], [1, object()], "abc", object(), {1: 2}])
def test_unsupported_types_are_type_errors(data):
    with pytest.raises(TypeError):
        sd.asarray(data)


@pytest.mark.parametrize("scalar", [5, 2.5, True, 1j])
def test_scalars_are_rejected_no_zero_dim(scalar):
    with pytest.raises(ValueError):  # docs/api/array.md §1.2
        sd.asarray(scalar)


def test_int_out_of_int64_range_overflows_only_when_target_is_int64():
    with pytest.raises(OverflowError):
        sd.asarray([2**63])
    with pytest.raises(OverflowError):
        sd.asarray([2**63], dtype=sd.int64)
    assert sd.asarray([2**70], dtype=sd.float64).tolist() == [float(2**70)]


# -------------------------------------------------- explicit dtype= rules --

def test_float_to_int_truncates_and_saturates():
    a = sd.asarray([1.5, -1.5, 2.7, math.nan, math.inf, -math.inf, 1e300], dtype=sd.int64)
    assert a.tolist() == [1, -1, 2, 0, INT64_MAX, INT64_MIN, INT64_MAX]


@pytest.mark.parametrize("target", [sd.int64, sd.float64])
def test_complex_to_real_is_type_error(target):
    with pytest.raises(TypeError):
        sd.asarray([1 + 0j], dtype=target)


def test_anything_to_bool_is_nonzero():
    assert sd.asarray([0j, 1j, 2, -0.0, math.nan, 0], dtype=sd.bool).tolist() == [
        False, True, True, False, True, False
    ]


def test_upcasts():
    assert sd.asarray([True, 3], dtype=sd.complex128).tolist() == [1 + 0j, 3 + 0j]


# ------------------------------------------------------- ndarray inputs ----

def test_ndarray_input_is_returned_as_is():
    a = sd.asarray([1, 2])
    assert sd.asarray(a) is a
    assert sd.asarray(a, dtype=sd.int64) is a


def test_copy_true_makes_a_new_array():
    a = sd.asarray([1, 2])
    b = sd.asarray(a, copy=True)
    assert b is not a and b.tolist() == a.tolist()


def test_copy_false_refuses_to_copy():
    a = sd.asarray([1, 2])
    with pytest.raises(ValueError):
        sd.asarray(a, dtype=sd.float64, copy=False)
    with pytest.raises(ValueError):
        sd.asarray([1, 2], copy=False)
    assert sd.asarray(a, copy=False) is a


def test_dtype_conversion_of_ndarray():
    a = sd.asarray([1.9, -1.9])
    assert sd.asarray(a, dtype=sd.int64).tolist() == [1, -1]
    with pytest.raises(TypeError):
        sd.asarray(sd.asarray([1j]), dtype=sd.float64)


def test_ndarrays_nested_in_lists():
    a = sd.asarray([1, 2])
    b = sd.asarray([3.5, 4])
    assert_same(sd.asarray([a, b]), np.asarray([[1, 2], [3.5, 4]]))
    with pytest.raises(ValueError):
        sd.asarray([a, sd.asarray([1, 2, 3])])


@pytest.mark.parametrize("bad", [1, "yes", 0])
def test_copy_argument_must_be_bool_or_none(bad):
    with pytest.raises(TypeError):
        sd.asarray([1], copy=bad)


def test_keyword_only_and_positional_only():
    with pytest.raises(TypeError):
        sd.asarray([1], sd.int64)  # options are keyword-only (docs/api/array.md §1.1)
    with pytest.raises(TypeError):
        sd.asarray(obj=[1])  # the array argument is positional-only


# ------------------------------------------------------ property tests -----

scalars = st.one_of(
    st.booleans(),
    st.integers(INT64_MIN, INT64_MAX),
    st.floats(allow_nan=True, allow_infinity=True),
    st.complex_numbers(allow_nan=True, allow_infinity=True),
)


@st.composite
def nested(draw):
    shape = draw(st.lists(st.integers(0, 4), min_size=1, max_size=4))
    kind = draw(st.sampled_from([st.booleans(), st.integers(INT64_MIN, INT64_MAX), st.floats(), scalars]))

    def build(dims):
        if not dims:
            return draw(kind)
        return [build(dims[1:]) for _ in range(dims[0])]

    return build(shape)


@given(nested())
def test_nested_roundtrip_matches_numpy(data):
    want = np.asarray(data)
    if want.dtype == object:  # mixed huge ints etc. that numpy keeps as objects
        return
    assert_same(sd.asarray(data), want)
