"""M1: building Series/DataFrame and getting data back out (API.md §10, §11)."""

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust_pandas as sdpd

NA = sdpd.NA


# --- list inference -----------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([True, False], "bool"),
        ([1, 2], "int64"),
        ([1.5, 2.0], "float64"),
        ([1, 2.5], "float64"),
        (["a", "b"], "string"),
        ([], "float64"),
        ([None, NA], "float64"),
        ([1, None], "int64"),
        ([math.nan], "float64"),
        ((1, 2), "int64"),
        (range(3), "int64"),
        ([np.int32(1), np.int64(2)], "int64"),
        ([np.float32(1.5)], "float64"),
        ([np.bool_(True)], "bool"),
        ([np.str_("x")], "string"),
    ],
)
def test_inference(values, dtype):
    assert sdpd.Series(values).dtype == dtype


def test_none_and_na_are_na_but_nan_is_a_value():
    s = sdpd.Series([1.0, None, NA, math.nan])
    out = s.to_list()
    assert out[0] == 1.0 and out[1] is NA and out[2] is NA and math.isnan(out[3])


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ([True, 1], TypeError),
        ([True, 1.5], TypeError),
        (["a", 1], TypeError),
        (["a", True], TypeError),
        ([2**63], OverflowError),
        ([-(2**63) - 1], OverflowError),
        ([2**53 + 1, 0.5], ValueError),
        ([object()], TypeError),
    ],
)
def test_inference_errors(values, error):
    with pytest.raises(error):
        sdpd.Series(values)


def test_big_exact_int_mixed_with_float_is_fine():
    assert sdpd.Series([2**60, 0.5]).to_list() == [float(2**60), 0.5]


@settings(max_examples=200, deadline=None)
@given(st.lists(st.none() | st.integers(-(2**63), 2**63 - 1)))
def test_int_round_trip(values):
    s = sdpd.Series(values, dtype="int64")
    assert s.to_list() == [NA if v is None else v for v in values]


@settings(max_examples=200, deadline=None)
@given(st.lists(st.none() | st.text()))
def test_string_round_trip(values):
    s = sdpd.Series(values, dtype="string")
    assert s.to_list() == [NA if v is None else v for v in values]


@settings(max_examples=200, deadline=None)
@given(st.lists(st.none() | st.floats()))
def test_float_round_trip_is_bit_exact(values):
    out = sdpd.Series(values, dtype="float64").to_list()
    for v, o in zip(values, out):
        if v is None:
            assert o is NA
        else:
            assert np.float64(v).tobytes() == np.float64(o).tobytes()  # keeps -0.0 and NaN payloads


# --- dtype= -------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["int64", int])
def test_dtype_spellings(spelling):
    assert sdpd.Series([1], dtype=spelling).dtype == "int64"


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1, 2], "float64", [1.0, 2.0]),
        ([1.0, -2.0, None], "int64", [1, -2, NA]),
        ([], "string", []),
        ([None], "bool", [NA]),
        ([2**60], "float64", [float(2**60)]),
    ],
)
def test_lossless_dtype(values, dtype, expected):
    s = sdpd.Series(values, dtype=dtype)
    assert s.dtype == dtype and s.to_list() == expected


@pytest.mark.parametrize(
    ("values", "dtype", "error"),
    [
        ([1.5], "int64", ValueError),
        ([math.nan], "int64", ValueError),
        ([1e19], "int64", ValueError),
        ([2**53 + 1], "float64", ValueError),
        ([True], "int64", TypeError),
        ([1], "bool", TypeError),
        ([1], "string", TypeError),
        (["1"], "int64", TypeError),
        ([1], "int32", TypeError),
        ([1], "object", TypeError),
    ],
)
def test_lossy_dtype_errors(values, dtype, error):
    with pytest.raises(error):
        sdpd.Series(values, dtype=dtype)


# --- numpy in -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("arr", "dtype"),
    [
        (np.array([1, 2], dtype=np.int8), "int64"),
        (np.array([1, 2], dtype=np.int32), "int64"),
        (np.array([1, 2], dtype=np.uint32), "int64"),
        (np.array([2**63 - 1], dtype=np.uint64), "int64"),
        (np.array([1.5], dtype=np.float16), "float64"),
        (np.array([1.5], dtype=np.float32), "float64"),
        (np.array([True, False]), "bool"),
        (np.array(["a", "bc"]), "string"),
        (np.array([1, None], dtype=object), "int64"),
        (np.array([1, 2], dtype=">i8"), "int64"),
    ],
)
def test_numpy_widening(arr, dtype):
    s = sdpd.Series(arr)
    assert s.dtype == dtype
    assert s.to_list() == [NA if v is None else v for v in arr.tolist()]


def test_numpy_errors():
    with pytest.raises(OverflowError):
        sdpd.Series(np.array([2**63], dtype=np.uint64))
    with pytest.raises(TypeError):
        sdpd.Series(np.array([1 + 2j]))
    with pytest.raises(ValueError):
        sdpd.Series(np.zeros((2, 2)))


def test_numpy_is_copied():
    arr = np.array([1, 2, 3])
    s = sdpd.Series(arr)
    arr[0] = 100
    assert s.to_list() == [1, 2, 3]  # CoW cannot see writes into foreign memory (API.md §10)


def test_numpy_strided():
    arr = np.arange(20, dtype=np.int16)[::-3]
    assert sdpd.Series(arr).to_list() == arr.tolist()


def test_numpy_nan_is_a_value():
    s = sdpd.Series(np.array([1.0, np.nan]))
    assert s.to_numpy().tolist()[0] == 1.0 and math.isnan(s.to_list()[1])


# --- to_numpy -----------------------------------------------------------------


def test_to_numpy_is_a_read_only_view():
    s = sdpd.Series([1, 2, 3])
    a, b = s.to_numpy(), s.to_numpy()
    assert not a.flags.writeable
    assert np.shares_memory(a, b)
    c = s.to_numpy(copy=True)
    assert c.flags.writeable and not np.shares_memory(a, c)


def test_to_numpy_view_of_a_slice():
    s = sdpd.Series(list(range(10)))
    assert s.tail(3).to_numpy().tolist() == [7, 8, 9]


def test_to_numpy_view_outlives_the_series():
    a = sdpd.Series([1.5, 2.5]).to_numpy()
    assert a.tolist() == [1.5, 2.5]


@pytest.mark.parametrize(
    ("values", "kwargs", "expected", "np_dtype"),
    [
        ([1, None], {"na_value": -1}, [1, -1], np.int64),
        ([True, None], {"na_value": False}, [True, False], np.bool_),
        ([1.5, None], {"na_value": 0}, [1.5, 0.0], np.float64),
        ([1, None], {"dtype": "float64", "na_value": math.inf}, [1.0, math.inf], np.float64),
        (["a", None], {"na_value": None}, ["a", None], object),
        (["a", "b"], {}, ["a", "b"], object),
    ],
)
def test_to_numpy_fill(values, kwargs, expected, np_dtype):
    arr = sdpd.Series(values).to_numpy(**kwargs)
    assert arr.dtype == np_dtype and arr.tolist() == expected


@pytest.mark.parametrize(
    ("values", "kwargs", "error"),
    [
        ([1, None], {}, ValueError),
        (["a", None], {}, ValueError),
        ([1, None], {"na_value": math.nan}, TypeError),
        ([1, None], {"na_value": True}, TypeError),
        ([True, None], {"na_value": 0}, TypeError),
        (["a", None], {"na_value": 0}, TypeError),
        ([1], {"na_value": 0.5}, TypeError),  # checked even without NA: errors do not depend on the data
        ([2**53 + 1], {"dtype": "float64"}, ValueError),
        ([True], {"dtype": "int64"}, TypeError),
    ],
)
def test_to_numpy_errors(values, kwargs, error):
    with pytest.raises(error):
        sdpd.Series(values).to_numpy(**kwargs)


def test_numpy_protocol():
    assert np.asarray(sdpd.Series([1, 2])).tolist() == [1, 2]
    assert np.asarray(sdpd.Series([1, 2]), dtype=np.float32).dtype == np.float32


def test_frame_to_numpy():
    df = sdpd.DataFrame({"a": [1, 2], "b": [0.5, None]})
    assert df.to_numpy(na_value=math.inf).tolist() == [[1.0, 0.5], [2.0, math.inf]]
    assert sdpd.DataFrame({"a": [1], "b": [2]}).to_numpy().dtype == np.int64
    assert sdpd.DataFrame({"a": ["x"], "b": ["y"]}).to_numpy().tolist() == [["x", "y"]]
    assert sdpd.DataFrame({"a": [1, 2]}).to_numpy().flags.writeable
    with pytest.raises(TypeError):
        sdpd.DataFrame({"a": [1], "b": ["x"]}).to_numpy()
    with pytest.raises(TypeError):
        sdpd.DataFrame({"a": [1], "b": [True]}).to_numpy()
    with pytest.raises(ValueError):
        df.to_numpy()
    with pytest.raises(TypeError):  # dtype= converts without loss only; bool -> int64 is astype's job
        sdpd.DataFrame({"a": [True], "b": [1]}).to_numpy(dtype="int64")


# --- Series -------------------------------------------------------------------


def test_series_basics():
    s = sdpd.Series([1, 2, 3], name="v")
    assert (len(s), s.shape, s.name, s.dtype) == (3, (3,), "v", "int64")
    assert s.index.equals(sdpd.RangeIndex(3))
    with pytest.raises(ValueError):
        bool(s)


def test_series_name_must_be_str():
    with pytest.raises(TypeError):
        sdpd.Series([1], name=0)
    s = sdpd.Series([1])
    with pytest.raises(TypeError):
        s.name = 1
    s.name = "x"
    assert s.name == "x"


def test_series_from_series_shares_the_buffer():
    s = sdpd.Series([1, 2], name="a")
    t = sdpd.Series(s)
    assert t.name == "a" and t._col._shares_values_with(s._col)
    assert sdpd.Series(s, name="b").name == "b"
    assert sdpd.Series(s, dtype="float64").to_list() == [1.0, 2.0]


def test_series_rejects():
    with pytest.raises(TypeError):
        sdpd.Series(5)
    with pytest.raises(TypeError):
        sdpd.Series({"a": 1})  # labels come through index= (API.md §14)


def test_only_the_main_argument_is_positional():  # API.md §11 C2
    with pytest.raises(TypeError):
        sdpd.Series([1], "int64")
    with pytest.raises(TypeError):
        sdpd.DataFrame([[1]], ["a"])
    with pytest.raises(TypeError):
        sdpd.Series([1]).to_numpy("int64")


@pytest.mark.parametrize("n", [0, 1, 3, 5, 7, -1, -3, -7, 100, -100])
def test_head_tail_match_python_slicing(n):
    values = list(range(7))
    s = sdpd.Series(values)
    expected_head = values[:n] if n >= 0 else values[: max(len(values) + n, 0)]
    expected_tail = (values[-n:] if n > 0 else []) if n >= 0 else values[-n:]
    assert s.head(n).to_list() == expected_head
    assert s.tail(n).to_list() == expected_tail
    assert s.tail(n).index.to_list() == [values.index(v) for v in expected_tail]


def test_head_shares_the_buffer_and_recounts_na():
    s = sdpd.Series([1, None, 3, 4])
    h = s.tail(2)
    assert h._col._shares_values_with(s._col)
    assert h._col.null_count == 0 and s.head(2)._col.null_count == 1


# --- DataFrame ----------------------------------------------------------------


def test_frame_from_dict():
    df = sdpd.DataFrame({"a": [1, 2], "b": np.array([0.5, 1.5]), "c": sdpd.Series(["x", None])})
    assert df.columns.to_list() == ["a", "b", "c"]
    assert df.dtypes.to_list() == ["int64", "float64", "string"] and df.dtypes.index.equals(df.columns)
    assert df.shape == (2, 3) and len(df) == 2
    assert df["c"].to_list() == ["x", NA] and df["c"].name == "c"


def test_frame_dict_scalars_broadcast():
    df = sdpd.DataFrame({"a": [1, 2, 3], "b": 0.5})
    assert df["b"].to_list() == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        sdpd.DataFrame({"a": 1, "b": 2})


def test_frame_dict_errors():
    with pytest.raises(ValueError):
        sdpd.DataFrame({"a": [1, 2], "b": [1]})
    with pytest.raises(TypeError):
        sdpd.DataFrame({1: [1]})
    # Series with different labels are aligned to the union of labels (API.md §14).
    df = sdpd.DataFrame({"a": sdpd.Series([1, 2]), "b": sdpd.Series([1, 2, 3]).tail(2)})
    assert df.index.to_list() == [0, 1, 2] and df["a"].to_list() == [1, 2, NA] and df["b"].to_list() == [NA, 2, 3]


def test_frame_columns_select_and_fill():
    df = sdpd.DataFrame({"a": [1, 2], "b": [3, 4]}, columns=["b", "z"])
    assert df.columns.to_list() == ["b", "z"]
    assert df["z"].dtype == "float64" and df["z"].to_list() == [NA, NA]
    with pytest.raises(ValueError):
        sdpd.DataFrame({"a": [1]}, columns=["a", "a"])


def test_frame_from_records():
    df = sdpd.DataFrame([{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}])
    assert df.columns.to_list() == ["a", "b", "c"]
    assert df["a"].to_list() == [1, NA] and df["c"].to_list() == [NA, 2.5]


def test_frame_from_rows_needs_columns():
    df = sdpd.DataFrame([[1, "x"], (2, "y")], columns=["n", "s"])
    assert df.dtypes.to_list() == ["int64", "string"]
    with pytest.raises(ValueError):
        sdpd.DataFrame([[1, 2]])
    with pytest.raises(ValueError):
        sdpd.DataFrame([[1, 2], [3]], columns=["a", "b"])
    with pytest.raises(TypeError):
        sdpd.DataFrame([1, 2, 3])


def test_frame_from_2d_numpy():
    arr = np.arange(6, dtype=np.int32).reshape(3, 2)
    df = sdpd.DataFrame(arr, columns=["a", "b"])
    assert df["b"].to_list() == [1, 3, 5] and df["a"].dtype == "int64"
    assert sdpd.DataFrame(np.array([["x", "y"]]), columns=["p", "q"])["q"].to_list() == ["y"]
    with pytest.raises(ValueError):
        sdpd.DataFrame(arr)
    with pytest.raises(ValueError):
        sdpd.DataFrame(arr, columns=["a"])


def test_frame_dtype_applies_to_every_column():
    df = sdpd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}, dtype="float64")
    assert df.dtypes.to_list() == ["float64", "float64"]
    with pytest.raises(ValueError):
        sdpd.DataFrame({"b": [3.5]}, dtype="int64")


def test_frame_empty():
    assert sdpd.DataFrame().shape == (0, 0)
    assert sdpd.DataFrame([]).shape == (0, 0)
    df = sdpd.DataFrame(columns=["a"], dtype="string")
    assert df.shape == (0, 1) and df["a"].dtype == "string"


def test_frame_from_frame_shares_columns():
    df = sdpd.DataFrame({"a": [1, 2]})
    assert sdpd.DataFrame(df)["a"]._col._shares_values_with(df["a"]._col)


def test_frame_getitem():
    df = sdpd.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        df["b"]
    with pytest.raises(TypeError):
        df[0]


def test_frame_head_tail():
    df = sdpd.DataFrame({"a": list(range(10)), "b": [str(i) for i in range(10)]})
    t = df.tail(3)
    assert t["b"].to_list() == ["7", "8", "9"] and t.index.to_list() == [7, 8, 9]
    assert df.head(-8)["a"].to_list() == [0, 1]


# --- NA -----------------------------------------------------------------------


def test_na_singleton():
    import pickle

    assert repr(NA) == "<NA>"
    assert type(NA)() is NA
    assert pickle.loads(pickle.dumps(NA)) is NA
    with pytest.raises(TypeError):
        bool(NA)
