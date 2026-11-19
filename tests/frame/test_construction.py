"""Building Series/DataFrame and getting data back out (docs/api/frame.md §2, §1.1)."""

import math
import operator

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples

NA = sd.frame.NA


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
    assert str(sd.frame.Series(values).dtype) == dtype


def test_none_and_na_are_na_but_nan_is_a_value():
    s = sd.frame.Series([1.0, None, NA, math.nan])
    out = s.tolist()
    assert out[0] == 1.0 and out[1] is NA and out[2] is NA and math.isnan(out[3])


@pytest.mark.parametrize(
    ("values", "error"),
    [
        (["a", 1], TypeError),
        (["a", True], TypeError),
        ([2**63], OverflowError),
        ([-(2**63) - 1], OverflowError),
        ([object()], TypeError),
    ],
)
def test_inference_errors(values, error):
    with pytest.raises(error):
        sd.frame.Series(values)


def test_inference_of_ints_with_floats_rounds_as_the_array_area():
    # int with float is float64, rounded; an int outside int64 becomes a float (docs/api/frame.md §2.2)
    assert sd.frame.Series([2**53 + 1, 0.5]).tolist() == [float(2**53 + 1), 0.5]
    assert sd.frame.Series([1.5, 2**70]).tolist() == [1.5, float(2**70)]
    assert sd.frame.Series([2**70], dtype="float64").tolist() == [float(2**70)]


def test_bool_promotes_in_inference_as_the_array_area():
    # bool < int64 < float64, as sd.asarray (DESIGN.md §10.1 #2, docs/api/frame.md §2.2)
    s = sd.frame.Series([True, 1, None])
    assert s.dtype == sd.int64 and s.tolist() == [1, 1, NA]
    assert sd.frame.Series([False, 1.5]).tolist() == [0.0, 1.5]
    assert sd.frame.Series([True, 2**70, 0.5]).tolist() == [1.0, float(2**70), 0.5]
    assert sd.frame.Series([np.True_, np.int32(3)]).tolist() == [1, 3]
    assert sd.frame.DataFrame([[True, 1], [2, False]], columns=["a", "b"]).dtypes == {"a": sd.int64, "b": sd.int64}
    for values in ([True, 1], [True, 1.5], [True, 1, 2.5]):
        assert sd.frame.Series(values).tolist() == sd.asarray(values).tolist()
    with pytest.raises(TypeError, match="str values cannot be mixed with bool, int values"):
        sd.frame.Series(["a", True, 1])


def test_big_exact_int_mixed_with_float_is_fine():
    assert sd.frame.Series([2**60, 0.5]).tolist() == [float(2**60), 0.5]


@settings(max_examples=examples(200), deadline=None)
@given(st.lists(st.none() | st.integers(-(2**63), 2**63 - 1)))
def test_int_round_trip(values):
    s = sd.frame.Series(values, dtype="int64")
    assert s.tolist() == [NA if v is None else v for v in values]


@settings(max_examples=examples(200), deadline=None)
@given(st.lists(st.none() | st.text()))
def test_string_round_trip(values):
    s = sd.frame.Series(values, dtype="string")
    assert s.tolist() == [NA if v is None else v for v in values]


@settings(max_examples=examples(200), deadline=None)
@given(st.lists(st.none() | st.floats()))
def test_float_round_trip_is_bit_exact(values):
    out = sd.frame.Series(values, dtype="float64").tolist()
    for v, o in zip(values, out):
        if v is None:
            assert o is NA
        else:
            assert np.float64(v).tobytes() == np.float64(o).tobytes()  # keeps -0.0 and NaN payloads


# --- dtype= -------------------------------------------------------------------


@pytest.mark.parametrize("spelling", ["int64", sd.int64])
def test_dtype_spellings(spelling):
    assert sd.frame.Series([1], dtype=spelling).dtype is sd.int64


@pytest.mark.parametrize("bad", [int, float, bool, str, sd.complex128, "complex128", "float32"])
def test_dtype_rejects_python_types_and_other_dtypes(bad):
    # the library's dtype objects or their names only (API.md §1)
    with pytest.raises(TypeError):
        sd.frame.Series([1], dtype=bad)


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
    s = sd.frame.Series(values, dtype=dtype)
    assert str(s.dtype) == dtype and s.tolist() == expected


@pytest.mark.parametrize(
    ("values", "dtype", "expected"),
    [
        ([1.5, -2.7], "int64", [1, -2]),
        ([math.nan], "int64", [0]),
        ([1e19, -1e19], "int64", [2**63 - 1, -(2**63)]),
        ([2**53 + 1], "float64", [float(2**53 + 1)]),
        ([True], "int64", [1]),
        ([1, 0], "bool", [True, False]),
        ([1], "string", ["1"]),
        (["1"], "int64", [1]),
    ],
)
def test_dtype_is_an_explicit_conversion(values, dtype, expected):
    # dtype= converts as astype does, as in the array area (docs/api/frame.md §2.3)
    s = sd.frame.Series(values, dtype=dtype)
    assert str(s.dtype) == dtype and s.tolist() == expected


@pytest.mark.parametrize("dtype", ["int32", "object"])
def test_unknown_dtype_errors(dtype):
    with pytest.raises(TypeError):
        sd.frame.Series([1], dtype=dtype)


# --- numpy in -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("arr", "dtype"),
    [
        (np.array([1, 2], dtype=np.int8), "int64"),
        (np.array([1, 2], dtype=np.int32), "int64"),
        (np.array([1, 2], dtype=np.uint32), "int64"),
        (np.array([1.5], dtype=np.float16), "float64"),
        (np.array([1.5], dtype=np.float32), "float64"),
        (np.array([True, False]), "bool"),
        (np.array(["a", "bc"]), "string"),
        (np.array([1, None], dtype=object), "int64"),
        (np.array([1, 2], dtype=">i8"), "int64"),
    ],
)
def test_numpy_widening(arr, dtype):
    s = sd.frame.Series(arr)
    assert str(s.dtype) == dtype
    assert s.tolist() == [NA if v is None else v for v in arr.tolist()]


def test_numpy_errors():
    # uint64 by its dtype, whatever the values, as sd.asarray (DESIGN.md §10.1 #10)
    for arr in (np.array([1], dtype=np.uint64), np.array([2**63], dtype=np.uint64)):
        with pytest.raises(TypeError, match=r"^Series\(\): uint64 buffers cannot be converted to int64 without loss$"):
            sd.frame.Series(arr)
        with pytest.raises(TypeError, match="uint64"):
            sd.frame.Series(arr, dtype="float64")
        with pytest.raises(TypeError, match="uint64"):
            sd.frame.DataFrame({"a": arr})
        with pytest.raises(TypeError, match="uint64"):
            sd.frame.Index(arr)
    with pytest.raises(TypeError):
        sd.frame.Series(np.array([1 + 2j]))
    with pytest.raises(ValueError):
        sd.frame.Series(np.zeros((2, 2)))


def test_numpy_is_copied():
    arr = np.array([1, 2, 3])
    s = sd.frame.Series(arr)
    arr[0] = 100
    assert s.tolist() == [1, 2, 3]  # CoW cannot see writes into foreign memory (docs/api/frame.md §2.4)


def test_numpy_strided():
    arr = np.arange(20, dtype=np.int16)[::-3]
    assert sd.frame.Series(arr).tolist() == arr.tolist()


def test_numpy_nan_is_a_value():
    s = sd.frame.Series(np.array([1.0, np.nan]))
    assert s.to_numpy().tolist()[0] == 1.0 and math.isnan(s.tolist()[1])


# --- to_numpy -----------------------------------------------------------------


def test_to_numpy_is_a_read_only_view():
    s = sd.frame.Series([1, 2, 3])
    a, b = s.to_numpy(), s.to_numpy()
    assert not a.flags.writeable
    assert np.shares_memory(a, b)
    c = s.to_numpy(copy=True)
    assert c.flags.writeable and not np.shares_memory(a, c)


@pytest.mark.parametrize(
    ("values", "kwargs"),
    [
        ([1, None], {"na_value": 0}),  # filling NA rows
        (["a"], {}),  # a string column becomes an object array
        ([1, 2], {"dtype": "float64"}),  # a dtype conversion
    ],
)
def test_to_numpy_copy_as_in_asarray(values, kwargs):
    # copy=None copies only when it must, True always, False never (docs/api/frame.md §2.7)
    s = sd.frame.Series(values)
    for copy in (None, True):
        assert s.to_numpy(copy=copy, **kwargs).tolist() == s.to_numpy(**kwargs).tolist()
    with pytest.raises(ValueError):
        s.to_numpy(copy=False, **kwargs)


def test_to_numpy_copy_without_a_copy_needed():
    s = sd.frame.Series([1.5, 2.5])
    view = s.to_numpy(copy=False)
    assert not view.flags.writeable and np.shares_memory(view, s.to_numpy())
    assert np.shares_memory(view, s.to_numpy(dtype="float64", copy=False))  # the same dtype is no conversion
    assert s.to_numpy(copy=True).flags.writeable
    for bad in (0, 1, "yes"):
        with pytest.raises(TypeError):
            s.to_numpy(copy=bad)
    with pytest.raises(ValueError):  # numpy asks with copy=False through the protocol
        np.asarray(sd.frame.Series([1, None]), copy=False)
    assert np.asarray(s, copy=True).flags.writeable and not np.asarray(s).flags.writeable
    with pytest.raises(ValueError):
        np.asarray(s, dtype=np.float32, copy=False)


def test_to_numpy_view_of_a_slice():
    s = sd.frame.Series(list(range(10)))
    assert s.tail(3).to_numpy().tolist() == [7, 8, 9]


def test_to_numpy_view_outlives_the_series():
    a = sd.frame.Series([1.5, 2.5]).to_numpy()
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
    arr = sd.frame.Series(values).to_numpy(**kwargs)
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
    ],
)
def test_to_numpy_errors(values, kwargs, error):
    with pytest.raises(error):
        sd.frame.Series(values).to_numpy(**kwargs)


def test_to_numpy_dtype_is_an_explicit_conversion():
    assert sd.frame.Series([2**53 + 1]).to_numpy(dtype="float64").tolist() == [float(2**53 + 1)]
    assert sd.frame.Series([True]).to_numpy(dtype="int64").tolist() == [1]


def test_numpy_protocol():
    assert np.asarray(sd.frame.Series([1, 2])).tolist() == [1, 2]
    assert np.asarray(sd.frame.Series([1, 2]), dtype=np.float32).dtype == np.float32


def test_frame_to_numpy():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [0.5, None]})
    assert df.to_numpy(na_value=math.inf).tolist() == [[1.0, 0.5], [2.0, math.inf]]
    assert sd.frame.DataFrame({"a": [1], "b": [2]}).to_numpy().dtype == np.int64
    assert sd.frame.DataFrame({"a": ["x"], "b": ["y"]}).to_numpy().tolist() == [["x", "y"]]
    assert sd.frame.DataFrame({"a": [1, 2]}).to_numpy().flags.writeable
    with pytest.raises(TypeError):
        sd.frame.DataFrame({"a": [1], "b": ["x"]}).to_numpy()
    # the common dtype promotes bool < int64 < float64, as in the array area (DESIGN.md §10.1 #2)
    mixed = sd.frame.DataFrame({"a": [1], "b": [True]}).to_numpy()
    assert mixed.dtype == np.int64 and mixed.tolist() == [[1, 1]]
    assert sd.frame.DataFrame({"a": [True], "b": [0.5]}).to_numpy().tolist() == [[1.0, 0.5]]
    assert sd.frame.DataFrame({"a": [True], "b": [False]}).to_numpy().dtype == np.bool_
    with pytest.raises(TypeError):
        sd.frame.DataFrame({"a": [True], "b": ["x"]}).to_numpy()
    with pytest.raises(ValueError):
        df.to_numpy()
    # a 2-D array of separate columns is always new: copy=False cannot hold (docs/api/frame.md §2.7)
    assert df.to_numpy(copy=True, na_value=0).tolist() == df.to_numpy(copy=None, na_value=0).tolist()
    with pytest.raises(ValueError):
        sd.frame.DataFrame({"a": [1.0]}).to_numpy(copy=False)
    with pytest.raises(TypeError):
        sd.frame.DataFrame({"a": [1.0]}).to_numpy(copy=1)
    # dtype= converts as astype does
    assert sd.frame.DataFrame({"a": [True], "b": [1]}).to_numpy(dtype="int64").tolist() == [[1, 1]]


# --- Series -------------------------------------------------------------------


def test_series_basics():
    s = sd.frame.Series([1, 2, 3], name="v")
    assert (len(s), s.shape, s.name, s.dtype) == (3, (3,), "v", sd.int64)
    assert s.index.equals(sd.frame.RangeIndex(3))
    with pytest.raises(TypeError):  # as an array's (DESIGN.md §10.1 #9)
        bool(s)


def test_series_name_must_be_str():
    with pytest.raises(TypeError):
        sd.frame.Series([1], name=0)
    s = sd.frame.Series([1])
    with pytest.raises(TypeError):
        s.name = 1
    s.name = "x"
    assert s.name == "x"


def test_series_from_series_shares_the_buffer():
    s = sd.frame.Series([1, 2], name="a")
    t = sd.frame.Series(s)
    assert t.name == "a" and t._col._shares_values_with(s._col)
    assert sd.frame.Series(s, name="b").name == "b"
    assert sd.frame.Series(s, dtype="float64").tolist() == [1.0, 2.0]


def test_series_rejects():
    with pytest.raises(TypeError):
        sd.frame.Series(5)
    with pytest.raises(TypeError):
        sd.frame.Series({"a": 1})  # labels come through index= (docs/api/frame.md §2.1)


def test_only_the_main_argument_is_positional():  # docs/api/frame.md §1.1
    with pytest.raises(TypeError):
        sd.frame.Series([1], "int64")
    with pytest.raises(TypeError):
        sd.frame.DataFrame([[1]], ["a"])
    with pytest.raises(TypeError):
        sd.frame.Series([1]).to_numpy("int64")


def test_main_arguments_are_positional_only(tmp_path):  # docs/api/frame.md §1.1: `/` before the main argument
    s, df = sd.frame.Series([1, 2]), sd.frame.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = tmp_path / "x.csv"
    path.write_text("a\n1\n")
    keyword_calls = [
        lambda: sd.frame.Series(data=[1]),
        lambda: sd.frame.DataFrame(data={"a": [1]}),
        lambda: sd.frame.Index(data=[1]),
        lambda: sd.frame.RangeIndex(stop=3),
        lambda: sd.frame.RangeIndex(0, 3, step=1),
        lambda: s.head(n=1),
        lambda: df.tail(n=1),
        lambda: s.astype(dtype="float64"),
        lambda: df.astype(dtype="float64"),
        lambda: s.fillna(value=0),
        lambda: df.fillna(value=0),
        lambda: s.reindex(labels=[0]),
        lambda: df.reindex(labels=[0]),
        lambda: s.index.equals(other=s.index),
        lambda: sd.frame.Index([1]).equals(other=sd.frame.Index([1])),
        lambda: df.duplicated(subset=["a"]),
        lambda: df.drop_duplicates(subset=["a"]),
        lambda: df.value_counts(subset=["a"]),
        lambda: df.groupby(by="a"),
        lambda: df.sort_values(by="a"),
        lambda: sd.frame.merge(left=df, right=df, on="a"),
        lambda: sd.frame.merge(df, right=df, on="a"),
        lambda: df.merge(right=df, on="a"),
        lambda: sd.frame.concat(objs=[s]),
        lambda: sd.frame.read_csv(path=path),
    ]
    for call in keyword_calls:
        with pytest.raises(TypeError):
            call()
    # the same calls with the main argument in place work
    assert sd.frame.RangeIndex(0, 3, 1).tolist() == [0, 1, 2]
    assert s.head(1).tolist() == [1] and df.tail(1)["b"].tolist() == ["y"]
    assert s.astype("float64").tolist() == [1.0, 2.0] and s.reindex([1]).tolist() == [2]
    assert df.sort_values("a", descending=True)["a"].tolist() == [2, 1]
    assert sd.frame.merge(df, df, on="a").shape == (2, 3) and df.merge(df, on="b").shape == (2, 3)
    assert sd.frame.concat([s, s]).tolist() == [1, 2, 1, 2] and sd.frame.read_csv(path)["a"].tolist() == [1]
    with pytest.raises(TypeError):
        s.astype("float64", copy=False)  # no copy: Copy-on-Write (docs/api/frame.md §1.1)


def test_tolist_has_no_alias():  # docs/api/frame.md §1.1
    for obj in (sd.frame.Series([1]), sd.frame.Index([1]), sd.frame.RangeIndex(2)):
        assert obj.tolist() == list(obj)
        with pytest.raises(AttributeError):
            obj.to_list()


def test_submodules_are_private():
    # sd.frame.index was a module, not an Index (docs/api/frame.md §1.1)
    for name in ("frame", "index", "io", "series", "groupby"):
        assert not hasattr(sd.frame, name), name
    assert sd.frame.merge.__module__ == "stardust.frame._merge" and callable(sd.frame.merge)
    assert sd.frame.__all__ == ["NA", "DataFrame", "Index", "RangeIndex", "Series", "concat", "merge", "read_csv"]


@pytest.mark.parametrize("n", [0, 1, 3, 5, 7, -1, -3, -7, 100, -100])
def test_head_tail_match_python_slicing(n):
    values = list(range(7))
    s = sd.frame.Series(values)
    expected_head = values[:n] if n >= 0 else values[: max(len(values) + n, 0)]
    expected_tail = (values[-n:] if n > 0 else []) if n >= 0 else values[-n:]
    assert s.head(n).tolist() == expected_head
    assert s.tail(n).tolist() == expected_tail
    assert s.tail(n).index.tolist() == [values.index(v) for v in expected_tail]


def test_head_shares_the_buffer_and_recounts_na():
    s = sd.frame.Series([1, None, 3, 4])
    h = s.tail(2)
    assert h._col._shares_values_with(s._col)
    assert h._col.null_count == 0 and s.head(2)._col.null_count == 1


# --- DataFrame ----------------------------------------------------------------


def test_frame_from_dict():
    df = sd.frame.DataFrame({"a": [1, 2], "b": np.array([0.5, 1.5]), "c": sd.frame.Series(["x", None])})
    assert df.columns.tolist() == ["a", "b", "c"]
    assert [str(d) for d in df.dtypes.values()] == ["int64", "float64", "string"] and list(
        df.dtypes
    ) == df.columns.tolist()
    assert df.shape == (2, 3) and len(df) == 2
    assert df["c"].tolist() == ["x", NA] and df["c"].name == "c"


def test_frame_dict_scalars_broadcast():
    df = sd.frame.DataFrame({"a": [1, 2, 3], "b": 0.5})
    assert df["b"].tolist() == [0.5, 0.5, 0.5]
    with pytest.raises(ValueError):
        sd.frame.DataFrame({"a": 1, "b": 2})


def test_frame_dict_errors():
    with pytest.raises(ValueError):
        sd.frame.DataFrame({"a": [1, 2], "b": [1]})
    with pytest.raises(TypeError):
        sd.frame.DataFrame({1: [1]})
    # Series with different labels are aligned to the union of labels (docs/api/frame.md §2.5).
    df = sd.frame.DataFrame({"a": sd.frame.Series([1, 2]), "b": sd.frame.Series([1, 2, 3]).tail(2)})
    assert df.index.tolist() == [0, 1, 2] and df["a"].tolist() == [1, 2, NA] and df["b"].tolist() == [NA, 2, 3]


def test_frame_columns_select_and_fill():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [3, 4]}, columns=["b", "z"])
    assert df.columns.tolist() == ["b", "z"]
    assert df["z"].dtype is sd.float64 and df["z"].tolist() == [NA, NA]
    with pytest.raises(ValueError):
        sd.frame.DataFrame({"a": [1]}, columns=["a", "a"])


def test_frame_from_records():
    df = sd.frame.DataFrame([{"a": 1, "b": "x"}, {"b": "y", "c": 2.5}])
    assert df.columns.tolist() == ["a", "b", "c"]
    assert df["a"].tolist() == [1, NA] and df["c"].tolist() == [NA, 2.5]


def test_frame_from_rows_needs_columns():
    df = sd.frame.DataFrame([[1, "x"], (2, "y")], columns=["n", "s"])
    assert [str(d) for d in df.dtypes.values()] == ["int64", "string"]
    with pytest.raises(ValueError):
        sd.frame.DataFrame([[1, 2]])
    with pytest.raises(ValueError):
        sd.frame.DataFrame([[1, 2], [3]], columns=["a", "b"])
    with pytest.raises(TypeError):
        sd.frame.DataFrame([1, 2, 3])


def test_frame_from_2d_numpy():
    arr = np.arange(6, dtype=np.int32).reshape(3, 2)
    df = sd.frame.DataFrame(arr, columns=["a", "b"])
    assert df["b"].tolist() == [1, 3, 5] and df["a"].dtype is sd.int64
    assert sd.frame.DataFrame(np.array([["x", "y"]]), columns=["p", "q"])["q"].tolist() == ["y"]
    with pytest.raises(ValueError):
        sd.frame.DataFrame(arr)
    with pytest.raises(ValueError):
        sd.frame.DataFrame(arr, columns=["a"])


def test_frame_dtype_applies_to_every_column():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}, dtype="float64")
    assert [str(d) for d in df.dtypes.values()] == ["float64", "float64"]
    assert sd.frame.DataFrame({"b": [3.5]}, dtype="int64")["b"].tolist() == [3]  # explicit, as astype


def test_frame_empty():
    assert sd.frame.DataFrame().shape == (0, 0)
    assert sd.frame.DataFrame([]).shape == (0, 0)
    df = sd.frame.DataFrame(columns=["a"], dtype="string")
    assert df.shape == (0, 1) and df["a"].dtype is sd.string


def test_frame_from_frame_shares_columns():
    df = sd.frame.DataFrame({"a": [1, 2]})
    assert sd.frame.DataFrame(df)["a"]._col._shares_values_with(df["a"]._col)


def test_frame_getitem():
    df = sd.frame.DataFrame({"a": [1]})
    with pytest.raises(KeyError):
        df["b"]
    with pytest.raises(TypeError):
        df[0]


def test_frame_head_tail():
    df = sd.frame.DataFrame({"a": list(range(10)), "b": [str(i) for i in range(10)]})
    t = df.tail(3)
    assert t["b"].tolist() == ["7", "8", "9"] and t.index.tolist() == [7, 8, 9]
    assert df.head(-8)["a"].tolist() == [0, 1]


# --- NA -----------------------------------------------------------------------


def test_na_singleton():
    import pickle

    assert repr(NA) == "<NA>"
    assert type(NA)() is NA
    assert pickle.loads(pickle.dumps(NA)) is NA
    with pytest.raises(TypeError):
        bool(NA)
    assert {NA: 1}[NA] == 1 and len({NA, NA}) == 1  # hashable; containers look up by identity


SCALARS = [0, 1, -3, 2.5, math.nan, True, False, "a", "", np.int32(2), np.float64(0.5), np.True_]
OPERATORS = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
    operator.and_,
    operator.or_,
    operator.xor,
    operator.lshift,
    operator.rshift,
]


def _one_row(op, x, na_left):
    """What a one-row Series gives: Series([x]) with NA on the same side (or a TypeError)."""
    s = sd.frame.Series([x])
    try:
        out = op(NA, s) if na_left else op(s, NA)
    except TypeError:
        return TypeError
    return out.tolist()[0]


@pytest.mark.parametrize("op", OPERATORS, ids=lambda f: f.__name__)
def test_na_scalar_operators_are_a_one_row_series(op):
    # DESIGN.md §10.1 #5, docs/api/frame.md §1.4: NA op x is Series([x])'s rule, unwrapped
    for x in SCALARS:
        for na_left in (True, False):
            want = _one_row(op, x, na_left)
            if want is TypeError:
                with pytest.raises(TypeError):
                    op(NA, x) if na_left else op(x, NA)
            else:
                got = op(NA, x) if na_left else op(x, NA)
                assert got is want or got == want, (op, x, na_left)
    assert op(NA, NA) is NA  # NA with NA is NA for every operator, == included


def test_na_scalar_logic_is_kleene():
    assert (NA | True) is True and (True | NA) is True and (NA | False) is NA
    assert (NA & False) is False and (False & NA) is False and (NA & True) is NA
    assert (NA ^ True) is NA and (NA | 1) is NA  # | with an int is bitwise: NA
    assert (NA == NA) is NA and (NA != 1) is NA and (1 < NA) is NA and (NA + "a") is NA and (NA**0) is NA
    for f in (operator.neg, operator.pos, abs, operator.invert):
        assert f(NA) is NA
    for bad in (lambda: NA - "a", lambda: NA & 1.5, lambda: NA << 0.5, lambda: NA + 2**70):
        with pytest.raises((TypeError, OverflowError)):
            bad()
    for other in ([1], None, np.array([1]), sd.asarray([1]), object()):  # as Series == other (DESIGN.md §10.1 #1)
        for op in (operator.eq, operator.ne, operator.add, operator.lt):
            with pytest.raises(TypeError):
                op(NA, other)
            with pytest.raises(TypeError):
                op(other, NA)
    s, df = sd.frame.Series([1, None]), sd.frame.DataFrame({"a": [True, None]})
    assert (NA + s).tolist() == [NA, NA] and (NA == s).tolist() == [NA, NA] and (NA | df)["a"].tolist() == [True, NA]
    assert (np.int64(1) + NA) is NA and (np.float64(1.5) < NA) is NA  # numpy scalars defer to NA


# --- pickling (DESIGN.md §10.1 #12) ---------------------------------------------


def test_frame_objects_pickle():
    import pickle

    s = sd.frame.Series([1.5, None, math.nan, -0.0], index=["a", "b", "c", "d"], name="v")
    df = sd.frame.DataFrame({"i": [1, None, 3], "s": ["x", None, "é"], "b": [True, False, None]}, index=[5, 6, 7])
    for obj in (s, df, s.iloc[1:3], df.iloc[1:], s.index, df.index, sd.frame.RangeIndex(2, 9, 3, name="r"), df.columns):
        back = pickle.loads(pickle.dumps(obj))
        assert type(back) is type(obj) and repr(back) == repr(obj)
    back = pickle.loads(pickle.dumps(s))
    assert back.index.equals(s.index) and back.name == "v"
    bits = lambda x: [np.float64(v).tobytes() for v in x.fillna(0.0).tolist()]
    assert bits(back) == bits(s)
    idx = sd.frame.Index(["a", "b"])
    assert "a" in idx  # builds the lookup table, which is not pickled
    assert pickle.loads(pickle.dumps(idx)).tolist() == ["a", "b"]
    sliced = pickle.loads(pickle.dumps(df.iloc[1:]))  # a slice pickles its own rows
    assert sliced["s"].tolist() == [NA, "é"] and sliced["b"].tolist() == [False, NA] and sliced.index.tolist() == [6, 7]
    assert (sliced["i"] + 1).tolist() == [NA, 4]


def test_construction_messages_name_the_problem():
    # DESIGN.md §8.4 and §10.3: the message says what is wrong
    with pytest.raises(ValueError, match=r"^DataFrame\(\): column 'a' has 2 values, but 'index' has 3 labels$"):
        sd.frame.DataFrame({"a": [1, 2]}, index=[1, 2, 3])
    with pytest.raises(ValueError, match=r"^DataFrame\(\): all columns must have the same length$"):
        sd.frame.DataFrame({"a": [1, 2], "b": [1]})
    for labels in ([None], [None, None], [sd.frame.NA]):
        with pytest.raises(ValueError, match=r"^Index\(\): an index cannot hold NA$"):
            sd.frame.Index(labels)
    with pytest.raises(TypeError, match=r"^Index\(\): index labels must be int64 or string, not float64$"):
        sd.frame.Index([1.5, None])
    df = sd.frame.DataFrame({"a": [1, None]})
    with pytest.raises(KeyError, match=r"astype\(\): no column 'zz'"):
        df.astype({"zz": "float64"})
    with pytest.raises(KeyError, match=r"fillna\(\): no column 'zz'"):
        df.fillna({"a": 0, "zz": 1})
    # column names: one message everywhere
    for call in (
        lambda: df[1],
        lambda: df.__setitem__(1, [1, 2]),
        lambda: sd.frame.DataFrame({1: [1]}),
        lambda: df.groupby("a")[1],
        lambda: sd.frame.merge(df, df, on=[1]),
    ):
        with pytest.raises(TypeError, match=r"column names must be str, not int$"):
            call()


def test_complex_values_are_values_the_frame_cannot_store():
    msg = r"cannot store an object of type 'complex' in a column \(only bool, int, float, str and None\)$"
    df = sd.frame.DataFrame({"a": [1, 2]})
    for call in (
        lambda: sd.frame.Series([1j]),
        lambda: sd.frame.DataFrame({"a": [1], "b": 1j}),
        lambda: df.__setitem__("c", 1j),
        lambda: sd.frame.Series([1, 2]).iloc.__setitem__(0, 1j),
    ):
        with pytest.raises(TypeError, match=msg):
            call()
    with pytest.raises(TypeError, match=r"^fillna\(\): 'value' must be None, a bool, int, float or str, not complex$"):
        sd.frame.Series([1.0, None]).fillna(1j)


def test_dropna_of_a_frame_without_columns():
    assert sd.frame.DataFrame().dropna().shape == (0, 0)
    df = sd.frame.DataFrame(index=[3, 4, 5])
    for kwargs in ({}, {"how": "all"}, {"subset": []}):
        got = df.dropna(**kwargs)  # no columns to look at: every row stays
        assert got.shape == (3, 0) and got.index.tolist() == [3, 4, 5]
    assert df.dropna(ignore_index=True).index.tolist() == [0, 1, 2]


def test_int_beyond_float_names_the_function():
    # the same message as the array area gives, with the public function in front (API.md §2.1).
    big = 10**400
    for call, name in (
        (lambda: sd.frame.Series([big, 0.5]), "Series"),
        (lambda: sd.frame.Series([big], dtype="float64"), "Series"),
        (lambda: sd.frame.DataFrame({"a": [big, 0.5]}), "DataFrame"),
        (lambda: sd.frame.Index([big, 0.5]), "Index"),
        (lambda: sd.frame.Series([1.0, None]).fillna(big), "fillna"),
    ):
        with pytest.raises(OverflowError, match=rf"^{name}\(\): Python int too large to convert to float64$"):
            call()
    with pytest.raises(OverflowError, match="^Python int too large to convert to float64$"):  # an operator: no name
        sd.frame.Series([1.0]) + big


def test_to_numpy_of_a_converted_column_is_writable():
    # A dtype conversion makes a new array: writable, as to_array's (docs/api/frame.md §2.7, §2.8).
    s = sd.frame.Series([1, 2, 3])
    for dtype, expect in (("float64", [1.0, 2.0, 3.0]), ("bool", [True, True, True])):
        for copy in (None, True):
            arr = s.to_numpy(dtype=dtype, copy=copy)
            assert arr.flags.writeable and arr.dtype == np.dtype(dtype) and arr.tolist() == expect
            arr[0] = 0
            assert s.tolist() == [1, 2, 3] and s.to_numpy(dtype=dtype).tolist() == expect
    assert not s.to_numpy().flags.writeable and not s.to_numpy(dtype="int64").flags.writeable  # no conversion: the view
    assert sd.frame.Series([1, None]).to_numpy(dtype="float64", na_value=0.5).flags.writeable
    arr = sd.frame.Series(["1", "2"]).to_numpy(dtype="int64")
    arr += 1
    assert arr.tolist() == [2, 3]


@pytest.mark.parametrize(
    "values,na_value,expected",
    [
        ([1, None], np.int64(-1), [1, -1]),
        ([1, None], np.int8(-1), [1, -1]),
        ([1, None], np.uint16(7), [1, 7]),
        ([1.5, None], np.float64(0.5), [1.5, 0.5]),
        ([1.5, None], np.float32(0.5), [1.5, 0.5]),
        ([1.5, None], np.int64(2), [1.5, 2.0]),
        ([True, None], np.bool_(False), [True, False]),
        ([True, None], np.True_, [True, True]),
    ],
)
def test_na_value_takes_numpy_scalars(values, na_value, expected):  # as fillna reads them
    s = sd.frame.Series(values)
    assert s.to_numpy(na_value=na_value).tolist() == expected
    assert s.to_array(na_value=na_value).tolist() == expected
    assert s.fillna(na_value).tolist() == expected
    assert sd.frame.DataFrame({"a": values}).to_numpy(na_value=na_value).tolist() == [[v] for v in expected]


def test_na_value_numpy_scalars_follow_their_python_kind():
    i, b = sd.frame.Series([1, None]), sd.frame.Series([True, None])
    for s, bad in ((i, np.float64(1.5)), (i, np.bool_(True)), (b, np.int64(1)), (b, np.float64(0.0))):
        with pytest.raises(TypeError, match=r"^to_numpy\(\): na_value=np\.\S+ does not fit an array of dtype"):
            s.to_numpy(na_value=bad)
        with pytest.raises(TypeError, match=r"^to_array\(\): na_value="):
            s.to_array(na_value=bad)


def test_frame_dict_values():  # docs/api/frame.md §2.5
    df = sd.frame.DataFrame({"t": (1, 2), "r": range(2), "a": sd.asarray([1.5, 2.5]), "s": 7, "n": None, "w": "x"})
    assert [df[k].tolist() for k in df] == [[1, 2], [0, 1], [1.5, 2.5], [7, 7], [NA, NA], ["x", "x"]]
    assert [str(d) for d in df.dtypes.values()] == ["int64", "int64", "float64", "int64", "float64", "string"]
    arr = sd.asarray([1.5, 2.5])
    df = sd.frame.DataFrame({"a": arr})
    arr[0] = 9.0  # always a copy (docs/api/frame.md §2.8)
    assert df["a"].tolist() == [1.5, 2.5]
    assert sd.frame.DataFrame({"a": 1}, index=[3, 4])["a"].tolist() == [1, 1]
    with pytest.raises(ValueError, match=r"^DataFrame\(\): cannot make a DataFrame from scalars alone"):
        sd.frame.DataFrame({"a": 1})
    with pytest.raises(ValueError, match=r"^DataFrame\(\): expected a 1-D array, got 2 dimensions$"):
        sd.frame.DataFrame({"a": sd.asarray([[1.5, 2.5]])})
