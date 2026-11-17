"""Index, selection, alignment and assignment with Copy-on-Write (docs/api/frame.md §5).

pandas is the oracle where the rules agree (lookups on unique labels,
position slicing, repr of named and string indexes). Where they differ on
purpose (alignment order, scalar lookups on duplicates, masks) the oracle
is plain Python over the labels.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

NA = sd.frame.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))


def pd_index_repr(idx):
    # the dtype as the array's repr writes it: dtype=int64, not dtype='int64' (DESIGN.md §10.2)
    return repr(idx).replace("dtype='str'", "dtype=string").replace("dtype='int64'", "dtype=int64")


# --- Index ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "labels,name",
    [([], None), (["a"], None), (["a", "b"], "n"), (list("abcdefghijklmnopqrstuvwxyz") * 5, None),
     (["it's", "tab\there"], "k"), ([3, 1, 2], "x"), (list(range(150)), "long name")],
)
def test_index_repr_matches_pandas(labels, name):
    ours = sd.frame.Index(labels, name=name)
    theirs = pd.Index(labels, name=name, dtype="int64" if not labels or isinstance(labels[0], int) else "str")
    assert repr(ours) == pd_index_repr(theirs)


def test_range_index_repr_with_name():
    assert repr(sd.frame.RangeIndex(3, name="r")) == repr(pd.RangeIndex(3, name="r"))


def test_index_api():
    idx = sd.frame.Index(["a", "b", "a"], name="k")
    assert idx.name == "k" and idx.dtype is sd.string and len(idx) == 3 and not idx.is_unique
    assert list(idx) == ["a", "b", "a"] and "b" in idx and "z" not in idx and 1 not in idx
    assert sd.frame.Index(["a"], name="x").equals(sd.frame.Index(["a"], name="y"))  # names are not compared
    assert sd.frame.RangeIndex(3).equals(sd.frame.Index([0, 1, 2])) and sd.frame.RangeIndex(3).is_unique
    r = sd.frame.RangeIndex(1, 10**9, 3)  # `in` is arithmetic: no table of a billion labels (docs/design/frame.md §6.1)
    assert 4 in r and 5 not in r and 10**9 not in r and "4" not in r and 2**70 not in r
    # `in` is the Series rule (docs/api/frame.md §5.8): a lossless conversion finds 4.0 and True (1)
    assert 4.0 in r and True in r and False not in r and 4.5 not in r and math.nan not in r and NA not in r
    assert np.int64(7) in r and r._labels is None
    ints = sd.frame.Index([1, 2])
    assert 1.0 in ints and 2 in ints and True in ints and 1.5 not in ints and "1" not in ints and NA not in ints
    assert all((x in ints) == (x in sd.frame.Series([1, 2])) for x in (1, 1.0, True, False, 1.5, "1", None, NA))
    assert sd.frame.Index([]).dtype is sd.int64
    for bad in ([1.5], [True], [1, None]):
        with pytest.raises((TypeError, ValueError)):
            sd.frame.Index(bad)
    with pytest.raises(TypeError):
        sd.frame.Index([1], name=1)
    with pytest.raises(TypeError):
        sd.frame.Index(x for x in [1, 2])  # no generators, as in the array area (DESIGN.md §10.1 #12)


@pytest.mark.parametrize("idx", [sd.frame.Index([1, 2]), sd.frame.Index(["a"]), sd.frame.RangeIndex(3)])
def test_an_index_is_not_a_value(idx):
    # == and != are TypeErrors as < is; labels are compared with equals() (DESIGN.md §10.1 #1b)
    for op in (lambda a, b: a == b, lambda a, b: a != b, lambda a, b: a < b):
        with pytest.raises(TypeError):
            op(idx, idx)
        with pytest.raises(TypeError):
            op(idx, [1, 2])
    with pytest.raises(TypeError):
        hash(idx)  # unhashable, as a Series (DESIGN.md §10.1 #12)
    with pytest.raises(TypeError):
        bool(idx)  # as an array's and a Series' (DESIGN.md §10.1 #9)
    with pytest.raises(TypeError):
        bool(sd.frame.Index([]))  # empty too: not len()
    assert idx.equals(idx)


def test_range_index_arguments_are_int_arguments():
    # the first argument is required and all three are ints in int64, as arange's (DESIGN.md §10.1 #8, §10.2)
    with pytest.raises(TypeError):
        sd.frame.RangeIndex()
    assert sd.frame.RangeIndex(-(2**63), 2**63 - 1, 2**62).tolist() == [-(2**63), -(2**62), 0, 2**62]
    for args in ((2**63,), (0, 2**63), (0, 1, -(2**63) - 1)):
        with pytest.raises(OverflowError, match=r"^RangeIndex\(\): '(start|stop|step)' is out of range$"):
            sd.frame.RangeIndex(*args)
    for bad in (True, 1.0, np.True_, "3", None):
        with pytest.raises(TypeError, match=r"^RangeIndex\(\): 'start' must be an int, not "):
            sd.frame.RangeIndex(bad)
    # numpy integers are ints wherever an int is (DESIGN.md §10.1 #13)
    r = sd.frame.RangeIndex(np.int64(1), np.int32(7), np.int8(2))
    assert r.tolist() == [1, 3, 5] and all(type(v) is int for v in r.tolist())
    with pytest.raises(TypeError, match=r"'step' must be an int, not bool"):
        sd.frame.RangeIndex(0, 3, True)


def test_label_types_must_match():
    s = sd.frame.Series([1, 2], index=[10, 20])
    assert s.loc[np.int64(10)] == 1
    for bad in (10.0, True, "10"):
        with pytest.raises(TypeError):
            s.loc[bad]
    with pytest.raises(TypeError):
        sd.frame.Series([1], index=["a"]).loc[0]


# --- selection ------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_selection_matches_pandas_on_unique_labels(data):
    n = data.draw(st.integers(1, 30))
    use_str = data.draw(st.booleans())
    pool = st.text(max_size=3) if use_str else st.integers(-50, 50)
    labels = data.draw(st.lists(pool, min_size=n, max_size=n, unique=True))
    values = data.draw(st.lists(st.integers(-9, 9), min_size=n, max_size=n))
    ours = sd.frame.Series(values, index=labels)
    theirs = pd.Series(values, index=pd.Index(labels, dtype="str" if use_str else "int64"))
    one = data.draw(st.sampled_from(labels))
    assert ours.loc[one] == theirs.loc[one]
    some = data.draw(st.lists(st.sampled_from(labels), max_size=6))
    assert ours.loc[some].tolist() == theirs.loc[some].tolist()
    assert ours.loc[some].index.tolist() == some
    a, b = data.draw(st.sampled_from(labels)), data.draw(st.sampled_from(labels))
    step = data.draw(st.sampled_from([None, 1, 2, -1, -2]))
    # pandas' label slice on unique labels matches ours (positions of a and b, both ends kept).
    assert ours.loc[a:b:step].tolist() == theirs.loc[a:b:step].tolist()
    start, stop = data.draw(st.integers(-n - 2, n + 2)), data.draw(st.integers(-n - 2, n + 2))
    pstep = data.draw(st.sampled_from([None, 1, 3, -1]))
    assert ours.iloc[start:stop:pstep].tolist() == theirs.iloc[start:stop:pstep].tolist()
    pos = data.draw(st.integers(-n, n - 1))
    assert ours.iloc[pos] == theirs.iloc[pos]
    plist = data.draw(st.lists(st.integers(-n, n - 1), max_size=5))
    assert ours.iloc[plist].tolist() == theirs.iloc[plist].tolist()


def test_duplicate_labels():
    s = sd.frame.Series([10, 20, 30], index=["a", "b", "a"])
    with pytest.raises(KeyError):
        s.loc["a"]  # a scalar lookup must find one row (docs/api/frame.md §5.4)
    assert s.loc[["a", "b"]].tolist() == [10, 30, 20]
    assert s["b"] == 20
    with pytest.raises(KeyError):
        s.loc["a":"b"]  # slice ends must be unique


def test_missing_labels_and_positions():
    s = sd.frame.Series([1, 2], index=["x", "y"])
    with pytest.raises(KeyError):
        s.loc["q"]
    with pytest.raises(KeyError):
        s.loc[["x", "q"]]
    with pytest.raises(KeyError):
        s.loc["w":"y"]  # no search over sorted labels (docs/design/frame.md §6.4); pandas returns rows here
    with pytest.raises(IndexError):
        s.iloc[2]
    with pytest.raises(IndexError):
        s.iloc[[0, -3]]


def test_slices_keep_range_index_and_share_buffers():
    s = sd.frame.Series(list(range(10)))
    part = s.iloc[2:8:2]
    assert isinstance(part.index, sd.frame.RangeIndex) and part.index.tolist() == [2, 4, 6]
    shared = s.loc[3:6]
    assert shared._col._shares_values_with(s._col) and shared.tolist() == [3, 4, 5, 6]
    assert isinstance(s.loc[[1, 2]].index, sd.frame.Index)


def test_frame_selection():
    df = sd.frame.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.5, 1.5, 2.5]}, index=["r1", "r2", "r3"])
    row = df.loc["r2"]
    assert isinstance(row, sd.frame.DataFrame) and row.shape == (1, 3) and [str(d) for d in row.dtypes.values()] == ["int64", "string", "float64"]
    assert df.loc["r2", "b"] == "y" and df.iloc[2, 0] == 3 and df.iloc[-1, -1] == 2.5
    assert df.loc[["r3", "r1"], "a"].tolist() == [3, 1]
    assert df.loc[:, "b":"c"].columns.tolist() == ["b", "c"]
    assert df.loc["r2", ["c", "a"]].columns.tolist() == ["c", "a"]
    assert df.iloc[1:, [2, 0]].index.tolist() == ["r2", "r3"]
    assert df[["c", "a"]].columns.tolist() == ["c", "a"]
    assert df.iloc[0].shape == (1, 3)
    with pytest.raises(KeyError):
        df.loc["r2", "zz"]
    with pytest.raises(ValueError):
        df[["a", "a"]]


# --- alignment ------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_alignment_order_and_values(data):
    pool = data.draw(st.sampled_from([st.integers(0, 15), st.sampled_from(list("abcdefghij"))]))
    la = data.draw(st.lists(pool, min_size=1, max_size=10, unique=True))
    lb = data.draw(st.lists(pool, min_size=1, max_size=10, unique=True))
    va = data.draw(st.lists(st.integers(-5, 5), min_size=len(la), max_size=len(la)))
    vb = data.draw(st.lists(st.integers(-5, 5), min_size=len(lb), max_size=len(lb)))
    a, b = sd.frame.Series(va, index=la), sd.frame.Series(vb, index=lb)
    out = a + b
    union = la + [x for x in lb if x not in la]  # left order, then right-only labels (docs/api/frame.md §1.7)
    da, db = dict(zip(la, va)), dict(zip(lb, vb))
    if la == lb:
        assert out.index.tolist() == la
    else:
        assert out.index.tolist() == union
    expected = [da[k] + db[k] if k in da and k in db else NA for k in out.index.tolist()]
    assert out.tolist() == expected


def test_alignment_errors_and_names():
    a = sd.frame.Series([1, 2], index=sd.frame.Index(["x", "y"], name="k"))
    b = sd.frame.Series([1, 2], index=sd.frame.Index(["y", "z"], name="k"))
    assert (a + b).index.name == "k"
    assert (a + sd.frame.Series([1], index=sd.frame.Index(["x"], name="j"))).index.name is None
    with pytest.raises(ValueError):
        a + sd.frame.Series([1, 2], index=["y", "y"])
    with pytest.raises(TypeError):
        sd.frame.Series([1], index=[0]) + sd.frame.Series([1], index=["0"])
    same = sd.frame.Series([1, 2], index=["d", "d"])
    assert (same + same).tolist() == [2, 4]  # identical labels: by position, duplicates fine


def test_frame_alignment():
    left = sd.frame.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}, index=["x", "y"])
    right = sd.frame.DataFrame({"c": ["p", "q"], "a": [10, 20]}, index=["y", "z"])
    out = left + sd.frame.DataFrame({"a": [10, 20], "b": [0.5, 0.5]}, index=["y", "z"])
    assert out.index.tolist() == ["x", "y", "z"] and out["a"].tolist() == [NA, 12, NA]
    mixed = left + right  # "b" and "c" are each on one side: computed against an NA column of their dtype
    assert mixed.columns.tolist() == ["a", "b", "c"] and mixed["c"].dtype is sd.string
    assert mixed["c"].tolist() == [NA, NA, NA] and mixed["a"].tolist() == [NA, 12, NA]
    df = sd.frame.DataFrame({"p": sd.frame.Series([1, 2], index=["b", "a"]), "q": sd.frame.Series([3], index=["c"])})
    assert df.index.tolist() == ["b", "a", "c"] and df["q"].tolist() == [NA, NA, 3]
    given = sd.frame.DataFrame({"p": sd.frame.Series([1, 2], index=["b", "a"]), "r": [7, 8, 9]}, index=["a", "c", "b"])
    assert given["p"].tolist() == [2, NA, 1] and given["r"].tolist() == [7, 8, 9]


def test_mask_alignment():
    s = sd.frame.Series([1, 2, 3], index=["a", "b", "c"])
    mask = sd.frame.Series([True, False, True], index=["c", "b", "a"])
    assert s[mask].tolist() == [1, 3]
    with pytest.raises(ValueError):
        s[sd.frame.Series([True, False], index=["a", "b"])]  # "c" missing
    with pytest.raises(ValueError):
        s[sd.frame.Series([True, True, True, True], index=["a", "b", "c", "d"])]  # "d" extra; pandas ignores it
    # Duplicate labels here: a mask of the same length must still hold the same set (docs/design/frame.md §6.3).
    dup = sd.frame.Series([10, 20, 30], index=[1, 1, 2])
    with pytest.raises(ValueError):
        dup[sd.frame.Series([True, False, True], index=[1, 2, 3])]  # 3 is not a label of dup
    assert dup[sd.frame.Series([True, False, True], index=[1, 1, 2])].tolist() == [10, 30]  # the same index: by position


def test_reindex():
    s = sd.frame.Series([1, 2, 3], index=["a", "b", "c"], name="v")
    r = s.reindex(["c", "z", "a"])
    assert r.tolist() == [3, NA, 1] and r.index.tolist() == ["c", "z", "a"] and r.name == "v"
    with pytest.raises(ValueError):
        sd.frame.Series([1, 2], index=["a", "a"]).reindex(["a"])
    df = sd.frame.DataFrame({"x": [1, 2]}, index=[5, 6])
    assert df.reindex([6, 7])["x"].tolist() == [2, NA]


def test_constructor_index():
    s = sd.frame.Series([1, 2], index=["a", "b"], name="v")
    assert s.index.tolist() == ["a", "b"]
    assert sd.frame.Series(s, index=["b", "c"]).tolist() == [2, NA]
    assert sd.frame.Series(index=["a", "b"]).tolist() == [NA, NA]
    with pytest.raises(ValueError):
        sd.frame.Series([1, 2], index=["a"])
    assert sd.frame.DataFrame([[1], [2]], columns=["a"], index=["x", "y"]).loc["y", "a"] == 2
    with pytest.raises(ValueError):
        sd.frame.DataFrame({"a": [1, 2]}, index=["x"])


# --- assignment and Copy-on-Write ------------------------------------------------


def test_series_assignment():
    s = sd.frame.Series([1, 2, 3, 4], index=["a", "b", "c", "d"])
    s.loc["b"] = 20
    s.iloc[-1] = NA
    s.loc[["a", "c"]] = [100, 300]
    s.loc["c":"d"] = 7
    assert s.tolist() == [100, 20, 7, 7]
    s[s > 50] = 0
    assert s.tolist() == [0, 20, 7, 7]
    s.loc[["d", "a"]] = sd.frame.Series([1, 2], index=["a", "d"])  # matched by label
    assert s.tolist() == [1, 20, 7, 2]
    for bad in (1.5, 7.0, "x"):  # by dtype only: float64 into int64 is down, whatever the value (docs/api/frame.md §5.7)
        with pytest.raises(TypeError):
            s.loc["a"] = bad
    with pytest.raises(KeyError):
        s.loc["new"] = 1  # no enlargement (docs/api/frame.md §5.7)
    with pytest.raises(ValueError):
        s.iloc[0:2] = [1, 2, 3]
    t = sd.frame.Series(["x", None, "z"])
    t.iloc[1] = "yy"
    t.iloc[[0, 2]] = NA
    assert t.tolist() == [NA, "yy", NA]


def test_copy_on_write():
    df = sd.frame.DataFrame({"a": [1, 2, 3]})
    col = df["a"]
    col.iloc[0] = 99
    assert col.tolist() == [99, 2, 3] and df["a"].tolist() == [1, 2, 3]
    s = sd.frame.Series([1, 2, 3, 4])
    head = s.head(2)
    head.iloc[0] = -1
    assert s.tolist() == [1, 2, 3, 4] and head.tolist() == [-1, 2]
    view = s.to_numpy()
    s.iloc[1] = 50
    assert view.tolist() == [1, 2, 3, 4]
    other = sd.frame.Series(s)
    other.iloc[0] = 0
    assert s.tolist() == [1, 50, 3, 4]


def test_write_in_place_when_nothing_else_holds_the_buffer():
    s = sd.frame.Series(list(range(1000)))
    address = s.to_numpy().__array_interface__["data"][0]  # the view is dropped right away
    s.iloc[0] = 7
    assert s.to_numpy().__array_interface__["data"][0] == address


def test_frame_assignment():
    df = sd.frame.DataFrame({"a": [1, 2]}, index=["x", "y"])
    df["b"] = [0.5, 1.5]
    df["c"] = "k"
    df["d"] = sd.frame.Series([9], index=["y"])
    df["a"] = ["p", "q"]  # replacing a whole column may change its dtype
    assert df.columns.tolist() == ["a", "b", "c", "d"]
    assert df["d"].tolist() == [NA, 9] and df["a"].dtype is sd.string and df["c"].tolist() == ["k", "k"]
    df.loc["y", "b"] = 2
    df.iloc[0, 1] = NA
    df.loc[df["b"].notna(), "c"] = "z"
    assert df["b"].tolist() == [NA, 2.0] and df["c"].tolist() == ["k", "z"]
    alias = df["b"]
    df.loc["y", "b"] = 3.0
    assert alias.tolist() == [NA, 2.0]  # the Series taken before the write keeps its values
    with pytest.raises(KeyError):
        df.loc["y", "new"] = 1
    with pytest.raises(TypeError):
        df.loc["y", ["a", "b"]] = 1
    with pytest.raises(ValueError):
        df["e"] = [1, 2, 3]
    empty = sd.frame.DataFrame()
    empty["a"] = [1, 2, 3]
    assert empty.shape == (3, 1)


# --- iteration, in, and the return types that were once provisional (list, dict) ---


def test_iteration_and_contains():
    s = sd.frame.Series([1.0, math.nan, None], index=["a", "b", "c"])
    assert len(list(s)) == 3 and list(s)[2] is NA
    assert 1.0 in s and math.nan in s and NA in s and "a" not in s and "a" in s.index
    assert 2.0 not in s and "x" not in s
    df = sd.frame.DataFrame({"a": [1], "b": [2]})
    assert list(df) == ["a", "b"] and "a" in df and 1 not in df


def test_columns_dtypes_nunique_are_index_and_series():
    df = sd.frame.DataFrame({"a": [1, 1], "b": ["x", "y"]})
    assert isinstance(df.columns, sd.frame.Index) and df.columns.dtype is sd.string
    assert df.dtypes["b"] is sd.string and df.nunique().loc["b"] == 2 and df.nunique().dtype is sd.int64


# --- repr with labels -------------------------------------------------------------

LABEL_OPTS = dict(max_rows=60, min_rows=10, max_cols=20, max_colwidth=50, show_dimensions="truncate", line_width=None)


@pytest.mark.parametrize("n", [3, 70])
@pytest.mark.parametrize("name", [None, "key"])
def test_repr_with_string_and_named_index(n, name):
    labels = [f"r{i * 7 % 101}" for i in range(n)]
    ours = sd.frame.Series(list(range(n)), index=sd.frame.Index(labels, name=name), name="v")
    theirs = pd.Series(list(range(n)), index=pd.Index(labels, name=name, dtype="str"), name="v")
    assert repr(ours) == repr(theirs)
    df = sd.frame.DataFrame({"a": list(range(n)), "b": [str(i) for i in range(n)]}, index=sd.frame.Index(labels, name=name))
    pdf = pd.DataFrame({"a": list(range(n)), "b": pd.array([str(i) for i in range(n)], dtype="string[python]")},
                       index=pd.Index(labels, name=name, dtype="str"))
    assert repr(df) == pdf.to_string(**LABEL_OPTS)
    ints = sd.frame.DataFrame({"a": [1.5] * n}, index=sd.frame.Index(list(range(100, 100 + n)), name=name))
    pints = pd.DataFrame({"a": [1.5] * n}, index=pd.Index(list(range(100, 100 + n)), name=name))
    assert repr(ints) == pints.to_string(**LABEL_OPTS)


def test_position_keys_of_the_wrong_kind_are_index_errors():
    # as an array index of the wrong kind (DESIGN.md §10.3, API.md §2)
    s = sd.frame.Series([10, 20, 30])
    df = sd.frame.DataFrame({"a": [1, 2], "b": [3, 4]})
    for key, shown in ((True, "bool"), (1.0, "float"), ("a", "str"), (None, "NoneType")):
        with pytest.raises(IndexError, match=rf"^cannot select rows by position with {shown}; "):
            s.iloc[key]
        with pytest.raises(IndexError, match=rf"^cannot select rows by position with {shown}; "):
            df.iloc[key]
    for key, shown in (([1.0], "float"), ([0, True], "bool"), (np.array([0.5]), "float64")):
        with pytest.raises(IndexError, match=rf"^positions must be ints, not {shown}$"):
            s.iloc[key]
    with pytest.raises(IndexError, match=r"^cannot select columns by position with float; "):
        df.iloc[0, 1.0]
    # a column position out of range says columns (DESIGN.md §8.4)
    with pytest.raises(IndexError, match=r"^position 9 is out of range for 2 columns$"):
        df.iloc[0, 9]
    with pytest.raises(IndexError, match=r"^position -3 is out of range for 2 columns$"):
        df.iloc[:, [0, -3]]
    with pytest.raises(IndexError, match=r"^position 5 is out of range for 2 rows$"):
        df.iloc[5, 0]
    assert s.iloc[np.int64(1)] == 20 and df.iloc[np.int64(1), np.int64(0)] == 2


def test_label_type_messages_use_not():
    s = sd.frame.Series([1, 2], index=[10, 20])
    with pytest.raises(TypeError, match=r"^a label of an int64 index must be an int, not float$"):
        s.loc[1.5]
    with pytest.raises(TypeError, match=r"^a label of a string index must be a str, not int$"):
        sd.frame.Series([1], index=["a"]).loc[0]


# --- backward slices, column keys, empty label lists ---------------------------


@pytest.mark.parametrize("key", [slice(None, None, -1), slice(4, None, -1), slice(None, None, -2), slice(3, 0, -1),
                                 slice(1, None, -1), slice(None, None, 2), slice(0, None, -1), slice(2, 2, -1)])
@pytest.mark.parametrize("index", [None, sd.frame.RangeIndex(10, 0, -2), sd.frame.RangeIndex(3, 18, 3)])
def test_range_index_slices_backwards_down_to_position_zero(key, index):
    values = [10, 20, 30, 40, 50]
    s = sd.frame.Series(values, index=index)
    labels = list(range(5)) if index is None else index.tolist()
    got = s.iloc[key]
    assert got.tolist() == values[key] and got.index.tolist() == labels[key]
    assert isinstance(got.index, sd.frame.RangeIndex)  # a slice of a RangeIndex stays one (docs/api/frame.md §5.4)
    repr(got)
    df = sd.frame.DataFrame({"a": values}, index=index)
    assert df.iloc[key]["a"].tolist() == values[key] and df.iloc[key].index.tolist() == labels[key]


def test_label_slices_backwards():
    s = sd.frame.Series([1, 2, 3, 4, 5])
    assert s.loc[::-1].tolist() == [5, 4, 3, 2, 1] and s.loc[::-1].index.tolist() == [4, 3, 2, 1, 0]
    assert repr(s.loc[::-1].index) == "RangeIndex(start=4, stop=-1, step=-1)"
    assert s.loc[3:1:-1].tolist() == [4, 3, 2]
    assert s.loc[2::-1].index.tolist() == [2, 1, 0]


def test_column_slice_with_a_missing_name_is_a_key_error():
    df = sd.frame.DataFrame({"a": [1], "price": [1.0], "q": ["x"]})
    for key in (slice("price", "zz"), slice("zz", "q"), slice("zz", None)):
        with pytest.raises(KeyError, match="^'zz'$"):  # as df.loc[:, "zz"]
            df.loc[:, key]
    assert list(df.loc[:, "price":"q"].columns) == ["price", "q"]


def test_column_key_of_the_wrong_type_says_what_a_column_key_is():
    df = sd.frame.DataFrame({"a": [1], "b": [2]})
    for bad, name in ((0, "int"), (1.5, "float"), (None, "NoneType")):
        with pytest.raises(TypeError, match=rf"^column names must be str \(or a list or slice of str\), not {name}$"):
            df.loc[:, bad]
    with pytest.raises(TypeError, match="^column names must be str, not int$"):
        df.loc[:, [0]]


def test_frame_getitem_takes_an_array_of_names():  # as df.loc[:, array] (docs/api/frame.md §5.3, §5.4)
    df = sd.frame.DataFrame({"a": [1, 2], "b": [3, 4], "q": ["x", "y"]})
    for key in (np.array(["q", "a"]), np.array(["q", "a"], dtype=object)):
        got = df[key]
        assert list(got.columns) == ["q", "a"] and got["a"].tolist() == [1, 2]
        assert list(df.loc[:, key].columns) == ["q", "a"]
    assert df[np.array([], dtype=str)].shape == (2, 0)
    with pytest.raises(KeyError, match="^'zz'$"):
        df[np.array(["a", "zz"])]
    with pytest.raises(TypeError, match="^column names must be str, not int64$"):
        df[np.array([0])]
    with pytest.raises(ValueError, match="^column names must be unique$"):
        df[np.array(["a", "a"])]


def test_reindex_with_no_labels():
    s = sd.frame.Series([1, 2], index=["a", "b"], name="n")
    for empty in ([], ()):
        got = s.reindex(empty)
        assert got.tolist() == [] and str(got.dtype) == "int64" and got.name == "n"
        assert str(got.index.dtype) == "string"  # as s.loc[[]]
        assert str(s.loc[[]].index.dtype) == "string"
    df = sd.frame.DataFrame({"a": [1.5]}, index=["x"])
    got = df.reindex([])
    assert got.shape == (0, 1) and str(got.index.dtype) == "string" and str(got["a"].dtype) == "float64"
    assert str(sd.frame.Series([1, 2]).reindex([]).index.dtype) == "int64"
    assert sd.frame.Series(s, index=[]).tolist() == []


LABEL_MIX = r"cannot align (int64 labels with string|string labels with int64) labels$"


def test_int_and_string_labels_say_the_same_thing_everywhere():  # docs/api/frame.md §5.6
    i, t = sd.frame.Series([1, 2]), sd.frame.Series([1, 2], index=["a", "b"])
    df = sd.frame.DataFrame({"a": [1, 2]})
    cases = {
        "": [lambda: i + t, lambda: t + i, lambda: df + sd.frame.DataFrame({"a": [1, 2]}, index=["a", "b"]),
             lambda: i[sd.frame.Series([True, False], index=["a", "b"])], lambda: df.__setitem__("c", t),
             lambda: i.__setitem__(slice(None), t)],
        r"reindex\(\): ": [lambda: i.reindex(t.index), lambda: t.reindex([0, 1]), lambda: df.reindex(["a"])],
        r"Series\(\): ": [lambda: sd.frame.Series(i, index=["a", "b"])],
        r"DataFrame\(\): ": [lambda: sd.frame.DataFrame({"a": i, "b": t}), lambda: sd.frame.DataFrame(df, index=["a", "b"])],
    }
    for prefix, calls in cases.items():
        for call in calls:
            with pytest.raises(TypeError, match="^" + prefix + LABEL_MIX):
                call()
    with pytest.raises(TypeError, match=r"^concat\(\): cannot join int64 labels with string labels$"):
        sd.frame.concat([i, t])


def test_row_assignment_of_a_list_mixing_bool_and_numbers():  # docs/api/frame.md §5.7
    mixed = "^cannot assign a list that mixes bool values with int or float values$"
    for values in ([True, 2], (True, 2), [2, False], [True, 2.5], [None, True, 2], [np.bool_(True), np.int64(2)]):
        s = sd.frame.Series([1, 2, 3])
        with pytest.raises(TypeError, match=mixed):
            s.iloc[: len(values)] = values
        with pytest.raises(TypeError, match=mixed):
            s.loc[list(range(len(values)))] = values
        assert s.tolist() == [1, 2, 3]
        df = sd.frame.DataFrame({"a": [1, 2, 3], "f": [1.0, 2.0, 3.0]})
        for col in ("a", "f"):
            with pytest.raises(TypeError, match=mixed):
                df.loc[list(range(len(values))), col] = values
            with pytest.raises(TypeError, match=mixed):
                df.iloc[: len(values), 0] = values
    # bools alone, and a bool scalar, go up into an int or float column (as fillna, §3.6)
    s = sd.frame.Series([1, 2, 3])
    s.iloc[:2] = [True, False]
    s.iloc[2] = True
    assert s.tolist() == [1, 0, 1] and str(s.dtype) == "int64"
    f = sd.frame.Series([1.0, 2.0])
    f.iloc[:] = [None, 2]
    assert f.tolist() == [NA, 2.0]
    # a whole column is made anew from the list, as in construction (§2.2, §5.7)
    df = sd.frame.DataFrame({"a": [1, 2]})
    df["a"] = [True, 2]
    assert df["a"].tolist() == [1, 2] and str(df["a"].dtype) == "int64"
    assert sd.frame.Series([True, 2]).tolist() == [1, 2]


def test_assignment_takes_tuples_ranges_and_arrays():  # docs/api/frame.md §5.7
    df = sd.frame.DataFrame({"a": [1, 2]})
    df["t"], df["r"], df["n"], df["s"] = (1, 2), range(2), np.array([1.5, 2.5]), sd.asarray([1.5, 2.5])
    assert [df[k].tolist() for k in "trns"] == [[1, 2], [0, 1], [1.5, 2.5], [1.5, 2.5]]
    arr = sd.asarray([7.0, 8.0])
    df["c"] = arr
    arr[0] = 0.0  # always a copy (docs/api/frame.md §2.8)
    assert df["c"].tolist() == [7.0, 8.0]
    with pytest.raises(ValueError, match="^3 values for 2 rows$"):
        df["c"] = sd.asarray([1.0, 2.0, 3.0])
    s = sd.frame.Series([1.0, 2.0, 3.0])
    for values in ((5, 6), range(5, 7), np.array([5, 6]), sd.asarray([5, 6])):
        s.iloc[:2] = values
        assert s.tolist() == [5.0, 6.0, 3.0]
        s.iloc[:2] = [1.0, 2.0]


def test_column_from_a_series_needs_unique_labels_in_the_series_only():  # docs/api/frame.md §5.6, §5.7
    df = sd.frame.DataFrame({"a": [1, 2, 3]}, index=[5, 5, 6])
    df["c"] = sd.frame.Series([10, 20, 99], index=[5, 6, 7])  # 7 is not a row: dropped
    assert df["c"].tolist() == [10, 10, 20]
    df["d"] = sd.frame.Series([10], index=[6])
    assert df["d"].tolist() == [NA, NA, 10]
    with pytest.raises(ValueError, match="^cannot reindex or align from an index with duplicate labels$"):
        df["e"] = sd.frame.Series([1, 2, 3], index=[5, 5, 7])
    df["e"] = sd.frame.Series([1, 2, 3], index=[5, 5, 6])  # the same labels: by position
    assert df["e"].tolist() == [1, 2, 3]
    s = sd.frame.Series([0, 0, 0], index=[5, 5, 6])
    s.loc[[5]] = sd.frame.Series([7, 8], index=[5, 6])
    assert s.tolist() == [7, 7, 0]
