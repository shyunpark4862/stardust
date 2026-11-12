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
    return repr(idx).replace("dtype='str'", "dtype='string'")


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
    assert idx.name == "k" and idx.dtype == "string" and len(idx) == 3 and not idx.is_unique
    assert list(idx) == ["a", "b", "a"] and "b" in idx and "z" not in idx and 1 not in idx
    assert sd.frame.Index(["a"], name="x").equals(sd.frame.Index(["a"], name="y"))  # names are not compared
    assert sd.frame.RangeIndex(3).equals(sd.frame.Index([0, 1, 2])) and sd.frame.RangeIndex(3).is_unique
    r = sd.frame.RangeIndex(1, 10**9, 3)  # `in` is arithmetic: no table of a billion labels (docs/design/frame.md §6.1)
    assert 4 in r and 5 not in r and 10**9 not in r and "4" not in r and True not in r and 4.0 not in r
    assert np.int64(7) in r and r._labels is None
    assert sd.frame.Index([]).dtype == "int64"
    for bad in ([1.5], [True], [1, None]):
        with pytest.raises((TypeError, ValueError)):
            sd.frame.Index(bad)
    with pytest.raises(TypeError):
        sd.frame.Index([1], name=1)


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
    assert ours.loc[some].to_list() == theirs.loc[some].tolist()
    assert ours.loc[some].index.to_list() == some
    a, b = data.draw(st.sampled_from(labels)), data.draw(st.sampled_from(labels))
    step = data.draw(st.sampled_from([None, 1, 2, -1, -2]))
    # pandas' label slice on unique labels matches ours (positions of a and b, both ends kept).
    assert ours.loc[a:b:step].to_list() == theirs.loc[a:b:step].tolist()
    start, stop = data.draw(st.integers(-n - 2, n + 2)), data.draw(st.integers(-n - 2, n + 2))
    pstep = data.draw(st.sampled_from([None, 1, 3, -1]))
    assert ours.iloc[start:stop:pstep].to_list() == theirs.iloc[start:stop:pstep].tolist()
    pos = data.draw(st.integers(-n, n - 1))
    assert ours.iloc[pos] == theirs.iloc[pos]
    plist = data.draw(st.lists(st.integers(-n, n - 1), max_size=5))
    assert ours.iloc[plist].to_list() == theirs.iloc[plist].tolist()


def test_duplicate_labels():
    s = sd.frame.Series([10, 20, 30], index=["a", "b", "a"])
    with pytest.raises(KeyError):
        s.loc["a"]  # a scalar lookup must find one row (docs/api/frame.md §5.4)
    assert s.loc[["a", "b"]].to_list() == [10, 30, 20]
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
    assert isinstance(part.index, sd.frame.RangeIndex) and part.index.to_list() == [2, 4, 6]
    shared = s.loc[3:6]
    assert shared._col._shares_values_with(s._col) and shared.to_list() == [3, 4, 5, 6]
    assert isinstance(s.loc[[1, 2]].index, sd.frame.Index)


def test_frame_selection():
    df = sd.frame.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [0.5, 1.5, 2.5]}, index=["r1", "r2", "r3"])
    row = df.loc["r2"]
    assert isinstance(row, sd.frame.DataFrame) and row.shape == (1, 3) and row.dtypes.to_list() == ["int64", "string", "float64"]
    assert df.loc["r2", "b"] == "y" and df.iloc[2, 0] == 3 and df.iloc[-1, -1] == 2.5
    assert df.loc[["r3", "r1"], "a"].to_list() == [3, 1]
    assert df.loc[:, "b":"c"].columns.to_list() == ["b", "c"]
    assert df.loc["r2", ["c", "a"]].columns.to_list() == ["c", "a"]
    assert df.iloc[1:, [2, 0]].index.to_list() == ["r2", "r3"]
    assert df[["c", "a"]].columns.to_list() == ["c", "a"]
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
        assert out.index.to_list() == la
    else:
        assert out.index.to_list() == union
    expected = [da[k] + db[k] if k in da and k in db else NA for k in out.index.to_list()]
    assert out.to_list() == expected


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
    assert (same + same).to_list() == [2, 4]  # identical labels: by position, duplicates fine


def test_frame_alignment():
    left = sd.frame.DataFrame({"a": [1, 2], "b": [1.5, 2.5]}, index=["x", "y"])
    right = sd.frame.DataFrame({"c": ["p", "q"], "a": [10, 20]}, index=["y", "z"])
    out = left + sd.frame.DataFrame({"a": [10, 20], "b": [0.5, 0.5]}, index=["y", "z"])
    assert out.index.to_list() == ["x", "y", "z"] and out["a"].to_list() == [NA, 12, NA]
    mixed = left + right  # "b" and "c" are each on one side: computed against an NA column of their dtype
    assert mixed.columns.to_list() == ["a", "b", "c"] and mixed["c"].dtype == "string"
    assert mixed["c"].to_list() == [NA, NA, NA] and mixed["a"].to_list() == [NA, 12, NA]
    df = sd.frame.DataFrame({"p": sd.frame.Series([1, 2], index=["b", "a"]), "q": sd.frame.Series([3], index=["c"])})
    assert df.index.to_list() == ["b", "a", "c"] and df["q"].to_list() == [NA, NA, 3]
    given = sd.frame.DataFrame({"p": sd.frame.Series([1, 2], index=["b", "a"]), "r": [7, 8, 9]}, index=["a", "c", "b"])
    assert given["p"].to_list() == [2, NA, 1] and given["r"].to_list() == [7, 8, 9]


def test_mask_alignment():
    s = sd.frame.Series([1, 2, 3], index=["a", "b", "c"])
    mask = sd.frame.Series([True, False, True], index=["c", "b", "a"])
    assert s[mask].to_list() == [1, 3]
    with pytest.raises(ValueError):
        s[sd.frame.Series([True, False], index=["a", "b"])]  # "c" missing
    with pytest.raises(ValueError):
        s[sd.frame.Series([True, True, True, True], index=["a", "b", "c", "d"])]  # "d" extra; pandas ignores it
    # Duplicate labels here: a mask of the same length must still hold the same set (docs/design/frame.md §6.3).
    dup = sd.frame.Series([10, 20, 30], index=[1, 1, 2])
    with pytest.raises(ValueError):
        dup[sd.frame.Series([True, False, True], index=[1, 2, 3])]  # 3 is not a label of dup
    assert dup[sd.frame.Series([True, False, True], index=[1, 1, 2])].to_list() == [10, 30]  # the same index: by position


def test_reindex():
    s = sd.frame.Series([1, 2, 3], index=["a", "b", "c"], name="v")
    r = s.reindex(["c", "z", "a"])
    assert r.to_list() == [3, NA, 1] and r.index.to_list() == ["c", "z", "a"] and r.name == "v"
    with pytest.raises(ValueError):
        sd.frame.Series([1, 2], index=["a", "a"]).reindex(["a"])
    df = sd.frame.DataFrame({"x": [1, 2]}, index=[5, 6])
    assert df.reindex([6, 7])["x"].to_list() == [2, NA]


def test_constructor_index():
    s = sd.frame.Series([1, 2], index=["a", "b"], name="v")
    assert s.index.to_list() == ["a", "b"]
    assert sd.frame.Series(s, index=["b", "c"]).to_list() == [2, NA]
    assert sd.frame.Series(index=["a", "b"]).to_list() == [NA, NA]
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
    s.loc["c":"d"] = 7.0  # lossless float -> int (docs/api/frame.md §5.7)
    assert s.to_list() == [100, 20, 7, 7]
    s[s > 50] = 0
    assert s.to_list() == [0, 20, 7, 7]
    s.loc[["d", "a"]] = sd.frame.Series([1, 2], index=["a", "d"])  # matched by label
    assert s.to_list() == [1, 20, 7, 2]
    for bad, error in ((1.5, ValueError), ("x", TypeError), (True, TypeError)):
        with pytest.raises(error):
            s.loc["a"] = bad
    with pytest.raises(KeyError):
        s.loc["new"] = 1  # no enlargement (docs/api/frame.md §5.7)
    with pytest.raises(ValueError):
        s.iloc[0:2] = [1, 2, 3]
    t = sd.frame.Series(["x", None, "z"])
    t.iloc[1] = "yy"
    t.iloc[[0, 2]] = NA
    assert t.to_list() == [NA, "yy", NA]


def test_copy_on_write():
    df = sd.frame.DataFrame({"a": [1, 2, 3]})
    col = df["a"]
    col.iloc[0] = 99
    assert col.to_list() == [99, 2, 3] and df["a"].to_list() == [1, 2, 3]
    s = sd.frame.Series([1, 2, 3, 4])
    head = s.head(2)
    head.iloc[0] = -1
    assert s.to_list() == [1, 2, 3, 4] and head.to_list() == [-1, 2]
    view = s.to_numpy()
    s.iloc[1] = 50
    assert view.tolist() == [1, 2, 3, 4]
    other = sd.frame.Series(s)
    other.iloc[0] = 0
    assert s.to_list() == [1, 50, 3, 4]


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
    assert df.columns.to_list() == ["a", "b", "c", "d"]
    assert df["d"].to_list() == [NA, 9] and df["a"].dtype == "string" and df["c"].to_list() == ["k", "k"]
    df.loc["y", "b"] = 2
    df.iloc[0, 1] = NA
    df.loc[df["b"].notna(), "c"] = "z"
    assert df["b"].to_list() == [NA, 2.0] and df["c"].to_list() == ["k", "z"]
    alias = df["b"]
    df.loc["y", "b"] = 3.0
    assert alias.to_list() == [NA, 2.0]  # the Series taken before the write keeps its values
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
    assert isinstance(df.columns, sd.frame.Index) and df.columns.dtype == "string"
    assert df.dtypes.loc["b"] == "string" and df.nunique().loc["b"] == 2 and df.nunique().dtype == "int64"


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
