"""repr matches pandas 3.1.1 (docs/api/frame.md §11).

pandas is the oracle. Each column becomes the pandas object that pandas
formats the way docs/api/frame.md §11 says ours is formatted:
- no NA: the numpy dtype (int64, float64, bool) or string[python];
- int/bool/string with NA: the nullable Int64/boolean/string[python];
- float64 with NA: float64 with NaN in the NA rows, printed with na_rep="<NA>"
  (so those columns hold no real NaN).
pandas' nullable Int64 prints negative numbers one space wider than int64;
ours always follows int64 (docs/api/frame.md §11), so ints with NA are drawn non-negative
here and the difference has its own test.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples

NA = sd.frame.NA
EXAMPLES = examples(300)  # SD_TIER=deep for a long oracle run
PD_NAMES = {"Int64": "int64", "boolean": "bool", "string": "string", "str": "string"}

SERIES_OPTS = dict(name=True, dtype=True, min_rows=10, max_rows=60, length="truncate")
FRAME_OPTS = dict(max_rows=60, min_rows=10, max_cols=20, max_colwidth=50, show_dimensions="truncate", line_width=None)


def pandas_column(values, dtype):
    has_na = any(v is NA for v in values)
    if dtype == "int64":
        return (
            pd.array([None if v is NA else v for v in values], dtype="Int64")
            if has_na
            else np.array(values, dtype=np.int64)
        )
    if dtype == "float64":
        return np.array([math.nan if v is NA else v for v in values], dtype=np.float64)
    if dtype == "bool":
        return (
            pd.array([None if v is NA else v for v in values], dtype="boolean")
            if has_na
            else np.array(values, dtype=bool)
        )
    return pd.array([None if v is NA else v for v in values], dtype="string[python]")


def fix_dtype_names(text):
    head, sep, pd_name = text.rpartition("dtype: ")
    tail = ""
    if pd_name.endswith(")"):  # "Series([], dtype: x)"
        pd_name, tail = pd_name[:-1], ")"
    return head + sep + PD_NAMES.get(pd_name, pd_name) + tail


finite = st.floats(allow_nan=False, allow_infinity=False)
special = st.sampled_from([0.0, -0.0, 1e-7, -3e-9, 1e6, 1.5e6, 1e15, 123456789.0, 0.1, 2.5, math.inf, -math.inf])
float_values = st.one_of(
    finite, special, st.floats(min_value=-1e3, max_value=1e3), st.integers(-(10**6), 10**6).map(float)
)
text = st.text(
    st.characters(codec="utf-8", exclude_categories=["Cs", "Cc", "Zl", "Zp", "Cf", "Mn", "Me"])
    | st.sampled_from("\t\n\r"),
    max_size=60,
)


@st.composite
def column(draw, n):
    dtype = draw(st.sampled_from(["int64", "float64", "bool", "string"]))
    with_na = draw(st.booleans())
    if dtype == "int64":
        elems = st.integers(0 if with_na else -(2**63), 2**63 - 1) | st.integers(-1000 * (not with_na), 1000)
    elif dtype == "float64":
        elems = float_values | (st.just(math.nan) if not with_na else float_values)
    elif dtype == "bool":
        elems = st.booleans()
    else:
        elems = text
    if with_na:
        elems = elems | st.just(NA)
    return dtype, draw(st.lists(elems, min_size=n, max_size=n))


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_series_repr_matches_pandas(data):
    n = data.draw(st.sampled_from([0, 1, 2, 3, 10, 59, 60, 61, 100]) | st.integers(0, 70))
    dtype, values = data.draw(column(n))
    name = data.draw(st.none() | text)
    ours = sd.frame.Series(values, dtype=dtype, name=name)
    theirs = pd.Series(pandas_column(values, dtype), name=name)
    na_rep = "<NA>" if dtype == "float64" and any(x is NA for x in values) else "NaN"
    assert repr(ours) == fix_dtype_names(theirs.to_string(na_rep=na_rep, **SERIES_OPTS))


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_frame_repr_matches_pandas(data):
    n = data.draw(st.sampled_from([0, 1, 3, 59, 60, 61, 100]) | st.integers(0, 70))
    ncols = data.draw(st.sampled_from([0, 1, 4, 20, 21, 23]))
    names = data.draw(st.lists(text, min_size=ncols, max_size=ncols, unique=True))
    columns = [data.draw(column(n)) for _ in names]
    if ncols > 20:
        # pandas bug: after cutting columns it picks a header's leading space
        # from the dtype at the same position in the uncut frame. Ours uses the
        # shown column's own dtype; skip the cases where the two differ.
        shown = list(range(10)) + list(range(ncols - 10, ncols))
        is_str = [dt == "string" for dt, _ in columns]
        assume(all(is_str[p] == is_str[i] for i, p in enumerate(shown)))
    ours = sd.frame.DataFrame({k: sd.frame.Series(v, dtype=dt) for k, (dt, v) in zip(names, columns)})
    theirs = pd.DataFrame(
        {k: pandas_column(v, dt) for k, (dt, v) in zip(names, columns)}, index=pd.RangeIndex(n if names else 0)
    )
    # na_rep only reaches float columns; they never mix NaN and NA here.
    na_rep = "<NA>" if any(dt == "float64" and any(x is NA for x in v) for dt, v in columns) else "NaN"
    if na_rep == "<NA>" and any(dt == "float64" and any(x is not NA and x != x for x in v) for dt, v in columns):
        return  # a NaN column next to an NA column: one na_rep cannot express both
    assert repr(ours) == theirs.to_string(na_rep=na_rep, **FRAME_OPTS)


def test_example_from_api_md():
    df = sd.frame.DataFrame(
        {"a": [1, None, 3], "b": [1.5, math.nan, 3.0], "c": ["x", None, "zz"], "d": [True, None, False]}
    )
    assert repr(df) == (
        "      a    b     c      d\n0     1  1.5     x   True\n1  <NA>  NaN  <NA>   <NA>\n2     3  3.0    zz  False"
    )
    assert repr(sd.frame.Series([0.1, 1e10, 123.456], name="v")) == (
        "0    1.000000e-01\n1    1.000000e+10\n2    1.234560e+02\nName: v, dtype: float64"
    )


def test_nan_and_na_in_one_float_column():
    assert repr(sd.frame.Series([1.25, math.nan, None])) == "0    1.25\n1     NaN\n2    <NA>\ndtype: float64"


def test_negative_int_with_na_follows_int64_formatting():
    # pandas' Int64 would print " -1" (one space wider); int64 prints "-1".
    assert repr(sd.frame.Series([-1, None])) == "0      -1\n1    <NA>\ndtype: int64"
    assert repr(sd.frame.Series([-1, 2])) == repr(pd.Series([-1, 2]))


def test_empty():
    assert repr(sd.frame.Series([], dtype="string", name="s")) == "Series([], Name: s, dtype: string)"
    assert repr(sd.frame.DataFrame()) == "Empty DataFrame\nColumns: []\nIndex: []"
    assert repr(sd.frame.DataFrame({"a": []})) == "Empty DataFrame\nColumns: [a]\nIndex: []"


def test_cut_columns_use_their_own_dtype_for_header_space():
    names = [f"c{i}" for i in range(21)]
    df = sd.frame.DataFrame({k: ["x"] if i == 11 else [0] for i, k in enumerate(names)})
    header = repr(df).splitlines()[0]
    assert "...  c11" not in header and "... c11" in header  # c11 is a string column: no extra space


@pytest.mark.parametrize("n", [5, 61, 1000])
def test_head_tail_repr_keep_labels(n):
    s = sd.frame.Series(list(range(n)))
    assert repr(s.tail(3)) == repr(pd.Series(range(n)).tail(3))
    df = sd.frame.DataFrame({"a": list(range(n))})
    assert repr(df.tail(70)) == pd.DataFrame({"a": range(n)}).tail(70).to_string(**FRAME_OPTS)
