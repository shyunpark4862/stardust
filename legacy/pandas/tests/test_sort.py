"""M7: sort_values and sort_index (API.md §17).

The oracle is Python's sorted(), which is stable in both directions as the
rules ask; several keys are applied last key first (an LSD sort), each key
with its own direction and NA position. Large inputs are checked against
numpy's stable argsort.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust_pandas as sdpd

NA = sdpd.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))

ELEMS = {
    "int64": st.integers(-(2**63), 2**63 - 1) | st.integers(-3, 3),
    "float64": st.floats() | st.sampled_from([0.0, -0.0, math.inf, -math.inf, math.nan]),
    "string": st.text(max_size=3) | st.sampled_from(["", "a", "é", "B"]),
    "bool": st.booleans(),
}


def value_key(v):
    if isinstance(v, float) and math.isnan(v):
        return (1, 0.0)  # NaN: larger than every number (API.md §17)
    return (0, v)


def oracle(columns, ascending, na_last):
    order = list(range(len(columns[0]) if columns else 0))
    for values, asc, last in reversed(list(zip(columns, ascending, na_last))):
        valid = [r for r in order if values[r] is not NA]
        missing = [r for r in order if values[r] is NA]
        valid = sorted(valid, key=lambda r: value_key(values[r]), reverse=not asc)  # stable either way
        order = valid + missing if last else missing + valid
    return order


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_sort_values_matches_python_sorted(data):
    n = data.draw(st.integers(0, 60))
    nkeys = data.draw(st.integers(1, 3))
    dts = [data.draw(st.sampled_from(list(ELEMS))) for _ in range(nkeys)]
    cols = {f"c{j}": data.draw(st.lists(ELEMS[dt] | st.none(), min_size=n, max_size=n)) for j, dt in enumerate(dts)}
    df = sdpd.DataFrame({k: sdpd.Series(v, dtype=dt) for (k, v), dt in zip(cols.items(), dts)})
    asc = data.draw(st.lists(st.booleans(), min_size=nkeys, max_size=nkeys))
    nas = data.draw(st.lists(st.sampled_from(["first", "last"]), min_size=nkeys, max_size=nkeys))
    by = list(cols)
    values = [df[k].to_list() for k in by]
    expected = oracle(values, asc, [p == "last" for p in nas])
    out = df.sort_values(by, ascending=asc, na_position=nas)
    assert out.index.to_list() == expected
    if nkeys == 1:
        s = df[by[0]].sort_values(ascending=asc[0], na_position=nas[0])
        assert s.index.to_list() == expected


def test_float_order_rules():
    s = sdpd.Series([1.0, math.nan, -math.inf, math.inf, None, -0.0, 0.0])
    up = s.sort_values().to_list()
    assert up[:3] == [-math.inf, -0.0, 0.0] and math.copysign(1, up[1]) == -1  # -0.0 == 0.0: original order
    assert up[3:5] == [1.0, math.inf] and math.isnan(up[5]) and up[6] is NA
    down = s.sort_values(ascending=False, na_position="first").to_list()
    assert down[0] is NA and math.isnan(down[1]) and down[2:] == [math.inf, 1.0, -0.0, 0.0, -math.inf]


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("ascending", [True, False])
def test_large_matches_numpy_stable_argsort(dtype, ascending):
    rng = np.random.default_rng(7)
    values = rng.integers(-1000, 1000, 200_000).astype(dtype)
    if dtype == "float64":
        values *= 1.5
    s = sdpd.Series(values)
    expected = np.argsort(values if ascending else -values, kind="stable")  # negating keeps ties in order
    assert s.sort_values(ascending=ascending).index.to_list() == expected.tolist()


@pytest.mark.parametrize("ascending", [True, False])
def test_strings_match_pandas_stable(ascending):
    words = ["pear", "Apple", "apple", "é", "", "pear", "b"] * 3
    got = sdpd.Series(words).sort_values(ascending=ascending).index.to_list()
    want = pd.Series(words).sort_values(ascending=ascending, kind="stable").index.tolist()
    assert got == want


def test_sort_index():
    s = sdpd.Series([1, 2, 3], index=["c", "a", "b"], name="v")
    assert s.sort_index().index.to_list() == ["a", "b", "c"] and s.sort_index().to_list() == [2, 3, 1]
    assert s.sort_index(ascending=False).to_list() == [1, 3, 2]
    df = sdpd.DataFrame({"x": [1, 2, 3]}, index=[30, 10, 20])
    assert df.sort_index()["x"].to_list() == [2, 3, 1]
    assert df.sort_index(ignore_index=True).index.equals(sdpd.RangeIndex(3))


def test_result_keeps_labels_and_names():
    s = sdpd.Series([3, 1, 2], index=sdpd.Index(["a", "b", "c"], name="k"), name="v")
    out = s.sort_values()
    assert out.index.to_list() == ["b", "c", "a"] and out.index.name == "k" and out.name == "v"
    assert isinstance(sdpd.Series([2, 1]).sort_values().index, sdpd.Index)
    assert sdpd.Series([2, 1]).sort_values(ignore_index=True).index.equals(sdpd.RangeIndex(2))


def test_arguments():
    df = sdpd.DataFrame({"a": [1, 2], "b": [3, 4]})
    for bad, error in (
        (lambda: df.sort_values(["a", "b"], ascending=[True]), ValueError),
        (lambda: df.sort_values("a", ascending=1), TypeError),
        (lambda: df.sort_values("a", na_position="middle"), ValueError),
        (lambda: df.sort_values("a", na_position=["first", "last"]), ValueError),
        (lambda: df.sort_values([]), ValueError),
        (lambda: df.sort_values("zz"), KeyError),
        (lambda: df.sort_values("a", True), TypeError),  # keyword-only (API.md §11 C2)
        (lambda: df["a"].sort_values(kind="stable"), TypeError),  # no kind: always stable (C4)
        (lambda: df.sort_index(na_position="last"), TypeError),  # no NA in an Index (C4)
    ):
        with pytest.raises(error):
            bad()
