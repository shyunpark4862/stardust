"""M3: factorize and the unique family (API.md §13).

The oracle is a plain-Python reference built on dict: Python already treats
0.0 and -0.0 as one dict key, so only NaN and NA need a stand-in key. Where
pandas means the same thing (ints and strings in nullable dtypes), pandas
is checked too.
"""

import math
import os

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust_pandas as sdpd

NA = sdpd.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))
NAN_KEY, NA_KEY = object(), object()


def key(v):
    if v is None or v is NA:
        return NA_KEY
    if isinstance(v, float) and math.isnan(v):
        return NAN_KEY
    return v


def ref_codes(rows):
    """First-appearance codes for rows (tuples of values)."""
    seen, codes = {}, []
    for r in rows:
        codes.append(seen.setdefault(tuple(key(v) for v in r), len(seen)))
    return codes


def same(x, y):
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or (x == y and math.copysign(1, x) == math.copysign(1, y))
    return x == y


def same_list(xs, ys):
    return len(xs) == len(ys) and all(same(x, y) for x, y in zip(xs, ys))


FLOATS = st.sampled_from([0.0, -0.0, math.nan, -math.nan, 1.5, -1.5, math.inf]) | st.floats(-3, 3).map(lambda x: round(x))
ELEMS = {
    "int64": st.integers(-3, 3) | st.integers(-(2**63), 2**63 - 1),
    "float64": FLOATS.map(float),
    "bool": st.booleans(),
    "string": st.sampled_from(["", "a", "b", "é", "ab"]) | st.text(max_size=3),
}


@st.composite
def column(draw, n, dtype=None):
    dtype = dtype or draw(st.sampled_from(list(ELEMS)))
    return dtype, draw(st.lists(ELEMS[dtype] | st.none(), min_size=n, max_size=n))


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_series_unique_family(data):
    n = data.draw(st.integers(0, 60))
    dtype, values = data.draw(column(n))
    s = sdpd.Series(values, dtype=dtype, name="v")
    vals = s.to_list()
    codes = ref_codes([(v,) for v in vals])
    first = {}
    for i, c in enumerate(codes):
        first.setdefault(c, i)
    last = {c: i for i, c in enumerate(codes)}
    counts = {c: codes.count(c) for c in set(codes)}

    u = s.unique()
    assert same_list(u.to_list(), [vals[first[c]] for c in sorted(first)])  # first appearance, first value
    assert u.name == "v" and u.index.equals(sdpd.RangeIndex(len(first)))
    has_na = any(v is NA for v in vals)
    assert s.nunique(dropna=False) == len(first) and s.nunique() == len(first) - has_na

    assert s.duplicated().to_list() == [first[c] != i for i, c in enumerate(codes)]
    assert s.duplicated(keep="last").to_list() == [last[c] != i for i, c in enumerate(codes)]
    assert s.duplicated(keep="none").to_list() == [counts[c] > 1 for c in codes]
    d = s.drop_duplicates(keep="last")
    assert d.index.to_list() == sorted(last.values())

    for dropna in (True, False):
        for ascending in (False, True):
            kept = [c for c in sorted(first) if not (dropna and vals[first[c]] is NA)]
            total = sum(counts[c] for c in kept)
            order = sorted(kept, key=lambda c: counts[c], reverse=not ascending)  # stable: ties keep first appearance
            vc = s.value_counts(dropna=dropna, ascending=ascending)
            assert vc.columns == ["v", "count"]
            assert same_list(vc["v"].to_list(), [vals[first[c]] for c in order])
            assert vc["count"].to_list() == [counts[c] for c in order]
            unsorted = s.value_counts(dropna=dropna, sort=False, normalize=True)
            assert same_list(unsorted["v"].to_list(), [vals[first[c]] for c in kept])
            assert unsorted["proportion"].to_list() == [counts[c] / total for c in kept]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_frame_duplicates_and_value_counts(data):
    n = data.draw(st.integers(0, 40))
    ncols = data.draw(st.integers(1, 4))
    names = [f"c{j}" for j in range(ncols)]
    cols = {k: data.draw(column(n)) for k in names}
    df = sdpd.DataFrame({k: sdpd.Series(v, dtype=dt) for k, (dt, v) in cols.items()})
    subset = data.draw(st.lists(st.sampled_from(names), min_size=1, unique=True))
    rows = list(zip(*[df[k].to_list() for k in subset])) if n else []
    codes = ref_codes(rows)
    first = {}
    for i, c in enumerate(codes):
        first.setdefault(c, i)
    assert df.duplicated(subset).to_list() == [first[c] != i for i, c in enumerate(codes)]
    assert df.drop_duplicates(subset, ignore_index=True).shape[0] == len(first)

    vc = df.value_counts(subset, dropna=False)
    counts = {c: codes.count(c) for c in first}
    order = sorted(first, key=lambda c: counts[c], reverse=True)
    assert vc.columns == subset + ["count"]
    for j, k in enumerate(subset):
        assert same_list(vc[k].to_list(), [rows[first[c]][j] for c in order])
    assert vc["count"].to_list() == [counts[c] for c in order]
    with_na = {c for c in first if any(v is NA for v in rows[first[c]])}
    assert df.value_counts(subset)["count"].to_list() == [counts[c] for c in order if c not in with_na]


def test_many_columns_compact_the_combined_code():
    # 7 columns of ~3000 distinct values: 3000^7 > 2^62, so the mixed-radix
    # combination must compact in between (hash.cpp factorize_rows).
    n = 2999  # rows are distinct: 2999 is prime, so i -> i * (j + 7) mod 2999 is one-to-one
    df = sdpd.DataFrame({f"c{j}": [(i * (j + 7) + j) % 2999 for i in range(n)] for j in range(7)})
    rows = list(zip(*[df[f"c{j}"].to_list() for j in range(7)]))
    codes = ref_codes(rows)
    assert df.duplicated().to_list() == [codes.index(c) != i for i, c in enumerate(codes)]
    doubled = sdpd.DataFrame({k: v.to_list() * 2 for k, v in ((f"c{j}", df[f"c{j}"]) for j in range(7))})
    assert doubled.duplicated().to_list() == [False] * n + [True] * n


def test_large_column_grows_the_table():
    values = [(i * 7919) % 100_003 for i in range(200_000)]
    s = sdpd.Series(values)
    assert s.nunique() == len(set(values))
    assert s.unique().to_list() == list(dict.fromkeys(values))


@pytest.mark.parametrize("dtype", ["Int64", "string[python]"])
def test_value_counts_matches_pandas(dtype):
    values = [3, None, 1, 3, None, 2, 1, 3] if dtype == "Int64" else ["b", None, "a", "b", None, "c", "a", "b"]
    ours = sdpd.Series(values, name="k")
    theirs = pd.Series(values, dtype=dtype, name="k")
    for kwargs in ({}, {"ascending": True}, {"sort": False}, {"normalize": True}):
        vc, pvc = ours.value_counts(**kwargs), theirs.value_counts(**kwargs)
        assert vc["k"].to_list() == [NA if v is pd.NA else v for v in pvc.index.tolist()]
        assert vc[pvc.name].to_list() == pvc.tolist()
    assert ours.nunique() == theirs.nunique() and ours.nunique(dropna=False) == theirs.nunique(dropna=False)
    assert ours.duplicated(keep="none").to_list() == theirs.duplicated(keep=False).tolist()


def test_na_key_keeps_its_first_appearance():
    # pandas' value_counts(dropna=False) puts NA last among ties (and last with
    # sort=False); its unique() keeps NA in place. Ours keeps first appearance
    # in both (API.md §13).
    s = sdpd.Series([3, None, 1, 3, None, 2, 1, 3])
    assert s.value_counts(dropna=False)["value"].to_list() == [3, NA, 1, 2]
    assert s.value_counts(dropna=False, sort=False)["value"].to_list() == [3, NA, 1, 2]
    assert s.unique().to_list() == [3, NA, 1, 2]


def test_float_keys_differ_from_pandas_on_nan():
    s = sdpd.Series([1.0, math.nan, -0.0, 0.0, math.nan])
    assert s.nunique() == 3  # NaN is a value (API.md §13); pandas says 2
    vc = s.value_counts()
    assert same_list(vc["value"].to_list(), [math.nan, -0.0, 1.0])
    assert vc["count"].to_list() == [2, 2, 1]


def test_errors_and_signatures():
    s = sdpd.Series([1, 1], name="count")
    with pytest.raises(ValueError):
        s.value_counts()  # the key column would clash with "count"
    assert s.value_counts(normalize=True).columns == ["count", "proportion"]
    with pytest.raises(ValueError):
        sdpd.Series([1]).duplicated(keep=False)  # keep="none" replaces False (API.md §11 C5)
    with pytest.raises(TypeError):
        sdpd.Series([1]).duplicated("first")  # keyword-only (C2)
    df = sdpd.DataFrame({"a": [1, 1], "count": [2, 2]})
    with pytest.raises(ValueError):
        df.value_counts()
    assert df.value_counts("a")["count"].to_list() == [2]
    with pytest.raises(KeyError):
        df.duplicated("zz")
    with pytest.raises(ValueError):
        df.duplicated([])
    assert df.nunique() == {"a": 1, "count": 1}
    assert sdpd.Series([], dtype="int64").unique().to_list() == []
    assert sdpd.Series([], dtype="string").value_counts().shape == (0, 2)
