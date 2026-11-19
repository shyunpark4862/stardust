"""factorize and the unique family (docs/api/frame.md §4).

The oracle is a plain-Python reference built on dict: Python already treats
0.0 and -0.0 as one dict key, so only NaN and NA need a stand-in key. Where
pandas means the same thing (ints and strings in nullable dtypes), pandas
is checked too.
"""

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples

NA = sd.frame.NA
EXAMPLES = examples(300)
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


def sort_key(v):
    """Sorted order of unique_values (docs/api/frame.md §4.2): values, then NaN, then NA."""
    if v is NA:
        return (2, 0)
    if isinstance(v, float) and math.isnan(v):
        return (1, 0)
    return (0, v)


FLOATS = st.sampled_from([0.0, -0.0, math.nan, -math.nan, 1.5, -1.5, math.inf]) | st.floats(-3, 3).map(
    lambda x: round(x)
)
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
    s = sd.frame.Series(values, dtype=dtype, name="v")
    vals = s.tolist()
    codes = ref_codes([(v,) for v in vals])
    first = {}
    for i, c in enumerate(codes):
        first.setdefault(c, i)
    last = {c: i for i, c in enumerate(codes)}
    counts = {c: codes.count(c) for c in set(codes)}

    u = s.unique_values()
    reps = [vals[first[c]] for c in sorted(first)]  # each key's first value
    assert same_list(u.tolist(), sorted(reps, key=sort_key))  # sorted, NaN once, NA last
    assert u.name == "v" and u.index.equals(sd.frame.RangeIndex(len(first)))
    has_na = any(v is NA for v in vals)
    assert s.nunique(dropna=False) == len(first) and s.nunique() == len(first) - has_na

    assert s.duplicated().tolist() == [first[c] != i for i, c in enumerate(codes)]
    assert s.duplicated(keep="last").tolist() == [last[c] != i for i, c in enumerate(codes)]
    assert s.duplicated(keep="none").tolist() == [counts[c] > 1 for c in codes]
    d = s.drop_duplicates(keep="last")
    assert d.index.tolist() == sorted(last.values())

    for dropna in (True, False):
        for descending in (True, False):
            kept = [c for c in sorted(first) if not (dropna and vals[first[c]] is NA)]
            total = sum(counts[c] for c in kept)
            order = sorted(kept, key=lambda c: counts[c], reverse=descending)  # stable: ties keep first appearance
            vc = s.value_counts(dropna=dropna, descending=descending)
            assert vc.columns.tolist() == ["v", "counts"]
            assert same_list(vc["v"].tolist(), [vals[first[c]] for c in order])
            assert vc["counts"].tolist() == [counts[c] for c in order]
            unsorted = s.value_counts(dropna=dropna, sort=False, normalize=True)
            assert same_list(unsorted["v"].tolist(), [vals[first[c]] for c in kept])
            assert unsorted["proportion"].tolist() == [counts[c] / total for c in kept]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_frame_duplicates_and_value_counts(data):
    n = data.draw(st.integers(0, 40))
    ncols = data.draw(st.integers(1, 4))
    names = [f"c{j}" for j in range(ncols)]
    cols = {k: data.draw(column(n)) for k in names}
    df = sd.frame.DataFrame({k: sd.frame.Series(v, dtype=dt) for k, (dt, v) in cols.items()})
    subset = data.draw(st.lists(st.sampled_from(names), min_size=1, unique=True))
    rows = list(zip(*[df[k].tolist() for k in subset])) if n else []
    codes = ref_codes(rows)
    first = {}
    for i, c in enumerate(codes):
        first.setdefault(c, i)
    assert df.duplicated(subset).tolist() == [first[c] != i for i, c in enumerate(codes)]
    assert df.drop_duplicates(subset, ignore_index=True).shape[0] == len(first)

    vc = df.value_counts(subset, dropna=False)
    counts = {c: codes.count(c) for c in first}
    order = sorted(first, key=lambda c: counts[c], reverse=True)
    assert vc.columns.tolist() == subset + ["counts"]
    for j, k in enumerate(subset):
        assert same_list(vc[k].tolist(), [rows[first[c]][j] for c in order])
    assert vc["counts"].tolist() == [counts[c] for c in order]
    with_na = {c for c in first if any(v is NA for v in rows[first[c]])}
    assert df.value_counts(subset)["counts"].tolist() == [counts[c] for c in order if c not in with_na]


def test_many_columns_compact_the_combined_code():
    # 7 columns of ~3000 distinct values: 3000^7 > 2^62, so the mixed-radix
    # combination must compact in between (hash.cpp factorize_rows).
    n = 2999  # rows are distinct: 2999 is prime, so i -> i * (j + 7) mod 2999 is one-to-one
    df = sd.frame.DataFrame({f"c{j}": [(i * (j + 7) + j) % 2999 for i in range(n)] for j in range(7)})
    rows = list(zip(*[df[f"c{j}"].tolist() for j in range(7)]))
    codes = ref_codes(rows)
    assert df.duplicated().tolist() == [codes.index(c) != i for i, c in enumerate(codes)]
    doubled = sd.frame.DataFrame({k: v.tolist() * 2 for k, v in ((f"c{j}", df[f"c{j}"]) for j in range(7))})
    assert doubled.duplicated().tolist() == [False] * n + [True] * n


def test_large_column_grows_the_table():
    values = [(i * 7919) % 100_003 for i in range(200_000)]
    s = sd.frame.Series(values)
    assert s.nunique() == len(set(values))
    assert s.unique_values().tolist() == sorted(set(values))
    assert s.drop_duplicates().tolist() == list(dict.fromkeys(values))  # first appearance


@pytest.mark.parametrize("dtype", ["Int64", "string[python]"])
def test_value_counts_matches_pandas(dtype):
    values = [3, None, 1, 3, None, 2, 1, 3] if dtype == "Int64" else ["b", None, "a", "b", None, "c", "a", "b"]
    ours = sd.frame.Series(values, name="k")
    theirs = pd.Series(values, dtype=dtype, name="k")
    for kwargs, pkwargs in (
        ({}, {}),
        ({"descending": False}, {"ascending": True}),
        ({"sort": False}, {"sort": False}),
        ({"normalize": True}, {"normalize": True}),
    ):
        vc, pvc = ours.value_counts(**kwargs), theirs.value_counts(**pkwargs)
        assert vc["k"].tolist() == [NA if v is pd.NA else v for v in pvc.index.tolist()]
        assert vc["counts" if pvc.name == "count" else pvc.name].tolist() == pvc.tolist()  # [pandas: count]
    assert ours.nunique() == theirs.nunique() and ours.nunique(dropna=False) == theirs.nunique(dropna=False)
    assert ours.duplicated(keep="none").tolist() == theirs.duplicated(keep=False).tolist()


def test_na_key_keeps_its_first_appearance():
    # pandas' value_counts(dropna=False) puts NA last among ties (and last with
    # sort=False). Ours keeps first appearance (docs/api/frame.md §4.3);
    # unique_values is sorted with NA last (§4.2).
    s = sd.frame.Series([3, None, 1, 3, None, 2, 1, 3])
    assert s.value_counts(dropna=False)["values"].tolist() == [3, NA, 1, 2]
    assert s.value_counts(dropna=False, sort=False)["values"].tolist() == [3, NA, 1, 2]
    assert s.unique_values().tolist() == [1, 2, 3, NA]


def test_unique_values_order():
    # Sorted like the array area's unique_values; NaN is one key (also with
    # another payload), -0.0 and 0.0 are one key shown as the first one, NA last.
    neg_nan = -math.nan
    s = sd.frame.Series([2.5, None, math.nan, -0.0, 0.0, -math.inf, neg_nan, 1.0, None, math.inf], name="u")
    u = s.unique_values()
    assert same_list(u.tolist(), [-math.inf, -0.0, 1.0, 2.5, math.inf, math.nan, NA])
    assert u.name == "u" and u.index.equals(sd.frame.RangeIndex(7))
    assert math.copysign(1, sd.frame.Series([0.0, -0.0]).unique_values().tolist()[0]) == 1
    assert sd.frame.Series(["b", None, "é", "B", "", "b"]).unique_values().tolist() == ["", "B", "b", "é", NA]
    assert sd.frame.Series([True, None, False, True]).unique_values().tolist() == [False, True, NA]
    assert sd.frame.Series([None, None], dtype="int64").unique_values().tolist() == [NA]
    # the array area gives the same values in the same order (where neither has NaN or NA)
    for values in ([3, -1, 3, 2**63 - 1, -(2**63), 0], [1.5, -0.0, 0.0, -2.0, 1.5, math.inf]):
        assert sd.frame.Series(values).unique_values().tolist() == sd.unique_values(sd.asarray(values)).tolist()


def test_unique_is_gone():
    with pytest.raises(AttributeError):
        sd.frame.Series([1]).unique()  # renamed unique_values, no alias (docs/api/frame.md §1.1)


def test_float_keys_differ_from_pandas_on_nan():
    s = sd.frame.Series([1.0, math.nan, -0.0, 0.0, math.nan])
    assert s.nunique() == 3  # NaN is a value (docs/api/frame.md §4.2); pandas says 2
    vc = s.value_counts()
    assert same_list(vc["values"].tolist(), [math.nan, -0.0, 1.0])
    assert vc["counts"].tolist() == [2, 2, 1]


def test_errors_and_signatures():
    s = sd.frame.Series([1, 1], name="counts")
    with pytest.raises(ValueError):
        s.value_counts()  # the key column would clash with "counts"
    assert s.value_counts(normalize=True).columns.tolist() == ["counts", "proportion"]
    with pytest.raises(TypeError, match=r"^duplicated\(\): 'keep' must be 'first', 'last' or 'none', not bool$"):
        sd.frame.Series([1]).duplicated(keep=False)  # keep="none" replaces False; a string option (API.md §2)
    with pytest.raises(ValueError, match=r"^drop_duplicates\(\): 'keep' must be 'first', 'last' or 'none', not 'x'$"):
        sd.frame.Series([1]).drop_duplicates(keep="x")
    for bad in (dict(normalize=1), dict(sort=None), dict(dropna=0)):  # bool options (DESIGN.md §10.1 #4)
        with pytest.raises(TypeError, match=r"^value_counts\(\): '\w+' must be True or False, not "):
            sd.frame.Series([1]).value_counts(**bad)
    with pytest.raises(TypeError, match=r"^nunique\(\): 'dropna' must be True or False, not int$"):
        sd.frame.Series([1]).nunique(dropna=1)
    with pytest.raises(TypeError):
        sd.frame.Series([1]).duplicated("first")  # keyword-only (docs/api/frame.md §1.1)
    df = sd.frame.DataFrame({"a": [1, 1], "counts": [2, 2]})
    with pytest.raises(ValueError):
        df.value_counts()
    assert df.value_counts("a")["counts"].tolist() == [2]
    with pytest.raises(KeyError):
        df.duplicated("zz")
    with pytest.raises(ValueError):
        df.duplicated([])
    assert df.nunique().tolist() == [1, 1] and df.nunique().index.tolist() == ["a", "counts"]
    assert sd.frame.Series([], dtype="int64").unique_values().tolist() == []
    with pytest.raises(TypeError):
        sd.frame.Series([1]).value_counts(descending=1)  # a bool option takes True or False (API.md §2)
    with pytest.raises(TypeError):
        sd.frame.Series([1]).value_counts(ascending=True)  # renamed: descending (docs/api/frame.md §4.1)
    with pytest.raises(TypeError):
        df.duplicated(subset=["a"])  # subset is positional-only (docs/api/frame.md §1.1)
    assert df.value_counts("a", descending=False)["counts"].tolist() == [2]
    assert sd.frame.Series([], dtype="string").value_counts().shape == (0, 2)
