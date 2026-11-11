"""M6: groupby (API.md §16).

The oracle for each group is M5 itself: a group's result must equal, bit for
bit, the same reduction of a Series holding just that group's rows (that is
why group sums are pairwise per group). Group order is checked against a
plain-Python sort; integer and string keys are also checked against pandas.
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
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 200))


def same(x, y):
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or np.float64(x).tobytes() == np.float64(y).tobytes()
    return x == y and type(x) is type(y)


def key_of(v):  # the sort key of API.md §16 (NA last, NaN after +inf, -0.0 == 0.0)
    if v is NA:
        return (2,)
    if isinstance(v, float) and math.isnan(v):
        return (1,)
    return (0, v)


KEYS = {"int64": st.integers(-3, 3), "float64": st.sampled_from([0.0, -0.0, 1.5, -2.0, math.nan, math.inf]),
        "string": st.sampled_from(["", "a", "b", "é"]), "bool": st.booleans()}


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_groups_match_reductions_of_their_rows(data):
    n = data.draw(st.integers(0, 80))
    nkeys = data.draw(st.integers(1, 2))
    kdts = [data.draw(st.sampled_from(list(KEYS))) for _ in range(nkeys)]
    keys = {f"k{j}": data.draw(st.lists(KEYS[dt] | st.none(), min_size=n, max_size=n)) for j, dt in enumerate(kdts)}
    f = data.draw(st.lists(st.floats(-1e6, 1e6) | st.sampled_from([math.nan, -0.0]) | st.none(), min_size=n, max_size=n))
    i = data.draw(st.lists(st.integers(-(2**62), 2**62) | st.none(), min_size=n, max_size=n))
    sort, dropna = data.draw(st.booleans()), data.draw(st.booleans())
    df = sdpd.DataFrame({**{k: sdpd.Series(v, dtype=dt) for (k, v), dt in zip(keys.items(), kdts)},
                         "f": sdpd.Series(f, dtype="float64"), "i": sdpd.Series(i, dtype="int64")})
    by = list(keys)
    g = df.groupby(by, sort=sort, dropna=dropna)

    # Expected groups: rows by key tuple (NaN equals NaN, -0.0 equals 0.0), first appearance.
    def norm(v):
        return ("nan",) if isinstance(v, float) and math.isnan(v) else (("na",) if v is NA else v)

    rows = [tuple(df[k].to_list()[r] for k in by) for r in range(n)]
    groups = {}
    for r, t in enumerate(rows):
        groups.setdefault(tuple(norm(v) for v in t), []).append(r)
    members = list(groups.values())
    if dropna:
        members = [m for m in members if not any(v is NA for v in rows[m[0]])]
    if sort:
        members.sort(key=lambda m: tuple(key_of(v) for v in rows[m[0]]))

    out = g.sum()
    assert out.shape[0] == len(members) and out.index.equals(sdpd.RangeIndex(len(members)))
    for j, k in enumerate(by):
        assert all(same(a, rows[m[0]][j]) for a, m in zip(out[k].to_list(), members))  # first row's key values

    results = {op: getattr(g, op)() for op in ("sum", "mean", "min", "max", "count", "var", "std", "first", "last")}
    sizes = g.size()["size"].to_list()
    for gi, m in enumerate(members):
        assert sizes[gi] == len(m)
        for col in ("f", "i"):
            part = sdpd.Series([df[col].to_list()[r] for r in m], dtype=df[col].dtype)
            for op in ("sum", "mean", "min", "max", "count", "var", "std"):
                assert same(results[op][col].to_list()[gi], getattr(part, op)()), (op, col, m)
            present = [v for v in part.to_list() if v is not NA]
            assert same(results["first"][col].to_list()[gi], present[0] if present else NA)
            assert same(results["last"][col].to_list()[gi], present[-1] if present else NA)


def test_float_group_sum_is_bitwise_the_series_sum():
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 3, 10_000)
    vals = rng.standard_normal(10_000) * 10.0 ** rng.integers(-8, 8, 10_000)
    df = sdpd.DataFrame({"k": keys, "v": vals})
    out = df.groupby("k").sum()
    for row, k in enumerate(out["k"].to_list()):
        assert same(out["v"].to_list()[row], float(np.sum(vals[keys == k])))  # numpy pairwise on the group's rows


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "first", "last", "size"])
def test_int_and_string_keys_match_pandas(op):
    k = ["b", "a", "b", "c", "a", "b"]
    i = [3, None, 5, 7, 1, None]
    s = ["x", "y", None, "w", "v", "u"]
    ours = sdpd.DataFrame({"k": k, "i": i, "s": s})
    theirs = pd.DataFrame({"k": k, "i": pd.array(i, dtype="Int64"), "s": pd.array(s, dtype="string[python]")})
    cols = ["i"] if op in ("sum", "mean") else ["i", "s"]
    if op == "size":
        assert ours.groupby("k").size()["size"].to_list() == theirs.groupby("k").size().tolist()
        return
    got = getattr(ours.groupby("k")[cols], op)()
    want = getattr(theirs.groupby("k", as_index=False)[cols], op)()
    assert got["k"].to_list() == want["k"].tolist()
    for c in cols:
        assert got[c].to_list() == [NA if v is pd.NA else v for v in want[c].tolist()]


def test_sort_order_rules():
    df = sdpd.DataFrame({"k": [2.0, math.nan, -0.0, None, math.inf, 0.0, -1.0], "v": list(range(7))})
    out = df.groupby("k", dropna=False).sum()
    keys = out["k"].to_list()
    assert keys[:3] == [-1.0, 0.0, 2.0] and math.copysign(1, keys[1]) == -1  # -0.0 came first: its value is kept
    assert keys[3] == math.inf and math.isnan(keys[4]) and keys[5] is NA  # NaN after +inf, NA last
    assert out["v"].to_list() == [6, 2 + 5, 0, 4, 1, 3]
    assert df.groupby("k").sum().shape[0] == 5  # dropna drops NA, keeps NaN (API.md §16)
    unsorted = df.groupby("k", sort=False, dropna=False).sum()["v"].to_list()
    assert unsorted == [0, 1, 2 + 5, 3, 4, 6]
    multi = sdpd.DataFrame({"a": ["x", "x", "y", None], "b": [2, 1, 1, 1], "v": [1, 2, 3, 4]})
    m = multi.groupby(["a", "b"], dropna=False).sum()
    assert m["a"].to_list() == ["x", "x", "y", NA] and m["b"].to_list() == [1, 2, 1, 1]


def test_skipna_and_all_na_groups():
    df = sdpd.DataFrame({"k": [1, 1, 2], "v": [None, 3.0, None]})
    g = df.groupby("k")
    assert g.sum()["v"].to_list() == [3.0, 0.0]
    assert g.mean()["v"].to_list() == [3.0, NA] and g.count()["v"].to_list() == [1, 0]
    assert g.sum(skipna=False)["v"].to_list() == [NA, NA]
    assert g.first()["v"].to_list() == [3.0, NA] and g.first(skipna=False)["v"].to_list() == [NA, NA]
    assert g.last(skipna=False)["v"].to_list() == [3.0, NA]
    assert g.var()["v"].to_list() == [NA, NA] and g.var(ddof=0)["v"].to_list() == [0.0, NA]


def test_agg_forms():
    df = sdpd.DataFrame({"k": ["a", "b", "a"], "x": [1, 2, 3], "y": [0.5, 1.0, 1.5]})
    g = df.groupby("k")
    assert g.agg("sum").columns.to_list() == ["k", "x", "y"]
    assert g.agg({"y": "max", "x": "min"}).columns.to_list() == ["k", "y", "x"]
    named = g.agg(total=("x", "sum"), top=("x", "max"), avg=("y", "mean"))
    assert named.columns.to_list() == ["k", "total", "top", "avg"] and named["total"].to_list() == [4, 2]
    for bad, error in (
        (lambda: g.agg(["sum", "mean"]), TypeError),
        (lambda: g.agg({"x": ["sum"]}), TypeError),
        (lambda: g.agg("median"), ValueError),
        (lambda: g.agg(k=("x", "sum")), ValueError),
        (lambda: g.agg(z=("k", "sum")), KeyError),
        (lambda: g.agg("sum", z=("x", "sum")), TypeError),
        (lambda: g.agg(), TypeError),
    ):
        with pytest.raises(error):
            bad()


def test_selection_and_errors():
    df = sdpd.DataFrame({"k": ["a", "b"], "s": ["x", "y"], "v": [1, 2]})
    with pytest.raises(TypeError):
        df.groupby("k").sum()  # the string column has no sum (API.md §15)
    assert df.groupby("k")["v"].sum()["v"].to_list() == [1, 2]
    assert df.groupby("k").max()["s"].to_list() == ["x", "y"]
    with pytest.raises(KeyError):
        df.groupby("zz")
    with pytest.raises(ValueError):
        df.groupby("k")["k"]
    with pytest.raises(ValueError):
        df.groupby([])
    with pytest.raises(TypeError):
        df.groupby("k", True)  # keyword-only (API.md §11 C2)
    empty = sdpd.DataFrame({"k": [None, None], "v": [1, 2]}, dtype=None)
    assert empty.groupby("k").sum().shape == (0, 2)
