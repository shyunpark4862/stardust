"""groupby (docs/api/frame.md §7).

The oracle for each group is the reduction of a Series holding just that
group's rows: a group's result must equal it, bit for bit (that is why group
sums are pairwise per group; docs/design/frame.md §9.1, §13.1). Group order is checked against a
plain-Python sort; integer and string keys are also checked against pandas.
"""

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples

NA = sd.frame.NA
EXAMPLES = examples(200)


def same(x, y):
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) and isinstance(y, float):
        return (math.isnan(x) and math.isnan(y)) or np.float64(x).tobytes() == np.float64(y).tobytes()
    return x == y and type(x) is type(y)


def key_of(v):  # the sort key of docs/api/frame.md §7.2 (NA last, NaN after +inf, -0.0 == 0.0)
    if v is NA:
        return (2,)
    if isinstance(v, float) and math.isnan(v):
        return (1,)
    return (0, v)


KEYS = {
    "int64": st.integers(-3, 3),
    "float64": st.sampled_from([0.0, -0.0, 1.5, -2.0, math.nan, math.inf]),
    "string": st.sampled_from(["", "a", "b", "é"]),
    "bool": st.booleans(),
}


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_groups_match_reductions_of_their_rows(data):
    n = data.draw(st.integers(0, 80))
    nkeys = data.draw(st.integers(1, 2))
    kdts = [data.draw(st.sampled_from(list(KEYS))) for _ in range(nkeys)]
    keys = {f"k{j}": data.draw(st.lists(KEYS[dt] | st.none(), min_size=n, max_size=n)) for j, dt in enumerate(kdts)}
    f = data.draw(
        st.lists(st.floats(-1e6, 1e6) | st.sampled_from([math.nan, -0.0]) | st.none(), min_size=n, max_size=n)
    )
    i = data.draw(st.lists(st.integers(-(2**62), 2**62) | st.none(), min_size=n, max_size=n))
    sort, dropna = data.draw(st.booleans()), data.draw(st.booleans())
    df = sd.frame.DataFrame(
        {
            **{k: sd.frame.Series(v, dtype=dt) for (k, v), dt in zip(keys.items(), kdts)},
            "f": sd.frame.Series(f, dtype="float64"),
            "i": sd.frame.Series(i, dtype="int64"),
        }
    )
    by = list(keys)
    g = df.groupby(by, sort=sort, dropna=dropna)

    # Expected groups: rows by key tuple (NaN equals NaN, -0.0 equals 0.0), first appearance.
    def norm(v):
        return ("nan",) if isinstance(v, float) and math.isnan(v) else (("na",) if v is NA else v)

    rows = [tuple(df[k].tolist()[r] for k in by) for r in range(n)]
    groups = {}
    for r, t in enumerate(rows):
        groups.setdefault(tuple(norm(v) for v in t), []).append(r)
    members = list(groups.values())
    if dropna:
        members = [m for m in members if not any(v is NA for v in rows[m[0]])]
    if sort:
        members.sort(key=lambda m: tuple(key_of(v) for v in rows[m[0]]))

    out = g.sum()
    assert out.shape[0] == len(members) and out.index.equals(sd.frame.RangeIndex(len(members)))
    for j, k in enumerate(by):
        assert all(same(a, rows[m[0]][j]) for a, m in zip(out[k].tolist(), members))  # first row's key values

    results = {op: getattr(g, op)() for op in ("sum", "mean", "min", "max", "count", "var", "std", "first", "last")}
    sizes = g.size()["size"].tolist()
    for gi, m in enumerate(members):
        assert sizes[gi] == len(m)
        for col in ("f", "i"):
            part = sd.frame.Series([df[col].tolist()[r] for r in m], dtype=df[col].dtype)
            for op in ("sum", "mean", "min", "max", "count", "var", "std"):
                assert same(results[op][col].tolist()[gi], getattr(part, op)()), (op, col, m)
            present = [v for v in part.tolist() if v is not NA]
            assert same(results["first"][col].tolist()[gi], present[0] if present else NA)
            assert same(results["last"][col].tolist()[gi], present[-1] if present else NA)


def test_float_group_sum_is_bitwise_the_series_sum():
    rng = np.random.default_rng(3)
    keys = rng.integers(0, 3, 10_000)
    vals = rng.standard_normal(10_000) * 10.0 ** rng.integers(-8, 8, 10_000)
    df = sd.frame.DataFrame({"k": keys, "v": vals})
    out = df.groupby("k").sum()
    for row, k in enumerate(out["k"].tolist()):
        assert same(out["v"].tolist()[row], float(np.sum(vals[keys == k])))  # numpy pairwise on the group's rows


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "count", "first", "last", "size"])
def test_int_and_string_keys_match_pandas(op):
    k = ["b", "a", "b", "c", "a", "b"]
    i = [3, None, 5, 7, 1, None]
    s = ["x", "y", None, "w", "v", "u"]
    ours = sd.frame.DataFrame({"k": k, "i": i, "s": s})
    theirs = pd.DataFrame({"k": k, "i": pd.array(i, dtype="Int64"), "s": pd.array(s, dtype="string[python]")})
    cols = ["i"] if op in ("sum", "mean") else ["i", "s"]
    if op == "size":
        assert ours.groupby("k").size()["size"].tolist() == theirs.groupby("k").size().tolist()
        return
    got = getattr(ours.groupby("k")[cols], op)()
    want = getattr(theirs.groupby("k", as_index=False)[cols], op)()
    assert got["k"].tolist() == want["k"].tolist()
    for c in cols:
        assert got[c].tolist() == [NA if v is pd.NA else v for v in want[c].tolist()]


def test_sort_order_rules():
    df = sd.frame.DataFrame({"k": [2.0, math.nan, -0.0, None, math.inf, 0.0, -1.0], "v": list(range(7))})
    out = df.groupby("k", dropna=False).sum()
    keys = out["k"].tolist()
    assert keys[:3] == [-1.0, 0.0, 2.0] and math.copysign(1, keys[1]) == -1  # -0.0 came first: its value is kept
    assert keys[3] == math.inf and math.isnan(keys[4]) and keys[5] is NA  # NaN after +inf, NA last
    assert out["v"].tolist() == [6, 2 + 5, 0, 4, 1, 3]
    assert df.groupby("k").sum().shape[0] == 5  # dropna drops NA, keeps NaN (docs/api/frame.md §7.2)
    unsorted = df.groupby("k", sort=False, dropna=False).sum()["v"].tolist()
    assert unsorted == [0, 1, 2 + 5, 3, 4, 6]
    multi = sd.frame.DataFrame({"a": ["x", "x", "y", None], "b": [2, 1, 1, 1], "v": [1, 2, 3, 4]})
    m = multi.groupby(["a", "b"], dropna=False).sum()
    assert m["a"].tolist() == ["x", "x", "y", NA] and m["b"].tolist() == [1, 2, 1, 1]


def test_skipna_and_all_na_groups():
    df = sd.frame.DataFrame({"k": [1, 1, 2], "v": [None, 3.0, None]})
    g = df.groupby("k")
    assert g.sum()["v"].tolist() == [3.0, 0.0]
    assert g.mean()["v"].tolist() == [3.0, NA] and g.count()["v"].tolist() == [1, 0]
    assert g.sum(skipna=False)["v"].tolist() == [NA, NA]
    assert g.first()["v"].tolist() == [3.0, NA] and g.first(skipna=False)["v"].tolist() == [NA, NA]
    assert g.last(skipna=False)["v"].tolist() == [3.0, NA]
    assert g.var()["v"].tolist() == [0.0, NA] and g.var(correction=1)["v"].tolist() == [NA, NA]


def test_agg_forms():
    df = sd.frame.DataFrame({"k": ["a", "b", "a"], "x": [1, 2, 3], "y": [0.5, 1.0, 1.5]})
    g = df.groupby("k")
    assert g.agg("sum").columns.tolist() == ["k", "x", "y"]
    assert g.agg({"y": "max", "x": "min"}).columns.tolist() == ["k", "y", "x"]
    named = g.agg(total=("x", "sum"), top=("x", "max"), avg=("y", "mean"))
    assert named.columns.tolist() == ["k", "total", "top", "avg"] and named["total"].tolist() == [4, 2]
    # "size" is an aggregation too, as size() (DESIGN.md §10.2)
    assert repr(g.agg("size")) == repr(g.size()) and g.agg("size")["size"].tolist() == [2, 1]
    assert g.agg({"x": "size"})["x"].tolist() == [2, 1] and g.agg(n=("y", "size"))["n"].tolist() == [2, 1]
    # every form of agg("std") is std(): correction=0
    for f in ("std", "var"):
        assert repr(g.agg(f)) == repr(getattr(g, f)()) == repr(g.agg({"x": f, "y": f}))
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
    df = sd.frame.DataFrame({"k": ["a", "b"], "s": ["x", "y"], "v": [1, 2]})
    with pytest.raises(TypeError):
        df.groupby("k").sum()  # the string column has no sum (docs/api/frame.md §6.3)
    assert df.groupby("k")["v"].sum()["v"].tolist() == [1, 2]
    assert df.groupby("k").max()["s"].tolist() == ["x", "y"]
    with pytest.raises(KeyError):
        df.groupby("zz")
    with pytest.raises(ValueError):
        df.groupby("k")["k"]
    with pytest.raises(ValueError):
        df.groupby([])
    with pytest.raises(TypeError):
        df.groupby("k", True)  # keyword-only (docs/api/frame.md §1.1)
    for bad in (dict(sort=1), dict(dropna=None)):  # bool options (DESIGN.md §10.1 #4)
        with pytest.raises(TypeError, match=r"^groupby\(\): '(sort|dropna)' must be True or False, not "):
            df.groupby("k", **bad)
    for bad in ((c for c in ["k"]), df["k"], {"k"}):  # a name or a list or tuple of names (DESIGN.md §10.1 #12)
        with pytest.raises(TypeError, match=r"^groupby\(\): 'by' must be a column name or a list of them, not "):
            df.groupby(bad)
    assert df.groupby(("k",))["v"].sum()["v"].tolist() == [1, 2]
    with pytest.raises(TypeError):
        df.groupby("k").first(skipna=0)
    empty = sd.frame.DataFrame({"k": [None, None], "v": [1, 2]}, dtype=None)
    assert empty.groupby("k").sum().shape == (0, 2)


def test_groupby_sees_the_frame_as_it_was():
    # The grouping is computed on first use, so the groupby keeps the columns
    # of the moment it was made; later writes to df do not mix in
    # (docs/design/frame.md §5.3).
    df = sd.frame.DataFrame({"k": [1, 1, 2], "v": [1.0, 2.0, 3.0]})
    g = df.groupby("k")
    before = g.sum()
    df["k"] = [5, 6, 6]
    df["v"] = [10.0, 20.0, 30.0]
    df.loc[0, "v"] = 100.0
    for result in (g.sum(), g["v"].sum()):
        assert result["k"].tolist() == [1, 2]
        assert result["v"].tolist() == [3.0, 3.0]
    assert before["v"].tolist() == [3.0, 3.0]
    g2 = df.groupby("k")  # made after the writes: sees them
    df["v"] = [0.0, 0.0, 0.0]
    assert g2.sum()["v"].tolist() == [100.0, 50.0]


def test_groupby_is_not_iterable():  # docs/api/frame.md §7.1
    g = sd.frame.DataFrame({"q": ["x", "y"], "v": [1, 2]}).groupby("q")
    for call in (lambda: list(g), lambda: iter(g), lambda: [x for x in g]):
        with pytest.raises(TypeError, match="^'GroupBy' object is not iterable$"):
            call()
    assert g["v"].sum()["v"].tolist() == [1, 2]  # g[...] still selects value columns


def test_agg_with_a_callable_names_what_agg_takes():  # docs/api/frame.md §7.4
    g = sd.frame.DataFrame({"q": ["x", "y"], "v": [1, 2]}).groupby("q")
    accepted = (
        r"; pass a function name \(a str such as 'sum'\), a dict of \{column: function name\}, "
        r"or named aggregations \(name=\(column, function name\)\)$"
    )
    for func in (sum, lambda x: x, max):
        with pytest.raises(TypeError, match=r"^agg\(\): a callable is not supported" + accepted):
            g.agg(func)
    with pytest.raises(TypeError, match=r"^agg\(\): int is not supported" + accepted):
        g.agg(5)
    for funcs in (["sum"], ("sum", "max")):
        with pytest.raises(TypeError, match=r"^agg\(\): a list of functions is not supported"):
            g.agg(funcs)
