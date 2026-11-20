"""sort_values and sort_index (docs/api/frame.md §8).

The oracle is Python's sorted(), which is stable in both directions as the
rules ask; several keys are applied last key first (an LSD sort), each key
with its own direction and NA position. Large inputs are checked against
numpy's stable argsort.
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
EXAMPLES = examples(300)

ELEMS = {
    "int64": st.integers(-(2**63), 2**63 - 1) | st.integers(-3, 3),
    "float64": st.floats() | st.sampled_from([0.0, -0.0, math.inf, -math.inf, math.nan]),
    "string": st.text(max_size=3) | st.sampled_from(["", "a", "é", "B"]),
    "bool": st.booleans(),
}


def value_key(v):
    if isinstance(v, float) and math.isnan(v):
        return (1, 0.0)  # NaN: larger than every number (docs/api/frame.md §8.2)
    return (0, v)


def oracle(columns, descending, na_last):
    order = list(range(len(columns[0]) if columns else 0))
    for values, desc, last in reversed(list(zip(columns, descending, na_last))):
        valid = [r for r in order if values[r] is not NA]
        missing = [r for r in order if values[r] is NA]
        valid = sorted(valid, key=lambda r: value_key(values[r]), reverse=desc)  # stable either way
        order = valid + missing if last else missing + valid
    return order


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_sort_values_matches_python_sorted(data):
    n = data.draw(st.integers(0, 60))
    nkeys = data.draw(st.integers(1, 3))
    dts = [data.draw(st.sampled_from(list(ELEMS))) for _ in range(nkeys)]
    cols = {f"c{j}": data.draw(st.lists(ELEMS[dt] | st.none(), min_size=n, max_size=n)) for j, dt in enumerate(dts)}
    df = sd.frame.DataFrame({k: sd.frame.Series(v, dtype=dt) for (k, v), dt in zip(cols.items(), dts)})
    desc = data.draw(st.lists(st.booleans(), min_size=nkeys, max_size=nkeys))
    nas = data.draw(st.lists(st.sampled_from(["first", "last"]), min_size=nkeys, max_size=nkeys))
    stable = data.draw(st.booleans())  # the sort is stable either way (docs/api/frame.md §8.1)
    by = list(cols)
    values = [df[k].tolist() for k in by]
    expected = oracle(values, desc, [p == "last" for p in nas])
    out = df.sort_values(by, descending=desc, stable=stable, na_position=nas)
    assert out.index.tolist() == expected
    if nkeys == 1:
        s = df[by[0]].sort_values(descending=desc[0], stable=stable, na_position=nas[0])
        assert s.index.tolist() == expected


def test_float_order_rules():
    s = sd.frame.Series([1.0, math.nan, -math.inf, math.inf, None, -0.0, 0.0])
    up = s.sort_values().tolist()
    assert up[:3] == [-math.inf, -0.0, 0.0] and math.copysign(1, up[1]) == -1  # -0.0 == 0.0: original order
    assert up[3:5] == [1.0, math.inf] and math.isnan(up[5]) and up[6] is NA
    down = s.sort_values(descending=True, na_position="first").tolist()
    assert down[0] is NA and math.isnan(down[1]) and down[2:] == [math.inf, 1.0, -0.0, 0.0, -math.inf]


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("stable", [True, False])
def test_large_matches_numpy_stable_argsort(dtype, descending, stable):
    rng = np.random.default_rng(7)
    values = rng.integers(-1000, 1000, 200_000).astype(dtype)
    if dtype == "float64":
        values *= 1.5
    s = sd.frame.Series(values)
    expected = np.argsort(-values if descending else values, kind="stable")  # negating keeps ties in order
    assert s.sort_values(descending=descending, stable=stable).index.tolist() == expected.tolist()


@pytest.mark.parametrize("descending", [False, True])
def test_strings_match_pandas_stable(descending):
    words = ["pear", "Apple", "apple", "é", "", "pear", "b"] * 3
    got = sd.frame.Series(words).sort_values(descending=descending).index.tolist()
    want = pd.Series(words).sort_values(ascending=not descending, kind="stable").index.tolist()
    assert got == want


def test_sort_index():
    s = sd.frame.Series([1, 2, 3], index=["c", "a", "b"], name="v")
    assert s.sort_index().index.tolist() == ["a", "b", "c"] and s.sort_index().tolist() == [2, 3, 1]
    assert s.sort_index(descending=True).tolist() == [1, 3, 2]
    assert s.sort_index(descending=True, stable=False).tolist() == [1, 3, 2]
    df = sd.frame.DataFrame({"x": [1, 2, 3]}, index=[30, 10, 20])
    assert df.sort_index()["x"].tolist() == [2, 3, 1]
    assert df.sort_index(descending=True, stable=False)["x"].tolist() == [1, 3, 2]
    assert df.sort_index(ignore_index=True).index.equals(sd.frame.RangeIndex(3))


def test_result_keeps_labels_and_names():
    s = sd.frame.Series([3, 1, 2], index=sd.frame.Index(["a", "b", "c"], name="k"), name="v")
    out = s.sort_values()
    assert out.index.tolist() == ["b", "c", "a"] and out.index.name == "k" and out.name == "v"
    assert isinstance(sd.frame.Series([2, 1]).sort_values().index, sd.frame.Index)
    assert sd.frame.Series([2, 1]).sort_values(ignore_index=True).index.equals(sd.frame.RangeIndex(2))


def test_arguments():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [3, 4]})
    for bad, error in (
        (lambda: df.sort_values(["a", "b"], descending=[True]), ValueError),
        (lambda: df.sort_values("a", descending=1), TypeError),  # a bool option takes True or False (API.md §2)
        (lambda: df.sort_values("a", stable=None), TypeError),
        (lambda: df["a"].sort_values(stable=1), TypeError),
        (lambda: df.sort_index(descending="yes"), TypeError),
        (lambda: df.sort_values("a", ascending=True), TypeError),  # renamed: descending (docs/api/frame.md §8.1)
        (lambda: df.sort_values(by="a"), TypeError),  # by is positional-only (docs/api/frame.md §1.1)
        (lambda: df.sort_values("a", na_position="middle"), ValueError),
        (lambda: df.sort_values("a", na_position=["first", "last"]), ValueError),
        (lambda: df.sort_values([]), ValueError),
        (lambda: df.sort_values("zz"), KeyError),
        (lambda: df.sort_values("a", True), TypeError),  # keyword-only (docs/api/frame.md §1.1)
        (lambda: df["a"].sort_values(kind="stable"), TypeError),  # no kind: always stable (docs/api/frame.md §8.1)
        (lambda: df.sort_index(na_position="last"), TypeError),  # no NA in an Index (docs/api/frame.md §8.1)
    ):
        with pytest.raises(error):
            bad()


# --- long common prefixes and one-byte keys -------------------------------------


@pytest.mark.parametrize("n", [5, 300, 20_000, 300_000])
def test_strings_with_a_long_common_prefix(n):
    # Runs of equal 8-byte prefixes go one radix level deeper (docs/design/frame.md §9.4),
    # down to strings that differ only in trailing NUL bytes; the order is the byte order.
    rng = np.random.default_rng(n)
    vals = [f"customer_{int(v):07d}" for v in rng.integers(0, 10**7, n)]
    vals[: n // 10] = ["https://example.com/" + "ab\0"[int(v) % 3] * int(v % 9) for v in rng.integers(0, 100, n // 10)]
    vals[-3:] = ["https://example.com/", "https://example.com/\0", "https://example.com/\0\0"]
    expected = sorted(vals, key=lambda v: v.encode())
    results, default = set(), sd.get_num_threads()
    for t in sorted({1, 3, default}):
        sd.set_num_threads(t)
        try:
            s = sd.frame.Series(vals)
            got = s.sort_values()
            down = s.sort_values(descending=True).tolist()
            uniq = tuple(s.unique_values().tolist()) if n <= 20_000 else None
        finally:
            sd.set_num_threads(default)
        assert got.tolist() == expected
        assert got.index.tolist() == sorted(range(n), key=lambda i: vals[i].encode())  # stable
        assert down == sorted(vals, key=lambda v: v.encode(), reverse=True)
        results.add((tuple(got.index.tolist()), uniq))
    assert len(results) == 1


def test_bool_keys_sort_stably_in_one_pass():
    b = (np.random.default_rng(1).random(300_000) < 0.5).tolist()
    s = sd.frame.Series(b)
    assert s.sort_values().index.tolist() == sorted(range(len(b)), key=lambda i: b[i])
    assert s.sort_values(descending=True).index.tolist() == sorted(range(len(b)), key=lambda i: not b[i])
    df = sd.frame.DataFrame({"b": b, "i": list(range(len(b)))})
    assert df.sort_values(["b", "i"], descending=[True, False])["i"].tolist() == sorted(
        range(len(b)), key=lambda i: (not b[i], i)
    )
