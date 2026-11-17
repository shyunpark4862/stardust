"""merge and concat (docs/api/frame.md §9).

The merge oracle is a nested loop written straight from the order rules of
docs/api/frame.md §9.1; each side carries a row-id column so the test sees exactly which
rows were paired, in which order.
"""

import math
import os

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

NA = sd.frame.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))
KEYS = {"int64": st.integers(-2, 2), "float64": st.sampled_from([0.0, -0.0, 1.5, math.nan]),
        "string": st.sampled_from(["a", "b", ""]), "bool": st.booleans()}


def norm(v):
    return ("nan",) if isinstance(v, float) and math.isnan(v) else v  # NaN = NaN, -0.0 = 0.0 (Python already)


def oracle(lk, rk, how):
    """(left row, right row) pairs in the order of docs/api/frame.md §9.1; None = no row."""
    def matches(a, b):
        return all(x is not NA and y is not NA and norm(x) == norm(y) for x, y in zip(a, b))

    def one_sided(probe, build, keep_probe, keep_build):
        out, hit = [], set()
        for p, pk in enumerate(probe):
            found = [b for b, bk in enumerate(build) if matches(pk, bk)]
            hit.update(found)
            out += [(p, b) for b in found] or ([(p, None)] if keep_probe else [])
        if keep_build:
            out += [(None, b) for b in range(len(build)) if b not in hit]
        return out

    if how == "right":
        return [(l, r) for r, l in one_sided(rk, lk, True, False)]
    return one_sided(lk, rk, how != "inner", how == "outer")


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_merge_rows_and_order(data):
    nkeys = data.draw(st.integers(1, 2))
    dts = [data.draw(st.sampled_from(list(KEYS))) for _ in range(nkeys)]
    nl, nr = data.draw(st.integers(0, 12)), data.draw(st.integers(0, 12))
    lk = [tuple(data.draw(KEYS[dt] | st.none()) for dt in dts) for _ in range(nl)]
    rk = [tuple(data.draw(KEYS[dt] | st.none()) for dt in dts) for _ in range(nr)]
    names = [f"k{j}" for j in range(nkeys)]
    left = sd.frame.DataFrame({**{k: sd.frame.Series([t[j] for t in lk], dtype=dt) for j, (k, dt) in enumerate(zip(names, dts))},
                           "lid": list(range(nl))}, dtype=None)
    right = sd.frame.DataFrame({**{k: sd.frame.Series([t[j] for t in rk], dtype=dt) for j, (k, dt) in enumerate(zip(names, dts))},
                            "rid": list(range(nr))})
    lk = [tuple(left[k].tolist()[i] for k in names) for i in range(nl)]  # as stored (None -> NA)
    rk = [tuple(right[k].tolist()[i] for k in names) for i in range(nr)]
    how = data.draw(st.sampled_from(["inner", "left", "right", "outer"]))
    out = sd.frame.merge(left, right, how=how, on=names)
    pairs = oracle(lk, rk, how)
    assert out["lid"].tolist() == [NA if l is None else l for l, _ in pairs]
    assert out["rid"].tolist() == [NA if r is None else r for _, r in pairs]
    assert out.columns.tolist() == names + ["lid", "rid"] and out.index.equals(sd.frame.RangeIndex(len(pairs)))
    for j, k in enumerate(names):  # one key column: left's value, else right's
        want = [lk[l][j] if l is not None else rk[r][j] for l, r in pairs]
        got = out[k].tolist()
        assert all((g is NA and w is NA) or norm(g) == norm(w) for g, w in zip(got, want))
    if pairs:
        srt = sd.frame.merge(left, right, how=how, on=names, sort=True)
        expected = sd.frame.DataFrame({k: out[k] for k in names}).sort_values(names).index.tolist()
        assert srt["lid"].tolist() == [out["lid"].tolist()[i] for i in expected]


@pytest.mark.parametrize("how", ["inner", "left"])
def test_matches_pandas_without_na(how):
    left = {"k": ["b", "a", "b", "c"], "lid": [0, 1, 2, 3]}
    right = {"k": ["b", "d", "a", "b"], "rid": [0, 1, 2, 3]}
    ours = sd.frame.merge(sd.frame.DataFrame(left), sd.frame.DataFrame(right), how=how, on="k")
    theirs = pd.merge(pd.DataFrame(left), pd.DataFrame(right), how=how, on="k")
    assert ours["lid"].tolist() == theirs["lid"].tolist()
    assert ours["rid"].tolist() == [NA if math.isnan(v) else int(v) for v in theirs["rid"].astype(float).tolist()]


def test_left_on_right_on_and_suffixes():
    left = sd.frame.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    right = sd.frame.DataFrame({"key": [2, 3], "v": ["x", "y"], "k": [7, 8]})
    out = sd.frame.merge(left, right, how="outer", left_on="k", right_on="key", suffixes=("_l", "_r"))
    assert out.columns.tolist() == ["k_l", "v_l", "key", "v_r", "k_r"]
    assert out["k_l"].tolist() == [1, 2, NA] and out["key"].tolist() == [NA, 2, 3]
    srt = sd.frame.merge(left, right, how="outer", left_on="k", right_on="key", sort=True, suffixes=("_l", "_r"))
    assert srt["v_r"].tolist() == [NA, "x", "y"]
    with pytest.raises(ValueError):  # "v" + "" collides with "v" from the right side... both suffixes empty
        sd.frame.merge(left, sd.frame.DataFrame({"k": [1], "v": [1]}), on="k", suffixes=("", ""))


def test_merge_errors():
    a = sd.frame.DataFrame({"k": [1], "x": [1]})
    b = sd.frame.DataFrame({"k": [1.0], "y": [2]})
    for bad, error in (
        (lambda: sd.frame.merge(a, b, on="k"), TypeError),  # int64 key with float64 key (docs/api/frame.md §9.1)
        (lambda: sd.frame.merge(a, a), ValueError),  # keys must be given
        (lambda: sd.frame.merge(a, a, on="k", left_on="k", right_on="k"), ValueError),
        (lambda: sd.frame.merge(a, a, how="cross", on="k"), ValueError),
        (lambda: sd.frame.merge(a, a, on="zz"), KeyError),
        (lambda: sd.frame.merge(a, a["k"], on="k"), TypeError),
        (lambda: sd.frame.merge(a, a, left_on=["k", "x"], right_on=["k"]), ValueError),
        (lambda: a.merge(a, "k"), TypeError),  # keyword-only (docs/api/frame.md §1.1)
    ):
        with pytest.raises(error):
            bad()
    assert a.merge(a, on="k").columns.tolist() == ["k", "x_x", "x_y"]


def test_concat_frames():
    a = sd.frame.DataFrame({"x": [1, 2], "y": [1.5, 2.5]}, index=["p", "q"])
    b = sd.frame.DataFrame({"y": [3.5], "z": ["s"]}, index=["p"])
    out = sd.frame.concat([a, b])
    assert out.columns.tolist() == ["x", "y", "z"] and out.index.tolist() == ["p", "q", "p"]
    assert out["x"].tolist() == [1, 2, NA] and out["z"].tolist() == [NA, NA, "s"] and out["x"].dtype is sd.int64
    inner = sd.frame.concat([a, b], join="inner", ignore_index=True)
    assert inner.columns.tolist() == ["y"] and inner.index.equals(sd.frame.RangeIndex(3))
    assert sd.frame.concat([sd.frame.DataFrame({"a": [1]}), sd.frame.DataFrame({"a": [2]})]).index.tolist() == [0, 0]
    # dtypes promote, as the array area's concat (DESIGN.md §10.1 #2): int64 + float64 is float64
    up = sd.frame.concat([a, sd.frame.DataFrame({"x": [0.5], "z": ["t"]}, index=["r"])])
    assert up["x"].dtype is sd.float64 and up["x"].tolist() == [1.0, 2.0, 0.5] and up["y"].tolist() == [1.5, 2.5, NA]
    for bad, error in (
        (lambda: sd.frame.concat([a, sd.frame.DataFrame({"x": ["s"]}, index=["r"])]), TypeError),  # int64 + string
        (lambda: sd.frame.concat([a, sd.frame.DataFrame({"x": [1]})]), TypeError),  # string labels + int labels
        (lambda: sd.frame.concat(x for x in [a, a]), TypeError),  # a list or tuple, as sd.concat
        (lambda: sd.frame.concat(a), TypeError),
        (lambda: sd.frame.concat([a], join=0), TypeError),
        (lambda: sd.frame.concat([a], ignore_index=1), TypeError),
        (lambda: sd.frame.concat([]), ValueError),
        (lambda: sd.frame.concat([a, a["x"]]), TypeError),
        (lambda: sd.frame.concat([a], join="left"), ValueError),
    ):
        with pytest.raises(error):
            bad()


def test_concat_series():
    s = sd.frame.Series([1, 2], name="v")
    out = sd.frame.concat([s, s.tail(1)])
    assert out.tolist() == [1, 2, 2] and out.index.tolist() == [0, 1, 1] and out.name == "v"
    assert sd.frame.concat([s, sd.frame.Series([3], name="w")]).name is None
    assert sd.frame.concat((s, s)).tolist() == [1, 2, 1, 2]  # a tuple too
    # bool < int64 < float64, as sd.concat (DESIGN.md §10.1 #2)
    for parts in (([1, 2], [1.5]), ([1, 2], [True]), ([True], [0.5, None]), ([True, False], [False])):
        got = sd.frame.concat([sd.frame.Series(p) for p in parts])
        want = sd.concat([sd.asarray([0 if v is None else v for v in p]) for p in parts])
        assert got.dtype == want.dtype
        assert [0 if v is NA else v for v in got.tolist()] == want.tolist()
    with pytest.raises(TypeError):
        sd.frame.concat([s, sd.frame.Series(["a"])])


def test_suffixes_is_a_tuple_of_two_str():  # docs/api/frame.md §9.1
    a = sd.frame.DataFrame({"k": [1, 2], "v": [1, 2]})
    b = sd.frame.DataFrame({"k": [1, 2], "v": [3, 4]})
    for bad in (["_a", "_b"], ("_a",), ("_a", "_b", "_c"), ("_a", None), "ab", None):
        with pytest.raises(TypeError, match=r"^merge\(\): 'suffixes' must be a tuple of two str$"):
            sd.frame.merge(a, b, on="k", suffixes=bad)
    assert list(sd.frame.merge(a, b, on="k", suffixes=("", "_b")).columns) == ["k", "v", "v_b"]
