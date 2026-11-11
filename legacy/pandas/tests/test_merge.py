"""M8: merge and concat (API.md §5, §18).

The merge oracle is a nested loop written straight from the order rules of
API.md §5; each side carries a row-id column so the test sees exactly which
rows were paired, in which order.
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
KEYS = {"int64": st.integers(-2, 2), "float64": st.sampled_from([0.0, -0.0, 1.5, math.nan]),
        "string": st.sampled_from(["a", "b", ""]), "bool": st.booleans()}


def norm(v):
    return ("nan",) if isinstance(v, float) and math.isnan(v) else v  # NaN = NaN, -0.0 = 0.0 (Python already)


def oracle(lk, rk, how):
    """(left row, right row) pairs in the order of API.md §5; None = no row."""
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
    left = sdpd.DataFrame({**{k: sdpd.Series([t[j] for t in lk], dtype=dt) for j, (k, dt) in enumerate(zip(names, dts))},
                           "lid": list(range(nl))}, dtype=None)
    right = sdpd.DataFrame({**{k: sdpd.Series([t[j] for t in rk], dtype=dt) for j, (k, dt) in enumerate(zip(names, dts))},
                            "rid": list(range(nr))})
    lk = [tuple(left[k].to_list()[i] for k in names) for i in range(nl)]  # as stored (None -> NA)
    rk = [tuple(right[k].to_list()[i] for k in names) for i in range(nr)]
    how = data.draw(st.sampled_from(["inner", "left", "right", "outer"]))
    out = sdpd.merge(left, right, how=how, on=names)
    pairs = oracle(lk, rk, how)
    assert out["lid"].to_list() == [NA if l is None else l for l, _ in pairs]
    assert out["rid"].to_list() == [NA if r is None else r for _, r in pairs]
    assert out.columns.to_list() == names + ["lid", "rid"] and out.index.equals(sdpd.RangeIndex(len(pairs)))
    for j, k in enumerate(names):  # one key column: left's value, else right's
        want = [lk[l][j] if l is not None else rk[r][j] for l, r in pairs]
        got = out[k].to_list()
        assert all((g is NA and w is NA) or norm(g) == norm(w) for g, w in zip(got, want))
    if pairs:
        srt = sdpd.merge(left, right, how=how, on=names, sort=True)
        expected = sdpd.DataFrame({k: out[k] for k in names}).sort_values(names).index.to_list()
        assert srt["lid"].to_list() == [out["lid"].to_list()[i] for i in expected]


@pytest.mark.parametrize("how", ["inner", "left"])
def test_matches_pandas_without_na(how):
    left = {"k": ["b", "a", "b", "c"], "lid": [0, 1, 2, 3]}
    right = {"k": ["b", "d", "a", "b"], "rid": [0, 1, 2, 3]}
    ours = sdpd.merge(sdpd.DataFrame(left), sdpd.DataFrame(right), how=how, on="k")
    theirs = pd.merge(pd.DataFrame(left), pd.DataFrame(right), how=how, on="k")
    assert ours["lid"].to_list() == theirs["lid"].tolist()
    assert ours["rid"].to_list() == [NA if math.isnan(v) else int(v) for v in theirs["rid"].astype(float).tolist()]


def test_left_on_right_on_and_suffixes():
    left = sdpd.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    right = sdpd.DataFrame({"key": [2, 3], "v": ["x", "y"], "k": [7, 8]})
    out = sdpd.merge(left, right, how="outer", left_on="k", right_on="key", suffixes=("_l", "_r"))
    assert out.columns.to_list() == ["k_l", "v_l", "key", "v_r", "k_r"]
    assert out["k_l"].to_list() == [1, 2, NA] and out["key"].to_list() == [NA, 2, 3]
    srt = sdpd.merge(left, right, how="outer", left_on="k", right_on="key", sort=True, suffixes=("_l", "_r"))
    assert srt["v_r"].to_list() == [NA, "x", "y"]
    with pytest.raises(ValueError):  # "v" + "" collides with "v" from the right side... both suffixes empty
        sdpd.merge(left, sdpd.DataFrame({"k": [1], "v": [1]}), on="k", suffixes=("", ""))


def test_merge_errors():
    a = sdpd.DataFrame({"k": [1], "x": [1]})
    b = sdpd.DataFrame({"k": [1.0], "y": [2]})
    for bad, error in (
        (lambda: sdpd.merge(a, b, on="k"), TypeError),  # int64 key with float64 key (API.md §18)
        (lambda: sdpd.merge(a, a), ValueError),  # keys must be given
        (lambda: sdpd.merge(a, a, on="k", left_on="k", right_on="k"), ValueError),
        (lambda: sdpd.merge(a, a, how="cross", on="k"), ValueError),
        (lambda: sdpd.merge(a, a, on="zz"), KeyError),
        (lambda: sdpd.merge(a, a["k"], on="k"), TypeError),
        (lambda: sdpd.merge(a, a, left_on=["k", "x"], right_on=["k"]), ValueError),
        (lambda: a.merge(a, "k"), TypeError),  # keyword-only (API.md §11 C2)
    ):
        with pytest.raises(error):
            bad()
    assert a.merge(a, on="k").columns.to_list() == ["k", "x_x", "x_y"]


def test_concat_frames():
    a = sdpd.DataFrame({"x": [1, 2], "y": [1.5, 2.5]}, index=["p", "q"])
    b = sdpd.DataFrame({"y": [3.5], "z": ["s"]}, index=["p"])
    out = sdpd.concat([a, b])
    assert out.columns.to_list() == ["x", "y", "z"] and out.index.to_list() == ["p", "q", "p"]
    assert out["x"].to_list() == [1, 2, NA] and out["z"].to_list() == [NA, NA, "s"] and out["x"].dtype == "int64"
    inner = sdpd.concat([a, b], join="inner", ignore_index=True)
    assert inner.columns.to_list() == ["y"] and inner.index.equals(sdpd.RangeIndex(3))
    assert sdpd.concat([sdpd.DataFrame({"a": [1]}), sdpd.DataFrame({"a": [2]})]).index.to_list() == [0, 0]
    for bad, error in (
        (lambda: sdpd.concat([a, sdpd.DataFrame({"x": [0.5]}, index=["r"])]), TypeError),  # int64 + float64
        (lambda: sdpd.concat([a, sdpd.DataFrame({"x": [1]})]), TypeError),  # string labels + int labels
        (lambda: sdpd.concat([]), ValueError),
        (lambda: sdpd.concat([a, a["x"]]), TypeError),
        (lambda: sdpd.concat([a], join="left"), ValueError),
    ):
        with pytest.raises(error):
            bad()


def test_concat_series():
    s = sdpd.Series([1, 2], name="v")
    out = sdpd.concat([s, s.tail(1)])
    assert out.to_list() == [1, 2, 2] and out.index.to_list() == [0, 1, 1] and out.name == "v"
    assert sdpd.concat([s, sdpd.Series([3], name="w")]).name is None
    with pytest.raises(TypeError):
        sdpd.concat([s, sdpd.Series([1.5])])
