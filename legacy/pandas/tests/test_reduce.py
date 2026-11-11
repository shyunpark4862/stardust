"""M5: reductions (API.md §15).

Float sums are numpy's pairwise summation, so numpy run over the same values
(NA rows hold 0.0, ARCHITECTURE.md §2.4) is a bit-for-bit oracle; without NA,
pandas is too. int64 means sum exactly, which Python's own int and float
arithmetic reproduce.
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


def bits(x):
    return np.float64(x).tobytes()


def same(x, y):
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) or isinstance(y, float):
        return isinstance(x, float) and isinstance(y, float) and (bits(x) == bits(y) or (math.isnan(x) and math.isnan(y)))
    return x == y and type(x) is type(y)


FLOATS = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12) | st.sampled_from([0.0, -0.0, 1e-300])
LENGTHS = st.sampled_from([0, 1, 7, 8, 9, 127, 128, 129, 300, 1000, 5000]) | st.integers(0, 64)


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_float_reductions_match_numpy_bitwise(data):
    n = data.draw(LENGTHS)
    values = data.draw(st.lists(FLOATS | st.none(), min_size=n, max_size=n))
    s = sdpd.Series(values, dtype="float64")
    filled = np.array([0.0 if v is None else v for v in values], dtype=np.float64)
    valid = np.array([v is not None for v in values], dtype=bool)
    count = int(valid.sum())
    assert same(s.sum(), float(np.sum(filled)))
    assert s.count() == count
    if count == 0:
        assert s.mean() is NA and s.min() is NA and s.var() is NA
        return
    mean = np.sum(filled) / count
    assert same(s.mean(), float(mean))
    assert same(s.min(), float(np.min(filled[valid])))
    assert same(s.max(), float(np.max(filled[valid])))
    for ddof in (0, 1):
        if count - ddof <= 0:
            assert s.var(ddof=ddof) is NA
            continue
        sqr = np.where(valid, (filled - mean) ** 2, 0.0)  # pandas' nanvar, step by step
        var = np.sum(sqr) / (count - ddof)
        assert same(s.var(ddof=ddof), float(var))
        assert same(s.std(ddof=ddof), float(np.sqrt(var)))
    if count == n:  # no NA: pandas itself is the oracle
        p = pd.Series(values, dtype="float64")
        assert same(s.sum(), float(p.sum())) and same(s.mean(), float(p.mean()))
        if n > 1:
            assert same(s.var(), float(p.var())) and same(s.std(), float(p.std()))


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(st.integers(-(2**63), 2**63 - 1) | st.none(), max_size=300))
def test_int_reductions(values):
    s = sdpd.Series(values, dtype="int64")
    present = [v for v in values if v is not None]
    wrapped = (sum(present) + 2**63) % 2**64 - 2**63  # int64 sum wraps (API.md §3)
    assert s.sum() == wrapped and type(s.sum()) is int
    if not present:
        assert s.mean() is NA and s.min() is NA and s.max() is NA
        return
    assert same(s.mean(), float(sum(present)) / len(present))  # exact sum, one rounding, then divide
    assert s.min() == min(present) and s.max() == max(present)
    mean = float(sum(present)) / len(present)
    filled = np.array([float(v) if v is not None else 0.0 for v in values])
    sqr = np.where([v is not None for v in values], (filled - mean) ** 2, 0.0)
    if len(present) > 1:
        assert same(s.var(), float(np.sum(sqr) / (len(present) - 1)))


def test_nan_propagates_whatever_skipna_says():
    s = sdpd.Series([1.0, math.nan, None])
    for op in ("sum", "mean", "min", "max", "std", "var"):
        assert math.isnan(getattr(s, op)())
        assert getattr(s, op)(skipna=False) is NA  # NA first: skipna=False makes any NA the answer
    assert s.count() == 2


def test_signed_zeros():
    for values in ([0.0, -0.0], [-0.0, 0.0], [-0.0] * 9):
        s = sdpd.Series(values)
        assert same(s.min(), float(np.min(values))) and same(s.max(), float(np.max(values)))
        assert same(s.sum(), float(np.sum(values)))


def test_all_na_and_empty():
    for s in (sdpd.Series([None, None], dtype="int64"), sdpd.Series([], dtype="float64")):
        assert s.sum() == 0 and s.count() == 0
        assert s.mean() is NA and s.min() is NA and s.max() is NA and s.std() is NA and s.var() is NA
    assert sdpd.Series([5.0]).var() is NA and sdpd.Series([5.0]).var(ddof=0) == 0.0  # pandas gives NaN for the first
    assert sdpd.Series([1, None]).sum(skipna=False) is NA


def test_bool_reductions():
    b = sdpd.Series([True, False, True, None])
    assert b.sum() == 2 and type(b.sum()) is int
    assert b.mean() == 2 / 3 and b.min() is False and b.max() is True and b.count() == 3
    for op in ("std", "var"):
        with pytest.raises(TypeError):
            getattr(b, op)()


def test_string_reductions():
    s = sdpd.Series(["b", "é", None, "a", "B"])
    assert s.min() == min(["b", "é", "a", "B"]) and s.max() == "é" and s.count() == 4
    for op in ("sum", "mean", "std", "var"):
        with pytest.raises(TypeError):
            getattr(s, op)()
    assert sdpd.Series([None], dtype="string").min() is NA


def test_arguments():
    s = sdpd.Series([1.0, 2.0])
    for bad in (-1, 1.5, True):
        with pytest.raises(ValueError):
            s.var(ddof=bad)
    with pytest.raises(TypeError):
        s.sum(True)  # keyword-only (API.md §11 C2)
    with pytest.raises(TypeError):
        s.count(skipna=True)  # count has no arguments (C3)


def test_frame_reductions_are_one_row():
    df = sdpd.DataFrame({"a": [1, 2, None], "b": [0.5, 1.5, 2.5], "t": [True, False, True]})
    out = df.sum()
    assert isinstance(out, sdpd.DataFrame) and out.shape == (1, 3) and out.index.to_list() == ["sum"]
    assert out.loc["sum", "a"] == 3 and out.loc["sum", "t"] == 2
    assert out.dtypes.to_list() == ["int64", "float64", "int64"]
    m = df.mean()
    assert m.dtypes.to_list() == ["float64"] * 3 and m.loc["mean", "a"] == 1.5
    assert df.count().loc["count", "a"] == 2
    assert df.max().dtypes.to_list() == ["int64", "float64", "bool"]
    with pytest.raises(TypeError):
        df.std()  # the bool column has no std
    mixed = sdpd.DataFrame({"s": ["x", "y"], "n": [1, 2]})
    assert mixed.min().loc["min", "s"] == "x"
    with pytest.raises(TypeError):
        mixed.sum()
    assert repr(df[["a", "b"]].sum()) == "     a    b\nsum  3  4.5"
