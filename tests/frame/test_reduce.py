"""Reductions (docs/api/frame.md §6).

Float sums are numpy's pairwise summation, so numpy run over the same values
(NA rows hold 0.0, docs/design/frame.md §4.5) is a bit-for-bit oracle; without NA,
pandas is too. int64 means are the array area's float64 sum over the count
(DESIGN.md §9.1 #11), so sd.sum(..., dtype=sd.float64) is their oracle.
"""

import math
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

NA = sd.frame.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))


def bits(x):
    return np.float64(x).tobytes()


def same(x, y):
    if x is NA or y is NA:
        return x is y
    if isinstance(x, float) or isinstance(y, float):
        return (
            isinstance(x, float) and isinstance(y, float) and (bits(x) == bits(y) or (math.isnan(x) and math.isnan(y)))
        )
    return x == y and type(x) is type(y)


FLOATS = st.floats(allow_nan=False, allow_infinity=False, min_value=-1e12, max_value=1e12) | st.sampled_from(
    [0.0, -0.0, 1e-300]
)
LENGTHS = st.sampled_from([0, 1, 7, 8, 9, 127, 128, 129, 300, 1000, 5000]) | st.integers(0, 64)


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_float_reductions_match_numpy_bitwise(data):
    n = data.draw(LENGTHS)
    values = data.draw(st.lists(FLOATS | st.none(), min_size=n, max_size=n))
    s = sd.frame.Series(values, dtype="float64")
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
    for correction in (0, 1):
        if count - correction <= 0:
            assert s.var(correction=correction) is NA
            continue
        sqr = np.where(valid, (filled - mean) ** 2, 0.0)  # pandas' nanvar, step by step
        var = np.sum(sqr) / (count - correction)
        assert same(s.var(correction=correction), float(var))
        assert same(s.std(correction=correction), float(np.sqrt(var)))
    if count == n:  # no NA: pandas itself is the oracle
        p = pd.Series(values, dtype="float64")
        assert same(s.sum(), float(p.sum())) and same(s.mean(), float(p.mean()))
        if n > 1:
            assert same(s.var(correction=1), float(p.var())) and same(s.std(correction=1), float(p.std()))
            assert same(s.var(), float(p.var(ddof=0)))  # the default is correction=0, as in the array area


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(st.integers(-(2**63), 2**63 - 1) | st.none(), max_size=300))
def test_int_reductions(values):
    s = sd.frame.Series(values, dtype="int64")
    present = [v for v in values if v is not None]
    wrapped = (sum(present) + 2**63) % 2**64 - 2**63  # int64 sum wraps (docs/api/frame.md §1.5)
    assert s.sum() == wrapped and type(s.sum()) is int
    if not present:
        assert s.mean() is NA and s.min() is NA and s.max() is NA
        return
    # the array area's float64 sum of the column (NA rows hold 0), then divide (DESIGN.md §9.1 #11)
    filled_i = [0 if v is None else v for v in values]
    mean = float(sd.sum(sd.asarray(filled_i), dtype=sd.float64)) / len(present)
    assert same(s.mean(), mean)
    if len(present) == len(values):  # short and without NA: numpy's mean too
        assert same(s.mean(), float(np.mean(np.array(values, dtype=np.int64))))
    assert s.min() == min(present) and s.max() == max(present)
    filled = np.array([float(v) if v is not None else 0.0 for v in values])
    sqr = np.where([v is not None for v in values], (filled - mean) ** 2, 0.0)
    assert same(s.var(), float(np.sum(sqr) / len(present)))
    if len(present) > 1:
        assert same(s.var(correction=1), float(np.sum(sqr) / (len(present) - 1)))


def test_nan_propagates_whatever_skipna_says():
    s = sd.frame.Series([1.0, math.nan, None])
    for op in ("sum", "mean", "min", "max", "std", "var"):
        assert math.isnan(getattr(s, op)())
        assert getattr(s, op)(skipna=False) is NA  # NA first: skipna=False makes any NA the answer
    assert s.count() == 2


def test_signed_zeros():
    for values in ([0.0, -0.0], [-0.0, 0.0], [-0.0] * 9):
        s = sd.frame.Series(values)
        assert same(s.min(), float(np.min(values))) and same(s.max(), float(np.max(values)))
        assert same(s.sum(), float(np.sum(values)))


def test_all_na_and_empty():
    for s in (sd.frame.Series([None, None], dtype="int64"), sd.frame.Series([], dtype="float64")):
        assert s.sum() == 0 and s.count() == 0
        assert s.mean() is NA and s.min() is NA and s.max() is NA and s.std() is NA and s.var() is NA
    assert sd.frame.Series([5.0]).var(correction=1) is NA and sd.frame.Series([5.0]).var() == 0.0
    assert sd.frame.Series([1, None]).sum(skipna=False) is NA


def test_bool_reductions():
    b = sd.frame.Series([True, False, True, None])
    assert b.sum() == 2 and type(b.sum()) is int
    assert b.mean() == 2 / 3 and b.min() is False and b.max() is True and b.count() == 3
    # std and var of 0 and 1, as sd.std on bool (DESIGN.md §10.1 #2)
    assert b.var() == sd.var(sd.asarray([True, False, True]))
    assert b.std(correction=1) == sd.std(sd.asarray([True, False, True]), correction=1)


@settings(max_examples=60, deadline=None)
@given(st.lists(st.booleans(), min_size=1, max_size=3000), st.sampled_from([0, 1, 0.5]))
def test_bool_std_var_have_the_array_bits(values, correction):
    s, x = sd.frame.Series(values), sd.asarray(values)
    for op in ("var", "std"):
        ours = getattr(s, op)(correction=correction)
        theirs = getattr(sd, op)(x, correction=correction)
        assert ours is NA if len(values) <= correction else ours == theirs or (math.isnan(ours) and math.isnan(theirs))
    keys = [i % 3 for i in range(len(values))]
    g = sd.frame.DataFrame({"k": keys, "b": values}).groupby("k").var()
    assert g["b"].dtype is sd.float64 and len(g) == len(set(keys))


def test_string_reductions():
    s = sd.frame.Series(["b", "é", None, "a", "B"])
    assert s.min() == min(["b", "é", "a", "B"]) and s.max() == "é" and s.count() == 4
    for op in ("sum", "mean", "std", "var"):
        with pytest.raises(TypeError):
            getattr(s, op)()
    assert sd.frame.Series([None], dtype="string").min() is NA


def test_arguments():
    s = sd.frame.Series([1.0, 2.0])
    for bad in (True, "1", None):  # an int or a float, as in the array area
        with pytest.raises(TypeError):
            s.var(correction=bad)
    for op in ("sum", "mean", "min", "max", "std", "var"):  # bool options take True or False (API.md §2)
        for bad in (1, 0, None, "yes"):
            with pytest.raises(TypeError, match=rf"^{op}\(\): 'skipna' must be True or False, not "):
                getattr(s, op)(skipna=bad)
    assert s.var(correction=0.5) == 0.5 / 1.5
    with pytest.raises(TypeError):
        s.sum(True)  # keyword-only (docs/api/frame.md §1.1)
    with pytest.raises(TypeError):
        s.count(skipna=True)  # count has no arguments (docs/api/frame.md §6.1)


def test_frame_reductions_are_one_row():
    df = sd.frame.DataFrame({"a": [1, 2, None], "b": [0.5, 1.5, 2.5], "t": [True, False, True]})
    out = df.sum()
    assert isinstance(out, sd.frame.DataFrame) and out.shape == (1, 3) and out.index.tolist() == ["sum"]
    assert out.loc["sum", "a"] == 3 and out.loc["sum", "t"] == 2
    assert [str(d) for d in out.dtypes.values()] == ["int64", "float64", "int64"]
    m = df.mean()
    assert [str(d) for d in m.dtypes.values()] == ["float64"] * 3 and m.loc["mean", "a"] == 1.5
    assert df.count().loc["count", "a"] == 2
    assert [str(d) for d in df.max().dtypes.values()] == ["int64", "float64", "bool"]
    assert df.std().loc["std", "t"] == sd.std(sd.asarray([True, False, True]))  # bool has a std now
    mixed = sd.frame.DataFrame({"s": ["x", "y"], "n": [1, 2]})
    assert mixed.min().loc["min", "s"] == "x"
    with pytest.raises(TypeError):
        mixed.sum()
    assert repr(df[["a", "b"]].sum()) == "     a    b\nsum  3  4.5"


@pytest.mark.parametrize("op", ["sum", "mean", "min", "max", "std", "var", "count"])
def test_unexpected_arguments_name_the_public_class(op):
    # Python's own message, as for every other method: the class is the
    # public one, not the mixin the reductions are written in.
    s = sd.frame.Series([1.0, 2.0])
    df = sd.frame.DataFrame({"k": [1, 1], "v": [1.0, 2.0]})
    for obj, cls in ((s, "Series"), (df, "DataFrame"), (df.groupby("k"), "GroupBy")):
        for bad in ("axis", "ddof", "numeric_only"):
            with pytest.raises(TypeError, match=rf"^{cls}\.{op}\(\) got an unexpected keyword argument '{bad}'$"):
                getattr(obj, op)(**{bad: 1})
        with pytest.raises(TypeError, match=rf"^{cls}\.{op}\(\) takes 1 positional argument but 2 were given$"):
            getattr(obj, op)(True)
        method = getattr(type(obj), op)
        assert method.__qualname__ == f"{cls}.{op}" and method.__name__ == op
    assert sd.frame.Series.sum is not sd.frame.DataFrame.sum
