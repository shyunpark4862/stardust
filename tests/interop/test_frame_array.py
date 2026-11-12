"""Series/DataFrame <-> sd.ndarray (docs/api/frame.md §2.8, DESIGN.md §4.3): what
is shared and what is copied, read-only views and Copy-on-Write across the
boundary, and the rules both areas now share."""

import math

import numpy as np
import pytest

import stardust as sd

S, DF, NA = sd.frame.Series, sd.frame.DataFrame, sd.frame.NA


# --- frame -> array -------------------------------------------------------------

@pytest.mark.parametrize(("values", "dtype"), [([1, 2, 3], sd.int64), ([0.5, -1.0], sd.float64), ([True, False], sd.bool)])
def test_to_array_is_a_read_only_view(values, dtype):
    a = S(values).to_array()
    assert isinstance(a, sd.ndarray) and a.dtype is dtype and a.tolist() == values
    with pytest.raises(ValueError):
        a[0] = a[1]


def test_frame_write_copies_first_and_the_view_keeps_its_values():
    s = S([1, 2, 3])
    a = s.to_array()
    s.iloc[0] = 100  # the view holds a reference to the Buffer: Copy-on-Write copies
    assert a.tolist() == [1, 2, 3] and s.tolist() == [100, 2, 3]
    df = DF({"x": [1.0, 2.0]})
    b = df["x"].to_array()
    df.loc[0, "x"] = 9.0
    assert b.tolist() == [1.0, 2.0] and df["x"].tolist() == [9.0, 2.0]


def test_to_array_copy_and_conversions():
    s = S([1, 2])
    c = s.to_array(copy=True)
    c[0] = 7  # writable, and the column does not see it
    assert c.tolist() == [7, 2] and s.tolist() == [1, 2]
    assert s.to_array(dtype="float64").dtype is sd.float64
    with pytest.raises(ValueError):
        s.to_array(dtype="float64", copy=False)
    f = S([1.0, None]).to_array(na_value=-1.0)
    assert f.tolist() == [1.0, -1.0]
    f[0] = 5.0  # a freshly filled column: nobody else holds it, so it is writable
    with pytest.raises(ValueError):
        S([1.0, None]).to_array()  # NA needs na_value=
    with pytest.raises(ValueError):
        S([1.0, None]).to_array(na_value=0.0, copy=False)
    with pytest.raises(TypeError):
        S(["a"]).to_array()  # the array area has no string dtype
    with pytest.raises(TypeError):
        s.to_array(copy="yes")


def test_slice_view_starts_at_its_row():
    s = S(list(range(10)))
    assert s.iloc[3:6].to_array().tolist() == [3, 4, 5]


def test_dataframe_to_array_is_f_order():
    df = DF({"a": [1, 2, 3], "b": [0.5, 1.5, None]})
    m = df.to_array(na_value=math.inf)
    assert m.dtype is sd.float64 and m.shape == (3, 2) and m.strides == (8, 24)
    assert m.tolist() == [[1.0, 0.5], [2.0, 1.5], [3.0, math.inf]]
    with pytest.raises(ValueError):
        df.to_array(na_value=0.0, copy=False)
    with pytest.raises(TypeError):
        DF({"a": [1], "s": ["x"]}).to_array()
    assert DF({"a": [1, 2]}).to_array().tolist() == [[1], [2]]


def test_linalg_takes_a_frame_through_to_array():
    df = DF({"x": [1.0, 2.0, 3.0], "one": [1.0, 1.0, 1.0]})
    y = S([3.0, 5.0, 7.0]).to_array()
    coef = sd.linalg.lstsq(df.to_array(), y).solution.tolist()
    assert all(math.isclose(c, e, abs_tol=1e-12) for c, e in zip(coef, [2.0, 1.0]))


# --- array -> frame -------------------------------------------------------------

def test_series_from_an_array_copies():
    x = sd.asarray([1, 2, 3])
    s = S(x)
    x[0] = 100
    assert s.tolist() == [1, 2, 3] and s.dtype is sd.int64
    assert S(sd.asarray([1.0, 2.0, 3.0, 4.0])[::2]).tolist() == [1.0, 3.0]  # any strides
    assert S(sd.asarray([True, False]), dtype="int64").tolist() == [1, 0]  # dtype= is explicit
    with pytest.raises(TypeError):
        S(sd.asarray([1j]))
    with pytest.raises(ValueError):
        S(sd.zeros((2, 2)))


def test_dataframe_from_a_2d_array_copies():
    x = sd.asarray([[1, 2], [3, 4]])
    df = DF(x, columns=["p", "q"])
    x[0, 0] = 9
    assert df["p"].tolist() == [1, 3] and df["q"].tolist() == [2, 4]
    assert DF({"a": sd.asarray([0.5, 1.5])})["a"].tolist() == [0.5, 1.5]
    with pytest.raises(ValueError):
        DF(sd.asarray([1, 2]), columns=["a"])


def test_roundtrip():
    s = S([0.1, -0.0, math.nan, 1e300])
    back = S(s.to_array())
    assert [v.hex() for v in back.tolist()] == [v.hex() for v in s.tolist()]


# --- rules the two areas share (DESIGN.md §9) -------------------------------------

def test_int_mean_has_the_array_bits():
    rng = np.random.default_rng(0)
    for n in (5, 5000, 300_000):
        v = rng.integers(-(2**62), 2**62, n)
        assert S(v).mean() == float(sd.mean(sd.asarray(v)))


def test_comparison_and_float_to_int_conversion_agree():
    i, f = [2**53 + 1, 5], [2.0**53, 5.0]
    assert (S(i) == S(f)).tolist() == sd.equal(sd.asarray(i), sd.asarray(f)).tolist()
    vals = [math.nan, math.inf, -math.inf, 1e300, -2.7]
    assert S(vals).astype("int64").tolist() == sd.astype(sd.asarray(vals), sd.int64).tolist()


def test_dtype_objects_are_shared():
    s, a = S([1.0]), sd.asarray([1.0])
    assert s.dtype is a.dtype is sd.float64
    assert S([1], dtype=sd.float64).dtype is S([1], dtype="float64").dtype
