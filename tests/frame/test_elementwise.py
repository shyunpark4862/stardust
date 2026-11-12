"""Element-wise operators, astype, NA handling and mask filters (docs/api/frame.md §3, §5.5).

Oracles: numpy for int64/float64 arithmetic (pandas runs numpy underneath),
Python itself for comparisons (Python compares int with float exactly, as
docs/api/frame.md §3.3 asks), pandas' nullable boolean for Kleene logic, and Python's
str()/float() for string conversions.
"""

import math
import operator
import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

NA = sd.frame.NA
EXAMPLES = int(os.environ.get("SDPD_EXAMPLES", 300))

I64 = st.integers(-(2**63), 2**63 - 1)
SMALL = st.integers(-20, 20)
FLOATS = st.floats() | st.sampled_from([0.0, -0.0, math.inf, -math.inf, math.nan, 1e308, 5e-324])


def with_na(elems):
    return elems | st.just(None)


def same(x, y):
    """Bitwise float equality (NaN == NaN, -0.0 != 0.0), plain == otherwise."""
    if isinstance(x, float) and isinstance(y, float):
        return np.float64(x).tobytes() == np.float64(y).tobytes() or (math.isnan(x) and math.isnan(y))
    return x == y and type(x) is type(y)


def assert_values(series, expected):
    got = series.to_list()
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert (g is NA and e is NA) or same(g, e), (got, expected)


# --- arithmetic ---------------------------------------------------------------

ARITH = {"+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
         "//": operator.floordiv, "%": operator.mod, "**": operator.pow}
NP_ARITH = {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.true_divide,
            "//": np.floor_divide, "%": np.remainder, "**": np.power}


def numpy_oracle(op, x, y, dtx, dty):
    """One element through numpy, plus our NA rules for ints (docs/api/frame.md §1.5, §3.2)."""
    if x is None or y is None:
        return NA
    ints = dtx == dty == "int64"
    if ints and op in ("//", "%") and y == 0:
        return NA
    if ints and op == "**" and y < 0:
        return NA
    with np.errstate(all="ignore"):
        r = NP_ARITH[op](np.array([x], dtype=dtx), np.array([y], dtype=dty))[0]
    return r.item()


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_arithmetic_matches_numpy(data):
    op = data.draw(st.sampled_from(list(ARITH)))
    dtx, dty = data.draw(st.sampled_from(["int64", "float64"])), data.draw(st.sampled_from(["int64", "float64"]))
    n = data.draw(st.integers(0, 20))
    def elems(dt):
        if dt == "float64":
            return FLOATS
        return SMALL if op == "**" else (I64 | SMALL)
    xs = data.draw(st.lists(with_na(elems(dtx)), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(elems(dty)), min_size=n, max_size=n))
    a, b = sd.frame.Series(xs, dtype=dtx), sd.frame.Series(ys, dtype=dty)
    expected = [numpy_oracle(op, x, y, dtx, dty) for x, y in zip(xs, ys)]
    assert_values(ARITH[op](a, b), expected)
    # Scalar on either side gives the same as a repeated column.
    if n and ys[0] is not None:
        assert_values(ARITH[op](a, ys[0]), [numpy_oracle(op, x, ys[0], dtx, dty) for x in xs])
        assert_values(ARITH[op](ys[0], a), [numpy_oracle(op, ys[0], x, dty, dtx) for x in xs])


def test_result_dtypes():
    i, f = sd.frame.Series([1, 2]), sd.frame.Series([1.5, 2.5])
    assert (i + i).dtype is sd.int64 and (i / i).dtype is sd.float64 and (i + f).dtype is sd.float64
    assert (i // i).dtype is sd.int64 and (i ** i).dtype is sd.int64 and (i + 1.5).dtype is sd.float64


def test_int_edge_cases():
    m = -(2**63)
    s = sd.frame.Series([m, 7, -7, 0])
    assert (s // -1).to_list() == [m, -7, 7, 0]  # wraps (docs/api/frame.md §1.5)
    assert (s % -1).to_list() == [0, 0, 0, 0]
    assert abs(s).to_list() == [m, 7, 7, 0]
    assert (s // 0).to_list() == [NA] * 4 and (s % 0).to_list() == [NA] * 4
    assert (sd.frame.Series([2, -1, 0]) ** -1).to_list() == [NA, NA, NA]
    assert (sd.frame.Series([0]) ** 0).to_list() == [1]
    assert (sd.frame.Series([2**62]) * 4).to_list() == [0]
    assert (sd.frame.Series([7, -7]) // 2).to_list() == [3, -4] and (sd.frame.Series([7, -7]) % 2).to_list() == [1, 1]


def test_na_propagates_without_exceptions():
    s = sd.frame.Series([1, None])
    assert (s ** 0).to_list() == [1, NA]  # pandas gives 1 for NA ** 0 (docs/api/frame.md §3.2)
    assert (1 ** s).to_list() == [1, NA]
    assert (s + NA).to_list() == [NA, NA] and (s + NA).dtype is sd.int64
    assert (s / NA).dtype is sd.float64


def test_strings_concatenate():
    s = sd.frame.Series(["a", None, "é"])
    assert (s + "!").to_list() == ["a!", NA, "é!"]
    assert ("<" + s).to_list() == ["<a", NA, "<é"]
    assert (s + s).to_list() == ["aa", NA, "éé"]


@pytest.mark.parametrize(
    "expr",
    [
        lambda: sd.frame.Series(["a"]) * 2,
        lambda: sd.frame.Series(["a"]) - "a",
        lambda: sd.frame.Series(["a"]) + 1,
        lambda: -sd.frame.Series(["a"]),
        lambda: +sd.frame.Series(["a"]),
        lambda: ~sd.frame.Series([1]),
        lambda: sd.frame.Series([1]) & sd.frame.Series([1]),
        lambda: sd.frame.Series([1]) + [1],
        lambda: sd.frame.Series([1]) + 2**63,
    ],
)
def test_unsupported_operations(expr):
    with pytest.raises((TypeError, OverflowError)):
        expr()


def test_bool_arithmetic_is_int64_arithmetic():
    # bool takes part as int64, as in the array area (docs/api/frame.md §3.2)
    t = sd.frame.Series([True, False, None])
    assert (t + t).dtype is sd.int64 and (t + t).to_list() == [2, 0, NA]
    assert (t + 1).to_list() == [2, 1, NA] and (2 * t).to_list() == [2, 0, NA]
    q = (t / t).to_list()
    assert (t + 0.5).dtype is sd.float64 and q[0] == 1.0 and math.isnan(q[1]) and q[2] is NA
    assert (t // t).to_list() == [1, NA, NA]  # the int rule: // 0 is NA
    assert (-t).to_list() == [-1, 0, NA] and abs(t).to_list() == [1, 0, NA]


def test_numpy_does_not_hijack_operators():
    with pytest.raises(TypeError):
        np.array([1]) + sd.frame.Series([1])


def test_names_follow_pandas():
    a, b = sd.frame.Series([1], name="x"), sd.frame.Series([1], name="x")
    assert (a + b).name == "x" and (a + sd.frame.Series([1], name="y")).name is None and (a * 2).name == "x"


def test_operands_with_different_indexes_align():
    s = sd.frame.Series([1, 2, 3])
    out = s.tail(2) + s.head(2)  # labels 1, 2 and 0, 1: union 1, 2, 0 (docs/api/frame.md §5.6)
    assert out.index.to_list() == [1, 2, 0] and out.to_list() == [4, NA, NA]
    with pytest.raises(ValueError):  # the kernel itself still wants equal lengths
        sd.frame._core.binary_op("add", s._col, s.head(2)._col, 0)


# --- comparison ---------------------------------------------------------------

COMPARE = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
NEAR_2_53 = st.sampled_from([2**53, 2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63), 0]) | I64


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_numeric_comparison_converts_ints_to_float(data):
    op = data.draw(st.sampled_from(list(COMPARE)))
    dtx, dty = data.draw(st.sampled_from(["int64", "float64"])), data.draw(st.sampled_from(["int64", "float64"]))
    n = data.draw(st.integers(0, 20))
    floats = FLOATS | NEAR_2_53.map(float) | st.sampled_from([2.0**63, -(2.0**63), 9007199254740993.0, 0.5, -0.5])
    elems = {"int64": NEAR_2_53, "float64": floats}
    xs = data.draw(st.lists(with_na(elems[dtx]), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(elems[dty]), min_size=n, max_size=n))
    got = COMPARE[op](sd.frame.Series(xs, dtype=dtx), sd.frame.Series(ys, dtype=dty))
    # int with float compares as float64, as in the array area, numpy and pandas (docs/api/frame.md §3.3).
    as_float = dtx != dty or dtx == "float64"
    xs_v = [None if x is None else (float(x) if as_float else x) for x in xs]
    ys_v = [None if y is None else (float(y) if as_float else y) for y in ys]
    expected = [NA if x is None or y is None else COMPARE[op](x, y) for x, y in zip(xs_v, ys_v)]
    assert got.dtype is sd.bool and got.to_list() == expected


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_string_and_bool_comparison(data):
    op = data.draw(st.sampled_from(list(COMPARE)))
    dtype = data.draw(st.sampled_from(["string", "bool"]))
    elems = st.text(max_size=5) if dtype == "string" else st.booleans()
    n = data.draw(st.integers(0, 20))
    xs = data.draw(st.lists(with_na(elems), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(elems), min_size=n, max_size=n))
    got = COMPARE[op](sd.frame.Series(xs, dtype=dtype), sd.frame.Series(ys, dtype=dtype))
    assert got.to_list() == [NA if x is None or y is None else COMPARE[op](x, y) for x, y in zip(xs, ys)]


def test_comparison_examples():
    assert (sd.frame.Series([2**53 + 1]) == 2.0**53).to_list() == [True]  # as numpy, pandas and the array area
    assert (sd.frame.Series([math.nan]) != sd.frame.Series([math.nan])).to_list() == [True]
    assert (1 < sd.frame.Series([0, 2])).to_list() == [False, True]
    assert (sd.frame.Series([1, None]) == NA).to_list() == [NA, NA]


@pytest.mark.parametrize("pair", [(["a"], 1), (["a"], True), ([True], 1), ([True], 1.5), ([1], "1")])
def test_incomparable_dtypes_raise_even_for_equality(pair):
    values, other = pair
    for op in COMPARE.values():
        with pytest.raises(TypeError):
            op(sd.frame.Series(values), other)


# --- logic --------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_kleene_logic_matches_pandas_boolean(data):
    n = data.draw(st.integers(0, 30))
    xs = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    a, b = sd.frame.Series(xs, dtype="bool"), sd.frame.Series(ys, dtype="bool")
    pa, pb = pd.array(xs, dtype="boolean"), pd.array(ys, dtype="boolean")
    to = lambda arr: [NA if v is pd.NA else bool(v) for v in arr]
    assert (a & b).to_list() == to(pa & pb)
    assert (a | b).to_list() == to(pa | pb)
    assert (a ^ b).to_list() == [NA if x is None or y is None else x != y for x, y in zip(xs, ys)]
    assert (~a).to_list() == to(~pa)
    for scalar in (True, False, NA):
        ps = pd.NA if scalar is NA else scalar
        assert (a & scalar).to_list() == to(pa & ps) and (scalar | a).to_list() == to(ps | pa)


# --- astype -------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(st.floats())))
def test_float_to_string_is_python_str(values):
    assert sd.frame.Series(values, dtype="float64").astype("string").to_list() == [NA if v is None else str(v) for v in values]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(I64)))
def test_int_to_string_and_back(values):
    s = sd.frame.Series(values, dtype="int64").astype("string")
    assert s.to_list() == [NA if v is None else str(v) for v in values]
    assert s.astype("int64").to_list() == [NA if v is None else v for v in values]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(st.floats()))
def test_float_string_round_trip(values):
    back = sd.frame.Series(values, dtype="float64").astype("string").astype("float64").to_list()
    assert all(same(v, b) for v, b in zip(values, back))


GOOD_FLOATS = st.from_regex(r"\A[+-]?([0-9]{1,5}(\.[0-9]{0,4})?|\.[0-9]{1,4})([eE][+-]?[0-9]{1,3})?\Z") | st.sampled_from(
    ["nan", "-NaN", "inf", "+Infinity", "-iNf", "1e400", "-1e400", "1e-400", "4.9e-324", "2.5e-310"]
)


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(GOOD_FLOATS))
def test_string_to_float_matches_python_float(values):
    got = sd.frame.Series(values, dtype="string").astype("float64").to_list()
    assert all(same(g, float(v)) for g, v in zip(got, values))


@pytest.mark.parametrize("text", [" 1", "1 ", "1_000", "", "+", "1e", "e5", ".", "0x10", "nan(1)", "١", "1.5.2", "--1"])
def test_strict_float_parsing_rejects(text):
    with pytest.raises(ValueError):
        sd.frame.Series([text]).astype("float64")


@pytest.mark.parametrize("text", [" 1", "1_0", "", "+", "1.0", "1e3", "٣", "9223372036854775808", "0x1"])
def test_strict_int_parsing_rejects(text):
    with pytest.raises(ValueError):
        sd.frame.Series([text]).astype("int64")


def test_string_to_bool():
    assert sd.frame.Series(["True", "False", None]).astype("bool").to_list() == [True, False, NA]
    for text in ("true", "1", "", "False "):
        with pytest.raises(ValueError):
            sd.frame.Series([text]).astype("bool")


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(st.floats(min_value=-(2.0**63), max_value=2.0**63, exclude_max=True))))
def test_float_to_int_truncates(values):
    got = sd.frame.Series(values, dtype="float64").astype("int64").to_list()
    assert got == [NA if v is None else math.trunc(v) for v in values]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(math.nan, 0), (math.inf, 2**63 - 1), (-math.inf, -(2**63)), (2.0**63, 2**63 - 1), (-(2.0**63) * 2, -(2**63))],
)
def test_float_to_int_saturates(value, expected):
    # the array area's rule (docs/api/frame.md §3.5)
    assert sd.frame.Series([value]).astype("int64").to_list() == [expected]


def test_bool_and_number_conversions():
    assert sd.frame.Series([True, False, None]).astype("int64").to_list() == [1, 0, NA]
    assert sd.frame.Series([True, None]).astype("float64").to_list() == [1.0, NA]
    assert sd.frame.Series([0, 5, None]).astype("bool").to_list() == [False, True, NA]
    assert sd.frame.Series([0.0, -0.0, math.nan]).astype("bool").to_list() == [False, False, True]
    assert sd.frame.Series([True, None]).astype("string").to_list() == ["True", NA]
    assert sd.frame.Series([2**53 + 1]).astype("float64").to_list() == [float(2**53)]  # astype rounds


# --- NA handling --------------------------------------------------------------


def test_isna_notna():
    s = sd.frame.Series([1.0, math.nan, None])
    assert s.isna().to_list() == [False, False, True]  # NaN is a value (docs/api/frame.md §1.3)
    assert s.notna().to_list() == [True, True, False]


def test_fillna():
    # the fill value converts by dtype only, as an assignment (docs/api/frame.md §3.6, §5.7)
    assert sd.frame.Series([1.5, None]).fillna(0).to_list() == [1.5, 0.0]
    assert sd.frame.Series(["a", None]).fillna("z").to_list() == ["a", "z"]
    assert sd.frame.Series([True, None]).fillna(False).to_list() == [True, False]
    assert sd.frame.Series([1, None]).fillna(NA).to_list() == [1, NA]
    assert sd.frame.Series([1, None]).fillna(True).to_list() == [1, 1]  # bool -> int64 goes up
    for value in (2.0, 1.5):  # float64 -> int64 goes down, whatever the value
        with pytest.raises(TypeError):
            sd.frame.Series([1, None]).fillna(value)
    with pytest.raises(TypeError):
        sd.frame.Series(["a", None]).fillna(1)
    df = sd.frame.DataFrame({"a": [1, None], "b": ["x", None]})
    assert df.fillna({"a": 0})["b"].to_list() == ["x", NA] and df.fillna({"a": 0})["a"].to_list() == [1, 0]
    with pytest.raises(KeyError):
        df.fillna({"zz": 0})


def test_dropna():
    s = sd.frame.Series([1, None, 3], name="v")
    d = s.dropna()
    assert d.to_list() == [1, 3] and d.index.to_list() == [0, 2] and d.name == "v"
    assert s.dropna(ignore_index=True).index.equals(sd.frame.RangeIndex(2))
    df = sd.frame.DataFrame({"a": [1, None, None], "b": ["x", "y", None]})
    assert df.dropna().index.to_list() == [0]
    assert df.dropna(how="all").index.to_list() == [0, 1]
    assert df.dropna(subset="b").index.to_list() == [0, 1]
    assert df.dropna(subset=[]).index.to_list() == [0, 1, 2]
    with pytest.raises(ValueError):
        df.dropna(how="some")
    with pytest.raises(KeyError):
        df.dropna(subset=["zz"])
    with pytest.raises(TypeError):
        df.dropna("any")  # keyword-only (docs/api/frame.md §1.1)


# --- mask filters and Index ---------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_mask_filter(data):
    n = data.draw(st.integers(0, 40))
    values = data.draw(st.lists(with_na(st.text(max_size=3)), min_size=n, max_size=n))
    mask = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    s = sd.frame.Series(values, dtype="string").tail(n)
    m = sd.frame.Series(mask, dtype="bool").tail(n)
    kept = [i for i, k in enumerate(mask) if k]  # NA counts as False (docs/api/frame.md §1.4)
    out = s[m]
    assert out.to_list() == [NA if values[i] is None else values[i] for i in kept]
    assert out.index.to_list() == kept and isinstance(out.index, sd.frame.Index)


def test_filter_keeps_labels_of_a_sliced_range():
    s = sd.frame.Series(list(range(10))).tail(5)
    assert s[s % 2 == 0].index.to_list() == [6, 8]
    f = s[s > 0]
    assert f[f > 7].index.to_list() == [8, 9]  # filtering an Index takes its labels


def test_frame_filter():
    df = sd.frame.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = df[df["a"] >= 2]
    assert out["b"].to_list() == ["y", "z"] and out.index.to_list() == [1, 2]
    with pytest.raises(TypeError):
        df[df["a"]]
    with pytest.raises(ValueError):  # the mask must hold the same labels (docs/api/frame.md §5.5)
        df[df["a"].tail(2) > 0]


def test_index_basics():
    idx = sd.frame.Index([3, 1, 2])
    assert idx.to_list() == [3, 1, 2] and idx.dtype is sd.int64 and len(idx) == 3
    assert sd.frame.Index([0, 1]).equals(sd.frame.RangeIndex(2)) and sd.frame.RangeIndex(2).equals(sd.frame.Index([0, 1]))
    with pytest.raises(ValueError):
        sd.frame.Index([1, None])


@pytest.mark.parametrize("values", [[], [7], [0, 2, 5], list(range(30)), [x * 1000 for x in range(30)],
                                     list(range(0, 400, 2)), list(range(101)), list(range(100)), [-5, 10**15, 3]])
def test_index_repr_matches_pandas(values):
    assert repr(sd.frame.Index(values)) == repr(pd.Index(np.array(values, dtype=np.int64)))


@pytest.mark.parametrize("n", [5, 70])
def test_filtered_repr_matches_pandas(n):
    s = sd.frame.Series([float(i) / 3 for i in range(n)], name="x")
    p = pd.Series([float(i) / 3 for i in range(n)], name="x")
    assert repr(s[s * 3 % 2 == 0]) == repr(p[p * 3 % 2 == 0])
    df, pdf = sd.frame.DataFrame({"x": list(range(n))}), pd.DataFrame({"x": range(n)})
    assert repr(df[df["x"] % 3 == 1]) == pdf[pdf["x"] % 3 == 1].to_string(max_rows=60, min_rows=10, show_dimensions="truncate")


# --- DataFrame operators ------------------------------------------------------


def test_frame_operators():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [0.5, None]})
    assert (df + 1)["a"].to_list() == [2, 3] and (2 * df)["b"].to_list() == [1.0, NA]
    other = sd.frame.DataFrame({"b": [1.0, 1.0], "a": [10, 20]})
    s = df + other
    assert s.columns.to_list() == ["a", "b"] and s["a"].to_list() == [11, 22]  # left's column order
    assert (df > 1)["a"].to_list() == [False, True]
    assert (-df)["a"].to_list() == [-1, -2] and abs(-df)["a"].to_list() == [1, 2]
    assert df.isna()["b"].to_list() == [False, True]
    assert [str(d) for d in df.astype({"a": "float64"}).dtypes.values()] == ["float64", "float64"]
    assert df.astype("string")["b"].to_list() == ["0.5", NA]
    with pytest.raises(TypeError):
        df + df["a"]
    partial = df + sd.frame.DataFrame({"a": [1, 2]})  # "b" is only on the left: NA there (docs/api/frame.md §5.6)
    assert partial["a"].to_list() == [2, 4] and partial["b"].to_list() == [NA, NA] and partial["b"].dtype is sd.float64
    with pytest.raises(KeyError):
        df.astype({"zz": "int64"})


# --- the Eisel-Lemire float parser (docs/design/frame.md §10.4) -----------------

HARD_FLOATS = ["0", "-0", "0.0", "1e22", "1e23", "9007199254740993", "9007199254740992.5",
               "2.2250738585072011e-308", "2.2250738585072014e-308", "4.9406564584124654e-324",
               "2.4703282292062327e-324", "2.4703282292062328e-324", "1.7976931348623157e308",
               "1.7976931348623158e308", "1.7976931348623159e308", "1e-400", "1e400", "123456789012345678",
               "1234567890123456789", "12345678901234567890", ".5", "5.", "7.2057594037927933e16",
               "0.000000000000000000000000000001", "9007199254740991.4999999999999999999"]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(
    st.integers(0, 2**64 - 1).map(lambda b: repr(np.frombuffer(np.uint64(b).tobytes(), dtype=np.float64)[0].item()))
    | st.builds(lambda m, e: f"{m}e{e}", st.integers(0, 10**19), st.integers(-345, 310))
    | st.builds(lambda a, b, e: f"{a}.{b}e{e}", st.integers(0, 10**10), st.integers(0, 10**12), st.integers(-30, 30)),
    max_size=50))
def test_float_parsing_is_correctly_rounded(texts):
    texts = [t for t in texts if "nan" not in t and "inf" not in t] + HARD_FLOATS
    got = sd.frame.Series(texts).astype("float64").to_list()
    assert all(same(g, float(t)) for g, t in zip(got, texts))
