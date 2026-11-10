"""M2: element-wise operators, astype, NA handling and mask filters (API.md §12).

Oracles: numpy for int64/float64 arithmetic (pandas runs numpy underneath),
Python itself for comparisons (Python compares int with float exactly, as
API.md §12 asks), pandas' nullable boolean for Kleene logic, and Python's
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

import stardust_pandas as sdpd

NA = sdpd.NA
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
    """One element through numpy, plus our NA rules for ints (API.md §3, §12)."""
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
    a, b = sdpd.Series(xs, dtype=dtx), sdpd.Series(ys, dtype=dty)
    expected = [numpy_oracle(op, x, y, dtx, dty) for x, y in zip(xs, ys)]
    assert_values(ARITH[op](a, b), expected)
    # Scalar on either side gives the same as a repeated column.
    if n and ys[0] is not None:
        assert_values(ARITH[op](a, ys[0]), [numpy_oracle(op, x, ys[0], dtx, dty) for x in xs])
        assert_values(ARITH[op](ys[0], a), [numpy_oracle(op, ys[0], x, dty, dtx) for x in xs])


def test_result_dtypes():
    i, f = sdpd.Series([1, 2]), sdpd.Series([1.5, 2.5])
    assert (i + i).dtype == "int64" and (i / i).dtype == "float64" and (i + f).dtype == "float64"
    assert (i // i).dtype == "int64" and (i ** i).dtype == "int64" and (i + 1.5).dtype == "float64"


def test_int_edge_cases():
    m = -(2**63)
    s = sdpd.Series([m, 7, -7, 0])
    assert (s // -1).to_list() == [m, -7, 7, 0]  # wraps (API.md §12)
    assert (s % -1).to_list() == [0, 0, 0, 0]
    assert abs(s).to_list() == [m, 7, 7, 0]
    assert (s // 0).to_list() == [NA] * 4 and (s % 0).to_list() == [NA] * 4
    assert (sdpd.Series([2, -1, 0]) ** -1).to_list() == [NA, NA, NA]
    assert (sdpd.Series([0]) ** 0).to_list() == [1]
    assert (sdpd.Series([2**62]) * 4).to_list() == [0]
    assert (sdpd.Series([7, -7]) // 2).to_list() == [3, -4] and (sdpd.Series([7, -7]) % 2).to_list() == [1, 1]


def test_na_propagates_without_exceptions():
    s = sdpd.Series([1, None])
    assert (s ** 0).to_list() == [1, NA]  # pandas gives 1 for NA ** 0 (API.md §12)
    assert (1 ** s).to_list() == [1, NA]
    assert (s + NA).to_list() == [NA, NA] and (s + NA).dtype == "int64"
    assert (s / NA).dtype == "float64"


def test_strings_concatenate():
    s = sdpd.Series(["a", None, "é"])
    assert (s + "!").to_list() == ["a!", NA, "é!"]
    assert ("<" + s).to_list() == ["<a", NA, "<é"]
    assert (s + s).to_list() == ["aa", NA, "éé"]


@pytest.mark.parametrize(
    "expr",
    [
        lambda: sdpd.Series([True]) + sdpd.Series([True]),
        lambda: sdpd.Series([True]) + 1,
        lambda: sdpd.Series([1]) + True,
        lambda: sdpd.Series(["a"]) * 2,
        lambda: sdpd.Series(["a"]) - "a",
        lambda: sdpd.Series(["a"]) + 1,
        lambda: -sdpd.Series([True]),
        lambda: -sdpd.Series(["a"]),
        lambda: +sdpd.Series(["a"]),
        lambda: ~sdpd.Series([1]),
        lambda: sdpd.Series([1]) & sdpd.Series([1]),
        lambda: sdpd.Series([1]) + [1],
        lambda: sdpd.Series([1]) + 2**63,
    ],
)
def test_unsupported_operations(expr):
    with pytest.raises((TypeError, OverflowError)):
        expr()


def test_numpy_does_not_hijack_operators():
    with pytest.raises(TypeError):
        np.array([1]) + sdpd.Series([1])


def test_names_follow_pandas():
    a, b = sdpd.Series([1], name="x"), sdpd.Series([1], name="x")
    assert (a + b).name == "x" and (a + sdpd.Series([1], name="y")).name is None and (a * 2).name == "x"


def test_operands_need_equal_indexes_until_m4():
    s = sdpd.Series([1, 2, 3])
    with pytest.raises(NotImplementedError):
        s.tail(2) + s.head(2)
    with pytest.raises(ValueError):  # equal indexes, so lengths match; this is a different length
        sdpd._core.binary_op("add", s._col, s.head(2)._col, 0)


# --- comparison ---------------------------------------------------------------

COMPARE = {"==": operator.eq, "!=": operator.ne, "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
NEAR_2_53 = st.sampled_from([2**53, 2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63), 0]) | I64


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_numeric_comparison_is_exact(data):
    op = data.draw(st.sampled_from(list(COMPARE)))
    dtx, dty = data.draw(st.sampled_from(["int64", "float64"])), data.draw(st.sampled_from(["int64", "float64"]))
    n = data.draw(st.integers(0, 20))
    floats = FLOATS | NEAR_2_53.map(float) | st.sampled_from([2.0**63, -(2.0**63), 9007199254740993.0, 0.5, -0.5])
    elems = {"int64": NEAR_2_53, "float64": floats}
    xs = data.draw(st.lists(with_na(elems[dtx]), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(elems[dty]), min_size=n, max_size=n))
    got = COMPARE[op](sdpd.Series(xs, dtype=dtx), sdpd.Series(ys, dtype=dty))
    # Python compares int with float exactly; the values here are what the columns hold.
    xs_v = [None if x is None else (float(x) if dtx == "float64" else x) for x in xs]
    ys_v = [None if y is None else (float(y) if dty == "float64" else y) for y in ys]
    expected = [NA if x is None or y is None else COMPARE[op](x, y) for x, y in zip(xs_v, ys_v)]
    assert got.dtype == "bool" and got.to_list() == expected


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_string_and_bool_comparison(data):
    op = data.draw(st.sampled_from(list(COMPARE)))
    dtype = data.draw(st.sampled_from(["string", "bool"]))
    elems = st.text(max_size=5) if dtype == "string" else st.booleans()
    n = data.draw(st.integers(0, 20))
    xs = data.draw(st.lists(with_na(elems), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(elems), min_size=n, max_size=n))
    got = COMPARE[op](sdpd.Series(xs, dtype=dtype), sdpd.Series(ys, dtype=dtype))
    assert got.to_list() == [NA if x is None or y is None else COMPARE[op](x, y) for x, y in zip(xs, ys)]


def test_comparison_examples():
    assert (sdpd.Series([2**53 + 1]) == 2.0**53).to_list() == [False]  # pandas says True
    assert (sdpd.Series([math.nan]) != sdpd.Series([math.nan])).to_list() == [True]
    assert (1 < sdpd.Series([0, 2])).to_list() == [False, True]
    assert (sdpd.Series([1, None]) == NA).to_list() == [NA, NA]


@pytest.mark.parametrize("pair", [(["a"], 1), (["a"], True), ([True], 1), ([True], 1.5), ([1], "1")])
def test_incomparable_dtypes_raise_even_for_equality(pair):
    values, other = pair
    for op in COMPARE.values():
        with pytest.raises(TypeError):
            op(sdpd.Series(values), other)


# --- logic --------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_kleene_logic_matches_pandas_boolean(data):
    n = data.draw(st.integers(0, 30))
    xs = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    a, b = sdpd.Series(xs, dtype="bool"), sdpd.Series(ys, dtype="bool")
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
    assert sdpd.Series(values, dtype="float64").astype("string").to_list() == [NA if v is None else str(v) for v in values]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(I64)))
def test_int_to_string_and_back(values):
    s = sdpd.Series(values, dtype="int64").astype("string")
    assert s.to_list() == [NA if v is None else str(v) for v in values]
    assert s.astype("int64").to_list() == [NA if v is None else v for v in values]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(st.floats()))
def test_float_string_round_trip(values):
    back = sdpd.Series(values, dtype="float64").astype("string").astype("float64").to_list()
    assert all(same(v, b) for v, b in zip(values, back))


GOOD_FLOATS = st.from_regex(r"\A[+-]?([0-9]{1,5}(\.[0-9]{0,4})?|\.[0-9]{1,4})([eE][+-]?[0-9]{1,3})?\Z") | st.sampled_from(
    ["nan", "-NaN", "inf", "+Infinity", "-iNf", "1e400", "-1e400", "1e-400", "4.9e-324", "2.5e-310"]
)


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(GOOD_FLOATS))
def test_string_to_float_matches_python_float(values):
    got = sdpd.Series(values, dtype="string").astype("float64").to_list()
    assert all(same(g, float(v)) for g, v in zip(got, values))


@pytest.mark.parametrize("text", [" 1", "1 ", "1_000", "", "+", "1e", "e5", ".", "0x10", "nan(1)", "١", "1.5.2", "--1"])
def test_strict_float_parsing_rejects(text):
    with pytest.raises(ValueError):
        sdpd.Series([text]).astype("float64")


@pytest.mark.parametrize("text", [" 1", "1_0", "", "+", "1.0", "1e3", "٣", "9223372036854775808", "0x1"])
def test_strict_int_parsing_rejects(text):
    with pytest.raises(ValueError):
        sdpd.Series([text]).astype("int64")


def test_string_to_bool():
    assert sdpd.Series(["True", "False", None]).astype("bool").to_list() == [True, False, NA]
    for text in ("true", "1", "", "False "):
        with pytest.raises(ValueError):
            sdpd.Series([text]).astype("bool")


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(st.floats(min_value=-(2.0**63), max_value=2.0**63, exclude_max=True))))
def test_float_to_int_truncates(values):
    got = sdpd.Series(values, dtype="float64").astype("int64").to_list()
    assert got == [NA if v is None else math.trunc(v) for v in values]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, 2.0**63, -(2.0**63) * 2])
def test_float_to_int_rejects(value):
    with pytest.raises(ValueError):
        sdpd.Series([value]).astype("int64")


def test_bool_and_number_conversions():
    assert sdpd.Series([True, False, None]).astype("int64").to_list() == [1, 0, NA]
    assert sdpd.Series([True, None]).astype("float64").to_list() == [1.0, NA]
    assert sdpd.Series([0, 5, None]).astype("bool").to_list() == [False, True, NA]
    assert sdpd.Series([0.0, -0.0, math.nan]).astype("bool").to_list() == [False, False, True]
    assert sdpd.Series([True, None]).astype("string").to_list() == ["True", NA]
    assert sdpd.Series([2**53 + 1]).astype("float64").to_list() == [float(2**53)]  # astype rounds


# --- NA handling --------------------------------------------------------------


def test_isna_notna():
    s = sdpd.Series([1.0, math.nan, None])
    assert s.isna().to_list() == [False, False, True]  # NaN is a value (API.md §1)
    assert s.notna().to_list() == [True, True, False]


def test_fillna():
    assert sdpd.Series([1, None]).fillna(2.0).to_list() == [1, 2]
    assert sdpd.Series([1.5, None]).fillna(0).to_list() == [1.5, 0.0]
    assert sdpd.Series(["a", None]).fillna("z").to_list() == ["a", "z"]
    assert sdpd.Series([True, None]).fillna(False).to_list() == [True, False]
    assert sdpd.Series([1, None]).fillna(NA).to_list() == [1, NA]
    with pytest.raises(ValueError):
        sdpd.Series([1, None]).fillna(1.5)
    with pytest.raises(TypeError):
        sdpd.Series(["a", None]).fillna(1)
    with pytest.raises(TypeError):
        sdpd.Series([1, None]).fillna(True)
    df = sdpd.DataFrame({"a": [1, None], "b": ["x", None]})
    assert df.fillna({"a": 0})["b"].to_list() == ["x", NA] and df.fillna({"a": 0})["a"].to_list() == [1, 0]
    with pytest.raises(KeyError):
        df.fillna({"zz": 0})


def test_dropna():
    s = sdpd.Series([1, None, 3], name="v")
    d = s.dropna()
    assert d.to_list() == [1, 3] and d.index.to_list() == [0, 2] and d.name == "v"
    assert s.dropna(ignore_index=True).index.equals(sdpd.RangeIndex(2))
    df = sdpd.DataFrame({"a": [1, None, None], "b": ["x", "y", None]})
    assert df.dropna().index.to_list() == [0]
    assert df.dropna(how="all").index.to_list() == [0, 1]
    assert df.dropna(subset="b").index.to_list() == [0, 1]
    assert df.dropna(subset=[]).index.to_list() == [0, 1, 2]
    with pytest.raises(ValueError):
        df.dropna(how="some")
    with pytest.raises(KeyError):
        df.dropna(subset=["zz"])
    with pytest.raises(TypeError):
        df.dropna("any")  # keyword-only (API.md §11 C2)


# --- mask filters and Index ---------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_mask_filter(data):
    n = data.draw(st.integers(0, 40))
    values = data.draw(st.lists(with_na(st.text(max_size=3)), min_size=n, max_size=n))
    mask = data.draw(st.lists(with_na(st.booleans()), min_size=n, max_size=n))
    s = sdpd.Series(values, dtype="string").tail(n)
    m = sdpd.Series(mask, dtype="bool").tail(n)
    kept = [i for i, k in enumerate(mask) if k]  # NA counts as False (API.md §2)
    out = s[m]
    assert out.to_list() == [NA if values[i] is None else values[i] for i in kept]
    assert out.index.to_list() == kept and isinstance(out.index, sdpd.Index)


def test_filter_keeps_labels_of_a_sliced_range():
    s = sdpd.Series(list(range(10))).tail(5)
    assert s[s % 2 == 0].index.to_list() == [6, 8]
    f = s[s > 0]
    assert f[f > 7].index.to_list() == [8, 9]  # filtering an Index takes its labels


def test_frame_filter():
    df = sdpd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = df[df["a"] >= 2]
    assert out["b"].to_list() == ["y", "z"] and out.index.to_list() == [1, 2]
    with pytest.raises(TypeError):
        df[df["a"]]
    with pytest.raises(NotImplementedError):
        df[df["a"].tail(2) > 0]


def test_index_basics():
    idx = sdpd.Index([3, 1, 2])
    assert idx.to_list() == [3, 1, 2] and idx.dtype == "int64" and len(idx) == 3
    assert sdpd.Index([0, 1]).equals(sdpd.RangeIndex(2)) and sdpd.RangeIndex(2).equals(sdpd.Index([0, 1]))
    with pytest.raises(ValueError):
        sdpd.Index([1, None])


@pytest.mark.parametrize("values", [[], [7], [0, 2, 5], list(range(30)), [x * 1000 for x in range(30)],
                                     list(range(0, 400, 2)), list(range(101)), list(range(100)), [-5, 10**15, 3]])
def test_index_repr_matches_pandas(values):
    assert repr(sdpd.Index(values)) == repr(pd.Index(np.array(values, dtype=np.int64)))


@pytest.mark.parametrize("n", [5, 70])
def test_filtered_repr_matches_pandas(n):
    s = sdpd.Series([float(i) / 3 for i in range(n)], name="x")
    p = pd.Series([float(i) / 3 for i in range(n)], name="x")
    assert repr(s[s * 3 % 2 == 0]) == repr(p[p * 3 % 2 == 0])
    df, pdf = sdpd.DataFrame({"x": list(range(n))}), pd.DataFrame({"x": range(n)})
    assert repr(df[df["x"] % 3 == 1]) == pdf[pdf["x"] % 3 == 1].to_string(max_rows=60, min_rows=10, show_dimensions="truncate")


# --- DataFrame operators ------------------------------------------------------


def test_frame_operators():
    df = sdpd.DataFrame({"a": [1, 2], "b": [0.5, None]})
    assert (df + 1)["a"].to_list() == [2, 3] and (2 * df)["b"].to_list() == [1.0, NA]
    other = sdpd.DataFrame({"b": [1.0, 1.0], "a": [10, 20]})
    s = df + other
    assert s.columns == ["a", "b"] and s["a"].to_list() == [11, 22]  # left's column order
    assert (df > 1)["a"].to_list() == [False, True]
    assert (-df)["a"].to_list() == [-1, -2] and abs(-df)["a"].to_list() == [1, 2]
    assert df.isna()["b"].to_list() == [False, True]
    assert df.astype({"a": "float64"}).dtypes == {"a": "float64", "b": "float64"}
    assert df.astype("string")["b"].to_list() == ["0.5", NA]
    with pytest.raises(TypeError):
        df + df["a"]
    with pytest.raises(NotImplementedError):
        df + sdpd.DataFrame({"a": [1, 2]})
    with pytest.raises(KeyError):
        df.astype({"zz": "int64"})
