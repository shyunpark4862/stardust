"""Element-wise operators, astype, NA handling and mask filters (docs/api/frame.md §3, §5.5).

Oracles: numpy for int64/float64 arithmetic (pandas runs numpy underneath),
Python itself for comparisons (Python compares int with float exactly, as
docs/api/frame.md §3.3 asks), pandas' nullable boolean for Kleene logic, and Python's
str()/float() for string conversions.
"""

import math
import operator

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples

NA = sd.frame.NA
EXAMPLES = examples(300)

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
    got = series.tolist()
    assert len(got) == len(expected)
    for g, e in zip(got, expected):
        assert (g is NA and e is NA) or same(g, e), (got, expected)


# --- arithmetic ---------------------------------------------------------------

ARITH = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "//": operator.floordiv,
    "%": operator.mod,
    "**": operator.pow,
}
NP_ARITH = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": np.true_divide,
    "//": np.floor_divide,
    "%": np.remainder,
    "**": np.power,
}


def numpy_oracle(op, x, y, dtx, dty):
    """One element through numpy, plus our NA rules for ints (docs/api/frame.md §1.5, §3.2)."""
    if x is None or y is None:
        return NA
    ints = dtx == dty == "int64"
    if ints and op in ("//", "%") and y == 0:
        return NA
    if ints and op == "**" and y < 0:
        return NA
    if op == "**" and not ints and float(y) in (2.0, -1.0, 0.5):
        # the exact forms of these exponents (docs/api/array.md): x*x, 1/x and sqrt with C99
        # pow's special values; numpy's one-element array exponent goes through libm pow,
        # which is not correctly rounded in about 0.1% of these cases
        xf, e = float(x), float(y)
        with np.errstate(all="ignore"):
            if e == 2.0:
                return (np.float64(xf) * np.float64(xf)).item()
            if e == -1.0:
                return (np.float64(1.0) / np.float64(xf)).item()
        if xf == 0.0 or xf == -math.inf:
            return abs(xf)
        return math.sqrt(xf) if xf >= 0 or math.isnan(xf) else math.nan
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


def test_power_is_the_array_kernel():
    # frame ** and array ** are one kernel (libm's pow with the exact forms of the exponents
    # 2, 0, 1, -1 and 0.5): the same bits for scalar and Series exponents, int and float
    # operands; NA rows stay NA
    rng = np.random.default_rng(5)
    x = np.exp(rng.uniform(-700, 700, 2000)) * rng.choice([1.0, -1.0], 2000)
    x[:9] = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, 1e-310, 1.0, -1.0]
    e = rng.uniform(-20, 20, 2000)
    e[::5] = rng.choice([2.0, 0.0, -0.0, 1.0, -1.0, 0.5], 400)
    xs, es = sd.asarray(x), sd.asarray(e)
    s, t = sd.frame.Series(x), sd.frame.Series(e)
    assert np.asarray((s**t).to_array()).tobytes() == np.asarray(xs**es).tobytes()
    for b in (2.0, 0.0, 1.0, -1.0, 0.5, 3.0, -2.5, 0.25, math.nan, math.inf):
        assert np.asarray((s**b).to_array()).tobytes() == np.asarray(xs**b).tobytes()
        assert np.asarray((b**s).to_array()).tobytes() == np.asarray(b**xs).tobytes()
    i = rng.integers(-5, 6, 2000)
    assert (
        np.asarray((sd.frame.Series(i) ** t).to_array()).tobytes()
        == np.asarray(sd.asarray(i.astype(float)) ** es).tobytes()
    )
    assert (
        np.asarray((s ** sd.frame.Series(i)).to_array()).tobytes()
        == np.asarray(xs ** sd.asarray(i.astype(float))).tobytes()
    )
    na = sd.frame.Series([2.0, None, 3.0]) ** sd.frame.Series([0.5, 2.0, None])
    assert na.tolist()[1:] == [NA, NA] and na.tolist()[0] == math.sqrt(2.0)


def test_result_dtypes():
    i, f = sd.frame.Series([1, 2]), sd.frame.Series([1.5, 2.5])
    assert (i + i).dtype is sd.int64 and (i / i).dtype is sd.float64 and (i + f).dtype is sd.float64
    assert (i // i).dtype is sd.int64 and (i**i).dtype is sd.int64 and (i + 1.5).dtype is sd.float64


def test_int_edge_cases():
    m = -(2**63)
    s = sd.frame.Series([m, 7, -7, 0])
    assert (s // -1).tolist() == [m, -7, 7, 0]  # wraps (docs/api/frame.md §1.5)
    assert (s % -1).tolist() == [0, 0, 0, 0]
    assert abs(s).tolist() == [m, 7, 7, 0]
    assert (s // 0).tolist() == [NA] * 4 and (s % 0).tolist() == [NA] * 4
    assert (sd.frame.Series([2, -1, 0]) ** -1).tolist() == [NA, NA, NA]
    assert (sd.frame.Series([0]) ** 0).tolist() == [1]
    assert (sd.frame.Series([2**62]) * 4).tolist() == [0]
    assert (sd.frame.Series([7, -7]) // 2).tolist() == [3, -4] and (sd.frame.Series([7, -7]) % 2).tolist() == [1, 1]


def test_na_propagates_without_exceptions():
    s = sd.frame.Series([1, None])
    assert (s**0).tolist() == [1, NA]  # pandas gives 1 for NA ** 0 (docs/api/frame.md §3.2)
    assert (1**s).tolist() == [1, NA]
    assert (s + NA).tolist() == [NA, NA] and (s + NA).dtype is sd.int64
    assert (s / NA).dtype is sd.float64


def test_strings_concatenate():
    s = sd.frame.Series(["a", None, "é"])
    assert (s + "!").tolist() == ["a!", NA, "é!"]
    assert ("<" + s).tolist() == ["<a", NA, "<é"]
    assert (s + s).tolist() == ["aa", NA, "éé"]


@pytest.mark.parametrize(
    "expr",
    [
        lambda: sd.frame.Series(["a"]) * 2,
        lambda: sd.frame.Series(["a"]) - "a",
        lambda: sd.frame.Series(["a"]) + 1,
        lambda: -sd.frame.Series(["a"]),
        lambda: +sd.frame.Series(["a"]),
        lambda: ~sd.frame.Series([1.5]),
        lambda: ~sd.frame.Series(["a"]),
        lambda: sd.frame.Series([1.5]) & sd.frame.Series([1]),
        lambda: sd.frame.Series([1]) & 1.0,
        lambda: sd.frame.Series(["a"]) | sd.frame.Series(["b"]),
        lambda: sd.frame.Series([True]) << 1,  # shifts take int64 only, as in the array area
        lambda: sd.frame.Series([1]) >> True,
        lambda: sd.frame.Series([1.0]) << 1,
        lambda: sd.frame.Series([1]) << 2**64,
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
    assert (t + t).dtype is sd.int64 and (t + t).tolist() == [2, 0, NA]
    assert (t + 1).tolist() == [2, 1, NA] and (2 * t).tolist() == [2, 0, NA]
    q = (t / t).tolist()
    assert (t + 0.5).dtype is sd.float64 and q[0] == 1.0 and math.isnan(q[1]) and q[2] is NA
    assert (t // t).tolist() == [1, NA, NA]  # the int rule: // 0 is NA
    assert (-t).tolist() == [-1, 0, NA] and abs(t).tolist() == [1, 0, NA]
    assert (+t).dtype is sd.int64 and (+t).tolist() == [1, 0, NA]


def test_numpy_does_not_hijack_operators():
    with pytest.raises(TypeError):
        np.array([1]) + sd.frame.Series([1])


def test_names_follow_pandas():
    a, b = sd.frame.Series([1], name="x"), sd.frame.Series([1], name="x")
    assert (a + b).name == "x" and (a + sd.frame.Series([1], name="y")).name is None and (a * 2).name == "x"


def test_operands_with_different_indexes_align():
    s = sd.frame.Series([1, 2, 3])
    out = s.tail(2) + s.head(2)  # labels 1, 2 and 0, 1: union 1, 2, 0 (docs/api/frame.md §5.6)
    assert out.index.tolist() == [1, 2, 0] and out.tolist() == [4, NA, NA]
    with pytest.raises(ValueError):  # the kernel itself still wants equal lengths
        sd.frame._core.binary_op("add", s._col, s.head(2)._col, 0)


# --- comparison ---------------------------------------------------------------

COMPARE = {
    "==": operator.eq,
    "!=": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}
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
    assert got.dtype is sd.bool and got.tolist() == expected


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
    assert got.tolist() == [NA if x is None or y is None else COMPARE[op](x, y) for x, y in zip(xs, ys)]


def test_comparison_examples():
    assert (sd.frame.Series([2**53 + 1]) == 2.0**53).tolist() == [True]  # as numpy, pandas and the array area
    assert (sd.frame.Series([math.nan]) != sd.frame.Series([math.nan])).tolist() == [True]
    assert (1 < sd.frame.Series([0, 2])).tolist() == [False, True]
    assert (sd.frame.Series([1, None]) == NA).tolist() == [NA, NA]


@pytest.mark.parametrize("pair", [(["a"], 1), (["a"], True), ([True], "1"), ([1], "1")])
def test_incomparable_dtypes_raise_even_for_equality(pair):
    values, other = pair
    for op in COMPARE.values():
        with pytest.raises(TypeError):
            op(sd.frame.Series(values), other)


def test_bool_compares_with_numbers_as_int64():
    # a bool promotes, as in the array area (DESIGN.md §10.1 #2, docs/api/frame.md §3.3)
    b = sd.frame.Series([True, False, None])
    assert (b == 1).tolist() == [True, False, NA]
    assert (b < 0.5).tolist() == [False, True, NA]
    assert (sd.frame.Series([0, 1, 2]) >= sd.frame.Series([False, True, True])).tolist() == [True, True, True]
    for op in COMPARE.values():
        x, y = [True, False, True], [1, 0, 2]
        assert op(sd.frame.Series(x), sd.frame.Series(y)).tolist() == op(sd.asarray(x), sd.asarray(y)).tolist()
        assert op(sd.frame.Series(x), 0.5).tolist() == op(sd.asarray(x), 0.5).tolist()


@pytest.mark.parametrize("other", [[1, 2], (1, 2), None, np.array([1, 2]), sd.asarray([1, 2]), {1: 2}, object()])
def test_equality_with_an_unsupported_operand_raises(other):
    # == and != raise as < does, not Python's identity fallback (DESIGN.md §10.1 #1)
    s, df = sd.frame.Series([1, 2]), sd.frame.DataFrame({"a": [1, 2]})
    for obj in (s, df):
        for op in (lambda a, b: a == b, lambda a, b: a != b, lambda a, b: b == a, lambda a, b: b != a):
            with pytest.raises(TypeError):
                op(obj, other)
    with pytest.raises(TypeError, match=r"^'==' not supported between instances of 'Series' and 'list'$"):
        s == [1, 2]  # noqa: B015
    with pytest.raises(TypeError, match="DataFrame and a Series"):
        s == df  # noqa: B015


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
    assert (a & b).tolist() == to(pa & pb)
    assert (a | b).tolist() == to(pa | pb)
    assert (a ^ b).tolist() == [NA if x is None or y is None else x != y for x, y in zip(xs, ys)]
    assert (~a).tolist() == to(~pa)
    for scalar in (True, False, NA):
        ps = pd.NA if scalar is NA else scalar
        assert (a & scalar).tolist() == to(pa & ps) and (scalar | a).tolist() == to(ps | pa)


# --- int bitwise operators ------------------------------------------------------

BITWISE = {
    "&": (operator.and_, "bitwise_and"),
    "|": (operator.or_, "bitwise_or"),
    "^": (operator.xor, "bitwise_xor"),
    "<<": (operator.lshift, "bitwise_left_shift"),
    ">>": (operator.rshift, "bitwise_right_shift"),
}
SHIFTS = st.sampled_from([-(2**63), -65, -64, -63, -1, 0, 1, 2, 31, 62, 63, 64, 65, 100, 2**63 - 1]) | st.integers(
    -130, 130
)


def array_oracle(fn, xs, ys):
    """The array area on the rows where neither side is NA; NA elsewhere (docs/api/frame.md §3.4)."""
    rows = [i for i, (x, y) in enumerate(zip(xs, ys)) if x is not None and y is not None]
    got = (
        getattr(sd, fn)(
            sd.asarray([xs[i] for i in rows], dtype=sd.int64), sd.asarray([ys[i] for i in rows], dtype=sd.int64)
        ).tolist()
        if rows
        else []
    )
    out = [NA] * len(xs)
    for i, v in zip(rows, got):
        out[i] = v
    return out


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.data())
def test_int_bitwise_matches_the_array_area(data):
    op = data.draw(st.sampled_from(list(BITWISE)))
    f, fn = BITWISE[op]
    n = data.draw(st.integers(0, 30))
    xs = data.draw(st.lists(with_na(I64 | SMALL), min_size=n, max_size=n))
    ys = data.draw(st.lists(with_na(SHIFTS if op in ("<<", ">>") else I64 | SMALL), min_size=n, max_size=n))
    a, b = sd.frame.Series(xs, dtype="int64"), sd.frame.Series(ys, dtype="int64")
    out = f(a, b)
    assert out.dtype is sd.int64 and out.tolist() == array_oracle(fn, xs, ys)
    if n:  # a scalar on either side
        y = ys[0] if ys[0] is not None else 3
        assert f(a, y).tolist() == array_oracle(fn, xs, [y] * n)
        assert f(y, a).tolist() == array_oracle(fn, [y] * n, xs)
    inv = (~a).tolist()
    assert inv == [
        NA if x is None else v
        for x, v in zip(xs, sd.bitwise_invert(sd.asarray([x or 0 for x in xs], dtype=sd.int64)).tolist())
    ]


def test_shift_edge_cases():
    # The count is read as unsigned, as in the array area: 64 or more, or
    # negative, shifts every bit out (<< gives 0, >> the sign). Overflow wraps.
    values = [1, -8, None, 2**62, -1, 7]
    s = sd.frame.Series(values)

    def wrap(v):
        return (v + 2**63) % 2**64 - 2**63

    for k in (-(2**63), -64, -1, 0, 1, 63, 64, 65, 2**63 - 1):
        inside = 0 <= k < 64
        assert (s << k).tolist() == [NA if v is None else (wrap(v << k) if inside else 0) for v in values]
        assert (s >> k).tolist() == [NA if v is None else (v >> k if inside else -(v < 0)) for v in values]
        assert (s << k).tolist() == array_oracle("bitwise_left_shift", values, [k] * len(values))
        assert (s >> k).tolist() == array_oracle("bitwise_right_shift", values, [k] * len(values))
    assert (s << 63).tolist()[0] == -(2**63)
    assert (sd.frame.Series([-5]) >> -1).tolist() == [-1] and (sd.frame.Series([5]) >> 64).tolist() == [0]


def test_bitwise_with_bool_and_na():
    i, t = sd.frame.Series([3, 2, None, 6]), sd.frame.Series([True, False, True, None])
    # bool takes part as int64 next to an int, as in the array area
    assert (i & t).dtype is sd.int64 and (i & t).tolist() == [1, 0, NA, NA]
    assert (t | i).tolist() == [3, 2, NA, NA] and (i ^ True).tolist() == [2, 3, NA, 7]
    assert (t & 1).dtype is sd.int64 and (t & 1).tolist() == [1, 0, 1, NA]
    assert sd.bitwise_and(sd.asarray([True, False]), sd.asarray([3, 2])).tolist() == (t & i).tolist()[:2]
    # two bools keep Kleene logic (docs/api/frame.md §1.4)
    assert (t & False).dtype is sd.bool and (t & False).tolist() == [False, False, False, False]
    assert (t | NA).tolist() == [True, NA, True, NA] and (~t).tolist() == [False, True, False, NA]
    # NA with an int column is an int64 NA: every row NA
    assert (i << NA).dtype is sd.int64 and (i & NA).tolist() == [NA] * 4
    assert (~i).tolist() == [-4, -3, NA, -7]


def test_frame_bitwise_operators():
    df = sd.frame.DataFrame({"a": [1, 6], "t": [True, None]})
    assert (df & 3)["a"].tolist() == [1, 2] and (df & 3)["t"].tolist() == [1, NA]
    assert (df[["a"]] << 2)["a"].tolist() == [4, 24]
    assert (1 << df[["a"]])["a"].tolist() == [2, 64] and (df[["a"]] >> df[["a"]])["a"].tolist() == [0, 0]
    assert (~df)["a"].tolist() == [-2, -7] and (~df)["t"].tolist() == [False, NA]
    assert (df ^ df)["t"].tolist() == [False, NA] and (df | df)["a"].tolist() == [1, 6]
    with pytest.raises(TypeError):
        df << 1  # the bool column cannot shift
    with pytest.raises(TypeError):
        sd.frame.DataFrame({"f": [1.5]}) & 1


# --- astype -------------------------------------------------------------------


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(st.floats())))
def test_float_to_string_is_python_str(values):
    assert sd.frame.Series(values, dtype="float64").astype("string").tolist() == [
        NA if v is None else str(v) for v in values
    ]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(I64)))
def test_int_to_string_and_back(values):
    s = sd.frame.Series(values, dtype="int64").astype("string")
    assert s.tolist() == [NA if v is None else str(v) for v in values]
    assert s.astype("int64").tolist() == [NA if v is None else v for v in values]


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(st.floats()))
def test_float_string_round_trip(values):
    back = sd.frame.Series(values, dtype="float64").astype("string").astype("float64").tolist()
    assert all(same(v, b) for v, b in zip(values, back))


GOOD_FLOATS = st.from_regex(
    r"\A[+-]?([0-9]{1,5}(\.[0-9]{0,4})?|\.[0-9]{1,4})([eE][+-]?[0-9]{1,3})?\Z"
) | st.sampled_from(["nan", "-NaN", "inf", "+Infinity", "-iNf", "1e400", "-1e400", "1e-400", "4.9e-324", "2.5e-310"])


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(GOOD_FLOATS))
def test_string_to_float_matches_python_float(values):
    got = sd.frame.Series(values, dtype="string").astype("float64").tolist()
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
    assert sd.frame.Series(["True", "False", None]).astype("bool").tolist() == [True, False, NA]
    for text in ("true", "1", "", "False "):
        with pytest.raises(ValueError):
            sd.frame.Series([text]).astype("bool")


@settings(max_examples=EXAMPLES, deadline=None)
@given(st.lists(with_na(st.floats(min_value=-(2.0**63), max_value=2.0**63, exclude_max=True))))
def test_float_to_int_truncates(values):
    got = sd.frame.Series(values, dtype="float64").astype("int64").tolist()
    assert got == [NA if v is None else math.trunc(v) for v in values]


@pytest.mark.parametrize(
    ("value", "expected"),
    [(math.nan, 0), (math.inf, 2**63 - 1), (-math.inf, -(2**63)), (2.0**63, 2**63 - 1), (-(2.0**63) * 2, -(2**63))],
)
def test_float_to_int_saturates(value, expected):
    # the array area's rule (docs/api/frame.md §3.5)
    assert sd.frame.Series([value]).astype("int64").tolist() == [expected]


def test_bool_and_number_conversions():
    assert sd.frame.Series([True, False, None]).astype("int64").tolist() == [1, 0, NA]
    assert sd.frame.Series([True, None]).astype("float64").tolist() == [1.0, NA]
    assert sd.frame.Series([0, 5, None]).astype("bool").tolist() == [False, True, NA]
    assert sd.frame.Series([0.0, -0.0, math.nan]).astype("bool").tolist() == [False, False, True]
    assert sd.frame.Series([True, None]).astype("string").tolist() == ["True", NA]
    assert sd.frame.Series([2**53 + 1]).astype("float64").tolist() == [float(2**53)]  # astype rounds


# --- NA handling --------------------------------------------------------------


def test_isna_notna():
    s = sd.frame.Series([1.0, math.nan, None])
    assert s.isna().tolist() == [False, False, True]  # NaN is a value (docs/api/frame.md §1.3)
    assert s.notna().tolist() == [True, True, False]


def test_isnan():
    # NaN, not NA: NA rows stay NA as in a comparison; other dtypes have no NaN (docs/api/frame.md §3.6)
    s = sd.frame.Series([1.0, math.nan, None, -math.nan, math.inf, -0.0], name="v", index=list("abcdef"))
    out = s.isnan()
    assert out.dtype is sd.bool and out.tolist() == [False, True, NA, True, False, False]
    assert out.name == "v" and out.index.tolist() == list("abcdef")
    assert s.isnan().tolist()[:2] == sd.isnan(sd.asarray([1.0, math.nan])).tolist()
    assert sd.frame.Series([1, None, 3]).isnan().tolist() == [False, NA, False]
    assert sd.frame.Series([True, None]).isnan().tolist() == [False, NA]
    assert sd.frame.Series(["nan", None]).isnan().tolist() == [False, NA]
    assert sd.frame.Series([], dtype="float64").isnan().tolist() == []
    assert s[s.isnan()].index.tolist() == ["b", "d"]  # an NA mask row is dropped (docs/api/frame.md §5.5)
    assert s.iloc[1:3].isnan().tolist() == [True, NA]  # a slice keeps its own NA rows
    df = sd.frame.DataFrame({"f": [math.nan, None, 0.5], "i": [1, 2, None], "s": ["a", None, "b"]})
    got = df.isnan()
    assert got.columns.tolist() == ["f", "i", "s"]
    assert got["f"].tolist() == [True, NA, False] and got["i"].tolist() == [False, False, NA]
    assert got["s"].tolist() == [False, NA, False]
    big = sd.frame.Series([math.nan if k % 3 == 0 else (None if k % 5 == 0 else float(k)) for k in range(300_000)])
    expected = [True if k % 3 == 0 else (NA if k % 5 == 0 else False) for k in range(300_000)]
    assert big.isnan().tolist() == expected


def test_fillna():
    # the fill value converts by dtype only, as an assignment (docs/api/frame.md §3.6, §5.7)
    assert sd.frame.Series([1.5, None]).fillna(0).tolist() == [1.5, 0.0]
    assert sd.frame.Series(["a", None]).fillna("z").tolist() == ["a", "z"]
    assert sd.frame.Series([True, None]).fillna(False).tolist() == [True, False]
    assert sd.frame.Series([1, None]).fillna(NA).tolist() == [1, NA]
    assert sd.frame.Series([1, None]).fillna(True).tolist() == [1, 1]  # bool -> int64 goes up
    for value in (2.0, 1.5):  # float64 -> int64 goes down, whatever the value
        with pytest.raises(TypeError):
            sd.frame.Series([1, None]).fillna(value)
    with pytest.raises(TypeError):
        sd.frame.Series(["a", None]).fillna(1)
    df = sd.frame.DataFrame({"a": [1, None], "b": ["x", None]})
    assert df.fillna({"a": 0})["b"].tolist() == ["x", NA] and df.fillna({"a": 0})["a"].tolist() == [1, 0]
    with pytest.raises(KeyError):
        df.fillna({"zz": 0})


def test_dropna():
    s = sd.frame.Series([1, None, 3], name="v")
    d = s.dropna()
    assert d.tolist() == [1, 3] and d.index.tolist() == [0, 2] and d.name == "v"
    assert s.dropna(ignore_index=True).index.equals(sd.frame.RangeIndex(2))
    df = sd.frame.DataFrame({"a": [1, None, None], "b": ["x", "y", None]})
    assert df.dropna().index.tolist() == [0]
    assert df.dropna(how="all").index.tolist() == [0, 1]
    assert df.dropna(subset="b").index.tolist() == [0, 1]
    assert df.dropna(subset=[]).index.tolist() == [0, 1, 2]
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
    assert out.tolist() == [NA if values[i] is None else values[i] for i in kept]
    assert out.index.tolist() == kept and isinstance(out.index, sd.frame.Index)


def test_filter_keeps_labels_of_a_sliced_range():
    s = sd.frame.Series(list(range(10))).tail(5)
    assert s[s % 2 == 0].index.tolist() == [6, 8]
    f = s[s > 0]
    assert f[f > 7].index.tolist() == [8, 9]  # filtering an Index takes its labels


def test_frame_filter():
    df = sd.frame.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    out = df[df["a"] >= 2]
    assert out["b"].tolist() == ["y", "z"] and out.index.tolist() == [1, 2]
    with pytest.raises(TypeError):
        df[df["a"]]
    with pytest.raises(ValueError):  # the mask must hold the same labels (docs/api/frame.md §5.5)
        df[df["a"].tail(2) > 0]


def test_index_basics():
    idx = sd.frame.Index([3, 1, 2])
    assert idx.tolist() == [3, 1, 2] and idx.dtype is sd.int64 and len(idx) == 3
    assert sd.frame.Index([0, 1]).equals(sd.frame.RangeIndex(2)) and sd.frame.RangeIndex(2).equals(
        sd.frame.Index([0, 1])
    )
    with pytest.raises(ValueError):
        sd.frame.Index([1, None])


@pytest.mark.parametrize(
    "values",
    [
        [],
        [7],
        [0, 2, 5],
        list(range(30)),
        [x * 1000 for x in range(30)],
        list(range(0, 400, 2)),
        list(range(101)),
        list(range(100)),
        [-5, 10**15, 3],
    ],
)
def test_index_repr_matches_pandas(values):
    # but dtype=int64, as the array's repr (DESIGN.md §10.2) [pandas: dtype='int64']
    assert repr(sd.frame.Index(values)) == repr(pd.Index(np.array(values, dtype=np.int64))).replace(
        "dtype='int64'", "dtype=int64"
    )


@pytest.mark.parametrize("n", [5, 70])
def test_filtered_repr_matches_pandas(n):
    s = sd.frame.Series([float(i) / 3 for i in range(n)], name="x")
    p = pd.Series([float(i) / 3 for i in range(n)], name="x")
    assert repr(s[s * 3 % 2 == 0]) == repr(p[p * 3 % 2 == 0])
    df, pdf = sd.frame.DataFrame({"x": list(range(n))}), pd.DataFrame({"x": range(n)})
    assert repr(df[df["x"] % 3 == 1]) == pdf[pdf["x"] % 3 == 1].to_string(
        max_rows=60, min_rows=10, show_dimensions="truncate"
    )


# --- DataFrame operators ------------------------------------------------------


def test_frame_operators():
    df = sd.frame.DataFrame({"a": [1, 2], "b": [0.5, None]})
    assert (df + 1)["a"].tolist() == [2, 3] and (2 * df)["b"].tolist() == [1.0, NA]
    other = sd.frame.DataFrame({"b": [1.0, 1.0], "a": [10, 20]})
    s = df + other
    assert s.columns.tolist() == ["a", "b"] and s["a"].tolist() == [11, 22]  # left's column order
    assert (df > 1)["a"].tolist() == [False, True]
    assert (-df)["a"].tolist() == [-1, -2] and abs(-df)["a"].tolist() == [1, 2]
    assert df.isna()["b"].tolist() == [False, True]
    assert [str(d) for d in df.astype({"a": "float64"}).dtypes.values()] == ["float64", "float64"]
    assert df.astype("string")["b"].tolist() == ["0.5", NA]
    with pytest.raises(TypeError):
        df + df["a"]
    partial = df + sd.frame.DataFrame({"a": [1, 2]})  # "b" is only on the left: NA there (docs/api/frame.md §5.6)
    assert partial["a"].tolist() == [2, 4] and partial["b"].tolist() == [NA, NA] and partial["b"].dtype is sd.float64
    with pytest.raises(KeyError):
        df.astype({"zz": "int64"})


# --- the Eisel-Lemire float parser (docs/design/frame.md §10.4) -----------------

HARD_FLOATS = [
    "0",
    "-0",
    "0.0",
    "1e22",
    "1e23",
    "9007199254740993",
    "9007199254740992.5",
    "2.2250738585072011e-308",
    "2.2250738585072014e-308",
    "4.9406564584124654e-324",
    "2.4703282292062327e-324",
    "2.4703282292062328e-324",
    "1.7976931348623157e308",
    "1.7976931348623158e308",
    "1.7976931348623159e308",
    "1e-400",
    "1e400",
    "123456789012345678",
    "1234567890123456789",
    "12345678901234567890",
    ".5",
    "5.",
    "7.2057594037927933e16",
    "0.000000000000000000000000000001",
    "9007199254740991.4999999999999999999",
]


@settings(max_examples=EXAMPLES, deadline=None)
@given(
    st.lists(
        st.integers(0, 2**64 - 1).map(lambda b: repr(np.frombuffer(np.uint64(b).tobytes(), dtype=np.float64)[0].item()))
        | st.builds(lambda m, e: f"{m}e{e}", st.integers(0, 10**19), st.integers(-345, 310))
        | st.builds(
            lambda a, b, e: f"{a}.{b}e{e}", st.integers(0, 10**10), st.integers(0, 10**12), st.integers(-30, 30)
        ),
        max_size=50,
    )
)
def test_float_parsing_is_correctly_rounded(texts):
    texts = [t for t in texts if "nan" not in t and "inf" not in t] + HARD_FLOATS
    got = sd.frame.Series(texts).astype("float64").tolist()
    assert all(same(g, float(t)) for g, t in zip(got, texts))


# --- operands that are not scalars, and unary + --------------------------------


@pytest.mark.parametrize(
    "value",
    [np.complex128(1 + 2j), np.complex64(1j), np.datetime64("2020-01-01"), np.timedelta64(1, "s"), np.bytes_(b"a")],
)
def test_numpy_scalars_without_a_column_dtype_are_not_operands(value):
    # As Python's complex: a TypeError, never the real part with a ComplexWarning.
    import warnings

    s = sd.frame.Series([1.0, 2.0])
    df = sd.frame.DataFrame({"a": [1.0]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for f in (lambda: s + value, lambda: value + s, lambda: s == value, lambda: s < value, lambda: df * value):
            with pytest.raises(TypeError):
                f()


def test_unary_plus_returns_a_new_object():
    # Copy-on-Write (docs/api/frame.md §5.7): writing to +s does not change s.
    s = sd.frame.Series([1, 2])
    t = +s
    assert t is not s
    t.iloc[0] = 99
    assert s.tolist() == [1, 2] and t.tolist() == [99, 2]
    df = sd.frame.DataFrame({"a": [1, 2]})
    d2 = +df
    assert d2 is not df
    d2["b"] = [3, 4]
    assert list(df) == ["a"]
