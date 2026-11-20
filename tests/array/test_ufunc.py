"""Every elementwise function on every dtype, against numpy.

The expected loop dtype comes from the spec below (which dtypes each function
has a loop for); numpy then computes the reference on inputs already
converted to that dtype. Where stardust deliberately differs from numpy
(integer division and power, bool arithmetic: docs/api/array.md §1.6, §1.4)
the reference is plain Python instead.
"""

import itertools
import math
import operator

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust as sd
from bench.accuracy.common import ulp_distance

from . import refdata
from .conftest import NP_DTYPE, assert_same, assert_ulp

B, I, F, C = sd.bool, sd.int64, sd.float64, sd.complex128
ORDER = [B, I, F, C]
INT_MAX, INT_MIN = 2**63 - 1, -(2**63)

_CF = [0.0, -0.0, 1.5, -2.5, math.inf, -math.inf, math.nan, 0.7, 1e-8, 2.0]
VALUES = {
    B: np.array([True, False]),
    I: np.array([0, 1, -1, 2, -7, 3, 5, INT_MAX, INT_MIN, 12345678901, 63, 64, -64]),
    F: np.array(
        [
            0.0,
            -0.0,
            1.5,
            -2.5,
            0.5,
            3.0,
            math.inf,
            -math.inf,
            math.nan,
            1e308,
            5e-324,
            -1e-310,
            2.0,
            100.0,
            0.7,
            -0.3,
            710.0,
            1e-8,
            -1.0,
            1.0,
        ]
    ),
    C: np.array([complex(r, i) for r in _CF for i in _CF]),
}

# Which dtypes each function has a loop for (the spec; mirrors scalar_ops.hpp).
ALL, NUM, INEXACT, REAL, INT_FLOAT, BOOL_INT = set(ORDER), {I, F, C}, {F, C}, {B, I, F}, {I, F}, {B, I}
UNARY = {
    "negative": NUM,
    "positive": NUM,
    "abs": NUM,
    "square": NUM,
    "sign": NUM,
    "sqrt": INEXACT,
    "exp": INEXACT,
    "expm1": INEXACT,
    "log": INEXACT,
    "log1p": INEXACT,
    "log2": INEXACT,
    "log10": INEXACT,
    "sin": INEXACT,
    "cos": INEXACT,
    "tan": INEXACT,
    "asin": INEXACT,
    "acos": INEXACT,
    "atan": INEXACT,
    "sinh": INEXACT,
    "cosh": INEXACT,
    "tanh": INEXACT,
    "asinh": INEXACT,
    "acosh": INEXACT,
    "atanh": INEXACT,
    "floor": REAL,
    "ceil": REAL,
    "trunc": REAL,
    "round": ALL,
    "isnan": ALL,
    "isinf": ALL,
    "isfinite": ALL,
    "conj": ALL,
    "logical_not": {B},
    "bitwise_invert": BOOL_INT,
}
BINARY = {
    "add": NUM,
    "subtract": NUM,
    "multiply": NUM,
    "divide": INEXACT,
    "pow": NUM,
    "floor_divide": INT_FLOAT,
    "remainder": INT_FLOAT,
    "atan2": {F},
    "maximum": REAL,
    "minimum": REAL,
    "equal": ALL,
    "not_equal": ALL,
    "less": REAL,
    "less_equal": REAL,
    "greater": REAL,
    "greater_equal": REAL,
    "logical_and": {B},
    "logical_or": {B},
    "logical_xor": {B},
    "bitwise_and": BOOL_INT,
    "bitwise_or": BOOL_INT,
    "bitwise_xor": BOOL_INT,
    "bitwise_left_shift": {I},
    "bitwise_right_shift": {I},
}
# float64 functions that run on vForce: accurate to 8 ULP by contract
# (docs/api/array.md §6.4); and pow (libm, whose exact forms for the exponents 2, -1 and
# 0.5 are correctly rounded where numpy's libm is not, docs/design/array.md §5.4).
# Everything else must match numpy bit for bit.
VFORCE = {
    "exp",
    "log",
    "log2",
    "log10",
    "sin",
    "asin",
    "acos",
    "atan",
    "atan2",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "pow",
}


def loop_dtype(loops, d):
    return next((k for k in ORDER[ORDER.index(d) :] if k in loops), None)


def wrap(v):
    return (v + 2**63) % 2**64 - 2**63


def py_int_binary(name, a, b):
    """docs/api/array.md §1.6 and the shift rule, as plain Python on Python ints."""
    if name == "floor_divide":
        return 0 if b == 0 else wrap(a // b)
    if name == "remainder":
        return a if b == 0 else a % b
    if name == "pow":
        if b < 0:
            return 1 if a == 1 else ((-1 if b % 2 else 1) if a == -1 else 0)
        return wrap(pow(a, b, 2**64))
    if name == "bitwise_left_shift":
        return 0 if not 0 <= b < 64 else wrap(a << b)
    if name == "bitwise_right_shift":
        return (-1 if a < 0 else 0) if not 0 <= b < 64 else a >> b
    raise AssertionError(name)


def reference_unary(name, n, loop):
    x = n.astype(NP_DTYPE[loop])
    if name == "conj" and loop != C:
        return x
    if name in ("floor", "ceil", "trunc", "round") and loop in (B, I):
        return x  # already whole; numpy returns float16 for bool
    return getattr(np, name)(x)


def reference_binary(name, a, b, loop):
    a, b = np.broadcast_arrays(a.astype(NP_DTYPE[loop]), b.astype(NP_DTYPE[loop]))
    if loop is I and name in ("floor_divide", "remainder", "pow", "bitwise_left_shift", "bitwise_right_shift"):
        out = [py_int_binary(name, x, y) for x, y in zip(a.ravel().tolist(), b.ravel().tolist())]
        return np.array(out, dtype=np.int64).reshape(a.shape)
    return getattr(np, name)(a, b)


def check(got, want, name, loop):
    if name in VFORCE and loop is F:
        assert_ulp(got, want, 8)
    elif name == "abs" and loop is C:  # sqrt(fma(re, re, im*im)): 3 ULP contract against Arb; numpy's formula differs
        assert_ulp(got, want, 4)
    else:
        assert_same(got, want)


@pytest.mark.parametrize("name", sorted(UNARY))
def test_unary_all_dtypes(name, dtype):
    loop = loop_dtype(UNARY[name], dtype)
    n = VALUES[dtype]
    fn = getattr(sd, name)
    if loop is None:
        with pytest.raises(TypeError):
            fn(sd.asarray(n))
        return
    with np.errstate(all="ignore"):
        want = reference_unary(name, n, loop)
    check(fn(sd.asarray(n)), want, name, loop)  # contiguous
    check(fn(sd.asarray(n)[::-1]), want[::-1], name, loop)  # strided


@pytest.mark.parametrize("name", sorted(BINARY))
@pytest.mark.parametrize("da,db", list(itertools.product(ORDER, ORDER)), ids=lambda d: str(d))
def test_binary_all_dtype_pairs(name, da, db):
    a, b = VALUES[da][:, None], VALUES[db][None, :]
    fn = getattr(sd, name)
    loop = loop_dtype(BINARY[name], max(da, db, key=ORDER.index))
    if name.startswith("bitwise_") and "shift" in name and B in (da, db):
        loop = None  # shifts take integers only
    if loop is None:
        with pytest.raises(TypeError):
            fn(sd.asarray(a), sd.asarray(b))
        return
    with np.errstate(all="ignore"):
        want = reference_binary(name, a, b, loop)
    check(fn(sd.asarray(a), sd.asarray(b)), want, name, loop)  # scalar-broadcast inner loops
    # The same pairs as contiguous arrays and as strided views: each layout has
    # its own kernel path (docs/design/array.md §5.1), and all must give the same bits.
    fa, fb = (np.ascontiguousarray(v) for v in np.broadcast_arrays(a, b))
    check(fn(sd.asarray(fa), sd.asarray(fb)), want, name, loop)
    check(fn(sd.asarray(fa.T), sd.asarray(fb.T)), want.T, name, loop)


_ELEMENTS = {"allow_nan": True, "allow_infinity": True}


@given(st.data())
def test_random_strided_broadcast_operands(data):
    # Arbitrary layouts (reversed, stepped, transposed), broadcasting and mixed
    # dtypes all go through the same iterator: compare a few ops with numpy.
    da, db = data.draw(st.sampled_from(ORDER)), data.draw(st.sampled_from(ORDER))
    shape = data.draw(hnp.array_shapes(min_dims=1, max_dims=3, max_side=4))
    a = data.draw(hnp.arrays(NP_DTYPE[da], shape, elements=None if da in (B, I) else _ELEMENTS))
    b = data.draw(hnp.arrays(NP_DTYPE[db], shape[data.draw(st.integers(0, len(shape) - 1)) :]))
    a = a[tuple(data.draw(st.sampled_from([slice(None), slice(None, None, -1), slice(None, None, 2)])) for _ in shape)]
    b = b.T if data.draw(st.booleans()) and b.ndim > 1 and len(set(b.shape)) == 1 else b
    if not all(x == y or x == 1 or y == 1 for x, y in zip(a.shape[::-1], b.shape[::-1])):
        return
    name = data.draw(st.sampled_from(["add", "multiply", "subtract", "equal", "maximum", "divide"]))
    loop = loop_dtype(BINARY[name], max(da, db, key=ORDER.index))
    if loop is None:
        return
    with np.errstate(all="ignore"):
        want = reference_binary(name, a, b, loop)
    check(getattr(sd, name)(sd.asarray(a), sd.asarray(b)), want, name, loop)


def test_complex_arithmetic_on_random_values():
    # Complex multiply/divide/square use numpy's exact formulas, including
    # where its compiler fused a multiply-add (see scalar_ops.hpp).
    rng = np.random.default_rng(0)
    a = rng.standard_normal(5000) * 10.0 ** rng.integers(-5, 5, 5000) + 1j * rng.standard_normal(5000)
    b = rng.standard_normal(5000) + 1j * rng.standard_normal(5000) * 10.0 ** rng.integers(-5, 5, 5000)
    sa, sb = sd.asarray(a), sd.asarray(b)
    with np.errstate(all="ignore"):  # numpy warns on overflow; stardust never does
        for name in ("add", "subtract", "multiply", "divide", "pow"):
            assert_same(getattr(sd, name)(sa, sb), getattr(np, name)(a, b))
        for name in ("square", "sqrt", "exp", "log", "log1p", "log2", "log10", "sign", "tanh"):
            assert_same(getattr(sd, name)(sa), getattr(np, name)(a))
        assert_ulp(sd.abs(sa), np.abs(a), 4)  # not numpy's cabs formula: each within its ULP of the exact value
        # sin and cos of the imaginary part may come from one sincos call: 2 ULP of numpy
        finite = np.isfinite(np.expm1(a))
        assert_ulp(sd.expm1(sa)[sd.asarray(finite)], np.expm1(a)[finite], 2)
    small = sd.asarray(np.arange(-5, 6) + 0j)
    assert_same(sd.pow(sa[:11], small), np.power(a[:11], np.arange(-5, 6) + 0j))  # integer-exponent path


# ----------------------------------- integer division and bool arithmetic


def test_integer_division_never_fails():
    a = sd.asarray([7, -7, 0, INT_MIN, INT_MIN])
    b = sd.asarray([0, 0, 0, -1, 1])
    assert (a // b).tolist() == [0, 0, 0, INT_MIN, INT_MIN]
    assert (a % b).tolist() == [7, -7, 0, 0, 0]
    # a == (a // b) * b + a % b, even for b == 0
    assert ((a // b) * b + a % b == a).tolist() == [True] * 5


def test_floor_semantics_match_python():
    for x, y in itertools.product([7, -7, 6, -6, 1], [3, -3, 2, -2, 7]):
        assert sd.floor_divide(x, y) == x // y
        assert sd.remainder(x, y) == x % y


def test_integer_power():
    base = sd.asarray([2, -1, -1, 1, 0, 3, 2])
    exp = sd.asarray([10, -3, -4, -9, -2, -1, 64])
    assert (base**exp).tolist() == [1024, -1, 1, 1, 0, 0, 0]  # 2**64 wraps to 0
    assert (sd.asarray([3]) ** 40).tolist() == [wrap(3**40)]


def _exact_power(x, e):
    """The exact forms of x ** e for e in 2, 0, 1, -1, 0.5 with C99 pow's special values."""
    with np.errstate(all="ignore"):
        if e == 2:
            return x * x
        if e == 0:
            return np.ones_like(x)
        if e == 1:
            return x.copy()
        if e == -1:
            return 1.0 / x
        r = np.sqrt(x)
        r[x == 0] = 0.0  # pow(-0, 0.5) = +0 (sqrt keeps -0)
        r[x == -math.inf] = math.inf  # pow(-inf, 0.5) = +inf (sqrt gives NaN)
        return r


EXACT_EXPONENTS = [2.0, 0.0, -0.0, 1.0, -1.0, 0.5]


@pytest.mark.parametrize("e", EXACT_EXPONENTS)
def test_float_power_exact_exponents(e):
    # x ** e for the exponents with an exact form (numpy's fast paths; C99 pow's special
    # values), whatever the exponent's layout: a scalar (Python float, and int where it is
    # one), an array of e, a broadcast row of e; out= and in place; strided x. Special
    # values, subnormals and overflow included.
    rng = np.random.default_rng(7)
    x = rng.standard_normal(5000) * 10.0 ** rng.integers(-170, 170, 5000)
    x[:14] = [
        math.nan,
        math.inf,
        -math.inf,
        -0.0,
        0.0,
        5e-324,
        -5e-324,
        1e-160,
        1e200,
        -1e200,
        1.5e154,
        2.0**-537,
        -1.0,
        1.0,
    ]
    want = _exact_power(x, e).tobytes()
    if e != 0.5:  # numpy's scalar fast paths agree (its 0.5 keeps sqrt's -0 and NaN)
        with np.errstate(all="ignore"):
            assert (x**e).tobytes() == want
    with np.errstate(all="ignore"):  # C99 pow's special values (numpy's array-exponent pow is libm's)
        sp = x[:14]
        assert np.array_equal(_exact_power(sp, e), np.power(sp, np.full(14, e)), equal_nan=True)
        assert np.signbit(_exact_power(sp, e)).tolist() == np.signbit(np.power(sp, np.full(14, e))).tolist()
    xs = sd.asarray(x)
    scalars = [e] + ([int(e)] if e == int(e) and not (e == 0 and math.copysign(1, e) < 0) else [])
    outs = [xs**v for v in scalars] + [sd.pow(xs, v) for v in scalars]
    outs += [sd.pow(xs, sd.full(x.shape, e)), sd.pow(xs.reshape((50, 100)), sd.full((1, 100), e)).reshape((5000,))]
    for got in outs:
        assert np.asarray(got).tobytes() == want
    out = sd.empty(x.shape)
    sd.pow(xs, e, out=out)
    assert np.asarray(out).tobytes() == want
    z = sd.asarray(x.copy())
    z **= e
    assert np.asarray(z).tobytes() == want
    assert np.asarray(xs[::3] ** e).tobytes() == _exact_power(x[::3], e).tobytes()
    # mixed exponents: the elements with this exponent take the exact form, the rest libm pow
    ex = np.where(np.arange(5000) % 2 == 0, e, 3.0)
    got = np.asarray(sd.pow(xs, sd.asarray(ex)))
    assert got[::2].tobytes() == _exact_power(x[::2], e).tobytes()
    assert np.asarray(sd.pow(xs, 3.0))[1::2].tobytes() == got[1::2].tobytes()
    # a 1-element result goes through the same rule
    assert np.asarray(sd.pow(xs[3:4], e)).tobytes() == _exact_power(x[3:4], e).tobytes()


def _libm_pow():
    import ctypes
    import ctypes.util

    f = ctypes.CDLL(ctypes.util.find_library("m") or None).pow
    f.restype, f.argtypes = ctypes.c_double, [ctypes.c_double, ctypes.c_double]
    return f


def test_float_power_is_libm_pow():
    # every exponent but the exact forms' (2, 0, 1, -1, 0.5) is libm's pow, element by element,
    # for a scalar and an array exponent: the same bits as numpy and the frame
    pw = _libm_pow()
    rng = np.random.default_rng(11)
    n = 3000
    x = np.exp(rng.uniform(-700, 700, n)) * rng.choice([1.0, -1.0], n)
    e = rng.uniform(-50, 50, n)
    e[::7] = np.round(e[::7])
    e[np.isin(e, EXACT_EXPONENTS)] = 3.0
    special = [0.0, -0.0, math.inf, -math.inf, math.nan, 5e-324, -5e-324, 1e-310, 1.0, -1.0, 0.5, 2.0, 1e308]
    sx = np.array([a for a in special for _ in special])
    se = np.array([b for _ in special for b in special + [3.0, -2.5, 1e300, -1e300]][: len(sx)])
    xs_all = np.concatenate([x, sx])
    es_all = np.concatenate([e, se])
    want = np.array([pw(float(a), float(b)) for a, b in zip(xs_all, es_all)])
    exact = np.isin(es_all, EXACT_EXPONENTS)
    got = np.asarray(sd.pow(sd.asarray(xs_all), sd.asarray(es_all)))
    assert got[~exact].tobytes() == want[~exact].tobytes()
    with np.errstate(all="ignore"):
        assert got[~exact].tobytes() == np.power(xs_all, es_all)[~exact].tobytes()
    for b in (3.0, -2.5, 7.25, 1e-3, -37.0, 1e300, math.inf, -math.inf, math.nan):
        got = np.asarray(sd.asarray(xs_all) ** b)
        assert got.tobytes() == np.array([pw(float(a), b) for a in xs_all]).tobytes()


def test_integer_power_untouched_by_float_fast_paths():
    assert (sd.asarray([3, -4, 2**32]) ** 2).tolist() == [9, 16, 0]
    assert (sd.asarray([3, -4]) ** 2).dtype == sd.int64
    assert (sd.asarray([3, -4, 0]) ** 0).tolist() == [1, 1, 1]
    assert (sd.asarray([3, -4]) ** 1).tolist() == [3, -4]
    assert (sd.asarray([1, -1, 2]) ** -1).tolist() == [1, -1, 0]


def test_bool_arithmetic_is_int64():
    t = sd.asarray([True, False])
    for got, want in [
        (t + t, [2, 0]),
        (t - t, [0, 0]),
        (t * t, [1, 0]),
        (-t, [-1, 0]),
        (+t, [1, 0]),
        (abs(t), [1, 0]),
        (sd.square(t), [1, 0]),
        (sd.sign(t), [1, 0]),
        (t**t, [1, 1]),
    ]:
        assert got.dtype is sd.int64 and got.tolist() == want
    assert (t / t).dtype is sd.float64
    # ... but logic and bit operations keep bool (docs/api/array.md §1.4)
    for got in (t & t, t | t, t ^ t, ~t, sd.maximum(t, t), t == t):
        assert got.dtype is sd.bool
    assert (~t).tolist() == [False, True]


def test_shift_counts_are_unsigned():
    x = sd.asarray([1, -8, 5])
    assert (x << sd.asarray([63, 1, 64])).tolist() == [INT_MIN, -16, 0]
    assert (x >> sd.asarray([-1, 70, 1])).tolist() == [0, -1, 2]


@pytest.mark.parametrize("fn", [sd.logical_and, sd.logical_or, sd.logical_xor])
def test_logical_takes_bool_only(fn):
    t = sd.asarray([True, False])
    assert fn(t, True).dtype is sd.bool
    with pytest.raises(TypeError):
        fn(sd.asarray([1, 0]), t)
    with pytest.raises(TypeError):
        fn(t, 1)


def test_complex_has_no_order():
    c = sd.asarray([1j])
    for fn in (sd.less, sd.maximum, sd.minimum, sd.floor_divide, sd.remainder, sd.atan2):
        with pytest.raises(TypeError):
            fn(c, c)
    with pytest.raises(TypeError):
        c < c  # noqa: B015
    with pytest.raises(TypeError):
        sd.floor(c)
    assert (c == c).tolist() == [True]


def test_maximum_minimum_signed_zero_and_nan():
    a = sd.asarray([-0.0, 0.0, math.nan, 1.0])
    b = sd.asarray([0.0, -0.0, 1.0, math.nan])
    assert_same(sd.maximum(a, b), np.maximum(np.asarray(a), np.asarray(b)))
    assert_same(sd.minimum(a, b), np.minimum(np.asarray(a), np.asarray(b)))
    assert [math.copysign(1, v) for v in sd.maximum(a, b).tolist()[:2]] == [1, 1]
    assert [math.copysign(1, v) for v in sd.minimum(a, b).tolist()[:2]] == [-1, -1]


# ----------------------------------------------------------- Python scalars


def test_python_scalar_promotion():
    i = sd.asarray([1, 2])
    assert (i + 1).dtype is sd.int64
    assert (i + 1.5).dtype is sd.float64 and (i + 1.5).tolist() == [2.5, 3.5]
    assert (i * 1j).dtype is sd.complex128
    assert (sd.asarray([True]) + True).tolist() == [2]
    assert (sd.asarray([1.0]) + 2**70).tolist() == [float(2**70)]  # float target is fine (docs/api/array.md §1.5)
    with pytest.raises(OverflowError):
        i + 2**70  # must become int64 (docs/api/array.md §1.5)
    with pytest.raises(OverflowError):
        i < 2**70  # noqa: B015


@pytest.mark.parametrize(
    "name",
    [
        "add",
        "subtract",
        "multiply",
        "divide",
        "floor_divide",
        "remainder",
        "pow",
        "atan2",
        "maximum",
        "minimum",
        "equal",
        "less",
        "bitwise_and",
    ],
)
def test_big_python_int_one_rule_for_every_binary_function(name):
    # A Python int becomes an int64 when the promoted operand dtype is bool or
    # int64, whatever loop runs: divide and atan2 compute in float64 but still
    # raise (docs/api/array.md §1.5).
    fn = getattr(sd, name)
    for x in (sd.asarray([1, 2]), sd.asarray([True, False]), 1, True):
        for big in (2**63, 2**64, -(2**63) - 1):
            with pytest.raises(OverflowError):
                fn(x, big)
            with pytest.raises(OverflowError):
                fn(big, x)
    if name != "bitwise_and":  # next to a float, the int becomes a float
        assert fn(sd.asarray([1.0]), 2**70).dtype in (sd.float64, sd.bool)
        assert isinstance(fn(2**70, 1.5), (float, bool))


def test_big_python_int_in_unary_functions():
    with pytest.raises(OverflowError):
        sd.sqrt(2**64)  # an int operand is an int64, even though sqrt computes in float64
    with pytest.raises(OverflowError):
        sd.exp(-(2**63) - 1)
    assert sd.sqrt(2**63 - 1) == math.sqrt(2**63 - 1)


def test_scalars_only_give_python_scalars():
    assert sd.add(1, 2) == 3 and type(sd.add(1, 2)) is int
    assert sd.add(2**62, 2**62) == INT_MIN  # int64 semantics, wraps (docs/design/array.md §2.2)
    assert sd.add(True, True) == 2
    assert sd.divide(1, 2) == 0.5
    assert math.isnan(sd.sqrt(-1.0))
    assert sd.sqrt(-1 + 0j) == 1j
    assert sd.less(1, 2) is True
    assert sd.floor_divide(7, 0) == 0


def test_operands_must_be_arrays_or_python_scalars():
    x = sd.asarray([1, 2])
    for bad in ([1, 2], (1, 2), np.array([1, 2]), "1", None):
        with pytest.raises(TypeError):
            sd.add(x, bad)
    with pytest.raises(TypeError):
        x + [1, 2]
    with pytest.raises(TypeError):
        np.array([1, 2]) + x  # numpy steps aside (__array_ufunc__ = None)
    with pytest.raises(TypeError):
        x + np.array([1, 2])


@pytest.mark.parametrize("bad", [[1, 2], (1, 2), np.array([1, 2]), "1", None, object(), sd.int64])
def test_equality_with_a_rejected_operand_is_a_type_error(bad):
    # == and != reject what the other operators reject, whichever side the
    # array is on: no fallback to identity (docs/api/array.md §10.3).
    x = sd.asarray([1, 2])
    for op in (operator.eq, operator.ne):
        with pytest.raises(TypeError):
            op(x, bad)
        with pytest.raises(TypeError):
            op(bad, x)
    for fn in (sd.equal, sd.not_equal, sd.add):
        with pytest.raises(TypeError):
            fn(x, bad)
    assert (x == x).tolist() == [True, True] and (x != 2).tolist() == [True, False]


@pytest.mark.parametrize(
    "scalar, dtype",
    [
        (np.int64(3), sd.int64),
        (np.int8(-3), sd.int64),
        (np.uint32(3), sd.int64),
        (np.uint64(3), sd.int64),
        (np.bool_(True), sd.bool),
        (np.float32(0.5), sd.float64),
        (np.float16(0.5), sd.float64),
        (np.float64(0.5), sd.float64),
        (np.complex64(1j), sd.complex128),
        (np.complex128(1j), sd.complex128),
    ],
)
def test_numpy_scalars_are_operands_like_python_scalars(scalar, dtype):
    # numpy's bool, integer, floating and complexfloating scalars count as the
    # Python scalar of their kind (docs/api/array.md §6.2), wherever a Python
    # scalar is taken: operators, functions, asarray, fill values, assignment.
    py = scalar.item()
    for x in (sd.asarray([True, False]), sd.asarray([1, 2]), sd.asarray([1.0, 2.0])):
        assert_same(x + scalar, np.asarray(x + py))
        assert_same(scalar * x, np.asarray(py * x))
        assert_same(x == scalar, np.asarray(x == py))
    assert sd.add(scalar, 0) == py + 0 and type(sd.multiply(scalar, True)) is type(sd.multiply(py, True))
    assert sd.asarray([scalar]).dtype is dtype
    assert sd.full(2, scalar).dtype is dtype
    z = sd.zeros(1, dtype=sd.complex128)
    z[0] = scalar
    assert z.tolist() == [complex(py)]


def test_numpy_scalars_follow_the_python_rules():
    with pytest.raises(OverflowError):
        sd.asarray([1]) + np.uint64(2**64 - 1)  # becomes an int64, like a Python int
    assert (sd.asarray([1.0]) + np.uint64(2**64 - 1)).tolist() == [1.0 + 2**64]
    i = sd.asarray([1, 2])
    with pytest.raises(TypeError):
        i += np.float32(0.5)  # implicit downcast, as with 0.5
    with pytest.raises(TypeError):
        sd.asarray([np.timedelta64(1, "s")])  # not a number
    with pytest.raises(TypeError):
        sd.add(i, np.datetime64("2020-01-01"))


# ------------------------------------------------------------------- out=


def test_out_basic_and_upcast():
    a, b = sd.asarray([1, 2, 3]), sd.asarray([10, 20, 30])
    out = sd.zeros(3, dtype=sd.int64)
    assert sd.add(a, b, out=out) is out and out.tolist() == [11, 22, 33]
    wide = sd.zeros(3, dtype=sd.complex128)
    sd.multiply(a, b, out=wide)  # int64 result widened into complex128
    assert wide.tolist() == [10, 40, 90]
    big = sd.zeros((2, 3))
    sd.add(a, 0.5, out=big)  # broadcast into a larger out
    assert big.tolist() == [[1.5, 2.5, 3.5]] * 2


def test_out_rules():
    a = sd.asarray([1.0, 2.0])
    with pytest.raises(TypeError):
        sd.add(a, a, out=sd.zeros(2, dtype=sd.int64))  # docs/api/array.md §1.5
    assert sd.less(a, a, out=sd.ones(2)).tolist() == [0.0, 0.0]  # bool -> float64 is upward: fine
    ro = np.zeros(2)
    ro.flags.writeable = False
    with pytest.raises(ValueError):
        sd.add(a, a, out=sd.asarray(ro))
    with pytest.raises(ValueError):
        sd.add(sd.zeros((2, 3)), 1.0, out=sd.zeros(3))  # result does not fit out
    with pytest.raises(TypeError):
        sd.add(a, a, sd.zeros(2))  # out is keyword-only


def test_out_overlapping_inputs():
    n = np.arange(10.0)
    a = sd.asarray(n.copy())
    sd.add(a[1:], a[:-1], out=a[1:])
    n[1:] = n[1:] + n[:-1]
    assert_same(a, n)
    b = sd.arange(6.0)
    sd.multiply(b, b[::-1], out=b)
    assert b.tolist() == [0.0, 4.0, 6.0, 6.0, 4.0, 0.0]


# ------------------------------------------------------------- operators


def test_operators_match_functions():
    x = sd.asarray([[5, -3], [7, 2]])
    y = sd.asarray([2, 3])
    pairs = [
        (x + y, sd.add),
        (x - y, sd.subtract),
        (x * y, sd.multiply),
        (x / y, sd.divide),
        (x // y, sd.floor_divide),
        (x % y, sd.remainder),
        (x**y, sd.pow),
        (x & y, sd.bitwise_and),
        (x | y, sd.bitwise_or),
        (x ^ y, sd.bitwise_xor),
        (x << y, sd.bitwise_left_shift),
        (x >> y, sd.bitwise_right_shift),
        (x == y, sd.equal),
        (x != y, sd.not_equal),
        (x < y, sd.less),
        (x <= y, sd.less_equal),
        (x > y, sd.greater),
        (x >= y, sd.greater_equal),
    ]
    for got, fn in pairs:
        assert_same(got, np.asarray(fn(x, y)))
    for got, fn in [(-x, sd.negative), (+x, sd.positive), (abs(x), sd.abs), (~x, sd.bitwise_invert)]:
        assert_same(got, np.asarray(fn(x)))


def test_reflected_operators():
    x = sd.asarray([1, 2, 4])
    assert (10 - x).tolist() == [9, 8, 6]
    assert (1 / x).tolist() == [1.0, 0.5, 0.25]
    assert (2**x).tolist() == [2, 4, 16]
    assert (7 // x).tolist() == [7, 3, 1]
    assert (1 << x).tolist() == [2, 4, 16]
    assert (3 > x).tolist() == [True, True, False]  # Python turns it into x < 3


def test_inplace_operators():
    x = sd.asarray([1.0, 2.0])
    before = x
    x += 1
    x *= sd.asarray([2.0, 3.0])
    x **= 2
    assert x is before and x.tolist() == [16.0, 81.0]
    i = sd.asarray([1, 2])
    with pytest.raises(TypeError):
        i += 0.5  # float64 cannot go into int64 (docs/api/array.md §1.5)
    with pytest.raises(TypeError):
        i /= 2  # divide gives float64
    b = sd.asarray([True])
    with pytest.raises(TypeError):
        b += True  # bool + bool is int64 (docs/api/array.md §1.4)
    a = sd.arange(5)
    a[1:] += a[:-1]
    assert a.tolist() == [0, 1, 3, 5, 7]


def test_result_layout_follows_input():
    f = sd.asarray(np.asfortranarray(np.ones((3, 4))))
    assert (f + 1)._flags()["f_contiguous"]
    assert sd.sqrt(f)._flags()["f_contiguous"]


# ------------------------------------------------------------ where, clip


def test_where():
    c = sd.asarray([True, False, True])
    assert_same(sd.where(c, sd.asarray([1, 2, 3]), 0.5), np.array([1.0, 0.5, 3.0]))
    assert sd.where(True, 1, 2) == 1
    got = sd.where(sd.asarray([[True], [False]]), sd.arange(3), sd.asarray([1j]))
    assert_same(got, np.where([[True], [False]], np.arange(3), 1j))
    with pytest.raises(TypeError):
        sd.where(sd.asarray([1, 0]), 1, 2)  # condition must be bool
    out = sd.zeros(3)
    assert sd.where(c, 1.0, 2.0, out=out) is out and out.tolist() == [1.0, 2.0, 1.0]


def test_clip():
    x = sd.asarray([-5, 0, 5, 10])
    assert sd.clip(x, min=0, max=6).tolist() == [0, 0, 5, 6]
    assert sd.clip(x, min=1).tolist() == [1, 1, 5, 10]
    assert sd.clip(x, max=1).tolist() == [-5, 0, 1, 1]
    y = sd.clip(x)
    assert y.tolist() == x.tolist() and y is not x
    assert sd.clip(x, min=sd.asarray([0, 1, 2, 3]), max=4).tolist() == [0, 1, 4, 4]
    # An array min/max broadcasts to x, never x to it: the result has x's shape (docs/api/array.md §6.4).
    m = sd.asarray([[-5, 0, 5, 10], [1, 2, 3, 4]])
    assert sd.clip(m, min=sd.asarray([0, 1, 2, 3]), max=sd.asarray([[4], [2]])).tolist() == [[0, 1, 4, 4], [1, 2, 2, 2]]
    with pytest.raises(
        ValueError,
        match=r"^clip\(\): 'min' of shape \(2, 4\) cannot be broadcast to the shape of "
        r"'x' \(4,\)$",
    ):
        sd.clip(x, min=sd.zeros((2, 4), dtype=sd.int64))
    with pytest.raises(
        ValueError,
        match=r"^clip\(\): 'max' of shape \(3,\) cannot be broadcast to the shape of "
        r"'x' \(4,\)$",
    ):
        sd.clip(x, max=sd.asarray([1, 2, 3]))
    assert sd.clip(x, min=5, max=1).tolist() == [1, 1, 1, 1]  # min > max: max wins, like numpy
    f = sd.asarray([math.nan, 1.0, 3.0, -0.0])
    got = sd.clip(f, min=0.0, max=2.0).tolist()
    assert math.isnan(got[0]) and got[1:] == [1.0, 2.0, 0.0]
    # Signed zeros follow maximum/minimum (IEEE: -0 < +0). numpy itself is not
    # consistent here: scalar bounds keep x, array bounds pick the bound.
    assert math.copysign(1, got[3]) == 1
    assert math.isnan(sd.clip(sd.asarray([1.0]), min=math.nan).tolist()[0])
    with pytest.raises(TypeError):
        sd.clip(x, min=0.5)  # keeps x's dtype: float bound into int64 (docs/api/array.md §1.5)
    with pytest.raises(TypeError):
        sd.clip(sd.asarray([1j]), min=0)
    with pytest.raises(TypeError):
        sd.clip(x, 0, 1)  # bounds are keyword-only


def test_clip_and_scalar_maximum_vector_paths_keep_the_rules():
    # Long contiguous arrays take the NEON paths: NaN and signed zeros must
    # come out exactly as minimum(maximum(x, lo), hi) element by element.
    specials = [math.nan, -0.0, 0.0, 1.0, -1.0, math.inf, -math.inf, 0.5]
    x = np.array(specials * 50)
    for lo in specials:
        for hi in (0.0, -0.0, 0.7, math.nan, math.inf):
            got = sd.clip(sd.asarray(x), min=lo, max=hi)
            ref = [sd.minimum(sd.maximum(v, lo), hi) for v in specials]  # one element at a time
            assert_same(got, np.array(ref * 50))
        assert_same(sd.maximum(sd.asarray(x), lo), np.maximum(x, lo))
        assert_same(sd.minimum(lo, sd.asarray(x)), np.minimum(lo, x))


def test_complex_abs_contract():
    import flint

    rng = np.random.default_rng(10)
    z = rng.standard_normal(2000) * np.exp(rng.uniform(-5, 5, 2000)) + 1j * rng.standard_normal(2000) * np.exp(
        rng.uniform(-5, 5, 2000)
    )
    z = np.concatenate([z, [3e-200 + 4e-200j, 1e-310 + 1e-311j, 1e300 + 1e300j, 5e-324 + 0j, 1e200 + 1e-200j]])
    got = np.asarray(sd.abs(sd.asarray(z)))
    prec = flint.ctx.prec
    flint.ctx.prec = 200
    try:
        for zi, g in zip(z, got):
            t = float((flint.arb(zi.real) ** 2 + flint.arb(zi.imag) ** 2).sqrt())
            assert ulp_distance(g, t) <= refdata.ULP_BOUND["cabs"], zi
    finally:
        flint.ctx.prec = prec
    special = np.asarray(
        sd.abs(sd.asarray([complex("inf+nanj"), complex("nan+1j"), 0j, -0.0 - 0.0j, 1.7e308 + 1e308j]))
    )
    assert (
        special[0] == np.inf and np.isnan(special[1]) and special[2] == 0 and special[3] == 0 and special[4] == np.inf
    )
