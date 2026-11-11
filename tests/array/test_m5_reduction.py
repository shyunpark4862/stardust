"""Reductions against numpy.

Integer, bool and min/max/arg* results must be exact. Floating sums are
compared with a tolerance: the summation order is deliberately not numpy's
(docs/api/array.md §7.8), only deterministic.
"""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust as sd
from .conftest import ALL_DTYPES, NP_DTYPE, assert_same

INT_MAX, INT_MIN = 2**63 - 1, -(2**63)
AXES = [None, 0, 1, -1, (0, 2), (2, 0), (), (0, 1, 2)]


def data(dtype, shape=(3, 4, 5), seed=0):
    rng = np.random.default_rng(seed)
    n = rng.integers(-3, 4, shape)
    if dtype is sd.bool:
        return n > 0
    if dtype is sd.int64:
        return n.astype(np.int64)
    if dtype is sd.float64:
        return rng.random(shape) + 0.5
    return (rng.random(shape) + 0.5) + 1j * (rng.random(shape) - 0.5)


def layouts(n):
    """The same values in several memory layouts."""
    yield n
    yield np.asfortranarray(n)
    t = np.ascontiguousarray(n.transpose(2, 0, 1)).transpose(1, 2, 0)  # permuted memory order
    yield t
    big = np.zeros(tuple(2 * s for s in n.shape), n.dtype)
    big[::2, ::2, ::2] = n
    yield big[::2, ::2, ::2]  # strided view


def compare(got, want, exact):
    want = np.asarray(want)
    if want.ndim == 0:
        assert not isinstance(got, sd.ndarray), "a full reduction must give a Python scalar (docs/api/array.md §1.2)"
        got_np = np.asarray(got, dtype=want.dtype)
        assert type(got) is {"b": bool, "i": int, "f": float, "c": complex}[want.dtype.kind]
    else:
        assert isinstance(got, sd.ndarray) and got.shape == want.shape
        assert NP_DTYPE[got.dtype] == want.dtype, (got.dtype, want.dtype)
        got_np = np.asarray(got)
    if exact:
        assert_same(sd.asarray(got_np.reshape(-1)), want.reshape(-1))
    else:
        np.testing.assert_allclose(got_np, want, rtol=1e-13, atol=0)


@pytest.mark.parametrize("axis", AXES, ids=str)
@pytest.mark.parametrize("keepdims", [False, True])
@pytest.mark.parametrize("name", ["sum", "prod", "any", "all", "mean", "min", "max"])
def test_reductions_match_numpy(name, axis, keepdims, dtype):
    n = data(dtype)
    fn = getattr(sd, name)
    if name in ("min", "max") and dtype is sd.complex128:
        with pytest.raises(TypeError):
            fn(sd.asarray(n), axis=axis)
        return
    want = getattr(np, name)(n, axis=axis, keepdims=keepdims)
    exact = name in ("any", "all", "min", "max") or dtype in (sd.bool, sd.int64) and name != "mean"
    for layout in layouts(n):
        compare(fn(sd.asarray(layout), axis=axis, keepdims=keepdims), want, exact)


@pytest.mark.parametrize("axis", AXES, ids=str)
@pytest.mark.parametrize("correction", [0, 1, 0.5])
@pytest.mark.parametrize("name", ["var", "std"])
def test_var_std_match_numpy(name, axis, correction, dtype):
    n = data(dtype)
    if axis == () and correction >= 1:
        want = np.full(n.shape, np.nan)  # one element per output: N - correction <= 0 -> nan
    else:
        want = getattr(np, name)(n, axis=axis, ddof=correction)
    for layout in layouts(n):
        got = getattr(sd, name)(sd.asarray(layout), axis=axis, correction=correction)
        compare(got, np.asarray(want, dtype=np.float64), exact=False)


def test_methods_are_the_functions():
    n = data(sd.float64)
    x = sd.asarray(n)
    for name in ("sum", "prod", "min", "max", "mean", "var", "std", "argmin", "argmax", "any", "all"):
        assert_same(getattr(x, name)(axis=1), np.asarray(getattr(sd, name)(x, axis=1)))
    assert x.sum() == sd.sum(x)
    with pytest.raises(TypeError):
        x.sum(1)  # axis is keyword-only


# ----------------------------------------------------------- result dtypes

def test_result_dtypes():
    b, i, f, c = (sd.asarray(data(d)) for d in ALL_DTYPES)
    assert sd.sum(b, axis=0).dtype is sd.int64  # docs/api/array.md §1.4
    assert sd.prod(b, axis=0).dtype is sd.int64
    assert sd.sum(i, axis=0).dtype is sd.int64
    assert sd.max(b, axis=0).dtype is sd.bool
    for x in (b, i):
        assert sd.mean(x, axis=0).dtype is sd.float64
        assert sd.var(x, axis=0).dtype is sd.float64
    assert sd.mean(c, axis=0).dtype is sd.complex128
    assert sd.var(c, axis=0).dtype is sd.float64 and sd.std(c, axis=0).dtype is sd.float64
    assert sd.any(f, axis=0).dtype is sd.bool and sd.argmax(f, axis=0).dtype is sd.int64


def test_dtype_argument():
    f = sd.asarray([1.9, 2.9, -0.5])
    assert sd.sum(f, dtype=sd.int64) == 1 + 2 + 0  # explicit: each element truncated first
    assert sd.sum(sd.asarray([1, 2]), dtype=sd.float64) == 3.0
    assert type(sd.prod(sd.asarray([2, 3]), dtype=sd.complex128)) is complex
    with pytest.raises(TypeError):
        sd.sum(f, dtype=sd.bool)  # no bool accumulation (docs/api/array.md §7.5)
    with pytest.raises(TypeError):
        sd.sum(sd.asarray([1j]), dtype=sd.float64)  # docs/api/array.md §1.5


def test_integer_sum_wraps():
    assert sd.sum(sd.asarray([INT_MAX, 1])) == INT_MIN
    assert sd.prod(sd.asarray([2**32, 2**32])) == 0


# ------------------------------------------------------------- empty arrays

def test_empty_rules():
    e = sd.zeros((0,))
    assert sd.sum(e) == 0 and sd.prod(e) == 1
    assert sd.any(e) is False and sd.all(e) is True
    for fn in (sd.mean, sd.var, sd.std):
        assert math.isnan(fn(e))
    for fn in (sd.min, sd.max, sd.argmin, sd.argmax):
        with pytest.raises(ValueError):
            fn(e)
    z = sd.zeros((3, 0))
    assert sd.sum(z, axis=1).tolist() == [0, 0, 0]
    with pytest.raises(ValueError):
        sd.max(z, axis=1)  # every output would fold nothing
    assert sd.max(z, axis=0).shape == (0,)  # nothing to compute: fine
    with pytest.raises(ValueError):
        sd.argmax(z, axis=1)
    assert sd.argmax(z, axis=0).shape == (0,)


def test_sum_signed_zero_like_numpy():
    z = sd.asarray([-0.0, -0.0])
    assert math.copysign(1, sd.sum(z)) == 1  # starts from +0, as numpy
    assert [math.copysign(1, v) for v in sd.cumulative_sum(z).tolist()] == [-1, -1]


# ---------------------------------------------------------------- NaN rules

def test_nan_rules():
    x = sd.asarray([1.0, math.nan, 3.0, math.nan])
    assert math.isnan(sd.max(x)) and math.isnan(sd.min(x))
    assert sd.argmax(x) == 1 and sd.argmin(x) == 1  # the first NaN
    assert math.isnan(sd.sum(x)) and math.isnan(sd.mean(x))
    assert sd.any(sd.asarray([0.0, math.nan])) is True  # nan != 0
    m = sd.asarray([[1.0, math.nan], [2.0, 0.5]])
    assert sd.argmax(m, axis=1).tolist() == [1, 0]


def test_ties_give_the_first_index():
    x = sd.asarray([3, 1, 3, 1])
    assert sd.argmax(x) == 0 and sd.argmin(x) == 1
    t = sd.asarray(np.array([[1, 5], [5, 1]]).T)  # transposed: flat index is C order of the view
    assert sd.argmax(t) == int(np.argmax(np.asarray(t)))


@pytest.mark.parametrize("axis", [None, 0, 1, 2, -1])
@pytest.mark.parametrize("keepdims", [False, True])
def test_argmin_argmax_match_numpy(axis, keepdims, dtype):
    if dtype is sd.complex128:
        with pytest.raises(TypeError):
            sd.argmax(sd.asarray(data(dtype)))
        return
    n = data(dtype)  # small integers: many ties
    for layout in layouts(n):
        for name in ("argmin", "argmax"):
            got = getattr(sd, name)(sd.asarray(layout), axis=axis, keepdims=keepdims)
            compare(got, getattr(np, name)(n, axis=axis, keepdims=keepdims), exact=True)


def test_argmax_axis_must_be_int():
    with pytest.raises(TypeError):
        sd.argmax(sd.zeros((2, 2)), axis=(0, 1))  # docs/api/array.md §7.2


# ------------------------------------------------------------- var details

def test_var_degrees_of_freedom():
    one = sd.asarray([5.0])
    assert sd.var(one) == 0.0
    assert math.isnan(sd.var(one, correction=1))  # N - correction <= 0 -> nan (docs/api/array.md §7.4)
    assert math.isnan(sd.std(sd.asarray([1.0, 2.0]), correction=2))
    assert sd.var(sd.asarray([1.0, 2.0, 3.0, 4.0]), correction=1) == pytest.approx(5 / 3)
    with pytest.raises(TypeError):
        sd.var(one, correction=True)


def test_var_complex_is_mean_squared_modulus():
    z = np.array([1 + 1j, -1 - 1j, 2j])
    assert sd.var(sd.asarray(z)) == pytest.approx(np.var(z))


# --------------------------------------------------------------- cumulative

@pytest.mark.parametrize("name,np_name", [("cumulative_sum", "cumulative_sum"), ("cumulative_prod", "cumulative_prod")])
@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("include_initial", [False, True])
def test_cumulative_match_numpy(name, np_name, axis, include_initial, dtype):
    n = data(dtype)
    want = getattr(np, np_name)(n, axis=axis, include_initial=include_initial)
    if dtype is sd.bool:
        want = want.astype(np.int64)
    for layout in layouts(n):
        got = getattr(sd, name)(sd.asarray(layout), axis=axis, include_initial=include_initial)
        compare(got, want, exact=dtype in (sd.bool, sd.int64))


def test_cumulative_axis_rules():
    v = sd.asarray([1, 2, 3])
    assert sd.cumulative_sum(v).tolist() == [1, 3, 6]  # 1-D: axis may be omitted
    with pytest.raises(ValueError):
        sd.cumulative_sum(sd.zeros((2, 2)))  # ndim > 1 needs axis (docs/api/array.md §7.7)
    with pytest.raises(TypeError):
        sd.cumulative_sum(v, axis=(0,))
    assert sd.cumulative_sum(sd.zeros(0), include_initial=True).tolist() == [0.0]
    assert sd.cumulative_prod(sd.asarray([True, True])).dtype is sd.int64
    with pytest.raises(TypeError):
        sd.cumulative_sum(v, dtype=sd.bool)


# --------------------------------------------------------- axes and inputs

def test_axis_errors_and_inputs():
    x = sd.zeros((2, 3))
    with pytest.raises(ValueError):
        sd.sum(x, axis=2)
    with pytest.raises(ValueError):
        sd.sum(x, axis=(0, 0))
    with pytest.raises(TypeError):
        sd.sum([1, 2, 3])  # arrays only
    with pytest.raises(TypeError):
        sd.sum(x, 0)


def test_full_reduction_types():
    assert type(sd.sum(sd.asarray([1, 2]))) is int
    assert type(sd.max(sd.asarray([True]))) is bool
    assert type(sd.mean(sd.asarray([1, 2]))) is float
    assert type(sd.argmax(sd.asarray([1, 2]))) is int
    k = sd.sum(sd.ones((2, 3)), keepdims=True)
    assert isinstance(k, sd.ndarray) and k.shape == (1, 1)


def test_deterministic():
    n = np.random.default_rng(3).standard_normal((257, 129))
    x = sd.asarray(n)
    for fn in (sd.sum, sd.mean, sd.var):
        first = np.asarray(fn(x, axis=0))
        for _ in range(3):
            assert np.array_equal(np.asarray(fn(x, axis=0)).view(np.int64), first.view(np.int64))


@given(st.data())
def test_random_shapes_and_axes(data_):
    shape = data_.draw(hnp.array_shapes(min_dims=1, max_dims=4, min_side=0, max_side=4))
    n = data_.draw(hnp.arrays(np.int64, shape, elements=st.integers(-100, 100)))
    axis = data_.draw(st.one_of(st.none(), st.integers(-len(shape), len(shape) - 1),
                                st.lists(st.integers(0, len(shape) - 1), unique=True).map(tuple)))
    keepdims = data_.draw(st.booleans())
    x = sd.asarray(n)
    for name in ("sum", "prod", "any", "all"):
        compare(getattr(sd, name)(x, axis=axis, keepdims=keepdims),
                getattr(np, name)(n, axis=axis, keepdims=keepdims), exact=True)
    ax = axis if axis is None or isinstance(axis, tuple) else (axis,)
    reduced = range(len(shape)) if ax is None else [a % len(shape) for a in ax]
    if all(shape[a] > 0 for a in reduced):
        compare(sd.max(x, axis=axis, keepdims=keepdims), np.max(n, axis=axis, keepdims=keepdims), exact=True)


# ------------------------------------------------------ vector kernel paths

@given(st.lists(st.sampled_from([0.0, -0.0, 1.0, 2.0, -3.0, math.inf, -math.inf, math.nan]), min_size=1, max_size=40))
def test_vector_folds_match_numpy(values):
    # Long enough lists reach the NEON paths (pairs of lanes, 8-wide folds);
    # ties, NaNs and signed zeros land in every lane position.
    n = np.array(values)
    x = sd.asarray(n)
    for name in ("argmax", "argmin"):
        assert getattr(sd, name)(x) == getattr(np, name)(n)
    for name in ("max", "min"):
        got, want = getattr(sd, name)(x), getattr(np, name)(n)
        assert (math.isnan(got) and math.isnan(want)) or (got == want and math.copysign(1, got) == math.copysign(1, want))


def test_pairwise_sum_accuracy():
    x = np.full(10**6, 0.1) + np.random.default_rng(7).random(10**6) * 1e-3
    exact = math.fsum(x.tolist())
    assert abs(sd.sum(sd.asarray(x)) - exact) / exact < 1e-15  # a running sum would be ~1e-14
    c = x + 1j * x[::-1]
    got = sd.sum(sd.asarray(c))
    assert abs(got.real - exact) / exact < 1e-15 and abs(got.imag - exact) / exact < 1e-15


@pytest.mark.parametrize("where", [0, 1, 4095, 4096, 4097, 99_999])
def test_any_all_stop_early_but_stay_right(where):
    z = np.zeros(100_000, bool)
    z[where] = True
    o = ~z
    for arr in (z, z[::-1], z.reshape(400, 250), z.reshape(400, 250).T):
        assert sd.any(sd.asarray(arr)) is True
        assert sd.all(sd.asarray(~arr)) is False
        assert np.array_equal(np.asarray(sd.any(sd.asarray(z.reshape(400, 250)), axis=1)), z.reshape(400, 250).any(axis=1))
    assert sd.any(sd.asarray(np.zeros(5000, bool))) is False and sd.all(sd.asarray(o | True)) is True
    assert sd.any(sd.asarray(np.where(z, 2.5, 0.0))) is True  # non-bool input: through the cast


@pytest.mark.parametrize("n", [1, 15, 16, 17, 31, 32, 33, 1000])
def test_int_min_max_vector_fold_finds_every_position(n):
    # The NEON fold keeps 16 lanes; the extreme may sit in any lane or the tail.
    for pos in sorted({0, min(1, n - 1), n // 2, n - 1, min(16, n - 1)}):
        x = np.arange(n, dtype=np.int64) * 7 - 3 * n
        x[pos] = INT_MAX
        assert sd.max(sd.asarray(x)) == INT_MAX
        x[pos] = INT_MIN
        assert sd.min(sd.asarray(x)) == INT_MIN
        assert sd.max(sd.asarray(x)) == x.max()


# Float prod and long cumulative scans run in fixed blocks
# (docs/design/array.md §6.6, §6.7); the results are compared with numpy
# within rounding, and must not depend on the layout of the same lane's
# memory more than rounding does.

@pytest.mark.parametrize("n", [0, 1, 7, 8, 9, 15, 16, 17, 4095, 4096, 4097, 8191, 8192, 8193, 40_000])
def test_blocked_float_prod_is_close(n):
    rng = np.random.default_rng(n)
    x = 1.0 + (rng.random(n) - 0.5) * 1e-3
    c = np.exp(1j * rng.standard_normal(n) * 1e-2)
    assert sd.prod(sd.asarray(x)) == pytest.approx(np.prod(x), rel=1e-12, abs=0)
    assert sd.prod(sd.asarray(c)) == pytest.approx(np.prod(c), rel=1e-12, abs=0)
    assert sd.prod(sd.asarray(x[::-1])) == pytest.approx(np.prod(x), rel=1e-12, abs=0)
    k = np.resize(np.array([1, 2, 3, -1, 1, 1, 5]), n)
    assert sd.prod(sd.asarray(k)) == np.prod(k)  # int: exact (wrapping, order-free)


def test_blocked_prod_special_values():
    ones = [1.0] * 30
    assert math.isnan(sd.prod(sd.asarray([2.0, math.inf, 0.0] + ones)))
    assert math.isnan(sd.prod(sd.asarray(ones + [math.nan])))
    assert sd.prod(sd.asarray(ones + [-0.0])) == 0.0
    assert math.copysign(1.0, sd.prod(sd.asarray(ones + [-0.0]))) == -1.0
    assert sd.prod(sd.asarray(ones + [math.inf, -2.0])) == -math.inf


@pytest.mark.parametrize("n", [8191, 8192, 8193, 16_384, 16_385, 4096 * 4 * 3 + 17, 100_000])
@pytest.mark.parametrize("name", ["cumulative_sum", "cumulative_prod"])
def test_blocked_scans_are_close(n, name):
    rng = np.random.default_rng(n)
    x = 1.0 + (rng.random(n) - 0.5) * 1e-3
    c = x * np.exp(1j * rng.standard_normal(n) * 1e-3)
    i = rng.integers(-5, 6, n)
    fn, ref = getattr(sd, name), getattr(np, name)
    for arr in (x, x[::-1], c):
        np.testing.assert_allclose(np.asarray(fn(sd.asarray(arr))), ref(arr), rtol=1e-12)
    m = np.stack([x, x[::-1], x * 1.0001]).T  # three long strided lanes
    np.testing.assert_allclose(np.asarray(fn(sd.asarray(m), axis=0)), ref(m, axis=0), rtol=1e-12)
    got = np.asarray(fn(sd.asarray(x), include_initial=True))
    np.testing.assert_allclose(got, ref(x, include_initial=True), rtol=1e-12)
    if name == "cumulative_sum":
        assert np.array_equal(np.asarray(fn(sd.asarray(i))), ref(i))  # int: exact


def test_blocked_scan_keeps_the_first_element():
    x = np.zeros(20_000)
    x[0] = -0.0
    got = np.asarray(sd.cumulative_sum(sd.asarray(x)))
    assert math.copysign(1.0, got[0]) == -1.0  # the first output is x[0] itself
    x[4096] = math.nan
    got = np.asarray(sd.cumulative_sum(sd.asarray(x)))
    assert not np.isnan(got[:4096]).any() and np.isnan(got[4096:]).all()
