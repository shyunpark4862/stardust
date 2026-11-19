"""quantile / median, the nan* reductions, cov and corrcoef against numpy (T0).

quantile, median and their nan forms are numpy's bits: the same order
statistics and numpy's interpolation formula (docs/api/array.md §7.10). The
result is always float64. The sums inside nanmean, nanvar and cov use our
summation order, so those are compared with a tolerance -- and with the plain
reduction's bits when there is no NaN.
"""

import math
import warnings

import numpy as np
import pytest

import stardust as sd

from .conftest import assert_same

METHODS = [
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]
QS = [0.0, 1e-300, 0.01, 0.1, 0.25, 0.3, 1 / 3, 0.5, 0.6, 2 / 3, 0.75, 0.9, 0.99, 1 - 1e-16, 1.0]
NAN = float("nan")


def np_quiet(f, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kw)


def check(got, want):
    """`got` has the bits of numpy's `want`, always as float64."""
    want = np.asarray(want, dtype=np.float64)
    if want.ndim == 0:
        assert type(got) is float
        assert_same(sd.asarray([got]), want.reshape(1))
    else:
        assert_same(got, want)


def samples():
    rng = np.random.default_rng(1)
    yield rng.random(1)
    yield rng.random(2)
    yield rng.random(7)
    yield rng.random(10)
    yield rng.standard_normal(101)
    yield rng.integers(0, 5, 40).astype(np.float64)  # many ties
    yield np.array([-np.inf, -1.0, 0.0, np.inf])
    yield np.array([1e308, -1e308, 1e308, 5e-324])


# ------------------------------------------------------------------ quantile


@pytest.mark.parametrize("method", METHODS)
def test_quantile_bits_every_method(method):
    for x in samples():
        a = sd.asarray(x)
        for q in QS:
            check(sd.quantile(a, q, method=method), np_quiet(np.quantile, x, q, method=method))
        check(sd.quantile(a, sd.asarray(QS), method=method), np_quiet(np.quantile, x, QS, method=method))


@pytest.mark.parametrize("method", METHODS)
def test_quantile_signed_zero_ties_equal_numpy_as_values(method):
    # Which of -0.0 / +0.0 a tie gives is not specified (docs/api/array.md §7.10).
    x = np.array([-0.0, 0.0, -0.0, 1.0])
    got = np.asarray(sd.quantile(sd.asarray(x), sd.asarray(QS), method=method))
    assert np.array_equal(got, np.quantile(x, QS, method=method).astype(np.float64))


@pytest.mark.parametrize("method", METHODS)
def test_quantile_int_and_bool_are_float64_with_numpys_values(method):
    x = np.array([7, -3, 2, 2, 9, 0, 11, 5])
    got = sd.quantile(sd.asarray(x), sd.asarray(QS), method=method)
    check(got, np.quantile(x, QS, method=method).astype(np.float64))
    b = np.array([True, False, True, True])
    check(sd.quantile(sd.asarray(b), 0.5, method=method), np.quantile(b.astype(np.float64), 0.5, method=method))


@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2), (2, 0), (0, 1, 2), ()])
@pytest.mark.parametrize("keepdims", [False, True])
def test_quantile_axes_layouts_keepdims(axis, keepdims):
    rng = np.random.default_rng(2)
    x = rng.standard_normal((4, 5, 6))
    for layout in (x, np.asfortranarray(x), x.transpose(2, 0, 1).copy().transpose(1, 2, 0), x[::-1, :, ::2]):
        a = sd.asarray(layout)
        for q in (0.3, [0.1, 0.5, 0.95]):
            sq = q if isinstance(q, float) else sd.asarray(q)
            for method in ("linear", "hazen", "nearest", "midpoint"):
                want = np.quantile(layout, q, axis=axis, method=method, keepdims=keepdims)
                check(sd.quantile(a, sq, axis=axis, method=method, keepdims=keepdims), want)


def test_quantile_nan_lane_is_nan():
    x = np.array([[1.0, NAN, 3.0], [4.0, 5.0, 6.0]])
    for method in METHODS:
        check(sd.quantile(sd.asarray(x), 0.5, axis=1, method=method), np.quantile(x, 0.5, axis=1, method=method))
    assert math.isnan(sd.quantile(sd.asarray(x), 0.5))


def test_quantile_empty_lane_is_nan():
    check(sd.quantile(sd.zeros((0,)), 0.5), np.float64(NAN))
    check(sd.quantile(sd.zeros((0, 3)), sd.asarray([0.2, 0.7]), axis=0), np.full((2, 3), NAN))
    check(sd.quantile(sd.zeros((3, 0)), 0.5, axis=0), np.zeros((0,)))
    check(sd.median(sd.zeros((0,))), np.float64(NAN))
    for method in METHODS:
        assert math.isnan(sd.quantile(sd.zeros((0,), dtype=sd.int64), 0.5, method=method))


def test_quantile_errors():
    x = sd.asarray([1.0, 2.0])
    for q in (-0.1, 1.1, NAN):
        with pytest.raises(ValueError, match=r"quantile\(\): 'q' must be in the range \[0, 1\]"):
            sd.quantile(x, q)
    with pytest.raises(ValueError, match="'q' must be in the range"):
        sd.quantile(x, sd.asarray([0.5, 2.0]))
    with pytest.raises(ValueError, match="'q' must be a scalar or 1-D"):
        sd.quantile(x, sd.asarray([[0.5]]))
    # the number-argument form with sd.ndarray (API.md §2.1); an int is a number too
    for bad, shown in [([0.5], "list"), (None, "NoneType"), (True, "bool")]:
        with pytest.raises(
            TypeError, match=rf"^quantile\(\): 'q' must be an int, a float or an sd.ndarray, not {shown}$"
        ):
            sd.quantile(x, bad)
    assert sd.quantile(x, 1) == sd.quantile(x, 1.0)
    with pytest.raises(TypeError, match="'q' must be real"):
        sd.quantile(x, sd.asarray([0.5j]))
    # The string-option message (API.md §2.1): the same list for a wrong value and a wrong type.
    methods = r"'method' must be 'inverted_cdf', 'averaged_inverted_cdf', .*, 'midpoint' or 'nearest', not "
    with pytest.raises(ValueError, match=r"^quantile\(\): " + methods + "'foo'$"):
        sd.quantile(x, 0.5, method="foo")
    with pytest.raises(TypeError, match=r"^quantile\(\): " + methods + "int$"):
        sd.quantile(x, 0.5, method=1)
    with pytest.raises(TypeError, match=r"^nanquantile\(\): " + methods + "NoneType$"):
        sd.nanquantile(x, 0.5, method=None)
    with pytest.raises(TypeError, match="not supported for complex128"):
        sd.quantile(sd.asarray([1j]), 0.5)
    with pytest.raises(TypeError, match="'keepdims' must be True or False"):
        sd.quantile(x, 0.5, keepdims=1)
    with pytest.raises(TypeError):
        sd.quantile([1.0], 0.5)


# -------------------------------------------------------------------- median


def test_median_bits():
    for x in samples():
        check(sd.median(sd.asarray(x)), np.median(x))
    x = np.random.default_rng(3).standard_normal((6, 7, 8))
    for axis in (None, 0, 1, 2, (0, 2)):
        for keepdims in (False, True):
            check(sd.median(sd.asarray(x), axis=axis, keepdims=keepdims), np.median(x, axis=axis, keepdims=keepdims))
    i = np.array([[3, 1, 2, 10], [-5, 7, 7, 0]])
    check(sd.median(sd.asarray(i), axis=1), np.median(i, axis=1))
    check(sd.median(sd.asarray([NAN, 1.0])), np.float64(NAN))


@pytest.mark.parametrize("length", [1, 2, 3, 4, 5, 6, 7, 10, 16, 31, 33, 100, 300])
def test_median_short_lanes(length):
    # Lanes under 6 take an insertion sort, longer ones the NEON sort (stack buffer up to 256).
    x = np.random.default_rng(length).standard_normal((50, length))
    check(sd.median(sd.asarray(x), axis=1), np.median(x, axis=1))
    check(
        sd.quantile(sd.asarray(x), sd.asarray(QS), axis=1, method="hazen"), np.quantile(x, QS, axis=1, method="hazen")
    )


def test_median_signed_zero_like_numpy():
    # numpy's median is mean() of the middle values: a sum that starts at +0.0.
    check(sd.median(sd.asarray([-0.0])), np.median(np.array([-0.0])))
    check(sd.median(sd.asarray([-0.0, -0.0])), np.median(np.array([-0.0, -0.0])))


# ------------------------------------------------------ nanquantile, nanmedian


def nan_data(seed=4):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((5, 9))
    x[rng.random(x.shape) < 0.3] = np.nan
    x[2] = np.nan  # an all-NaN row
    return x


@pytest.mark.parametrize("method", METHODS)
def test_nanquantile_bits(method):
    x = nan_data()
    a = sd.asarray(x)
    for axis in (None, 0, 1):
        for q in (0.0, 0.3, 0.5, 1.0):
            check(
                sd.nanquantile(a, q, axis=axis, method=method), np_quiet(np.nanquantile, x, q, axis=axis, method=method)
            )
        check(
            sd.nanquantile(a, sd.asarray(QS), axis=axis, method=method),
            np_quiet(np.nanquantile, x, QS, axis=axis, method=method),
        )


def test_nanmedian_bits():
    x = nan_data()
    for axis in (None, 0, 1):
        for keepdims in (False, True):
            check(
                sd.nanmedian(sd.asarray(x), axis=axis, keepdims=keepdims),
                np_quiet(np.nanmedian, x, axis=axis, keepdims=keepdims),
            )


def test_nan_quantiles_without_nan_equal_the_plain_ones():
    x = np.random.default_rng(5).standard_normal((7, 8))
    a = sd.asarray(x)
    check(sd.nanmedian(a, axis=0), np.asarray(sd.median(a, axis=0)))
    check(sd.nanquantile(a, 0.37, axis=1), np.asarray(sd.quantile(a, 0.37, axis=1)))
    i = sd.asarray([[1, 4, 2], [9, 9, 0]])
    check(sd.nanquantile(i, 0.5, axis=1), np.asarray(sd.quantile(i, 0.5, axis=1)))


def test_nan_quantile_all_nan_and_empty_are_nan_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert math.isnan(sd.nanmedian(sd.asarray([NAN, NAN])))
        assert math.isnan(sd.nanquantile(sd.asarray([NAN]), 0.5))
        assert math.isnan(sd.nanmedian(sd.zeros((0,))))


# ------------------------------------------------------------ nan reductions

FLOAT_FUNCS = {
    "nansum": (sd.nansum, np.nansum, sd.sum),
    "nanprod": (sd.nanprod, np.nanprod, sd.prod),
    "nanmean": (sd.nanmean, np.nanmean, sd.mean),
}


@pytest.mark.parametrize("name", list(FLOAT_FUNCS))
@pytest.mark.parametrize("axis", [None, 0, 1, (0, 1)])
def test_nan_folds_against_numpy(name, axis):
    ours, theirs, plain = FLOAT_FUNCS[name]
    x = nan_data() * 0.5 + 1.0
    got = ours(sd.asarray(x), axis=axis)
    want = np_quiet(theirs, x, axis=axis)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-14, atol=0)
    # Without NaN: the plain reduction's bits.
    y = np.random.default_rng(6).random((6, 7)) + 0.5
    assert_same(sd.asarray([ours(sd.asarray(y))]), np.array([plain(sd.asarray(y))]))
    assert_same(ours(sd.asarray(y), axis=0), np.asarray(plain(sd.asarray(y), axis=0)))


def test_nan_folds_all_nan_and_empty():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        allnan = sd.asarray([[NAN, NAN], [1.0, NAN]])
        check(sd.nanmean(allnan, axis=1), np.array([NAN, 1.0]))
        check(sd.nansum(allnan, axis=1), np.array([0.0, 1.0]))
        check(sd.nanprod(allnan, axis=1), np.array([1.0, 1.0]))
        check(sd.nanvar(allnan, axis=1), np.array([NAN, 0.0]))
        check(sd.nanstd(allnan, axis=1), np.array([NAN, 0.0]))
        check(sd.nanmin(allnan, axis=1), np.array([NAN, 1.0]))
        check(sd.nanmax(allnan, axis=1), np.array([NAN, 1.0]))
        assert math.isnan(sd.nanmean(sd.zeros((0,))))
        assert sd.nansum(sd.zeros((0,))) == 0.0


@pytest.mark.parametrize("correction", [0, 1, 1.5])
@pytest.mark.parametrize("axis", [None, 0, 1])
def test_nanvar_nanstd(correction, axis):
    x = nan_data()
    for ours, theirs in ((sd.nanvar, np.nanvar), (sd.nanstd, np.nanstd)):
        got = np.asarray(ours(sd.asarray(x), axis=axis, correction=correction))
        want = np_quiet(theirs, x, axis=axis, ddof=correction)
        np.testing.assert_allclose(got, want, rtol=1e-13, atol=0)
    y = np.random.default_rng(7).random((5, 6))
    for ours, plain in ((sd.nanvar, sd.var), (sd.nanstd, sd.std)):
        assert_same(
            ours(sd.asarray(y), axis=axis if axis is not None else 0, correction=correction),
            np.asarray(plain(sd.asarray(y), axis=axis if axis is not None else 0, correction=correction)),
        )


def test_nan_complex():
    z = np.array([1 + 2j, complex(NAN, 0), 3 - 1j, complex(0, NAN)])
    np.testing.assert_allclose(sd.nansum(sd.asarray(z)), np.nansum(z), rtol=1e-15)
    np.testing.assert_allclose(sd.nanmean(sd.asarray(z)), np.nanmean(z), rtol=1e-15)
    np.testing.assert_allclose(sd.nanvar(sd.asarray(z)), np.nanvar(z), rtol=1e-15)
    with pytest.raises(TypeError, match="not supported for complex128"):
        sd.nanmin(sd.asarray(z))
    with pytest.raises(TypeError, match="not supported for complex128"):
        sd.nanargmax(sd.asarray(z))


def test_nanmin_nanmax_exact():
    x = nan_data()
    x[0, 0] = -0.0
    x[0, 1] = 0.0
    for axis in (None, 0, 1):
        for ours, theirs in ((sd.nanmin, np.nanmin), (sd.nanmax, np.nanmax)):
            check(ours(sd.asarray(x), axis=axis), np_quiet(theirs, x, axis=axis))
    big = np.random.default_rng(8).standard_normal(100_000)
    big[::7] = np.nan
    check(sd.nanmin(sd.asarray(big)), np.nanmin(big))
    check(sd.nanmax(sd.asarray(big)), np.nanmax(big))
    with pytest.raises(ValueError, match=r"nanmin\(\): undefined for an empty array"):
        sd.nanmin(sd.zeros((0,)))
    assert sd.nanmin(sd.asarray([3, 1, 2])) == 1 and type(sd.nanmin(sd.asarray([3, 1, 2]))) is int


def test_nanargmin_nanargmax():
    x = np.array([[NAN, 2.0, 1.0, 1.0], [5.0, NAN, 5.0, -np.inf], [np.inf, NAN, np.inf, NAN]])
    for axis in (None, 1):
        for ours, theirs in ((sd.nanargmin, np.nanargmin), (sd.nanargmax, np.nanargmax)):
            got = ours(sd.asarray(x), axis=axis)
            want = theirs(x, axis=axis)
            assert np.array_equal(np.asarray(got), want), (ours, axis)
    with pytest.raises(ValueError, match=r"nanargmin\(\): All-NaN slice encountered"):
        sd.nanargmin(sd.asarray([[1.0, NAN], [NAN, NAN]]), axis=1)
    with pytest.raises(ValueError, match="All-NaN slice encountered"):
        sd.nanargmax(sd.asarray([NAN]))
    assert sd.nanargmax(sd.asarray([1, 3, 2])) == 1


@pytest.mark.parametrize("axis", [0, 1])
def test_nancumulative(axis):
    x = nan_data()
    for ours, theirs in ((sd.nancumulative_sum, np.nancumsum), (sd.nancumulative_prod, np.nancumprod)):
        got = np.asarray(ours(sd.asarray(x), axis=axis))
        np.testing.assert_allclose(got, theirs(x, axis=axis), rtol=1e-13, atol=0)
    got = sd.nancumulative_sum(sd.asarray([NAN, 1.0, NAN, 2.0]), include_initial=True)
    check(got, np.array([0.0, 0.0, 1.0, 1.0, 3.0]))


def test_nan_functions_on_integers_are_the_plain_ones():
    i = sd.asarray([[3, 1, 2], [0, 5, 4]])
    for nanf, plain in (
        (sd.nansum, sd.sum),
        (sd.nanprod, sd.prod),
        (sd.nanmin, sd.min),
        (sd.nanmax, sd.max),
        (sd.nanargmin, sd.argmin),
        (sd.nanargmax, sd.argmax),
        (sd.nanmean, sd.mean),
        (sd.nanvar, sd.var),
        (sd.nanstd, sd.std),
    ):
        assert nanf(i) == plain(i)
        assert np.array_equal(np.asarray(nanf(i, axis=1)), np.asarray(plain(i, axis=1)))


# --------------------------------------------------------------- cov, corrcoef


@pytest.mark.parametrize("correction", [0, 1, 2])
def test_cov(correction):
    rng = np.random.default_rng(9)
    x = rng.standard_normal((4, 30))
    np.testing.assert_allclose(
        np.asarray(sd.cov(sd.asarray(x), correction=correction)), np.cov(x, ddof=correction), rtol=1e-13, atol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(sd.cov(sd.asarray(x.T), rowvar=False, correction=correction)),
        np.cov(x.T, rowvar=False, ddof=correction),
        rtol=1e-13,
        atol=1e-15,
    )
    one = sd.cov(sd.asarray(x[0]), correction=correction)
    assert type(one) is float
    np.testing.assert_allclose(one, np.cov(x[0], ddof=correction), rtol=1e-13)
    z = x[:2] + 1j * x[2:]
    np.testing.assert_allclose(
        np.asarray(sd.cov(sd.asarray(z), correction=correction)), np.cov(z, ddof=correction), rtol=1e-13, atol=1e-15
    )
    i = rng.integers(-5, 5, (3, 8))
    np.testing.assert_allclose(np.asarray(sd.cov(sd.asarray(i))), np.cov(i, ddof=0), rtol=1e-13, atol=1e-15)


def test_cov_default_correction_is_zero_and_too_few_is_nan():
    x = sd.asarray([[1.0, 2.0, 4.0], [0.0, 1.0, 1.0]])
    np.testing.assert_allclose(np.asarray(sd.cov(x)), np.cov(np.asarray(x), ddof=0), rtol=1e-14)
    assert np.isnan(np.asarray(sd.cov(x, correction=3))).all()
    assert math.isnan(sd.cov(sd.asarray([1.0]), correction=1))


def test_cov_rowvar_false_always_transposes_and_no_variables_is_empty():
    x = np.array([[1.0, 2.0, 4.0]])
    np.testing.assert_allclose(np.asarray(sd.cov(sd.asarray(x), rowvar=False)), np.cov(x, rowvar=False, ddof=0))
    assert sd.cov(sd.zeros((0, 3))).shape == (0, 0)
    assert sd.corrcoef(sd.zeros((3, 0)), rowvar=False).shape == (0, 0)


def test_corrcoef():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((5, 40))
    x[1] = 2 * x[0] + 1  # perfectly correlated: clipped to exactly 1
    got = np.asarray(sd.corrcoef(sd.asarray(x)))
    np.testing.assert_allclose(got, np.corrcoef(x), rtol=1e-13, atol=1e-15)
    assert np.all(np.abs(got) <= 1.0)
    np.testing.assert_allclose(
        np.asarray(sd.corrcoef(sd.asarray(x.T), rowvar=False)), np.corrcoef(x), rtol=1e-13, atol=1e-15
    )
    assert sd.corrcoef(sd.asarray([1.0, 2.0, 3.0])) == 1.0
    assert math.isnan(sd.corrcoef(sd.asarray([1.0, 1.0])))
    z = x[:2] + 1j * x[2:4]
    np.testing.assert_allclose(np.asarray(sd.corrcoef(sd.asarray(z))), np.corrcoef(z), rtol=1e-13, atol=1e-15)


@pytest.mark.parametrize("shape", [(40, 100), (40, 1000), (3, 5000)])  # syrk (n <= 4p) and gemm paths
def test_cov_exactly_symmetric(shape):
    rng = np.random.default_rng(16)
    x = rng.standard_normal(shape)
    c = np.asarray(sd.cov(sd.asarray(x)))
    assert np.array_equal(c, c.T)
    np.testing.assert_allclose(c, np.cov(x, ddof=0), rtol=1e-12, atol=1e-14)
    z = x + 1j * rng.standard_normal(shape)
    cz = np.asarray(sd.cov(sd.asarray(z)))
    assert np.array_equal(cz, cz.conj().T) and np.all(cz.diagonal().imag == 0)
    np.testing.assert_allclose(cz, np.cov(z, ddof=0), rtol=1e-12, atol=1e-14)


def test_cov_errors():
    with pytest.raises(ValueError, match=r"cov\(\): 'x' must be 1-D or 2-D"):
        sd.cov(sd.zeros((2, 2, 2)))
    with pytest.raises(TypeError, match="'rowvar' must be True or False"):
        sd.cov(sd.zeros((2, 2)), rowvar=0)
    with pytest.raises(TypeError, match="'correction' must be an int or a float"):
        sd.cov(sd.zeros((2, 2)), correction="1")


# ---------------------------------------------------------------- threads


def test_quantile_same_bits_at_every_thread_count():
    x = sd.asarray(np.random.default_rng(11).standard_normal((300, 400)))
    before = sd.get_num_threads()
    try:
        results = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            results.append(np.asarray(sd.quantile(x, sd.asarray([0.1, 0.5, 0.9]), axis=0)))
            results.append(np.asarray(sd.nanmedian(x, axis=1)))
        assert all(np.array_equal(results[i], results[i % 2]) for i in range(len(results)))
    finally:
        sd.set_num_threads(before)


# ------------------------------------------------- optimized paths (A-C review)


def test_quantile_signed_zero_ties_total_order_and_thread_independent():
    # -0.0 ranks below +0.0 on every path (sort, select, parallel sort), so
    # the result is the same at every thread count (docs/api/array.md §7.10).
    rng = np.random.default_rng(12)
    qs = sd.asarray(np.linspace(0, 1, 41))
    before = sd.get_num_threads()
    try:
        for n in (2, 17, 3000, 5000, 300_000):
            x = rng.choice([-0.0, 0.0, -1.0, 1.0], n)
            total = np.concatenate(
                [
                    np.sort(x[x < 0]),
                    -np.zeros(np.sum((x == 0) & np.signbit(x))),
                    np.zeros(np.sum((x == 0) & ~np.signbit(x))),
                    np.sort(x[x > 0]),
                ]
            )
            want = total[np.floor((n - 1) * np.linspace(0, 1, 41)).astype(int)]
            got = []
            for t in (1, 3, 10):
                sd.set_num_threads(t)
                got.append(np.asarray(sd.quantile(sd.asarray(x), qs, method="lower")))
                assert np.array_equal(got[-1].view(np.int64), want.view(np.int64)), (n, t)
    finally:
        sd.set_num_threads(before)


def nan_copy(x, value):
    c = np.copy(x, order="K")
    c[np.isnan(c)] = value
    return c


@pytest.mark.parametrize("layout", ["C", "F", "T"])
@pytest.mark.parametrize("cplx", [False, True])
def test_nan_reductions_are_the_plain_ones_on_a_filled_copy(layout, cplx):
    # The nan* folds read NaN as the identity instead of copying: the bits
    # must be those of the plain reduction on the NaN-replaced copy.
    rng = np.random.default_rng(13)
    x = rng.standard_normal((64, 2000)) * 3
    if cplx:
        x = x + 1j * rng.standard_normal(x.shape)
    x[rng.random(x.shape) < 0.2] = np.nan
    x = {"C": x, "F": np.asfortranarray(x), "T": x.T}[layout]
    a, f0, f1 = sd.asarray(x), sd.asarray(nan_copy(x, 0)), sd.asarray(nan_copy(x, 1))
    for axis in (None, 0, 1, (0, 1)):
        for got, want in (
            (sd.nansum(a, axis=axis), sd.sum(f0, axis=axis)),
            (sd.nanprod(a, axis=axis), sd.prod(f1, axis=axis)),
        ):
            if isinstance(got, sd.ndarray):
                assert_same(got, np.asarray(want))
            else:
                assert_same(sd.asarray([got]), np.asarray([want]))
        nz = sd.asarray(np.asarray(x) != 0)
        assert np.array_equal(np.asarray(sd.count_nonzero(a, axis=axis)), np.asarray(sd.count_nonzero(nz, axis=axis)))
    for axis in (0, 1):
        assert_same(sd.nancumulative_sum(a, axis=axis), np.asarray(sd.cumulative_sum(f0, axis=axis)))
    if not cplx:
        for axis in (None, 0, 1):
            fi, fn = sd.asarray(nan_copy(x, np.inf)), sd.asarray(nan_copy(x, -np.inf))
            assert np.array_equal(np.asarray(sd.nanargmin(a, axis=axis)), np.asarray(sd.argmin(fi, axis=axis)))
            assert np.array_equal(np.asarray(sd.nanargmax(a, axis=axis)), np.asarray(sd.argmax(fn, axis=axis)))


def test_stable_sort_keeps_the_input_order_of_mixed_zeros():
    # A stable float sort with zeros of both signs takes the NEON sort and writes the zeros
    # back in input order (it used to fall to merge sort: 64 ms for 10^6 values, now 14).
    rng = np.random.default_rng(7)
    for n in (40, 1000, 200_000):
        a = rng.standard_normal(n).round(1)
        a[rng.uniform(size=n) < 0.2] = 0.0
        a[rng.uniform(size=n) < 0.1] = -0.0
        got = np.asarray(sd.sort(sd.asarray(a)))
        assert got.tobytes() == np.sort(a, kind="stable").tobytes()
        got = np.asarray(sd.sort(sd.asarray(a), descending=True))
        assert got.tobytes() == a[np.argsort(-a, kind="stable")].tobytes()
