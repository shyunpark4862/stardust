"""vecdot, outer, trace, diagonal, matrix_power, vector_norm, matrix_norm."""

import math
import os

import numpy as np
import pytest
import stardust as sd
from .conftest import DTYPES, LAYOUTS, batch_shapes, compute_dtype, npy, random_values, as_sd, well_conditioned, with_layout
from hypothesis import given, settings
from hypothesis import strategies as st


SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
INF = math.inf


def f_order_strides(shape, itemsize):
    m, n = shape[-2], shape[-1]
    st_ = [0] * len(shape)
    st_[-2], st_[-1] = itemsize, itemsize * m
    s = itemsize * m * n
    for d in range(len(shape) - 3, -1, -1):
        st_[d] = s
        s *= shape[d]
    return tuple(st_)


def as_result(r):
    """A Python scalar or an sd array -> numpy."""
    return np.asarray(r) if isinstance(r, sd.ndarray) else np.asarray(r)


# ---------------------------------------------------------------- vecdot --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    b1=batch_shapes(),
    b2=batch_shapes(),
    n=st.integers(0, 5),
    d1=st.sampled_from(DTYPES),
    d2=st.sampled_from(DTYPES),
)
def test_vecdot_matches_numpy(seed, b1, b2, n, d1, d2):
    try:
        np.broadcast_shapes(b1, b2)
    except ValueError:
        b2 = ()
    rng = np.random.default_rng(seed)
    x1, x2 = random_values(rng, (*b1, n), d1), random_values(rng, (*b2, n), d2)
    ct = compute_dtype(d1, d2)
    got = sd.linalg.vecdot(as_sd(x1), as_sd(x2))
    want = np.vecdot(x1.astype(ct), x2.astype(ct))  # conjugates x1, like the Array API
    g = as_result(got)
    assert g.dtype == ct and g.shape == want.shape
    if g.ndim == 0:
        assert isinstance(got, complex if ct == np.complex128 else float)
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-12)


def test_vecdot_axis():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((3, 4)), rng.standard_normal((2, 3, 4))
    np.testing.assert_allclose(npy(sd.linalg.vecdot(as_sd(a), as_sd(b), axis=-2)), np.vecdot(a, b, axis=-2), rtol=1e-14)
    np.testing.assert_allclose(npy(sd.linalg.vecdot(as_sd(a), as_sd(b), axis=1)), np.vecdot(a, b, axis=-2), rtol=1e-14)
    with pytest.raises(ValueError, match="same size"):
        sd.linalg.vecdot(as_sd(a), as_sd(b), axis=0)  # sizes 1 (padded) and 2 along the contracted axis
    with pytest.raises(ValueError, match="out of bounds"):
        sd.linalg.vecdot(as_sd(a), as_sd(b), axis=3)
    with pytest.raises(TypeError):
        sd.linalg.vecdot(as_sd(a), as_sd(b), axis=True)


def test_vecdot_conjugates_x1():
    x = as_sd(np.array([1j, 2]))
    assert sd.linalg.vecdot(x, x) == 5 + 0j


# ----------------------------------------------------------------- outer --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    b1=batch_shapes(),
    b2=batch_shapes(),
    m=st.integers(0, 4),
    n=st.integers(0, 4),
    d1=st.sampled_from(DTYPES),
    d2=st.sampled_from(DTYPES),
)
def test_outer_matches_numpy(seed, b1, b2, m, n, d1, d2):
    try:
        np.broadcast_shapes(b1, b2)
    except ValueError:
        b2 = ()
    rng = np.random.default_rng(seed)
    x1, x2 = random_values(rng, (*b1, m), d1), random_values(rng, (*b2, n), d2)
    ct = compute_dtype(d1, d2)
    got = sd.linalg.outer(as_sd(x1), as_sd(x2))
    want = x1.astype(ct)[..., :, None] * x2.astype(ct)[..., None, :]
    assert npy(got).dtype == ct and got.shape == want.shape
    assert got.strides == f_order_strides(want.shape, npy(got).itemsize)
    assert np.array_equal(npy(got), want)  # one product per element: exact


# --------------------------------------------------------- trace, diagonal --


def saturating_int64(x):
    """float -> int64 as the array area converts explicitly: toward zero, saturating, NaN -> 0."""
    t = np.trunc(np.nan_to_num(x, nan=0.0, posinf=2.0**64, neginf=-(2.0**64)))
    out = np.where(t >= 2.0**63, np.iinfo(np.int64).max, 0)
    out = np.where(t < -(2.0**63), np.iinfo(np.int64).min, out)
    mid = (t < 2.0**63) & (t >= -(2.0**63))
    return np.where(mid, t.clip(-(2.0**63), 2.0**63 - 1024).astype(np.int64), out)


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(0, 5),
    n=st.integers(0, 5),
    k=st.integers(-6, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_trace_and_diagonal_match_numpy(seed, batch, m, n, k, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    diag = np.diagonal(x, offset=k, axis1=-2, axis2=-1)
    # diagonal: a copy with the input dtype (docs/design/linalg.md §10.8)
    d = sd.linalg.diagonal(as_sd(x), k=k)
    assert npy(d).dtype == x.dtype and np.array_equal(npy(d), diag)
    # trace, default dtype: computed in float64/complex128
    ct = compute_dtype(dtype)
    t = sd.linalg.trace(as_sd(x), k=k)
    want = diag.astype(ct).sum(axis=-1)
    if not batch:
        assert isinstance(t, complex if ct == np.complex128 else float)
    np.testing.assert_allclose(as_result(t), want, rtol=1e-14, atol=1e-14)
    # trace with dtype=: an explicit conversion, then the sum (as sd.sum)
    if dtype != "complex128":
        ti = sd.linalg.trace(as_sd(x), k=k, dtype=sd.int64)
        conv = saturating_int64(diag) if dtype == "float64" else diag.astype(np.int64)
        assert np.array_equal(as_result(ti), conv.sum(axis=-1, dtype=np.int64))
        if not batch:
            assert type(ti) is int


def test_diagonal_is_a_writable_copy():
    a = sd.asarray([[1, 2], [3, 4]])
    d = sd.linalg.diagonal(a)
    d[0] = 9
    assert npy(a)[0, 0] == 1 and npy(d).tolist() == [9, 4]
    assert npy(sd.linalg.diagonal(sd.eye(3, dtype=sd.bool))).dtype == np.bool_


def test_trace_dtype_rules():
    i = sd.asarray([[2**62, 0], [0, 2**62]])
    assert sd.linalg.trace(i, dtype=sd.int64) == -(2**63)  # wraps, like sd.sum
    assert sd.linalg.trace(i) == 2.0**63
    assert sd.linalg.trace(sd.asarray([[1e300, 0], [0, 1e300]]), dtype=sd.int64) == -2  # saturate, then wrap
    with pytest.raises(TypeError, match="bool"):
        sd.linalg.trace(i, dtype=sd.bool)
    c = as_sd(np.array([[1j, 0], [0, 1]]))
    with pytest.raises(TypeError, match="imaginary part"):
        sd.linalg.trace(c, dtype=sd.float64)
    assert sd.linalg.trace(c, dtype=sd.complex128) == 1 + 1j
    with pytest.raises(TypeError, match="dtype"):
        sd.linalg.trace(i, dtype=float)
    with pytest.raises(TypeError):
        sd.linalg.trace(i, k=True)


# --------------------------------------------------------- matrix_power --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(0, 5),
    n=st.integers(-4, 9),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_matrix_power_matches_numpy(seed, batch, m, n, dtype, layout):
    rng = np.random.default_rng(seed)
    if n < 0:
        x = well_conditioned(rng, batch, m, dtype)
    else:
        x = random_values(rng, (*batch, m, m), dtype)
        if dtype in ("float64", "complex128"):
            x = x / max(m, 1)  # keep powers bounded
    x = with_layout(x, layout)
    ct = compute_dtype(dtype)
    got = sd.linalg.matrix_power(as_sd(x), n)
    want = np.linalg.matrix_power(x.astype(ct), n)
    g = npy(got)
    assert g.dtype == ct and g.shape == x.shape
    assert got.strides == f_order_strides(x.shape, g.itemsize)
    scale = max(np.abs(want).max(initial=0.0), 1.0)
    np.testing.assert_allclose(g, want, rtol=1e-11, atol=1e-11 * scale)


def test_matrix_power_exact_cases():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 4, 4)) + 8 * np.eye(4)
    assert np.array_equal(npy(sd.linalg.matrix_power(as_sd(x), -1)), npy(sd.linalg.inv(as_sd(x))))
    assert np.array_equal(npy(sd.linalg.matrix_power(as_sd(x), 1)), x)
    nan = np.full((3, 3), np.nan)
    assert np.array_equal(npy(sd.linalg.matrix_power(as_sd(nan), 0)), np.eye(3))  # x^0 = I whatever x holds
    assert np.isnan(npy(sd.linalg.matrix_power(as_sd(nan), 2))).all()  # n >= 0: no check, NaN propagates
    with pytest.raises(ValueError, match="NaN or infinity"):
        sd.linalg.matrix_power(as_sd(nan), -1)  # n < 0 goes through LAPACK: checked (docs/design/linalg.md §2.3)
    with pytest.raises(sd.linalg.LinAlgError, match="singular"):
        sd.linalg.matrix_power(sd.zeros((2, 2)), -2)
    big = sd.linalg.matrix_power(sd.eye(2), 2**62 + 1)
    assert np.array_equal(npy(big), np.eye(2))
    assert np.array_equal(npy(sd.linalg.matrix_power(sd.eye(2), -(2**63))), np.eye(2))
    with pytest.raises(TypeError):
        sd.linalg.matrix_power(sd.eye(2), 2.0)
    with pytest.raises(ValueError, match="square"):
        sd.linalg.matrix_power(sd.ones((2, 3)), 2)


# ----------------------------------------------------------- vector_norm --

ORDS = [2, 1, INF, -INF, 0, 3, 0.5, -1, -2.5]


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    shape=st.lists(st.integers(1, 4), min_size=1, max_size=3),
    dtype=st.sampled_from(DTYPES),
    ord_=st.sampled_from(ORDS),
    keepdims=st.booleans(),
    axis_kind=st.sampled_from(["none", "int", "tuple"]),
    layout=st.sampled_from(["C", "strided", "reversed"]),
    data=st.data(),
)
def test_vector_norm_matches_numpy(seed, shape, dtype, ord_, keepdims, axis_kind, layout, data):
    rng = np.random.default_rng(seed)
    x = random_values(rng, tuple(shape), dtype)
    if len(shape) >= 2:
        x = with_layout(x, layout)
    nd = x.ndim
    if axis_kind == "none":
        axis = None
    elif axis_kind == "int":
        axis = data.draw(st.integers(-nd, nd - 1))
    else:
        axis = tuple(data.draw(st.lists(st.integers(-nd, nd - 1), min_size=1, max_size=nd, unique_by=lambda a: a % nd)))
    ct = compute_dtype(dtype)
    with np.errstate(divide="ignore"):
        want = np.linalg.vector_norm(x.astype(ct), axis=axis, keepdims=keepdims, ord=ord_)
    got = sd.linalg.vector_norm(as_sd(x), axis=axis, keepdims=keepdims, ord=ord_)
    g = as_result(got)
    assert g.dtype == np.float64 and g.shape == np.shape(want)
    if g.ndim == 0:
        assert isinstance(got, float)
    np.testing.assert_allclose(g, want, rtol=1e-13, atol=0)


def test_vector_norm_no_overflow_or_underflow():
    for v, want in [(1e200, 1e200 * math.sqrt(2)), (1e-200, 1e-200 * math.sqrt(2)), (1e-160, 1e-160 * math.sqrt(2))]:
        assert sd.linalg.vector_norm(as_sd(np.array([v, v]))) == pytest.approx(want, rel=1e-15)
    assert sd.linalg.vector_norm(as_sd(np.array([1e150, 1e150])), ord=3) == pytest.approx(1e150 * 2 ** (1 / 3), rel=1e-15)
    assert sd.linalg.vector_norm(as_sd(np.array([3e200j, 4e200]))) == pytest.approx(5e200, rel=1e-15)
    assert sd.linalg.matrix_norm(as_sd(np.full((2, 2), 1e200))) == pytest.approx(2e200, rel=1e-15)


def test_vector_norm_special_values():
    nan = as_sd(np.array([1.0, np.nan, 2.0]))
    for o in (2, 1, INF, -INF, 3):
        assert math.isnan(sd.linalg.vector_norm(nan, ord=o))
    assert sd.linalg.vector_norm(nan, ord=0) == 3.0  # NaN != 0 counts
    inf = as_sd(np.array([1.0, np.inf]))
    assert sd.linalg.vector_norm(inf) == INF and sd.linalg.vector_norm(inf, ord=3) == INF
    assert math.isnan(sd.linalg.vector_norm(as_sd(np.array([np.inf, np.nan]))))
    assert sd.linalg.vector_norm(as_sd(np.array([0.0, 2.0])), ord=-1) == 0.0


def test_vector_norm_empty_and_errors():
    e = sd.zeros((0,))
    for o in ORDS:
        assert sd.linalg.vector_norm(e, ord=o) == 0.0  # docs/api/linalg.md §1.10
    assert npy(sd.linalg.vector_norm(sd.zeros((3, 0)), axis=1)).tolist() == [0.0, 0.0, 0.0]
    assert sd.linalg.vector_norm(sd.zeros((3, 0)), axis=0).shape == (0,)
    x = sd.ones((2, 3))
    with pytest.raises(ValueError, match="repeated"):
        sd.linalg.vector_norm(x, axis=(0, -2))
    with pytest.raises(ValueError, match="out of bounds"):
        sd.linalg.vector_norm(x, axis=2)
    with pytest.raises(TypeError):
        sd.linalg.vector_norm(x, ord=True)
    with pytest.raises(TypeError):
        sd.linalg.vector_norm(x, ord="fro")
    with pytest.raises(ValueError, match="NaN"):
        sd.linalg.vector_norm(x, ord=math.nan)
    assert sd.linalg.vector_norm(x, axis=(0, 1), keepdims=True).shape == (1, 1)


def test_vector_norm_axis_list_is_a_tuple():
    # A list is taken like a tuple (DESIGN.md §9.6, docs/api/linalg.md §3).
    rng = np.random.default_rng(5)
    n = rng.standard_normal((2, 3, 4))
    x = as_sd(n)
    for axis in ([0], [2, 0], [-1], [], [0, 1, 2]):
        got = sd.linalg.vector_norm(x, axis=axis, keepdims=True)
        assert np.array_equal(npy(got), npy(sd.linalg.vector_norm(x, axis=tuple(axis), keepdims=True)))
    with pytest.raises(ValueError, match="repeated"):
        sd.linalg.vector_norm(x, axis=[0, -3])
    with pytest.raises(TypeError):
        sd.linalg.vector_norm(x, axis=[0.0])


@pytest.mark.parametrize("call", [
    lambda o: sd.linalg.vector_norm(sd.ones((2, 2)), ord=o),
    lambda o: sd.linalg.matrix_norm(sd.ones((2, 2)), ord=o),
    lambda o: sd.linalg.cond(sd.eye(2), ord=o),
])
def test_ord_of_a_wrong_type_is_a_type_error_everywhere(call):
    # The same error kind in the three functions (DESIGN.md §9.6).
    for bad in (True, False, [2], (1,), 1j, b"fro"):
        with pytest.raises(TypeError, match="'ord' must be"):
            call(bad)


@pytest.mark.parametrize("call", [
    lambda v: sd.linalg.trace(sd.eye(2), k=v),
    lambda v: sd.linalg.diagonal(sd.eye(2), k=v),
    lambda v: sd.linalg.matrix_power(sd.eye(2), v),
    lambda v: sd.linalg.vecdot(sd.ones(2), sd.ones(2), axis=v),
    lambda v: sd.linalg.cross(sd.ones(3), sd.ones(3), axis=v),
    lambda v: sd.linalg.vector_norm(sd.ones(2), axis=v),
    lambda v: sd.linalg.vector_norm(sd.ones(2), axis=(v,)),
    lambda v: sd.linalg.solve_banded(sd.ones((1, 2)), sd.ones(2), lower_bandwidth=v, upper_bandwidth=0),
])
def test_int_argument_outside_int64_is_overflow_error(call):
    # API.md §2: an int that must become an int64 (was ValueError).
    for big in (2**63, -(2**63) - 1, 2**100):
        with pytest.raises(OverflowError, match="out of range"):
            call(big)


# ----------------------------------------------------------- matrix_norm --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(1, 5),
    n=st.integers(1, 5),
    dtype=st.sampled_from(DTYPES),
    ord_=st.sampled_from(["fro", 1, -1, INF, -INF]),
    keepdims=st.booleans(),
    layout=st.sampled_from(LAYOUTS),
)
def test_matrix_norm_matches_numpy(seed, batch, m, n, dtype, ord_, keepdims, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    ct = compute_dtype(dtype)
    want = np.linalg.matrix_norm(x.astype(ct), keepdims=keepdims, ord=ord_)
    got = sd.linalg.matrix_norm(as_sd(x), keepdims=keepdims, ord=ord_)
    g = as_result(got)
    assert g.dtype == np.float64 and g.shape == np.shape(want)
    np.testing.assert_allclose(g, want, rtol=1e-13, atol=0)


def test_matrix_norm_rules():
    x = sd.ones((2, 3))
    for o in (3, "max", 0, "FRO", 1.5):
        with pytest.raises(ValueError, match="ord"):
            sd.linalg.matrix_norm(x, ord=o)
    with pytest.raises(ValueError, match="2 dimensions"):
        sd.linalg.matrix_norm(sd.ones((3,)))
    for shape in [(0, 0), (3, 0), (0, 3)]:
        for o in ("fro", 1, -1, INF, -INF):
            assert sd.linalg.matrix_norm(sd.zeros(shape), ord=o) == 0.0  # docs/api/linalg.md §1.10
    assert math.isnan(sd.linalg.matrix_norm(as_sd(np.array([[1.0, np.nan], [0, 1]])), ord=1))
    assert sd.linalg.matrix_norm(sd.ones((4, 2, 3)), keepdims=True).shape == (4, 1, 1)


# ----------------------------------------------------------------- common --


@pytest.mark.parametrize(
    "call",
    [
        lambda b: sd.linalg.vecdot(b, b),
        lambda b: sd.linalg.outer(b, b),
        lambda b: sd.linalg.trace(b),
        lambda b: sd.linalg.diagonal(b),
        lambda b: sd.linalg.matrix_power(b, 2),
        lambda b: sd.linalg.vector_norm(b),
        lambda b: sd.linalg.matrix_norm(b),
    ],
)
@pytest.mark.parametrize("bad", [[[1.0]], np.eye(2), None, 1.0])
def test_only_sd_arrays(call, bad):
    with pytest.raises(TypeError, match=r"^\w+\(\): '\w+' must be an sd\.ndarray, not "):
        call(bad)


def test_no_nan_check_outside_lapack():
    nan = as_sd(np.array([[np.nan, 1.0], [2.0, 3.0]]))
    assert math.isnan(sd.linalg.trace(nan))
    assert math.isnan(npy(sd.linalg.vecdot(nan, nan))[0])
    assert math.isnan(npy(sd.linalg.outer(nan[0], nan[1]))[0, 0])


def test_keyword_only_options():
    x = sd.eye(2)
    with pytest.raises(TypeError):
        sd.linalg.trace(x, 1)
    with pytest.raises(TypeError):
        sd.linalg.vector_norm(x, None)
    with pytest.raises(TypeError):
        sd.linalg.matrix_power(x, n=2)


def test_outer_equals_broadcast_multiply():
    rng = np.random.default_rng(5)
    x = as_sd(rng.standard_normal((3, 4)) + 1j * rng.standard_normal((3, 4)))
    y = as_sd(rng.standard_normal((3, 5)) + 1j * rng.standard_normal((3, 5)))
    want = sd.multiply(sd.expand_dims(x, axis=-1), sd.expand_dims(y, axis=-2))
    assert np.array_equal(npy(sd.linalg.outer(x, y)), npy(want))


@pytest.mark.parametrize("k", [-(2**63), -(10**18), -3, 4, 10**18, 2**63 - 1])
def test_offset_past_the_matrix_is_an_empty_diagonal(k):
    # k = -2**63 once negated -k (signed overflow, undefined behaviour); any
    # offset past the matrix is an empty diagonal (docs/api/linalg.md §2).
    a = sd.ones((3, 4))
    assert sd.linalg.trace(a, k=k) == 0.0
    assert sd.linalg.diagonal(a, k=k).shape == (0,)
    assert sd.linalg.diagonal(sd.ones((5, 3, 4)), k=k).shape == (5, 0)
