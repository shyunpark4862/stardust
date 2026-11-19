"""solve, inv, det, slogdet and the shared pipeline under them."""

import cmath
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .conftest import (
    DTYPES,
    LAYOUTS,
    as_sd,
    batch_shapes,
    compute_dtype,
    npy,
    random_values,
    residual_ok,
    well_conditioned,
    with_layout,
)

SETTINGS = settings(max_examples=int(__import__("os").environ.get("SDLA_EXAMPLES", 150)), deadline=None)


def f_order_strides(shape, itemsize):
    """Strides of a result: every matrix column-major, batch axes outside in C order."""
    m, n = shape[-2], shape[-1]
    st_ = [0] * len(shape)
    st_[-2], st_[-1] = itemsize, itemsize * m
    s = itemsize * m * n
    for d in range(len(shape) - 3, -1, -1):
        st_[d] = s
        s *= shape[d]
    return tuple(st_)


# ---------------------------------------------------------------- properties --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_det_slogdet_match_numpy(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    a = with_layout(random_values(rng, (*batch, n, n), dtype), layout)
    ref = a.astype(compute_dtype(dtype))
    d = sd.linalg.det(as_sd(a))
    s = sd.linalg.slogdet(as_sd(a))
    want_d = np.linalg.det(ref)
    want_s = np.linalg.slogdet(ref)
    scale = np.prod(np.maximum(np.abs(ref).sum(axis=-1), 1.0), axis=-1)  # Hadamard-style bound on |det|
    if not batch:
        assert isinstance(d, complex if dtype == "complex128" else float)
        assert isinstance(s.sign, complex if dtype == "complex128" else float)
        assert isinstance(s.logabsdet, float)
    else:
        assert npy(d).dtype == compute_dtype(dtype) and npy(d).shape == batch
        assert npy(s.logabsdet).dtype == np.float64
    assert np.all(np.abs(np.asarray(d) - want_d) <= 1e-12 * scale)
    # where numpy finds the matrix singular (exact zero pivot), so must we
    sing = np.asarray(want_s.sign) == 0
    assert np.array_equal(np.asarray(s.sign) == 0, sing)
    ok = ~sing
    np.testing.assert_allclose(np.asarray(s.sign)[ok], np.asarray(want_s.sign)[ok], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(s.logabsdet)[ok], np.asarray(want_s.logabsdet)[ok], rtol=0, atol=1e-12 * max(n, 1)
    )
    assert np.all(np.isneginf(np.asarray(s.logabsdet)[sing]))


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_inv(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    a = with_layout(well_conditioned(rng, batch, n, dtype), layout)
    x = sd.linalg.inv(as_sd(a))
    got = npy(x)
    ref = a.astype(compute_dtype(dtype))
    assert got.dtype == ref.dtype and got.shape == a.shape
    assert x.strides == f_order_strides(a.shape, got.itemsize)
    assert residual_ok(ref, got, np.broadcast_to(np.eye(n), ref.shape))
    np.testing.assert_allclose(got, np.linalg.inv(ref), rtol=1e-12, atol=1e-14)
    # inv(a) is solve(a, I) bit for bit (docs/design/linalg.md §10.2)
    eye = np.broadcast_to(np.eye(n), a.shape)
    assert np.array_equal(got, npy(sd.linalg.solve(as_sd(a), as_sd(eye))))


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch_a=batch_shapes(),
    batch_b=batch_shapes(),
    n=st.integers(0, 6),
    k=st.integers(0, 3),
    vector=st.booleans(),
    dtype_a=st.sampled_from(DTYPES),
    dtype_b=st.sampled_from(DTYPES),
    layout_a=st.sampled_from(LAYOUTS),
    layout_b=st.sampled_from(LAYOUTS),
)
def test_solve(seed, batch_a, batch_b, n, k, vector, dtype_a, dtype_b, layout_a, layout_b):
    try:
        np.broadcast_shapes(batch_a, batch_b)
    except ValueError:
        batch_b = ()
    rng = np.random.default_rng(seed)
    a = with_layout(well_conditioned(rng, batch_a, n, dtype_a), layout_a)
    if vector:
        b = random_values(rng, (n,), dtype_b)
    else:
        b = with_layout(random_values(rng, (*batch_b, n, k), dtype_b), layout_b)
    x = sd.linalg.solve(as_sd(a), as_sd(b))
    got = npy(x)
    ct = compute_dtype(dtype_a, dtype_b)
    ra, rb = a.astype(ct), b.astype(ct)
    want = np.linalg.solve(ra, rb)  # numpy 2 uses the same vector/matrix rule for b
    assert got.dtype == ct and got.shape == want.shape
    if not vector:
        assert x.strides == f_order_strides(want.shape, got.itemsize)
    assert residual_ok(ra, got, rb if vector else np.broadcast_to(rb, want.shape))
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------- rules --


def test_result_is_fresh_writable_memory():
    a = sd.asarray([[2.0, 1.0], [1.0, 3.0]])
    x = sd.linalg.inv(a)
    x[0, 0] = 7.0
    assert npy(a)[0, 0] == 2.0 and npy(x)[0, 0] == 7.0
    # A fresh result owns its memory like every new array: no .base
    # (docs/api/linalg.md §1.7, docs/api/array.md §10.1).
    assert x.base is None


@pytest.mark.parametrize("bad", [[[1.0, 0.0], [0.0, 1.0]], np.eye(2), 3.0, None])
def test_only_sd_arrays(bad):
    for f in (sd.linalg.det, sd.linalg.slogdet, sd.linalg.inv):
        with pytest.raises(TypeError, match=r"^\w+\(\): '\w+' must be an sd\.ndarray, not "):
            f(bad)
    with pytest.raises(TypeError, match="'a' must be"):
        sd.linalg.solve(bad, sd.ones((2,)))
    with pytest.raises(TypeError, match="'b' must be"):
        sd.linalg.solve(sd.eye(2), bad)


def test_positional_only():
    with pytest.raises(TypeError):
        sd.linalg.det(x=sd.eye(2))
    with pytest.raises(TypeError):
        sd.linalg.solve(sd.eye(2), b=sd.ones((2,)))


@pytest.mark.parametrize("f", [sd.linalg.det, sd.linalg.slogdet, sd.linalg.inv])
def test_shape_errors(f):
    with pytest.raises(ValueError, match="at least 2 dimensions"):
        f(sd.ones((3,)))
    with pytest.raises(ValueError, match="square"):
        f(sd.ones((2, 3)))


def test_solve_shape_errors():
    with pytest.raises(ValueError, match="rows"):
        sd.linalg.solve(sd.eye(3), sd.ones((2,)))
    with pytest.raises(ValueError, match="rows"):
        sd.linalg.solve(sd.eye(3), sd.ones((2, 1)))
    with pytest.raises(ValueError, match="broadcast"):
        sd.linalg.solve(sd.ones((2, 3, 3)), sd.ones((4, 3, 1)))
    with pytest.raises(ValueError, match="square"):
        sd.linalg.solve(sd.ones((2, 3)), sd.ones((2,)))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, complex(0, math.nan)])
def test_nonfinite_input_is_value_error(bad):
    a = np.eye(3, dtype=complex if isinstance(bad, complex) else float)
    a[1, 2] = bad
    for f in (sd.linalg.det, sd.linalg.slogdet, sd.linalg.inv):
        with pytest.raises(ValueError, match="NaN or infinity") as e:
            f(as_sd(a))
        assert not isinstance(e.value, sd.linalg.LinAlgError)
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sd.linalg.solve(as_sd(a), sd.ones((3,)))
    b = np.ones(3, dtype=a.dtype)
    b[0] = bad
    with pytest.raises(ValueError, match="'b' contains NaN"):
        sd.linalg.solve(sd.eye(3), as_sd(b))


def test_singular_raises_linalgerror_with_smallest_batch_index():
    assert issubclass(sd.linalg.LinAlgError, ValueError)
    a = np.stack([np.eye(2)] * 4)
    a[2] = 0.0
    a[1, 1] = 0.0  # batch elements 1 and 2 are singular; 1 must be reported
    with pytest.raises(sd.linalg.LinAlgError, match=r"singular matrix \(at batch index \(1,\)\)"):
        sd.linalg.inv(as_sd(a))
    with pytest.raises(sd.linalg.LinAlgError, match=r"batch index \(1,\)"):
        sd.linalg.solve(as_sd(a), sd.ones((2,)))
    with pytest.raises(sd.linalg.LinAlgError, match=r"^inv\(\): singular matrix$"):
        sd.linalg.inv(sd.zeros((2, 2)))
    with pytest.raises(sd.linalg.LinAlgError, match=r"batch index \(0, 1\)"):
        sd.linalg.inv(as_sd(np.stack([np.eye(2), np.zeros((2, 2))])[None]))


def test_det_slogdet_of_singular_do_not_raise():
    a = np.stack([np.eye(2), np.zeros((2, 2)), [[1.0, 2.0], [2.0, 4.0]]])
    assert np.array_equal(npy(sd.linalg.det(as_sd(a))), [1.0, 0.0, 0.0])
    s = sd.linalg.slogdet(as_sd(a))
    assert np.array_equal(npy(s.sign), [1.0, 0.0, 0.0])
    assert np.array_equal(npy(s.logabsdet), [0.0, -np.inf, -np.inf])
    assert sd.linalg.det(sd.zeros((3, 3))) == 0.0
    assert sd.linalg.slogdet(sd.zeros((3, 3))) == (0.0, -math.inf)


def test_empty():
    assert sd.linalg.det(sd.zeros((0, 0))) == 1.0
    assert sd.linalg.slogdet(sd.zeros((0, 0))) == (1.0, 0.0)
    assert sd.linalg.det(as_sd(np.zeros((0, 0), complex))) == 1 + 0j
    assert sd.linalg.inv(sd.zeros((0, 0))).shape == (0, 0)
    assert sd.linalg.solve(sd.zeros((0, 0)), sd.zeros((0,))).shape == (0,)
    assert sd.linalg.solve(sd.zeros((0, 0)), sd.zeros((0, 3))).shape == (0, 3)
    assert sd.linalg.solve(sd.eye(3), sd.zeros((3, 0))).shape == (3, 0)
    # a batch of zero matrices
    assert npy(sd.linalg.det(sd.zeros((0, 3, 3)))).shape == (0,)
    assert sd.linalg.slogdet(sd.zeros((2, 0, 3, 3))).sign.shape == (2, 0)
    assert sd.linalg.inv(sd.zeros((0, 3, 3))).shape == (0, 3, 3)
    assert sd.linalg.solve(sd.zeros((0, 3, 3)), sd.ones((3,))).shape == (0, 3)


def test_solve_broadcasting_and_vector_rule():
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 1, 3, 3)) + 4 * np.eye(3)
    b = rng.standard_normal((4, 3, 2))
    assert sd.linalg.solve(as_sd(a), as_sd(b)).shape == (2, 4, 3, 2)
    # 1-D b is one vector for every matrix of the batch
    v = rng.standard_normal(3)
    x = npy(sd.linalg.solve(as_sd(a), as_sd(v)))
    assert x.shape == (2, 1, 3)
    np.testing.assert_allclose(x, np.linalg.solve(a, v), rtol=1e-12)
    # (3, 1) is one matrix with one column, not a batch of vectors
    assert sd.linalg.solve(sd.eye(3), sd.ones((3, 1))).shape == (3, 1)


def test_dtypes():
    i = sd.asarray([[2, 1], [1, 3]])
    assert sd.linalg.det(i) == 5.0 and isinstance(sd.linalg.det(i), float)
    assert npy(sd.linalg.inv(i)).dtype == np.float64
    assert sd.linalg.det(sd.eye(2, dtype=sd.bool)) == 1.0
    c = as_sd(np.array([[1j, 0], [0, 2]]))
    assert sd.linalg.det(c) == 2j
    s = sd.linalg.slogdet(c)
    assert s.sign == 1j and s.logabsdet == pytest.approx(math.log(2))
    assert npy(sd.linalg.solve(sd.eye(2), as_sd(np.array([1j, 1])))).dtype == np.complex128
    assert npy(sd.linalg.solve(c, sd.ones((2,)))).dtype == np.complex128


def test_complex_slogdet_sign_is_unit():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 4, 4)) + 1j * rng.standard_normal((5, 4, 4))
    s = sd.linalg.slogdet(as_sd(a))
    np.testing.assert_allclose(np.abs(npy(s.sign)), 1.0, rtol=1e-14)
    d = npy(s.sign) * np.exp(npy(s.logabsdet))
    np.testing.assert_allclose(d, np.linalg.det(a), rtol=1e-12)
    assert cmath.isclose(sd.linalg.det(as_sd(a[0])), np.linalg.det(a[0]), rel_tol=1e-12)


def test_large_values_det_overflows_slogdet_does_not():
    a = np.eye(200) * 1e10
    assert sd.linalg.det(as_sd(a)) == math.inf
    s = sd.linalg.slogdet(as_sd(a))
    assert s.sign == 1.0 and s.logabsdet == pytest.approx(200 * math.log(1e10))
