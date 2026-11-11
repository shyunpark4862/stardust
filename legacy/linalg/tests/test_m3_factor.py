"""M3: solve_triangular, cholesky, cholesky_solve, lu, lu_solve."""

import os

import numpy as np
import pytest
import scipy.linalg as sl
import stardust_numpy as sdnp
from conftest import (
    DTYPES,
    LAYOUTS,
    batch_shapes,
    compute_dtype,
    npy,
    random_values,
    residual_ok,
    sd,
    well_conditioned,
    with_layout,
)
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust_linalg as sdla

SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
FLOATS = ["float64", "complex128"]


def f_order_strides(shape, itemsize):
    m, n = shape[-2], shape[-1]
    st_ = [0] * len(shape)
    st_[-2], st_[-1] = itemsize, itemsize * m
    s = itemsize * m * n
    for d in range(len(shape) - 3, -1, -1):
        st_[d] = s
        s *= shape[d]
    return tuple(st_)


def poison(x, keep_lower, keep_diag=True):
    """Fill the triangle a function must not read with NaN (float dtypes only)."""
    x = x.copy()
    n = x.shape[-1]
    mask = np.tril(np.ones((n, n), bool), 0 if keep_diag else -1) if keep_lower else np.triu(
        np.ones((n, n), bool), 0 if keep_diag else 1)
    x[..., ~mask] = np.nan
    return x


def same_values_as_layout(x, layout):
    """The values with_layout(x, layout) will show: 'bcast' repeats batch element 0."""
    return np.ascontiguousarray(with_layout(x, layout)) if layout == "bcast" else x


def spd(rng, batch, n, dtype):
    """Hermitian positive definite matrices."""
    a = random_values(rng, (*batch, n, n), dtype)
    if dtype in ("bool", "int64"):
        a = a.astype(np.int64)
        return a @ np.swapaxes(a, -1, -2) + n * np.eye(n, dtype=np.int64) + np.eye(n, dtype=np.int64)
    return a @ np.conj(np.swapaxes(a, -1, -2)) + n * np.eye(n)


# -------------------------------------------------------- solve_triangular --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    k=st.integers(0, 3),
    vector=st.booleans(),
    upper=st.booleans(),
    unit=st.booleans(),
    da=st.sampled_from(FLOATS),
    db=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_solve_triangular(seed, batch, n, k, vector, upper, unit, da, db, layout):
    rng = np.random.default_rng(seed)
    a = same_values_as_layout(random_values(rng, (*batch, n, n), da) + (2 * n + 2) * np.eye(n), layout)
    tri = np.triu(a) if upper else np.tril(a)
    if unit:
        tri = tri - np.diagonal(tri, axis1=-2, axis2=-1)[..., None, :] * np.eye(n) + np.eye(n)
    a_in = with_layout(poison(a, keep_lower=not upper, keep_diag=not unit), layout)  # the unread part is NaN
    b = random_values(rng, (n,) if vector else (*batch, n, k), db)
    ct = compute_dtype(da, db)
    got = sdla.solve_triangular(sd(a_in), sd(b), upper=upper, unit_diagonal=unit)
    want = np.linalg.solve(tri.astype(ct), b.astype(ct))
    g = npy(got)
    assert g.dtype == ct and g.shape == want.shape
    if not vector:
        assert got.strides == f_order_strides(want.shape, g.itemsize)
    np.testing.assert_allclose(g, want, rtol=1e-12, atol=1e-13)
    assert residual_ok(tri.astype(ct), g, b.astype(ct) if vector else np.broadcast_to(b.astype(ct), want.shape))


def test_solve_triangular_rules():
    a = np.triu(np.ones((3, 3)))
    a[1, 1] = 0.0
    with pytest.raises(sdla.LinAlgError, match="^singular matrix$"):
        sdla.solve_triangular(sd(a), sdnp.ones((3,)), upper=True)
    # with unit_diagonal the diagonal is not read, so the zero does not matter
    x = sdla.solve_triangular(sd(a), sdnp.ones((3,)), upper=True, unit_diagonal=True)
    np.testing.assert_allclose(npy(x), np.linalg.solve(np.triu(np.ones((3, 3))), np.ones(3)))
    with pytest.raises(TypeError):
        sdla.solve_triangular(sdnp.eye(3), sdnp.ones((3,)))  # upper is required (S3)
    with pytest.raises(TypeError):
        sdla.solve_triangular(sdnp.eye(3), sdnp.ones((3,)), True)  # keyword-only
    bad = np.eye(3)
    bad[2, 0] = np.nan
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sdla.solve_triangular(sd(bad), sdnp.ones((3,)), upper=False)
    sdla.solve_triangular(sd(bad), sdnp.ones((3,)), upper=True)  # the NaN is in the unread triangle
    with pytest.raises(ValueError, match="'b' contains NaN"):
        sdla.solve_triangular(sdnp.eye(3), sd(np.array([1.0, np.nan, 0])), upper=True)
    assert sdla.solve_triangular(sdnp.zeros((0, 0)), sdnp.zeros((0,)), upper=True).shape == (0,)


# ---------------------------------------------------------------- cholesky --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(["int64", "float64", "complex128"]),
    layout=st.sampled_from(LAYOUTS),
)
def test_cholesky(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    s = same_values_as_layout(spd(rng, batch, n, dtype), layout)
    ct = compute_dtype(dtype)
    s_in = with_layout(poison(s.astype(ct), keep_lower=True) if dtype != "int64" else s, layout)
    l = sdla.cholesky(sd(s_in))
    u = sdla.cholesky(sd(s_in), upper=True)
    gl, gu = npy(l), npy(u)
    assert gl.dtype == ct and gl.shape == s.shape
    assert l.strides == f_order_strides(s.shape, gl.itemsize)
    assert np.array_equal(gl, np.tril(gl)) and np.array_equal(gu, np.triu(gu))
    # U is L^H bit for bit (DESIGN.md §2.12)
    assert np.array_equal(gu, np.conj(np.swapaxes(gl, -1, -2)))
    want = np.linalg.cholesky(s.astype(ct))
    np.testing.assert_allclose(gl, want, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gl @ np.conj(np.swapaxes(gl, -1, -2)), s, rtol=1e-12, atol=1e-12 * max(n, 1) ** 2)


def test_cholesky_rules():
    with pytest.raises(sdla.LinAlgError, match="^matrix is not positive definite$"):
        sdla.cholesky(sd(np.array([[1.0, 2.0], [2.0, 1.0]])))
    batch = np.stack([np.eye(2), np.eye(2), -np.eye(2)])
    with pytest.raises(sdla.LinAlgError, match=r"positive definite \(at batch index \(2,\)\)"):
        sdla.cholesky(sd(batch))
    up = np.eye(3)
    up[0, 2] = np.inf  # unread upper triangle
    assert np.array_equal(npy(sdla.cholesky(sd(up))), np.eye(3))
    lo = np.eye(3)
    lo[2, 0] = np.inf
    with pytest.raises(ValueError, match="'x' contains NaN"):
        sdla.cholesky(sd(lo))
    assert sdla.cholesky(sdnp.zeros((0, 0))).shape == (0, 0)
    assert sdla.cholesky(sdnp.zeros((2, 0, 0)), upper=True).shape == (2, 0, 0)
    assert np.array_equal(npy(sdla.cholesky(sdnp.eye(3, dtype=sdnp.bool))), np.eye(3))


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    k=st.integers(0, 3),
    vector=st.booleans(),
    upper=st.booleans(),
    dtype=st.sampled_from(FLOATS),
    db=st.sampled_from(DTYPES),
)
def test_cholesky_solve(seed, batch, n, k, vector, upper, dtype, db):
    rng = np.random.default_rng(seed)
    s = spd(rng, batch, n, dtype)
    f = sdla.cholesky(sd(s), upper=upper)
    f_in = poison(npy(f), keep_lower=not upper)  # the factor's other triangle is not read
    b = random_values(rng, (n,) if vector else (*batch, n, k), db)
    ct = compute_dtype(dtype, db)
    got = npy(sdla.cholesky_solve(sd(f_in), sd(b), upper=upper))
    want = np.linalg.solve(s.astype(ct), b.astype(ct))
    assert got.dtype == ct and got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-12)


def test_cholesky_solve_rules():
    l = np.eye(3)
    l[1, 1] = 0.0
    with pytest.raises(sdla.LinAlgError, match="singular"):
        sdla.cholesky_solve(sd(l), sdnp.ones((3,)))
    with pytest.raises(ValueError, match="'l' contains NaN"):
        sdla.cholesky_solve(sd(np.full((2, 2), np.nan)), sdnp.ones((2,)))
    with pytest.raises(ValueError, match="rows"):
        sdla.cholesky_solve(sdnp.eye(3), sdnp.ones((2,)))


# ---------------------------------------------------------------------- lu --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(0, 6),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_lu(seed, batch, m, n, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    r = sdla.lu(sd(x))
    assert type(r).__name__ == "LUResult" and r._fields == ("L", "U", "perm")
    L, U, p = npy(r.L), npy(r.U), npy(r.perm)
    k = min(m, n)
    ct = compute_dtype(dtype)
    assert L.shape == (*batch, m, k) and U.shape == (*batch, k, n) and p.shape == (*batch, m)
    assert L.dtype == ct and U.dtype == ct and p.dtype == np.int64
    assert r.L.strides == f_order_strides(L.shape, L.itemsize)
    assert np.array_equal(L, np.tril(L)) and np.array_equal(U, np.triu(U))
    assert np.all(np.diagonal(L, axis1=-2, axis2=-1) == 1)
    assert np.all(np.sort(p, axis=-1) == np.arange(m))
    xp = np.take_along_axis(x.astype(ct), p[..., None], axis=-2)  # x[perm]
    np.testing.assert_allclose(L @ U, xp, rtol=1e-12, atol=1e-12 * max(n, 1))
    # Partial pivoting bounds L's entries: |L| <= 1 for real input. zgetrf picks
    # the pivot by the cheaper |re| + |im| (izamax), and |z| <= |re|+|im| <= sqrt(2)|z|,
    # so for complex input only |L| <= sqrt(2) is guaranteed.
    bound = np.sqrt(2) if ct == np.complex128 else 1.0
    assert np.all(np.abs(L) <= bound * (1 + 1e-15))
    if not batch and m and n:  # same factors as scipy (same getrf)
        sp_l, sp_u = sl.lu(x.astype(ct), permute_l=False, p_indices=True)[1:]
        np.testing.assert_allclose(L, sp_l, rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(U, sp_u, rtol=1e-12, atol=1e-14)


def test_lu_singular_is_not_an_error():
    r = sdla.lu(sd(np.array([[1.0, 2.0], [2.0, 4.0]])))
    assert npy(r.U)[1, 1] == 0.0
    r = sdla.lu(sdnp.zeros((3, 3)))
    assert np.array_equal(npy(r.U), np.zeros((3, 3))) and npy(r.perm).tolist() == [0, 1, 2]
    with pytest.raises(ValueError, match="NaN"):
        sdla.lu(sd(np.array([[np.nan]])))


# ---------------------------------------------------------------- lu_solve --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    k=st.integers(0, 3),
    vector=st.booleans(),
    da=st.sampled_from(DTYPES),
    db=st.sampled_from(DTYPES),
)
def test_lu_solve_equals_solve(seed, batch, n, k, vector, da, db):
    rng = np.random.default_rng(seed)
    a = well_conditioned(rng, batch, n, da)
    b = random_values(rng, (n,) if vector else (*batch, n, k), db)
    want = sdla.solve(sd(a), sd(b))
    got = sdla.lu_solve(*sdla.lu(sd(a)), sd(b))
    # The same steps as solve after getrf, so the same bits -- when both factor
    # in the same type. With a real `a` and a complex `b`, solve factors a
    # complex copy of `a` (zgetrf) while lu factors it as real (dgetrf); the
    # values then agree only to rounding (DESIGN.md §2.12).
    if da == "complex128" or db != "complex128":
        assert np.array_equal(npy(got), npy(want))
    else:
        np.testing.assert_allclose(npy(got), npy(want), rtol=1e-12, atol=1e-14)


def test_lu_solve_with_own_factors():
    rng = np.random.default_rng(1)
    n = 5
    l = np.tril(rng.standard_normal((n, n)), -1) + 7 * np.eye(n)  # the diagonal of l is not read (taken as 1)
    u = np.triu(rng.standard_normal((n, n))) + 4 * np.eye(n)
    p = rng.permutation(n)
    b = rng.standard_normal((n, 2))
    got = npy(sdla.lu_solve(sd(poison(l, True)), sd(poison(u, False)), sd(p), sd(b)))
    l1 = np.tril(l, -1) + np.eye(n)
    np.testing.assert_allclose(l1 @ u @ got, b[p], rtol=1e-12, atol=1e-12)


def test_lu_solve_rules():
    l, u, b = sdnp.eye(3), sdnp.eye(3), sdnp.ones((3,))
    for bad in ([0, 0, 1], [0, 1, 3], [-1, 0, 1]):
        with pytest.raises(ValueError, match="permutation"):
            sdla.lu_solve(l, u, sd(np.array(bad)), b)
    with pytest.raises(TypeError, match="int64"):
        sdla.lu_solve(l, u, sd(np.array([0.0, 1.0, 2.0])), b)
    with pytest.raises(ValueError, match="length"):
        sdla.lu_solve(l, u, sd(np.array([0, 1])), b)
    with pytest.raises(ValueError, match="'u' is"):
        sdla.lu_solve(l, sdnp.eye(2), sd(np.array([0, 1, 2])), b)
    u0 = np.eye(3)
    u0[2, 2] = 0
    with pytest.raises(sdla.LinAlgError, match="singular"):
        sdla.lu_solve(l, sd(u0), sd(np.array([0, 1, 2])), b)
    with pytest.raises(ValueError, match="'u' contains NaN"):
        sdla.lu_solve(l, sd(np.full((3, 3), np.nan)), sd(np.array([0, 1, 2])), b)
    # batches of permutations broadcast like everything else
    ps = sd(np.array([[0, 1, 2], [2, 1, 0]]))
    x = npy(sdla.lu_solve(l, u, ps, sd(np.array([1.0, 2.0, 3.0]))))
    assert x.tolist() == [[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]


@pytest.mark.parametrize(
    "call",
    [
        lambda z: sdla.solve_triangular(z, z, upper=True),
        lambda z: sdla.cholesky(z),
        lambda z: sdla.cholesky_solve(z, z),
        lambda z: sdla.lu(z),
        lambda z: sdla.lu_solve(z, z, z, z),
    ],
)
@pytest.mark.parametrize("bad", [[[1.0]], np.eye(2), None])
def test_only_sdnp_arrays(call, bad):
    with pytest.raises(TypeError, match="stardust_numpy.ndarray"):
        call(bad)


@SETTINGS
@given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 6), dtype=st.sampled_from(FLOATS))
def test_results_do_not_depend_on_input_layout(seed, n, dtype):
    """Packing turns every layout into the same F-order buffer, so the bits agree."""
    rng = np.random.default_rng(seed)
    s = spd(rng, (2,), n, dtype)
    b = random_values(rng, (2, n, 2), dtype)
    def run(layout):
        S = sd(with_layout(s, layout))
        B = sd(with_layout(b, layout))
        r = sdla.lu(S)
        return [npy(sdla.cholesky(S)), npy(sdla.solve_triangular(S, B, upper=False)), npy(sdla.solve(S, B)),
                npy(r.L), npy(r.U), npy(sdla.inv(S)), npy(sdla.det(S))]
    ref = run("C")
    for layout in ("F", "strided", "reversed"):
        for got, want in zip(run(layout), ref):
            assert np.array_equal(got, want)
