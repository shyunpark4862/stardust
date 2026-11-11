"""M5: eigh, eigvalsh, svd, svdvals, matrix_rank, pinv, cond, matrix_norm (2, -2, nuc)."""

import math
import os

import numpy as np
import pytest
import scipy.linalg as sl
import stardust_numpy as sdnp
from conftest import DTYPES, LAYOUTS, batch_shapes, compute_dtype, npy, random_values, sd, with_layout
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust_linalg as sdla

SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
EPS = np.finfo(np.float64).eps
INF = math.inf


def mH(x):
    return np.conj(np.swapaxes(x, -1, -2))


def same_values_as_layout(x, layout):
    return np.ascontiguousarray(with_layout(x, layout)) if layout == "bcast" else x


def hermitian(rng, batch, n, dtype):
    a = random_values(rng, (*batch, n, n), dtype)
    if dtype in ("bool", "int64"):
        a = a.astype(np.int64)
        return a + np.swapaxes(a, -1, -2)
    return (a + mH(a)) / 2


def poison_upper(x):
    """NaN above the diagonal: the triangle eigh must not read (API.md §0.2)."""
    x = x.copy()
    n = x.shape[-1]
    x[..., np.triu(np.ones((n, n), bool), 1)] = np.nan
    return x


def low_rank(rng, batch, m, n, r, dtype):
    """m x n matrices of rank min(r, m, n) (exactly, up to rounding)."""
    left = random_values(rng, (*batch, m, r), "float64" if dtype != "complex128" else dtype)
    right = random_values(rng, (*batch, r, n), "float64" if dtype != "complex128" else dtype)
    return left @ right


# ----------------------------------------------------------- eigh, eigvalsh --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_eigh(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    h = same_values_as_layout(hermitian(rng, batch, n, dtype), layout)
    ct = compute_dtype(dtype)
    h_in = with_layout(poison_upper(h.astype(ct)) if dtype in ("float64", "complex128") else h, layout)
    r = sdla.eigh(sd(h_in))
    assert type(r).__name__ == "EighResult" and r._fields == ("eigenvalues", "eigenvectors")
    w, v = npy(r.eigenvalues), npy(r.eigenvectors)
    assert w.dtype == np.float64 and w.shape == (*batch, n)
    assert v.dtype == ct and v.shape == (*batch, n, n)
    assert np.all(np.diff(w, axis=-1) >= 0)  # ascending
    hc = h.astype(ct)
    scale = max(np.abs(hc).max(initial=0), 1)
    np.testing.assert_allclose(hc @ v, v * w[..., None, :], atol=20 * max(n, 1) * EPS * scale)
    np.testing.assert_allclose(mH(v) @ v, np.broadcast_to(np.eye(n), v.shape), atol=20 * max(n, 1) * EPS)
    # The same LAPACK call (syevd/heevd, lower triangle) as numpy: the same bits.
    want = np.linalg.eigh(hc)
    assert np.array_equal(w, want.eigenvalues) and np.array_equal(v, want.eigenvectors)
    # Values only (a faster LAPACK mode): equal to rounding, not to the bit (DESIGN.md §2.14).
    wv = npy(sdla.eigvalsh(sd(h_in)))
    np.testing.assert_allclose(wv, w, rtol=0, atol=20 * max(n, 1) * EPS * scale)


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch_a=batch_shapes(),
    batch_b=batch_shapes(),
    n=st.integers(0, 5),
    da=st.sampled_from(["int64", "float64", "complex128"]),
    db=st.sampled_from(["float64", "complex128"]),
)
def test_eigh_generalized(seed, batch_a, batch_b, n, da, db):
    try:
        np.broadcast_shapes(batch_a, batch_b)
    except ValueError:
        batch_b = ()
    rng = np.random.default_rng(seed)
    a = hermitian(rng, batch_a, n, da)
    g = random_values(rng, (*batch_b, n, n), db)
    b = g @ mH(g) + n * np.eye(n) + np.eye(n)  # Hermitian positive definite
    ct = compute_dtype(da, db)
    r = sdla.eigh(sd(a), sd(poison_upper(b)))
    w, v = npy(r.eigenvalues), npy(r.eigenvectors)
    shape = np.broadcast_shapes(batch_a, batch_b)
    assert w.shape == (*shape, n) and v.dtype == ct
    ac, bc = a.astype(ct), b.astype(ct)
    scale = max(np.abs(ac).max(initial=0), np.abs(bc).max(initial=0), 1)
    # a v = w b v, normalized so that v^H b v = I
    np.testing.assert_allclose(ac @ v, (bc @ v) * w[..., None, :], atol=100 * max(n, 1) * EPS * scale**2)
    np.testing.assert_allclose(mH(v) @ bc @ v, np.broadcast_to(np.eye(n), v.shape), atol=100 * max(n, 1) * EPS * scale)
    wv = npy(sdla.eigvalsh(sd(a), sd(b)))
    np.testing.assert_allclose(wv, w, rtol=1e-10, atol=1e-12 * scale)
    if not shape and n:
        np.testing.assert_allclose(w, sl.eigh(ac, bc, eigvals_only=True), rtol=1e-10, atol=1e-12 * scale)


def test_eigh_rules():
    with pytest.raises(sdla.LinAlgError, match="^matrix is not positive definite$"):
        sdla.eigh(sdnp.eye(2), sd(np.array([[1.0, 0.0], [0.0, -1.0]])))
    with pytest.raises(sdla.LinAlgError, match="positive definite"):
        sdla.eigvalsh(sdnp.eye(2), sd(-np.eye(2)))
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sdla.eigh(sd(np.array([[np.nan, 0.0], [0.0, 1.0]])))
    with pytest.raises(ValueError, match="'b' contains NaN"):
        sdla.eigh(sdnp.eye(2), sd(np.array([[1.0, 0.0], [np.nan, 1.0]])))
    with pytest.raises(ValueError, match="'b' is 3x3"):
        sdla.eigh(sdnp.eye(2), sdnp.eye(3))
    with pytest.raises(ValueError, match="square"):
        sdla.eigvalsh(sdnp.ones((2, 3)))
    assert sdla.eigh(sdnp.zeros((0, 0))).eigenvectors.shape == (0, 0)
    assert sdla.eigvalsh(sdnp.zeros((2, 0, 0))).shape == (2, 0)
    # b is positional (API.md §4: eigh(a, b=None, /))
    with pytest.raises(TypeError):
        sdla.eigh(sdnp.eye(2), b=sdnp.eye(2))


# ---------------------------------------------------------- svd, svdvals --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(0, 6),
    n=st.integers(0, 6),
    mode=st.sampled_from(["reduced", "complete"]),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_svd(seed, batch, m, n, mode, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    ct = compute_dtype(dtype)
    r = sdla.svd(sd(x), mode=mode)
    assert type(r).__name__ == "SVDResult" and r._fields == ("U", "S", "Vh")
    U, S, Vh = npy(r.U), npy(r.S), npy(r.Vh)
    k = min(m, n)
    uc, vr = (m, n) if mode == "complete" else (k, k)
    assert U.shape == (*batch, m, uc) and S.shape == (*batch, k) and Vh.shape == (*batch, vr, n)
    assert U.dtype == ct and S.dtype == np.float64 and Vh.dtype == ct
    assert np.all(np.diff(S, axis=-1) <= 0) and np.all(S >= 0)
    scale = max(np.abs(x).max(initial=0), 1)
    np.testing.assert_allclose((U[..., :, :k] * S[..., None, :]) @ Vh[..., :k, :], x.astype(ct),
                               atol=20 * max(m, n, 1) * EPS * scale)
    np.testing.assert_allclose(mH(U) @ U, np.broadcast_to(np.eye(uc), (*batch, uc, uc)), atol=20 * max(m, 1) * EPS)
    np.testing.assert_allclose(Vh @ mH(Vh), np.broadcast_to(np.eye(vr), (*batch, vr, vr)), atol=20 * max(n, 1) * EPS)
    if k:
        # Not bit for bit: gesdd picks its internal path by the workspace size, and
        # numpy and scipy size it differently from LAPACK's own query, which we use
        # (DESIGN.md §2.14). Singular vectors are unique only up to phase, so
        # compare the singular values directly and the vectors through x = U S Vh above.
        want = np.linalg.svd(x.astype(ct), full_matrices=mode == "complete")
        np.testing.assert_allclose(S, want.S, rtol=0, atol=20 * max(m, n, 1) * EPS * scale)
    sv = npy(sdla.svdvals(sd(x)))
    np.testing.assert_allclose(sv, S, rtol=0, atol=20 * max(m, n, 1) * EPS * scale)


def test_svd_rules():
    u, s, vh = sdla.svd(sdnp.zeros((3, 0)), mode="complete")
    assert np.array_equal(npy(u), np.eye(3)) and s.shape == (0,) and vh.shape == (0, 0)
    u, s, vh = sdla.svd(sdnp.zeros((0, 2)), mode="complete")
    assert u.shape == (0, 0) and np.array_equal(npy(vh), np.eye(2))
    assert sdla.svd(sdnp.zeros((3, 0))).U.shape == (3, 0)
    with pytest.raises(ValueError, match="'reduced' or 'complete'"):
        sdla.svd(sdnp.eye(2), mode="full")
    with pytest.raises(ValueError, match="NaN"):
        sdla.svdvals(sd(np.array([[np.inf]])))


# ------------------------------------------------------ matrix_rank, pinv --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(max_dims=1),
    m=st.integers(0, 6),
    n=st.integers(0, 6),
    r=st.integers(1, 6),
    dtype=st.sampled_from(["float64", "complex128"]),
)
def test_matrix_rank_and_pinv(seed, batch, m, n, r, dtype):
    rng = np.random.default_rng(seed)
    x = low_rank(rng, batch, m, n, r, dtype)
    rank = sdla.matrix_rank(sd(x))
    want = np.linalg.matrix_rank(x)  # numpy's default tolerance is ours: max(m, n) * eps * s_max
    assert np.array_equal(np.asarray(rank), want)
    if not batch:
        assert type(rank) is int
    p = npy(sdla.pinv(sd(x)))
    assert p.shape == (*batch, n, m) and p.dtype == x.dtype
    np.testing.assert_allclose(p, np.linalg.pinv(x), rtol=1e-8, atol=1e-10 * max(np.abs(p).max(initial=0), 1))
    # Moore-Penrose: x p x = x
    np.testing.assert_allclose(x @ p @ x, x, atol=1e-9 * max(np.abs(x).max(initial=0), 1))
    # the rank lstsq reports agrees (both by singular values <= rtol * max)
    if m and n and not batch:
        assert sdla.lstsq(sd(x), sdnp.zeros((m,))).rank == rank


def test_rank_and_pinv_rtol():
    d = sd(np.diag([3.0, 1.0, 0.0]))
    assert sdla.matrix_rank(d) == 2
    assert sdla.matrix_rank(d, rtol=1 / 3) == 1  # equal to rtol * max counts as zero
    assert sdla.matrix_rank(d, rtol=0.3) == 2
    assert sdla.matrix_rank(d, rtol=1.0) == 0 and sdla.matrix_rank(d, rtol=5) == 0
    assert np.array_equal(npy(sdla.pinv(d, rtol=1.0)), np.zeros((3, 3)))
    np.testing.assert_allclose(npy(sdla.pinv(d, rtol=0.5)), np.diag([1 / 3, 0, 0]))
    assert sdla.matrix_rank(sdnp.zeros((3, 4))) == 0 and sdla.matrix_rank(sdnp.zeros((0, 4))) == 0
    assert sdla.pinv(sdnp.zeros((0, 4))).shape == (4, 0)
    assert np.array_equal(npy(sdla.pinv(sdnp.zeros((2, 3)))), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="rtol"):
        sdla.matrix_rank(d, rtol=-1)
    with pytest.raises(TypeError):
        sdla.pinv(d, 0.5)


# -------------------------------------------------------------------- cond --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(max_dims=1),
    n=st.integers(1, 6),
    dtype=st.sampled_from(["int64", "float64", "complex128"]),
    ord_=st.sampled_from([2, -2, "fro", "nuc", 1, -1, INF, -INF]),
)
def test_cond_matches_numpy(seed, batch, n, dtype, ord_):
    rng = np.random.default_rng(seed)
    x = random_values(rng, (*batch, n, n), dtype) + (3 * n + 1) * np.eye(n, dtype=np.int64)
    got = sdla.cond(sd(x), ord=ord_)
    want = np.linalg.cond(x.astype(compute_dtype(dtype)), p=ord_)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-10)
    if not batch:
        assert type(got) is float


def test_cond_rules():
    z = sdnp.zeros((3, 3))
    for o in (2, "fro", "nuc", 1, -1, INF, -INF):
        assert sdla.cond(z, ord=o) == INF  # a singular matrix's condition number (DESIGN.md §2.14)
    assert sdla.cond(z, ord=-2) == 0.0
    s = sd(np.array([[1.0, 2.0], [2.0, 4.0]]))
    assert sdla.cond(s, ord=1) == INF
    r = sd(np.random.default_rng(0).standard_normal((4, 2)))
    assert sdla.cond(r) == pytest.approx(np.linalg.cond(np.asarray(r)))  # 2-norm: any shape
    with pytest.raises(ValueError, match="square"):
        sdla.cond(r, ord=1)
    with pytest.raises(ValueError, match="empty"):
        sdla.cond(sdnp.zeros((0, 0)))
    with pytest.raises(ValueError, match="NaN"):
        sdla.cond(sd(np.array([[np.nan]])), ord=1)
    with pytest.raises(ValueError, match="ord"):
        sdla.cond(sdnp.eye(2), ord=3)


# ------------------------------------------------- matrix_norm 2, -2, nuc --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(1, 5),
    n=st.integers(1, 5),
    dtype=st.sampled_from(DTYPES),
    ord_=st.sampled_from([2, -2, "nuc"]),
    keepdims=st.booleans(),
)
def test_matrix_norm_svd_ords(seed, batch, m, n, dtype, ord_, keepdims):
    rng = np.random.default_rng(seed)
    x = random_values(rng, (*batch, m, n), dtype)
    got = sdla.matrix_norm(sd(x), ord=ord_, keepdims=keepdims)
    want = np.linalg.matrix_norm(x.astype(compute_dtype(dtype)), ord=ord_, keepdims=keepdims)
    assert np.shape(got) == np.shape(want)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-14)


def test_matrix_norm_svd_ords_rules():
    nan = sd(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    for o in (2, -2, "nuc"):
        with pytest.raises(ValueError, match="NaN"):  # LAPACK is called: checked (DESIGN.md §2.14)
            sdla.matrix_norm(nan, ord=o)
        assert sdla.matrix_norm(sdnp.zeros((3, 0)), ord=o) == 0.0
    assert math.isnan(sdla.matrix_norm(nan))  # 'fro' does not call LAPACK: NaN propagates


@pytest.mark.parametrize(
    "call",
    [
        lambda z: sdla.eigh(z),
        lambda z: sdla.eigvalsh(z),
        lambda z: sdla.svd(z),
        lambda z: sdla.svdvals(z),
        lambda z: sdla.matrix_rank(z),
        lambda z: sdla.pinv(z),
        lambda z: sdla.cond(z),
    ],
)
@pytest.mark.parametrize("bad", [[[1.0]], np.eye(2), None])
def test_only_sdnp_arrays(call, bad):
    with pytest.raises(TypeError, match="stardust_numpy.ndarray"):
        call(bad)
