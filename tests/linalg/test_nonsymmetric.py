"""eig, eigvals, schur, hessenberg, ldl."""

import numpy as np
import pytest
import scipy.linalg as sl
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples
from .conftest import DTYPES, LAYOUTS, as_sd, batch_shapes, compute_dtype, npy, random_values, with_layout

SETTINGS = settings(max_examples=examples(150), deadline=None)
EPS = np.finfo(np.float64).eps


def mH(x):
    return np.conj(np.swapaxes(x, -1, -2))


def f_order_strides(shape, itemsize):
    m, n = shape[-2], shape[-1]
    st_ = [0] * len(shape)
    st_[-2], st_[-1] = itemsize, itemsize * m
    s = itemsize * m * n
    for d in range(len(shape) - 3, -1, -1):
        st_[d] = s
        s *= shape[d]
    return tuple(st_)


def lex_sorted(w):
    """Sorted by (real, imag) along the last axis, stably (docs/api/linalg.md §5)."""
    order = np.lexsort((w.imag, w.real), axis=-1)
    return np.take_along_axis(w, order, axis=-1)


# ----------------------------------------------------------- eig, eigvals --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_eig(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, n, n), dtype), layout)
    r = sd.linalg.eig(as_sd(x))
    assert type(r).__name__ == "EigResult" and r._fields == ("eigenvalues", "eigenvectors")
    w, v = npy(r.eigenvalues), npy(r.eigenvectors)
    # always complex128, sorted by (real, imag) (docs/api/linalg.md §1.5, §5)
    assert w.dtype == np.complex128 and v.dtype == np.complex128
    assert w.shape == (*batch, n) and v.shape == (*batch, n, n)
    assert r.eigenvectors.strides == f_order_strides(v.shape, 16)
    assert np.array_equal(w, lex_sorted(w))
    xc = x.astype(np.complex128)
    scale = max(np.abs(x).max(initial=0), 1)
    # A v = v w (a backward-stable residual, whatever the conditioning)
    np.testing.assert_allclose(xc @ v, v * w[..., None, :], atol=50 * max(n, 1) * EPS * scale)
    np.testing.assert_allclose(np.linalg.norm(v, axis=-2), np.ones((*batch, n)), atol=1e-13)
    if dtype != "complex128":  # a real matrix's eigenvalues come in exact conjugate pairs
        assert np.array_equal(lex_sorted(np.conj(w)), w)
    wv = npy(sd.linalg.eigvals(as_sd(x)))
    assert wv.dtype == np.complex128 and np.array_equal(wv, lex_sorted(wv))
    np.testing.assert_allclose(wv, w, atol=1e-7 * scale)  # values-only mode: rounding-level agreement
    want = lex_sorted(np.linalg.eigvals(x.astype(compute_dtype(dtype))).astype(np.complex128))
    np.testing.assert_allclose(w, want, atol=1e-7 * scale)


def test_eig_order_and_rules():
    # a rotation: eigenvalues -i, i (negative imaginary part first)
    w = npy(sd.linalg.eigvals(as_sd(np.array([[0.0, -1.0], [1.0, 0.0]]))))
    assert w.tolist() == [-1j, 1j]
    w = npy(sd.linalg.eigvals(as_sd(np.diag([3.0, -1.0, 2.0]))))
    assert w.tolist() == [-1, 2, 3] and w.dtype == np.complex128
    with pytest.raises(ValueError, match="NaN"):
        sd.linalg.eig(as_sd(np.array([[np.nan]])))
    with pytest.raises(ValueError, match="square"):
        sd.linalg.eigvals(sd.ones((2, 3)))
    assert sd.linalg.eig(sd.zeros((0, 0))).eigenvalues.shape == (0,)
    assert npy(sd.linalg.eigvals(sd.zeros((2, 0, 0)))).dtype == np.complex128


# ------------------------------------------------------ schur, hessenberg --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_schur(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, n, n), dtype), layout)
    ct = compute_dtype(dtype)
    r = sd.linalg.schur(as_sd(x))
    assert type(r).__name__ == "SchurResult" and r._fields == ("T", "Z")
    T, Z = npy(r.T), npy(r.Z)
    assert T.dtype == ct and Z.dtype == ct and T.shape == x.shape
    scale = max(np.abs(x).max(initial=0), 1)
    np.testing.assert_allclose(Z @ T @ mH(Z), x.astype(ct), atol=50 * max(n, 1) * EPS * scale)
    np.testing.assert_allclose(mH(Z) @ Z, np.broadcast_to(np.eye(n), Z.shape), atol=50 * max(n, 1) * EPS)
    assert np.array_equal(np.tril(T, -2), np.zeros_like(T))
    sub = np.diagonal(T, offset=-1, axis1=-2, axis2=-1)
    if ct == np.complex128:
        assert np.all(sub == 0)  # complex Schur: triangular
    else:  # real Schur: 2x2 blocks only, never two in a row
        assert not np.any((sub[..., :-1] != 0) & (sub[..., 1:] != 0))
    if not batch and n:  # the same gees call as scipy
        st_, sz = sl.schur(x.astype(ct), output="complex" if ct == np.complex128 else "real")
        assert np.array_equal(T, st_) and np.array_equal(Z, sz)


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_hessenberg(seed, batch, n, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, n, n), dtype), layout)
    ct = compute_dtype(dtype)
    r = sd.linalg.hessenberg(as_sd(x))
    assert type(r).__name__ == "HessenbergResult" and r._fields == ("H", "Q")
    H, Q = npy(r.H), npy(r.Q)
    assert H.dtype == ct and Q.dtype == ct
    assert np.array_equal(np.tril(H, -2), np.zeros_like(H))
    scale = max(np.abs(x).max(initial=0), 1)
    np.testing.assert_allclose(Q @ H @ mH(Q), x.astype(ct), atol=50 * max(n, 1) * EPS * scale)
    np.testing.assert_allclose(mH(Q) @ Q, np.broadcast_to(np.eye(n), Q.shape), atol=50 * max(n, 1) * EPS)
    # scipy returns n <= 2 matrices unchanged (H = x, Q = I); LAPACK's zgehrd still
    # applies a length-1 reflector that makes a complex subdiagonal real. Both are
    # Hessenberg forms of x, so compare with scipy only where it calls LAPACK too.
    if not batch and (n >= 3 or (n and ct == np.float64)):
        sh, sq = sl.hessenberg(x.astype(ct), calc_q=True)
        np.testing.assert_allclose(H, sh, rtol=1e-13, atol=1e-13 * scale)
        np.testing.assert_allclose(Q, sq, rtol=1e-13, atol=1e-13)


# --------------------------------------------------------------------- ldl --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 7),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
    small_diag=st.booleans(),
)
def test_ldl(seed, batch, n, dtype, layout, small_diag):
    rng = np.random.default_rng(seed)
    a = random_values(rng, (*batch, n, n), dtype)
    if dtype == "bool":
        a = a.astype(np.int64)
    h = a + mH(a) if dtype == "complex128" else a + np.swapaxes(a, -1, -2)
    if small_diag and dtype in ("float64", "complex128"):  # forces 2x2 pivots
        h = h - np.diagonal(h, axis1=-2, axis2=-1)[..., None, :] * np.eye(n) * 0.999
    if dtype == "bool":
        h = h.astype(bool)  # 0/1 symmetric
    ct = compute_dtype(dtype)
    h = np.ascontiguousarray(with_layout(h, layout)) if layout == "bcast" else h
    h_in = h.astype(ct)
    if dtype in ("float64", "complex128"):  # NaN above the diagonal: not read (docs/api/linalg.md §1.4)
        h_in = h_in.copy()
        h_in[..., np.triu(np.ones((n, n), bool), 1)] = np.nan
    r = sd.linalg.ldl(as_sd(with_layout(h_in if dtype in ("float64", "complex128") else h, layout)))
    assert type(r).__name__ == "LDLResult" and r._fields == ("L", "D", "perm")
    L, D, p = npy(r.L), npy(r.D), npy(r.perm)
    assert L.dtype == ct and D.dtype == ct and p.dtype == np.int64 and p.shape == (*batch, n)
    assert np.array_equal(np.tril(L), L) and np.all(np.diagonal(L, axis1=-2, axis2=-1) == 1)
    assert np.all(np.sort(p, axis=-1) == np.arange(n))
    assert np.array_equal(np.tril(np.triu(D, -1), 1), D) and np.allclose(D, mH(D))  # Hermitian, tridiagonal blocks
    hc = h.astype(ct)
    hp = np.take_along_axis(np.take_along_axis(hc, p[..., :, None], axis=-2), p[..., None, :], axis=-1)
    scale = max(np.abs(hc).max(initial=0), 1)
    np.testing.assert_allclose(L @ D @ mH(L), hp, atol=200 * max(n, 1) * EPS * scale)
    if not batch and n:  # scipy.linalg.ldl, read the same way (docs/design/linalg.md §10.6)
        sL, sD, sp = sl.ldl(hc, lower=True, hermitian=True)
        assert np.array_equal(p, sp) and np.array_equal(L, sL[sp]) and np.array_equal(D, sD)


def test_ldl_rules():
    r = sd.linalg.ldl(sd.zeros((3, 3)))  # singular: no error (like lu)
    assert np.array_equal(npy(r.D), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="NaN"):
        sd.linalg.ldl(as_sd(np.array([[1.0, 0.0], [np.nan, 1.0]])))
    assert sd.linalg.ldl(sd.zeros((0, 0))).perm.shape == (0,)


@pytest.mark.parametrize(
    "call",
    [
        lambda z: sd.linalg.eig(z),
        lambda z: sd.linalg.eigvals(z),
        lambda z: sd.linalg.schur(z),
        lambda z: sd.linalg.hessenberg(z),
        lambda z: sd.linalg.ldl(z),
    ],
)
@pytest.mark.parametrize("bad", [[[1.0]], np.eye(2), None])
def test_only_sd_arrays(call, bad):
    with pytest.raises(TypeError, match=r"^\w+\(\): '\w+' must be an sd\.ndarray, not "):
        call(bad)
