"""qr, qr_pivoted, lstsq."""

import math
import os

import numpy as np
import pytest
import scipy.linalg as sl
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .conftest import DTYPES, LAYOUTS, as_sd, batch_shapes, compute_dtype, npy, random_values, with_layout

SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
EPS = np.finfo(np.float64).eps


def f_order_strides(shape, itemsize):
    m, n = shape[-2], shape[-1]
    st_ = [0] * len(shape)
    st_[-2], st_[-1] = itemsize, itemsize * m
    s = itemsize * m * n
    for d in range(len(shape) - 3, -1, -1):
        st_[d] = s
        s *= shape[d]
    return tuple(st_)


def mH(x):
    return np.conj(np.swapaxes(x, -1, -2))


# ---------------------------------------------------------------------- qr --


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
def test_qr(seed, batch, m, n, mode, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    ct = compute_dtype(dtype)
    r = sd.linalg.qr(as_sd(x), mode=mode)
    assert type(r).__name__ == "QRResult" and r._fields == ("Q", "R")
    Q, R = npy(r.Q), npy(r.R)
    k = min(m, n)
    qc = m if mode == "complete" else k
    assert Q.shape == (*batch, m, qc) and R.shape == (*batch, qc, n)
    assert Q.dtype == ct and R.dtype == ct
    assert r.Q.strides == f_order_strides(Q.shape, Q.itemsize)
    assert np.array_equal(R, np.triu(R))
    np.testing.assert_allclose(mH(Q) @ Q, np.broadcast_to(np.eye(qc), (*batch, qc, qc)), atol=10 * max(m, 1) * EPS)
    np.testing.assert_allclose(Q @ R, x.astype(ct), atol=10 * max(m, n, 1) * EPS * max(np.abs(x).max(initial=0), 1))
    # The same LAPACK calls (geqrf, orgqr) as numpy, so the same bits.
    want = np.linalg.qr(x.astype(ct), mode=mode)
    assert np.array_equal(Q, want.Q) and np.array_equal(R, want.R)


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
def test_qr_pivoted(seed, batch, m, n, mode, dtype, layout):
    rng = np.random.default_rng(seed)
    x = with_layout(random_values(rng, (*batch, m, n), dtype), layout)
    ct = compute_dtype(dtype)
    r = sd.linalg.qr_pivoted(as_sd(x), mode=mode)
    assert type(r).__name__ == "QRPivotedResult" and r._fields == ("Q", "R", "perm")
    Q, R, p = npy(r.Q), npy(r.R), npy(r.perm)
    assert p.dtype == np.int64 and p.shape == (*batch, n)
    assert np.all(np.sort(p, axis=-1) == np.arange(n))
    xp = np.take_along_axis(x.astype(ct), p[..., None, :], axis=-1)  # x[:, perm]
    np.testing.assert_allclose(Q @ R, xp, atol=10 * max(m, n, 1) * EPS * max(np.abs(x).max(initial=0), 1))
    d = np.abs(np.diagonal(R, axis1=-2, axis2=-1))
    assert np.all(d[..., :-1] >= d[..., 1:] * (1 - 1e-12))  # column pivoting: |R_jj| non-increasing
    if not batch and m and n:
        _, sr, sp = sl.qr(x.astype(ct), mode="economic" if mode == "reduced" else "full", pivoting=True)
        assert np.array_equal(p, sp)
        np.testing.assert_allclose(R, sr, rtol=1e-12, atol=1e-14)


def test_qr_rules():
    with pytest.raises(ValueError, match="'reduced' or 'complete'"):
        sd.linalg.qr(sd.eye(2), mode="r")
    with pytest.raises(TypeError):
        sd.linalg.qr(sd.eye(2), mode=1)
    with pytest.raises(TypeError):
        sd.linalg.qr(sd.eye(2), "complete")  # keyword-only
    with pytest.raises(ValueError, match="NaN"):
        sd.linalg.qr(as_sd(np.array([[np.nan]])))
    with pytest.raises(ValueError, match="2 dimensions"):
        sd.linalg.qr_pivoted(sd.ones((3,)))
    # empty: complete Q of an (m, 0) matrix is the identity
    q, r = sd.linalg.qr(sd.zeros((3, 0)), mode="complete")
    assert np.array_equal(npy(q), np.eye(3)) and r.shape == (3, 0)
    assert sd.linalg.qr(sd.zeros((0, 3))).Q.shape == (0, 0)
    assert sd.linalg.qr_pivoted(sd.zeros((2, 0, 3))).perm.shape == (2, 3)


# ------------------------------------------------------------------- lstsq --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(max_dims=1),
    m=st.integers(0, 7),
    n=st.integers(0, 7),
    k=st.integers(0, 3),
    vector=st.booleans(),
    deficient=st.booleans(),
    da=st.sampled_from(DTYPES),
    db=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
)
def test_lstsq_matches_numpy(seed, batch, m, n, k, vector, deficient, da, db, layout):
    rng = np.random.default_rng(seed)
    a = random_values(rng, (*batch, m, n), da)
    if deficient and n >= 2:  # exactly rank deficient: the last column repeats the first
        a[..., -1] = a[..., 0]
    a = with_layout(a, layout)
    b = random_values(rng, (m,) if vector else (*batch, m, k), db)
    ct = compute_dtype(da, db)
    r = sd.linalg.lstsq(as_sd(a), as_sd(b))
    assert type(r).__name__ == "LstsqResult" and r._fields == ("solution", "rank", "singular_values")
    x, s = npy(r.solution), npy(r.singular_values)
    assert x.dtype == ct and s.dtype == np.float64
    assert s.shape == (*batch, min(m, n))
    if not batch:
        assert type(r.rank) is int
    else:
        assert npy(r.rank).dtype == np.int64 and npy(r.rank).shape == batch
    # numpy's lstsq is not batched: compare element by element
    ra, rb = a.astype(ct), b.astype(ct)
    for idx in np.ndindex(*batch):
        bb = rb if vector else rb[idx]
        wx, _, wrank, ws = np.linalg.lstsq(ra[idx], bb)  # numpy's default rcond is max(m, n) * eps, as ours
        gx = x[idx] if batch else x
        gr = npy(r.rank)[idx] if batch else r.rank
        assert gr == wrank
        np.testing.assert_allclose(gx, wx, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(s[idx] if batch else s, ws, rtol=1e-12, atol=1e-13)


def test_lstsq_minimum_norm_and_rank():
    a = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]])  # rank 1
    r = sd.linalg.lstsq(as_sd(a), as_sd(np.array([2.0, 2.0, 0.0])))
    assert r.rank == 1
    np.testing.assert_allclose(npy(r.solution), [1.0, 1.0])  # the minimum-norm one of all x1 + x2 = 2
    # rtol >= 1 treats every singular value as zero: rank 0, solution 0. (LAPACK's
    # xLALSD would silently use machine epsilon instead; numpy inherits that.)
    for rtol in (1.0, 1.5, 1e10):
        r = sd.linalg.lstsq(as_sd(a), as_sd(np.array([2.0, 2.0, 0.0])), rtol=rtol)
        assert r.rank == 0 and np.array_equal(npy(r.solution), [0.0, 0.0])
        assert npy(r.singular_values)[0] == pytest.approx(2.0)
    # the boundary: a singular value equal to rtol * max counts as zero
    d = as_sd(np.array([[3.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    assert sd.linalg.lstsq(d, sd.ones((3,)), rtol=1 / 3).rank == 1
    assert sd.linalg.lstsq(d, sd.ones((3,)), rtol=0.3).rank == 2
    # underdetermined: the minimum-norm solution
    r = sd.linalg.lstsq(as_sd(np.array([[1.0, 2.0]])), as_sd(np.array([5.0])))
    np.testing.assert_allclose(npy(r.solution), [1.0, 2.0])


def test_lstsq_rules():
    a = sd.ones((3, 2))
    with pytest.raises(ValueError, match="'a' has 3 rows but 'b' has 2 rows"):
        sd.linalg.lstsq(a, sd.ones((2,)))
    for bad in (-1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="rtol"):
            sd.linalg.lstsq(a, sd.ones((3,)), rtol=bad)
    for bad in (True, "1e-10"):
        with pytest.raises(TypeError, match="rtol"):
            sd.linalg.lstsq(a, sd.ones((3,)), rtol=bad)
    with pytest.raises(ValueError, match="'a' contains NaN"):
        sd.linalg.lstsq(as_sd(np.array([[np.nan]])), sd.ones((1,)))
    with pytest.raises(ValueError, match="'b' contains NaN"):
        sd.linalg.lstsq(a, as_sd(np.array([1.0, np.nan, 1.0])))
    # empty systems: the minimum-norm solution is zero, rank 0
    r = sd.linalg.lstsq(sd.zeros((0, 3)), sd.zeros((0,)))
    assert r.rank == 0 and npy(r.solution).tolist() == [0.0, 0.0, 0.0] and r.singular_values.shape == (0,)
    r = sd.linalg.lstsq(sd.zeros((3, 0)), sd.ones((3, 2)))
    assert r.solution.shape == (0, 2) and r.rank == 0


@pytest.mark.parametrize(
    "call", [lambda z: sd.linalg.qr(z), lambda z: sd.linalg.qr_pivoted(z), lambda z: sd.linalg.lstsq(z, z)]
)
@pytest.mark.parametrize("bad", [[[1.0]], np.eye(2), None])
def test_only_sd_arrays(call, bad):
    with pytest.raises(TypeError, match=r"^\w+\(\): '\w+' must be an sd\.ndarray, not "):
        call(bad)
