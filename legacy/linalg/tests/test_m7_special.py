"""M7: solve_tridiagonal, solve_banded, solve_sylvester."""

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
EPS = np.finfo(np.float64).eps
VEC_LAYOUTS = ["C", "strided", "reversed"]


def vec_layout(x, layout):
    """The same values as `x` (numpy, ndim >= 1) with a different last-axis stride."""
    if layout == "C":
        return np.ascontiguousarray(x)
    if layout == "strided":
        big = np.zeros((*x.shape[:-1], 2 * x.shape[-1] + 1), x.dtype)
        big[..., 1::2] = x
        return big[..., 1::2]
    return np.ascontiguousarray(x[..., ::-1])[..., ::-1]


def tridiagonal_dense(dl, d, du):
    """The dense (..., n, n) matrices of broadcast dl, d, du."""
    n = d.shape[-1]
    batch = np.broadcast_shapes(dl.shape[:-1], d.shape[:-1], du.shape[:-1])
    a = np.zeros((*batch, n, n), np.result_type(dl, d, du, np.float64))
    i = np.arange(n)
    a[..., i, i] = d
    a[..., i[1:], i[:-1]] = dl
    a[..., i[:-1], i[1:]] = du
    return a


def tri_parts(rng, batch, n, dtypes):
    """dl, d, du, safely nonsingular in every dtype (a dominant diagonal; bool: unit lower bidiagonal)."""
    dt_l, dt_d, dt_u = dtypes
    m = max(n - 1, 0)
    dl = random_values(rng, (*batch, m), dt_l)
    du = random_values(rng, (*batch, m), dt_u)
    if dt_d == "bool":  # unit lower bidiagonal: determinant 1
        d = np.ones((*batch, n), np.bool_)
        du = np.zeros_like(du)
    else:  # |d| >= 7 > |dl| + |du| for int entries in [-3, 3]; ample for normal floats
        d = random_values(rng, (*batch, n), dt_d) + 10
    return dl, d, du


# ------------------------------------------------------- solve_tridiagonal --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 7),
    dtypes=st.tuples(*[st.sampled_from(DTYPES)] * 4),
    layouts=st.tuples(*[st.sampled_from(VEC_LAYOUTS)] * 3),
    b_layout=st.sampled_from(LAYOUTS),
    k=st.one_of(st.none(), st.integers(0, 3)),
    bcast=st.integers(0, 3),
)
def test_solve_tridiagonal(seed, batch, n, dtypes, layouts, b_layout, k, bcast):
    rng = np.random.default_rng(seed)
    dl, d, du = tri_parts(rng, batch, n, dtypes[:3])
    # one of the four arguments without batch axes (broadcast)
    parts = [dl, d, du]
    if bcast < 3 and batch:
        parts[bcast] = parts[bcast][(0,) * len(batch)] if 0 not in batch else parts[bcast]
    dl, d, du = (vec_layout(p, lay) for p, lay in zip(parts, layouts))
    if k is None:
        b = random_values(rng, (n,), dtypes[3])
    else:
        b = with_layout(random_values(rng, (*batch, n, k), dtypes[3]), b_layout)
    x = npy(sdla.solve_tridiagonal(sd(dl), sd(d), sd(du), sd(b)))
    want_dt = compute_dtype(*dtypes)
    assert x.dtype == want_dt
    a = tridiagonal_dense(dl, d, du)
    out_batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2] if k is not None else ())
    assert x.shape == ((*out_batch, n) if k is None else (*out_batch, n, k))
    a_b = np.broadcast_to(a, (*out_batch, n, n))
    b_b = np.broadcast_to(b, x.shape)
    assert residual_ok(a_b, x, b_b)
    # bit for bit equal to scipy.linalg.solve_banded((1, 1), ...), which calls gtsv too
    if n >= 2 and x.size:
        for idx in np.ndindex(*out_batch):
            ab = np.zeros((3, n), want_dt)
            ai = a_b[idx]
            ab[0, 1:], ab[1], ab[2, :-1] = np.diag(ai, 1), np.diag(ai), np.diag(ai, -1)
            want = sl.solve_banded((1, 1), ab, np.asarray(b_b[idx], want_dt))
            assert np.array_equal(x[idx], want)


def test_tridiagonal_rules():
    rng = np.random.default_rng(0)
    dl, d, du = rng.standard_normal(4), rng.standard_normal(5) + 5, rng.standard_normal(4)
    # a 1-D b is one vector, broadcast against the batch of the diagonals
    x = npy(sdla.solve_tridiagonal(sd(np.stack([dl, dl])), sd(d), sd(du), sd(np.ones(5))))
    assert x.shape == (2, 5)
    # empty: n = 0 takes length-0 off-diagonals
    assert npy(sdla.solve_tridiagonal(sd(np.zeros(0)), sd(np.zeros(0)), sd(np.zeros(0)), sd(np.ones((0, 2))))).shape == (0, 2)
    with pytest.raises(ValueError, match=r"'dl' must have length 4 \(one less than 'd'\), got 5"):
        sdla.solve_tridiagonal(sd(np.zeros(5)), sd(d), sd(du), sd(np.ones(5)))
    with pytest.raises(ValueError, match=r"'du' must have length 4"):
        sdla.solve_tridiagonal(sd(dl), sd(d), sd(np.zeros(3)), sd(np.ones(5)))
    with pytest.raises(ValueError, match=r"the matrix is 5x5 but 'b' has 4 rows"):
        sdla.solve_tridiagonal(sd(dl), sd(d), sd(du), sd(np.ones(4)))
    with pytest.raises(TypeError):
        sdla.solve_tridiagonal(dl, sd(d), sd(du), sd(np.ones(5)))
    # NaN: named by argument, in argument order
    for i, name in enumerate(["dl", "d", "du", "b"]):
        args = [dl.copy(), d.copy(), du.copy(), np.ones(5)]
        args[i][1] = np.nan
        with pytest.raises(ValueError, match=f"'{name}' contains NaN or infinity"):
            sdla.solve_tridiagonal(*map(sd, args))
    # an exact zero pivot is a LinAlgError, reported before a NaN in b (as in solve)
    z = np.zeros(5)
    with pytest.raises(sdla.LinAlgError, match="singular matrix"):
        sdla.solve_tridiagonal(sd(np.zeros(4)), sd(z), sd(np.zeros(4)), sd(np.full(5, np.nan)))
    dd = np.stack([d, d, z])
    with pytest.raises(sdla.LinAlgError, match=r"singular matrix \(at batch index \(2,\)\)"):
        sdla.solve_tridiagonal(sd(np.zeros(4)), sd(dd), sd(np.zeros(4)), sd(np.ones(5)))


# ------------------------------------------------------------ solve_banded --


def band_parts(rng, batch, n, kl, ku, dtype):
    """Band storage of a safely nonsingular band matrix (bool: unit lower band);
    ab's corners outside the matrix are NaN (float) or junk (int, bool)."""
    a = well_conditioned(rng, batch, n, dtype) if n else np.zeros((*batch, 0, 0), {"bool": np.bool_, "int64": np.int64, "float64": np.float64, "complex128": np.complex128}[dtype])
    i, j = np.indices((n, n))
    a = np.where((i - j <= kl) & (j - i <= ku), a, np.zeros((), a.dtype))
    if dtype in ("float64", "complex128"):
        ab = np.full((*batch, kl + ku + 1, n), np.nan, a.dtype)
    else:
        ab = random_values(rng, (*batch, kl + ku + 1, n), dtype)
    for jj in range(n):
        for ii in range(max(0, jj - ku), min(n, jj + kl + 1)):
            ab[..., ku + ii - jj, jj] = a[..., ii, jj]
    return ab


def band_dense(ab, kl, ku):
    """The dense matrices that band storage `ab` describes."""
    n = ab.shape[-1]
    a = np.zeros((*ab.shape[:-2], n, n), ab.dtype)
    for jj in range(n):
        for ii in range(max(0, jj - ku), min(n, jj + kl + 1)):
            a[..., ii, jj] = ab[..., ku + ii - jj, jj]
    return a


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 7),
    kl=st.integers(0, 4),
    ku=st.integers(0, 4),
    dtypes=st.tuples(st.sampled_from(DTYPES), st.sampled_from(DTYPES)),
    layout=st.sampled_from(LAYOUTS),
    k=st.one_of(st.none(), st.integers(0, 3)),
)
def test_solve_banded(seed, batch, n, kl, ku, dtypes, layout, k):
    rng = np.random.default_rng(seed)
    ab = with_layout(band_parts(rng, batch, n, kl, ku, dtypes[0]), layout)
    a = band_dense(ab, kl, ku)  # after the layout: "bcast" repeats batch element 0
    b = random_values(rng, (n,), dtypes[1]) if k is None else random_values(rng, (*batch, n, k), dtypes[1])
    x = npy(sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=kl, upper_bandwidth=ku))
    want_dt = compute_dtype(*dtypes)
    assert x.dtype == want_dt
    assert x.shape == ((*batch, n) if k is None else (*batch, n, k))
    b_b = np.broadcast_to(b, x.shape)
    assert residual_ok(a, x, b_b)
    # bit for bit equal to scipy (gbsv; gtsv for (1, 1)); scipy divides directly when n = 1
    if n >= 2 and x.size:
        for idx in np.ndindex(*batch):
            abz = np.nan_to_num(np.asarray(ab[idx], want_dt))  # scipy copies the corners into gbsv's input
            want = sl.solve_banded((kl, ku), abz, np.asarray(b_b[idx], want_dt), check_finite=False)
            assert np.array_equal(x[idx], want)


def test_banded_rules():
    rng = np.random.default_rng(1)
    dl, d, du = rng.standard_normal(5), rng.standard_normal(6) + 5, rng.standard_normal(5)
    ab = np.zeros((3, 6))
    ab[0, 1:], ab[1], ab[2, :-1] = du, d, dl
    b = rng.standard_normal((6, 2))
    # (1, 1) goes through gtsv: the same bits as solve_tridiagonal (DESIGN.md §2.16)
    x1 = npy(sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=1, upper_bandwidth=1))
    x2 = npy(sdla.solve_tridiagonal(sd(dl), sd(d), sd(du), sd(b)))
    assert np.array_equal(x1, x2)
    # (1, 1) does not read the corners either
    abn = ab.copy()
    abn[0, 0] = abn[2, -1] = np.nan
    assert np.array_equal(npy(sdla.solve_banded(sd(abn), sd(b), lower_bandwidth=1, upper_bandwidth=1)), x1)
    # (0, 0): a diagonal matrix
    x = npy(sdla.solve_banded(sd(d[None, :]), sd(d), lower_bandwidth=0, upper_bandwidth=0))
    assert np.array_equal(x, np.ones(6))
    # bandwidths wider than the matrix are fine (the extra rows are never read)
    ab_wide = np.full((9, 2), np.nan)
    ab_wide[4] = [2.0, 4.0]  # the diagonal
    ab_wide[3, 1] = ab_wide[5, 0] = 0.0  # a[0, 1], a[1, 0]: the only other elements inside the matrix
    assert npy(sdla.solve_banded(sd(ab_wide), sd(np.array([2.0, 8.0])), lower_bandwidth=4, upper_bandwidth=4)).tolist() == [1.0, 2.0]
    with pytest.raises(ValueError, match=r"'ab' must have lower_bandwidth \+ upper_bandwidth \+ 1 rows \(2 \+ 1 \+ 1\), got 3"):
        sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=2, upper_bandwidth=1)
    with pytest.raises(ValueError, match="must have lower_bandwidth"):  # no overflow in the sum
        sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=2**62, upper_bandwidth=2**62)
    with pytest.raises(ValueError, match="bandwidths must be >= 0"):
        sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=-1, upper_bandwidth=3)
    with pytest.raises(TypeError):
        sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=True, upper_bandwidth=1)
    with pytest.raises(TypeError):
        sdla.solve_banded(sd(ab), sd(b), 1, 1)  # keyword-only
    with pytest.raises(TypeError):
        sdla.solve_banded(sd(ab), sd(b), lower_bandwidth=1)  # both required
    with pytest.raises(ValueError, match=r"the matrix is 6x6 but 'b' has 5 rows"):
        sdla.solve_banded(sd(ab), sd(b[:5]), lower_bandwidth=1, upper_bandwidth=1)
    ab2 = np.zeros((4, 6))
    ab2[1] = 1.0
    ab2[3, :-2] = np.nan  # inside the matrix: read and checked
    with pytest.raises(ValueError, match="'ab' contains NaN or infinity"):
        sdla.solve_banded(sd(ab2), sd(b), lower_bandwidth=2, upper_bandwidth=1)
    with pytest.raises(ValueError, match="'b' contains NaN or infinity"):
        sdla.solve_banded(sd(np.ones((1, 6))), sd(np.full(6, np.nan)), lower_bandwidth=0, upper_bandwidth=0)
    # singular first, then b (as in solve)
    with pytest.raises(sdla.LinAlgError, match="singular matrix"):
        sdla.solve_banded(sd(np.zeros((4, 6))), sd(np.full(6, np.nan)), lower_bandwidth=2, upper_bandwidth=1)
    abs_ = np.stack([np.ones((1, 6)), np.zeros((1, 6))])
    with pytest.raises(sdla.LinAlgError, match=r"singular matrix \(at batch index \(1,\)\)"):
        sdla.solve_banded(sd(abs_), sd(np.ones(6)), lower_bandwidth=0, upper_bandwidth=0)
    assert npy(sdla.solve_banded(sd(np.zeros((3, 0))), sd(np.zeros(0)), lower_bandwidth=1, upper_bandwidth=1)).shape == (0,)


# --------------------------------------------------------- solve_sylvester --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    m=st.integers(0, 6),
    n=st.integers(0, 6),
    dtypes=st.tuples(*[st.sampled_from(DTYPES)] * 3),
    layouts=st.tuples(*[st.sampled_from(LAYOUTS)] * 3),
)
def test_solve_sylvester(seed, batch, m, n, dtypes, layouts):
    rng = np.random.default_rng(seed)
    # well_conditioned: eigenvalues with positive real parts, so a and -b never share one
    a = with_layout(well_conditioned(rng, batch, m, dtypes[0]), layouts[0])
    b = with_layout(well_conditioned(rng, batch, n, dtypes[1]), layouts[1])
    c = with_layout(random_values(rng, (*batch, m, n), dtypes[2]), layouts[2])
    x = npy(sdla.solve_sylvester(sd(a), sd(b), sd(c)))
    want_dt = compute_dtype(*dtypes)
    assert x.dtype == want_dt and x.shape == (*batch, m, n)
    if x.size:
        assert x.strides[-2] == x.itemsize and x.strides[-1] == x.itemsize * m  # F-order
        af, bf, cf = (np.asarray(t, want_dt) for t in (a, b, c))
        r = af @ x + x @ bf - cf
        fro = lambda t: np.linalg.norm(t, axis=(-2, -1))  # noqa: E731
        assert np.all(fro(r) <= 100 * (m + n) * EPS * ((fro(af) + fro(bf)) * fro(x) + fro(cf)))
        if not batch and dtypes[0] == dtypes[1] == dtypes[2]:
            np.testing.assert_allclose(x, sl.solve_sylvester(af, bf, cf), rtol=1e-9, atol=1e-12)


def test_sylvester_rules():
    rng = np.random.default_rng(2)
    a, b, c = rng.standard_normal((3, 3)), rng.standard_normal((2, 2)), rng.standard_normal((3, 2))
    # the scalar case: x = c / (a + b)
    assert npy(sdla.solve_sylvester(sd(np.array([[2.0]])), sd(np.array([[3.0]])), sd(np.array([[10.0]])))).tolist() == [[2.0]]
    # a and -b with a common eigenvalue: no solution (or infinitely many)
    with pytest.raises(sdla.LinAlgError, match="a and -b have a common or close eigenvalue"):
        sdla.solve_sylvester(sd(np.eye(2)), sd(-np.eye(2)), sd(np.ones((2, 2))))
    aa = np.stack([np.eye(2), np.eye(2)])
    bb = np.stack([np.eye(2), -np.eye(2)])
    with pytest.raises(sdla.LinAlgError, match=r"close eigenvalue \(at batch index \(1,\)\)"):
        sdla.solve_sylvester(sd(aa), sd(bb), sd(np.ones((2, 2))))
    with pytest.raises(ValueError, match="'c' must be 3x2 to match 'a' and 'b', got 3x3"):
        sdla.solve_sylvester(sd(a), sd(b), sd(np.ones((3, 3))))
    with pytest.raises(ValueError):
        sdla.solve_sylvester(sd(np.ones((3, 2))), sd(b), sd(c))  # a not square
    with pytest.raises(ValueError):
        sdla.solve_sylvester(sd(a), sd(b), sd(np.ones(3)))  # c must be a matrix
    for i, name in enumerate(["a", "b", "c"]):
        args = [a.copy(), b.copy(), c.copy()]
        args[i][0, 0] = np.inf
        with pytest.raises(ValueError, match=f"'{name}' contains NaN or infinity"):
            sdla.solve_sylvester(*map(sd, args))
    # empty
    assert npy(sdla.solve_sylvester(sd(np.zeros((0, 0))), sd(b), sd(np.zeros((0, 2))))).shape == (0, 2)
    assert npy(sdla.solve_sylvester(sd(a), sd(np.zeros((0, 0))), sd(np.zeros((3, 0))))).shape == (3, 0)
    # a real a and a complex b: computed in complex (API.md §0.3), with a's complex
    # Schur form. This a has a conjugate pair, where scipy 1.18.1 goes wrong
    # (a real quasi-triangular form fed to the complex trsyl, DESIGN.md §2.16).
    ar = np.array([[0.0, -2.0, 1.0], [1.0, 0.0, 0.5], [0.0, 0.3, 1.0]])
    bc = b + 1j * rng.standard_normal((2, 2))
    x = npy(sdla.solve_sylvester(sd(ar), sd(bc), sd(c)))
    assert x.dtype == np.complex128
    assert np.abs(ar @ x + x @ bc - c).max() < 1e-12
    # layout does not change the bits
    xf = npy(sdla.solve_sylvester(sd(np.asfortranarray(a)), sd(b[::-1, ::-1].copy()[::-1, ::-1]), sd(c)))
    assert np.array_equal(xf, npy(sdla.solve_sylvester(sd(a), sd(b), sd(c))))
