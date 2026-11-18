"""kron, cross, expm (docs/api/linalg.md §2, §7; docs/design/linalg.md §10.8, §11)."""

import hashlib
import math
import os
import warnings

import numpy as np
import pytest
import scipy.linalg as sl
import stardust as sd
from .conftest import DTYPES, LAYOUTS, batch_shapes, compute_dtype, npy, random_values, as_sd, with_layout
from hypothesis import given, settings
from hypothesis import strategies as st

from stardust._core import linalg as _core

SETTINGS = settings(max_examples=int(os.environ.get("SDLA_EXAMPLES", 150)), deadline=None)
EPS = np.finfo(np.float64).eps


def batch_kron(a, b):
    """numpy's product a[i, j] * b[k, l] at (i*P + k, j*Q + l), batch axes broadcast."""
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    m, n = a.shape[-2:]
    p, q = b.shape[-2:]
    prod = a[..., :, None, :, None] * b[..., None, :, None, :]
    return np.broadcast_to(prod, (*batch, m, p, n, q)).reshape(*batch, m * p, n * q)


# ------------------------------------------------------------------- kron --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    drop=st.integers(0, 2),
    shapes=st.tuples(*[st.integers(0, 4)] * 4),
    dtypes=st.tuples(st.sampled_from(DTYPES), st.sampled_from(DTYPES)),
    layouts=st.tuples(st.sampled_from(LAYOUTS), st.sampled_from(LAYOUTS)),
)
def test_kron(seed, batch, drop, shapes, dtypes, layouts):
    rng = np.random.default_rng(seed)
    m, n, p, q = shapes
    # x2 has fewer batch axes, or size-1 ones, so that they broadcast
    b2 = tuple(1 if i % 2 else s for i, s in enumerate(batch[drop:]))
    a = with_layout(random_values(rng, (*batch, m, n), dtypes[0]), layouts[0])
    b = with_layout(random_values(rng, (*b2, p, q), dtypes[1]), layouts[1])
    r = npy(sd.linalg.kron(as_sd(a), as_sd(b)))
    ct = compute_dtype(*dtypes)
    assert r.dtype == ct
    want = batch_kron(a.astype(ct), b.astype(ct))
    assert r.shape == want.shape
    # every product as numpy's multiply computes it: bit for bit
    assert np.array_equal(r, want)
    if r.ndim == 2 and r.size:
        assert r.flags.f_contiguous


def test_kron_rules():
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((2, 3)), rng.standard_normal((4, 2)) + 1j
    assert np.array_equal(npy(sd.linalg.kron(as_sd(a), as_sd(b))), np.kron(a, b))  # 2-D: np.kron itself
    # batches broadcast (np.kron would multiply the batch axes too)
    x, y = rng.standard_normal((5, 1, 2, 2)), rng.standard_normal((3, 3, 3))
    assert npy(sd.linalg.kron(as_sd(x), as_sd(y))).shape == (5, 3, 6, 6)
    assert npy(sd.linalg.kron(as_sd(np.eye(2, dtype=np.int64)), as_sd(np.ones((2, 2), np.bool_)))).dtype == np.float64
    with pytest.raises(ValueError):
        sd.linalg.kron(as_sd(np.ones(3)), as_sd(np.ones((2, 2))))  # 1-D: not a matrix
    with pytest.raises(ValueError):
        sd.linalg.kron(as_sd(np.ones((2, 2, 2))), as_sd(np.ones((3, 2, 2))))  # batches do not broadcast
    with pytest.raises(TypeError):
        sd.linalg.kron(np.ones((2, 2)), as_sd(np.ones((2, 2))))
    # NaN passes through, as in numpy
    r = npy(sd.linalg.kron(as_sd(np.array([[np.nan, 1.0]])), as_sd(np.ones((1, 1)))))
    assert np.isnan(r[0, 0]) and r[0, 1] == 1.0


# ------------------------------------------------------------------ cross --


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(max_dims=3),
    pos=st.integers(0, 3),
    drop=st.integers(0, 3),
    dtypes=st.tuples(st.sampled_from(DTYPES), st.sampled_from(DTYPES)),
    strided=st.booleans(),
)
def test_cross(seed, batch, pos, drop, dtypes, strided):
    rng = np.random.default_rng(seed)
    pos = min(pos, len(batch))
    shape = (*batch[:pos], 3, *batch[pos:])
    axis = pos - len(shape)  # negative: the same axis of the broadcast shape in both
    drop = min(drop, pos)  # x2 has fewer leading axes
    a = random_values(rng, shape, dtypes[0])
    b = random_values(rng, shape[drop:], dtypes[1])
    if strided:
        a = np.repeat(a, 2, axis=axis)[(slice(None),) * (a.ndim + axis) + (slice(None, None, 2),)]
    r = npy(sd.linalg.cross(as_sd(a), as_sd(b), axis=axis))
    ct = compute_dtype(*dtypes)
    assert r.dtype == ct
    want = np.cross(a.astype(ct), b.astype(ct), axis=axis)
    assert r.shape == want.shape
    assert np.array_equal(r, want)  # numpy's arithmetic, bit for bit


def test_cross_rules():
    e = np.eye(3)
    assert np.array_equal(npy(sd.linalg.cross(as_sd(e[0]), as_sd(e[1]))), e[2])
    # the axis counts in the broadcast shape (Array API): axis=1 is b's first axis here
    a, b = np.ones((2, 3, 4)), np.arange(12.0).reshape(3, 4)
    r = npy(sd.linalg.cross(as_sd(a), as_sd(b), axis=1))
    assert r.shape == (2, 3, 4) and np.array_equal(r, np.cross(a, b, axis=-2))
    assert r.flags.c_contiguous
    with pytest.raises(ValueError, match=r"must have size 3 along the axis, got 2 and 2"):
        sd.linalg.cross(as_sd(np.ones(2)), as_sd(np.ones(2)))
    with pytest.raises(ValueError, match=r"got 3 and 1"):
        sd.linalg.cross(as_sd(np.ones((3, 2))), as_sd(np.ones(2)), axis=0)  # x2 has no axis 0 of its own
    with pytest.raises(ValueError, match="out of bounds"):
        sd.linalg.cross(as_sd(np.ones(3)), as_sd(np.ones(3)), axis=1)
    with pytest.raises(TypeError):
        sd.linalg.cross(as_sd(np.ones(3)), as_sd(np.ones(3)), -1)  # axis is keyword-only
    z = npy(sd.linalg.cross(as_sd(np.zeros((0, 3))), as_sd(np.ones(3))))
    assert z.shape == (0, 3)
    assert npy(sd.linalg.cross(as_sd(np.ones(3, np.int64)), as_sd(np.ones(3, np.bool_)))).dtype == np.float64


# ------------------------------------------------------------------- expm --


def structured(rng, batch, n, dtype, structure, scale):
    a = random_values(rng, (*batch, n, n), dtype)
    if dtype in ("float64", "complex128"):
        a = a * (scale / max(n, 1))
    if structure == "upper":
        a = np.triu(a)
    elif structure == "lower":
        a = np.tril(a)
    elif structure == "diagonal":
        a = a * np.eye(n, dtype=a.dtype)
    return a


def rel_err(x, want):
    d = np.abs(x - want).sum(axis=-2).max(axis=-1)
    s = np.abs(want).sum(axis=-2).max(axis=-1)
    return np.max(d / s)


@SETTINGS
@given(
    seed=st.integers(0, 2**32 - 1),
    batch=batch_shapes(),
    n=st.integers(0, 6),
    dtype=st.sampled_from(DTYPES),
    layout=st.sampled_from(LAYOUTS),
    structure=st.sampled_from(["full", "full", "upper", "lower", "diagonal"]),
    scale=st.sampled_from([1e-3, 0.1, 1.0, 5.0, 30.0]),
)
def test_expm(seed, batch, n, dtype, layout, structure, scale):
    rng = np.random.default_rng(seed)
    a = with_layout(structured(rng, batch, n, dtype, structure, scale), layout)
    r = npy(sd.linalg.expm(as_sd(a)))
    ct = compute_dtype(dtype)
    assert r.dtype == ct and r.shape == a.shape
    if r.size == 0:
        return
    want = sl.expm(a.astype(ct))
    # well within what the conditioning allows for these sizes (scipy is as far
    # from the exact value: docs/design/linalg.md §11.5)
    assert rel_err(r, want) < 1e-12 * max(1.0, scale)
    if structure in ("upper", "lower", "diagonal"):  # the other triangle is exactly 0
        i, j = np.indices((n, n))
        off = (i > j) if structure == "upper" else (i < j) if structure == "lower" else (i != j)
        assert np.all(r[..., off] == 0)
        d = np.diagonal(a.astype(ct), axis1=-2, axis2=-1)
        assert np.allclose(np.diagonal(r, axis1=-2, axis2=-1), np.exp(d), rtol=4 * EPS, atol=0)


def test_expm_exact_cases():
    # 0 -> I, nilpotent -> I + N, diagonal -> exp of the diagonal (libm's exp, as math.exp)
    assert np.array_equal(npy(sd.linalg.expm(as_sd(np.zeros((3, 3))))), np.eye(3))
    assert np.array_equal(npy(sd.linalg.expm(as_sd(np.array([[0.0, 7.5], [0.0, 0.0]])))), [[1.0, 7.5], [0.0, 1.0]])
    d = np.array([-700.0, -1.5, 0.0, 0.3, 700.0])
    r = npy(sd.linalg.expm(as_sd(np.diag(d))))
    assert np.array_equal(np.diag(r), [math.exp(v) for v in d]) and np.count_nonzero(r - np.diag(np.diag(r))) == 0
    assert npy(sd.linalg.expm(as_sd(np.array([[2.0]])))) == math.exp(2.0)
    # 2x2 triangular: [[a, t], [0, c]] -> [[e^a, t (e^c - e^a) / (c - a)], [0, e^c]]
    for a_, c_, t in [(1.0, -1.0, 1e8), (0.3, 0.3 + 1e-10, 2.0), (-50.0, 40.0, 1.0), (2.0, 2.0, 3.0)]:
        r = npy(sd.linalg.expm(as_sd(np.array([[a_, t], [0.0, c_]]))))
        want12 = t * math.exp(a_) if a_ == c_ else t * math.exp((a_ + c_) / 2) * math.sinh((c_ - a_) / 2) / ((c_ - a_) / 2)
        assert r[1, 0] == 0 and r[0, 0] == math.exp(a_) and r[1, 1] == math.exp(c_)
        assert abs(r[0, 1] - want12) <= 4 * EPS * abs(want12)
    # a rotation generator -> the rotation
    th = 0.7
    r = npy(sd.linalg.expm(as_sd(np.array([[0.0, -th], [th, 0.0]]))))
    assert np.allclose(r, [[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], rtol=0, atol=2 * EPS)
    # exp(A) exp(-A) = I and exp(2A) = exp(A)^2
    rng = np.random.default_rng(1)
    a = rng.standard_normal((6, 6))
    e1, e2 = npy(sd.linalg.expm(as_sd(a))), npy(sd.linalg.expm(as_sd(-a)))
    assert np.allclose(e1 @ e2, np.eye(6), atol=1e-12)
    assert rel_err(npy(sd.linalg.expm(as_sd(2 * a))), e1 @ e1) < 1e-13


def test_expm_large_norms_no_overflow_trouble():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # ||A|| far above 2^100: the powers would overflow without the up-front scaling
        a = np.array([[-1e40, 1e39], [2e39, -1e40]])
        r = npy(sd.linalg.expm(as_sd(a)))
        assert np.all(np.isfinite(r)) and np.all(r == 0)  # exp of eigenvalues ~ -1e40
        q = np.linalg.qr(np.random.default_rng(2).standard_normal((4, 4)))[0]
        b = q @ np.diag([-1e35, -3e35, -2e35, -5e35]) @ q.T
        assert np.all(npy(sd.linalg.expm(as_sd(b))) == 0)
        # an exponential that overflows is inf, with no warning or error
        r = npy(sd.linalg.expm(as_sd(np.array([[1000.0, 1.0], [0.0, 999.0]]))))
        assert np.isinf(r[0, 0])


def test_expm_estimated_norms():
    # from n = 100 on, ||A^8|| and ||A^10|| are estimated (lacn2); a degree-13 matrix both ways
    rng = np.random.default_rng(3)
    for dt in ("float64", "complex128"):
        a = random_values(rng, (120, 120), dt) * (30 / 120)
        want = sl.expm(a)
        got_est = npy(sd.linalg.expm(as_sd(a)))
        _core._set_expm_estimate_min(1 << 40)
        try:
            got_exact = npy(sd.linalg.expm(as_sd(a)))
        finally:
            _core._set_expm_estimate_min(100)
        assert rel_err(got_est, want) < 1e-12 and rel_err(got_exact, want) < 1e-12


def test_expm_rules():
    rng = np.random.default_rng(4)
    assert npy(sd.linalg.expm(as_sd(np.zeros((0, 0))))).shape == (0, 0)
    assert npy(sd.linalg.expm(as_sd(np.zeros((0, 3, 3))))).shape == (0, 3, 3)
    assert npy(sd.linalg.expm(as_sd(np.eye(3, dtype=np.int64)))).dtype == np.float64
    assert npy(sd.linalg.expm(as_sd(np.eye(3, dtype=np.bool_)))).dtype == np.float64
    r = npy(sd.linalg.expm(as_sd(rng.standard_normal((4, 4)))))
    assert r.flags.f_contiguous
    with pytest.raises(ValueError):
        sd.linalg.expm(as_sd(np.ones((2, 3))))
    with pytest.raises(ValueError):
        sd.linalg.expm(as_sd(np.ones(3)))
    with pytest.raises(TypeError):
        sd.linalg.expm(np.eye(2))
    # NaN/inf: ValueError naming the smallest batch index -- a diagonal one included
    x = np.zeros((5, 3, 3))
    x[3, 0, 1] = np.inf
    x[2, 1, 1] = np.nan
    with pytest.raises(ValueError, match=r"'x' contains NaN or infinity \(at batch index \(2,\)\)"):
        sd.linalg.expm(as_sd(x))
    # the same matrix gives the same bits wherever it sits in a batch (docs/design/linalg.md §6.4)
    a = rng.standard_normal((7, 5, 5)) * 3
    whole = npy(sd.linalg.expm(as_sd(a)))
    for i in range(7):
        assert np.array_equal(whole[i], npy(sd.linalg.expm(as_sd(a[i]))))


# ------------------------------------------------------------ thread counts --


def test_same_bits_for_any_thread_count():
    rng = np.random.default_rng(5)
    default = sd.get_num_threads()
    calls = [
        lambda a, a32: sd.linalg.expm(a),
        # the n = 110 batch's kron on its leading 32 x 32 blocks (contiguous): the full one is
        # 6 x 12,100^2 doubles (7 GB) per result; a batch of six splits all the same
        lambda a, a32: sd.linalg.kron(a32, a32),
        lambda a, a32: sd.linalg.cross(a[..., 0, :3], a[..., 1, :3]),
    ]
    try:
        for batch, n in (((400,), 3), ((48,), 16), ((6,), 110)):
            an = rng.standard_normal((*batch, n, n)) * 2
            a = as_sd(an)
            a32 = as_sd(np.ascontiguousarray(an[..., :32, :32])) if n > 32 else a
            for call in calls:
                first = None  # a digest per thread count, compared to the first
                for t in sorted({1, 3, default}):
                    sd.set_num_threads(t)
                    d = hashlib.sha256(np.ascontiguousarray(np.asarray(call(a, a32))).data).digest()
                    first = d if first is None else first
                    assert d == first
    finally:
        sd.set_num_threads(default)
