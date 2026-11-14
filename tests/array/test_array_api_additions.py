"""T0b: the Array API 2024 functions the array area lacked, plus bincount and
histogram, against numpy.

Exact unless noted: the elementwise functions are numpy's scalar formulas,
the shape and search functions move or compare values only, and histogram
replays numpy's block sums. tensordot goes through matmul (BLAS), so it is
compared with a tolerance.
"""

import gc
import math
import warnings

import numpy as np
import pytest

import stardust as sd
from .conftest import assert_same

RNG = np.random.default_rng(0)
X3 = RNG.standard_normal((3, 4, 5))
SPECIAL = np.array([0.0, -0.0, 1.0, -1.0, np.inf, -np.inf, np.nan, 1e308, -1e308, 5e-324, -5e-324, 2.5, -7.25])


def same(got, want):
    assert_same(got, np.asarray(want))


# --------------------------------------------------------------- elementwise

@pytest.mark.parametrize("name", ["copysign", "hypot", "logaddexp", "nextafter"])
def test_binary_elementwise_bits(name):
    a = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL])
    b = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL[::-1]])
    aa, bb = np.meshgrid(SPECIAL, SPECIAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for x, y in ((a, b), (aa.ravel(), bb.ravel())):
            same(getattr(sd, name)(sd.asarray(x), sd.asarray(y)), getattr(np, name)(x, y))
        same(getattr(sd, name)(sd.asarray([1, 2, 3]), 2), getattr(np, name)(np.array([1.0, 2, 3]), 2.0))
    assert type(getattr(sd, name)(1.0, 2.0)) is float
    with pytest.raises(TypeError):
        getattr(sd, name)(sd.asarray([1j]), 1.0)


def test_reciprocal_and_signbit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x = np.concatenate([RNG.standard_normal(500) * 30, SPECIAL])
        same(sd.reciprocal(sd.asarray(x)), np.reciprocal(x))
        z = RNG.standard_normal(2000) * 10 + 1j * RNG.standard_normal(2000) * 10
        z = np.concatenate([z, [0j, complex(np.inf, 1), complex(1, np.inf), complex(np.nan, 0), complex(1e-310, 1e300)]])
        same(sd.reciprocal(sd.asarray(z)), np.reciprocal(z))
    same(sd.reciprocal(sd.asarray([2, 4])), np.array([0.5, 0.25]))  # integers in float64 (Array API)
    same(sd.signbit(sd.asarray(x)), np.signbit(x))
    same(sd.signbit(sd.asarray([-1, 0, 3])), np.array([True, False, False]))
    with pytest.raises(TypeError):
        sd.signbit(sd.asarray([1j]))


# ----------------------------------------------------------------- creation

@pytest.mark.parametrize("k", range(-6, 7))
def test_tril_triu(k):
    m = RNG.standard_normal((2, 4, 5))
    same(sd.tril(sd.asarray(m), k=k), np.tril(m, k))
    same(sd.triu(sd.asarray(m), k=k), np.triu(m, k))
    b = RNG.random((3, 3)) > 0.5
    same(sd.triu(sd.asarray(b), k=k), np.triu(b, k))


def test_tril_errors_and_copy():
    with pytest.raises(ValueError, match=r"tril\(\): 'x' must have at least 2 dimensions"):
        sd.tril(sd.asarray([1.0, 2.0]))
    x = sd.asarray(np.ones((2, 2)))
    t = sd.tril(x)
    t[0, 0] = 5.0
    assert x[0, 0] == 1.0


def test_meshgrid():
    a, b, c = np.array([1, 2, 3]), np.array([4.0, 5.0]), np.array([True, False, True, True])
    for indexing in ("xy", "ij"):
        got = sd.meshgrid(sd.asarray(a), sd.asarray(b), sd.asarray(c), indexing=indexing)
        want = np.meshgrid(a, b, c, indexing=indexing)
        assert isinstance(got, tuple) and len(got) == 3
        for g, w in zip(got, want):
            same(g, w)
    same(sd.meshgrid(sd.asarray(a))[0], np.meshgrid(a)[0])
    assert sd.meshgrid() == ()
    with pytest.raises(ValueError, match="'indexing' must be 'xy' or 'ij'"):
        sd.meshgrid(sd.asarray(a), indexing="xx")
    with pytest.raises(ValueError, match="every array must be 1-D"):
        sd.meshgrid(sd.zeros((2, 2)))


# ------------------------------------------------------------- manipulation

@pytest.mark.parametrize("axis", [None, 0, 1, -1, (0, 2)])
def test_flip_is_a_view(axis):
    a = sd.asarray(X3)
    f = sd.flip(a, axis=axis)
    same(f, np.flip(X3, axis))
    assert f.base is not None
    f[0, 0, 0] = 123.0
    assert 123.0 in np.asarray(a)


@pytest.mark.parametrize("shift,axis", [(7, None), (-3, None), (2, 0), (-1, (1, 2)), ((1, -2), (0, 2)), (100, 1), (0, 1)])
def test_roll(shift, axis):
    same(sd.roll(sd.asarray(X3), shift, axis=axis), np.roll(X3, shift, axis))
    same(sd.roll(sd.asarray(X3), shift=shift, axis=axis), np.roll(X3, shift, axis))


def test_roll_errors():
    with pytest.raises(TypeError, match="'shift' must be an int when 'axis' is None"):
        sd.roll(sd.asarray(X3), (1, 2))
    with pytest.raises(ValueError, match="same length"):
        sd.roll(sd.asarray(X3), (1, 2), axis=(0,))


def test_repeat():
    same(sd.repeat(sd.asarray(X3), 2), np.repeat(X3, 2))
    same(sd.repeat(sd.asarray(X3), 3, axis=1), np.repeat(X3, 3, axis=1))
    same(sd.repeat(sd.asarray(X3), sd.asarray([2, 0, 1, 3]), axis=1), np.repeat(X3, [2, 0, 1, 3], axis=1))
    same(sd.repeat(sd.asarray(X3), sd.asarray([2]), axis=2), np.repeat(X3, [2], axis=2))
    with pytest.raises(ValueError, match="non-negative"):
        sd.repeat(sd.asarray([1, 2]), -1)
    with pytest.raises(ValueError, match="'repeats' has length 3 but the axis has length 2"):
        sd.repeat(sd.asarray([1, 2]), sd.asarray([1, 2, 3]))
    with pytest.raises(TypeError, match="'repeats' must be an int or an int64 array"):
        sd.repeat(sd.asarray([1, 2]), 1.5)


@pytest.mark.parametrize("reps", [(2,), 2, (2, 1, 3), (1, 1, 1), (2, 1, 1, 3), (0, 2), (3, 2, 1, 2, 1)])
def test_tile(reps):
    same(sd.tile(sd.asarray(X3), reps), np.tile(X3, reps))
    same(sd.tile(sd.asarray([True, False]), reps), np.tile([True, False], reps))


def test_tile_many_dimensions():
    # 30-D: room for two interleaved axes per copy, so five repeated axes take three copies.
    x = np.arange(6).reshape((1,) * 28 + (2, 3))
    reps = (1,) * 25 + (2,) * 5
    same(sd.tile(sd.asarray(x), reps), np.tile(x, reps))
    # 32-D: no room at all, one block copy per repetition.
    y = np.arange(4).reshape((1,) * 30 + (2, 2))
    reps = (3,) + (1,) * 29 + (2, 1)
    same(sd.tile(sd.asarray(y), reps), np.tile(y, reps))


def test_unstack():
    for axis in (0, 1, -1):
        got = sd.unstack(sd.asarray(X3), axis=axis)
        assert isinstance(got, tuple) and len(got) == X3.shape[axis]
        for i, g in enumerate(got):
            same(g, np.take(X3, i, axis=axis))
            assert g.base is not None
    assert sd.unstack(sd.asarray([1, 2, 3])) == (1, 2, 3)  # no 0-d arrays: Python scalars


def test_diff():
    same(sd.diff(sd.asarray(X3)), np.diff(X3))
    same(sd.diff(sd.asarray(X3), axis=0, n=2), np.diff(X3, axis=0, n=2))
    same(sd.diff(sd.asarray(X3), n=9), np.diff(X3, n=9))
    same(sd.diff(sd.asarray(X3), prepend=0.5, append=sd.asarray(np.ones((3, 4, 2)))),
         np.diff(X3, prepend=0.5, append=np.ones((3, 4, 2))))
    same(sd.diff(sd.asarray([1, 4, 9, 16])), np.diff([1, 4, 9, 16]))
    same(sd.diff(sd.asarray([True, False, False, True])), np.array([-1, 0, 1]))  # subtract rule: int64
    x = sd.asarray([1.0, 2.0])
    assert sd.diff(x, n=0) is x
    with pytest.raises(ValueError, match="'n' must be non-negative"):
        sd.diff(x, n=-1)


# ---------------------------------------------------------- indexing, search

def test_take():
    idx = np.array([[0, 3], [-1, 2]])
    same(sd.take(sd.asarray(X3), sd.asarray(idx), axis=1), np.take(X3, idx, axis=1))
    same(sd.take(sd.asarray(X3), 2, axis=2), np.take(X3, 2, axis=2))
    assert sd.take(sd.asarray([5, 6, 7]), 1) == 6
    same(sd.take(sd.asarray([5, 6, 7]), sd.asarray([2, 0])), np.array([7, 5]))
    with pytest.raises(IndexError):
        sd.take(sd.asarray([5, 6, 7]), 3)
    with pytest.raises(ValueError, match="'axis' is required"):
        sd.take(sd.asarray(X3), 0)
    with pytest.raises(TypeError, match="'indices' must be an int64 array, not bool"):
        sd.take(sd.asarray([5, 6]), sd.asarray([True, False]))


def test_take_along_axis():
    for axis in (0, 1, -1):
        ix = np.argsort(X3, axis=axis)
        same(sd.take_along_axis(sd.asarray(X3), sd.asarray(ix), axis=axis), np.take_along_axis(X3, ix, axis=axis))
    ix = np.argsort(X3, axis=1)[:, :2]
    same(sd.take_along_axis(sd.asarray(X3), sd.asarray(ix), axis=1), np.take_along_axis(X3, ix, axis=1))
    with pytest.raises(ValueError, match="as many dimensions"):
        sd.take_along_axis(sd.asarray(X3), sd.asarray([0]), axis=0)


def test_nonzero_and_count_nonzero():
    z = RNG.integers(0, 3, (3, 4, 5)).astype(float)
    z[0, 0, 0] = np.nan  # NaN is nonzero
    got = sd.nonzero(sd.asarray(z))
    assert isinstance(got, tuple) and len(got) == 3
    for g, w in zip(got, np.nonzero(z)):
        same(g, w)
    for g, w in zip(sd.nonzero(sd.asarray([0j, 1j, 2])), np.nonzero(np.array([0j, 1j, 2]))):
        same(g, w)
    same(sd.nonzero(sd.zeros((0,)))[0], np.zeros(0, np.int64))
    assert sd.count_nonzero(sd.asarray(z)) == np.count_nonzero(z)
    same(sd.count_nonzero(sd.asarray(z), axis=(0, 2)), np.count_nonzero(z, axis=(0, 2)))
    same(sd.count_nonzero(sd.asarray(z), axis=1, keepdims=True), np.count_nonzero(z, axis=1, keepdims=True))


def test_searchsorted():
    s = np.sort(np.concatenate([RNG.standard_normal(100), [np.nan, 0.5, 0.5, -np.inf]]))
    v = np.concatenate([RNG.standard_normal(50), [np.nan, 0.5, -np.inf, np.inf]]).reshape(6, 9)
    for side in ("left", "right"):
        same(sd.searchsorted(sd.asarray(s), sd.asarray(v), side=side), np.searchsorted(s, v, side=side))
    assert sd.searchsorted(sd.asarray(s), 0.5) == np.searchsorted(s, 0.5)
    assert sd.searchsorted(sd.asarray([1, 2, 3]), 2.5) == 2
    u = RNG.standard_normal(30)
    so = np.argsort(u)
    same(sd.searchsorted(sd.asarray(u), sd.asarray(v), sorter=sd.asarray(so)), np.searchsorted(u, v, sorter=so))
    big = np.sort(RNG.standard_normal(10_000))
    keys = RNG.standard_normal(200_000)
    same(sd.searchsorted(sd.asarray(big), sd.asarray(keys)), np.searchsorted(big, keys))
    with pytest.raises(ValueError, match="'x1' must be 1-D"):
        sd.searchsorted(sd.zeros((2, 2)), 1.0)
    with pytest.raises(ValueError, match="'side' must be 'left' or 'right'"):
        sd.searchsorted(sd.asarray([1.0]), 1.0, side="middle")
    with pytest.raises(TypeError, match="complex128"):
        sd.searchsorted(sd.asarray([1.0]), 1j)


# ---------------------------------------------------------------- dtypes

def test_dtype_functions():
    assert sd.can_cast(sd.int64, sd.float64) and not sd.can_cast(sd.float64, sd.int64)
    assert sd.can_cast(sd.asarray([True]), "complex128")
    assert sd.result_type(sd.asarray([1]), 1.5) is sd.float64
    assert sd.result_type(sd.bool, True) is sd.bool
    assert sd.result_type(sd.asarray([1.0]), 1j, sd.int64) is sd.complex128
    with pytest.raises(ValueError):
        sd.result_type()
    with pytest.raises(TypeError, match="sd.string"):
        sd.can_cast(sd.string, sd.float64)
    kinds = {"bool": [sd.bool], "signed integer": [sd.int64], "unsigned integer": [], "integral": [sd.int64],
             "real floating": [sd.float64], "complex floating": [sd.complex128],
             "numeric": [sd.int64, sd.float64, sd.complex128]}
    for kind, members in kinds.items():
        for d in (sd.bool, sd.int64, sd.float64, sd.complex128):
            assert sd.isdtype(d, kind) == (d in members)
    assert sd.isdtype(sd.float64, ("bool", sd.float64)) and not sd.isdtype(sd.int64, sd.float64)
    with pytest.raises(ValueError, match="'kind' must be"):
        sd.isdtype(sd.int64, "integer")
    f, nf = sd.finfo(sd.float64), np.finfo(np.float64)
    assert (f.bits, f.eps, f.max, f.min, f.smallest_normal, f.dtype) == (64, nf.eps, nf.max, nf.min, nf.smallest_normal,
                                                                        sd.float64)
    assert sd.finfo(sd.asarray([1j])).dtype is sd.float64
    i = sd.iinfo(sd.int64)
    assert (i.bits, i.max, i.min, i.dtype) == (64, 2**63 - 1, -(2**63), sd.int64)
    with pytest.raises(TypeError):
        sd.finfo(sd.int64)
    with pytest.raises(TypeError):
        sd.iinfo(sd.float64)


# ----------------------------------------------------------- tensordot, vecdot

@pytest.mark.parametrize("axes,shape_b", [(2, (4, 5, 6)), (1, (5, 2)), (0, (2,)), (([1, 0], [0, 2]), (4, 2, 3)),
                                          (([2], [1]), (2, 5, 7))])
def test_tensordot(axes, shape_b):
    b = RNG.standard_normal(shape_b)
    got = sd.tensordot(sd.asarray(X3), sd.asarray(b), axes=axes)
    np.testing.assert_allclose(np.asarray(got), np.tensordot(X3, b, axes), rtol=1e-13, atol=1e-14)
    i = RNG.integers(-3, 4, (3, 4))
    same(sd.tensordot(sd.asarray(i), sd.asarray(i.T.copy()), axes=1), np.tensordot(i, i.T, 1))
    assert sd.tensordot(sd.asarray([1.0, 2.0]), sd.asarray([3.0, 4.0]), axes=1) == 11.0


def test_tensordot_errors():
    with pytest.raises(ValueError, match="shape mismatch"):
        sd.tensordot(sd.asarray(X3), sd.asarray(X3), axes=([0], [1]))
    with pytest.raises(ValueError, match="larger than"):
        sd.tensordot(sd.asarray([1.0]), sd.asarray([1.0]), axes=2)


def test_vecdot_is_linalg_vecdot():
    assert sd.vecdot is sd.linalg.vecdot


# ------------------------------------------------------------------ dlpack

def test_from_dlpack_shares_numpy_memory():
    x = np.arange(12.0).reshape(3, 4)[:, ::-1]
    a = sd.from_dlpack(x)
    same(a, x)
    x[0, 0] = 99.0
    assert a[0, 0] == 99.0
    assert sd.from_dlpack(x, copy=True).base is None
    for v in (np.array([1 + 2j]), np.array([True, False]), np.array([3, -4])):
        same(sd.from_dlpack(v), v)
    ro = np.arange(3.0)
    ro.flags.writeable = False
    with pytest.raises(ValueError, match="read-only"):
        sd.from_dlpack(ro)[0] = 1.0


def test_dlpack_export_to_numpy():
    x = sd.asarray(np.arange(6).reshape(2, 3))
    n = np.from_dlpack(x)
    assert np.array_equal(n, np.arange(6).reshape(2, 3)) and n.flags.writeable
    n[0, 0] = 42
    assert x[0, 0] == 42  # shared
    b = np.from_dlpack(sd.broadcast_to(sd.asarray([1.0, 2.0]), (3, 2)))
    assert not b.flags.writeable  # read-only travels with the versioned format
    assert x.__dlpack_device__() == (1, 0)
    with pytest.raises(BufferError):
        sd.broadcast_to(sd.asarray([1.0]), (2,)).__dlpack__()  # legacy capsule cannot say read-only
    rt = sd.from_dlpack(sd.asarray([[1.0, 2.0], [3.0, 4.0]]).mT)
    same(rt, np.array([[1.0, 3.0], [2.0, 4.0]]))


def test_dlpack_errors_and_lifetime():
    with pytest.raises(TypeError, match="float32"):
        sd.from_dlpack(np.zeros(3, np.float32))
    with pytest.raises(TypeError, match="DLPack protocol"):
        sd.from_dlpack([1.0, 2.0])
    with pytest.raises(ValueError, match="0-dimensional"):
        sd.from_dlpack(np.array(1.0))
    cap = sd.asarray([1.0, 2.0]).__dlpack__()
    del cap  # an unconsumed capsule frees its tensor
    for _ in range(3):
        big = np.ones(1_000_000)
        s = sd.from_dlpack(big)
        del big
        assert float(sd.sum(s)) == 1e6
        del s
    gc.collect()


# --------------------------------------------------------- bincount, histogram

def test_bincount():
    x = RNG.integers(0, 10, 1000)
    w = RNG.random(1000)
    same(sd.bincount(sd.asarray(x)), np.bincount(x))
    same(sd.bincount(sd.asarray(x), weights=sd.asarray(w), minlength=20), np.bincount(x, weights=w, minlength=20))
    same(sd.bincount(sd.zeros((0,), dtype=sd.int64), minlength=3), np.bincount(np.zeros(0, int), minlength=3))
    same(sd.bincount(sd.asarray([True, False, True])), np.array([1, 2]))
    with pytest.raises(ValueError, match="non-negative"):
        sd.bincount(sd.asarray([1, -1]))
    with pytest.raises(TypeError, match="bool or int64"):
        sd.bincount(sd.asarray([1.0]))


HIST_X = RNG.standard_normal(150_000)  # more than one 65536-element block
HIST_W = RNG.random(150_000)


@pytest.mark.parametrize("kw", [
    {}, {"bins": 37}, {"bins": 5, "range": (-1, 2)}, {"weights": HIST_W}, {"weights": HIST_W, "bins": 13, "range": (-3, 0.5)},
    {"weights": RNG.integers(-5, 6, 150_000), "bins": 9}, {"density": True}, {"density": True, "weights": HIST_W},
    {"bins": np.array([-2, -0.5, 0, 0, 1, 3.0])}, {"bins": np.array([-2, -0.5, 0, 1, 1.0])}, {"bins": np.array([-3, 0, 3])},
    {"bins": np.array([-3, -1, 0, 2.0]), "weights": HIST_W},
], ids=lambda kw: ",".join(kw) or "default")
def test_histogram_bits(kw):
    skw = {k: sd.asarray(v) if isinstance(v, np.ndarray) else v for k, v in kw.items()}
    got = sd.histogram(sd.asarray(HIST_X), **skw)
    hist, edges = np.histogram(HIST_X, **kw)
    assert got.hist is got[0] and got.bin_edges is got[1]
    same(got.hist, hist)
    same(got.bin_edges, edges)


def test_histogram_edges_cases():
    xi = RNG.integers(0, 10, 1000)
    for g, w in zip(sd.histogram(sd.asarray(xi)), np.histogram(xi)):
        same(g, w)
    for data in (np.ones(5), np.zeros(0)):
        for g, w in zip(sd.histogram(sd.asarray(data)), np.histogram(data)):
            same(g, w)
    xn = HIST_X[:1000].copy()
    xn[5] = np.nan
    with pytest.raises(ValueError, match="range is not finite"):
        sd.histogram(sd.asarray(xn))
    same(sd.histogram(sd.asarray(xn), range=(-1, 1)).hist, np.histogram(xn, range=(-1, 1))[0])
    with pytest.raises(ValueError, match="'bins' must increase monotonically"):
        sd.histogram(sd.asarray(xn), bins=sd.asarray([1.0, 0.0]))
    with pytest.raises(ValueError, match="'bins' must be positive"):
        sd.histogram(sd.asarray(xn), bins=0)
    with pytest.raises(TypeError, match="'bins' must be an int or a 1-D array, not str"):
        sd.histogram(sd.asarray(xn), bins="auto")
    with pytest.raises(ValueError, match="'weights' must have the same shape"):
        sd.histogram(sd.asarray([1.0, 2.0]), weights=sd.asarray([1.0]))
    with pytest.raises(TypeError, match="complex128"):
        sd.histogram(sd.asarray([1j]))
