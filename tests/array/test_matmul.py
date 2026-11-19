"""matmul and @ (Accelerate BLAS for float64/complex128, own loop for int64)."""

import numpy as np
import pytest

import stardust as sd
from .conftest import NP_DTYPE, assert_same

SHAPES = [
    ((3, 4), (4, 5)),
    ((4,), (4, 5)),
    ((3, 4), (4,)),
    ((4,), (4,)),
    ((1, 4), (4, 1)),
    ((5, 1), (1, 6)),
    ((2, 3, 4), (4, 2)),
    ((3, 4), (2, 4, 5)),
    ((2, 1, 3, 4), (5, 4, 2)),
    ((2, 3, 4), (2, 4, 6)),
    ((0, 4), (4, 3)),
    ((3, 0), (0, 2)),
    ((2, 0, 3), (3, 4)),
    ((3,), (2, 3, 4)),
]


def make(shape, dtype, seed):
    rng = np.random.default_rng(seed)
    if dtype is sd.bool:
        return rng.random(shape) > 0.5
    if dtype is sd.int64:
        return rng.integers(-9, 10, shape)
    if dtype is sd.float64:
        return rng.standard_normal(shape)
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def check(got, want, dtype):
    want = np.asarray(want)
    if want.ndim == 0:
        assert not isinstance(got, sd.ndarray)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12)
        return
    if dtype in (sd.bool, sd.int64):
        assert_same(got, want.astype(np.int64))
    else:
        assert got.shape == want.shape and NP_DTYPE[got.dtype] == want.dtype
        np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("sa,sb", SHAPES, ids=str)
def test_shapes(sa, sb, dtype):
    a, b = make(sa, dtype, 0), make(sb, dtype, 1)
    want = a.astype(np.int64) @ b.astype(np.int64) if dtype is sd.bool else a @ b
    check(sd.matmul(sd.asarray(a), sd.asarray(b)), want, dtype)
    check(sd.asarray(a) @ sd.asarray(b), want, dtype)


@pytest.mark.parametrize("layout", ["T", "F", "reversed", "stepped", "broadcast"])
def test_operand_layouts(layout, dtype):
    # BLAS reads row- or column-contiguous matrices in place; others are packed first.
    a, b = make((6, 5), dtype, 2), make((5, 7), dtype, 3)
    views = {
        "T": (a.T.copy().T, b.T.copy().T),
        "F": (np.asfortranarray(a), np.asfortranarray(b)),
        "reversed": (a[::-1, ::-1], b[::-1]),
        "stepped": (np.repeat(a, 2, axis=1)[:, ::2], np.repeat(b, 3, axis=0)[::3]),
        "broadcast": (np.broadcast_to(a[:1], a.shape), b),
    }[layout]
    want = views[0].astype(np.int64) @ views[1].astype(np.int64) if dtype is sd.bool else views[0] @ views[1]
    check(sd.asarray(views[0]) @ sd.asarray(views[1]), want, dtype)
    v = views[0][:, 0]  # a strided vector
    check(
        sd.asarray(v) @ sd.asarray(views[0]),
        (v.astype(np.int64) if dtype is sd.bool else v) @ (views[0].astype(np.int64) if dtype is sd.bool else views[0]),
        dtype,
    )


def test_mixed_dtypes_and_c13():
    i = sd.asarray([[1, 2], [3, 4]])
    f = sd.asarray([[0.5, 0.0], [0.0, 0.5]])
    assert (i @ f).dtype is sd.float64 and (i @ f).tolist() == [[0.5, 1.0], [1.5, 2.0]]
    t = sd.asarray([[True, True], [False, True]])
    assert (t @ t).dtype is sd.int64 and (t @ t).tolist() == [[1, 2], [0, 1]]  # docs/api/array.md §1.4
    assert (sd.asarray([1j, 1.0]) @ sd.asarray([1j, 2.0])) == -1 + 2  # no conjugation


def test_integer_matmul_wraps():
    big = sd.asarray([[2**62, 2**62]])
    assert (big @ sd.asarray([[2], [2]])).tolist() == [[0]]  # 2**64 wraps to 0


def test_errors():
    a = sd.zeros((2, 3))
    with pytest.raises(ValueError):
        a @ sd.zeros((2, 3))
    with pytest.raises(ValueError):
        sd.zeros((2, 2, 3)) @ sd.zeros((3, 3, 2))  # batch axes do not broadcast
    with pytest.raises(TypeError):
        a @ 2  # arrays only
    with pytest.raises(TypeError):
        sd.matmul(a, [[1], [2], [3]])
    with pytest.raises(TypeError):
        2 @ a


def test_inplace_matmul():
    a = sd.asarray([[1.0, 2.0], [3.0, 4.0]])
    before = a
    a @= sd.asarray([[0.0, 1.0], [1.0, 0.0]])
    assert a is before and a.tolist() == [[2.0, 1.0], [4.0, 3.0]]
    with pytest.raises(ValueError):
        a @= sd.zeros((2, 3))  # the result must keep a's shape
    i = sd.asarray([[1, 0], [0, 1]])
    with pytest.raises(TypeError):
        i @= sd.asarray([[0.5, 0.0], [0.0, 0.5]])  # docs/api/array.md §1.5


def test_large_against_numpy():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((300, 200)), rng.standard_normal((200, 250))
    np.testing.assert_allclose(np.asarray(sd.asarray(a) @ sd.asarray(b)), a @ b, rtol=1e-10, atol=1e-10)
    ia, ib = rng.integers(-100, 100, (120, 80)), rng.integers(-100, 100, (80, 90))
    assert_same(sd.asarray(ia) @ sd.asarray(ib), ia @ ib)


def test_small_batches_on_the_pool_match_numpy():
    # Batches of matrices up to 256^3 multiply-adds run on the pool, each
    # still one BLAS call (docs/design/array.md §9.4).
    rng = np.random.default_rng(5)
    for shape_a, shape_b in [
        ((1000, 8, 8), (1000, 8, 8)),
        ((3, 50, 4, 7), (50, 7, 5)),
        ((40, 128, 64), (64, 32)),
        ((300, 1, 6), (300, 6, 1)),
        ((4, 300, 300), (4, 300, 300)),
    ]:
        a, b = rng.random(shape_a), rng.random(shape_b)
        got = np.asarray(sd.asarray(a) @ sd.asarray(b))
        assert got.shape == (a @ b).shape
        np.testing.assert_allclose(got, a @ b, rtol=1e-13)
    c = rng.random((200, 5, 5)) + 1j * rng.random((200, 5, 5))
    np.testing.assert_allclose(np.asarray(sd.asarray(c) @ sd.asarray(c)), c @ c, rtol=1e-13)
