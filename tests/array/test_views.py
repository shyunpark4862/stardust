"""Shape manipulation. Views share memory; numpy is the oracle for values."""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

import stardust as sd
from .conftest import assert_same


def shares(view, owner_np):
    return np.shares_memory(np.asarray(view), owner_np)


def grid(*shape):
    n = np.arange(int(np.prod(shape)), dtype=np.float64).reshape(shape)
    return n, sd.asarray(n)


# --------------------------------------------------------------------- reshape

@pytest.mark.parametrize("src,dst", [
    ((12,), (3, 4)), ((3, 4), (12,)), ((3, 4), (2, 6)), ((2, 3, 4), (6, 4)),
    ((2, 3, 4), (2, -1)), ((3, 4), (-1,)), ((1, 12, 1), (3, 1, 4)), ((0, 3), (3, 0)), ((0, 3), (-1, 5)),
])
def test_reshape_contiguous_is_a_view(src, dst):
    n, a = grid(*src)
    r = sd.reshape(a, dst)
    assert_same(r, n.reshape(dst))
    assert shares(r, n) or n.size == 0


def test_reshape_strided_view_when_strides_allow():
    n = np.arange(48.0).reshape(4, 12)[:, ::2]  # (4, 6), rows not contiguous
    r = sd.reshape(sd.asarray(n), (4, 2, 3))  # splits only the inner axis
    assert_same(r, n.reshape(4, 2, 3))
    assert shares(r, n)


def test_reshape_needs_copy_for_transpose():
    n = np.arange(12.0).reshape(3, 4).T
    a = sd.asarray(n)
    r = sd.reshape(a, (12,))
    assert_same(r, n.reshape(12))
    assert not shares(r, n) and r.base is None
    with pytest.raises(ValueError):
        sd.reshape(a, (12,), copy=False)


def test_reshape_copy_true_always_copies():
    n, a = grid(3, 4)
    r = sd.reshape(a, (4, 3), copy=True)
    assert_same(r, n.reshape(4, 3))
    assert not shares(r, n)


@pytest.mark.parametrize("bad,exc", [
    ((5,), ValueError), ((-1, -1), ValueError), ((-1, 5), ValueError), ((), ValueError), ((-2, 6), ValueError),
    (3.0, TypeError), ([3, 4], TypeError),
])
def test_reshape_errors(bad, exc):
    _, a = grid(3, 4)
    with pytest.raises(exc):
        sd.reshape(a, bad)


def test_reshape_unknown_extent_of_empty_array():
    _, a = grid(0, 3)
    with pytest.raises(ValueError):
        sd.reshape(a, (-1, 0))  # any extent would do: ambiguous


def test_reshape_method_and_signature():
    n, a = grid(2, 3)
    assert_same(a.reshape((3, 2)), n.reshape(3, 2))
    assert_same(a.reshape(6), n.reshape(6))
    with pytest.raises(TypeError):
        a.reshape(3, 2)  # one shape argument (docs/api/array.md §1.3)
    with pytest.raises(TypeError):
        sd.reshape(a, (6,), False)  # copy is keyword-only


@given(st.data())
def test_reshape_matches_numpy_on_random_views(data):
    base = np.arange(2 * 3 * 4 * 5, dtype=np.int64).reshape(2, 3, 4, 5)
    index = tuple(data.draw(st.sampled_from([slice(None), slice(None, None, -1), slice(None, None, 2), slice(1, 2)]))
                  for _ in range(4))
    view = base[index]
    if data.draw(st.booleans()):
        view = view.transpose(data.draw(st.permutations(range(4))))
    # A random factorisation of the size into a random number of axes.
    size = view.size
    shape = []
    while size > 1:
        f = data.draw(st.sampled_from([d for d in range(2, size + 1) if size % d == 0]))
        shape.append(f)
        size //= f
    shape = shape or [1]
    shape.insert(data.draw(st.integers(0, len(shape))), 1)
    r = sd.reshape(sd.asarray(view), tuple(shape))
    want = view.reshape(shape)
    assert_same(r, want)
    # View exactly when numpy can view (same algorithm).
    assert shares(r, base) == np.shares_memory(want, base)


def test_ravel():
    n, a = grid(3, 4)
    r = sd.ravel(a)
    assert_same(r, n.ravel())
    assert shares(r, n)
    t = sd.ravel(a.T)
    assert_same(t, n.T.ravel())
    assert not shares(t, n)
    # A 1-D strided array is already "flat": ravel views it (numpy copies).
    s = sd.ravel(a[0, ::2])
    assert_same(s, n[0, ::2])
    assert shares(s, n)


# ----------------------------------------------------------------- transposes

def test_T_and_mT():
    n, a = grid(3, 4)
    assert_same(a.T, n.T)
    assert shares(a.T, n)
    n3, a3 = grid(2, 3, 4)
    assert_same(a3.mT, np.swapaxes(n3, -1, -2))
    assert_same(sd.matrix_transpose(a3), np.swapaxes(n3, -1, -2))
    with pytest.raises(ValueError):
        a3.T  # only 2-D (docs/api/array.md §10.1)
    with pytest.raises(ValueError):
        sd.arange(3).T
    with pytest.raises(ValueError):
        sd.arange(3).mT


def test_permute_dims():
    n, a = grid(2, 3, 4)
    for axes in [(0, 1, 2), (2, 0, 1), (1, 2, 0), (-1, 0, 1)]:
        assert_same(sd.permute_dims(a, axes), n.transpose(axes))
    for bad in [(0, 1), (0, 0, 1), (0, 1, 3)]:
        with pytest.raises(ValueError):
            sd.permute_dims(a, bad)
    assert_same(sd.permute_dims(sd.arange(3), 0), np.arange(3))
    # A list works like a tuple for every axis argument, as in sd.linalg (docs/api/array.md §1.3).
    assert_same(sd.permute_dims(a, [2, 1, 0]), n.transpose((2, 1, 0)))
    assert_same(sd.moveaxis(a, [0], (1,)), np.moveaxis(n, [0], (1,)))
    assert_same(sd.sum(a, axis=[0, 2]), np.sum(n, axis=(0, 2)))
    with pytest.raises(TypeError, match=r"^permute_dims\(\): 'axes' must be an int or a tuple or list of ints, not float$"):
        sd.permute_dims(a, (2, 1.0, 0))
    with pytest.raises(TypeError, match=r"^sum\(\): 'axis' must be an int or a tuple or list of ints, not float$"):
        sd.sum(a, axis=1.5)
    with pytest.raises(OverflowError):
        sd.permute_dims(a, (2**63, 0, 1))  # an int outside int64 (API.md §2)


@pytest.mark.parametrize("src,dst", [(0, 2), (-1, 0), ((0, 1), (2, 0)), ((0, 2), (-1, -3)), ((), ())])
def test_moveaxis(src, dst):
    n, a = grid(2, 3, 4)
    assert_same(sd.moveaxis(a, src, dst), np.moveaxis(n, src, dst))
    # source and destination are axes: lists work like tuples (docs/api/array.md §1.3, §4.1)
    as_list = (lambda v: list(v) if isinstance(v, tuple) else v)
    assert_same(sd.moveaxis(a, as_list(src), as_list(dst)), np.moveaxis(n, src, dst))


def test_moveaxis_errors():
    _, a = grid(2, 3, 4)
    with pytest.raises(ValueError):
        sd.moveaxis(a, (0, 1), 2)
    with pytest.raises(ValueError):
        sd.moveaxis(a, (0, 0), (1, 2))
    with pytest.raises(ValueError):
        sd.moveaxis(a, 3, 0)


# --------------------------------------------------- adding and removing axes

def test_squeeze():
    n, a = grid(1, 3, 1, 4)
    assert_same(sd.squeeze(a, axis=0), n.squeeze(0))
    assert_same(sd.squeeze(a, axis=(0, -2)), n.squeeze((0, 2)))
    with pytest.raises(ValueError):
        sd.squeeze(a, axis=1)  # extent 3
    with pytest.raises(ValueError):
        sd.squeeze(sd.zeros((1, 1)), axis=(0, 1))  # would be 0-d (docs/api/array.md §1.2)
    with pytest.raises(TypeError):
        sd.squeeze(a)  # axis is required


def test_expand_dims():
    n, a = grid(2, 3)
    for axis in [0, 1, 2, -1, -3, (0, 2), (0, -1), (1, 2, 3)]:
        got = sd.expand_dims(a, axis=axis)
        assert_same(got, np.expand_dims(n, axis))
        assert shares(got, n)
    for bad in [3, -4, (0, 0)]:
        with pytest.raises(ValueError):
            sd.expand_dims(a, axis=bad)
    with pytest.raises(ValueError):
        sd.expand_dims(sd.zeros((1,) * 31), axis=(0, 1))


# --------------------------------------------------------------- broadcasting

def test_broadcast_to_is_a_readonly_view():
    n = np.arange(3.0)
    b = sd.broadcast_to(sd.asarray(n), (2, 3))
    assert_same(b, np.broadcast_to(n, (2, 3)))
    assert b.strides == (0, 8)
    assert not b._flags()["writeable"]
    with pytest.raises(ValueError):
        b[0, 0] = 1.0
    assert shares(b, n)


@pytest.mark.parametrize("shape", [(2,), (3, 2), (), (-1, 3)])
def test_broadcast_to_errors(shape):
    with pytest.raises(ValueError):
        sd.broadcast_to(sd.arange(3), shape)


def test_broadcast_to_is_one_directional():
    with pytest.raises(ValueError):
        sd.broadcast_to(sd.zeros((2, 3)), (1, 3))


def test_broadcast_arrays():
    x, y = np.arange(3.0), np.arange(2).reshape(2, 1)
    got = sd.broadcast_arrays(sd.asarray(x), sd.asarray(y))
    want = np.broadcast_arrays(x, y)
    assert isinstance(got, tuple) and len(got) == 2
    for g, w in zip(got, want):
        assert_same(g, w)
        assert not g._flags()["writeable"]
    assert sd.broadcast_arrays() == ()
    with pytest.raises(ValueError):
        sd.broadcast_arrays(sd.zeros(2), sd.zeros(3))
    with pytest.raises(TypeError):
        sd.broadcast_arrays(sd.zeros(2), [1, 2])


# --------------------------------------------------------------- real / imag

def test_real_imag_of_complex_are_writeable_views():
    n = np.array([1 + 2j, 3 - 4j, -0.0 + 5j])
    a = sd.asarray(n)
    re, im = sd.real(a), sd.imag(a)
    assert_same(re, n.real)
    assert_same(im, n.imag)
    assert re.strides == (16,) and im.strides == (16,)
    im[1] = 9.0
    assert n[1] == 3 + 9j  # writes go through to the complex array


def test_real_imag_of_strided_complex():
    n = (np.arange(24.0) + 1j * np.arange(24.0)[::-1]).reshape(4, 6)[::-1, ::2]
    a = sd.asarray(n)
    assert_same(sd.real(a), n.real)
    assert_same(sd.imag(a).T, n.imag.T)


def test_real_imag_of_real_dtypes(dtype):
    if dtype is sd.complex128:
        return
    a = sd.ones((2, 3), dtype=dtype)
    assert sd.real(a) is a
    im = sd.imag(a)
    assert im.dtype is dtype and im.tolist() == [[0] * 3] * 2
    assert not im._flags()["writeable"]


@pytest.mark.parametrize("value", [True, False, 3, -7, 2.5, -0.0, 3 - 4j, complex(0, -0.0), np.int64(3),
                                   np.float32(2.5), np.bool_(True)])
def test_real_imag_conj_take_python_scalars(value):
    # Like conj (docs/api/array.md §6.4): the result is the Python scalar of the dtype in §6.3.
    def py(fn):
        r = fn(sd.asarray([value]))
        return r.tolist()[0]
    for fn in (sd.real, sd.imag, sd.conj):
        got = fn(value)
        want = py(fn)
        assert type(got) is type(want) and repr(got) == repr(want)
    out = sd.zeros((2, 2), dtype=sd.complex128)
    assert sd.imag(value, out=out) is out and out.tolist() == [[complex(value).imag] * 2] * 2
    with pytest.raises(TypeError):
        sd.real([value])
    with pytest.raises(TypeError, match=r"^imag\(\): 'x' must be an sd\.ndarray or a bool, int, float or complex, not str$"):
        sd.imag("1")


def test_real_imag_scalar_ints_follow_the_int64_rule():
    with pytest.raises(OverflowError):
        sd.real(2**63)  # an int is an int64 (docs/api/array.md §1.5), as with conj
    with pytest.raises(OverflowError):
        sd.conj(2**63)
    assert sd.real(2**63 - 1) == 2**63 - 1 and sd.imag(-(2**63)) == 0


def test_real_imag_out():
    a = sd.asarray([1 + 2j, 3 + 4j])
    out = sd.zeros((2, 2))
    assert sd.imag(a, out=out) is out
    assert out.tolist() == [[2, 4], [2, 4]]  # broadcast into out
    with pytest.raises(TypeError):
        sd.real(a, out=sd.zeros(2, dtype=sd.int64))  # float64 -> int64 (docs/api/array.md §1.5)
    with pytest.raises(TypeError):
        sd.real(a, sd.zeros(2))  # out is keyword-only


# ------------------------------------------------------------------- base

def test_base_is_the_final_owner():
    a = sd.zeros((4, 4))
    v = a[1:][:, 2:].T[None].reshape((2, 3))
    assert v.base is a
    n = np.zeros((4, 4))
    w = sd.asarray(n)[1:]
    assert w.base is n  # imported memory: numpy's array owns it
