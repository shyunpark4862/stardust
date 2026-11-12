"""Basic indexing (int, slice, ..., newaxis), assignment, len and iteration."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import stardust as sd
from .conftest import assert_same


def grid(*shape, dtype=np.float64):
    n = np.arange(int(np.prod(shape))).astype(dtype).reshape(shape)
    return n, sd.asarray(n.copy())


# ------------------------------------------------------------------ getitem

@pytest.mark.parametrize("key", [
    0, -1, slice(None), slice(1, 3), slice(None, None, -1), slice(None, None, 2), slice(5, 1, -2), slice(10, 20),
    (1, slice(None)), (slice(None), 2), (Ellipsis, 1), (1, Ellipsis), (None,), (None, 1, None),
    (Ellipsis, None), (slice(None), None, slice(1, None)), (), Ellipsis, None, (-1, -1, slice(None, None, -1)),
    (slice(0, 0),), (slice(3, 0, 1),),
])
def test_getitem_views_match_numpy(key):
    n, a = grid(3, 4, 5)
    got, want = a[key], n[key]
    assert_same(got, want)
    if want.size:
        assert np.shares_memory(np.asarray(got), np.asarray(a))


def test_full_integer_index_gives_python_scalars(dtype):
    a = sd.zeros((2, 3), dtype=dtype)
    v = a[1, 2]
    assert type(v) is {sd.bool: bool, sd.int64: int, sd.float64: float, sd.complex128: complex}[dtype]
    assert type(sd.arange(3)[-1]) is int
    assert type(a[1, ..., 2]) is type(v)  # never a 0-d array (docs/api/array.md §1.2)


@pytest.mark.parametrize("key,exc", [
    (3, IndexError), (-4, IndexError), ((0, 0, 0), IndexError), ((Ellipsis, Ellipsis), IndexError),
    (1.0, IndexError), ("a", IndexError), (slice(None, None, 0), ValueError), (2**70, IndexError),
    (True, IndexError),  # no 0-d masks (docs/api/array.md §5.1)
])
def test_getitem_errors(key, exc):
    _, a = grid(3, 4)
    with pytest.raises(exc):
        a[key]


def test_index_with_numpy_integer():
    _, a = grid(3, 4)
    assert_same(a[np.int64(1)], np.arange(4.0, 8.0))


def test_too_many_dimensions():
    a = sd.zeros((1,) * 30)
    assert a[None, None].ndim == 32
    with pytest.raises(IndexError):
        a[None, None, None]


@given(st.data())
def test_random_basic_indices_match_numpy(data):
    shape = data.draw(st.lists(st.integers(0, 4), min_size=1, max_size=4))
    n = np.arange(int(np.prod(shape)), dtype=np.int64).reshape(shape)
    a = sd.asarray(n)
    bound = st.one_of(st.none(), st.integers(-6, 6))
    def item(extent):
        if extent and data.draw(st.booleans()):
            return data.draw(st.integers(-extent, extent - 1))
        return slice(data.draw(bound), data.draw(bound), data.draw(st.sampled_from([None, 1, 2, -1, -3])))

    # Index some leading and some trailing axes; with an ellipsis between them
    # the axes in the middle stay whole (without one, only leading axes are indexed).
    lead = data.draw(st.integers(0, len(shape)))
    items = [item(e) for e in shape[:lead]]
    if data.draw(st.booleans()):
        trail = data.draw(st.integers(0, len(shape) - lead))
        items += [Ellipsis] + [item(e) for e in shape[len(shape) - trail:]]
    for _ in range(data.draw(st.integers(0, 2))):  # newaxis entries anywhere
        items.insert(data.draw(st.integers(0, len(items))), None)
    key = tuple(items)
    want = n[key]
    got = a[key]
    if want.ndim == 0:
        assert got == want.item() and type(got) is int
    else:
        assert_same(got, want)


# ------------------------------------------------------------------ setitem

def test_setitem_scalar_and_broadcast():
    n, a = grid(3, 4)
    a[1, 2] = 100.0
    n[1, 2] = 100.0
    a[0] = 7
    n[0] = 7
    a[:, 1] = sd.asarray([1.5, 2.5, 3.5])
    n[:, 1] = [1.5, 2.5, 3.5]
    a[..., ::-2] = sd.asarray([10, 20])
    n[..., ::-2] = [10, 20]
    a[1:, None, :] = [[[-1.0]]]
    n[1:, None, :] = -1.0
    assert_same(a, n)


def test_setitem_from_numpy_and_lists():
    _, a = grid(2, 3)
    a[:] = np.ones((2, 3))
    a[0] = [True, False, True]  # bool -> float64 is upward
    assert a.tolist() == [[1, 0, 1], [1, 1, 1]]


def test_setitem_overlap_reads_before_writing():
    n, a = grid(10)
    a[1:] = a[:-1]
    n[1:] = n[:-1].copy()
    assert_same(a, n)
    a[::-1] = a
    assert_same(a, n[::-1])


def test_setitem_refuses_implicit_downcast():
    i = sd.zeros(3, dtype=sd.int64)
    with pytest.raises(TypeError):
        i[0] = 1.5
    with pytest.raises(TypeError):
        i[:] = sd.asarray([1.0, 2.0, 3.0])
    b = sd.zeros(2, dtype=sd.bool)
    with pytest.raises(TypeError):
        b[0] = 1  # int -> bool would be a downcast (docs/api/array.md §1.5), even for 0 and 1
    b[0] = True
    f = sd.zeros(2)
    with pytest.raises(TypeError):
        f[0] = 1j
    i[0] = True
    assert i.tolist() == [1, 0, 0]


def test_setitem_integer_range():
    i = sd.zeros(2, dtype=sd.int64)
    with pytest.raises(OverflowError):
        i[0] = 2**63  # docs/api/array.md §1.5
    f = sd.zeros(2)
    f[0] = 2**70  # fine: the target is float64
    assert f[0] == float(2**70)


def test_setitem_strict_broadcasting():
    _, a = grid(2, 3)
    with pytest.raises(ValueError):
        a[0] = sd.asarray([[1.0, 2.0, 3.0]])  # (1, 3) into (3,): numpy would drop the leading 1
    # a single element takes a scalar only: anything else is a wrong type (DESIGN.md §10.3)
    msg = r"^only a bool, int, float or complex can be assigned to a single element, not "
    with pytest.raises(TypeError, match=msg + "ndarray$"):
        a[0, 0] = sd.asarray([5.0])  # an array never fits a single element
    with pytest.raises(TypeError, match=msg + "list$"):
        a[0, 0] = [5.0]
    with pytest.raises(TypeError, match=msg + "str$"):
        a[0, 0] = "a"
    with pytest.raises(TypeError, match=msg + "NoneType$"):
        a[1] = sd.zeros(3); a[1, 2] = None
    with pytest.raises(ValueError):
        a[:] = sd.asarray([1.0, 2.0])


def test_setitem_readonly():
    n = np.zeros(3)
    n.flags.writeable = False
    with pytest.raises(ValueError):
        sd.asarray(n)[0] = 1.0


def test_setitem_nan_and_complex():
    c = sd.zeros(2, dtype=sd.complex128)
    c[1] = complex(math.nan, -0.0)
    assert math.isnan(c[1].real) and math.copysign(1, c[1].imag) == -1


# ------------------------------------------------------- len and iteration

def test_len_and_iteration():
    n, a = grid(3, 2)
    assert len(a) == 3
    rows = list(a)
    assert len(rows) == 3
    for r, w in zip(rows, n):
        assert_same(r, w)
    assert list(sd.asarray([1, 2, 3])) == [1, 2, 3]
    assert all(type(v) is int for v in sd.arange(3))
    assert list(sd.zeros((0, 2))) == []


def test_iteration_yields_views():
    a = sd.zeros((2, 2))
    for row in a:
        row[0] = 5.0
    assert a.tolist() == [[5, 0], [5, 0]]


def test_in_is_not_supported():
    with pytest.raises(TypeError):
        1 in sd.arange(3)
