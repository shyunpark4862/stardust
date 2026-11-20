"""The n-d iterator through what it drives: copies and elementwise operations on any
layout (broadcast, extent-1 axes, transposed and stepped views, several operands that
disagree) give numpy's values, empty shapes stay empty, and incompatible shapes raise."""

import numpy as np
import pytest

import stardust as sd

BASE = np.arange(2.0 * 3 * 4 * 6).reshape(2, 3, 4, 6)

LAYOUTS = {
    "contiguous": lambda a: a,
    "transposed": lambda a: a.T,
    "every second element": lambda a: a[..., ::2],
    "every second row": lambda a: a[:, :, ::2],
    "reversed": lambda a: a[::-1, :, ::-1],
    "extent-1 axes": lambda a: a[:1, :, :1],
    "one element": lambda a: a[:1, :1, :1, :1],
}


def as_sd(np_array):
    return sd.asarray(np_array)  # zero-copy, keeps numpy's strides


@pytest.mark.parametrize("layout", LAYOUTS)
def test_copy_of_any_layout_matches_numpy(layout):
    view = LAYOUTS[layout](BASE)
    got = sd.astype(as_sd(view), sd.float64, copy=True)
    np.testing.assert_array_equal(np.asarray(got), view)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_operands_in_different_layouts(layout):
    view = LAYOUTS[layout](BASE)
    c = np.ascontiguousarray(view) * 0.5  # the other operand C-ordered
    np.testing.assert_array_equal(np.asarray(sd.add(as_sd(view), as_sd(c))), view + c)
    out = sd.zeros(view.shape)
    sd.subtract(as_sd(c), as_sd(view), out=out)
    np.testing.assert_array_equal(np.asarray(out), c - view)


def test_broadcast_operands():
    a, b = np.arange(3.0).reshape(3, 1), np.arange(4.0)
    np.testing.assert_array_equal(np.asarray(sd.add(as_sd(a), as_sd(b))), a + b)
    np.testing.assert_array_equal(np.asarray(sd.add(as_sd(a.T), as_sd(b[::-1, None]))), a.T + b[::-1, None])


def test_empty_iteration():
    got = sd.add(sd.zeros((3, 0, 2)), 1.0)
    assert got.shape == (3, 0, 2)


@pytest.mark.parametrize("shapes", [[(5, 1, 4), (3, 1)], [(1,), (2, 3)], [(2, 1, 3), (4, 1), (1, 3)], [(0, 3), (1, 3)]])
def test_broadcast_shape_matches_numpy(shapes):
    arrays = [sd.zeros(s) for s in shapes]
    got = sd.add(*arrays) if len(arrays) == 2 else sd.where(sd.astype(arrays[0], sd.bool), *arrays[1:])
    assert got.shape == np.broadcast_shapes(*shapes)


@pytest.mark.parametrize("shapes", [[(2, 3), (3, 2)], [(4,), (5,)], [(2, 1, 3), (4, 3, 1), (2,)]])
def test_incompatible_broadcast_raises(shapes):
    arrays = [sd.zeros(s) for s in shapes]
    with pytest.raises(ValueError):
        sd.add(*arrays) if len(arrays) == 2 else sd.where(sd.astype(arrays[0], sd.bool), *arrays[1:])


def test_broadcast_results_are_c_ordered():
    b = sd.broadcast_to(sd.ones(4), (3, 4))
    assert b.copy().strides == (32, 8) and (b + 1).strides == (32, 8)
    assert sd.astype(b, sd.int64).strides == (32, 8) and sd.sort(b).strides == (32, 8)
    assert (sd.broadcast_to(sd.ones(5), (2, 3, 5)) * 2).strides == (120, 40, 8)
    f = sd.asarray(np.asfortranarray(np.ones((3, 4))))
    assert (f + 1).strides == (8, 24)  # a real layout is still followed
    fb = sd.broadcast_to(sd.reshape(f, (3, 1, 4)), (3, 2, 4))
    assert (fb + 0).strides == (8, 24, 48)  # the stride-0 axis keeps its place
