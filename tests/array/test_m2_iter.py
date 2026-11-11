"""M2: the n-d iterator's plan (broadcast, drop, reorder, coalesce)."""

import numpy as np
import pytest

import stardust as sd

plan = sd._core._iter_plan


def as_sd(np_array):
    return sd.asarray(np_array)  # zero-copy, keeps numpy's strides


def test_contiguous_array_becomes_one_run():
    assert plan([sd.zeros((2, 3, 4))]) == {"shape": (24,), "strides": [(8,)], "empty": False}


def test_transposed_array_is_walked_in_memory_order():
    t = as_sd(np.zeros((2, 3, 4)).T)  # shape (4, 3, 2), strides (8, 32, 96)
    assert plan([t], "naive")["strides"] == [(8, 32, 96)]
    assert plan([t], "reorder") == {"shape": (2, 3, 4), "strides": [(96, 32, 8)], "empty": False}
    assert plan([t]) == {"shape": (24,), "strides": [(8,)], "empty": False}


def test_uniform_slices_coalesce_across_rows():
    # Every second element of a (4, 6) array is one evenly spaced run of 12.
    assert plan([as_sd(np.zeros((4, 6))[:, ::2])])["shape"] == (12,)
    # Every second row is not: two runs of 6, 96 bytes apart.
    assert plan([as_sd(np.zeros((4, 6))[::2])]) == {"shape": (2, 6), "strides": [(96, 8)], "empty": False}


def test_extent_one_axes_are_dropped():
    assert plan([sd.zeros((1, 5, 1))]) == {"shape": (5,), "strides": [(8,)], "empty": False}
    assert plan([sd.zeros((1, 1))]) == {"shape": (1,), "strides": [(0,)], "empty": False}


def test_broadcast_operands_get_stride_zero():
    p = plan([sd.zeros((3, 1)), sd.zeros(4)])
    assert p == {"shape": (3, 4), "strides": [(8, 0), (0, 8)], "empty": False}


def test_coalescing_requires_every_operand_to_agree():
    c = sd.zeros((3, 4))
    t = as_sd(np.zeros((4, 3)).T)
    p = plan([t, c])  # input transposed, output C-order
    assert p["shape"] == (3, 4)  # cannot merge: contiguous for one, not the other
    assert p["strides"][1] == (32, 8)  # the output (last operand) decides the order


def test_empty_iteration():
    assert plan([sd.zeros((3, 0, 2))])["empty"] is True


@pytest.mark.parametrize("shapes", [[(2, 3), (3, 2)], [(4,), (5,)], [(2, 1, 3), (4, 3, 1), (2,)]])
def test_incompatible_broadcast_raises(shapes):
    with pytest.raises(ValueError):
        plan([sd.zeros(s) for s in shapes])


@pytest.mark.parametrize(
    "shapes", [[(5, 1, 4), (3, 1)], [(1,), (2, 3)], [(2, 1, 3), (4, 1), (1, 3)], [(0, 3), (1, 3)]]
)
def test_broadcast_shape_matches_numpy(shapes):
    arrays = [sd.zeros(s) for s in shapes]
    want = np.broadcast_shapes(*shapes)
    got = plan(arrays, "naive")["shape"]
    assert got == want
