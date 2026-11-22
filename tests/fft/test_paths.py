"""A line's bits are those of the line transformed alone (docs/design/fft.md §1.3, §3.7, §4.5): in any
batch shape, memory layout (C, F, negative and gapped strides, the transform axis first, in the middle or
last), with cutting and padding, on both sides of the mixed-radix / Bluestein boundary and of the
driver's direct and gathered paths."""

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import signals as S

F = sd.fft


def _lines_alone(lines, f, **kw):
    return np.stack([np.asarray(f(sd.asarray(np.ascontiguousarray(v)), **kw)) for v in lines])


@pytest.mark.parametrize("n", [12, 13, 16, 1009])
def test_layouts(n):
    for f in (F.fft, F.ifft):
        ref = None
        for name, arr, axis in S.layouts(n=n, batch=5):
            if arr.shape[axis] == 0 or arr.size == 0:
                assert F.fft(sd.asarray(arr), axis=axis).shape == arr.shape
                continue
            got = np.asarray(f(sd.asarray(arr), axis=axis))
            lines = np.moveaxis(arr, axis, -1).reshape(-1, n)
            alone = _lines_alone(lines, f)
            assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, n), alone), (name, f.__name__)
            if ref is None:
                ref = alone[:5]


def test_every_axis_of_a_3d_array():
    g = np.random.default_rng(3)
    x = g.standard_normal((6, 7, 8)) + 1j * g.standard_normal((6, 7, 8))
    for axis in range(3):
        for n in (None, 5, 11):
            got = np.asarray(F.fft(sd.asarray(x), axis=axis, n=n))
            lines = np.moveaxis(x, axis, -1).reshape(-1, x.shape[axis])
            nn = n or x.shape[axis]
            alone = _lines_alone(lines, F.fft, n=nn)
            assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, nn), alone), (axis, n)


def test_direct_and_gathered_paths_agree():
    # the last axis, contiguous complex128, read in place; the same lines gathered (F order, a slice,
    # a cut) must give the same bytes
    g = np.random.default_rng(4)
    for n in (8, 9, 64, 13, 1000):
        x = g.uniform(-1, 1, (9, n + 3)) + 1j * g.uniform(-1, 1, (9, n + 3))
        direct = np.asarray(F.fft(sd.asarray(np.ascontiguousarray(x[:, :n]))))
        assert bits_equal(F.fft(sd.asarray(x), n=n), direct)  # in place with a cut
        assert bits_equal(F.fft(sd.asarray(np.asfortranarray(x[:, :n]))), direct)
        assert bits_equal(F.fft(sd.asarray(x[:, :n].T.copy()), axis=0).T, direct)


def test_boundary_lengths_of_the_paths():
    # 11-smooth lengths (mixed radix) next to lengths with a larger prime (Bluestein), and lengths
    # whose lines fill a task's block differently
    for n in (10, 11, 12, 13, 14, 22, 23, 26, 120, 121, 122, 143, 169, 2047, 2048, 2049, 9240, 9241):
        x = S.get(f"c/uniform/{n}").x
        batch = np.stack([x, 2 * x, x[::-1].copy()])
        got = np.asarray(F.fft(sd.asarray(batch)))
        assert bits_equal(got[0], F.fft(sd.asarray(x)))
        assert bits_equal(got[1], 2 * got[0])
        assert bits_equal(got[2], F.fft(sd.asarray(x[::-1].copy())))


def test_large_batches_split_like_one_line():
    # enough lines to split into tasks and to fill several gather blocks
    g = np.random.default_rng(5)
    for shape, axis in (((257, 24), 1), ((24, 257), 0), ((3, 40, 33), 1), ((70, 2, 13), 0)):
        x = g.uniform(-1, 1, shape) + 1j * g.uniform(-1, 1, shape)
        got = np.asarray(F.fft(sd.asarray(x), axis=axis))
        n = shape[axis]
        lines = np.moveaxis(x, axis, -1).reshape(-1, n)
        assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, n), _lines_alone(lines, F.fft)), shape
