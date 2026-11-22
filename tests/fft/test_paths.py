"""A line's bits are those of the line transformed alone (docs/design/fft.md §1.3, §3.7, §4.5): in any
batch shape, memory layout (C, F, negative and gapped strides, the transform axis first, in the middle or
last), with cutting and padding, on both sides of the mixed-radix / Bluestein boundary and of the
driver's direct and gathered paths; the same for the real and Hermitian transforms (milestone FT2), whose
even and odd lengths take different paths (decision FT43)."""

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
    # 31-smooth lengths (mixed radix, 13 to 31 among them) next to lengths with a larger prime (Bluestein:
    # 122, 2047, 2049, 9241), and lengths whose lines fill a task's block differently
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


R_FUNCS = ("rfft", "ihfft", "fft", "ifft")
H_FUNCS = ("irfft", "hfft")


@pytest.mark.parametrize("n", [12, 13, 16, 26, 1009, 1000])
def test_real_layouts(n):
    # the real parts of the complex layouts: float64 lines with a 16-byte stride, and copies of them
    for name in R_FUNCS:
        f = getattr(F, name)
        for lname, arr, axis in S.layouts(n=n, batch=5):
            if arr.size == 0:
                continue
            for r in (arr.real, np.array(arr.real, order="K")):
                got = np.asarray(f(sd.asarray(r), axis=axis))
                lines = np.moveaxis(r, axis, -1).reshape(-1, n)
                alone = _lines_alone(lines, f)
                assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, alone.shape[1]), alone), (lname, name)


@pytest.mark.parametrize("n", [12, 13, 16, 26, 1009, 1000])
def test_hermitian_layouts(n):
    # complex layouts as half spectra: lines of n // 2 + 1 entries cut from them, or padded
    for name in H_FUNCS:
        f = getattr(F, name)
        for lname, arr, axis in S.layouts(n=n, batch=5):
            if arr.size == 0:
                continue
            m = arr.shape[axis]
            for out_n in (n, 2 * (m - 1), 2 * m + 3):
                got = np.asarray(f(sd.asarray(arr), axis=axis, n=out_n))
                lines = np.moveaxis(arr, axis, -1).reshape(-1, m)
                alone = _lines_alone(lines, f, n=out_n)
                assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, out_n), alone), (lname, name, out_n)


def test_real_direct_and_gathered_paths_agree():
    g = np.random.default_rng(6)
    for n in (8, 9, 64, 13, 26, 1000, 1009):
        x = g.uniform(-1, 1, (9, n + 3))
        direct = np.asarray(F.rfft(sd.asarray(np.ascontiguousarray(x[:, :n]))))
        assert bits_equal(F.rfft(sd.asarray(x), n=n), direct)  # in place with a cut
        assert bits_equal(F.rfft(sd.asarray(np.asfortranarray(x[:, :n]))), direct)
        assert bits_equal(F.rfft(sd.asarray(x[:, :n].T.copy()), axis=0).T, direct)
        assert bits_equal(
            F.rfft(sd.asarray((x[:, :n] * 4).astype(np.int64))), F.rfft(sd.asarray(np.trunc(x[:, :n] * 4)))
        )
        h = direct + 0.25j
        back = np.asarray(F.irfft(sd.asarray(h), n=n))
        assert bits_equal(F.irfft(sd.asarray(np.asfortranarray(h)), n=n), back)
        assert bits_equal(F.irfft(sd.asarray(h.T.copy()), n=n, axis=0).T, back)
        wide = np.concatenate([h, h], axis=1)
        assert bits_equal(F.irfft(sd.asarray(wide), n=n), back)  # in place with a cut


def test_real_boundary_lengths_of_the_paths():
    # even lengths whose half is 31-smooth or not, odd lengths of both kinds, 1 and 2
    for n in (1, 2, 3, 4, 22, 24, 26, 27, 34, 121, 122, 143, 169, 2046, 2048, 2049, 2050, 2054, 18482):
        x = S.get(f"r/uniform/{n}").x
        batch = np.stack([x, 2 * x, x[::-1].copy()])
        got = np.asarray(F.rfft(sd.asarray(batch)))
        assert bits_equal(got[0], F.rfft(sd.asarray(x)))
        assert bits_equal(got[1], 2 * got[0])
        assert bits_equal(got[2], F.rfft(sd.asarray(x[::-1].copy())))
        back = np.asarray(F.irfft(sd.asarray(got), n=n))
        assert bits_equal(back[1], 2 * back[0])
        assert bits_equal(back[0], F.irfft(sd.asarray(got[0]), n=n))


def test_real_large_batches_split_like_one_line():
    g = np.random.default_rng(7)
    for shape, axis in (((257, 24), 1), ((24, 257), 0), ((3, 40, 33), 1), ((70, 2, 26), 0), ((40, 1000), 1)):
        x = g.uniform(-1, 1, shape)
        n = shape[axis]
        for name in ("rfft", "fft", "ihfft"):
            f = getattr(F, name)
            got = np.asarray(f(sd.asarray(x), axis=axis))
            lines = np.moveaxis(x, axis, -1).reshape(-1, n)
            alone = _lines_alone(lines, f)
            assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, alone.shape[1]), alone), (shape, name)
        h = np.asarray(F.rfft(sd.asarray(x), axis=axis))
        got = np.asarray(F.irfft(sd.asarray(h), axis=axis, n=n))
        lines = np.moveaxis(h, axis, -1).reshape(-1, h.shape[axis])
        assert bits_equal(np.moveaxis(got, axis, -1).reshape(-1, n), _lines_alone(lines, F.irfft, n=n)), shape
