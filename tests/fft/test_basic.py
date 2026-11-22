"""fft and ifft (docs/api/fft.md §2; docs/design/fft.md §2.2, §2.4, §2.5, §2.8): numpy 2.5.4 and scipy
1.18.1 live on the same inputs (the difference from either is a small part of the contract's bound,
tests/fft/refdata.py), R's recorded results, the shapes, dtypes, cutting and padding, norm, the
length-1 copy, the empty batch axis, the dtypes of the input, the exact properties of §4.5, the
inverse and non-finite values."""

import math

import numpy as np
import pytest
import scipy.fft as sf

import stardust as sd

from .._bits import bits_equal
from . import refdata
from .conftest import LIVE

F = sd.fft
LENGTHS = list(range(1, 130)) + [192, 240, 255, 256, 257, 500, 539, 1000, 1009, 1024, 1536, 2018, 2310, 3072, 4097]


def _rand(shape, seed, complex_=True):
    g = np.random.default_rng(seed)
    x = g.uniform(-1, 1, shape)
    return x + 1j * g.uniform(-1, 1, shape) if complex_ else x


def _units(got, want, n, xnorm, s=1.0):
    """||got - want|| in units of the contract's bound (want: another library's result, so the units
    are those of the difference of two results each within the contract: at most 2)."""
    return refdata.norm2(np.asarray(got) - want) / refdata.bound("c2c", n, xnorm, s)


@LIVE
@pytest.mark.parametrize("n", LENGTHS)
def test_against_numpy_and_scipy(n):
    x = _rand(n, n)
    xn = refdata.norm2(x)
    for norm, s in (("backward", 1.0), ("ortho", 1 / math.sqrt(n)), ("forward", 1 / n)):
        got = np.asarray(F.fft(sd.asarray(x), norm=norm))
        assert got.dtype == np.complex128 and got.shape == (n,)
        assert _units(got, np.fft.fft(x, norm=norm), n, xn, s) < 0.25
        assert _units(got, sf.fft(x, norm=norm), n, xn, s) < 0.25
        si = {"backward": 1 / n, "ortho": 1 / math.sqrt(n), "forward": 1.0}[norm]
        got = np.asarray(F.ifft(sd.asarray(x), norm=norm))
        assert _units(got, np.fft.ifft(x, norm=norm), n, xn, si) < 0.25
        assert _units(got, sf.ifft(x, norm=norm), n, xn, si) < 0.25


@LIVE
def test_cut_and_pad_like_numpy():
    x = _rand((3, 10), 1)
    for n in (1, 4, 10, 13, 32):
        for axis in (0, 1, -1):
            got = np.asarray(F.fft(sd.asarray(x), n=n, axis=axis))
            want = np.fft.fft(x, n=n, axis=axis)
            assert got.shape == want.shape and got.dtype == want.dtype
            assert np.allclose(got, want, rtol=0, atol=1e-13)
            # cutting and padding are exactly the transform of the cut or padded array
            shape = list(x.shape)
            shape[axis] = n
            pad = np.zeros(shape, np.complex128)
            k = min(n, x.shape[axis])
            sl = [slice(None)] * 2
            sl[axis] = slice(0, k)
            pad[tuple(sl)] = x[tuple(sl)]
            assert bits_equal(got, F.fft(sd.asarray(pad), axis=axis))


def test_norm_factors_are_one_multiplication():
    # 'forward' is the 'backward' result times fl(1/n), 'ortho' times fl(1/fl(sqrt(n))) (§2.5)
    for n in (3, 12, 13, 1000, 1009):
        x = sd.asarray(_rand(n, n + 7))
        y = np.asarray(F.fft(x))
        assert bits_equal(F.fft(x, norm="forward"), y * (1.0 / n))
        assert bits_equal(F.fft(x, norm="ortho"), y * (1.0 / math.sqrt(n)))
        yi = np.asarray(F.ifft(x, norm="forward"))
        assert bits_equal(F.ifft(x), yi * (1.0 / n))
        assert bits_equal(F.ifft(x, norm="ortho"), yi * (1.0 / math.sqrt(n)))


def test_length_one_is_a_copy():
    for v in (2.5 - 1j, complex(-0.0, 0.0), complex(np.nan, 1.0), complex(5e-324, -np.inf)):
        x = np.array([v])
        for f in (F.fft, F.ifft):
            for norm in ("backward", "ortho", "forward"):
                assert bits_equal(f(sd.asarray(x), norm=norm), x)
    x = _rand((4, 1), 3)
    assert bits_equal(F.fft(sd.asarray(x), axis=1), x)


def test_input_dtypes():
    # bool, int64 and float64 are taken as float64 values (FT1: through the complex path with zero
    # imaginary parts until milestone FT2 brings the real path, decision FT38)
    for a in (np.array([True, False, True]), np.array([1, 2, 3, 4]), np.array([0.5, -1.0, 2.0])):
        got = F.fft(sd.asarray(a))
        assert got.dtype == sd.complex128
        assert bits_equal(got, F.fft(sd.asarray(a.astype(np.complex128))))


def test_empty_batch_axis_and_shapes():
    assert F.fft(sd.zeros((0, 4))).shape == (0, 4)
    assert F.fft(sd.zeros((0, 4)), axis=1, n=7).shape == (0, 7)
    assert F.ifft(sd.zeros((3, 0, 2)), axis=0).shape == (3, 0, 2)
    y = F.fft(sd.ones((2, 3, 5)), axis=1, n=4)
    assert y.shape == (2, 4, 5) and y.dtype == sd.complex128
    assert np.asarray(y).flags["C_CONTIGUOUS"]


def test_result_is_new_and_input_untouched():
    x = _rand(16, 4)
    sx = sd.asarray(x)
    before = np.asarray(sx).copy()
    y = F.fft(sx)
    assert bits_equal(sx, before)
    assert not np.shares_memory(np.asarray(y), np.asarray(sx))


def test_scaling_by_powers_of_two_is_exact():
    # §4.5: the same operations with other exponents
    for n in (8, 12, 13, 1000, 1009, 4096):
        x = _rand(n, n)
        y = np.asarray(F.fft(sd.asarray(x)))
        for k in (2.0, 0.25, 2.0**40):
            assert bits_equal(F.fft(sd.asarray(x * k)), y * k), (n, k)


def test_inverse_is_the_swapped_forward():
    # swap(z) = i conj(z): ifft(x) * n == swap(fft(swap(x))) bit for bit (cfft.hpp's inverse)
    for n in (5, 16, 13, 360, 1009, 2048):
        x = _rand(n, 2 * n)
        sw = x.imag + 1j * x.real
        f = np.asarray(F.fft(sd.asarray(sw)))
        assert bits_equal(F.ifft(sd.asarray(x), norm="forward"), f.imag + 1j * f.real), n


def test_round_trip():
    for n in (7, 64, 97, 1000, 1009):
        x = _rand(n, n + 1)
        back = np.asarray(F.ifft(F.fft(sd.asarray(x))))
        assert refdata.norm2(back - x) <= 2 * refdata.bound("c2c", n, refdata.norm2(x), 1.0 / math.sqrt(n)) * math.sqrt(
            n
        )


def test_r_conventions():
    # R's fft(z, inverse=TRUE) is ifft(z, norm='forward'); mvfft is fft(Z, axis=0) (docs/api/fft.md §2)
    rec = refdata.data("r")
    from . import signals as S

    for key, val in rec.items():
        if not key.startswith("fft/"):
            continue
        s = S.get(key[len("fft/") :])
        want = np.array([complex(a, b) for a, b in val["values"]])
        got = np.asarray(F.fft(sd.asarray(s.x)))
        assert refdata.norm2(got - want) <= 2 * refdata.bound("c2c", s.n, refdata.norm2(s.x)), key
    val = rec["mvfft/c/uniform/24 as 12x2"]["values"]
    z = S.get("c/uniform/24").x.reshape(12, 2, order="F")
    got = np.asarray(F.fft(sd.asarray(z), axis=0)).ravel(order="F")
    want = np.array([complex(a, b) for a, b in val])
    assert refdata.norm2(got - want) <= 4 * refdata.bound("c2c", 12, refdata.norm2(z))


def test_nonfinite_is_ieee_and_deterministic():
    # FT17: no warning, IEEE through the computation; the pattern is the same on every run and in any
    # batch (the line's own bits)
    from . import signals as S

    for n in (4, 13, 16):
        cases = list(S.nonfinite(n))
        batch = np.array([z for sid, z in cases if sid.startswith("c/")])
        lines = np.asarray(F.fft(sd.asarray(batch)))
        for k, z in enumerate(batch):
            one = np.asarray(F.fft(sd.asarray(z)))
            assert bits_equal(one, lines[k])
            assert not np.all(np.isfinite(one))


def test_recorded_behaviours_of_numpy_and_scipy():
    # tests/fft/data/{numpy,scipy}.json (tools/fft_probe.py): the non-rounding behaviour is theirs
    from .._bits import value_equal

    def arr(v):
        a = np.array(v["array"])
        return a[..., 0] + 1j * a[..., 1] if v["dtype"] == "complex128" and a.size else a.reshape(v["shape"])

    for lib in ("numpy", "scipy"):
        rec = refdata.data(lib)
        assert bits_equal(F.fft(sd.asarray(np.array([2.5 - 1j]))), arr(rec["fft/length 1 is a copy"]["value"]))
        assert list(F.fft(sd.zeros((0, 4))).shape) == rec["fft/length-0 batch axis"]["value"]["shape"]
        for key in ("fft/length-0 transform axis", "fft/n=0", "fft/n=-1", "fft/empty"):
            assert rec[key]["error"] == "ValueError"
        want = arr(rec["ifft/[1,2,3,4] norm=forward"]["value"])
        assert value_equal(F.ifft(sd.asarray([1.0, 2.0, 3.0, 4.0]), norm="forward"), want)
        want = arr(rec["fft/int64"]["value"])
        assert value_equal(F.fft(sd.asarray(np.array([0, 1, 2, 3]))), want)
        want = arr(rec["fft/n=6 (zero padding)"]["value"])
        got = np.asarray(F.fft(sd.asarray([1.0, 2.0, 3.0, 4.0]), n=6))
        assert refdata.norm2(got - want) <= 2 * refdata.bound("c2c", 6, refdata.norm2(np.arange(1.0, 5.0)))
        assert rec["fft/scaling by 2 is exact"]["value"] is True
    # docs/design/fft.md §4.1: the entry whose true value is -2 + 0i is not exact in any of them
    y3 = complex(np.asarray(F.fft(sd.asarray([1.0, 2.0, 3.0, 4.0]), n=6))[3])
    assert y3.real == -2.0 and abs(y3.imag) <= refdata.bound("c2c", 6, refdata.norm2(np.arange(1.0, 5.0)))
