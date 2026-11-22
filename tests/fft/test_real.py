"""rfft, irfft, hfft, ihfft and fft of real input (docs/api/fft.md §1.2, §3; docs/design/fft.md §2.3, §3.4,
§4.5; milestone FT2): numpy 2.5.4 and scipy 1.18.1 live on the same inputs (the difference from either is
a small part of the contract's bound, tests/fft/refdata.py), the parts of a Hermitian input that are
dropped and the default lengths (numpy's and scipy's), odd lengths, cutting and padding, the input
dtypes, the bit relations (rfft is the front of fft of real input, the rest of which is mirrored exactly;
ifft of real input, ihfft and hfft are conjugates with the other direction's factor), the exact
properties of §4.5, the recorded behaviours of numpy and scipy, non-finite values."""

import math

import numpy as np
import pytest
import scipy.fft as sf

import stardust as sd

from .._bits import bits_equal, value_equal
from . import refdata
from .conftest import LIVE

F = sd.fft
LENGTHS = list(range(1, 130)) + [192, 240, 254, 255, 256, 257, 500, 539, 1000, 1009, 1024, 1536, 2018, 2310, 4097]
NORMS = ("backward", "ortho", "forward")
SWAP = {"backward": "forward", "ortho": "ortho", "forward": "backward"}


def _rand(n, seed):
    return np.random.default_rng(seed).uniform(-1, 1, n)


def _factor(norm, n, inverse):
    return {"backward": 1 / n if inverse else 1.0, "ortho": 1 / math.sqrt(n), "forward": 1.0 if inverse else 1 / n}[
        norm
    ]


def _half(n, seed):
    """A half spectrum for an output of length n with nonzero imaginary parts everywhere (the DC and Nyquist
    ones must be dropped)."""
    g = np.random.default_rng(seed)
    m = n // 2 + 1
    return g.uniform(-1, 1, m) + 1j * g.uniform(-1, 1, m)


@LIVE
@pytest.mark.parametrize("n", LENGTHS)
def test_against_numpy_and_scipy(n):
    x = _rand(n, n)
    xn = refdata.norm2(x)
    h = _half(n, n + 1)
    hn = refdata.hermitian_norm(h, n)
    for norm in NORMS:
        sf_, si = _factor(norm, n, False), _factor(norm, n, True)
        got = np.asarray(F.rfft(sd.asarray(x), norm=norm))
        assert got.dtype == np.complex128 and got.shape == (n // 2 + 1,)
        b = refdata.bound("real", n, xn, sf_)
        for want in (np.fft.rfft(x, norm=norm), sf.rfft(x, norm=norm)):
            assert refdata.norm2(got - want) / b < 0.25
        got = np.asarray(F.ihfft(sd.asarray(x), norm=norm))
        b = refdata.bound("real", n, xn, si)
        for want in (np.fft.ihfft(x, norm=norm), sf.ihfft(x, norm=norm)):
            assert refdata.norm2(got - want) / b < 0.25
        for f, nf, scf in ((F.fft, np.fft.fft, sf.fft), (F.ifft, np.fft.ifft, sf.ifft)):
            got = np.asarray(f(sd.asarray(x), norm=norm))
            b = refdata.bound("real", n, xn, si if f is F.ifft else sf_)
            for want in (nf(x, norm=norm), scf(x, norm=norm)):
                assert refdata.norm2(got - want) / b < 0.25
        got = np.asarray(F.irfft(sd.asarray(h), n=n, norm=norm))
        assert got.dtype == np.float64 and got.shape == (n,)
        b = refdata.bound("hermitian", n, hn, si)
        for want in (np.fft.irfft(h, n=n, norm=norm), sf.irfft(h, n=n, norm=norm)):
            assert refdata.norm2(got - want) / b < 0.25
        got = np.asarray(F.hfft(sd.asarray(h), n=n, norm=norm))
        b = refdata.bound("hermitian", n, hn, {"backward": 1.0, "ortho": 1 / math.sqrt(n), "forward": 1 / n}[norm])
        for want in (np.fft.hfft(h, n=n, norm=norm), sf.hfft(h, n=n, norm=norm)):
            assert refdata.norm2(got - want) / b < 0.25


@LIVE
def test_dropped_parts_and_default_lengths_like_numpy_and_scipy():
    for m in (2, 3, 4, 5, 9, 17, 64):
        h = _half(2 * (m - 1), m)
        for f, nf, scf in ((F.irfft, np.fft.irfft, sf.irfft), (F.hfft, np.fft.hfft, sf.hfft)):
            got = np.asarray(f(sd.asarray(h)))
            assert got.shape == nf(h).shape == scf(h).shape == (2 * (m - 1),)
            # the imaginary parts of DC and Nyquist are dropped: the same bits without them
            plain = h.copy()
            plain[0] = plain[0].real
            plain[-1] = plain[-1].real
            assert bits_equal(got, f(sd.asarray(plain)))
            np.testing.assert_allclose(got, nf(h), rtol=0, atol=1e-13 * max(1, m))
            np.testing.assert_allclose(got, scf(h), rtol=0, atol=1e-13 * max(1, m))
        for n in (2 * m - 1, 2 * m - 3, m, 3 * m + 1):  # odd n: only DC's imaginary part is dropped
            got = np.asarray(F.irfft(sd.asarray(h), n=n))
            np.testing.assert_allclose(got, np.fft.irfft(h, n=n), rtol=0, atol=1e-14)
            np.testing.assert_allclose(got, sf.irfft(h, n=n), rtol=0, atol=1e-14)
            used = h[: n // 2 + 1].copy()
            used[0] = used[0].real
            if n % 2 == 0 and used.size == n // 2 + 1:  # Nyquist present (not padded)
                used[-1] = used[-1].real
            assert bits_equal(got, F.irfft(sd.asarray(used), n=n))
    # rfft and ihfft default to the axis's length
    x = _rand(7, 3)
    assert F.rfft(sd.asarray(x)).shape == np.fft.rfft(x).shape == (4,)
    assert F.ihfft(sd.asarray(x)).shape == np.fft.ihfft(x).shape == (4,)


def test_cut_and_pad():
    g = np.random.default_rng(5)
    x = g.uniform(-1, 1, (3, 10))
    for n in (1, 2, 4, 7, 10, 13, 32):
        for axis in (0, 1):
            shape = list(x.shape)
            shape[axis] = n
            pad = np.zeros(shape)
            k = min(n, x.shape[axis])
            sl = [slice(None)] * 2
            sl[axis] = slice(0, k)
            pad[tuple(sl)] = x[tuple(sl)]
            for f in (F.rfft, F.ihfft, F.fft, F.ifft):
                assert bits_equal(f(sd.asarray(x), n=n, axis=axis), f(sd.asarray(pad), axis=axis)), (f, n, axis)
    h = g.uniform(-1, 1, (3, 6)) + 1j * g.uniform(-1, 1, (3, 6))
    for n in (1, 2, 5, 10, 11, 12, 20):
        m = n // 2 + 1
        pad = np.zeros((3, m), np.complex128)
        pad[:, : min(m, 6)] = h[:, :m]
        for f in (F.irfft, F.hfft):
            assert bits_equal(f(sd.asarray(h), n=n), f(sd.asarray(pad), n=n)), (f, n)


def test_rfft_is_the_front_of_fft_and_the_rest_is_mirrored():
    # decision FT16, docs/design/fft.md §4.5: bit for bit, on every length form and value kind
    from . import signals as S

    for n in LENGTHS + [2 * 1009, 3 * 10007, 4 * 257, 1 << 14, 65536]:
        for kind in ("uniform", "wide", "tiny", "impulse"):
            x = S.get(f"r/{kind}/{n}").x
            y = np.asarray(F.fft(sd.asarray(x)))
            r = np.asarray(F.rfft(sd.asarray(x)))
            assert bits_equal(y[: n // 2 + 1], r), (n, kind)
            k = np.arange(1, (n + 1) // 2)
            assert bits_equal(y[n - k], np.conj(y[k])), (n, kind)
            assert r[0].imag == 0 and not math.copysign(1, r[0].imag) < 0  # +0
            if n % 2 == 0:
                assert r[-1].imag == 0 and not math.copysign(1, r[-1].imag) < 0
            for norm in NORMS:
                assert bits_equal(
                    F.ifft(sd.asarray(x), norm=SWAP[norm]), np.conj(np.asarray(F.fft(sd.asarray(x), norm=norm)))
                )


def test_hermitian_pairs_are_conjugates_with_the_other_direction():
    # numpy's definitions: ihfft(x) = conj(rfft(x)) and hfft(x) = irfft(conj(x)), with the norm swapped
    for n in (1, 2, 3, 8, 13, 26, 100, 1009, 1024):
        x = _rand(n, 2 * n)
        h = _half(n, 3 * n)
        for norm in NORMS:
            assert bits_equal(
                F.ihfft(sd.asarray(x), norm=norm), np.conj(np.asarray(F.rfft(sd.asarray(x), norm=SWAP[norm])))
            )
            assert bits_equal(
                F.hfft(sd.asarray(h), n=n, norm=norm), F.irfft(sd.asarray(np.conj(h)), n=n, norm=SWAP[norm])
            )
    # so the DC imaginary part of ihfft is -0 (numpy; [scipy와 다름]: +0)
    y = np.asarray(F.ihfft(sd.asarray(np.array([1.0, 2.0, 3.0, 4.0]))))
    assert math.copysign(1, y[0].imag) == -1


def test_norm_factors_are_one_multiplication():
    for n in (3, 12, 13, 26, 1000, 1009):
        x = sd.asarray(_rand(n, n + 7))
        y = np.asarray(F.rfft(x))
        assert bits_equal(F.rfft(x, norm="forward"), y * (1.0 / n))
        assert bits_equal(F.rfft(x, norm="ortho"), y * (1.0 / math.sqrt(n)))
        h = sd.asarray(_half(n, n + 9))
        yi = np.asarray(F.irfft(h, n=n, norm="forward"))
        assert bits_equal(F.irfft(h, n=n), yi * (1.0 / n))
        assert bits_equal(F.irfft(h, n=n, norm="ortho"), yi * (1.0 / math.sqrt(n)))


def test_scaling_by_powers_of_two_is_exact():
    for n in (8, 12, 13, 26, 1000, 1009, 4096):
        x = _rand(n, n)
        h = _half(n, n)
        y = np.asarray(F.rfft(sd.asarray(x)))
        z = np.asarray(F.irfft(sd.asarray(h), n=n))
        for k in (2.0, 0.25, 2.0**40):
            assert bits_equal(F.rfft(sd.asarray(x * k)), y * k), (n, k)
            assert bits_equal(F.irfft(sd.asarray(h * k), n=n), z * k), (n, k)


def test_length_one_and_two():
    for v in (2.5, -0.0, 1e308, 5e-324, np.nan, -np.inf):
        x = np.array([v])
        for norm in NORMS:
            for f in (F.rfft, F.fft):
                assert bits_equal(f(sd.asarray(x), norm=norm), np.array([complex(v, 0.0)]))
            assert bits_equal(F.irfft(sd.asarray(np.array([complex(v, 3.0)])), n=1, norm=norm), x)
    x = np.array([3.0, -1.25])
    assert bits_equal(F.rfft(sd.asarray(x)), np.array([1.75 + 0j, 4.25 + 0j]))
    assert bits_equal(F.irfft(sd.asarray(np.array([1.75 + 5j, 4.25 - 7j])), norm="forward"), 2 * x)


def test_input_dtypes():
    # bool, int64 and float64 are float64 values; irfft and hfft read real input as complex with zero
    # imaginary parts; rfft and ihfft refuse complex input (numpy and scipy too, tests/fft/test_errors.py)
    for a in (np.array([True, False, True, True]), np.array([1, 2, 3, 4, -7]), np.array([0.5, -1.0, 2.0])):
        for f in (F.rfft, F.ihfft, F.fft, F.ifft):
            got = f(sd.asarray(a))
            assert got.dtype == sd.complex128
            assert bits_equal(got, f(sd.asarray(a.astype(np.float64))))
        for f in (F.irfft, F.hfft):
            got = f(sd.asarray(a))
            assert got.dtype == sd.float64
            assert bits_equal(got, f(sd.asarray(a.astype(np.complex128))))


def test_round_trip():
    for n in (1, 2, 7, 64, 97, 1000, 1009, 2018):
        x = _rand(n, n + 1)
        back = np.asarray(F.irfft(F.rfft(sd.asarray(x)), n=n))
        b = refdata.bound("real", n, refdata.norm2(x), 1 / math.sqrt(n)) * math.sqrt(n)
        assert refdata.norm2(back - x) <= 2 * b, n


def test_result_is_new_and_input_untouched():
    x = _rand(16, 4)
    sx = sd.asarray(x)
    for f in (F.rfft, F.ihfft, F.fft):
        y = f(sx)
        assert bits_equal(sx, x)
        assert not np.shares_memory(np.asarray(y), np.asarray(sx))
    h = sd.asarray(_half(16, 5))
    before = np.asarray(h).copy()
    for f in (F.irfft, F.hfft):
        f(h)
        assert bits_equal(h, before)


def test_empty_batch_axis_and_shapes():
    assert F.rfft(sd.zeros((0, 4))).shape == (0, 3)
    assert F.irfft(sd.zeros((0, 4))).shape == (0, 6)
    assert F.hfft(sd.zeros((3, 0, 2)), axis=0, n=5).shape == (5, 0, 2)
    y = F.rfft(sd.ones((2, 3, 5)), axis=1, n=4)
    assert y.shape == (2, 3, 5) and y.dtype == sd.complex128
    assert np.asarray(y).flags["C_CONTIGUOUS"]
    z = F.irfft(sd.ones((2, 3, 5)), axis=1)
    assert z.shape == (2, 4, 5) and z.dtype == sd.float64


def test_recorded_behaviours_of_numpy_and_scipy():
    # tests/fft/data/{numpy,scipy}.json (tools/fft_probe.py)
    def arr(v):
        a = np.array(v["array"])
        return a[..., 0] + 1j * a[..., 1] if v["dtype"] == "complex128" and a.size else a.reshape(v["shape"])

    for lib in ("numpy", "scipy"):
        rec = refdata.data(lib)
        assert value_equal(F.hfft(sd.asarray(np.array([1, 2 + 1j, 3 + 5j]))), arr(rec["hfft/[1,2+1j,3+5j]"]["value"]))
        assert rec["irfft/DC and Nyquist imag dropped"]["value"] is True
        assert value_equal(F.irfft(sd.asarray(np.array([0.5 + 0j, 0.75, 0.0]))).shape, (4,))
        assert value_equal(F.rfft(sd.asarray(np.array([1.0]))), arr(rec["rfft/n=1"]["value"]))
        want = arr(rec["irfft/n=5 odd"]["value"])
        assert rec["irfft/length 1 default n"]["error"] == "ValueError"
        assert rec["rfft/complex input"]["error"] == rec["ihfft/complex input"]["error"] == "TypeError"
        want = arr(rec["ihfft/[1,2,3,4]"]["value"])
        got = np.asarray(F.ihfft(sd.asarray(np.array([1.0, 2.0, 3.0, 4.0]))))
        assert np.array_equal(got, want)  # values; the sign of DC's imaginary zero is numpy's
    assert refdata.data("numpy")["ihfft/DC imag sign"]["value"] == -1.0  # ours too
    assert refdata.data("scipy")["ihfft/DC imag sign"]["value"] == 1.0  # [scipy와 다름]
    # numpy's rfft is not the front of its fft from length 6 on; scipy's is (and ours, above)
    assert refdata.data("numpy")["rfft/differs from the start of fft at lengths"]["value"][0] == 6
    assert refdata.data("scipy")["rfft/differs from the start of fft at lengths"]["value"] == []
    # real input to fft: DC's imaginary part +0 (numpy; scipy -0)
    y = np.asarray(F.fft(sd.asarray(np.array([True, False, True]))))
    assert value_equal(y, arr(refdata.data("numpy")["fft/bool"]["value"]))
    assert math.copysign(1, y[0].imag) == 1


def test_nonfinite_is_ieee_and_deterministic():
    # FT17: no warning; the pattern is the same in any batch (the line's own bits)
    from . import signals as S

    for n in (4, 13, 16, 26):
        batch = np.array([x for sid, x in S.nonfinite(n) if sid.startswith("r/")])
        for f in (F.rfft, F.fft, F.ihfft):
            lines = np.asarray(f(sd.asarray(batch)))
            for k, x in enumerate(batch):
                one = np.asarray(f(sd.asarray(x)))
                assert bits_equal(one, lines[k])
                assert not np.all(np.isfinite(one))
        hb = np.asarray(F.rfft(sd.asarray(batch)))
        hb[:, 1] = complex(np.nan, 1.0)
        lines = np.asarray(F.irfft(sd.asarray(hb), n=n))
        for k in range(hb.shape[0]):
            assert bits_equal(F.irfft(sd.asarray(hb[k]), n=n), lines[k])
