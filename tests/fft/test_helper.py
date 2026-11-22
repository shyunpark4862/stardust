"""fftfreq, rfftfreq, fftshift, ifftshift, next_fast_len and prev_fast_len (docs/api/fft.md §5; docs/design
/fft.md §2.7; decisions FT18, FT26): numpy's bits for the frequencies, roll's bits and dtype for the
shifts, scipy's values for the fast lengths (its recorded 1..2000 and live on large targets) and R's
nextn for the real ones."""

import numpy as np
import pytest
import scipy.fft as sf

import stardust as sd

from .._bits import bits_equal
from . import refdata
from .conftest import LIVE

F = sd.fft


@LIVE
def test_fftfreq_is_numpys_bits():
    for n in list(range(1, 40)) + [100, 1000, 1023, 4097]:
        for d in (1.0, 0.1, -0.5, 3, 1e-300, 7e300, np.float64(0.3), np.int64(2)):
            assert bits_equal(F.fftfreq(n, d=d), np.fft.fftfreq(n, d=d)), (n, d)
            assert bits_equal(F.rfftfreq(n, d=d), np.fft.rfftfreq(n, d=d)), (n, d)
    assert F.fftfreq(np.int64(4)).dtype == sd.float64


def test_fftfreq_records():
    for lib in ("numpy", "scipy"):
        rec = refdata.data(lib)
        assert np.asarray(F.fftfreq(5)).tolist() == rec["fftfreq/5"]["value"]["array"]
        assert bits_equal(F.fftfreq(4, d=-0.5), np.array(rec["fftfreq/d=-0.5"]["value"]["array"]))
        assert np.asarray(F.rfftfreq(6)).tolist() == rec["rfftfreq/6"]["value"]["array"]


@LIVE
def test_shifts_are_roll():
    g = np.random.default_rng(1)
    for shape in ((5,), (4,), (3, 4), (2, 5, 7), (0, 3)):
        for dtype in (np.bool_, np.int64, np.float64, np.complex128):
            x = (g.standard_normal(shape) * 3).astype(dtype)
            sx = sd.asarray(x)
            for axes in (None, 0, -1, (0,), [0, -1] if len(shape) > 1 else [0], tuple(range(len(shape)))):
                got = F.fftshift(sx, axes=axes)
                assert got.dtype == sx.dtype
                assert bits_equal(got, np.fft.fftshift(x, axes=axes)), (shape, dtype, axes)
                assert bits_equal(F.ifftshift(sx, axes=axes), np.fft.ifftshift(x, axes=axes))
                assert bits_equal(F.ifftshift(got, axes=axes), x)
    x = sd.arange(10).reshape((2, 5))
    assert bits_equal(F.fftshift(x), sd.roll(x, (1, 2), axis=(0, 1)))


def test_shift_returns_a_new_array():
    x = sd.arange(6)
    y = F.fftshift(x, axes=())
    assert bits_equal(y, x) and not np.shares_memory(np.asarray(y), np.asarray(x))


def test_fast_lengths_are_scipys():
    rec = refdata.data("scipy")
    for key, real, prev in (
        ("next_fast_len/1..2000", False, False),
        ("next_fast_len/real 1..2000", True, False),
        ("prev_fast_len/1..2000", False, True),
        ("prev_fast_len/real 1..2000", True, True),
    ):
        f = F.prev_fast_len if prev else F.next_fast_len
        assert [f(t, real=real) for t in range(1, 2001)] == rec[key]["value"], key
    assert [F.next_fast_len(t, real=True) for t in range(1, 2001)] == refdata.data("r")["nextn/1..2000"]["value"]


@LIVE
def test_fast_lengths_large_targets_live():
    g = np.random.default_rng(3)
    targets = [int(t) for t in g.integers(1, 2**60, 200)] + [2**40 + 1, 10**17, 2**60]  # scipy refuses far larger ones
    for t in targets:
        for real in (False, True):
            assert F.next_fast_len(t, real=real) == sf.next_fast_len(t, real=real), (t, real)
            assert F.prev_fast_len(t, real=real) == sf.prev_fast_len(t, real=real), (t, real)
    assert F.next_fast_len(np.int64(13)) == 14
    # every next fast length is a length the mixed radix transforms directly (no Bluestein)
    for t in range(1, 3000, 7):
        assert refdata.is_smooth11(F.next_fast_len(t))
        assert sd._core.fft._plan_info(F.next_fast_len(t))["path"] == "mixed"


def test_fast_lengths_beyond_int64():
    # scipy refuses targets above 2^62 ("too large to perform an FFT"); here every int64 target has an
    # answer unless the next fast length is beyond int64 (ValueError, docs/api/fft.md §5)
    assert F.prev_fast_len(2**63 - 1) == 9223306985706024960
    assert F.prev_fast_len(2**63 - 1, real=True) == 9216000000000000000
    msg = r"^next_fast_len\(\): the next fast length after 9223372036854775807 is beyond int64$"
    for real in (False, True):
        with pytest.raises(ValueError, match=msg):
            F.next_fast_len(2**63 - 1, real=real)
