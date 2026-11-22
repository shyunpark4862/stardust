"""The one-dimensional transforms against the Arb references (docs/design/fft.md §4, §7.2, §7.3): every
stored line of tests/fft/ref/c2c.npz (forward by fft, backward by ifft(norm='forward'): both
unnormalised), of real.npz (rfft, the whole spectrum of fft of real input, ihfft) and of hermitian.npz
(irfft(norm='forward'), hfft of the conjugate: both unnormalised) is within the contract, on every kind
of signal (uniform, normal, impulse, constant, sines, wide, huge, tiny) and every length form (2^k,
f 2^k, the other 11-smooth lengths, Bluestein; even and odd real lengths); and the long lines the files
leave out (2^16, 65537, 3 10007; decision FT33) with Arb computed here. The n-dimensional transforms
(milestone FT3): every stored line of nd.npz (fftn, ifftn(norm='forward')) and the points of
refdata.nd_points (Bluestein axes, rfftn and irfftn on even, odd and Bluestein last lengths, on uniform,
wide, huge and subnormal values) with Arb computed here, judged by refdata.bound_nd."""

import numpy as np
import pytest

import stardust as sd

from . import refdata
from . import signals as S

F = sd.fft
REF = refdata.ref("c2c")


def _points(method):
    m = REF.methods[method]
    for j, k in enumerate(m.index):
        yield str(REF.inputs["signal"][k]), REF.inputs["x"][k], int(REF.inputs["n"][k]), m.hi[j], m.lo[j]


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_every_stored_line_within_contract(method):
    worst = (0.0, "")
    count = 0
    for sid, x, n, hi, lo in _points(method):
        sx = sd.asarray(x)
        got = F.fft(sx) if method == "forward" else F.ifft(sx, norm="forward")
        u = refdata.units(refdata.error(np.asarray(got), hi, lo), refdata.bound("c2c", n, refdata.norm2(x)))
        assert u <= 1.0, (sid, u)
        worst = max(worst, (u, sid))
        count += 1
    assert count >= (313 if method == "forward" else 200)
    # FT1 measured at most 0.054 units (the tiny signals of n = 7 and 13; docs/design/fft.md §4.4)
    assert worst[0] < 0.1, worst


@pytest.mark.parametrize("sid", refdata.LONG[:3])
def test_long_lines_within_contract(sid):
    s = S.get(sid)
    for inverse in (False, True):
        hi, lo = refdata.arb_dft(s.x, inverse)
        got = F.ifft(sd.asarray(s.x), norm="forward") if inverse else F.fft(sd.asarray(s.x))
        u = refdata.units(refdata.error(np.asarray(got), hi, lo), refdata.bound("c2c", s.n, refdata.norm2(s.x)))
        assert u < 0.1, (sid, inverse, u)


def _family(name, method):
    r = refdata.ref(name)
    m = r.methods[method]
    for j, k in enumerate(m.index):
        yield str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]


def _whole(half, n):
    """The whole spectrum (n entries) of a half spectrum of a real signal."""
    return np.concatenate([half, np.conj(half[1 : (n + 1) // 2][::-1])])


def test_every_stored_real_line_within_contract():
    worst = (0.0, "")
    count = 0
    for sid, x, n, hi, lo in _family("real", "forward"):
        sx = sd.asarray(x)
        b = refdata.bound("real", n, refdata.norm2(x))
        for name, got, want in (
            ("rfft", F.rfft(sx), (hi, lo)),
            ("fft", F.fft(sx), (_whole(hi, n), _whole(lo, n))),
            ("ihfft", np.conj(np.asarray(F.ihfft(sx, norm="forward"))), (hi, lo)),
        ):
            u = refdata.units(refdata.error(np.asarray(got), *want), b)
            assert u <= 1.0, (name, sid, u)
            worst = max(worst, (u, f"{name} {sid}"))
        count += 1
    assert count >= 282
    # FT2 measured at most 0.042 units (docs/design/fft.md §4.4)
    assert worst[0] < 0.1, worst


def test_every_stored_hermitian_line_within_contract():
    worst = (0.0, "")
    count = 0
    for sid, x, n, hi, lo in _family("hermitian", "c2r"):
        b = refdata.bound("hermitian", n, refdata.hermitian_norm(x, n))
        for name, got in (
            ("irfft", F.irfft(sd.asarray(x), n=n, norm="forward")),
            ("hfft", F.hfft(sd.asarray(np.conj(x)), n=n)),
        ):
            u = refdata.units(refdata.error(np.asarray(got), hi, lo), b)
            assert u <= 1.0, (name, sid, u)
            worst = max(worst, (u, f"{name} {sid}"))
        count += 1
    assert count >= 113
    # FT2 measured at most 0.049 units (docs/design/fft.md §4.4)
    assert worst[0] < 0.1, worst


@pytest.mark.parametrize("sid", refdata.LONG_REAL[:2])
def test_long_real_lines_within_contract(sid):
    s = S.get(sid)
    hi, lo = refdata.arb_dft(s.x.astype(np.complex128))
    got = np.asarray(F.fft(sd.asarray(s.x)))
    u = refdata.units(refdata.error(got, hi, lo), refdata.bound("real", s.n, refdata.norm2(s.x)))
    assert u < 0.1, (sid, u)
    # the inverse: the half spectrum back to the n reals (times n)
    half = np.asarray(F.rfft(sd.asarray(s.x)))
    hi, lo = refdata.arb_dft(_whole(half, s.n), inverse=True)
    got = np.asarray(F.irfft(sd.asarray(half), n=s.n, norm="forward"))
    b = refdata.bound("hermitian", s.n, refdata.hermitian_norm(half, s.n))
    assert refdata.units(refdata.error(got, hi.real, lo.real), b) < 0.1, sid


def test_every_stored_nd_line_within_contract():
    r = refdata.ref("nd")
    worst = (0.0, "")
    for method in ("forward", "backward"):
        m = r.methods[method]
        for j, k in enumerate(m.index):
            shape = tuple(int(v) for v in r.inputs["shape"][k])
            x = r.inputs["x"][k].reshape(shape)
            sx = sd.asarray(x)
            got = F.fftn(sx) if method == "forward" else F.ifftn(sx, norm="forward")
            u = refdata.units(refdata.error(np.asarray(got).ravel(), m.hi[j], m.lo[j]),
                              refdata.bound_nd(shape, refdata.norm2(x)))  # fmt: skip
            assert u <= 1.0, (str(r.inputs["signal"][k]), method, u)
            worst = max(worst, (u, f"{method} {r.inputs['signal'][k]}"))
    assert worst[0] < 0.1, worst


def test_nd_points_within_contract():
    worst = {}
    for family, sid, x, s, hi, lo in refdata.nd_points():
        sx = sd.asarray(x)
        if family == "c2c":
            got, b = F.fftn(sx), refdata.bound_nd(s, refdata.norm2(x))
        elif family == "c2c-backward":
            got, b = F.ifftn(sx, norm="forward"), refdata.bound_nd(s, refdata.norm2(x))
        elif family == "real":
            got, b = F.rfftn(sx), refdata.bound_nd(s, refdata.norm2(x), real=True)
            # and the whole spectrum of fftn of real input: the same bound (decision FT45's form)
            whole = np.asarray(F.fftn(sx))
            assert np.array_equal(whole[..., : hi.shape[-1]], np.asarray(got), equal_nan=True), sid
        else:
            got = F.irfftn(sx, s=s, norm="forward")
            b = refdata.bound_nd(s, refdata.hermitian_norm_nd(x, s[-1]), real=True)
            hgot = F.hfftn(sd.asarray(np.conj(x)), s=s)
            assert refdata.units(refdata.error(np.asarray(hgot), hi, lo), b) <= 1.0, ("hfftn", sid)
        u = refdata.units(refdata.error(np.asarray(got), hi, lo), b)
        assert u <= 1.0, (family, sid, u)
        worst[family] = max(worst.get(family, (0.0, "")), (u, sid))
    for family, w in worst.items():
        assert w[0] < 0.2, (family, w)
