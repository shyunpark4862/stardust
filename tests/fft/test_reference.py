"""fft and ifft against the Arb references (docs/design/fft.md §4, §7.2, §7.3): every stored line of
tests/fft/ref/c2c.npz (forward by fft, backward by ifft(norm='forward'): both unnormalised) is within the
contract, on every kind of signal (uniform, normal, impulse, constant, sines, wide, huge, tiny) and every
length form (2^k, f 2^k, the other 11-smooth lengths, Bluestein); and the long lines the files leave out
(2^16, 65537, 3 10007; decision FT33) with Arb computed here."""

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
