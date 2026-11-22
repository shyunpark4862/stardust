"""The reference and comparison data of sd.fft (docs/design/fft.md §4, §7): every file belongs to the
test signals as they are (digests), the references pass checks of their own (exact rational sums,
Parseval's identity, the index reversal that relates the forward and backward DFT, the exact
zeros of constant and impulse signals), numpy 2.5.4 and scipy 1.18.1 are within the contract on
every reference (two other algorithms agreeing with Arb to a small part of the bound), and the
recorded files hold the behaviours the design quotes. The records are compared, not re-run."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.fft as sf

from . import refdata
from . import signals as S
from .conftest import LIVE

REFS = {f: refdata.ref(f) for f in refdata.FAMILIES}
NUMPY = refdata.data("numpy")
SCIPY = refdata.data("scipy")
R = refdata.data("r")
VDSP = refdata.data("vdsp")
PROTO = refdata.data("prototype")


def fr(v) -> Fraction:
    return Fraction(float(v))


def points(family, method):
    r = REFS[family]
    m = r.methods[method]
    for j, k in enumerate(m.index):
        yield str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j], k


# ---- the files belong to the signals ------------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == sorted(refdata.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0; acb.dft"), f
        assert r.meta["tool"] == "tools/fft_ref.py"
        assert r.meta["target_bits"] == 120
        assert r.meta["n_requested"] == r.points, f
        for m in r.methods.values():
            assert m.dropped.size == 0


def test_inputs_are_the_generators_signals():
    for f, r in REFS.items():
        for k in range(r.points):
            s = S.get(str(r.inputs["signal"][k]))
            x = r.inputs["x"][k]
            assert np.array_equal(x, np.asarray(s.x).ravel()), (f, s.id)
            assert x.dtype == s.x.dtype, (f, s.id)
            assert r.inputs["digest"][k] == S.digest(s.x), (f, s.id)
            assert int(r.inputs["n"][k]) == s.n
            if f == "nd":
                assert tuple(r.inputs["shape"][k]) == s.x.shape


def test_reference_shapes():
    for sid, _x, n, hi, lo, _ in points("c2c", "forward"):
        assert hi.shape == lo.shape == (n,) and hi.dtype == np.complex128, sid
    for sid, _x, n, hi, _lo, _ in points("real", "forward"):
        assert hi.shape == (n // 2 + 1,), sid
    for sid, x, n, hi, _lo, _ in points("hermitian", "c2r"):
        assert hi.shape == (n,) and hi.dtype == np.float64 and x.shape == (n // 2 + 1,), sid
    for method in refdata.R2R:
        for sid, _x, n, hi, _lo, _ in points("r2r", method):
            assert hi.shape == (n,) and hi.dtype == np.float64, (sid, method)
            assert method != "dct1" or n >= 2
    assert len(REFS["c2c"].methods["backward"].index) < len(REFS["c2c"].methods["forward"].index)


# ---- the references check themselves ------------------------------------------------------------


def dd_sum_sq(hi, lo):
    """sum |hi + lo|^2 as an exact rational."""
    out = Fraction(0)
    for h, l in zip(np.asarray(hi).ravel(), np.asarray(lo).ravel()):
        re, im = fr(h.real) + fr(l.real), fr(np.imag(h)) + fr(np.imag(l))
        out += re * re + im * im
    return out


def test_parseval():
    # ||F x||^2 = n ||x||^2 exactly; the stored double-doubles (each entry within 2^-120 ||y|| of
    # its value, plus the double-double's own rounding, 2^-106 of the entry) keep it to 2^-104
    checked = 0
    for sid, x, n, hi, lo, _ in points("c2c", "forward"):
        if n > 1024 or "tiny" in sid or "huge" in sid:
            continue
        want = n * sum((fr(v.real) ** 2 + fr(v.imag) ** 2 for v in x), Fraction(0))
        got = dd_sum_sq(hi, lo)
        assert abs(got - want) <= want * Fraction(1, 2**104), sid
        checked += 1
    assert checked > 150


def test_backward_is_the_forward_reversed():
    # sum_j x_j e^(+2 pi i jk/n) = Y_((n - k) mod n): the two methods were computed apart
    r = REFS["c2c"]
    fwd = dict(zip(r.methods["forward"].index.tolist(), zip(r.methods["forward"].hi, r.methods["forward"].lo)))
    for sid, _x, n, hi, lo, k in points("c2c", "backward"):
        fh, fl = fwd[k]
        rev = (-np.arange(n)) % n
        scale = refdata.norm2(fh)
        diff = (hi - fh[rev]) + (lo - fl[rev])
        assert refdata.norm2(diff) <= scale * 2.0**-104 + 2.0**-1074, sid


def test_exact_sums():
    # DC is the exact sum of the input, the Nyquist entry the alternating sum (real family); c2r's
    # first value is X_0 + 2 sum Re X_k (+ X_(n/2)) of the cleaned half spectrum
    for sid, x, n, hi, lo, _ in points("real", "forward"):
        dc = sum((fr(v) for v in x), Fraction(0))
        tol = Fraction(refdata.norm2(hi)) * Fraction(1, 2**104) + Fraction(1, 2**1074)
        assert abs(fr(hi[0].real) + fr(lo[0].real) - dc) <= tol, sid
        assert abs(fr(hi[0].imag) + fr(lo[0].imag)) <= tol, sid
        if n % 2 == 0:
            ny = sum((fr(v) * (-1) ** j for j, v in enumerate(x)), Fraction(0))
            assert abs(fr(hi[-1].real) + fr(lo[-1].real) - ny) <= tol, sid
    for sid, x, n, hi, lo, _ in points("hermitian", "c2r"):
        m = n // 2 + 1
        s = fr(x[0].real)
        for k in range(1, m):
            s += fr(x[k].real) * (1 if (n % 2 == 0 and k == m - 1) else 2)
        tol = Fraction(refdata.norm2(hi)) * Fraction(1, 2**104) + Fraction(1, 2**1074)
        assert abs(fr(hi[0]) + fr(lo[0]) - s) <= tol, sid
    for sid, x, _n, hi, lo, _ in points("r2r", "dct2"):
        tol = Fraction(refdata.norm2(hi)) * Fraction(1, 2**104) + Fraction(1, 2**1074)
        assert abs(fr(hi[0]) + fr(lo[0]) - 2 * sum((fr(v) for v in x), Fraction(0))) <= tol, sid


def test_structured_signals():
    # a constant signal's spectrum is n (1 + 1j) at k = 0 and 0 elsewhere; an impulse at p has
    # |Y_k| = 1 everywhere
    for sid, _x, n, hi, lo, _ in points("c2c", "forward"):
        kind = sid.split("/")[1]
        scale = refdata.norm2(hi)
        if kind == "constant":
            assert hi[0] == n * (1 + 1j) and abs(lo[0]) <= scale * 2.0**-118, sid
            assert np.all(np.abs(hi[1:]) <= scale * 2.0**-118), sid  # true zeros: the ball's midpoint
        if kind == "impulse":
            assert np.all(np.abs(np.abs(hi) - 1) <= 2.0**-52), sid


# ---- numpy and scipy are within the contract on every reference -------------------------------


def _units(lib_value, family, x, n, hi, lo, method=None):
    if family == "hermitian":
        b = refdata.bound("hermitian", n, refdata.hermitian_norm(x, n))
    elif family == "r2r":
        b = refdata.bound("r2r", n, refdata.norm2(x), method=method)
    elif family == "nd":
        b = refdata.bound("nd", x.shape, refdata.norm2(x))
    else:
        b = refdata.bound("c2c" if family == "c2c" else "real", n, refdata.norm2(x))
    return refdata.units(refdata.error(lib_value, hi, lo), b)


@LIVE
@pytest.mark.parametrize("lib", ["numpy", "scipy"])
def test_libraries_within_contract(lib):
    m = np.fft if lib == "numpy" else sf
    worst = {}
    for _sid, x, n, hi, lo, _ in points("c2c", "forward"):
        worst["c2c"] = max(worst.get("c2c", 0), _units(m.fft(x), "c2c", x, n, hi, lo))
    for _sid, x, n, hi, lo, _ in points("c2c", "backward"):
        worst["c2c"] = max(worst["c2c"], _units(m.ifft(x, norm="forward"), "c2c", x, n, hi, lo))
    for _sid, x, n, hi, lo, _ in points("real", "forward"):
        worst["real"] = max(worst.get("real", 0), _units(m.rfft(x), "real", x, n, hi, lo))
    for _sid, x, n, hi, lo, _ in points("hermitian", "c2r"):
        worst["hermitian"] = max(
            worst.get("hermitian", 0), _units(m.irfft(x, n, norm="forward"), "hermitian", x, n, hi, lo)
        )
    r = REFS["nd"]
    for method, f in (("forward", m.fftn), ("backward", lambda a: m.ifftn(a, norm="forward"))):
        for _sid, x, n, hi, lo, k in points("nd", method):
            shape = tuple(r.inputs["shape"][k])
            xs = x.reshape(shape)
            worst["nd"] = max(worst.get("nd", 0), _units(f(xs).ravel(), "nd", xs, n, hi, lo))
    if lib == "scipy":
        for method in refdata.R2R:
            f, t = (sf.dct if method.startswith("dct") else sf.dst), int(method[-1])
            for _sid, x, n, hi, lo, _ in points("r2r", method):
                worst["r2r"] = max(worst.get("r2r", 0), _units(f(x, type=t), "r2r", x, n, hi, lo, method))
    # far inside the proven bound: a regression of the references (or of the bound) shows here
    for fam, w in worst.items():
        assert w < 0.25, (lib, fam, w)


# ---- the recorded libraries ---------------------------------------------------------------------


def test_records_have_their_provenance():
    assert NUMPY["_provenance"]["numpy"] == "2.5.4"
    assert SCIPY["_provenance"]["scipy"] == "1.18.1"
    assert R["_provenance"]["R"].startswith("R version 4.6.2")
    for rec in (VDSP, PROTO):
        assert rec["_provenance"]["tool"] == "tools/fft_probe.py"


def test_recorded_departures():
    # the behaviours docs/design/fft.md §2 and docs/api/fft.md §1.6 quote
    assert SCIPY["irfft/n=0"]["value"]["shape"] == [1]  # scipy: a length-1 result
    assert NUMPY["irfft/n=0"]["error"] == "ValueError"
    assert "zero-sized" in SCIPY["child/sf.dct(np.array([2.0]), type=1)"]["msg"]
    assert SCIPY["child/sf.dst(np.array([2.0]), type=1)"]["stdout"] == "ndarray [4.0]"  # DST-I N = 1 is 2x
    assert SCIPY["helper/next_fast_len(0)"]["value"] == 0
    assert SCIPY["helper/next_fast_len(True)"]["value"] == 1
    for rec in (NUMPY, SCIPY):
        assert rec["fftfreq/0"]["error"] == rec["fftfreq/d=0"]["error"] == "ZeroDivisionError"
        assert rec["fftn/axes=() returns its input"]["value"] is True
        assert rec["fft/axis=1 of 1-d"]["error"] == "IndexError"
        assert rec["fftfreq/bits are k * (1 / (n d))"]["value"] is True
        assert rec["fftshift/bits are roll"]["value"] is True
        assert rec["fft/scaling by 2 is exact"]["value"] is True
        assert rec["fft/length-0 batch axis"]["value"]["shape"] == [0, 4]
        assert rec["fft/n=0"]["error"] == rec["fft/empty"]["error"] == "ValueError"
        assert rec["rfft/complex input"]["error"] == "TypeError"
    assert "value" in NUMPY["fftn/axes=(0,0)"] and SCIPY["fftn/axes=(0,0)"]["error"] == "ValueError"
    assert NUMPY["fftn/axes=(1,1) equals twice"]["value"] is True
    assert SCIPY["fft/n=True"]["value"]["shape"] == [1] and NUMPY["fft/n=True"]["error"] == "TypeError"
    assert NUMPY["fft/norm=None equals backward"]["value"] is True
    assert SCIPY["fft/plan=object()"]["error"] == "NotImplementedError"
    assert SCIPY["rfftn/axes=()"]["error"] == "ValueError"
    assert NUMPY["ihfft/DC imag sign"]["value"] == -1.0 and SCIPY["ihfft/DC imag sign"]["value"] == 1.0
    # numpy's rfft is not the start of its fft; scipy's is (decision FT16)
    assert len(NUMPY["rfft/differs from the start of fft at lengths"]["value"]) > 100
    assert SCIPY["rfft/differs from the start of fft at lengths"]["value"] == []
    nf = SCIPY["nonfinite/patterns differing from numpy"]
    assert nf["value"] > 0 and nf["of"] == 12474


def smooth(n, p):
    for q in (2, 3, 5, 7, 11):
        if q <= p:
            while n % q == 0:
                n //= q
    return n == 1


def test_fast_lengths_are_smooth_numbers():
    # FT18: next_fast_len is the next 11-smooth number (real=True: 5-smooth); R's nextn the 5-smooth
    nfl, nflr = SCIPY["next_fast_len/1..2000"]["value"], SCIPY["next_fast_len/real 1..2000"]["value"]
    pfl, pflr = SCIPY["prev_fast_len/1..2000"]["value"], SCIPY["prev_fast_len/real 1..2000"]["value"]
    for t in range(1, 2001):
        assert nfl[t - 1] == next(m for m in range(t, 4 * t + 8) if smooth(m, 11))
        assert nflr[t - 1] == next(m for m in range(t, 4 * t + 8) if smooth(m, 5))
        assert pfl[t - 1] == next(m for m in range(t, 0, -1) if smooth(m, 11))
        assert pflr[t - 1] == next(m for m in range(t, 0, -1) if smooth(m, 5))
    assert R["nextn/1..2000"]["value"] == nflr


def test_r_fft_within_contract():
    r = REFS["c2c"]
    by_signal = {str(r.inputs["signal"][k]): k for k in range(r.points)}
    for key, rec in R.items():
        if not key.startswith(("fft/", "inverse/")):
            continue
        what, sid = key.split("/", 1)
        k = by_signal[sid]
        x = r.inputs["x"][k]
        assert rec["digest"] == S.digest(x)
        method = "forward" if what == "fft" else "backward"
        m = r.methods[method]
        j = int(np.searchsorted(m.index, k))
        assert m.index[j] == k
        got = np.array([complex(a, b) for a, b in rec["values"]])
        u = _units(got, "c2c", x, len(x), m.hi[j], m.lo[j])
        # R's mixed radix (Singleton) is within the contract too; at the prime 1009 its O(n^2) sum
        # comes to about half the bound (recorded in docs/design/fft.md §4.4)
        assert u < (1.0 if sid.endswith("/1009") else 0.25), (key, u)


def test_vdsp_records():
    # docs/design/fft.md §3.1, §3.2: the length sets, determinism, no inner threads
    zop = VDSP["lengths/zop"]["odd_parts"]
    assert zop == {"1": {"min_power": 0, "count": 13}, "3": {"min_power": 3, "count": 8},
                   "5": {"min_power": 3, "count": 7}, "15": {"min_power": 3, "count": 6}}  # fmt: skip
    zrop = VDSP["lengths/zrop"]["odd_parts"]
    assert {f: v["min_power"] for f, v in zrop.items()} == {"1": 1, "3": 4, "5": 4, "15": 4}
    ilv = VDSP["lengths/ilv"]["odd_parts"]
    assert set(ilv) == {"1", "3", "5", "9", "15"} and "25" not in ilv
    assert not [r for r in VDSP["big"] if r["kind"] == "ilv" and r["ok"] and r["n"] > 4096]
    assert VDSP["lengths/dct"]["crashed"] and all(n % 4 == 2 for n in VDSP["lengths/dct"]["crashed"])
    for r in VDSP["det"]:
        assert r["repeat"] == r["misaligned"] == r["threads"] == r["in_place"] == 1, r
        assert r["recreated"] == r["shared_previous"] == 1, r
    conj = {r["n"]: r["inverse_is_conj"] for r in VDSP["det"] if r["kind"] == "zop"}
    assert conj[1024] == conj[65536] == 1 and conj[3072] == conj[15360] == 0
    assert all(abs(r["cpu_over_wall"] - 1.0) < 0.05 for r in VDSP["cpu"])


def test_prototype_records():
    # decision FT22 / FT30: every length class has a recorded path, from three runs
    keys = [k for k in PROTO if k.startswith("paths/")]
    assert sorted(keys) == sorted(f"paths/{kind}/{f}" for kind in ("c2c", "r2c") for f in (1, 3, 5, 15))
    for k in keys:
        rec = PROTO[k]
        assert rec["runs"] == 3 and rec["path"] in ("own", "vdsp"), k
        assert (rec["path"] == "own") == (rec["geomean_vdsp_over_own"] > 1.0), k
        assert all(v["rel_diff"] < 1e-14 for v in rec["lengths"]), k  # the two paths agree
    for k, rec in PROTO.items():
        if k.startswith("twiddle/"):
            assert rec["mu_over_u"] <= 1.5, k  # the twiddle bound the proofs use (FT31)
        if k.startswith("accuracy/"):
            for lib in ("numpy", "scipy", "own", "vdsp"):
                assert rec[lib]["units"] < 0.1, (k, lib)
            assert rec["digest"] == S.digest(S.get(rec["signal"]).x)
    assert math.isclose(refdata.KAPPA, 8.0) and refdata.C0 == 4.0 and refdata.C1 == 2.0 and refdata.C2 == 8.0
