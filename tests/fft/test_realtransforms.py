"""The discrete cosine and sine transforms (milestone FT4; docs/design/fft.md §2.6, §3.5, §4.2, §4.5, §7.3):
dct, idct, dst, idst and dctn, idctn, dstn, idstn.

- The contract against Arb: every stored line of tests/fft/ref/r2r.npz (140 lines, eight transforms, lengths
  1-64, 100, 127, 128, 255, 256, 1000, 1024, 4096 on uniform, normal, impulse, constant and wide values) at
  every norm, with and without orthogonalize (refdata.r2r_reference: the exact sqrt 2 and factor), through
  dct and dst and through idct and idst of the inverse type; and the lines of refdata.LONG_R2R the files
  leave out (a prime and 2 1009) with Arb computed here.
- scipy: every type, norm and orthogonalize on lengths 1-40 and a few long and Bluestein ones within twice
  the contract of each other; the n-dimensional ones over every form of s and axes; the boundaries (DST-I of
  one point, n, empty axes, complex, int and bool input) as scipy's, the differences marked.
- The bit relations (§4.5): DST-II, III, IV are the DCT of the same type with exact sign changes and
  reversals; idct of the inverse type with norm='forward' is dct (and idst, dst); a complex input's parts
  are the real transform of each part; dctn over one axis is dct, and with factor 1 dct along each axis from
  the last of `axes` to the first; a power-of-two scaling moves through exactly.
- Layouts and batches: a line's bits are those of the line alone (every axis of a 3-d array, C, F and
  strided, cut and padded, int64 and bool input, the empty batch).
- The wrong calls and their messages.
"""

import math

import numpy as np
import pytest
import scipy.fft as sf

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import signals as S

F = sd.fft
NORMS = ("backward", "ortho", "forward")
METHODS = refdata.R2R
INVERSE_TYPE = {1: 1, 2: 3, 3: 2, 4: 4}


def _fn(method, inverse=False):
    name = (
        ("dct" if method.startswith("dct") else "dst")
        if not inverse
        else ("idct" if method.startswith("dct") else "idst")
    )
    return getattr(F, name)


def _factor(method, n, norm, inverse=False):
    """The normalisation factor sd applies (docs/design/fft.md §2.5): 1, 1/N' or fl(1/fl(sqrt N'))."""
    nl = refdata.r2r_logical(method, n)
    if norm == "ortho":
        return 1.0 / math.sqrt(nl)
    if (norm == "backward") == (not inverse):
        return 1.0
    return 1.0 / nl


def _points(method):
    r = refdata.ref("r2r")
    m = r.methods[method]
    for j, k in enumerate(m.index):
        yield str(r.inputs["signal"][k]), r.inputs["x"][k], int(r.inputs["n"][k]), m.hi[j], m.lo[j]


def _units(got, method, x, n, hi, lo, s=1.0):
    b = refdata.bound("r2r", n, refdata.norm2(x), s=s, method=method)
    return refdata.units(refdata.error(np.asarray(got), hi, lo), b)


# ---------------------------------------------------------------- the contract (Arb)


@pytest.mark.parametrize("method", METHODS)
def test_every_stored_line_within_contract(method):
    t = int(method[-1])
    f, fi = _fn(method), _fn(method, inverse=True)
    worst = (0.0, "")
    count = 0
    for sid, x, n, hi, lo in _points(method):
        sx = sd.asarray(x)
        got = f(sx, type=t)
        u = _units(got, method, x, n, hi, lo)
        assert u <= 1.0, (method, sid, u)
        worst = max(worst, (u, sid))
        # the inverse of the inverse type at norm='forward' is the same transform with factor 1, bit for bit
        assert bits_equal(fi(sx, type=INVERSE_TYPE[t], norm="forward"), got), (method, sid)
        count += 1
    assert count >= (135 if method == "dct1" else 140)  # dct1 has no line of one point (five kinds)
    assert worst[0] < 0.1, worst


@pytest.mark.parametrize("method", METHODS)
def test_every_norm_and_orthogonalize_within_contract(method):
    t = int(method[-1])
    f, fi = _fn(method), _fn(method, inverse=True)
    worst = (0.0, "")
    for sid, x, n, hi, lo in _points(method):
        if n > 1024:
            continue
        sx = sd.asarray(x)
        for norm in NORMS:
            for ortho in (False, True):
                s = _factor(method, n, norm)
                rhi, rlo = refdata.r2r_reference(method, x, hi, lo, s, ortho)
                got = f(sx, type=t, norm=norm, orthogonalize=ortho)
                u = _units(got, method, x, n, rhi, rlo, s)
                assert u <= 1.0, (method, sid, norm, ortho, u)
                worst = max(worst, (u, f"{sid} {norm} {ortho}"))
                # the inverse transform computing the same type with the inverse direction's factor
                si = _factor(method, n, norm, inverse=True)
                rhi, rlo = refdata.r2r_reference(method, x, hi, lo, si, ortho)
                got = fi(sx, type=INVERSE_TYPE[t], norm=norm, orthogonalize=ortho)
                assert _units(got, method, x, n, rhi, rlo, si) <= 1.0, (method, sid, norm, ortho, "inverse")
    assert worst[0] < 0.1, worst


@pytest.mark.parametrize("sid", refdata.LONG_R2R[:2])
def test_long_lines_within_contract(sid):
    s = S.get(sid)
    for method in METHODS:
        hi, lo = refdata.arb_r2r(s.x, method)
        got = _fn(method)(sd.asarray(s.x), type=int(method[-1]))
        u = _units(got, method, s.x, s.n, hi, lo)
        assert u < 0.1, (sid, method, u)


def test_extreme_values_within_contract():
    # the kinds the stored family lacks: subnormal values (the floor tau), the edge of the overflow domain,
    # sines on and between the bins; Arb computed here
    worst = (0.0, "")
    for kind in ("tiny", "huge", "sine", "sine_half"):
        for n in (2, 5, 16, 26, 127):
            s = S.get(S.sid("r", kind, n))
            for method in METHODS:
                hi, lo = refdata.arb_r2r(s.x, method)
                got = _fn(method)(sd.asarray(s.x), type=int(method[-1]))
                u = _units(got, method, s.x, n, hi, lo)
                assert u <= 1.0, (s.id, method, u)
                worst = max(worst, (u, f"{method} {s.id}"))
    assert worst[0] < 0.2, worst


def test_bluestein_inner_lengths_take_their_bound():
    # decision FT53: an inner length with a prime factor above 31 (11 before decision FT64) widens the bound by
    # its rho and inner length
    for method, n in (("dct2", 127), ("dct2", 74), ("dct1", 38), ("dst1", 36), ("dct4", 37), ("dct4", 74)):
        h = refdata.r2r_inner(method, n)
        assert not refdata.is_mixed(h)
        plain = (refdata.KAPPA * refdata.levels(2 * n + 2) + refdata.C2) * refdata.U * refdata.norm_r2r(method, n)
        assert refdata.bound("r2r", n, 1.0, method=method) > plain
    assert refdata.r2r_inner("dct2", 1000) == 500 and refdata.r2r_inner("dct4", 1009) == 1009


# ---------------------------------------------------------------- scipy


LENGTHS = list(range(1, 41)) + [64, 100, 127, 255, 1009, 1024, 2018, 4097]


@pytest.mark.parametrize("name", ["dct", "idct", "dst", "idst"])
def test_scipy_within_contract(name):
    g = np.random.default_rng(41)
    f, ref = getattr(F, name), getattr(sf, name)
    inverse = name.startswith("i")
    for n in LENGTHS:
        x = g.uniform(-1, 1, n)
        for t in (1, 2, 3, 4):
            if t == 1 and name.endswith("ct") and n < 2:
                continue
            method = ("dct" if "dct" in name else "dst") + str(INVERSE_TYPE[t] if inverse else t)
            for norm in NORMS:
                for ortho in (None, False, True):
                    got = np.asarray(f(sd.asarray(x), type=t, norm=norm, orthogonalize=ortho))
                    want = ref(x, type=t, norm=norm, orthogonalize=ortho)
                    assert got.shape == want.shape and got.dtype == np.float64
                    s = _factor(method, n, norm, inverse)
                    b = refdata.bound("r2r", n, refdata.norm2(x), s=s, method=method)
                    assert refdata.norm2(got - want) <= 2 * b, (name, t, n, norm, ortho)


def test_scipy_n_dimensional():
    g = np.random.default_rng(42)
    x = g.uniform(-1, 1, (5, 6, 7))
    forms = ({}, {"axes": (0, 2)}, {"axes": [2, 0]}, {"s": (4, 9)}, {"s": (3, -1, 8), "axes": (2, 0, 1)},
             {"axes": -1}, {"s": 6}, {"s": [5, 6, 7]})  # fmt: skip
    for name in ("dctn", "idctn", "dstn", "idstn"):
        for t in (1, 2, 3, 4):
            for norm in NORMS:
                for kw in forms:
                    got = np.asarray(getattr(F, name)(sd.asarray(x), type=t, norm=norm, **kw))
                    want = getattr(sf, name)(x, type=t, norm=norm, **kw)
                    assert got.shape == want.shape, (name, kw)
                    assert np.abs(got - want).max() <= 1e-13 * np.abs(want).max(), (name, t, norm, kw)
    for o in (True, False):
        got = np.asarray(F.dctn(sd.asarray(x), type=3, norm="ortho", orthogonalize=o, axes=(1, 2)))
        assert np.allclose(got, sf.dctn(x, type=3, norm="ortho", orthogonalize=o, axes=(1, 2)), rtol=0, atol=1e-13)


def test_scipy_boundaries():
    # DST-I of one point is 2x, DCT-II of one point 2x, DCT-III and DCT-IV x and sqrt 2 x: as scipy
    for name, t in (("dst", 1), ("dct", 2), ("dct", 3), ("dct", 4), ("dst", 2), ("dst", 3), ("dst", 4)):
        x = np.array([2.0])
        got = np.asarray(getattr(F, name)(sd.asarray(x), type=t))
        assert np.allclose(got, getattr(sf, name)(x, type=t), rtol=1e-15), (name, t)
    # DCT-I of two points
    assert np.array_equal(np.asarray(F.dct(sd.asarray(np.array([1.0, 2.0])), type=1)), [3.0, -1.0])
    # n cuts and pads; int64 and bool input are float64; the result is a new C-order array
    x = np.arange(6.0)
    for n in (2, 6, 9):
        assert np.allclose(np.asarray(F.dct(sd.asarray(x), n=n)), sf.dct(x, n=n), rtol=0, atol=1e-13)
    gi = F.dct(sd.asarray(np.arange(5)), type=3)
    assert gi.dtype == sd.float64 and np.allclose(np.asarray(gi), sf.dct(np.arange(5), type=3))
    gb = F.dst(sd.asarray(np.array([True, False, True])), type=4)
    assert gb.dtype == sd.float64 and np.allclose(np.asarray(gb), sf.dst(np.array([True, False, True]), type=4))
    # complex input: complex128, the parts transformed separately (as scipy)
    z = np.array([1 + 2j, 3 - 1j, -0.5j, 2.0])
    gz = F.idct(sd.asarray(z), type=2, norm="ortho")
    assert gz.dtype == sd.complex128 and np.allclose(np.asarray(gz), sf.idct(z, type=2, norm="ortho"))
    # the empty batch: an empty result; an empty transform axis: ValueError (scipy: "invalid number of data
    # points (0) specified")
    assert np.asarray(F.dct(sd.asarray(np.zeros((0, 4))))).shape == (0, 4)
    assert np.asarray(F.dstn(sd.asarray(np.zeros((3, 0))), axes=0)).shape == (3, 0)
    for f in (F.dct, F.dctn):
        with pytest.raises(ValueError, match="cannot transform an axis of length 0"):
            f(sd.asarray(np.zeros(0)))
        with pytest.raises(ValueError):
            sf.dct(np.zeros(0))


def test_dct1_of_one_point_is_an_error():
    # [scipy와 다름] scipy 1.18.1 leaks DUCC0's assertion as RuntimeError "no zero-sized FFTs" (decision FT20)
    assert "zero-sized" in refdata.data("scipy")["child/sf.dct(np.array([2.0]), type=1)"]["msg"]
    x = sd.asarray(np.array([2.0]))
    for f, name in ((F.dct, "dct"), (F.idct, "idct")):
        with pytest.raises(ValueError, match=rf"^{name}\(\): type=1 needs at least 2 points, not 1$"):
            f(x, type=1)
    with pytest.raises(ValueError, match=r"^dct\(\): type=1 needs at least 2 points, not 1$"):
        F.dct(sd.asarray(np.ones(5)), type=1, n=1)
    with pytest.raises(ValueError, match=r"^idctn\(\): type=1 needs at least 2 points, not 1$"):
        F.idctn(sd.asarray(np.ones((3, 1))), type=1)
    assert np.asarray(F.dstn(sd.asarray(np.ones((3, 1))), type=1)).shape == (3, 1)


# ---------------------------------------------------------------- bit relations


def _neg_odd(v):
    v = np.array(v, copy=True)
    v[1::2] = -v[1::2]
    return v


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 13, 16, 17, 26, 100, 127, 1000, 1009])
def test_dst_is_the_dct_with_signs_and_reversal(n):
    x = np.random.default_rng(n).uniform(-1, 1, n)
    x[0] = 0.0  # a signed zero through the sign changes
    for norm in NORMS:
        for o in (None, False, True):
            kw = {"norm": norm, "orthogonalize": o}
            # DST-II(x)_k = DCT-II((-1)^n x_n)_(N-1-k)
            want = np.asarray(F.dct(sd.asarray(_neg_odd(x)), type=2, **kw))[::-1]
            assert bits_equal(F.dst(sd.asarray(x), type=2, **kw), want), (n, norm, o, 2)
            # DST-III and DST-IV: (-1)^k DCT(x reversed)_k
            for t in (3, 4):
                want = _neg_odd(np.asarray(F.dct(sd.asarray(x[::-1].copy()), type=t, **kw)))
                assert bits_equal(F.dst(sd.asarray(x), type=t, **kw), want), (n, norm, o, t)
            # and the inverses (idst type 2 computes DST-III, type 3 DST-II)
            want = _neg_odd(np.asarray(F.idct(sd.asarray(x[::-1].copy()), type=2, **kw)))
            assert bits_equal(F.idst(sd.asarray(x), type=2, **kw), want), (n, norm, o, "i2")


def test_complex_input_is_its_parts():
    g = np.random.default_rng(43)
    z = g.uniform(-1, 1, (6, 33)) + 1j * g.uniform(-1, 1, (6, 33))
    for name in ("dct", "idct", "dst", "idst"):
        for t in (1, 2, 3, 4):
            got = np.asarray(getattr(F, name)(sd.asarray(z), type=t, axis=0, norm="ortho"))
            re = np.asarray(getattr(F, name)(sd.asarray(np.ascontiguousarray(z.real)), type=t, axis=0, norm="ortho"))
            im = np.asarray(getattr(F, name)(sd.asarray(np.ascontiguousarray(z.imag)), type=t, axis=0, norm="ortho"))
            assert bits_equal(got.real.copy(), re) and bits_equal(got.imag.copy(), im), (name, t)
        got = np.asarray(getattr(F, name + "n")(sd.asarray(z), type=2))
        re = np.asarray(getattr(F, name + "n")(sd.asarray(np.ascontiguousarray(z.real)), type=2))
        assert bits_equal(got.real.copy(), re), name


def test_n_dimensional_is_one_dimensional_passes():
    g = np.random.default_rng(44)
    x = g.uniform(-1, 1, (7, 9, 12))
    sx = sd.asarray(x)
    for t in (1, 2, 3, 4):
        for nd, one in ((F.dctn, F.dct), (F.idstn, F.idst)):
            for norm in NORMS:  # one axis: the 1-d function
                assert bits_equal(nd(sx, type=t, axes=1, norm=norm), one(sx, type=t, axis=1, norm=norm)), (t, nd)
        # factor 1 (dctn 'backward', idctn 'forward'): dct along the axes from the last of `axes` to the first
        want = F.dct(F.dct(F.dct(sx, type=t, axis=2), type=t, axis=0), type=t, axis=1)
        assert bits_equal(F.dctn(sx, type=t, axes=(1, 0, 2)), want), t
        want = F.idst(F.idst(sx, type=t, axis=2, norm="forward"), type=t, axis=0, norm="forward")
        assert bits_equal(F.idstn(sx, type=t, axes=(0, 2), norm="forward"), want), t


def test_power_of_two_scaling_is_exact():
    x = np.random.default_rng(45).uniform(-1, 1, 77)
    for name in ("dct", "dst"):
        for t in (1, 2, 3, 4):
            a = np.asarray(getattr(F, name)(sd.asarray(x), type=t))
            b = np.asarray(getattr(F, name)(sd.asarray(4.0 * x), type=t))
            assert bits_equal(4.0 * a, b), (name, t)


# ---------------------------------------------------------------- layouts and batches


def test_a_line_is_the_line_alone():
    g = np.random.default_rng(46)
    t3 = g.uniform(-1, 1, (5, 26, 9))
    layouts = (t3, np.asfortranarray(t3), t3[::-1, :, ::2], np.ascontiguousarray(np.swapaxes(t3, 0, 2)).swapaxes(0, 2))
    for name in ("dct", "idst"):
        f = getattr(F, name)
        for t in (1, 2, 3, 4):
            for x in layouts:
                for axis in (0, 1, 2):
                    for n in (None, 7):
                        got = np.asarray(f(sd.asarray(x), type=t, axis=axis, n=n, norm="ortho"))
                        lines = np.moveaxis(x, axis, -1).reshape(-1, x.shape[axis])
                        outs = np.moveaxis(got, axis, -1).reshape(-1, got.shape[axis])
                        for k in range(0, lines.shape[0], 5):
                            one = f(sd.asarray(np.ascontiguousarray(lines[k])), type=t, n=n, norm="ortho")
                            assert bits_equal(outs[k], one), (name, t, axis, n, k)


def test_dtypes_take_the_float64_path():
    x = np.array([3, -1, 4, 1, -5, 9], dtype=np.int64)
    for t in (1, 2, 3, 4):
        assert bits_equal(F.dst(sd.asarray(x), type=t), F.dst(sd.asarray(x.astype(np.float64)), type=t)), t
        b = x > 0
        assert bits_equal(F.dctn(sd.asarray(b), type=t), F.dctn(sd.asarray(b.astype(np.float64)), type=t)), t


def test_nonfinite_values_pass_through():
    # IEEE arithmetic as it falls (decision FT17): no error, and the other lines are untouched
    x = np.random.default_rng(47).uniform(-1, 1, (3, 16))
    x[1, 5] = math.nan
    x[2, 0] = math.inf
    for t in (1, 2, 3, 4):
        got = np.asarray(F.dct(sd.asarray(x), type=t))
        assert np.isnan(got[1]).any() and not np.isfinite(got[2]).all()
        assert bits_equal(got[0], F.dct(sd.asarray(x[0].copy()), type=t))


# ---------------------------------------------------------------- the wrong calls


def test_wrong_calls():
    x = sd.asarray(np.ones(4))
    cases = [
        (lambda: F.dct(x, type=5), ValueError, r"^dct\(\): 'type' must be 1, 2, 3 or 4, not 5$"),
        (lambda: F.idst(x, type=0), ValueError, r"^idst\(\): 'type' must be 1, 2, 3 or 4, not 0$"),
        # [scipy와 다름] scipy's binding dumps its signature for type=2.0
        (lambda: F.dct(x, type=2.0), TypeError, r"^dct\(\): 'type' must be an int, not float$"),
        (lambda: F.dct(x, type=True), TypeError, r"^dct\(\): 'type' must be an int, not bool$"),
        (lambda: F.dct(x, type=None), TypeError, r"^dct\(\): 'type' must be an int, not NoneType$"),
        (lambda: F.dct(x, type=2**70), OverflowError, r"^dct\(\): 'type' is out of range$"),
        (
            lambda: F.dst(x, orthogonalize=1),
            TypeError,
            r"^dst\(\): 'orthogonalize' must be True, False or None, not int$",
        ),
        (lambda: F.dct(x, n=0), ValueError, r"^dct\(\): 'n' must be at least 1, not 0$"),
        (lambda: F.dct(x, axis=1), ValueError, r"^dct\(\): axis 1 is out of bounds for an array of dimension 1$"),
        (
            lambda: F.dct(x, norm=None),
            TypeError,
            r"^dct\(\): 'norm' must be 'backward', 'ortho' or 'forward', not NoneType$",
        ),
        (lambda: F.dct([1.0, 2.0]), TypeError, r"^dct\(\): 'x' must be an sd.ndarray, not list$"),
        (lambda: F.dct(np.ones(3)), TypeError, r"^dct\(\): 'x' must be an sd.ndarray, not numpy.ndarray$"),
        (lambda: F.dctn(sd.asarray(np.ones((2, 2))), axes=(0, 0)), ValueError, r"^dctn\(\): repeated axis$"),
        (lambda: F.dstn(sd.asarray(np.ones((2, 2))), axes=()), ValueError, r"^dstn\(\): 'axes' must not be empty$"),
        (
            lambda: F.idctn(sd.asarray(np.ones((2, 2))), s=(3, 3, 3)),
            ValueError,
            r"^idctn\(\): 's' has 3 entries for an array of dimension 2$",
        ),
        (
            lambda: F.idstn(sd.asarray(np.ones((2, 2))), s=(0,)),
            ValueError,
            r"^idstn\(\): the entries of 's' must be at least 1 \(or -1\), not 0$",
        ),
        (lambda: F.dct(x, 2), TypeError, None),  # options are keyword-only (FT6; [scipy와 다름])
    ]
    for call, exc, msg in cases:
        with pytest.raises(exc, match=msg):
            call()
    # the order of the checks is the signature's: x, type, n, axis, norm, orthogonalize
    with pytest.raises(TypeError, match="'x'"):
        F.dct([1.0], type=9)
    with pytest.raises(ValueError, match="'type'"):
        F.dct(x, type=9, n=0)
    with pytest.raises(ValueError, match="'n'"):
        F.dct(x, n=0, norm="bad")
    with pytest.raises(ValueError, match="'norm'"):
        F.dct(x, norm="bad", orthogonalize=1)
    with pytest.raises(ValueError, match="'type'"):
        F.dctn(x, type=9, s=(0,))
