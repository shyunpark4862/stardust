"""The n-dimensional transforms fft2, ifft2, fftn, ifftn, rfft2, irfft2, rfftn, irfftn, hfft2, ihfft2, hfftn
and ihfftn (docs/api/fft.md §2, §3; docs/design/fft.md §2.4, §3.7, §4.2; milestone FT3): every form of
`s` and `axes` (None, -1, cutting, padding, a subset, any order, negative, a list, `s` without `axes`)
against numpy 2.5.4 and scipy 1.18.1 live (within the n-dimensional contract, tests/fft/refdata.bound_nd),
the parts of a non-Hermitian input irfftn drops (numpy's and scipy's order), the result shapes and dtypes,
the bit relations (one axis is the one-dimensional function; the passes are the one-dimensional
transforms chained from the last axis to the first; rfftn is the front of fftn of real input; ihfftn and
hfftn are conjugates with the other direction's factor), the exact properties, the layouts, the errors and
the recorded behaviours of numpy and scipy."""

import math
import re

import numpy as np
import pytest
import scipy.fft as sf

import stardust as sd

from .._bits import bits_equal, value_equal
from . import refdata
from .conftest import LIVE

F = sd.fft
NORMS = ("backward", "ortho", "forward")
SWAP = {"backward": "forward", "ortho": "ortho", "forward": "backward"}
C2C = ("fft2", "ifft2", "fftn", "ifftn")
R2C = ("rfft2", "rfftn", "ihfft2", "ihfftn")
C2R = ("irfft2", "irfftn", "hfft2", "hfftn")


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def _c(shape, seed):
    g = np.random.default_rng(seed)
    return g.uniform(-1, 1, shape) + 1j * g.uniform(-1, 1, shape)


def _r(shape, seed):
    return np.random.default_rng(seed).uniform(-1, 1, shape)


def _resolve(x, s, axes, name):
    """(s, axes) as numbers: scipy's rules (decision FT14), irfftn's default last length 2 (m - 1)."""
    if axes is None:
        axes = list(range(x.ndim - len(s), x.ndim)) if s is not None else list(range(x.ndim))
    elif isinstance(axes, int):
        axes = [axes]
    axes = [a % x.ndim for a in axes]
    real_out = name in C2R
    if s is None:
        s = [x.shape[a] for a in axes]
        if real_out:
            s[-1] = 2 * (x.shape[axes[-1]] - 1)
    else:
        s = [x.shape[a] if v == -1 else v for v, a in zip(([s] if isinstance(s, int) else s), axes)]
    return s, axes


def _fit(x, s, axes):
    """x cut or zero-padded to s along axes (the input the transform sees)."""
    y = x
    for n, a in zip(s, axes):
        m = y.shape[a]
        if n < m:
            y = np.take(y, range(n), axis=a)
        elif n > m:
            pad = [(0, 0)] * y.ndim
            pad[a] = (0, n - m)
            y = np.pad(y, pad)
    return y


def _theirs(name, x, s, axes, norm):
    """numpy's result where numpy has the function, else scipy's (both with explicit axes)."""
    mod = sf if name.startswith(("hfft", "ihfft")) else np.fft
    return getattr(mod, name)(x, s=s, axes=axes, norm=norm)


def _bound(name, x, s, axes, norm):
    """The contract's bound for this call (tests/fft/refdata.bound_nd) on the input x."""
    real = name in R2C or name in C2R
    inverse = name in ("ifft2", "ifftn", "irfft2", "irfftn", "ihfft2", "ihfftn")
    count = math.prod(s)
    scale = {"backward": 1 / count if inverse else 1.0, "ortho": 1 / math.sqrt(count),
             "forward": 1.0 if inverse else 1 / count}[norm]  # fmt: skip
    if name in C2R:
        h = s[-1] // 2 + 1
        xf = np.moveaxis(_fit(x, list(s[:-1]) + [h], axes), axes, range(-len(axes), 0))
        if len(axes) > 1:
            xn = refdata.hermitian_norm_nd(xf, s[-1])
        else:  # a batch of lines: the norm of their H-norms
            xn = refdata.norm2([refdata.hermitian_norm(v, s[-1]) for v in xf.reshape(-1, h)])
    else:
        xn = refdata.norm2(_fit(x, s, axes))
    batch = x.size // math.prod(x.shape[a] for a in axes) if x.size else 1
    return refdata.bound_nd(s, xn, scale, real) + refdata.tau(count) * (math.sqrt(batch) - 1)


FORMS = [  # (shape, s, axes)
    ((6, 10), None, None),
    ((6, 10), (4, 13), None),
    ((6, 10), (-1, 7), (0, 1)),
    ((6, 10), None, (1, 0)),
    ((4, 6, 10), None, None),
    ((4, 6, 10), (5, 3), None),  # s without axes: the last len(s)
    ((4, 6, 10), (9, -1), [2, 0]),
    ((4, 6, 10), None, (-1, -3)),
    ((4, 6, 10), (8,), (1,)),
    ((3, 13, 8), None, (0, 1)),  # an axis of radix 13 (decision FT64)
    ((3, 37, 8), None, (0, 1)),  # a Bluestein axis (37)
    ((4, 41), (43, -1), None),  # Bluestein axes, one by padding (43) and one as given (41)
    ((5, 6, 9), (74,), (1,)),  # a Bluestein axis of 74 = 2 37 (a real axis's inner length 37)
    ((2, 37, 6), (-1, 74), (1, 2)),  # Bluestein on both axes (37 and 74)
    ((5, 7, 9), (17, 6), (2, 1)),
    ((2, 3, 4, 5), None, None),
]


@LIVE
@pytest.mark.parametrize("form", FORMS, ids=str)
@pytest.mark.parametrize("name", C2C + R2C + C2R)
def test_against_numpy_and_scipy(name, form):
    shape, s, axes = form
    if name.endswith("2") and (axes is None or len(axes) != 2) and not (axes is None and s is None):
        pytest.skip("the 2-d functions take two axes")
    f = getattr(F, name)
    real_in = name in R2C
    x = _r(shape, 7) if real_in else _c(shape, 7)
    kw = {} if (name.endswith("2") and axes is None) else {"axes": axes}
    for norm in NORMS:
        got = np.asarray(f(sd.asarray(x), s=s, norm=norm, **kw))
        rs, raxes = _resolve(x, s, kw.get("axes", (-2, -1) if name.endswith("2") else None), name)
        want = _theirs(name, x, rs, raxes, norm)
        assert got.shape == want.shape and got.dtype == want.dtype, (name, form, norm)
        b = _bound(name, x, rs, raxes, norm)
        err = refdata.norm2(got - want)
        assert err <= 2 * b, (name, form, norm, err / b)
        if name in ("rfftn", "irfftn", "hfftn", "ihfftn"):  # scipy too
            want2 = getattr(sf, name)(x, s=rs, axes=raxes, norm=norm)
            assert refdata.norm2(got - want2) <= 2 * b, (name, form, norm)


@LIVE
def test_real_and_integer_input_take_the_real_path():
    g = np.random.default_rng(5)
    ints = g.integers(-9, 9, (5, 6))
    for x in (ints, ints % 2 == 0, ints * 0.25):
        sx = sd.asarray(x)
        for name in ("fft2", "fftn", "ifft2", "ifftn"):
            got = np.asarray(getattr(F, name)(sx))
            want = getattr(np.fft, name)(x.astype(np.float64))
            assert got.dtype == np.complex128
            assert refdata.norm2(got - want) <= 2 * refdata.bound_nd((5, 6), refdata.norm2(x), 1.0), name
        # the first pass is the real one: rfftn is the front of fftn, bit for bit
        h = np.asarray(F.rfftn(sx))
        assert bits_equal(np.asarray(F.fftn(sx))[:, :4], h)
        for name in ("irfft2", "hfftn"):  # real input to the inverse real transforms: zero imaginary parts
            assert bits_equal(getattr(F, name)(sx), getattr(F, name)(sd.asarray(x.astype(np.complex128))))


@LIVE
def test_irfftn_drops_numpys_and_scipys_parts():
    # a non-Hermitian input: what is dropped depends on the order (the complex passes first, then irfft
    # drops the imaginary parts of each line's DC and Nyquist entries and reads n // 2 + 1 entries)
    for shape, s in (((6, 7), (6, 12)), ((6, 7), (6, 11)), ((4, 5, 6), (3, 4, 10)), ((5, 9), (8, 17))):
        x = _c(shape, 11)
        axes = list(range(len(shape)))
        hi, lo = refdata.arb_irfftn(_fit(x, list(s[:-1]) + [s[-1] // 2 + 1], axes), s[-1])
        b = _bound("irfftn", x, list(s), axes, "forward")
        for got in (np.asarray(F.irfftn(sd.asarray(x), s=s, norm="forward")),
                    np.fft.irfftn(x, s=s, axes=axes, norm="forward"), sf.irfftn(x, s=s, norm="forward")):  # fmt: skip
            assert refdata.units(refdata.error(got, hi, lo), b) <= 1.0, shape
        # the entries beyond n // 2 + 1 of the last axis are never read
        y = x.copy()
        y[..., s[-1] // 2 + 1 :] = np.nan
        assert bits_equal(F.irfftn(sd.asarray(y), s=s), F.irfftn(sd.asarray(x), s=s))


def test_one_axis_is_the_one_dimensional_function():
    x, r = _c((5, 12), 1), _r((5, 12), 2)
    pairs = [("fftn", "fft", x), ("ifftn", "ifft", x), ("fftn", "fft", r), ("ifftn", "ifft", r),
             ("rfftn", "rfft", r), ("ihfftn", "ihfft", r), ("irfftn", "irfft", x), ("hfftn", "hfft", x)]  # fmt: skip
    for nd, one, v in pairs:
        for axis in (0, 1):
            for norm in NORMS:
                for n in (None, 7, 16):
                    s = None if n is None else (n,)
                    got = getattr(F, nd)(sd.asarray(v), s=s, axes=(axis,), norm=norm)
                    want = getattr(F, one)(sd.asarray(v), n=n, axis=axis, norm=norm)
                    assert bits_equal(got, want), (nd, axis, norm, n)


def test_the_passes_are_the_one_dimensional_transforms_chained():
    # from the last of axes to the first (decision FT25); the factor is one of N, so 'backward' forward
    # and 'forward' inverse (factor 1) are the chains bit for bit
    x = _c((4, 6, 10), 3)
    sx = sd.asarray(x)
    for axes, s in (((0, 1, 2), None), ((2, 0), (7, 3)), ((1, 2), (-1, 12))):
        chain = sx
        lengths = s or [None] * len(axes)
        for a, n in zip(axes[::-1], lengths[::-1]):
            chain = F.fft(chain, axis=a, n=None if n == -1 else n)
        assert bits_equal(F.fftn(sx, s=s, axes=axes), chain), axes
        chain = sx
        for a, n in zip(axes[::-1], lengths[::-1]):
            chain = F.ifft(chain, axis=a, n=None if n == -1 else n, norm="forward")
        assert bits_equal(F.ifftn(sx, s=s, axes=axes, norm="forward"), chain), axes
    # rfftn: rfft along the last of axes, then fft; irfftn: ifft along the others, then irfft
    r = _r((4, 6, 10), 4)
    assert bits_equal(F.rfftn(sd.asarray(r)), F.fft(F.fft(F.rfft(sd.asarray(r)), axis=1), axis=0))
    want = F.irfft(F.ifft(F.ifft(sx, axis=1, norm="forward"), axis=0, norm="forward"), n=18, norm="forward")
    assert bits_equal(F.irfftn(sx, norm="forward"), want)
    want = F.irfft(F.ifft(sx, axis=0, n=5, norm="forward"), axis=2, n=9, norm="forward")
    assert bits_equal(F.irfftn(sx, s=(5, 9), axes=(0, 2), norm="forward"), want)


@pytest.mark.parametrize("norm", NORMS)
def test_conjugate_relations(norm):
    x, r = _c((6, 5, 7), 8), _r((6, 5, 7), 9)
    for shape_s in (None, (4, 8), (7, 9)):
        axes = None if shape_s is None else (0, 2)
        got = np.asarray(F.ihfftn(sd.asarray(r), s=shape_s, axes=axes, norm=norm))
        want = np.conj(np.asarray(F.rfftn(sd.asarray(r), s=shape_s, axes=axes, norm=SWAP[norm])))
        assert bits_equal(got, want)
        got = F.hfftn(sd.asarray(x), s=shape_s, axes=axes, norm=norm)
        want = F.irfftn(sd.asarray(np.conj(x)), s=shape_s, axes=axes, norm=SWAP[norm])
        assert bits_equal(got, want)
    assert bits_equal(F.hfft2(sd.asarray(x)), F.hfftn(sd.asarray(x), axes=(-2, -1)))
    assert bits_equal(F.ihfft2(sd.asarray(r)), F.ihfftn(sd.asarray(r), axes=(1, 2)))


def test_exact_properties():
    x = _c((6, 9), 12)
    # powers of 2 scale exactly; length 1 along every axis is a copy
    assert bits_equal(F.fft2(sd.asarray(2 * x)), 2 * np.asarray(F.fft2(sd.asarray(x))))
    assert bits_equal(F.fftn(sd.asarray(x[:1, :1])), x[:1, :1])
    assert bits_equal(F.ifftn(sd.asarray(x[:1, :1]), norm="ortho"), x[:1, :1])
    # irfftn's result does not depend on the parts beyond n // 2 + 1
    y = x.copy()
    y[:, 5:] = 1e300
    assert bits_equal(F.irfft2(sd.asarray(y), s=(6, 8)), F.irfft2(sd.asarray(x), s=(6, 8)))


def test_layouts_do_not_matter():
    x = _c((6, 8, 10), 13)
    for name in ("fftn", "ifftn", "irfftn", "hfftn"):
        f = getattr(F, name)
        ref = np.asarray(f(sd.asarray(np.ascontiguousarray(x))))
        for v in (np.asfortranarray(x), x.transpose(2, 0, 1).copy().transpose(1, 2, 0)):
            assert bits_equal(f(sd.asarray(v)), ref), name
        big = np.zeros((12, 8, 30), np.complex128)
        big[::2, :, ::3] = x
        assert bits_equal(f(sd.asarray(big[::2, :, ::3])), ref), name
        assert bits_equal(f(sd.asarray(np.ascontiguousarray(x[::-1])[::-1])), ref), name
    r = x.real.copy()
    ref = np.asarray(F.rfftn(sd.asarray(r)))
    for v in (np.asfortranarray(r), np.ascontiguousarray(r[:, ::-1])[:, ::-1]):
        assert bits_equal(F.rfftn(sd.asarray(v)), ref)
        assert bits_equal(np.asarray(F.fftn(sd.asarray(v)))[..., :6], ref)


def test_shapes_dtypes_and_empty_batches():
    x = sd.asarray(_c((3, 4, 5), 14))
    r = sd.asarray(_r((3, 4, 5), 14))
    assert F.fft2(x).shape == (3, 4, 5) and F.fft2(x).dtype == sd.complex128
    assert F.rfftn(r).shape == (3, 4, 3) and F.rfft2(r, s=(2, 8)).shape == (3, 2, 5)
    assert F.irfftn(x).shape == (3, 4, 8) and F.irfftn(x).dtype == sd.float64
    assert F.irfftn(x, s=(-1, -1)).shape == (3, 4, 5)  # -1: the input's length (numpy and scipy)
    assert F.hfft2(x, s=(2, 7)).shape == (3, 2, 7) and F.ihfft2(r).shape == (3, 4, 3)
    assert F.fftn(x, s=(6,)).shape == (3, 4, 6)
    assert F.fftn(x, s=7).shape == (3, 4, 7) and F.fftn(x, axes=0).shape == (3, 4, 5)  # an int for either
    for name in C2C + C2R + R2C:
        z = sd.zeros((0, 4, 6), dtype=sd.float64 if name in R2C else sd.complex128)
        got = getattr(F, name)(z, axes=(1, 2) if not name.endswith("2") else (-2, -1))
        want = getattr(sf, name)(np.zeros((0, 4, 6), np.float64 if name in R2C else np.complex128), axes=(1, 2))
        assert got.shape == want.shape, name
    # a transform axis of length 0 is padded when s gives its length
    assert value_equal(F.fft2(sd.zeros((3, 0), dtype=sd.complex128), s=(3, 2)), np.zeros((3, 2), np.complex128))
    assert value_equal(F.irfft2(sd.zeros((2, 0), dtype=sd.complex128), s=(2, 4)), np.zeros((2, 4)))


def test_errors():
    x = sd.asarray(_c((4, 6), 15))
    r = sd.asarray(_r((4, 6), 15))
    for name in C2C + R2C + C2R:
        f = getattr(F, name)
        v = r if name in R2C else x
        with raises(TypeError, f"{name}(): 'x' must be an sd.ndarray, not list"):
            f([[1.0]])
        with raises(TypeError, f"{name}(): 's' must be an int or a tuple or list of ints, not float"):
            f(v, s=(4, 2.0))
        with raises(TypeError, f"{name}(): 's' must be an int or a tuple or list of ints, not NoneType"):
            f(v, s=(None, 4))  # numpy deprecated None entries; scipy refuses them too
        with raises(TypeError, f"{name}(): 's' must be an int or a tuple or list of ints, not bool"):
            f(v, s=(True, 4))
        with raises(ValueError, f"{name}(): the entries of 's' must be at least 1 (or -1), not 0"):
            f(v, s=(0, 4))
        with raises(ValueError, f"{name}(): the entries of 's' must be at least 1 (or -1), not -2"):
            f(v, s=(4, -2))
        with raises(OverflowError, f"{name}(): 's' is out of range"):
            f(v, s=(2**64, 4))
        with raises(ValueError, f"{name}(): 's' and 'axes' must have the same length, not 3 and 2"):
            f(v, s=(4, 4, 4), axes=(0, 1))  # numpy and scipy: ValueError too
        with raises(ValueError, f"{name}(): repeated axis"):
            f(v, axes=(1, 1))  # [numpy와 다름]: numpy transforms the axis twice
        with raises(ValueError, f"{name}(): axis 2 is out of bounds for an array of dimension 2"):
            f(v, axes=(0, 2))
        with raises(ValueError, f"{name}(): axis -2 is out of bounds for an array of dimension 1"):
            f(sd.asarray(np.ones(4, np.complex128 if name not in R2C else np.float64)), axes=(-2, -1))
        with raises(TypeError, f"{name}(): 'axes' must be an int or a tuple or list of ints, not str"):
            f(v, axes="0")
        with raises(ValueError, f"{name}(): 'axes' must not be empty"):
            f(v, axes=())  # FT23
        with raises(TypeError, f"{name}(): 'norm' must be 'backward', 'ortho' or 'forward', not NoneType"):
            f(v, norm=None)
        with raises(TypeError, f"{name}(): takes 1 positional argument but 2 were given"):
            f(v, (4, 4))
        with raises(TypeError, f"{name}(): unexpected keyword argument 'workers'"):
            f(v, workers=2)
        with raises(ValueError, f"{name}(): cannot transform an axis of length 0"):
            f(sd.zeros((3, 0), dtype=v.dtype))
        if not name.endswith("2"):
            with raises(ValueError, f"{name}(): 's' has 3 entries for an array of dimension 2"):
                f(v, s=(2, 2, 2))
            with raises(ValueError, f"{name}(): 's' must not be empty"):
                f(v, s=())
    for name in R2C:
        with raises(TypeError, f"{name}(): 'x' must be real, not complex128"):
            getattr(F, name)(x)
    for name in C2R:
        with raises(ValueError, f"{name}(): the default s[-1] = 2 (m - 1) is 0 for an axis of length m = 1; give 's'"):
            getattr(F, name)(sd.ones((3, 1), dtype=sd.complex128))
        assert getattr(F, name)(sd.ones((3, 1), dtype=sd.complex128), s=(3, 1)).shape == (3, 1)


def test_recorded_behaviours_of_numpy_and_scipy():
    # tests/fft/data/{numpy,scipy}.json (tools/fft_probe.py)
    def arr(v):
        a = np.array(v["array"])
        if v["dtype"] == "complex128" and a.size:
            a = a[..., 0] + 1j * a[..., 1]
        return a.reshape(v["shape"])

    npr, spr = refdata.data("numpy"), refdata.data("scipy")
    # s without axes: the last len(s) axes (numpy warns; scipy does it silently)
    for rec in (npr, spr):
        assert list(F.fftn(sd.ones((2, 3)), s=(4,)).shape) == rec["fftn/s=(4,) without axes"]["value"]
        assert list(F.fftn(sd.ones((2, 3)), s=(-1, 4)).shape) == rec["fftn/s=(-1,4)"]["value"]
        assert list(F.fftn(sd.ones((2, 3, 4)), s=(4, 5)).shape) == rec["fftn/s=(4,5) without axes, 3-d"]["value"]
        assert list(F.irfftn(sd.ones((2, 3), dtype=sd.complex128)).shape) == rec["irfftn/default s"]["value"]
        assert rec["fftn/len(s) != len(axes)"]["error"] == "ValueError"
    # repeated axes: scipy refuses them (as here), numpy transforms twice
    assert spr["fftn/axes=(0,0)"]["error"] == "ValueError" and npr["fftn/axes=(1,1) equals twice"]["value"]
    # empty axes: both return the input (an error here, FT23); scipy refuses it for rfftn
    assert npr["fftn/axes=() returns its input"]["value"] and spr["rfftn/axes=()"]["error"] == "ValueError"
    # fft2 of a 1-d array: IndexError (numpy), ValueError (scipy and here)
    assert npr["fft2/1-d"]["error"] == "IndexError" and spr["fft2/1-d"]["error"] == "ValueError"
    assert value_equal(F.hfft2(sd.ones((2, 3), dtype=sd.complex128)), arr(spr["hfft2/ones (2,3)"]["value"]))
    got = np.asarray(F.ihfftn(sd.ones((2, 4))))
    assert np.array_equal(got, arr(spr["ihfftn/ones (2,4)"]["value"]))  # values; DC's imaginary zero is -0 here
    assert math.copysign(1, got[0, 0].imag) == -1  # conj(rfftn) (numpy's 1-d ihfft sign)
