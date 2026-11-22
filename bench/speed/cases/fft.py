"""sd.fft against numpy 2.5.4's numpy.fft, scipy 1.18.1's scipy.fft and R 4.6.2's stats::fft
(docs/design/fft.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the
sd.fft names it needs, and cases() keeps a spec once its names are public (FT1 brings fft, ifft and
the helpers, FT2 the real and Hermitian transforms, FT3 the n-dimensional ones, FT4 dct and dst,
FT5 FFTLog, FT6 cases of its new paths). The sizes are those of docs/design/fft.md §5, where milestone FT0 timed the
comparators and the prototype of the own path. numpy's pocketfft always runs on one thread; scipy
runs at its default workers=1 and, since sd runs on its pool by default, also at workers=10. R's
fft runs in the one R session.

Data: uniform values in [-1, 1) (complex for the complex transforms), numpy arrays made in the setup
from fixed seeds; sd gets them as sd.ndarray, outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Ref, RRef, count


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""


def _data(shape, complex_=True, seed=20261123):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, *shape, int(complex_)]))
        x = g.uniform(-1, 1, shape)
        if complex_:
            x = x + 1j * g.uniform(-1, 1, shape)
        d = {"x": x, "sx": sd.asarray(x)}
        if x.ndim == 1:
            d["r_re"], d["r_im"] = np.ascontiguousarray(x.real), np.ascontiguousarray(x.imag if complex_ else 0 * x)
        return d

    return setup


def _sd():
    import stardust as sd

    return sd.fft


def _sf():
    import scipy.fft as sf

    return sf


R_DATA = {"r_re": "r_re", "r_im": "r_im"}
R_SETUP = "z <- complex(real = r_re, imaginary = r_im)"


def _size(n: int) -> str:
    """The id notation (bench/README.md): 2^k, 10^k, or the plain count."""
    k = n.bit_length() - 1
    return f"2^{k}" if n == 1 << k else count(n)


def _complex_specs():
    out = []
    for n in (1024, 4096, 3072, 15360, 20480, 1 << 16, 1 << 20, 1000, 1009, 10007, 100003):
        big = n >= 1 << 16 or n in (10007, 100003)
        refs = [Ref("numpy", lambda d: lambda: np.fft.fft(d["x"])), Ref("scipy", lambda d: lambda: _sf().fft(d["x"]))]
        if big:
            refs.append(Ref("scipy workers=10", lambda d: lambda: _sf().fft(d["x"], workers=10), lib="scipy"))
        if n in (1024, 1 << 16, 1 << 20, 1000, 1009):
            refs.append(RRef("R", "fft(z)", data=R_DATA, setup=R_SETUP))
        note = {1009: "a prime: Bluestein", 10007: "a prime: Bluestein", 100003: "a prime: Bluestein"}.get(n, "")
        out.append(Spec("FT1", "complex", "fft", ("fft",), _size(n), _data((n,)),
                        lambda d: lambda: _sd().fft(d["sx"]), tuple(refs), quick=n in (1024, 1 << 16), note=note))  # fmt: skip
    for n in (1024, 1 << 16):
        refs = (Ref("numpy", lambda d: lambda: np.fft.ifft(d["x"])), Ref("scipy", lambda d: lambda: _sf().ifft(d["x"])))
        out.append(Spec("FT1", "complex", "ifft", ("ifft",), _size(n), _data((n,)),
                        lambda d: lambda: _sd().ifft(d["sx"]), refs))  # fmt: skip
    for axis in (-1, 0):
        refs = (Ref("numpy", lambda d, a=axis: lambda: np.fft.fft(d["x"], axis=a)),
                Ref("scipy", lambda d, a=axis: lambda: _sf().fft(d["x"], axis=a)),
                Ref("scipy workers=10", lambda d, a=axis: lambda: _sf().fft(d["x"], axis=a, workers=10), lib="scipy"))  # fmt: skip
        out.append(Spec("FT1", "batch", f"fft axis={axis}", ("fft",), "(1000, 1024)", _data((1000, 1024)),
                        lambda d, a=axis: lambda: _sd().fft(d["sx"], axis=a), refs, quick=axis == -1))  # fmt: skip
    return out


def _helper_specs():
    out = []
    refs = (Ref("numpy", lambda d: lambda: np.fft.fftfreq(1024, d=0.1)),
            Ref("scipy", lambda d: lambda: _sf().fftfreq(1024, d=0.1)))  # fmt: skip
    out.append(Spec("FT1", "helpers", "fftfreq", ("fftfreq",), "1024", dict, lambda d: lambda: _sd().fftfreq(1024, d=0.1),
                    refs, note="the call's cost"))  # fmt: skip
    refs = (
        Ref("numpy", lambda d: lambda: np.fft.fftshift(d["x"])),
        Ref("scipy", lambda d: lambda: _sf().fftshift(d["x"])),
    )
    out.append(Spec("FT1", "helpers", "fftshift", ("fftshift",), "(1000, 1024)", _data((1000, 1024)),
                    lambda d: lambda: _sd().fftshift(d["sx"]), refs))  # fmt: skip
    return out


def _real_specs():
    out = []
    for n in (1024, 1 << 16, 1 << 20):
        refs = (Ref("numpy", lambda d: lambda: np.fft.rfft(d["x"])), Ref("scipy", lambda d: lambda: _sf().rfft(d["x"])))
        out.append(Spec("FT2", "real", "rfft", ("rfft",), _size(n), _data((n,), complex_=False),
                        lambda d: lambda: _sd().rfft(d["sx"]), refs, quick=n == 1 << 16))  # fmt: skip
    for n in (1024, 1 << 16):

        def half(n=n):
            base = _data((n,), complex_=False)

            def setup():
                import stardust as sd

                d = base()
                d["X"] = np.fft.rfft(d["x"])
                d["sX"] = sd.asarray(d["X"])
                return d

            return setup

        refs = (
            Ref("numpy", lambda d: lambda: np.fft.irfft(d["X"])),
            Ref("scipy", lambda d: lambda: _sf().irfft(d["X"])),
        )
        out.append(
            Spec("FT2", "real", "irfft", ("irfft",), _size(n), half(), lambda d: lambda: _sd().irfft(d["sX"]), refs)
        )
    n = 1 << 16
    refs = (Ref("numpy", lambda d: lambda: np.fft.fft(d["x"])), Ref("scipy", lambda d: lambda: _sf().fft(d["x"])))
    out.append(Spec("FT2", "real", "fft of real input", ("fft", "rfft"), _size(n), _data((n,), complex_=False),
                    lambda d: lambda: _sd().fft(d["sx"]), refs, note="the real path (decision FT16)"))  # fmt: skip
    # odd and Bluestein real lengths (decision FT43): 1155 = 3 5 7 11 (the complex plan of n), 10007 (a
    # prime) and 20014 = 2 10007 (the real plan on a Bluestein half)
    for n, note in ((1155, "odd: the complex plan of n"), (10007, "a prime: Bluestein"),
                    (20014, "2 10007: a Bluestein half")):  # fmt: skip
        refs = (Ref("numpy", lambda d: lambda: np.fft.rfft(d["x"])), Ref("scipy", lambda d: lambda: _sf().rfft(d["x"])))
        out.append(Spec("FT2", "real", "rfft", ("rfft",), _size(n), _data((n,), complex_=False),
                        lambda d: lambda: _sd().rfft(d["sx"]), refs, note=note))  # fmt: skip
    n = 1 << 16
    refs = (Ref("numpy", lambda d: lambda: np.fft.ihfft(d["x"])), Ref("scipy", lambda d: lambda: _sf().ihfft(d["x"])))
    out.append(Spec("FT2", "real", "ihfft", ("ihfft",), _size(n), _data((n,), complex_=False),
                    lambda d: lambda: _sd().ihfft(d["sx"]), refs))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.hfft(d["x"])), Ref("scipy", lambda d: lambda: _sf().hfft(d["x"])))
    out.append(Spec("FT2", "real", "hfft", ("hfft",), _size(n // 2 + 1), _data((n // 2 + 1,)),
                    lambda d: lambda: _sd().hfft(d["sx"]), refs))  # fmt: skip
    for axis in (-1, 0):
        refs = (Ref("numpy", lambda d, a=axis: lambda: np.fft.rfft(d["x"], axis=a)),
                Ref("scipy", lambda d, a=axis: lambda: _sf().rfft(d["x"], axis=a)),
                Ref("scipy workers=10", lambda d, a=axis: lambda: _sf().rfft(d["x"], axis=a, workers=10), lib="scipy"))  # fmt: skip
        out.append(Spec("FT2", "batch", f"rfft axis={axis}", ("rfft",), "(1000, 1024)", _data((1000, 1024), complex_=False),
                        lambda d, a=axis: lambda: _sd().rfft(d["sx"], axis=a), refs))  # fmt: skip
    return out


def _nd_specs():
    out = []
    refs = (Ref("numpy", lambda d: lambda: np.fft.fft2(d["x"])), Ref("scipy", lambda d: lambda: _sf().fft2(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().fft2(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "fft2", ("fft2",), "(2048, 2048)", _data((2048, 2048)),
                    lambda d: lambda: _sd().fft2(d["sx"]), refs, quick=True))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.rfftn(d["x"])), Ref("scipy", lambda d: lambda: _sf().rfftn(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().rfftn(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "rfftn", ("rfftn",), "(256, 256, 256)", _data((256, 256, 256), complex_=False),
                    lambda d: lambda: _sd().rfftn(d["sx"]), refs))  # fmt: skip
    # milestone FT3's further cases: a small 3-d array (the call's cost and the split of small passes), the
    # inverse real transform (its complex passes first, decision FT25), the real 2-d transform and ifft2
    refs = (Ref("numpy", lambda d: lambda: np.fft.fftn(d["x"])), Ref("scipy", lambda d: lambda: _sf().fftn(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().fftn(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "fftn", ("fftn",), "(64, 64, 64)", _data((64, 64, 64)),
                    lambda d: lambda: _sd().fftn(d["sx"]), refs, quick=True))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.ifft2(d["x"])), Ref("scipy", lambda d: lambda: _sf().ifft2(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().ifft2(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "ifft2", ("ifft2",), "(1000, 1024)", _data((1000, 1024)),
                    lambda d: lambda: _sd().ifft2(d["sx"]), refs))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.rfft2(d["x"])), Ref("scipy", lambda d: lambda: _sf().rfft2(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().rfft2(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "rfft2", ("rfft2",), "(2048, 2048)", _data((2048, 2048), complex_=False),
                    lambda d: lambda: _sd().rfft2(d["sx"]), refs))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.irfftn(d["x"], axes=(0, 1, 2))),
            Ref("scipy", lambda d: lambda: _sf().irfftn(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().irfftn(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT3", "nd", "irfftn", ("irfftn",), "(128, 128, 65)", _data((128, 128, 65)),
                    lambda d: lambda: _sd().irfftn(d["sx"]), refs, note="the result is (128, 128, 128)"))  # fmt: skip
    return out


def _r2r_specs():
    out = []
    for name in ("dct", "dst"):
        for t in (1, 2, 3, 4):
            for n in (1 << 16,) + ((1 << 20,) if t == 2 else ()):
                refs = [Ref("scipy", lambda d, f=name, t=t: lambda: getattr(_sf(), f)(d["x"], type=t))]
                if n == 1 << 20:
                    refs.append(Ref("scipy workers=10", lambda d, f=name, t=t: lambda: getattr(_sf(), f)(
                        d["x"], type=t, workers=10), lib="scipy"))  # fmt: skip
                out.append(Spec("FT4", "r2r", f"{name} type={t}", (name,), _size(n), _data((n,), complex_=False),
                                lambda d, f=name, t=t: lambda: getattr(_sd(), f)(d["sx"], type=t), tuple(refs),
                                quick=(name, t, n) == ("dct", 2, 1 << 16)))  # fmt: skip
    # milestone FT4's further cases: the inverse (DCT-III), odd and Bluestein lengths (the complex plan of n,
    # the odd DCT-IV), batches along both axes and a 2-d transform
    refs = (Ref("scipy", lambda d: lambda: _sf().idct(d["x"])),)
    out.append(Spec("FT4", "r2r", "idct type=2", ("idct",), _size(1 << 16), _data((1 << 16,), complex_=False),
                    lambda d: lambda: _sd().idct(d["sx"]), refs, note="the DCT-III path"))  # fmt: skip
    for t, n, note in ((2, 1155, "odd: the complex plan of n"), (2, 10007, "a prime: Bluestein"),
                       (4, 1155, "odd: the odd DCT-IV")):  # fmt: skip
        refs = (Ref("scipy", lambda d, t=t: lambda: _sf().dct(d["x"], type=t)),)
        out.append(Spec("FT4", "r2r", f"dct type={t}", ("dct",), _size(n), _data((n,), complex_=False),
                        lambda d, t=t: lambda: _sd().dct(d["sx"], type=t), refs, note=note))  # fmt: skip
    for axis in (-1, 0):
        refs = (Ref("scipy", lambda d, a=axis: lambda: _sf().dct(d["x"], axis=a)),
                Ref("scipy workers=10", lambda d, a=axis: lambda: _sf().dct(d["x"], axis=a, workers=10), lib="scipy"))  # fmt: skip
        out.append(Spec("FT4", "batch", f"dct axis={axis}", ("dct",), "(1000, 1024)", _data((1000, 1024), complex_=False),
                        lambda d, a=axis: lambda: _sd().dct(d["sx"], axis=a), refs, quick=axis == -1))  # fmt: skip
    refs = (Ref("scipy", lambda d: lambda: _sf().dctn(d["x"])),
            Ref("scipy workers=10", lambda d: lambda: _sf().dctn(d["x"], workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT4", "nd", "dctn", ("dctn",), "(1024, 1024)", _data((1024, 1024), complex_=False),
                    lambda d: lambda: _sd().dctn(d["sx"]), refs))  # fmt: skip
    return out


def _fftlog_specs():
    """Milestone FT5: fht on logspace(-4, 4, n) samples of r^1.5 e^(-r^2 / 2) (dln = ln(1e8) / (n - 1), mu = 0.5,
    scipy's optimal offset), ifht with a bias, a batch along the last axis, a new plan each call (sd's plan cache
    cleared in the timed call: the coefficients' cost, which scipy pays every call) and fhtoffset."""
    out = []

    def data(shape, bias=0.0):
        def setup():
            import stardust as sd

            n = shape[-1]
            r = np.logspace(-4, 4, n)
            a = np.broadcast_to(r**1.5 * np.exp(-(r**2) / 2), shape).copy()
            if len(shape) > 1:
                a *= np.random.Generator(np.random.PCG64([20261123, *shape])).uniform(0.5, 1.5, shape)
            dln = float(np.log(r[1] / r[0]))
            off = float(_sf().fhtoffset(dln, 0.5, bias=bias))
            return {"a": a, "sa": sd.asarray(a), "dln": dln, "off": off, "bias": bias}

        return setup

    def sd_call(name, fresh=False):
        def make(d):
            import stardust as sd

            f = getattr(_sd(), name)
            if fresh:
                clear = sd._core.fft._fht_cache_clear
                return lambda: clear() or f(d["sa"], d["dln"], 0.5, offset=d["off"], bias=d["bias"])
            return lambda: f(d["sa"], d["dln"], 0.5, offset=d["off"], bias=d["bias"])

        return make

    def ref(name):
        def call(d):
            return lambda: getattr(_sf(), name)(d["a"], d["dln"], 0.5, offset=d["off"], bias=d["bias"])

        return (Ref("scipy", call),)

    for n in (1 << 12, 1 << 16, 1 << 20):
        out.append(Spec("FT5", "fftlog", "fht", ("fht",), _size(n), data((n,)), sd_call("fht"), ref("fht"),
                        quick=n == 1 << 12))  # fmt: skip
    out.append(Spec("FT5", "fftlog", "ifht bias=0.5", ("ifht",), _size(1 << 12), data((1 << 12,), 0.5), sd_call("ifht"),
                    ref("ifht"), note="the biases on both sides"))  # fmt: skip
    out.append(Spec("FT5", "fftlog", "fht", ("fht",), "(1000, 1024)", data((1000, 1024)), sd_call("fht"), ref("fht"),
                    note="1000 lines along the last axis"))  # fmt: skip
    for n in (1 << 12, 1 << 16):
        out.append(Spec("FT5", "fftlog", "fht new plan", ("fht",), _size(n), data((n,)), sd_call("fht", fresh=True),
                        ref("fht"), note="sd's plan cache cleared each call: the coefficients' cost"))  # fmt: skip
    refs = (Ref("scipy", lambda d: lambda: _sf().fhtoffset(0.01, 0.5, initial=0.3, bias=0.2)),)
    out.append(Spec("FT5", "fftlog", "fhtoffset", ("fhtoffset",), "1", dict,
                    lambda d: lambda: _sd().fhtoffset(0.01, 0.5, initial=0.3, bias=0.2), refs,
                    note="the call's cost"))  # fmt: skip
    return out


def _ft6_specs():
    """Milestone FT6: lengths with a radix of 13 to 31 (decision FT64; Bluestein before), a first-axis batch of the
    two-line kernel (decision FT67) and the DCT-I batch (inner length 1023 = 3 11 31)."""
    out = []
    for n, note in ((1001, "7 11 13: radix 13"), (32736, "3 11 31 32: radix 31, a four-step")):
        refs = (Ref("numpy", lambda d: lambda: np.fft.fft(d["x"])), Ref("scipy", lambda d: lambda: _sf().fft(d["x"])))
        out.append(Spec("FT6", "complex", "fft", ("fft",), _size(n), _data((n,)), lambda d: lambda: _sd().fft(d["sx"]),
                        refs, note=note))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.rfft(d["x"])), Ref("scipy", lambda d: lambda: _sf().rfft(d["x"])))
    out.append(Spec("FT6", "real", "rfft", ("rfft",), _size(4095), _data((4095,), complex_=False),
                    lambda d: lambda: _sd().rfft(d["sx"]), refs, note="3^2 5 7 13: radix 13"))  # fmt: skip
    refs = (Ref("numpy", lambda d: lambda: np.fft.fft(d["x"], axis=0)),
            Ref("scipy", lambda d: lambda: _sf().fft(d["x"], axis=0)),
            Ref("scipy workers=10", lambda d: lambda: _sf().fft(d["x"], axis=0, workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT6", "batch", "fft axis=0", ("fft",), "(1024, 1000)", _data((1024, 1000)),
                    lambda d: lambda: _sd().fft(d["sx"], axis=0), refs, note="the two-line kernel"))  # fmt: skip
    refs = (Ref("scipy", lambda d: lambda: _sf().dct(d["x"], type=1)),
            Ref("scipy workers=10", lambda d: lambda: _sf().dct(d["x"], type=1, workers=10), lib="scipy"))  # fmt: skip
    out.append(Spec("FT6", "batch", "dct type=1", ("dct",), "(1000, 1024)", _data((1000, 1024), complex_=False),
                    lambda d: lambda: _sd().dct(d["sx"], type=1), refs, note="inner length 1023 = 3 11 31"))  # fmt: skip
    return out


SPECS = _complex_specs() + _helper_specs() + _real_specs() + _nd_specs() + _r2r_specs() + _fftlog_specs() + _ft6_specs()


def _public(names) -> bool:
    import stardust as sd

    return all(n in sd.fft.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(area="fft", group=s.group, name=s.name, size=s.size, setup=s.setup, sd=s.sd, refs=s.refs, quick=s.quick,
             note=s.note)
        for s in SPECS
        if _public(s.needs)
    ]  # fmt: skip


TITLES = {
    "complex": "복소 변환 (`fft`, `ifft`)",
    "batch": "배치 (한 축의 여러 줄)",
    "helpers": "도우미 (`fftfreq`, `fftshift`)",
    "real": "실수와 에르미트 변환",
    "nd": "다차원 변환",
    "r2r": "DCT·DST",
    "fftlog": "FFTLog",
}
