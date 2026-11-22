"""sd.fft against numpy 2.5.4's numpy.fft, scipy 1.18.1's scipy.fft and R 4.6.2's stats::fft
(docs/design/fft.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the
sd.fft names it needs, and cases() keeps a spec once its names are public (FT1 brings fft, ifft and
the helpers, FT2 the real and Hermitian transforms, FT3 the n-dimensional ones, FT4 dct and dst,
FT5 FFTLog). The sizes are those of docs/design/fft.md §5, where milestone FT0 timed the
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
    return out


def _r2r_specs():
    out = []
    for name in ("dct", "dst"):
        for t in (1, 2, 3, 4):
            for n in (1 << 16,) + ((1 << 20,) if t == 2 else ()):
                refs = (Ref("scipy", lambda d, f=name, t=t: lambda: getattr(_sf(), f)(d["x"], type=t)),)
                out.append(Spec("FT4", "r2r", f"{name} type={t}", (name,), _size(n), _data((n,), complex_=False),
                                lambda d, f=name, t=t: lambda: getattr(_sd(), f)(d["sx"], type=t), refs,
                                quick=(name, t, n) == ("dct", 2, 1 << 16)))  # fmt: skip
    return out


def _fftlog_specs():
    n = 4096

    def setup():
        import stardust as sd

        r = np.logspace(-4, 4, n)
        a = r**1.5 * np.exp(-(r**2) / 2)
        return {"a": a, "sa": sd.asarray(a), "dln": float(np.log(r[1] / r[0]))}

    refs = (Ref("scipy", lambda d: lambda: _sf().fht(d["a"], d["dln"], 0.5)),)
    return [
        Spec("FT5", "fftlog", "fht", ("fht",), "4096", setup, lambda d: lambda: _sd().fht(d["sa"], d["dln"], 0.5), refs)
    ]


SPECS = _complex_specs() + _helper_specs() + _real_specs() + _nd_specs() + _r2r_specs() + _fftlog_specs()


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
