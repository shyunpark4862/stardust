"""sd.integrate against scipy.integrate 1.18.1 (docs/design/integrate.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names the sd.integrate
names it needs and its milestone, and cases() keeps a case only once the milestone is in DONE and
the names are public (C3 brought the sample rules and fixed_quad; C4 brings quad and the multiple
integrals, C5 quad's weights, C6 tanhsinh and nsum, C7 quad_vec and cubature). The comparator is scipy at its defaults (one thread); the integrands are
the same Python functions on both sides (quad's family: one float a call; tanhsinh and nsum:
numpy arrays for scipy, sd arrays for sd), so the times compare the work between the calls.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable

import numpy as np

from ..registry import Case, Check, Ref

DONE: tuple[str, ...] = ("C3", "C4", "C5", "C6", "C7")  # the milestones whose cases run
BATCH_SIZES = ((10, "10"), (1000, "10^3"), (10**5, "10^5"))
SAMPLE_SIZES = ((10**3, "10^3"), (10**6, "10^6"))


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]  # sd.integrate names the sd side calls
    sd: Callable  # data -> zero-argument callable
    scipy: Callable  # data -> zero-argument callable
    setup: Callable = dict
    size: str = ""
    n: int | None = None
    quick: bool = False
    note: str = ""


def _quiet(f):
    def g():
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            return f()

    return g


def _gauss(x):
    return math.exp(-x * x)


def _peak5(x):
    return 4.0**-5 / ((x - math.pi / 4) * (x - math.pi / 4) + 16.0**-5)


def _log0(x):
    return math.log(x) if x > 0 else 0.0


def _quad_specs():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    return [
        Spec("C4", "quad", "quad exp(-x^2) on (0, 1)", ("quad",), lambda d: lambda: I.quad(_gauss, 0.0, 1.0),
             lambda d: lambda: si.quad(_gauss, 0.0, 1.0), quick=True, note="21 calls"),
        Spec("C4", "quad", "quad exp(-x^2) on the real line", ("quad",), lambda d: lambda: I.quad(_gauss, -math.inf, math.inf),
             lambda d: lambda: si.quad(_gauss, -math.inf, math.inf), note="270 calls"),
        Spec("C4", "quad", "quad peak a=5", ("quad",), lambda d: lambda: I.quad(_peak5, 0.0, 1.0),
             lambda d: lambda: si.quad(_peak5, 0.0, 1.0), note="525 calls, 13 intervals"),
        Spec("C4", "quad", "dblquad exp(-x^2-y^2) unit square", ("dblquad",),
             lambda d: lambda: I.dblquad(lambda y, x: math.exp(-(x * x + y * y)), 0.0, 1.0, 0.0, 1.0),
             lambda d: lambda: si.dblquad(lambda y, x: math.exp(-(x * x + y * y)), 0.0, 1.0, 0.0, 1.0),
             note="441 calls"),
        Spec("C5", "quad", "quad weight=cos half line", ("quad",),
             lambda d: lambda: I.quad(lambda x: math.exp(-x), 0.0, math.inf, weight="cos", wvar=3.0),
             lambda d: lambda: si.quad(lambda x: math.exp(-x), 0.0, math.inf, weight="cos", wvar=3.0),
             note="QAWF"),
        Spec("C5", "quad", "quad weight=sin log(x) on (0, 1)", ("quad",),
             lambda d: lambda: I.quad(_log0, 0.0, 1.0, weight="sin", wvar=10 * math.pi),
             lambda d: lambda: si.quad(_log0, 0.0, 1.0, weight="sin", wvar=10 * math.pi), note="QAWO, 305 calls"),
        Spec("C5", "quad", "quad weight=alg-loga", ("quad",),
             lambda d: lambda: I.quad(_gauss, 0.0, 1.0, weight="alg-loga", wvar=(-0.5, 0.5)),
             lambda d: lambda: si.quad(_gauss, 0.0, 1.0, weight="alg-loga", wvar=(-0.5, 0.5)), note="QAWS"),
        Spec("C5", "quad", "quad weight=cauchy", ("quad",),
             lambda d: lambda: I.quad(_gauss, -1.0, 5.0, weight="cauchy", wvar=0.3),
             lambda d: lambda: si.quad(_gauss, -1.0, 5.0, weight="cauchy", wvar=0.3), note="QAWC"),
    ]  # fmt: skip


def _batch_setup(n):
    def setup():
        import stardust as sd

        b = np.linspace(0.5, 2.0, n)
        return {"b": b, "sb": sd.asarray(b)}

    return setup


def _batch_specs():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    out = []
    for n, size in BATCH_SIZES:
        out.append(Spec("C6", "batch", "tanhsinh exp", ("tanhsinh",), lambda d: lambda: I.tanhsinh(sd.exp, 0.0, d["sb"]),
                        lambda d: _quiet(lambda: si.tanhsinh(np.exp, 0.0, d["b"])), setup=_batch_setup(n), size=size,
                        n=n, quick=n == 1000))  # fmt: skip
    out.append(Spec("C6", "batch", "nsum 1/k^2", ("nsum",), lambda d: lambda: I.nsum(lambda k: 1 / (k * k), 1.0, math.inf),
                    lambda d: _quiet(lambda: si.nsum(lambda k: 1 / (k * k), 1.0, np.inf)), note="10 calls, 8561 terms"))  # fmt: skip
    return out


def _vector_specs():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741

    def g(X):
        return np.exp(-np.sum(X * X, axis=-1))

    def sg(X):
        return sd.exp(-sd.sum(X * X, axis=-1))

    def three(x):
        return np.array([math.exp(x), math.sin(x), x * x])

    def peaks(x):
        return np.array([1 / (1e-4 + (x - 0.3) ** 2), math.sqrt(abs(x - 0.7))])

    k = np.linspace(0.0, 30.0, 1000)
    sk = sd.asarray(k)

    def wide(x):
        return np.cos(k * x)

    def swide(x):
        return sd.cos(sk * x)

    inf = math.inf
    return [
        Spec("C7", "vector", "cubature gk21 2-D gauss", ("cubature",), lambda d: lambda: I.cubature(sg, [0.0, 0.0], [1.0, 1.0]),
             lambda d: lambda: si.cubature(g, [0.0, 0.0], [1.0, 1.0]), note="호출 2번, 점 982개"),
        Spec("C7", "vector", "cubature genz-malik 2-D gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [0.0, 0.0], [1.0, 1.0], rule="genz-malik"),
             lambda d: lambda: si.cubature(g, [0.0, 0.0], [1.0, 1.0], rule="genz-malik"), note="호출 106번"),
        Spec("C7", "vector", "cubature gk15 plane gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [-inf, -inf], [inf, inf], rule="gk15"),
             lambda d: lambda: si.cubature(g, [-inf, -inf], [inf, inf], rule="gk15"), note="호출 448번"),
        Spec("C7", "vector", "cubature genz-malik 4-D gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [0.0] * 4, [1.0] * 4, rule="genz-malik", rtol=1e-6),
             lambda d: lambda: si.cubature(g, [0.0] * 4, [1.0] * 4, rule="genz-malik", rtol=1e-6)),
        Spec("C7", "vector", "quad_vec three functions", ("quad_vec",), lambda d: lambda: I.quad_vec(three, 0.0, 1.0),
             lambda d: lambda: si.quad_vec(three, 0.0, 1.0), note="호출 63번"),
        Spec("C7", "vector", "quad_vec two peaks", ("quad_vec",), lambda d: lambda: I.quad_vec(peaks, 0.0, 1.0),
             lambda d: lambda: si.quad_vec(peaks, 0.0, 1.0), note="호출 819번"),
        Spec("C7", "vector", "quad_vec scalar real line", ("quad_vec",),
             lambda d: lambda: I.quad_vec(_gauss, -inf, inf), lambda d: lambda: si.quad_vec(_gauss, -inf, inf)),
        Spec("C7", "vector", "quad_vec cos(kx)", ("quad_vec",), lambda d: lambda: I.quad_vec(swide, 0.0, 2.0),
             lambda d: lambda: si.quad_vec(wide, 0.0, 2.0), size="(1000,)", n=1000),
    ]  # fmt: skip


def _samples_setup(n):
    def setup():
        import stardust as sd

        x = np.linspace(0.0, 1.0, n + 1)
        y = np.exp(x)
        return {"y": y, "sy": sd.asarray(y), "x": x, "sx": sd.asarray(x)}

    return setup


def _sample_specs():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    out = []
    for n, size in SAMPLE_SIZES:
        for name in ("trapezoid", "simpson", "cumulative_trapezoid", "cumulative_simpson"):
            out.append(Spec("C3", "samples", name, (name,), lambda d, name=name: lambda: getattr(I, name)(d["sy"], d["sx"]),
                            lambda d, name=name: lambda: getattr(si, name)(d["y"], x=d["x"]), setup=_samples_setup(n),
                            size=size, n=n + 1, quick=n == 10**6 and name == "trapezoid"))  # fmt: skip
    out.append(Spec("C3", "samples", "fixed_quad n=5", ("fixed_quad",), lambda d: lambda: I.fixed_quad(sd.exp, 0.0, 1.0),
                    lambda d: lambda: si.fixed_quad(np.exp, 0.0, 1.0)[0]))  # fmt: skip
    return out


def specs() -> list[Spec]:
    """Every planned case, public or not (the C0 scipy timings use them all)."""
    return _sample_specs() + _quad_specs() + _batch_specs() + _vector_specs()


def _public(spec) -> bool:
    import stardust as sd

    return spec.milestone in DONE and all(n in sd.integrate.__all__ for n in spec.needs)


def _pair(r):
    """The quad family's result as scipy's (integral, error) (C10)."""
    return tuple(r)


def _value(r):
    """tanhsinh's integral or nsum's sum (sd's or scipy's result) as an array."""
    return np.asarray(r.sum if hasattr(r, "sum") and not hasattr(r, "integral") else r.integral, dtype=np.float64)


def _vector_value(r):
    """quad_vec's integral (sd's result or scipy's tuple) or cubature's (sd's integral, scipy's
    estimate) as an array."""
    import stardust as sd

    v = r[0] if isinstance(r, tuple) else getattr(r, "estimate", None)
    if v is None:
        v = r.integral
    return np.asarray(v.tolist() if isinstance(v, sd.ndarray) else v, dtype=np.float64)


def cases() -> list[Case]:
    out = []
    for s in specs():
        if not _public(s):
            continue
        if s.group == "quad":
            check = Check(sd=_pair)
        elif s.group == "batch":
            check = Check(sd=_value, ref=lambda label, r: _value(r))
        elif s.group == "vector":
            check = Check(sd=_vector_value, ref=lambda label, r: _vector_value(r))
        else:
            check = Check()
        out.append(Case(area="integrate", group=s.group, name=s.name, size=s.size, n=s.n, setup=s.setup, sd=s.sd,
                        refs=(Ref("scipy", s.scipy),), quick=s.quick, note=s.note, check=check))  # fmt: skip
    return out


TITLES = {
    "samples": "표본 규칙과 고정 규칙",
    "quad": "QUADPACK (quad, dblquad)",
    "batch": "배치 tanh-sinh와 급수",
    "vector": "벡터 값과 다차원",
}
