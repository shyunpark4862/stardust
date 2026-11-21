"""sd.integrate against scipy.integrate 1.18.1 (docs/design/integrate.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names the sd.integrate
names it needs and its milestone, and cases() keeps a case only once the milestone is in DONE and
the names are public (C0 has none, so the area has no case yet; C3 brings the sample rules and
fixed_quad, C4 quad and the multiple integrals, C5 quad's weights, C6 tanhsinh and nsum, C7
quad_vec and cubature). The comparator is scipy at its defaults (one thread); the integrands are
the same Python functions on both sides (quad's family: one float a call; tanhsinh and nsum:
numpy arrays for scipy, sd arrays for sd), so the times compare the work between the calls.
"""

from __future__ import annotations

import dataclasses
import math
import warnings
from typing import Callable

import numpy as np

from ..registry import Case, Ref

DONE: tuple[str, ...] = ()  # the milestones whose cases run
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

    return [
        Spec("C7", "vector", "cubature gk21 2-D gauss", ("cubature",), lambda d: lambda: I.cubature(sg, [0.0, 0.0], [1.0, 1.0]),
             lambda d: lambda: si.cubature(g, [0.0, 0.0], [1.0, 1.0])),
        Spec("C7", "vector", "cubature genz-malik 2-D gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [0.0, 0.0], [1.0, 1.0], rule="genz-malik"),
             lambda d: lambda: si.cubature(g, [0.0, 0.0], [1.0, 1.0], rule="genz-malik")),
        Spec("C7", "vector", "quad_vec three functions", ("quad_vec",),
             lambda d: lambda: I.quad_vec(lambda x: np.array([math.exp(x), math.sin(x), x * x]), 0.0, 1.0),
             lambda d: lambda: si.quad_vec(lambda x: np.array([math.exp(x), math.sin(x), x * x]), 0.0, 1.0)),
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
                    lambda d: lambda: si.fixed_quad(np.exp, 0.0, 1.0)))  # fmt: skip
    return out


def specs() -> list[Spec]:
    """Every planned case, public or not (the C0 scipy timings use them all)."""
    return _sample_specs() + _quad_specs() + _batch_specs() + _vector_specs()


def _public(spec) -> bool:
    import stardust as sd

    return spec.milestone in DONE and all(n in sd.integrate.__all__ for n in spec.needs)


def cases() -> list[Case]:
    out = []
    for s in specs():
        if not _public(s):
            continue
        out.append(Case(area="integrate", group=s.group, name=s.name, size=s.size, n=s.n, setup=s.setup, sd=s.sd,
                        refs=(Ref("scipy", s.scipy),), quick=s.quick, note=s.note))  # fmt: skip
    return out


TITLES = {
    "samples": "표본 규칙과 고정 규칙",
    "quad": "QUADPACK (quad, dblquad)",
    "batch": "배치 tanh-sinh와 급수",
    "vector": "벡터 값과 다차원",
}
