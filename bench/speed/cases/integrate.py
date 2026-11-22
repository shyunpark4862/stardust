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

from ..registry import Case, Check, Ref, RRef

DONE: tuple[str, ...] = ("C3", "C4", "C5", "C6", "C7")  # the milestones whose cases run
BATCH_SIZES = ((10, "10"), (1000, "10^3"), (10**4, "10^4"), (10**5, "10^5"))
SAMPLE_SIZES = ((10**3, "10^3"), (10**5, "10^5"), (10**6, "10^6"))


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


def _cgauss(x):
    return complex(math.exp(-x * x), math.sin(x))


def _gauss3(z, y, x):
    return math.exp(-(x * x + y * y + z * z))


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
        Spec("C4", "quad", "quad points=[0.3]", ("quad",), lambda d: lambda: I.quad(_gauss, 0.0, 1.0, points=[0.3]),
             lambda d: lambda: si.quad(_gauss, 0.0, 1.0, points=[0.3]), note="QAGP, 호출 42번"),
        Spec("C4", "quad", "quad complex_func", ("quad",),
             lambda d: lambda: I.quad(_cgauss, 0.0, 1.0, complex_func=True),
             lambda d: lambda: si.quad(_cgauss, 0.0, 1.0, complex_func=True), note="호출 42번"),
        Spec("C4", "quad", "dblquad exp(-x^2-y^2) unit square", ("dblquad",),
             lambda d: lambda: I.dblquad(lambda y, x: math.exp(-(x * x + y * y)), 0.0, 1.0, 0.0, 1.0),
             lambda d: lambda: si.dblquad(lambda y, x: math.exp(-(x * x + y * y)), 0.0, 1.0, 0.0, 1.0),
             note="441 calls"),
        Spec("C4", "quad", "tplquad exp(-x^2-y^2-z^2) unit cube", ("tplquad",),
             lambda d: lambda: I.tplquad(_gauss3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0),
             lambda d: lambda: si.tplquad(_gauss3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0), note="호출 9,261번"),
        Spec("C4", "quad", "nquad exp(-x^2-y^2-z^2) unit cube", ("nquad",),
             lambda d: lambda: I.nquad(_gauss3, [(0.0, 1.0)] * 3),
             lambda d: lambda: si.nquad(_gauss3, [(0.0, 1.0)] * 3), note="호출 9,261번"),
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


def _power(k, p):
    return 1.0 / k**p


def _power_setup(n):
    def setup():
        import stardust as sd

        p = np.linspace(1.5, 4.0, n)
        return {"p": p, "sp": sd.asarray(p)}

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
    for n, size in (BATCH_SIZES[0], BATCH_SIZES[1]):
        out.append(Spec("C6", "batch", "nsum 1/k^p", ("nsum",), lambda d: lambda: I.nsum(_power, 1.0, math.inf, args=(d["sp"],)),
                        lambda d: _quiet(lambda: si.nsum(_power, 1.0, np.inf, args=(d["p"],))), setup=_power_setup(n),
                        size=size, n=n, note="p는 [1.5, 4]의 고른 점"))  # fmt: skip
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


def _romb_setup(n):
    def setup():
        import stardust as sd

        x = np.linspace(0.0, 1.0, n)
        y = np.exp(x)
        return {"y": y, "sy": sd.asarray(y), "dx": float(x[1] - x[0])}

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
    for k in (10, 20):
        n = 2**k + 1
        out.append(Spec("C3", "samples", "romb", ("romb",), lambda d: lambda: I.romb(d["sy"], dx=d["dx"]),
                        lambda d: lambda: si.romb(d["y"], dx=d["dx"]), setup=_romb_setup(n), size=f"{n} points", n=n))  # fmt: skip
    for n in (5, 200):
        out.append(Spec("C3", "samples", f"fixed_quad n={n}", ("fixed_quad",),
                        lambda d, n=n: lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=n),
                        lambda d, n=n: lambda: si.fixed_quad(np.exp, 0.0, 1.0, n=n)[0]))  # fmt: skip
    for n in (8, 20):
        out.append(Spec("C3", "samples", f"newton_cotes {n}", ("newton_cotes",), lambda d, n=n: lambda: I.newton_cotes(n),
                        lambda d, n=n: lambda: si.newton_cotes(n), note="sd는 가중치를 기억한다"))  # fmt: skip
    return out


def specs() -> list[Spec]:
    """Every planned case, public or not (the C0 scipy timings use them all)."""
    return (_sample_specs() + _sample_more() + _quad_specs() + _quad_more() + _quad_vectorized() + _batch_specs()
            + _batch_more() + _vector_specs() + _vector_more())  # fmt: skip


def _quad_vectorized():
    """quad(vectorized=True) (C127): the integrand on sd arrays, one call a rule application, against
    scipy's quad with the scalar integrand (scipy has no such option) and R's integrate (vectorised by
    nature: the same rows as the scalar ones)."""
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    inf = math.inf
    c = math.pi / 4

    def gauss(x):
        return sd.exp(-x * x)

    def peak5(x):
        d = x - c
        return 4.0**-5 / (d * d + 16.0**-5)

    def osc(x):
        return sd.abs(sd.sin(200.0 * x))

    def _osc(x):
        return abs(math.sin(200.0 * x))

    return [
        Spec("C4", "quad", "quad exp(-x^2) on the real line, vectorized=True", ("quad",),
             lambda d: lambda: I.quad(gauss, -inf, inf, vectorized=True), lambda d: lambda: si.quad(_gauss, -inf, inf),
             note="호출 9번(점 30개씩)"),
        Spec("C4", "quad", "quad peak a=5, vectorized=True", ("quad",),
             lambda d: lambda: I.quad(peak5, 0.0, 1.0, vectorized=True), lambda d: lambda: si.quad(_peak5, 0.0, 1.0),
             note="호출 25번(점 21개씩)"),
        Spec("C4", "quad", "quad |sin(200x)| limit reached, vectorized=True", ("quad",),
             lambda d: _quiet(lambda: I.quad(osc, 0.0, 10.0, vectorized=True)),
             lambda d: _quiet(lambda: si.quad(_osc, 0.0, 10.0)), note="구간 50개에 닿는다(status 1), 호출 99번"),
    ]  # fmt: skip


# -- the branches the first plan left out (review of 2026-11-22): the sample rules on a
# matrix along either axis, complex samples, spacing alone, an even count (Simpson's last
# interval), initial; Gauss-Legendre past the table and with an array-valued f; Newton-Cotes at
# given positions; quad's remaining weights, the half line, several breakpoints, a run that
# reaches 'limit'; the multiple integrals with function bounds, per-layer options, 2 and 4
# dimensions; tanhsinh's log, infinite ends, endpoint singularity, args; nsum's direct sum,
# both-sided infinite sum, log, step; quad_vec's norm, rules and points; cubature in 1 and 3
# dimensions, with points, on a quadrant, array-valued.


def _matrix_setup():
    import stardust as sd

    t = np.linspace(0.0, 1.0, 1000)
    Y = np.exp(np.outer(np.linspace(0.5, 1.5, 1000), t))
    return {"Y": Y, "sY": sd.asarray(Y), "dx": float(t[1] - t[0])}


def _complex_setup(n):
    def setup():
        import stardust as sd

        x = np.linspace(0.0, 1.0, n + 1)
        y = np.exp(1j * 3.0 * x)
        return {"y": y, "sy": sd.asarray(y), "dx": float(x[1] - x[0])}

    return setup


NC_POSITIONS = [0.0, 1.0, 2.5, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0]


def _sample_more():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    out = []
    for name in ("trapezoid", "simpson", "cumulative_trapezoid"):
        for axis in (-1, 0):
            out.append(Spec("C3", "samples", f"{name} axis={axis}", (name,),
                            lambda d, name=name, axis=axis: lambda: getattr(I, name)(d["sY"], dx=d["dx"], axis=axis),
                            lambda d, name=name, axis=axis: lambda: getattr(si, name)(d["Y"], dx=d["dx"], axis=axis),
                            setup=_matrix_setup, size="(1000, 1000)", n=10**6,
                            note="1,000줄: 줄마다 또는 열마다"))  # fmt: skip
    out += [
        Spec("C3", "samples", "trapezoid complex", ("trapezoid",), lambda d: lambda: I.trapezoid(d["sy"], dx=d["dx"]),
             lambda d: lambda: si.trapezoid(d["y"], dx=d["dx"]), setup=_complex_setup(10**6), size="10^6", n=10**6 + 1,
             note="실수부와 허수부를 따로"),
        Spec("C3", "samples", "simpson dx only", ("simpson",), lambda d: lambda: I.simpson(d["sy"], dx=1e-6),
             lambda d: lambda: si.simpson(d["y"], dx=1e-6), setup=_samples_setup(10**6), size="10^6", n=10**6 + 1),
        Spec("C3", "samples", "simpson even count", ("simpson",), lambda d: lambda: I.simpson(d["sy"][1:], d["sx"][1:]),
             lambda d: lambda: si.simpson(d["y"][1:], x=d["x"][1:]), setup=_samples_setup(10**6), size="10^6", n=10**6,
             note="짝수 개: 마지막 구간의 Cartwright 보정"),
        Spec("C3", "samples", "cumulative_trapezoid initial=0", ("cumulative_trapezoid",),
             lambda d: lambda: I.cumulative_trapezoid(d["sy"], d["sx"], initial=0),
             lambda d: lambda: si.cumulative_trapezoid(d["y"], x=d["x"], initial=0), setup=_samples_setup(10**6),
             size="10^6", n=10**6 + 1),
        Spec("C3", "samples", "cumulative_simpson dx initial=1", ("cumulative_simpson",),
             lambda d: lambda: I.cumulative_simpson(d["sy"], dx=1e-6, initial=1.0),
             lambda d: lambda: si.cumulative_simpson(d["y"], dx=1e-6, initial=1.0), setup=_samples_setup(10**6),
             size="10^6", n=10**6 + 1),
        Spec("C3", "samples", "fixed_quad n=1000", ("fixed_quad",), lambda d: lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=1000),
             lambda d: lambda: si.fixed_quad(np.exp, 0.0, 1.0, n=1000)[0],
             note="표 밖의 점(처음 계산 뒤 기억한 것)"),
        Spec("C3", "samples", "fixed_quad n=20 array f (100,)", ("fixed_quad",),
             lambda d: (lambda c: lambda: I.fixed_quad(lambda x: sd.exp(c * x), 0.0, 1.0, n=20))(
                 sd.asarray(np.linspace(0.5, 2.0, 100).reshape(100, 1))),
             lambda d: (lambda c: lambda: si.fixed_quad(lambda x: np.exp(c * x), 0.0, 1.0, n=20)[0])(
                 np.linspace(0.5, 2.0, 100).reshape(100, 1)), note="f가 (100, n)을 돌려준다"),
        Spec("C3", "samples", "newton_cotes positions", ("newton_cotes",), lambda d: lambda: I.newton_cotes(NC_POSITIONS),
             lambda d: lambda: si.newton_cotes(np.array(NC_POSITIONS)), note="고르지 않은 9점(sd는 기억한다)"),
    ]  # fmt: skip
    return out


def _kinks(x):
    return abs(math.sin(5 * math.pi * x))


def _osc(x):
    return abs(math.sin(200.0 * x))


def _quad_more():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    disk_lo = lambda x: -math.sqrt(max(1.0 - x * x, 0.0))  # noqa: E731
    disk_hi = lambda x: math.sqrt(max(1.0 - x * x, 0.0))  # noqa: E731
    g2 = lambda y, x: math.exp(-(x * x + y * y))  # noqa: E731
    g4 = lambda w, z, y, x: math.exp(-(x * x + y * y + z * z + w * w))  # noqa: E731
    out = [
        Spec(
            "C4",
            "quad",
            "quad exp(-x) on the half line",
            ("quad",),
            lambda d: lambda: I.quad(lambda x: math.exp(-x), 0.0, math.inf),
            lambda d: lambda: si.quad(lambda x: math.exp(-x), 0.0, math.inf),
            note="QAGI 한쪽",
        ),
        Spec(
            "C4",
            "quad",
            "quad |sin(5 pi x)| points x4",
            ("quad",),
            lambda d: lambda: I.quad(_kinks, 0.0, 1.0, points=[0.2, 0.4, 0.6, 0.8]),
            lambda d: lambda: si.quad(_kinks, 0.0, 1.0, points=[0.2, 0.4, 0.6, 0.8]),
            note="QAGP, 꺾임 넷",
        ),
        Spec(
            "C4",
            "quad",
            "quad |sin(200x)| limit reached",
            ("quad",),
            lambda d: _quiet(lambda: I.quad(_osc, 0.0, 10.0)),
            lambda d: _quiet(lambda: si.quad(_osc, 0.0, 10.0)),
            note="구간 50개에 닿는다(status 1)",
        ),
        Spec(
            "C4",
            "quad",
            "dblquad disk bounds",
            ("dblquad",),
            lambda d: lambda: I.dblquad(g2, -1.0, 1.0, disk_lo, disk_hi),
            lambda d: lambda: si.dblquad(g2, -1.0, 1.0, disk_lo, disk_hi),
            note="안쪽 끝이 x의 함수",
        ),
        Spec(
            "C4",
            "quad",
            "nquad 2-D range function",
            ("nquad",),
            lambda d: lambda: I.nquad(g2, [lambda x: (0.0, x), (0.0, 1.0)], opts={"limit": 100}),
            lambda d: lambda: si.nquad(g2, [lambda x: (0.0, x), (0.0, 1.0)], opts={"limit": 100}),
            note="삼각형, 모든 층에 같은 opts",
        ),
        Spec(
            "C4",
            "quad",
            "nquad 2-D per-layer opts",
            ("nquad",),
            lambda d: (
                lambda: I.nquad(
                    lambda y, x: _kinks(x) * math.exp(-y * y),
                    [(0.0, 1.0), (0.0, 1.0)],
                    opts=[{}, {"points": [0.2, 0.4, 0.6, 0.8]}],
                )
            ),
            lambda d: (
                lambda: si.nquad(
                    lambda y, x: _kinks(x) * math.exp(-y * y),
                    [(0.0, 1.0), (0.0, 1.0)],
                    opts=[{}, {"points": [0.2, 0.4, 0.6, 0.8]}],
                )
            ),
            note="바깥 층만 points",
        ),
        Spec(
            "C4",
            "quad",
            "nquad exp(-|x|^2) 4-D",
            ("nquad",),
            lambda d: lambda: I.nquad(g4, [(0.0, 1.0)] * 4),
            lambda d: lambda: si.nquad(g4, [(0.0, 1.0)] * 4),
            note="호출 194,481번",
        ),
        Spec(
            "C5",
            "quad",
            "quad weight=cos finite",
            ("quad",),
            lambda d: lambda: I.quad(_gauss, 0.0, 10.0, weight="cos", wvar=20.0),
            lambda d: lambda: si.quad(_gauss, 0.0, 10.0, weight="cos", wvar=20.0),
            note="QAWO",
        ),
    ]
    for w in ("alg", "alg-logb", "alg-log"):
        out.append(Spec("C5", "quad", f"quad weight={w}", ("quad",),
                        lambda d, w=w: lambda: I.quad(_gauss, 0.0, 1.0, weight=w, wvar=(-0.5, 0.3)),
                        lambda d, w=w: lambda: si.quad(_gauss, 0.0, 1.0, weight=w, wvar=(-0.5, 0.3)), note="QAWS"))  # fmt: skip
    return out  # fmt: skip


def _log_gauss(x, c):
    return -c * x * x


def _batch_more():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    inf = math.inf
    s = _batch_setup
    out = [
        Spec("C6", "batch", "tanhsinh log exp(-c x^2)", ("tanhsinh",),
             lambda d: lambda: I.tanhsinh(_log_gauss, 0.0, 2.0, args=(d["sb"],), log=True),
             lambda d: _quiet(lambda: si.tanhsinh(_log_gauss, 0.0, 2.0, args=(d["b"],), log=True)), setup=s(10**4),
             size="10^4", n=10**4, note="log=True, c는 [0.5, 2]"),
        Spec("C6", "batch", "tanhsinh exp(-c x) half line", ("tanhsinh",),
             lambda d: lambda: I.tanhsinh(lambda x, c: sd.exp(-c * x), 0.0, inf, args=(d["sb"],)),
             lambda d: _quiet(lambda: si.tanhsinh(lambda x, c: np.exp(-c * x), 0.0, np.inf, args=(d["b"],))),
             setup=s(10**3), size="10^3", n=10**3),
        Spec("C6", "batch", "tanhsinh exp(-c x^2) real line", ("tanhsinh",),
             lambda d: lambda: I.tanhsinh(lambda x, c: sd.exp(-c * x * x), -inf, inf, args=(d["sb"],)),
             lambda d: _quiet(lambda: si.tanhsinh(lambda x, c: np.exp(-c * x * x), -np.inf, np.inf, args=(d["b"],))),
             setup=s(10**4), size="10^4", n=10**4),
        Spec("C6", "batch", "tanhsinh c/sqrt(x) endpoint", ("tanhsinh",),
             lambda d: lambda: I.tanhsinh(lambda x, c: c / sd.sqrt(x), 0.0, 1.0, args=(d["sb"],)),
             lambda d: _quiet(lambda: si.tanhsinh(lambda x, c: c / np.sqrt(x), 0.0, 1.0, args=(d["b"],))),
             setup=s(10**3), size="10^3", n=10**3, note="끝점 특이점"),
        Spec("C6", "batch", "nsum 1/k^p direct", ("nsum",), lambda d: lambda: I.nsum(_power, 1.0, 1000.0, args=(d["sp"],)),
             lambda d: _quiet(lambda: si.nsum(_power, 1.0, 1000.0, args=(d["p"],))), setup=_power_setup(10**3),
             size="10^3", n=10**3, note="항 1,000개를 직접 더한다"),
        Spec("C6", "batch", "nsum exp(-k^2/c) both sides", ("nsum",),
             lambda d: lambda: I.nsum(lambda k, c: sd.exp(-k * k / c), -inf, inf, args=(d["sp"],)),
             lambda d: _quiet(lambda: si.nsum(lambda k, c: np.exp(-k * k / c), -np.inf, np.inf, args=(d["p"],))),
             setup=_power_setup(10**3), size="10^3", n=10**3, note="양쪽 무한: 호출마다 f를 두 번"),
        Spec("C6", "batch", "nsum log 1/k^p", ("nsum",),
             lambda d: lambda: I.nsum(lambda k, p: -p * sd.log(k), 1.0, inf, args=(d["sp"],), log=True),
             lambda d: _quiet(lambda: si.nsum(lambda k, p: -p * np.log(k), 1.0, np.inf, args=(d["p"],), log=True)),
             setup=_power_setup(10), size="10", n=10, note="log=True"),
        Spec("C6", "batch", "nsum 1/k^2 step=0.5", ("nsum",),
             lambda d: lambda: I.nsum(lambda k: 1 / (k * k), 1.0, inf, step=0.5),
             lambda d: _quiet(lambda: si.nsum(lambda k: 1 / (k * k), 1.0, np.inf, step=0.5))),
    ]  # fmt: skip
    return out


def _vector_more():
    import scipy.integrate as si

    import stardust as sd

    I = sd.integrate  # noqa: E741
    inf = math.inf

    def g(X):
        return np.exp(-np.sum(X * X, axis=-1))

    def sg(X):
        return sd.exp(-sd.sum(X * X, axis=-1))

    c3 = np.array([1.0, 2.0, 3.0])
    sc3 = sd.asarray(c3)

    def g3(X):
        return np.exp(-np.sum(X * X, axis=-1)[:, None] * c3)

    def sg3(X):
        return sd.exp(-sd.reshape(sd.sum(X * X, axis=-1), (-1, 1)) * sc3)

    def kink(X):
        return np.sqrt(np.abs(X[:, 0] - 0.5) + np.abs(X[:, 1] - 0.5))

    def skink(X):
        return sd.sqrt(sd.abs(X[:, 0] - 0.5) + sd.abs(X[:, 1] - 0.5))

    def three(x):
        return np.array([math.exp(x), math.sin(x), x * x])

    def peaks(x):
        return np.array([1 / (1e-4 + (x - 0.3) ** 2), math.sqrt(abs(x - 0.7))])

    def moments(x):
        e = math.exp(-x)
        return np.array([e, x * e, x * x * e])

    return [
        Spec("C7", "vector", "quad_vec three functions norm=max", ("quad_vec",),
             lambda d: lambda: I.quad_vec(three, 0.0, 1.0, norm="max"), lambda d: lambda: si.quad_vec(three, 0.0, 1.0, norm="max")),
        Spec("C7", "vector", "quad_vec three functions trapezoid", ("quad_vec",),
             lambda d: lambda: I.quad_vec(three, 0.0, 1.0, quadrature="trapezoid"),
             lambda d: lambda: si.quad_vec(three, 0.0, 1.0, quadrature="trapezoid")),
        Spec("C7", "vector", "quad_vec two peaks gk15", ("quad_vec",),
             lambda d: lambda: I.quad_vec(peaks, 0.0, 1.0, quadrature="gk15"),
             lambda d: lambda: si.quad_vec(peaks, 0.0, 1.0, quadrature="gk15")),
        Spec("C7", "vector", "quad_vec two peaks points", ("quad_vec",),
             lambda d: lambda: I.quad_vec(peaks, 0.0, 1.0, points=[0.3, 0.7]),
             lambda d: lambda: si.quad_vec(peaks, 0.0, 1.0, points=[0.3, 0.7])),
        Spec("C7", "vector", "quad_vec moments half line", ("quad_vec",), lambda d: lambda: I.quad_vec(moments, 0.0, inf),
             lambda d: lambda: si.quad_vec(moments, 0.0, inf), note="gk15, x = a + (1 - t)/t"),
        Spec("C7", "vector", "cubature gk21 1-D gauss", ("cubature",), lambda d: lambda: I.cubature(sg, [0.0], [1.0]),
             lambda d: lambda: si.cubature(g, [0.0], [1.0])),
        Spec("C7", "vector", "cubature gk21 3-D gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [0.0] * 3, [1.0] * 3), lambda d: lambda: si.cubature(g, [0.0] * 3, [1.0] * 3),
             note="영역마다 점 9,261개"),
        Spec("C7", "vector", "cubature gk21 2-D kink points", ("cubature",),
             lambda d: lambda: I.cubature(skink, [0.0, 0.0], [1.0, 1.0], points=[[0.5, 0.5]]),
             lambda d: lambda: si.cubature(kink, [0.0, 0.0], [1.0, 1.0], points=[np.array([0.5, 0.5])]),
             note="꺾인 점이 영역의 모서리에"),
        Spec("C7", "vector", "cubature gk15 quadrant gauss", ("cubature",),
             lambda d: lambda: I.cubature(sg, [0.0, 0.0], [inf, inf], rule="gk15"),
             lambda d: lambda: si.cubature(g, [0.0, 0.0], [inf, inf], rule="gk15"), note="반무한 두 축"),
        Spec("C7", "vector", "cubature genz-malik 2-D array (3,)", ("cubature",),
             lambda d: lambda: I.cubature(sg3, [0.0, 0.0], [1.0, 1.0], rule="genz-malik"),
             lambda d: lambda: si.cubature(g3, [0.0, 0.0], [1.0, 1.0], rule="genz-malik"), size="(3,)", n=3,
             note="배열 값"),
    ]  # fmt: skip


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


# R's stats::integrate (a C translation of QUADPACK's dqags and dqagi) on quad's unweighted rows,
# at quad's defaults. Its integrand is an R function, vectorised over the rule's points (one R call
# for 21 or 15 points), where sd and scipy call the Python function once a point.
R_QUAD_OPTS = "rel.tol = 1.49e-8, abs.tol = 1.49e-8, subdivisions = 50L, stop.on.error = FALSE"
R_QUAD = {
    "quad exp(-x^2) on (0, 1)": ("function(x) exp(-x^2)", "0", "1"),
    "quad exp(-x^2) on the real line": ("function(x) exp(-x^2)", "-Inf", "Inf"),
    "quad peak a=5": ("function(x) 4^-5 / ((x - pi / 4)^2 + 16^-5)", "0", "1"),
    "quad exp(-x) on the half line": ("function(x) exp(-x)", "0", "Inf"),
    "quad |sin(200x)| limit reached": ("function(x) abs(sin(200 * x))", "0", "10"),
}
# quad(vectorized=True)'s rows: R's integrand is the same vectorised one
for _k in ("quad exp(-x^2) on the real line", "quad peak a=5", "quad |sin(200x)| limit reached"):
    R_QUAD[_k + ", vectorized=True"] = R_QUAD[_k]
del _k
R_QUAD_NOTE = "R은 벡터화한 R 피적분함수(규칙마다 한 번 부른다)"


def _r_quad(name):
    f, a, b = R_QUAD[name]
    return RRef("R", f"integrate(f, {a}, {b}, {R_QUAD_OPTS})", setup=f"f <- {f}", value="c(.v$value, .v$abs.error)")


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
        elif s.name.startswith("newton_cotes"):
            check = Check(sd=lambda r: (np.asarray(r[0].tolist()), r[1]), ref=lambda label, r: (np.asarray(r[0]), r[1]))
        else:
            check = Check()
        out.append(Case(area="integrate", group=s.group, name=s.name, size=s.size, n=s.n, setup=s.setup, sd=s.sd,
                        refs=(Ref("scipy", s.scipy),) + ((_r_quad(s.name),) if s.name in R_QUAD else ()),
                        quick=s.quick, note=", ".join(x for x in (s.note, R_QUAD_NOTE if s.name in R_QUAD else "") if x),
                        check=check))  # fmt: skip
    return out


TITLES = {
    "samples": "표본 규칙과 고정 규칙",
    "quad": "QUADPACK (quad, dblquad, tplquad, nquad)",
    "batch": "배치 tanh-sinh와 급수",
    "vector": "벡터 값과 다차원",
}
