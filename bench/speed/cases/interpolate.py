"""sd.interpolate (and the array area's sd.interp) against scipy.interpolate 1.18.1, numpy 2.5.4's interp
and R 4.6.2's stats::approx and splinefun (docs/design/interpolate.md §5, §7.5).

The plan of the area's cases, milestone by milestone: every spec below names its milestone and the names
it needs, and cases() keeps a spec once its names are public (IP1 brings sd.interp, IP2 the piecewise
polynomials and the cubic family, IP3 the B-splines, IP4 the polynomial and rational interpolants, IP5
the grids, IP6 the scattered data). The sizes are those of docs/design/interpolate.md §5, where IP0 timed
the comparators. scipy's and numpy's interpolation kernels run on one thread; scipy's RBF solve and AAA's
SVD use Accelerate (threads="follow"). R runs in the one R session.

Data: nodes [0, 1) uniform with standard normal values, evaluation points uniform (or sorted, marked),
numpy arrays made in the setup from fixed seeds; sd gets them as sd.ndarray, outside the timing.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from ..registry import Case, Ref, RRef


@dataclasses.dataclass(frozen=True)
class Spec:
    milestone: str
    group: str
    name: str
    needs: tuple[str, ...]  # sd.interpolate names; "sd.interp" for the array area's function
    size: str
    setup: Callable[[], dict]
    sd: Callable
    refs: tuple = ()
    quick: bool = False
    note: str = ""


def _line(n, m, seed=20261123, sorted_points=False, periodic=False, columns=None):
    """n nodes on [0, 1] with normal values (columns: an (n, columns) batch), m uniform points."""

    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, m, columns or 0]))
        x = np.arange(n) / (n - 1)
        y = g.standard_normal((n, columns) if columns else n)
        if periodic:
            y[-1] = y[0]
        q = g.random(m)
        if sorted_points:
            q.sort()
        return {"x": x, "y": y, "q": q, "sx": sd.asarray(x), "sy": sd.asarray(y), "sq": sd.asarray(q)}

    return setup


def _grid(n, d, m, seed=20261124):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, d, m]))
        axis = np.arange(n) / (n - 1)
        v = g.standard_normal((n,) * d)
        q = g.random((m, d))
        return {"points": (axis,) * d, "v": v, "q": q, "spoints": (sd.asarray(axis),) * d, "sv": sd.asarray(v),
                "sq": sd.asarray(q)}  # fmt: skip

    return setup


def _scatter(n, m, seed=20261125):
    def setup():
        import stardust as sd

        g = np.random.Generator(np.random.PCG64([seed, n, m]))
        y, d, q = g.random((n, 2)), g.standard_normal(n), g.random((m, 2))
        return {"y": y, "d": d, "q": q, "sy": sd.asarray(y), "sd_": sd.asarray(d), "sq": sd.asarray(q)}

    return setup


def _with(setup, build_sd, build_ref):
    """The data with the interpolants built in the setup (the evaluation cases time the evaluation only)."""

    def wrapped():
        d = setup()
        d["obj"] = build_ref(d)
        d["sobj"] = build_sd(d)
        return d

    return wrapped


def _ip():
    import stardust as sd

    return sd.interpolate


def _si():
    import scipy.interpolate as si

    return si


R_LINE = {"x": "x", "y": "y", "q": "q"}


def _linear_specs():
    out = []
    for n in (10**3, 10**6):
        for srt in (False, True):
            name = f"interp ({_count(n)} nodes{', sorted points' if srt else ''})"
            refs = [Ref("numpy", lambda d: lambda: np.interp(d["q"], d["x"], d["y"]))]
            if n == 10**3 and not srt:
                refs.append(Ref("scipy k=1", lambda d: (lambda b: lambda: b(d["q"]))(
                    _si().make_interp_spline(d["x"], d["y"], k=1)), lib="scipy"))  # fmt: skip
                refs.append(RRef("R", "approx(x, y, xout = q, rule = 2)$y", data=R_LINE))
            out.append(Spec("IP1", "linear", name, ("sd.interp",), "10^6 points",
                            _line(n, 10**6, sorted_points=srt), lambda d: lambda: _sd().interp(d["sq"], d["sx"], d["sy"]),
                            tuple(refs), quick=(n, srt) == (10**3, False)))  # fmt: skip
    return out


def _count(n):
    from ..registry import count

    return count(n)


def _sd():
    import stardust as sd

    return sd


def _cubic_specs():
    out = []
    for bc in ("not-a-knot", "natural", "periodic"):
        setup = _line(10**6, 1, periodic=bc == "periodic")
        out.append(Spec("IP2", "cubic", f"CubicSpline build {bc}", ("CubicSpline",), "10^6 points", setup,
                        lambda d, bc=bc: lambda: _ip().CubicSpline(d["sx"], d["sy"], bc_type=bc),
                        (Ref("scipy", lambda d, bc=bc: lambda: _si().CubicSpline(d["x"], d["y"], bc_type=bc)),),
                        quick=bc == "not-a-knot"))  # fmt: skip
    out.append(Spec("IP2", "cubic", "CubicSpline build", ("CubicSpline",), "10^3 x (1000,)", _line(1000, 1, columns=1000),
                    lambda d: lambda: _ip().CubicSpline(d["sx"], d["sy"]),
                    (Ref("scipy", lambda d: lambda: _si().CubicSpline(d["x"], d["y"])),), note="1000 columns"))  # fmt: skip
    for n in (10**3, 10**6):
        setup = _with(_line(n, 10**6), lambda d: _ip().CubicSpline(d["sx"], d["sy"], bc_type="natural"),
                      lambda d: _si().CubicSpline(d["x"], d["y"], bc_type="natural"))  # fmt: skip
        refs = [Ref("scipy", lambda d: lambda: d["obj"](d["q"]))]
        if n == 10**3:
            refs.append(RRef("R", "f(q)", data=R_LINE, setup="f <- splinefun(x, y, method = 'natural')"))
        out.append(Spec("IP2", "cubic", f"CubicSpline eval ({_count(n)} nodes)", ("CubicSpline",), "10^6 points",
                        setup, lambda d: lambda: d["sobj"](d["sq"]), tuple(refs)))  # fmt: skip
    out.append(Spec("IP2", "cubic", "PchipInterpolator build", ("PchipInterpolator",), "10^6 points", _line(10**6, 1),
                    lambda d: lambda: _ip().PchipInterpolator(d["sx"], d["sy"]),
                    (Ref("scipy", lambda d: lambda: _si().PchipInterpolator(d["x"], d["y"])),)))  # fmt: skip
    for meth in ("akima", "makima"):
        out.append(Spec("IP2", "cubic", f"Akima1DInterpolator build {meth}", ("Akima1DInterpolator",), "10^6 points",
                        _line(10**6, 1), lambda d, m=meth: lambda: _ip().Akima1DInterpolator(d["sx"], d["sy"], method=m),
                        (Ref("scipy", lambda d, m=meth: lambda: _si().Akima1DInterpolator(d["x"], d["y"], method=m)),)))  # fmt: skip
    built = _with(
        _line(10**6, 1), lambda d: _ip().CubicSpline(d["sx"], d["sy"]), lambda d: _si().CubicSpline(d["x"], d["y"])
    )
    out.append(Spec("IP2", "ppoly", "derivative + antiderivative", ("CubicSpline",), "10^6 points", built,
                    lambda d: lambda: (d["sobj"].derivative(), d["sobj"].antiderivative()),
                    (Ref("scipy", lambda d: lambda: (d["obj"].derivative(), d["obj"].antiderivative())),)))  # fmt: skip
    out.append(Spec("IP2", "ppoly", "integrate", ("CubicSpline",), "10^6 points", built,
                    lambda d: lambda: d["sobj"].integrate(0.0, 1.0),
                    (Ref("scipy", lambda d: lambda: d["obj"].integrate(0.0, 1.0)),)))  # fmt: skip
    for n in (10**3, 10**6):
        setup = _with(
            _line(n, 1), lambda d: _ip().CubicSpline(d["sx"], d["sy"]), lambda d: _si().CubicSpline(d["x"], d["y"])
        )
        out.append(Spec("IP2", "ppoly", "roots", ("CubicSpline",), f"{_count(n)} points", setup,
                        lambda d: lambda: d["sobj"].roots(), (Ref("scipy", lambda d: lambda: d["obj"].roots()),),
                        note="scipy: closed forms up to degree 3"))  # fmt: skip
    return out


def _bspline_specs():
    out = [
        Spec("IP3", "bspline", "make_interp_spline k=3", ("make_interp_spline",), "10^6 points", _line(10**6, 1),
             lambda d: lambda: _ip().make_interp_spline(d["sx"], d["sy"]).c,
             (Ref("scipy", lambda d: lambda: _si().make_interp_spline(d["x"], d["y"]).c),), quick=True),
        Spec("IP3", "bspline", "make_interp_spline k=5", ("make_interp_spline",), "10^5 points", _line(10**5, 1),
             lambda d: lambda: _ip().make_interp_spline(d["sx"], d["sy"], k=5).c,
             (Ref("scipy", lambda d: lambda: _si().make_interp_spline(d["x"], d["y"], k=5).c),)),
    ]  # fmt: skip
    for n in (10**3, 10**4):
        setup = _with(_line(n, 10**5), lambda d: _ip().make_interp_spline(d["sx"], d["sy"]),
                      lambda d: _si().make_interp_spline(d["x"], d["y"]))  # fmt: skip
        out.append(Spec("IP3", "bspline", f"BSpline eval k=3 ({_count(n)} nodes)", ("make_interp_spline",), "10^5 points",
                        setup, lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),),
                        note="random order: scipy walks from the previous interval"))  # fmt: skip
    setup = _with(_line(10**6, 10**6, sorted_points=True), lambda d: _ip().make_interp_spline(d["sx"], d["sy"]),
                  lambda d: _si().make_interp_spline(d["x"], d["y"]))  # fmt: skip
    out.append(Spec("IP3", "bspline", "BSpline eval k=3 (10^6 nodes, sorted points)", ("make_interp_spline",), "10^6 points", setup,
                    lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),)))  # fmt: skip
    # random order on a long spline: scipy's walk from the previous interval did not finish in 3 minutes here (§3.1),
    # so it is timed only at the small sizes above
    setup = _with(_line(10**6, 10**6), lambda d: _ip().make_interp_spline(d["sx"], d["sy"]), lambda d: None)
    out.append(Spec("IP3", "bspline", "BSpline eval k=3 (10^6 nodes, random points)", ("make_interp_spline",), "10^6 points",
                    setup, lambda d: lambda: d["sobj"](d["sq"]), (), quick=True,
                    note="scipy: minutes (O(n) per point in random order), not timed"))  # fmt: skip
    built = _with(_line(10**6, 1), lambda d: _ip().make_interp_spline(d["sx"], d["sy"]),
                  lambda d: _si().make_interp_spline(d["x"], d["y"]))  # fmt: skip
    out.append(Spec("IP3", "bspline", "BSpline derivative + antiderivative", ("make_interp_spline",), "10^6 points", built,
                    lambda d: lambda: (d["sobj"].derivative().c, d["sobj"].antiderivative().c),
                    (Ref("scipy", lambda d: lambda: (d["obj"].derivative().c, d["obj"].antiderivative().c)),)))  # fmt: skip
    out.append(Spec("IP3", "bspline", "BSpline integrate", ("make_interp_spline",), "10^6 points", built,
                    lambda d: lambda: d["sobj"].integrate(0.1, 0.9),
                    (Ref("scipy", lambda d: lambda: d["obj"].integrate(0.1, 0.9)),)))  # fmt: skip
    out.append(Spec("IP3", "bspline", "PPoly.from_spline", ("make_interp_spline",), "10^6 points", built,
                    lambda d: lambda: _ip().PPoly.from_spline(d["sobj"]),
                    (Ref("scipy", lambda d: lambda: _si().PPoly.from_spline(d["obj"]),
                         differs="breakpoints: every knot, the repeated ones too (sd: the distinct knots of the base "
                                 "interval, IP54)"),)))  # fmt: skip
    return out


def _cheb(m):
    def setup():
        import stardust as sd

        x = np.cos(np.pi * np.arange(101) / 100)
        q = np.random.Generator(np.random.PCG64([20261126, m])).uniform(-1, 1, m)
        return {"x": x, "y": np.exp(x), "q": q, "sx": sd.asarray(x), "sy": sd.asarray(np.exp(x)), "sq": sd.asarray(q)}

    return setup


def _aaa(kind):
    def setup():
        import stardust as sd

        z = np.linspace(-1, 1, 1000)
        f = np.tan(np.pi * z / 2) if kind == "tan" else np.abs(z)
        return {"z": z, "f": f, "sz": sd.asarray(z), "sf": sd.asarray(f)}

    return setup


def _polynomial_specs():
    setup = _with(_cheb(10**6), lambda d: _ip().BarycentricInterpolator(d["sx"], d["sy"]),
                  lambda d: _si().BarycentricInterpolator(d["x"], d["y"], rng=0))  # fmt: skip
    out = [Spec("IP4", "polynomial", "BarycentricInterpolator eval (101 nodes)", ("BarycentricInterpolator",), "10^6 points",
                setup, lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),),
                quick=True)]  # fmt: skip
    for kind in ("tan", "abs"):
        out.append(Spec("IP4", "polynomial", f"AAA {kind}", ("AAA",), "10^3 points", _aaa(kind),
                        lambda d: lambda: _ip().AAA(d["sz"], d["sf"]),
                        (Ref("scipy", lambda d: lambda: _si().AAA(d["z"], d["f"]), threads="follow"),)))  # fmt: skip
    return out


def _grid_specs():
    out = []
    for n, dim in ((1000, 2), (100, 3)):
        for meth in ("linear", "nearest") if dim == 2 else ("linear",):
            setup = _with(_grid(n, dim, 10**6), lambda d, m=meth: _ip().RegularGridInterpolator(d["spoints"], d["sv"], method=m),
                          lambda d, m=meth: _si().RegularGridInterpolator(d["points"], d["v"], method=m))  # fmt: skip
            out.append(Spec("IP5", "grid", f"RegularGridInterpolator {meth} {'x'.join([str(n)] * dim)}",
                            ("RegularGridInterpolator",), "10^6 points", setup, lambda d: lambda: d["sobj"](d["sq"]),
                            (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),), quick=(meth, dim) == ("linear", 2)))  # fmt: skip
    out.append(Spec("IP5", "grid", "RegularGridInterpolator cubic build", ("RegularGridInterpolator",), "(100, 100)",
                    _grid(100, 2, 1), lambda d: lambda: _ip().RegularGridInterpolator(d["spoints"], d["sv"], method="cubic"),
                    (Ref("scipy", lambda d: lambda: _si().RegularGridInterpolator(d["points"], d["v"], method="cubic"),
                         differs="iterative solve (gcrotmk): values within its tolerance"),)))  # fmt: skip
    for meth, m in (("cubic", 10**5), ("pchip", 10**4)):
        setup = _with(_grid(100, 2, m), lambda d, k=meth: _ip().RegularGridInterpolator(d["spoints"], d["sv"], method=k),
                      lambda d, k=meth: _si().RegularGridInterpolator(d["points"], d["v"], method=k))  # fmt: skip
        out.append(Spec("IP5", "grid", f"RegularGridInterpolator {meth} 100x100", ("RegularGridInterpolator",),
                        f"{_count(m)} points", setup, lambda d: lambda: d["sobj"](d["sq"]),
                        (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),)))  # fmt: skip
    return out


def _rbf_specs():
    out = []
    for n in (1000, 4000):
        out.append(Spec("IP6", "scattered", "RBFInterpolator build", ("RBFInterpolator",), f"({n}, 2)", _scatter(n, 1),
                        lambda d: lambda: _ip().RBFInterpolator(d["sy"], d["sd_"]),
                        (Ref("scipy", lambda d: lambda: _si().RBFInterpolator(d["y"], d["d"]), threads="follow"),),
                        quick=n == 1000))  # fmt: skip
    setup = _with(_scatter(1000, 10**4), lambda d: _ip().RBFInterpolator(d["sy"], d["sd_"]),
                  lambda d: _si().RBFInterpolator(d["y"], d["d"]))  # fmt: skip
    out.append(Spec("IP6", "scattered", "RBFInterpolator eval (1000 nodes)", ("RBFInterpolator",), "10^4 points", setup,
                    lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),)))  # fmt: skip
    setup = _with(_scatter(10**5, 10**4), lambda d: _ip().RBFInterpolator(d["sy"], d["sd_"], neighbors=50),
                  lambda d: _si().RBFInterpolator(d["y"], d["d"], neighbors=50))  # fmt: skip
    out.append(Spec("IP6", "scattered", "RBFInterpolator neighbors=50 (10^5 nodes)", ("RBFInterpolator",), "10^4 points",
                    setup, lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),)))  # fmt: skip
    setup = _with(_scatter(10**5, 10**6), lambda d: _ip().NearestNDInterpolator(d["sy"], d["sd_"]),
                  lambda d: _si().NearestNDInterpolator(d["y"], d["d"]))  # fmt: skip
    out.append(Spec("IP6", "scattered", "NearestNDInterpolator (10^5 nodes)", ("NearestNDInterpolator",), "10^6 points",
                    setup, lambda d: lambda: d["sobj"](d["sq"]), (Ref("scipy", lambda d: lambda: d["obj"](d["q"])),)))  # fmt: skip
    return out


SPECS = _linear_specs() + _cubic_specs() + _bspline_specs() + _polynomial_specs() + _grid_specs() + _rbf_specs()


def _public(names) -> bool:
    import stardust as sd

    return all(hasattr(sd, "interp") if n == "sd.interp" else n in sd.interpolate.__all__ for n in names)


def cases() -> list[Case]:
    return [
        Case(area="interpolate", group=s.group, name=s.name, size=s.size, setup=s.setup, sd=s.sd, refs=s.refs,
             quick=s.quick, note=s.note)
        for s in SPECS
        if _public(s.needs)
    ]  # fmt: skip


TITLES = {
    "linear": "선형 (`sd.interp`)",
    "cubic": "3차 계열의 만들기와 평가",
    "ppoly": "조각 다항식의 연산",
    "bspline": "B-스플라인",
    "polynomial": "다항식·유리식",
    "grid": "격자",
    "scattered": "흩어진 점",
}
