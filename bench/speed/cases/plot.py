"""sd.plot's drawing times (docs/design/plot.md §3.7, §8): what a figure costs once its text is
in the cache, which is what a notebook pays for every figure after the first.

There is no comparator: no other library is installed for it, and the area's promise is the
picture, not a number. The cases time the steps a figure goes through (the layout into a scene,
then each output) on a small figure and on large data, and the function plots' sampling. The
text comes from the user's cache; the setup draws every figure once, so no TeX run is timed.
"""

from __future__ import annotations

import math

from ..registry import Case, Check, count

SKIP = Check(skip="a picture: nothing to compare")


def _data(n):
    import stardust as sd

    rng = sd.random.default_rng(20261122)
    x = sd.linspace(0.0, 100.0, n)
    return {"sd": sd, "x": x, "y": sd.sin(x) + (rng.random(n) - 0.5), "u": rng.random(n), "v": rng.random(n)}


def _figure(kind, n):
    def setup():
        d = _data(n)
        P = d["sd"].plot
        if kind == "line":
            fig = P.line(d["x"], d["y"]).xlabel(r"$x$").ylabel(r"$y$")
        elif kind == "scatter":
            fig = P.scatter(d["u"], d["v"], alpha=0.3, size=1.5)
        elif kind == "surface":
            sd = d["sd"]
            fig = P.surface(lambda x, y: sd.sin(sd.sqrt(x * x + y * y)), -6, 6, -6, 6)
        else:
            sd = d["sd"]
            fig = P.density(lambda x, y: sd.exp(-(x * x + y * y) / 2) * sd.cos(3 * x), -3, 3, -2, 2)
        fig.to_png(dpi=72)  # the text is typeset here, outside the timing
        d["fig"] = fig
        return d

    return setup


def _again(d):
    """The figure drawn afresh: the scene is dropped, so the layout is part of the time."""
    d["fig"]._touch()
    return d["fig"]


OUTPUTS = {
    "layout": lambda d: lambda: _again(d)._build("bench"),
    "png 300 dpi": lambda d: lambda: _again(d).to_png(dpi=300),
    "svg": lambda d: lambda: _again(d).to_svg(),
    "pdf": lambda d: lambda: _again(d).to_pdf(),
}
SAMPLING = {
    "function sin": lambda sd: lambda: sd.plot.function(sd.sin, 0, 2 * math.pi),
    "function tan": lambda sd: lambda: sd.plot.function(sd.tan, -2 * math.pi, 2 * math.pi),
    "function sin(1/x)": lambda sd: lambda: sd.plot.function(lambda x: sd.sin(1 / x), -0.5, 0.5),
    "implicit": lambda sd: (
        lambda: sd.plot.implicit(lambda x, y: (x * x + y * y - 1) ** 3 - x * x * y**3, -1.5, 1.5, -1.3, 1.5)
    ),
    "contour 8 levels": lambda sd: (
        lambda: sd.plot.contour(lambda x, y: sd.sin(x) * sd.cos(y), -3, 3, -2, 2, levels=8, labels=False)
    ),
}


def cases() -> list[Case]:
    out = []
    for kind, sizes in (
        ("line", (1000, 1_000_000)),
        ("scatter", (1000, 100_000, 1_000_000)),
        ("surface", (14641,)),
        ("density", (172800,)),
    ):
        for n in sizes:
            for name, make in OUTPUTS.items():
                out.append(
                    Case(area="plot", group=kind, name=name, size=count(n) + " points", n=n, setup=_figure(kind, n), sd=make,
                         quick=(kind, n, name) in {("line", 1000, "png 300 dpi"), ("scatter", 1_000_000, "png 300 dpi")}, check=SKIP)
                )  # fmt: skip
    for name, make in SAMPLING.items():
        out.append(
            Case(area="plot", group="sampling", name=name, setup=lambda: {"sd": __import__("stardust")}, sd=lambda d, make=make: make(d["sd"]),
                 threads=(1,), quick=name == "function tan", check=SKIP)
        )  # fmt: skip
    return out


TITLES = {"line": "선", "scatter": "점", "surface": "곡면 (3D)", "density": "밀도", "sampling": "함수 그림의 표본"}
