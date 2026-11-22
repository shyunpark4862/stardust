"""Golden cases for sd.plot (tools/golden.py has the contract; docs/design/plot.md §8).

The scene golden: every hand-built scene of tests/plot/scenes.py in both font families, as its
dict (shapes and strings, exact), as the SVG and PDF files (their hashes) and as the pixels the
rasterizer paints at three resolutions (the hash of the pixel buffer, not of the PNG file, so
zlib is no part of it). The text comes from the
recorded cache tests/plot/data/text, read frozen: no TeX runs and the results do not move with
the TeX installation. The engine's own results are cases too: layouts of recorded strings and of
numbers laid out from their characters, and the errors.
Each milestone adds the cases of what it brings (docs/design/plot.md §9); nothing is public yet.
"""

import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from stardust._core import plot as _core_plot  # noqa: E402
from stardust.plot import _pdf, _png, _svg, _text  # noqa: E402
from tests.plot import figures, scenes  # noqa: E402


def _frozen(fn):
    def thunk():
        with _text.cache(scenes.TEXT, frozen=True):
            return fn()

    return thunk


def _layout(family, s):
    lay = _text.engine(family).layout([s])[0]
    return {
        "box": [lay.width, lay.height, lay.depth],
        "fonts": [list(f) for f in lay.fonts],
        "glyphs": [list(g) for g in lay.glyphs],
        "rules": [list(r) for r in lay.rules],
    }


def _svg_case(name, family):
    text = _svg.write(scenes.SCENES[name](family))
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "bytes": len(text.encode()), "head": text[:200]}


def _png_case(name, family, dpi):
    w, h, px = _png.pixels(scenes.SCENES[name](family), dpi)
    return {"size": [w, h], "sha256": hashlib.sha256(px).hexdigest(), "ink": sum(px[3::4])}


def _pdf_case(name, family):
    data = _pdf.write(scenes.SCENES[name](family))
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _strokes():
    """The rasterizer alone: widths, caps, joins and dashes on a small canvas, every pixel."""
    from array import array

    out = {}
    pts = array("d", [3.3, 4.1, 20.5, 6.0, 12.2, 17.7, 27.9, 15.2]).tobytes()
    for cap in (0, 1, 2):
        for join in (0, 1):
            for width, dash in ((0.4, []), (1.7, []), (3.5, [4.0, 2.5])):
                c = _core_plot.Canvas(32, 22)
                c.stroke(b"MLLL", pts, (1.0, 0.0, 0.0, 1.0, 0.0, 0.0), (0.1, 0.2, 0.7, 0.8), (), width, cap, join, dash)
                out[f"cap{cap}/join{join}/w{width}/{'dash' if dash else 'solid'}"] = hashlib.sha256(
                    c.rgba8()
                ).hexdigest()
    c = _core_plot.Canvas(16, 12)
    c.fill(
        b"MCCZMLLZ",
        array("d", [2, 2, 14, 1, 15, 11, 3, 9, 1, 6, 0, 3, 2, 2, 6, 5, 9, 5, 7, 8]).tobytes(),
        (1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (0.9, 0.3, 0.0, 1.0),
        (1.5, 0.5, 13.25, 10.0),
    )
    out["fill"] = list(c.rgba8())
    return out


def _scene_summary(d):
    """A figure's scene in short: its size, the count of each kind of item, every string, and the
    hash of the whole (the exact numbers as float.hex), so the record stays small."""

    def exact(o):
        if isinstance(o, float):
            return o.hex()
        if isinstance(o, dict):
            return {k: exact(v) for k, v in o.items()}
        if isinstance(o, list):
            return [exact(v) for v in o]
        return o

    kinds = {}
    for item in d["items"]:
        kinds[item["type"]] = kinds.get(item["type"], 0) + 1
    text = json.dumps(exact(d), sort_keys=True, ensure_ascii=False)
    return {
        "size": [d["width"], d["height"]],
        "items": kinds,
        "strings": [i["s"] for i in d["items"] if i["type"] == "text"],
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _figure_case(name, family):
    fig = figures.FIGURES[name](family)
    svg, pdf = fig.to_svg(), fig.to_pdf()
    w, h, px = _png.pixels(fig._build("golden"), 150)
    return {
        "repr": repr(fig),
        "height": fig.height,
        "scene": _scene_summary(fig.scene()),
        "svg": hashlib.sha256(svg.encode()).hexdigest(),
        "pdf": hashlib.sha256(pdf).hexdigest(),
        "png": [w, h, hashlib.sha256(px).hexdigest()],
    }


def _raises(fn):
    def thunk():
        return fn()

    return thunk


def cases(L):
    sd = L.sd
    P = sd.plot
    yield "raster/strokes", _strokes
    for name in figures.FIGURES:
        for family in _text.FAMILIES:
            yield f"figure/{name}/{family}", _frozen(lambda name=name, family=family: _figure_case(name, family))
    # every public name once more by itself, and the errors' messages
    xs, ys = [0.0, 1.0, 2.0], [1.0, 3.0, 2.0]
    small = {
        "Figure": lambda: P.Figure(width=60, height=40, theme="boxed").line(xs, ys),
        "line": lambda: P.line(xs, ys, style="-.", width=0.8, color="#12345680"),
        "scatter": lambda: P.scatter(xs, ys, shape="^", size=6, alpha=0.5),
        "bar": lambda: P.bar(["Group A", "Residuals"], [1.0, -2.0], horizontal=True),
        "area": lambda: P.area(xs, ys, [0.5, 0.5, 0.5]),
        "step": lambda: P.step(xs, ys, where="pre"),
        "errorbar": lambda: P.errorbar(xs, ys, yerr=0.25, xerr=[0.1, 0.2, 0.3]),
        "hline": lambda: P.hline(0.5, style=":"),
        "vline": lambda: P.vline(-1.0, color="red"),
        "text": lambda: P.text(0.5, 0.5, "Group A", rotation=30),
        "annotate": lambda: P.annotate("Residuals", (0.0, 0.0), (1.0, 1.0)),
        "grid": lambda: P.grid([P.line(xs, ys), P.scatter(xs, ys)], labels="A"),
        "function": lambda: P.function(lambda x: 1 / (x - 0.3), 0, 1, asymptotes=True),
        "parametric": lambda: P.parametric(sd.cos, lambda t: sd.sin(2 * t), 0, 6.5),
        "polar": lambda: P.polar(lambda t: 1 + sd.cos(t), 0, 6.5, style="--"),
        "implicit": lambda: P.implicit(lambda x, y: y * y - x**3 + x, -2, 2, -2, 2, level=0.2),
        "region": lambda: P.region(lambda x, y: sd.minimum(x, y), -1, 1, -1, 1, alpha=1.0),
        "contour": lambda: P.contour(lambda x, y: x * y, -1, 1, -1, 1, levels=3, labels=False, color="black"),
        "density": lambda: P.density(lambda x, y: x * y, -1, 1, -1, 1, cmap="cividis", colorbar=False),
        "phase": lambda: P.phase(lambda z: z * z, -1, 1, -1, 1, contours=False),
        "surface": lambda: P.surface(lambda x, y: x * x + y, -1, 1, -1, 1, cmap="magma", azimuth=20, elevation=40),
        "parametric_surface": lambda: P.parametric_surface(
            lambda u, v: u, lambda u, v: v, lambda u, v: u * v, -1, 1, -1, 1, mesh=False
        ),
        "curve3d": lambda: P.curve3d(sd.cos, sd.sin, lambda t: t, 0, 6, perspective=False),
        "scatter3d": lambda: P.scatter3d(xs, ys, [0.0, 1.0, 4.0], size=4),
        "histogram": lambda: P.histogram(ys * 3, bins=3, density=True),
        "ecdf": lambda: P.ecdf(ys * 2),
        "boxplot": lambda: P.boxplot(None, xs + ys, points=True),
        "strip": lambda: P.strip(["Group A", "Residuals", "Group A"], ys, rng=3),
        "qq": lambda: P.qq(xs + ys, dist=sd.stats.Exponential(scale=1.0)),
        "heatmap": lambda: P.heatmap([xs, ys], colorbar=False),
        "coefficients": lambda: P.coefficients(
            sd.regression.ols(ys + xs, [[0.0], [1.0], [2.0], [3.0], [5.0], [9.0]]), intercept=True
        ),
        "distribution": lambda: P.distribution(sd.stats.Poisson(mu=3.0), kind="cdf"),
        "diagnostics": lambda: P.diagnostics(sd.regression.ols(ys + xs, [[0.0], [1.0], [2.0], [3.0], [5.0], [9.0]])),
        "pairs": lambda: P.pairs(sd.frame.DataFrame({"x": xs, "y": ys})),
    }
    for name, make in (
        ("tan", lambda: P.function(sd.tan, -5, 5)),
        ("sqrt", lambda: P.function(sd.sqrt, -1, 4)),
        ("floor", lambda: P.function(sd.floor, -2.5, 2.5)),
        ("sin1x", lambda: P.function(lambda x: sd.sin(1 / x), -0.5, 0.5)),
        ("gamma", lambda: P.function(sd.special.gamma, -4.2, 4.2)),
    ):
        yield f"info/{name}", (lambda make=make: [list(i) for i in make().info])
    for name, make in small.items():
        yield (
            f"each/{name}",
            _frozen(lambda make=make: (lambda f: {"repr": repr(f), "scene": _scene_summary(f.scene())})(make())),
        )
    errors = {
        "length": lambda: P.line(xs, [1.0]),
        "column": lambda: P.line("x", ys),
        "style": lambda: P.line(xs, ys, style="dashed"),
        "color": lambda: P.scatter(xs, ys, color="reddish"),
        "keyword": lambda: P.bar(xs, ys, fontsize=3),
        "save": lambda: P.line(xs, ys).save("a.jpg"),
        "cell": lambda: P.Figure(ncols=2)[2],
        "typeset": lambda: P.line(xs, ys).xlabel("50% of").to_svg(),
    }
    for name, make in errors.items():
        yield f"error/{name}", _frozen(make)
    for family in _text.FAMILIES:
        for name, make in scenes.SCENES.items():
            yield f"scene/{name}/{family}", (lambda make=make, family=family: make(family).to_dict())
            yield f"svg/{name}/{family}", _frozen(lambda name=name, family=family: _svg_case(name, family))
            yield f"pdf/{name}/{family}", _frozen(lambda name=name, family=family: _pdf_case(name, family))
            for dpi in (72, 100, 300):
                yield (
                    f"png/{name}/{family}/{dpi}",
                    _frozen(lambda name=name, family=family, dpi=dpi: _png_case(name, family, dpi)),
                )
        for s in scenes.STRINGS + scenes.NUMBERS + ["$-0.125$", "$10^{-15}$", "$31415.9$"]:
            yield f"layout/{family}/{s}", _frozen(lambda family=family, s=s: _layout(family, s))
        yield (
            f"error/frozen-miss/{family}",
            _frozen(lambda family=family: _text.engine(family).layout(["never recorded"])),
        )
    yield "error/family", _frozen(lambda: _text.engine("mono"))
