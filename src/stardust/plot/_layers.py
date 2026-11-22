"""The layers of a cell: what the basic plotting functions add (docs/api/plot.md §3).

A layer keeps its data as float64 sd arrays in data coordinates and knows three things: its
extent along an axis (for the automatic limits), how to draw itself into a Frame (the page
coordinates of a laid-out cell), and its legend entries. Lines break at values that are not
finite; such points are not drawn.
"""

import math
import re
from array import array

from .. import _core as sd
from .._core import plot as _core_plot
from . import _color, _scene
from ._figure import DASHES, interleave

_RUN = re.compile(b"\x01+")
EDGE_UPTO = 3000  # points up to which a scatter's markers get a white rim
THIN_FROM = 8000  # points from which a line is thinned for drawing
THIN_COLUMN = 0.05  # bp: a seventh of a pixel at 200 dpi, under half a pixel at 600


def finite_range(a, positive=False):
    """(min, max) of the finite (and, for a log axis, positive) values of an sd array, or None."""
    if a.shape[0] == 0:
        return None
    ok = sd.isfinite(a)
    if positive:
        ok = ok & (a > 0.0)
    if not bool(sd.any(ok)):
        return None
    b = sd.where(ok, a, math.nan)
    return float(sd.nanmin(b)), float(sd.nanmax(b))


def polyline(X, Y):
    """(ops, xy) of the runs of finite points of two page-coordinate arrays."""
    n = X.shape[0]
    xy = interleave(X, Y)
    ok = sd.isfinite(X) & sd.isfinite(Y)
    if bool(sd.all(ok)):
        runs = [xy] if n else []
    else:
        mask = bytes(memoryview(ok))
        runs = [xy[2 * a : 2 * b] for a, b in (m.span() for m in _RUN.finditer(mask))]
    ops, out = [], array("d")
    for run in runs:
        if len(run) > 2 * THIN_FROM:  # a long run: what a column narrower than any pixel cannot show is left out
            thin = array("d")
            thin.frombytes(_core_plot.decimate(run.tobytes(), THIN_COLUMN))
            run = thin
        ops.append("M" + "L" * (len(run) // 2 - 1))
        out.extend(run)
    return "".join(ops), out


def points(X, Y):
    """The finite points of two page-coordinate arrays as array('d') pairs."""
    xy = interleave(X, Y)
    ok = sd.isfinite(X) & sd.isfinite(Y)
    if bool(sd.all(ok)):
        return xy
    mask = bytes(memoryview(ok))
    out = array("d")
    for m in _RUN.finditer(mask):
        a, b = m.span()
        out.extend(xy[2 * a : 2 * b])
    return out


def _sample(X, Y, limit=200):
    """Up to `limit` finite points of a layer (for placing the legend)."""
    xy = points(X, Y)
    n = len(xy) // 2
    step = max(1, n // limit)
    return [(xy[2 * i], xy[2 * i + 1]) for i in range(0, n, step)]


# marker shapes: outlines around the origin for a marker of diameter 1
def _marker(shape):
    if shape == "o":
        return _scene.circle(0.0, 0.0, 0.5)
    if shape == "s":
        return _scene.rect(-0.44, -0.44, 0.88, 0.88)
    if shape == "^":
        return _scene.polyline([(0.0, -0.56), (0.52, 0.34), (-0.52, 0.34)], closed=True)
    if shape == "v":
        return _scene.polyline([(0.0, 0.56), (0.52, -0.34), (-0.52, -0.34)], closed=True)
    if shape == "D":
        return _scene.polyline([(0.0, -0.58), (0.46, 0.0), (0.0, 0.58), (-0.46, 0.0)], closed=True)
    if shape == "+":
        return "MLML", [-0.5, 0.0, 0.5, 0.0, 0.0, -0.5, 0.0, 0.5]
    return "MLML", [-0.4, -0.4, 0.4, 0.4, -0.4, 0.4, 0.4, -0.4]  # x


MARKERS = ("o", "s", "^", "v", "D", "+", "x")
_OPEN = ("+", "x")


def marker_item(shape, size, at, color, edge, edge_width, clip):
    ops, xy = _marker(shape)
    xy = [v * size for v in xy]
    if shape in _OPEN:
        return _scene.Markers(ops, xy, at, stroke=color, width=max(edge_width, 0.12 * size), clip=clip)
    return _scene.Markers(ops, xy, at, fill=color, stroke=edge, width=edge_width, clip=clip)


class Layer:
    label = None

    def extent(self, axis, positive):
        return None

    def draw(self, fr):
        raise NotImplementedError

    def entries(self):
        return []

    def strings(self):
        return []

    def sample(self, fr):
        return []


def _line_handle(color, width, dash):
    def handle(scene, x, y, w, th):
        scene.add(_scene.Path(*_scene.polyline([(x, y), (x + w, y)]), stroke=color, width=width, dash=dash))

    return handle


def _marker_handle(shape, size, color, edge, edge_width):
    def handle(scene, x, y, w, th):
        scene.add(marker_item(shape, size, [x + w / 2, y], color, edge, edge_width, None))

    return handle


def _patch_handle(color, edge=None):
    def handle(scene, x, y, w, th):
        scene.add(_scene.Path(*_scene.rect(x + 1.0, y - 3.0, w - 2.0, 6.0), fill=color, stroke=edge, width=0.4))

    return handle


class Line(Layer):
    def __init__(self, x, y, color, width, style, label, alpha=1.0):
        self.x, self.y, self.color, self.width, self.style, self.label = x, y, color, width, style, label

    def extent(self, axis, positive):
        r = finite_range(self.x if axis == "x" else self.y, positive)
        return None if r is None else (r[0], r[1], ())

    def draw(self, fr):
        ops, xy = polyline(fr.tx(self.x), fr.ty(self.y))
        if ops:
            w = fr.theme["line_width"] if self.width is None else self.width
            fr.scene.add(
                _scene.Path(ops, xy, stroke=self.color, width=w, dash=DASHES[self.style], join="round", clip=fr.clip)
            )

    def entries(self):
        return [
            (
                self.label,
                lambda s, x, y, w, th: _line_handle(
                    self.color, th["line_width"] if self.width is None else self.width, DASHES[self.style]
                )(s, x, y, w, th),
            )
        ]

    def sample(self, fr):
        return _sample(fr.tx(self.x), fr.ty(self.y))


class Scatter(Layer):
    def __init__(self, x, y, color, size, shape, label):
        self.x, self.y, self.color, self.size, self.shape, self.label = x, y, color, size, shape, label

    def extent(self, axis, positive):
        r = finite_range(self.x if axis == "x" else self.y, positive)
        return None if r is None else (r[0], r[1], ())

    def _size(self, th):
        return th["marker_size"] if self.size is None else self.size

    def draw(self, fr):
        at = points(fr.tx(self.x), fr.ty(self.y))
        if at:
            # markers are not cut at the plot's edge: a point on the edge shows whole
            edge, ew = self._edge(len(at) // 2, fr.theme)
            fr.scene.add(marker_item(self.shape, self._size(fr.theme), at, self.color, edge, ew, None))

    def _edge(self, n, th):
        """A thin white rim keeps neighbouring points apart; a cloud of thousands goes without
        (the rims would wash it out)."""
        if n > EDGE_UPTO or self._size(th) < 2.5:
            return None, 0.0
        return (1.0, 1.0, 1.0, self.color[3]), th["edge_width"]

    def entries(self):
        def handle(s, x, y, w, th):
            edge, ew = self._edge(1, th)
            _marker_handle(self.shape, self._size(th), self.color, edge, ew)(s, x, y, w, th)

        return [(self.label, handle)]

    def sample(self, fr):
        return _sample(fr.tx(self.x), fr.ty(self.y))


class Bars(Layer):
    """Rectangles from a base to a value at positions pos, each `width` wide (data units)."""

    def __init__(self, pos, value, base, width, horizontal, color, label):
        self.pos, self.value, self.base, self.width = pos, value, base, width
        self.horizontal, self.color, self.label = horizontal, color, label

    def extent(self, axis, positive):
        along = (axis == "y") == self.horizontal  # the axis the positions lie on
        if along:
            r = finite_range(self.pos, positive)
            return None if r is None else (r[0] - self.width / 2, r[1] + self.width / 2, ())
        r = finite_range(self.value, positive)
        if r is None:
            return None
        if positive:
            return (r[0], r[1], ())
        return (min(r[0], self.base), max(r[1], self.base), (self.base,))

    def draw(self, fr):
        half = self.width / 2
        base = self.base
        if (fr.xa if self.horizontal else fr.ya).scale == "log" and base <= 0.0:
            base = (fr.xa if self.horizontal else fr.ya).lo
        lo, hi = self.pos - half, self.pos + half
        v = self.value
        b = sd.full(v.shape, base)
        if self.horizontal:
            x0, x1, y0, y1 = fr.tx(b), fr.tx(v), fr.ty(lo), fr.ty(hi)
        else:
            x0, x1, y0, y1 = fr.tx(lo), fr.tx(hi), fr.ty(b), fr.ty(v)
        ok = bytes(memoryview(sd.isfinite(x0) & sd.isfinite(x1) & sd.isfinite(y0) & sd.isfinite(y1)))
        a0, a1, b0, b1 = (array("d", bytes(memoryview(a))) for a in (x0, x1, y0, y1))
        ops, xy = [], array("d")
        for i, good in enumerate(ok):
            if good:
                ops.append("MLLLZ")
                xy.extend((a0[i], b0[i], a1[i], b0[i], a1[i], b1[i], a0[i], b1[i]))
        if ops:
            fr.scene.add(
                _scene.Path(
                    "".join(ops),
                    xy,
                    fill=_color.fill_of(self.color),
                    stroke=self.color,
                    width=fr.theme["edge_width"],
                    clip=fr.clip,
                )
            )

    def entries(self):
        return [(self.label, _patch_handle(_color.fill_of(self.color), self.color))]

    def sample(self, fr):
        if self.horizontal:
            return _sample(fr.tx(self.value), fr.ty(self.pos))
        return _sample(fr.tx(self.pos), fr.ty(self.value))


class Area(Layer):
    def __init__(self, x, y1, y2, color, label):
        self.x, self.y1, self.y2, self.color, self.label = x, y1, y2, color, label

    def extent(self, axis, positive):
        if axis == "x":
            r = finite_range(self.x, positive)
            return None if r is None else (r[0], r[1], ())
        a, b = finite_range(self.y1, positive), finite_range(self.y2, positive)
        rs = [r for r in (a, b) if r is not None]
        return None if not rs else (min(r[0] for r in rs), max(r[1] for r in rs), ())

    def draw(self, fr):
        X, A, B = fr.tx(self.x), fr.ty(self.y1), fr.ty(self.y2)
        ok = bytes(memoryview(sd.isfinite(X) & sd.isfinite(A) & sd.isfinite(B)))
        top, bot = interleave(X, A), interleave(X, B)
        ops, xy = [], array("d")
        for m in _RUN.finditer(ok):
            a, b = m.span()
            if b - a < 2:
                continue
            ops.append("M" + "L" * (2 * (b - a) - 1) + "Z")
            xy.extend(top[2 * a : 2 * b])
            for i in range(b - 1, a - 1, -1):
                xy.extend(bot[2 * i : 2 * i + 2])
        if ops:
            fr.scene.add(_scene.Path("".join(ops), xy, fill=self.color, clip=fr.clip))

    def entries(self):
        return [(self.label, _patch_handle(self.color))]


class ErrorBars(Layer):
    """Markers with error bars: lo and hi arrays along y (and/or x), or None."""

    def __init__(self, x, y, ylo, yhi, xlo, xhi, color, cap, size, label):
        self.x, self.y, self.ylo, self.yhi, self.xlo, self.xhi = x, y, ylo, yhi, xlo, xhi
        self.color, self.cap, self.size, self.label = color, cap, size, label

    def extent(self, axis, positive):
        arrs = [self.x, self.xlo, self.xhi] if axis == "x" else [self.y, self.ylo, self.yhi]
        rs = [r for r in (finite_range(a, positive) for a in arrs if a is not None) if r is not None]
        return None if not rs else (min(r[0] for r in rs), max(r[1] for r in rs), ())

    def draw(self, fr):
        w = fr.theme["axis_width"]
        cap = self.cap
        ops, xy = [], array("d")
        X, Y = fr.tx(self.x), fr.ty(self.y)
        if self.ylo is not None:
            cols = [array("d", bytes(memoryview(a))) for a in (X, fr.ty(self.ylo), fr.ty(self.yhi))]
            for px, a, b in zip(*cols):
                if math.isfinite(px) and math.isfinite(a) and math.isfinite(b):
                    ops.append("ML" + ("MLML" if cap > 0 else ""))
                    xy.extend((px, a, px, b))
                    if cap > 0:
                        xy.extend((px - cap, a, px + cap, a, px - cap, b, px + cap, b))
        if self.xlo is not None:
            cols = [array("d", bytes(memoryview(a))) for a in (Y, fr.tx(self.xlo), fr.tx(self.xhi))]
            for py, a, b in zip(*cols):
                if math.isfinite(py) and math.isfinite(a) and math.isfinite(b):
                    ops.append("ML" + ("MLML" if cap > 0 else ""))
                    xy.extend((a, py, b, py))
                    if cap > 0:
                        xy.extend((a, py - cap, a, py + cap, b, py - cap, b, py + cap))
        if ops:
            fr.scene.add(_scene.Path("".join(ops), xy, stroke=self.color, width=w, clip=fr.clip))
        at = points(X, Y)
        if at:
            size = fr.theme["marker_size"] if self.size is None else self.size
            fr.scene.add(marker_item("o", size, at, self.color, None, 0.0, None))

    def entries(self):
        def handle(scene, x, y, w, th):
            scene.add(
                _scene.Path(
                    *_scene.polyline([(x + w / 2, y - 4.0), (x + w / 2, y + 4.0)]),
                    stroke=self.color,
                    width=th["axis_width"],
                )
            )
            scene.add(
                marker_item(
                    "o",
                    th["marker_size"] if self.size is None else self.size,
                    [x + w / 2, y],
                    self.color,
                    None,
                    0.0,
                    None,
                )
            )

        return [(self.label, handle)]

    def sample(self, fr):
        return _sample(fr.tx(self.x), fr.ty(self.y))


class Rule(Layer):
    """A horizontal or vertical line across the plot at a data value."""

    def __init__(self, value, vertical, color, width, style, label):
        self.value, self.vertical, self.color, self.width, self.style, self.label = (
            value,
            vertical,
            color,
            width,
            style,
            label,
        )

    def extent(self, axis, positive):
        if (axis == "x") != self.vertical or (positive and self.value <= 0.0):
            return None
        return (self.value, self.value, ())

    def draw(self, fr):
        w = fr.theme["axis_width"] if self.width is None else self.width
        if self.vertical:
            p = fr.px(self.value)
            pts = [(p, fr.y), (p, fr.y + fr.h)]
        else:
            p = fr.py(self.value)
            pts = [(fr.x, p), (fr.x + fr.w, p)]
        if math.isfinite(p):
            fr.scene.add(
                _scene.Path(*_scene.polyline(pts), stroke=self.color, width=w, dash=DASHES[self.style], clip=fr.clip)
            )

    def entries(self):
        return [
            (
                self.label,
                lambda s, x, y, w, th: _line_handle(
                    self.color, th["axis_width"] if self.width is None else self.width, DASHES[self.style]
                )(s, x, y, w, th),
            )
        ]


class Label(Layer):
    """A string at a data point; with an arrow to another point when xy is given."""

    def __init__(self, s, x, y, ha, va, size, color, rotation, target=None):
        self.s, self.x, self.y, self.ha, self.va, self.size, self.color = s, x, y, ha, va, size, color
        self.rotation, self.target = rotation, target

    def extent(self, axis, positive):
        vals = [self.x if axis == "x" else self.y]
        if self.target is not None:
            vals.append(self.target[0] if axis == "x" else self.target[1])
        vals = [v for v in vals if not positive or v > 0.0]
        return None if not vals else (min(vals), max(vals), ())

    def strings(self):
        return [self.s]

    def draw(self, fr):
        size = fr.theme["size"] if self.size is None else self.size
        px, py = fr.px(self.x), fr.py(self.y)
        if not (math.isfinite(px) and math.isfinite(py)):
            return
        if self.target is not None:
            tx, ty = fr.px(self.target[0]), fr.py(self.target[1])
            w, h, d = fr.engine.layout([self.s])[0].box(size)
            # the arrow leaves the text's box on the side facing the target
            dx, dy = tx - px, ty - py
            dist = math.hypot(dx, dy)
            if dist > 1e-9:
                ux, uy = dx / dist, dy / dist
                hw, hh = w / 2 + 2.0, (h + d) / 2 + 2.0
                t0 = min(hw / abs(ux) if ux else math.inf, hh / abs(uy) if uy else math.inf)
                if t0 < dist - 3.0:
                    sx, sy = px + ux * t0, py + uy * t0
                    ex, ey = tx - ux * 1.5, ty - uy * 1.5
                    head = 4.0
                    bx, by = ex - ux * head, ey - uy * head
                    fr.scene.add(
                        _scene.Path(
                            *_scene.polyline([(sx, sy), (bx, by)]), stroke=self.color, width=fr.theme["axis_width"]
                        )
                    )
                    fr.scene.add(
                        _scene.Path(
                            *_scene.polyline(
                                [(ex, ey), (bx - uy * 1.6, by + ux * 1.6), (bx + uy * 1.6, by - ux * 1.6)], closed=True
                            ),
                            fill=self.color,
                        )
                    )
            fr.scene.add(_scene.Text(self.s, px, py, size=size, ha="center", va="center", color=self.color))
        else:
            fr.scene.add(
                _scene.Text(self.s, px, py, size=size, ha=self.ha, va=self.va, rotation=self.rotation, color=self.color)
            )
