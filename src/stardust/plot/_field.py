"""Plots of functions of two variables (docs/api/plot.md §4.3, §4.4): implicit, region, contour,
density, phase.

The function is evaluated once on a grid (one call with two arrays). Level curves are traced by
marching squares: in every cell the level crosses, its crossings of the cell's edges are found by
linear interpolation and moved once more along the edge with a second evaluation (a secant
step), then the pieces are joined into curves. Regions are the cells' inside polygons, which
tile the region exactly. Densities and phase portraits are pictures of the grid's values.
"""

import math
from array import array

from .. import _core as sd
from .._core import _type_name
from .._core import plot as _core_plot
from ..optimize._common import _check_bool, _check_choice
from . import _color, _layers, _sample, _scene, _ticks
from ._basic import _alpha, _label, _opt_real, _style
from ._figure import integer, real
from ._function import _callable, _interval

NX, NY = 320, 240  # cells of the grid for curves and regions
FINE = 4  # a cell near a curve or a region's edge is sampled this many times finer along each direction
MAX_NEAR = 0.35
_PLACES = (0.5, 0.33, 0.67, 0.2, 0.8, 0.42, 0.58, 0.1, 0.9)  # along a level curve, where its number may stand
DX, DY = 480, 360  # samples of a density or a phase portrait


def call2(f, X, Y, who, arg="f"):
    out = f(X, Y)
    if isinstance(out, sd.ndarray):
        if tuple(out.shape) != tuple(X.shape):
            raise ValueError(
                f"{who}(): '{arg}' must return an array of its arguments' shape {tuple(X.shape)}, "
                f"not {tuple(out.shape)}"
            )
        if out.dtype == sd.complex128:
            raise TypeError(f"{who}(): '{arg}' must return real values, not complex128")
        return out if out.dtype == sd.float64 else out.astype(sd.float64)
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        return sd.full(tuple(X.shape), float(out))
    raise TypeError(f"{who}(): '{arg}' must return an sd.ndarray (it is called with two), not {_type_name(out)}")


def _box(who, x0, x1, y0, y1):
    x0, x1 = _interval(who, x0, x1, "x0", "x1")
    y0, y1 = _interval(who, y0, y1, "y0", "y1")
    return x0, x1, y0, y1


def grid(f, x0, x1, y0, y1, nx, ny, who):
    """(xs, ys, F): the corner coordinates (lists) and the values, F[j, i] = f(xs[i], ys[j])."""
    x, y = sd.linspace(x0, x1, nx + 1), sd.linspace(y0, y1, ny + 1)
    X, Y = sd.meshgrid(x, y, indexing="xy")
    return x.tolist(), y.tolist(), call2(f, X, Y, who)


def _stretch(F, axis):
    """A grid with FINE cells for each of its cells along an axis, the new values interpolated."""
    t = sd.linspace(0.0, 1.0, FINE + 1)[:-1]
    if axis == 1:
        a, b = F[:, :-1], F[:, 1:]
        mid = sd.reshape(a[:, :, None] * (1.0 - t) + b[:, :, None] * t, (F.shape[0], (F.shape[1] - 1) * FINE))
        return sd.concat([mid, F[:, -1:]], axis=1)
    return _stretch(F.T, 1).T


def _spread(mask):
    """Cells -> corners: the corners of the cells of a bool mask."""
    ny, nx = mask.shape
    out = sd.full((ny + 1, nx + 1), False)
    for dj in (0, 1):
        for di in (0, 1):
            out[dj : dj + ny, di : di + nx] |= mask
    return out


def near(f, xs, ys, F, level, who):
    """Samples again, FINE times finer along each direction, the band of cells the level passes
    through or close by (within the cell's own change of the function, and the cells around
    those) and the cells at the edge of the function's domain. Returns a finer (xs, ys, F):
    true values in the band, interpolated ones elsewhere, where only their side of the level
    matters. A curve that only touches the level between samples, a sharp corner or a crossing
    of two branches is drawn FINE times finer for the price of the band. Nothing is done when
    the band is empty or covers more than MAX_NEAR of the grid (a function rough everywhere)."""
    G = F - level
    fin = sd.isfinite(G)
    c = (G[:-1, :-1], G[:-1, 1:], G[1:, :-1], G[1:, 1:])
    k = (fin[:-1, :-1], fin[:-1, 1:], fin[1:, :-1], fin[1:, 1:])
    every, some = k[0] & k[1] & k[2] & k[3], k[0] | k[1] | k[2] | k[3]
    low = sd.minimum(sd.minimum(c[0], c[1]), sd.minimum(c[2], c[3]))
    high = sd.maximum(sd.maximum(c[0], c[1]), sd.maximum(c[2], c[3]))
    band = every & (low - (high - low) <= 0.0) & (high + (high - low) >= 0.0) | some & ~every
    return _again(f, xs, ys, F, band, who)


def swift(f, xs, ys, F, lv, who):
    """Samples again the cells where the function changes too fast for the levels lv: by more
    than half the distance between two levels within one cell (several level curves through a
    cell cannot be told apart), among the cells whose values are near the levels at all."""
    step = min(b - a for a, b in zip(lv, lv[1:]))
    fin = sd.isfinite(F)
    c = (F[:-1, :-1], F[:-1, 1:], F[1:, :-1], F[1:, 1:])
    k = (fin[:-1, :-1], fin[:-1, 1:], fin[1:, :-1], fin[1:, 1:])
    every, some = k[0] & k[1] & k[2] & k[3], k[0] | k[1] | k[2] | k[3]
    low = sd.minimum(sd.minimum(c[0], c[1]), sd.minimum(c[2], c[3]))
    high = sd.maximum(sd.maximum(c[0], c[1]), sd.maximum(c[2], c[3]))
    band = every & (high - low > 0.5 * step) & (high > lv[0] - step) & (low < lv[-1] + step) | some & ~every
    return _again(f, xs, ys, F, band, who)


def _again(f, xs, ys, F, band, who):
    """The grid FINE times finer, with true values in the cells of `band` and the cells around
    them and interpolated ones elsewhere (the grid itself when the band is empty or too wide)."""
    ny, nx = band.shape
    wide = sd.full((ny + 2, nx + 2), False)  # with the cells around
    for dj in (0, 1, 2):
        for di in (0, 1, 2):
            wide[dj : dj + ny, di : di + nx] |= band
    band = wide[1:-1, 1:-1]
    count = int(sd.sum(band))
    if count == 0 or count > MAX_NEAR * ny * nx:
        return xs, ys, F
    x = sd.reshape(_stretch(sd.reshape(sd.asarray(xs), (1, -1)), 1), (-1,))
    y = sd.reshape(_stretch(sd.reshape(sd.asarray(ys), (1, -1)), 1), (-1,))
    fine = _stretch(_stretch(F, 1), 0)
    nodes = _spread(sd.repeat(sd.repeat(band, FINE, axis=0), FINE, axis=1))
    jj, ii = sd.nonzero(nodes)
    n = jj.shape[0]
    fine[nodes] = sd.reshape(call2(f, sd.reshape(x[ii], (n, 1)), sd.reshape(y[jj], (n, 1)), who), (n,))
    return x.tolist(), y.tolist(), fine


class Grid:
    """A sampled grid for marching squares (csrc/plot/levels.cpp): made once, asked for any
    number of levels."""

    def __init__(self, xs, ys, F):
        self.core = _core_plot.Levels(
            bytes(memoryview(sd.reshape(F, (-1,)))), array("d", xs).tobytes(), array("d", ys).tobytes()
        )


class Cells:
    """The cells of a grid a level crosses, with the crossings of their edges. `grid` is the
    Grid of (xs, ys, F) when one was made already (many levels of one grid)."""

    def __init__(self, f, xs, ys, F, level, who, polish=True, grid=None):
        self.core, self.level = (grid or Grid(xs, ys, F)).core, level
        E = _doubles(self.core.edges(level))
        n = E.shape[0] // 8
        E = sd.reshape(E, (n, 8))
        px, py = E[:, 0], E[:, 1]
        if polish and n:
            # a secant step between the crossing estimate and the corner on the other side of the level
            v = call2(f, sd.reshape(px, (n, 1)), sd.reshape(py, (n, 1)), who)
            v = sd.reshape(v, (n,)) - level
            first = (E[:, 4] > 0.0) != (v > 0.0)  # the corner on the other side: the first, or the second
            cx, cy = sd.where(first, E[:, 2], E[:, 5]), sd.where(first, E[:, 3], E[:, 6])
            t = v / (v - sd.where(first, E[:, 4], E[:, 7]))
            use = sd.isfinite(v) & (v != 0.0) & (t >= 0.0) & (t <= 1.0)
            t = sd.where(use, t, 0.0)
            px, py = px + t * (cx - px), py + t * (cy - py)
        self.points = bytes(memoryview(sd.reshape(sd.stack([px, py], axis=1), (-1,)))) if n else b""

    def arrays(self):
        """The level set as two sd arrays with a NaN between curves; a closed curve repeats its
        first point."""
        out = _doubles(self.core.curves(self.level, self.points))
        return out[0::2], out[1::2]

    def curves(self):
        """The level set as polylines: lists of (x, y)."""
        X, Y = self.arrays()
        out, cur = [], []
        for x, y in zip(X.tolist(), Y.tolist()):
            if x != x:
                out.append(cur)
                cur = []
            else:
                cur.append((x, y))
        if cur:
            out.append(cur)
        return out

    def inside(self):
        """(ops, xy) of the region where the function is above the level: rectangles for runs of
        cells wholly inside, and the inside polygon of every cell the level crosses."""
        ops, xy = self.core.inside(self.level, self.points)
        return ops.decode(), array("d", xy)


def _doubles(b):
    return sd.asarray(array("d", b)) if b else sd.zeros(0)


class Fill(_layers.Layer):
    """Polygons in data coordinates, filled (a region)."""

    def __init__(self, ops, xy, box, color, label):
        self.ops, self.xy, self.box, self.color, self.label = ops, xy, box, color, label

    def extent(self, axis, positive):
        lo, hi = (self.box[0], self.box[1]) if axis == "x" else (self.box[2], self.box[3])
        return (lo, hi, (lo, hi))

    def draw(self, fr):
        if not self.ops:
            return
        pts = sd.asarray(self.xy)
        n = pts.shape[0] // 2
        P = sd.reshape(pts, (n, 2))
        X, Y = fr.tx(P[:, 0]), fr.ty(P[:, 1])
        out = array("d")
        out.frombytes(bytes(memoryview(sd.reshape(sd.stack([X, Y], axis=1), (-1,)))))
        fr.scene.add(_scene.Path(self.ops, out, fill=self.color, clip=fr.clip))

    def entries(self):
        return [(self.label, _layers._patch_handle(self.color))]


class Picture(_layers.Layer):
    """An image over a rectangle of data coordinates (row 0 is the top)."""

    def __init__(self, box, iw, ih, rgba, smooth):
        self.box, self.iw, self.ih, self.rgba, self.smooth = box, iw, ih, rgba, smooth

    def extent(self, axis, positive):
        lo, hi = (self.box[0], self.box[1]) if axis == "x" else (self.box[2], self.box[3])
        return (lo, hi, (lo, hi))

    def draw(self, fr):
        x0, x1, y0, y1 = self.box
        px0, px1, py0, py1 = fr.px(x0), fr.px(x1), fr.py(y1), fr.py(y0)
        fr.scene.add(
            _scene.Image(px0, py0, px1 - px0, py1 - py0, self.iw, self.ih, self.rgba, smooth=self.smooth, clip=fr.clip)
        )


class ContourLabels(_layers.Layer):
    """A number on each level curve, on a white patch that interrupts the line. Each level
    offers several places along its longest curve; the first that is inside the plot and clear
    of the numbers already set is taken, and a level with no such place goes without."""

    def __init__(self, items, size):
        self.items, self.size = items, size  # (string, [(x, y, the curve's direction dx, dy)])

    def strings(self):
        return [s for s, _ in self.items]

    def draw(self, fr):
        taken = []  # (x, y, radius) of the numbers set
        for s, places in self.items:
            w, h, d = fr.engine.layout([s])[0].box(self.size)
            hw, hh = w / 2 + 1.5, (h + d) / 2 + 1.0
            rad = math.hypot(hw, hh)
            for place in places:
                px, py = fr.px(place[0]), fr.py(place[1])
                inside = fr.x + rad <= px <= fr.x + fr.w - rad and fr.y + rad <= py <= fr.y + fr.h - rad
                if inside and all(math.hypot(px - qx, py - qy) >= rad + qr + 2.0 for qx, qy, qr in taken):
                    break
            else:
                continue
            x, y, dx, dy = place
            taken.append((px, py, rad))
            ax, ay = fr.px(x + dx) - px, fr.py(y + dy) - py
            ang = math.degrees(math.atan2(-ay, ax)) if (ax or ay) else 0.0
            if ang > 90.0:
                ang -= 180.0
            elif ang < -90.0:
                ang += 180.0
            a = math.radians(ang)
            c, sn = math.cos(a), math.sin(a)
            quad = [
                (px + c * ex + sn * ey, py - sn * ex + c * ey)
                for ex, ey in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))
            ]
            fr.scene.add(_scene.Path(*_scene.polyline(quad, closed=True), fill=(1.0, 1.0, 1.0, 1.0)))
            fr.scene.add(_scene.Text(s, px, py, size=self.size, ha="center", va="center", rotation=ang))


def _parse(self, who, color):
    palette = self._root._palette
    return _color.parse(color, who, "color", palette) if color is not None else self._cell.next_color(palette)


def implicit(self, f, x0, x1, y0, y1, /, *, level=0.0, color=None, width=None, style="-", label=None):
    who = "implicit"
    _callable(f, who, "f")
    x0, x1, y0, y1 = _box(who, x0, x1, y0, y1)
    level = real(level, who, "level")
    width, style, label = _opt_real(who, "width", width), _style(who, style), _label(who, label)
    xs, ys, F = grid(f, x0, x1, y0, y1, NX, NY, who)
    X, Y = Cells(f, *near(f, xs, ys, F, level, who), level, who).arrays()
    layer = _layers.Line(X, Y, _parse(self, who, color), width, style, label)
    layer.extent = lambda axis, positive: (x0, x1, (x0, x1)) if axis == "x" else (y0, y1, (y0, y1))
    self._cell.layers.append(layer)
    self._cell.aspect = 1.0
    return self._touch()


def region(self, f, x0, x1, y0, y1, /, *, color=None, alpha=0.3, label=None):
    who = "region"
    _callable(f, who, "f")
    x0, x1, y0, y1 = _box(who, x0, x1, y0, y1)
    a, label = _alpha(who, alpha), _label(who, label)
    xs, ys, F = grid(f, x0, x1, y0, y1, NX, NY, who)
    ops, xy = Cells(f, *near(f, xs, ys, F, 0.0, who), 0.0, who).inside()
    self._cell.layers.append(Fill(ops, xy, (x0, x1, y0, y1), _color.with_alpha(_parse(self, who, color), a), label))
    self._cell.aspect = 1.0
    return self._touch()


def _levels(who, levels, lo, hi):
    if isinstance(levels, bool) or not isinstance(levels, (int, list, tuple)):
        raise TypeError(f"{who}(): 'levels' must be an int or a sequence of numbers, not {_type_name(levels)}")
    if isinstance(levels, int):
        n = integer(levels, who, "levels", 1)
        if not lo < hi:
            return [lo]
        return [v for v in _ticks.linear(lo, hi, max(n, 2)) if lo < v < hi] or [0.5 * (lo + hi)]
    out = sorted(real(v, who, "levels") for v in levels)
    if not out:
        raise ValueError(f"{who}(): 'levels' is empty")
    return out


def _cmap(who, cmap, default="viridis"):
    return default if cmap is None else _check_choice(cmap, who, "cmap", tuple(_color.CMAPS))


def contour(self, f, x0, x1, y0, y1, /, *, levels=10, filled=False, labels=True, cmap=None, color=None):
    who = "contour"
    _callable(f, who, "f")
    x0, x1, y0, y1 = _box(who, x0, x1, y0, y1)
    _check_bool(filled, who, "filled")
    _check_bool(labels, who, "labels")
    if color is not None and cmap is not None:
        raise ValueError(f"{who}(): 'color' and 'cmap' cannot both be given")
    one = _color.parse(color, who, "color", self._root._palette) if color is not None else None
    name = _cmap(who, cmap)
    xs, ys, F = grid(f, x0, x1, y0, y1, NX, NY, who)
    r = _layers.finite_range(sd.reshape(F, (-1,)))
    if r is None:
        raise ValueError(f"{who}(): 'f' has no finite value on the grid")
    # the levels are spread over the bulk of the values: a pole does not take them all
    rr = _sample.robust_range(sd.reshape(F, (-1,)))
    lv = _levels(who, levels, rr[0], rr[1])
    if len(lv) > 1:
        xs, ys, F = swift(f, xs, ys, F, lv, who)
    whole = Grid(xs, ys, F)
    box = (x0, x1, y0, y1)
    span = (lv[-1] - lv[0]) or 1.0
    cell = self._cell

    def shade(v):
        return _color.cmap(name, (v - lv[0]) / span) if len(lv) > 1 else _color.cmap(name, 0.5)

    if filled:
        # bands painted from the lowest up: each level's region covers the part above it
        step = lv[1] - lv[0] if len(lv) > 1 else span
        cell.layers.append(
            Fill("MLLLZ", array("d", (x0, y0, x1, y0, x1, y1, x0, y1)), box, shade(lv[0] - step / 2), None)
        )
        for k, v in enumerate(lv):
            ops, xy = Cells(f, xs, ys, F, v, who, grid=whole).inside()
            top = lv[k + 1] if k + 1 < len(lv) else v + step
            cell.layers.append(Fill(ops, xy, box, shade((v + top) / 2), None))
    text, _ = _ticks.labels(lv)
    marks = []
    for k, v in enumerate(lv):
        cells = Cells(f, xs, ys, F, v, who, grid=whole)
        X, Y = cells.arrays()
        if X.shape[0] == 0:
            continue
        c = one if one is not None else (_scene.INK if filled else shade(v))
        layer = _layers.Line(X, Y, c, 0.5 if filled else 0.8, "-", None)
        layer.extent = lambda axis, positive: (x0, x1, (x0, x1)) if axis == "x" else (y0, y1, (y0, y1))
        cell.layers.append(layer)
        if labels and not filled:
            pts = max(cells.curves(), key=len)
            if len(pts) >= 12:
                places = []
                for t in _PLACES if k % 2 == 0 else _PLACES[1:] + _PLACES[:1]:
                    m = int(t * (len(pts) - 1))
                    (xa, ya), (xb, yb) = pts[max(m - 2, 0)], pts[min(m + 2, len(pts) - 1)]
                    places.append((pts[m][0], pts[m][1], xb - xa, yb - ya))
                marks.append((text[k], places))
    if filled and len(lv) > 1:
        cell.colorbar = (name, lv[0], lv[-1])  # filled bands are read from a colour bar, not from numbers on them
    if marks:
        cell.layers.append(ContourLabels(marks, 6.0))
    return self._touch()


def _picture(F, vmin, vmax, table):
    """RGBA bytes of a grid of values (row 0 the lowest y) with the rows turned top down."""
    flipped = sd.flip(F, axis=0)
    return _core_plot.colormap(bytes(memoryview(sd.reshape(flipped, (-1,)))), vmin, vmax, table)


def density(self, f, x0, x1, y0, y1, /, *, cmap=None, vmin=None, vmax=None, colorbar=True):
    who = "density"
    _callable(f, who, "f")
    x0, x1, y0, y1 = _box(who, x0, x1, y0, y1)
    name = _cmap(who, cmap)
    _check_bool(colorbar, who, "colorbar")
    # the samples are the centres of the picture's pixels
    hx, hy = (x1 - x0) / DX, (y1 - y0) / DY
    x = sd.linspace(x0 + hx / 2, x1 - hx / 2, DX)
    y = sd.linspace(y0 + hy / 2, y1 - hy / 2, DY)
    X, Y = sd.meshgrid(x, y, indexing="xy")
    F = call2(f, X, Y, who)
    r = _layers.finite_range(sd.reshape(F, (-1,))) or (0.0, 1.0)
    lo = r[0] if vmin is None else real(vmin, who, "vmin")
    hi = r[1] if vmax is None else real(vmax, who, "vmax")
    if not lo < hi:
        if vmin is not None and vmax is not None:
            raise ValueError(f"{who}(): 'vmin' must be less than 'vmax'")
        lo, hi = lo - 0.5, lo + 0.5
    self._cell.layers.append(Picture((x0, x1, y0, y1), DX, DY, _picture(F, lo, hi, _color.table(name)), True))
    if colorbar:
        self._cell.colorbar = (name, lo, hi)
    return self._touch()


def phase(self, f, re0, re1, im0, im1, /, *, contours=True):
    who = "phase"
    _callable(f, who, "f")
    x0, x1 = _interval(who, re0, re1, "re0", "re1")
    y0, y1 = _interval(who, im0, im1, "im0", "im1")
    _check_bool(contours, who, "contours")
    hx, hy = (x1 - x0) / DX, (y1 - y0) / DY
    x = sd.linspace(x0 + hx / 2, x1 - hx / 2, DX)
    y = sd.linspace(y1 - hy / 2, y0 + hy / 2, DY)  # the top row first
    X, Y = sd.meshgrid(x, y, indexing="xy")
    Z = X + Y * 1j
    W = f(Z)
    if not isinstance(W, sd.ndarray) or tuple(W.shape) != tuple(Z.shape):
        raise TypeError(
            f"{who}(): 'f' must return an sd.ndarray of its argument's shape {tuple(Z.shape)} "
            "(it is called with a complex128 array)"
        )
    if W.dtype != sd.complex128:
        W = W.astype(sd.complex128)
    re_, im_ = sd.real(W), sd.imag(W)
    rgba = _core_plot.phase_colors(
        bytes(memoryview(sd.reshape(re_, (-1,)))), bytes(memoryview(sd.reshape(im_, (-1,)))), contours
    )
    self._cell.layers.append(Picture((x0, x1, y0, y1), DX, DY, rgba, True))
    self._cell.aspect = 1.0
    return self._touch()


METHODS = {"implicit": implicit, "region": region, "contour": contour, "density": density, "phase": phase}
