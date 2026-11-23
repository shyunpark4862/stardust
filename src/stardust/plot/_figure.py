"""Figure: cells with axes, their layers, the layout and the outputs (docs/api/plot.md §2,
docs/design/plot.md §3.3, §3.4).

A Figure holds a grid of cells. A cell holds layers (what the plotting functions add: _layers.py)
and the axes' settings. Nothing is drawn until an output is asked for; then the figure is laid
out once (limits, ticks, the margins the measured text needs) into a scene, which the SVG, PDF
and PNG writers read. A method that changes the figure drops the scene.

Lengths the user gives are mm; everything inside is bp (1/72 inch), as the scene.
"""

import math
import numbers
import operator
import pathlib
from array import array

from .. import _core as sd
from .._core import _type_name
from ..optimize._common import _check_choice
from . import _color, _pdf, _png, _scene, _svg, _text, _ticks

MM = 72.0 / 25.4
INK = _scene.INK
WHITE = (1.0, 1.0, 1.0, 1.0)

# A theme: sizes in bp. 'paper' is the default (docs/design/plot.md §2.3): the left and bottom
# axis lines only, ticks outward, no grid. 'boxed' draws all four lines; 'grid' adds a light grid.
THEMES = {
    "paper": {
        "size": 8.0,
        "tick_size": 7.0,
        "title_size": 9.0,
        "legend_size": 7.0,
        "panel_size": 10.0,
        "axis_width": 0.6,
        "tick_len": 2.8,
        "tick_pad": 2.0,
        "label_pad": 3.5,
        "title_pad": 4.0,
        "line_width": 1.2,
        "marker_size": 3.8,
        "edge_width": 0.5,
        "margin": 3.0,
        "gap": 6.0,
        "spines": ("left", "bottom"),
        "grid": False,
        "pad": 0.04,
    },
}
THEMES["boxed"] = {**THEMES["paper"], "spines": ("left", "bottom", "right", "top")}
THEMES["grid"] = {**THEMES["paper"], "grid": True}
GRID_COLOR = (0.9, 0.9, 0.9, 1.0)
SCALES = ("linear", "log")
LEGEND_LOCS = ("best", "top", "right", "bottom", "upper right", "upper left", "lower left", "lower right")
DASHES = {"-": (), "--": (4.0, 2.0), ":": (1.0, 1.6), "-.": (4.0, 1.6, 1.0, 1.6)}


def real(v, who, arg, *, positive=False, finite=True):
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{who}(): '{arg}' must be an int or a float, not {_type_name(v)}")
    v = float(v)
    if finite and not math.isfinite(v):
        raise ValueError(f"{who}(): '{arg}' must be finite")
    if positive and not v > 0.0:
        raise ValueError(f"{who}(): '{arg}' must be positive")
    return v


def string(v, who, arg):
    if not isinstance(v, str):
        raise TypeError(f"{who}(): '{arg}' must be a str, not {_type_name(v)}")
    return v


def integer(v, who, arg, lo):
    if isinstance(v, bool) or not isinstance(v, numbers.Integral):
        raise TypeError(f"{who}(): '{arg}' must be an int, not {_type_name(v)}")
    v = operator.index(v)
    if v < lo:
        raise ValueError(f"{who}(): '{arg}' must be at least {lo}")
    return v


class Cell:
    """One set of axes."""

    def __init__(self):
        self.layers = []
        self.title = self.xlabel = self.ylabel = self.zlabel = None
        self.xlim = [None, None]
        self.ylim = [None, None]
        self.xscale = self.yscale = "linear"
        self.xticks = self.yticks = None  # None: automatic; (values, labels or None); 'pi'
        self.xcats = self.ycats = None  # category labels of a category axis
        self.legend = {"loc": "best", "title": None, "ncols": 1}
        self.aspect = None  # 1.0: equal data units
        self.ncolor = 0  # colours of the palette taken so far
        self.frame = True  # False: no axes at all (a 3D cell draws its own)
        self.colorbar = None  # (map name, lowest value, highest value) of a density
        self.headroom = 0  # brackets stacked above the data (compare)
        self.note = None  # a line of text inside the plot, in a free corner (a fit's equation)
        self.placed = None  # where a 'best' legend goes, found by the layout
        self.span = [None, None]  # x and y ranges shared with other cells (grid's sharex, sharey; the layout's)
        self.inner = [False, False]  # x and y tick labels left to a neighbouring cell (shared axes)
        self.twin = None  # a cell drawn against a second y axis at the right (twin()), sharing x
        self.insets = []  # (cell, (x, y, w, h) as shares of the plot rectangle) drawn over it (inset())

    def copy(self):
        """A cell of its own with the same content: its settings and layers are copied, so that
        changing one of the two leaves the other as it was (grid)."""
        import copy

        out = copy.copy(self)
        out.xlim, out.ylim, out.legend = list(self.xlim), list(self.ylim), dict(self.legend)
        out.layers = []
        for layer in self.layers:
            twin = copy.copy(layer)
            if hasattr(twin, "fork"):
                twin.fork(out)  # a layer with lists of its own (a 3D space) copies them
            out.layers.append(twin)
        if self.twin is not None:
            out.twin = self.twin.copy()
        out.insets = [(c.copy(), rect) for c, rect in self.insets]
        return out

    def next_color(self, palette):
        c = _color.cycle(palette, self.ncolor)
        self.ncolor += 1
        return c


class _Axis:
    """A laid-out axis: its range, scale and ticks."""

    __slots__ = ("lo", "hi", "scale", "ticks", "labels", "minor", "mult", "cats")

    def frac(self, v):
        """Where a data value (a float) falls in [0, 1] along the axis."""
        if self.scale == "log":
            return (
                (math.log10(v) - math.log10(self.lo)) / (math.log10(self.hi) - math.log10(self.lo))
                if v > 0
                else math.nan
            )
        return (v - self.lo) / (self.hi - self.lo)


class Frame:
    """What a layer needs to draw: the scene, the data-to-page maps and the plot rectangle."""

    def __init__(self, scene, x, y, w, h, xa, ya, theme, palette, engine, dpi=300.0):
        self.scene, self.x, self.y, self.w, self.h = scene, x, y, w, h
        self.dpi = dpi  # of the pictures a layer makes (a 3D scene)
        self.xa, self.ya, self.theme, self.palette, self.engine = xa, ya, theme, palette, engine
        self.clip = (x, y, w, h)

    def px(self, v):
        return self.x + self.xa.frac(v) * self.w

    def py(self, v):
        return self.y + self.h - self.ya.frac(v) * self.h

    def tx(self, a):
        """An sd array of x data as page coordinates."""
        xa = self.xa
        if xa.scale == "log":
            l0, l1 = math.log10(xa.lo), math.log10(xa.hi)
            return (sd.log10(sd.where(a > 0.0, a, math.nan)) - l0) * (self.w / (l1 - l0)) + self.x
        return (a - xa.lo) * (self.w / (xa.hi - xa.lo)) + self.x

    def ty(self, a):
        ya = self.ya
        if ya.scale == "log":
            l0, l1 = math.log10(ya.lo), math.log10(ya.hi)
            return (self.y + self.h) - (sd.log10(sd.where(a > 0.0, a, math.nan)) - l0) * (self.h / (l1 - l0))
        return (self.y + self.h) - (a - ya.lo) * (self.h / (ya.hi - ya.lo))


class Figure:
    def __init__(self, *, width=89.0, height=None, nrows=1, ncols=1, font="serif", theme="paper", palette="jco"):
        self._init("Figure", width, height, nrows, ncols, font, theme, palette)

    def _init(self, who, width, height, nrows, ncols, font, theme, palette):
        self._root, self._index = self, 0
        self._w = real(width, who, "width", positive=True)
        self._h = None if height is None else real(height, who, "height", positive=True)
        self._nrows, self._ncols = integer(nrows, who, "nrows", 1), integer(ncols, who, "ncols", 1)
        self._font = _check_choice(font, who, "font", _text.FAMILIES)
        self._theme = _check_choice(theme, who, "theme", tuple(THEMES))
        self._palette = _check_choice(palette, who, "palette", tuple(_color.PALETTES))
        self._cells = [Cell() for _ in range(self._nrows * self._ncols)]
        self._panels = None  # panel labels, a str per cell
        self._share = (False, False)  # sharex, sharey of a grid: False, True or 'all'
        self._scene = None
        self._infos = []
        return self

    # ---- views and attributes

    def _view(self, index):
        v = object.__new__(Figure)
        v._root, v._index = self._root, index
        return v

    @property
    def _base(self):
        """The cell itself: what the title, the x axis, the legend and the insets belong to."""
        return self._root._cells[self._index]

    @property
    def _cell(self):
        """The cell the plotting methods and the y settings act on: of a twin() view, the cell of
        the second y axis."""
        c = self._base
        return c.twin if self.__dict__.get("_twinned") else c

    def inset(self, figure, rect, /):
        """Draws the one cell of `figure` over this cell's plot, in `rect` = (x, y, w, h): shares
        of the plot rectangle from its lower left corner (its ticks and labels inside the rect)."""
        if not isinstance(figure, Figure) or len(figure._root._cells) != 1:
            raise TypeError(f"inset(): 'figure' must be a Figure of one cell, not {_type_name(figure)}")
        if not isinstance(rect, (tuple, list)):
            raise TypeError(f"inset(): 'rect' must be (x, y, w, h), not {_type_name(rect)}")
        if len(rect) != 4:
            raise ValueError(f"inset(): 'rect' must hold 4 numbers (x, y, w, h), not {len(rect)}")
        x, y, w, h = (real(v, "inset", "rect") for v in rect)
        if not (0.0 <= x and 0.0 <= y and 0.0 < w and 0.0 < h and x + w <= 1.0 + 1e-9 and y + h <= 1.0 + 1e-9):
            raise ValueError("inset(): 'rect' must lie in the plot: shares in [0, 1] with x + w and y + h at most 1")
        cell = self._base  # (of a twin() view too: the insets are the cell's)
        if not cell.frame:
            raise ValueError("inset(): an inset goes over a 2D plot's cell")
        cell.insets.append((figure._root._cells[0].copy(), (x, y, w, h)))
        return self._touch()

    def twin(self):
        """A view of the cell that draws against a second y axis at the right: its plotting
        methods and ylabel, ylim, yscale, yticks set that axis; the x axis is shared, so title,
        xlabel, xlim, xscale, xticks, legend and inset of the view are the cell's own."""
        if self.__dict__.get("_twinned"):
            raise ValueError("twin(): this is already the second y axis")
        cell = self._cell
        if not cell.frame or any(type(layer).__name__ == "Space" for layer in cell.layers):
            raise ValueError("twin(): a second y axis needs a 2D plot's cell")
        if cell.twin is None:
            cell.twin = Cell()
            cell.twin.ncolor = cell.ncolor  # the colours go on where the first axis's left off
        v = self._view(self._index)
        v._twinned = True
        return v

    def _touch(self):
        self._root._scene = None
        return self

    def __getitem__(self, key, /):
        r = self._root
        if r is not self:
            raise IndexError("a cell of a figure has no cells of its own")
        nr, nc = r._nrows, r._ncols
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError(f"a figure takes one or two indices, not {len(key)}")
            i, j = key
        elif nr == 1 or nc == 1:
            i, j = (0, key) if nr == 1 else (key, 0)
        else:
            raise IndexError(f"a figure of {nr} x {nc} cells takes two indices")
        out = []
        for v, n in ((i, nr), (j, nc)):
            if isinstance(v, bool) or not isinstance(v, numbers.Integral):
                raise IndexError(f"figure indices must be ints, not {_type_name(v)}")
            v = operator.index(v)
            if not -n <= v < n:
                raise IndexError(f"index {v} is out of bounds for a figure of {nr} x {nc} cells")
            out.append(v % n)
        return self._view(out[0] * nc + out[1])

    @property
    def width(self):
        return self._root._w

    @property
    def height(self):
        """The height in mm; of a figure made without one, the height its layout gives."""
        r = self._root
        return r._h if r._h is not None else r._build("height").height / MM

    @property
    def shape(self):
        return (self._root._nrows, self._root._ncols)

    @property
    def axes(self):
        r = self._root
        return tuple(r._view(i) for i in range(len(r._cells)))

    @property
    def info(self):
        """What the function plots did, a record per layer (docs/api/plot.md §4.6)."""
        return tuple(self._root._infos)

    def __repr__(self):
        r = self._root
        n = sum(len(c.layers) for c in r._cells)
        h = f"{r._h:g}" if r._h is not None else "auto"
        return f"Figure({r._w:g} x {h} mm, {len(r._cells)} axes, {n} layer{'s' if n != 1 else ''})"

    # ---- settings; each returns the figure (docs/api/plot.md §2.2)

    def title(self, text, /):
        self._base.title = string(text, "title", "text")
        return self._touch()

    def xlabel(self, text, /):
        self._base.xlabel = string(text, "xlabel", "text")
        return self._touch()

    def ylabel(self, text, /):
        self._cell.ylabel = string(text, "ylabel", "text")
        return self._touch()

    def zlabel(self, text, /):
        self._cell.zlabel = string(text, "zlabel", "text")
        return self._touch()

    def _lim(self, who, low, high):
        lo = None if low is None else real(low, who, "low")
        hi = None if high is None else real(high, who, "high")
        if lo is not None and hi is not None and not lo < hi:
            raise ValueError(f"{who}(): 'low' must be less than 'high'")
        return [lo, hi]

    def _flat(self, who):
        """Settings of a plot's axes, which a 3D cell does not have: ValueError there."""
        if any(type(layer).__name__ == "Space" for layer in self._cell.layers):
            raise ValueError(f"{who}(): a 3D cell has no such axis setting; its ranges are those of the data")

    def xlim(self, low, high, /):
        self._flat("xlim")
        self._base.xlim = self._lim("xlim", low, high)
        return self._touch()

    def ylim(self, low, high, /):
        self._flat("ylim")
        self._cell.ylim = self._lim("ylim", low, high)
        return self._touch()

    def xscale(self, kind, /):
        self._flat("xscale")
        self._base.xscale = _check_choice(kind, "xscale", "kind", SCALES)
        return self._touch()

    def yscale(self, kind, /):
        self._flat("yscale")
        self._cell.yscale = _check_choice(kind, "yscale", "kind", SCALES)
        return self._touch()

    def _ticks(self, who, values, labels):
        if isinstance(values, str):
            if values != "pi":
                raise ValueError(f"{who}(): 'values' must be numbers or 'pi', not {values!r}")
            if labels is not None:
                raise ValueError(f"{who}(): 'labels' cannot be given with values='pi'")
            return "pi"
        try:
            vals = [real(v, who, "values") for v in values]
        except TypeError:
            raise TypeError(
                f"{who}(): 'values' must be a sequence of numbers or 'pi', not {_type_name(values)}"
            ) from None
        if labels is not None:
            labels = [string(s, who, "labels") for s in labels]
            if len(labels) != len(vals):
                raise ValueError(
                    f"{who}(): 'values' and 'labels' must have the same length: {len(vals)} and {len(labels)}"
                )
        return (vals, labels)

    def xticks(self, values, /, *, labels=None):
        self._flat("xticks")
        self._base.xticks = self._ticks("xticks", values, labels)
        return self._touch()

    def yticks(self, values, /, *, labels=None):
        self._flat("yticks")
        self._cell.yticks = self._ticks("yticks", values, labels)
        return self._touch()

    def legend(self, *, loc="best", title=None, ncols=1):
        if loc is not None:
            _check_choice(loc, "legend", "loc", LEGEND_LOCS)
        if title is not None:
            string(title, "legend", "title")
        self._base.legend = {"loc": loc, "title": title, "ncols": integer(ncols, "legend", "ncols", 1)}
        return self._touch()

    def size(self, width, height=None, /):
        r = self._root
        r._w = real(width, "size", "width", positive=True)
        r._h = None if height is None else real(height, "size", "height", positive=True)
        return self._touch()

    # ---- outputs (docs/api/plot.md §2.3, §2.4)

    def _build(self, who, dpi=300.0):
        r = self._root
        # a figure with a 3D cell holds a picture made for one resolution
        if r._scene is not None and getattr(r, "_scene_dpi", dpi) != dpi and any(not c.frame for c in r._cells):
            r._scene = None
        if r._scene is None:
            try:
                r._scene = _layout(r, dpi)
                r._scene_dpi = dpi
            except _text.TypesetError as e:
                raise ValueError(f"{who}(): {e}") from None
            except _text.MissingTool as e:
                raise RuntimeError(f"{who}(): {e}") from None
        return r._scene

    def _write(self, who, fn, *args, **kw):
        scene = self._build(who, args[0] if args else 300.0)
        try:
            return fn(scene, *args, **kw)
        except _text.TypesetError as e:
            raise ValueError(f"{who}(): {e}") from None
        except _text.MissingTool as e:
            raise RuntimeError(f"{who}(): {e}") from None

    def scene(self):
        return self._build("scene").to_dict()

    def to_svg(self):
        return self._write("to_svg", _svg.write)

    def to_pdf(self):
        return self._write("to_pdf", _pdf.write)

    def to_png(self, *, dpi=300):
        return self._write("to_png", _png.write, real(dpi, "to_png", "dpi", positive=True))

    def save(self, path, /, *, dpi=300):
        if not isinstance(path, (str, pathlib.PurePath)):
            raise TypeError(f"save(): 'path' must be a str or a path, not {_type_name(path)}")
        dpi = real(dpi, "save", "dpi", positive=True)
        p = pathlib.Path(path)
        ext = p.suffix.lower()
        if ext == ".pdf":
            data = self._write("save", _pdf.write, dpi)
        elif ext == ".svg":
            data = self._write("save", _svg.write, dpi).encode()
        elif ext == ".png":
            data = self._write("save", _png.write, dpi)
        else:
            raise ValueError(f"save(): 'path' must end in '.pdf', '.svg' or '.png', not {p.suffix!r}")
        p.write_bytes(data)
        return self

    def _pictured(self):
        """Whether the figure holds a 3D cell or a layer drawn as a picture (a density, a phase
        portrait, a heatmap, a hist2d): a notebook shows it as PNG, the rest as SVG."""
        for cell in self._root._cells:
            for layer in cell.layers:
                if type(layer).__name__ in ("Space", "Picture", "Counts"):
                    return True
        return False

    def _repr_mimebundle_(self, include=None, exclude=None):
        """The notebook's view: one image, PNG at 200 dpi or SVG (docs/api/plot.md §2.4), and
        the one-line repr as plain text. Only one of the two is made."""
        if self._pictured():
            data = {"image/png": self._write("Figure", _png.write, 200.0, level=1)}
        else:
            data = {"image/svg+xml": self._write("Figure", _svg.write, 200.0)}
        data["text/plain"] = repr(self)
        return data


# ---- layout


def _extent(cell, axis, scale):
    """(lo, hi, sticky values) of the layers' data along an axis, or None (x: with the layers of
    a second y axis, which share it)."""
    lo, hi, sticky = math.inf, -math.inf, set()
    layers = cell.layers + (cell.twin.layers if axis == "x" and cell.twin is not None else [])
    for layer in layers:
        e = layer.extent(axis, scale == "log")
        if e is None:
            continue
        lo, hi = min(lo, e[0]), max(hi, e[1])
        sticky.update(e[2])
    return (lo, hi, sticky) if lo <= hi else None


def _range(cell, axis, th):
    """The axis range: the user's limits, else the data's extent padded (not past a sticky
    value such as a bar's base)."""
    lim = cell.xlim if axis == "x" else cell.ylim
    scale = cell.xscale if axis == "x" else cell.yscale
    cats = cell.xcats if axis == "x" else cell.ycats
    lo, hi = lim
    span = cell.span[0 if axis == "x" else 1]
    if span is not None and lo is None and hi is None:
        return span
    if lo is not None and hi is not None:
        if scale == "log" and lo <= 0.0:
            raise ValueError(f"{axis}lim(): a log axis needs positive limits")
        return lo, hi
    if cats is not None:
        a, b = -0.6, len(cats) - 0.4
    else:
        e = _extent(cell, axis, scale)
        if e is None:
            a, b = (1.0, 10.0) if scale == "log" else (0.0, 1.0)
        else:
            a, b, sticky = e
            pad = th["pad"]
            if scale == "log":
                la, lb = math.log10(a), math.log10(b)
                if la == lb:
                    la, lb = la - 0.5, lb + 0.5
                d = (lb - la) * pad
                a, b = 10.0 ** (la - d), 10.0 ** (lb + d)
            else:
                if a == b:
                    d = 0.5 if a == 0.0 else abs(a) * 0.05
                    a, b = a - d, b + d
                    sticky = set()
                d = (b - a) * pad
                a, b = (a if a in sticky else a - d), (b if b in sticky else b + d)
    lo = a if lo is None else lo
    hi = b if hi is None else hi
    if not lo < hi:
        lo, hi = (lo, lo + 1.0) if lim[0] is not None else (hi - 1.0, hi)
    return lo, hi


def _pi_ticks(lo, hi, m):
    """Multiples of pi/d inside [lo, hi] with their labels."""
    span = (hi - lo) / math.pi
    best = None
    for num, den in ((1, 4), (1, 2), (1, 1), (2, 1), (4, 1), (8, 1), (16, 1), (1, 8), (1, 16)):
        k = span * den / num
        score = abs(k + 1 - m)
        if best is None or score < best[0]:
            best = (score, num, den)
    _, num, den = best
    step = math.pi * num / den
    out, labels = [], []
    for n in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1):
        p, q = n * num, den
        g = math.gcd(abs(p), q)
        p, q = p // g, q // g
        sign = "-" if p < 0 else ""
        p = abs(p)
        if p == 0:
            s = "$0$"
        elif q == 1:
            s = f"${sign}{'' if p == 1 else p}\\pi$"
        else:
            s = f"${sign}\\frac{{{'' if p == 1 else p}\\pi}}{{{q}}}$"
        out.append(n * step)
        labels.append(s)
    return out, labels


def _wanted(axis, length):
    """How many ticks an axis of `length` bp asks for: one per 36 bp along x (labels are wide),
    one per 30 bp along y."""
    return max(2, min(10, round(length / (36.0 if axis == "x" else 30.0)) + 1))


def _axis(cell, axis, length, th, want=None):
    """The laid-out axis for a plot side of `length` bp."""
    a = _Axis()
    a.scale = cell.xscale if axis == "x" else cell.yscale
    a.cats = cell.xcats if axis == "x" else cell.ycats
    a.lo, a.hi = _range(cell, axis, th)
    a.minor, a.mult = [], None
    fixed = cell.xticks if axis == "x" else cell.yticks
    m = _wanted(axis, length) if want is None else want
    if fixed == "pi":
        a.ticks, a.labels = _pi_ticks(a.lo, a.hi, m)
    elif fixed is not None:
        vals, labels = fixed
        keep = [i for i, v in enumerate(vals) if a.lo <= v <= a.hi and (a.scale != "log" or v > 0)]
        a.ticks = [vals[i] for i in keep]
        if labels is not None:
            a.labels = [labels[i] for i in keep]
        elif a.scale == "log":
            a.labels = _ticks.log_labels(a.ticks)
        else:
            a.labels, a.mult = _ticks.labels(a.ticks)
    elif a.cats is not None:
        a.ticks = [float(i) for i in range(len(a.cats)) if a.lo <= i <= a.hi]
        a.labels = [a.cats[int(i)] for i in a.ticks]
    elif a.scale == "log":
        a.ticks, a.minor = _ticks.log(a.lo, a.hi, m)
        a.ticks = [v for v in a.ticks if a.lo <= v <= a.hi]
        a.labels = _ticks.log_labels(a.ticks)
    else:
        a.ticks = _ticks.linear(a.lo, a.hi, m)
        a.labels, a.mult = _ticks.labels(a.ticks)
    return a


class _Measured:
    """The decorations of a cell for a given plot size: axes and the margins they need."""


def _legend_entries(cell):
    seen, out = set(), []
    for layer in cell.layers + (cell.twin.layers if cell.twin is not None else []):
        for label, handle in layer.entries():
            if label is not None and label not in seen:
                seen.add(label)
                out.append((label, handle))
    return out


def _static_strings(fig):
    """Every string of the figure that does not depend on the layout (one TeX batch)."""
    out = []
    for cell in fig._cells:
        out += [s for s in (cell.title, cell.xlabel, cell.ylabel, cell.zlabel, cell.legend["title"], cell.note) if s]
        if cell.twin is not None:
            out += [s for s in (cell.twin.ylabel,) if s] + [s for layer in cell.twin.layers for s in layer.strings()]
        for inner, _ in cell.insets:
            out += [s for s in (inner.title, inner.xlabel, inner.ylabel) if s]
            out += [s for layer in inner.layers for s in layer.strings()]
        out += [label for label, _ in _legend_entries(cell)]
        for cats in (cell.xcats, cell.ycats):
            out += cats or []
        for t in (cell.xticks, cell.yticks):
            if isinstance(t, tuple) and t[1]:
                out += t[1]
        for layer in cell.layers:
            out += layer.strings()
    if fig._panels:
        out += [rf"\textbf{{{p}}}" for p in fig._panels if p]
    return list(dict.fromkeys(out))


def _measure(fig, cell, pw, ph, th, engine, want=(None, None)):
    """Axes and margins of a cell whose plot rectangle is pw x ph. `want`: the numbers of ticks
    asked of the x and y axes (None: what the side's length asks for), which the cells of a
    shared axis take from one another so that they choose the same ticks."""

    def box(s, size):
        return engine.layout([s])[0].box(size)

    m = _Measured()
    m.legend = m.bar = m.fit = None
    m.want = [None, None]  # the numbers of ticks the automatic axes ended up asking for
    m.title = None  # (height, depth) of the title
    if not cell.frame:
        m.xa = m.ya = None
        m.left = m.right = m.top = m.bottom = 0.0
        space = cell.layers[0] if cell.layers and hasattr(cell.layers[0], "margins") else None
        if space is not None:  # a 3D cell: labels in the margins, the box in the plot rectangle
            m.left, m.right, m.top, m.bottom = space.margins(th, engine)
            entries = _legend_entries(cell)
            m.loc = None if cell.legend["loc"] is None else "right"  # beside the box: it has no free corner
            if entries and m.loc:
                m.legend = _legend_box(cell, entries, th, engine, pw, "right")
                m.right += m.legend["w"] + th["gap"]
                pw = max(pw - m.legend["w"] - th["gap"], 10.0)
            a = space.aspect()
            fw = min(pw, ph / a)
            m.fit = (fw, fw * a)
        if cell.title:
            _, h, d = box(cell.title, th["title_size"])
            m.title = (h, d)
            m.top += h + d + th["title_pad"]
        return m
    m.xa, m.ya = _axis(cell, "x", pw, th, want[0]), _axis(cell, "y", ph, th, want[1])

    def blank():
        for k, a in enumerate((m.xa, m.ya)):
            if cell.inner[k]:  # a shared axis: the outer cell carries the numbers
                a.labels, a.mult = [""] * len(a.labels), None

    blank()
    if cell.headroom and cell.ylim[1] is None:
        # room above the data for the brackets: the range grows so that they clear it
        # (the ticks stay those of the data: none beside the brackets); on a log axis it grows
        # in the logarithm
        need = min(0.7, (13.0 * cell.headroom + 12.0) / ph)
        data = _axis(cell, "y", ph * (1.0 - need), th)
        m.ya.ticks, m.ya.labels, m.ya.mult, m.ya.minor = data.ticks, data.labels, data.mult, data.minor
        if m.ya.scale == "log":
            a, b = math.log10(m.ya.lo), math.log10(m.ya.hi)
            m.ya.hi = 10.0 ** (a + (b - a) / (1.0 - need))
        else:
            m.ya.hi = m.ya.lo + (m.ya.hi - m.ya.lo) / (1.0 - need)
    m.fit = None
    if cell.aspect is not None and m.xa.scale == "linear" and m.ya.scale == "linear":
        # equal data units: the plot rectangle is the largest of the data's shape that fits
        k = min(pw / (m.xa.hi - m.xa.lo), ph / (m.ya.hi - m.ya.lo))
        fw, fh = (m.xa.hi - m.xa.lo) * k, (m.ya.hi - m.ya.lo) * k
        m.fit = (fw, fh)
        if fw < pw - 1e-9 or fh < ph - 1e-9:  # the ticks for the sides as they are drawn
            cx, cy = _axis(cell, "x", fw, th, want[0]), _axis(cell, "y", fh, th, want[1])
            m.xa, m.ya = cx, cy
            blank()
        pw, ph = fw, fh
    for k, (name, a, side) in enumerate((("x", m.xa, pw), ("y", m.ya, ph))):
        fixed = cell.xticks if k == 0 else cell.yticks
        if fixed is None and a.cats is None and not (k == 1 and cell.headroom):
            m.want[k] = _wanted(name, side) if want[k] is None else want[k]
    ts, tl, tp = th["tick_size"], th["tick_len"], th["tick_pad"]
    # the labels the layout made (pi ticks, a common power of ten) are typeset in one batch
    engine.layout([s for s in (*m.xa.labels, *m.ya.labels, m.xa.mult, m.ya.mult) if s])
    xb = [box(s, ts) for s in m.xa.labels]
    yb = [box(s, ts) for s in m.ya.labels]

    def crowded():
        """Whether two neighbouring x labels come closer than a third of the text size."""
        at = sorted((m.xa.frac(v) * pw, w) for v, (w, _, _) in zip(m.xa.ticks, xb))
        return any(b[0] - b[1] / 2 - (a[0] + a[1] / 2) < 0.35 * ts for a, b in zip(at, at[1:]))

    m.xrot = 0.0
    if cell.xticks is None and m.xa.cats is None:
        while crowded() and m.want[0] > 2:  # fewer ticks rather than labels that touch
            m.want[0] -= 1
            m.xa = _axis(cell, "x", pw, th, m.want[0])
            xb = [box(s, ts) for s in m.xa.labels]
    elif crowded():  # labels the user or the data gave: they are turned
        m.xrot = 40.0
        c, sn = math.cos(math.radians(m.xrot)), math.sin(math.radians(m.xrot))
        xb = [(w * c + (h + d) * sn, 0.0, w * sn + (h + d) * c) for w, h, d in xb]
    # how far the first and last x labels reach past the plot's ends
    over_l = over_r = 0.0
    for v, (w, _, _) in zip(m.xa.ticks, xb):
        f = m.xa.frac(v)
        over_l = max(over_l, (w if m.xrot else w / 2) - f * pw)  # a turned label hangs to the left of its tick
        over_r = max(over_r, (0.0 if m.xrot else w / 2) - (1 - f) * pw)
    over_t = over_b = 0.0
    for v, (_, h, d) in zip(m.ya.ticks, yb):
        f = m.ya.frac(v)
        over_b = max(over_b, (h + d) / 2 - f * ph)
        over_t = max(over_t, (h + d) / 2 - (1 - f) * ph)
    m.ytick_w = max((w for w, _, _ in yb), default=0.0)
    m.xtick_h = max((h for _, h, _ in xb), default=0.0)
    m.xtick_d = max((d for _, _, d in xb), default=0.0)
    m.left = tl + tp + m.ytick_w
    m.bottom = tl + tp + m.xtick_h + m.xtick_d
    if cell.ylabel:
        _, h, d = box(cell.ylabel, th["size"])
        m.left += th["label_pad"] + h + d
    if cell.xlabel:
        _, h, d = box(cell.xlabel, th["size"])
        m.bottom += th["label_pad"] + h + d
    m.left = max(m.left, over_l)
    m.right = max(over_r, 0.0)
    m.ya2 = None
    if cell.twin is not None and cell.twin.layers:  # the second y axis: its ticks and name at the right
        twin = cell.twin
        m.ya2 = _axis(twin, "y", ph, th)
        engine.layout([s for s in (*m.ya2.labels, m.ya2.mult) if s])
        m.ytick2_w = max((box(s, ts)[0] for s in m.ya2.labels), default=0.0)
        need = tl + tp + m.ytick2_w
        if twin.ylabel:
            _, h, d = box(twin.ylabel, th["size"])
            need += th["label_pad"] + h + d
        m.right = max(m.right, need)
    m.bottom = max(m.bottom, over_b)
    m.top = max(over_t, 0.0)
    if m.ya.mult:
        _, h, d = box(m.ya.mult, ts)
        m.top = max(m.top, h + d + 2.0)
    if m.xa.mult:
        w, _, _ = box(m.xa.mult, ts)
        m.right = max(m.right, w + 2.0)
    if cell.title:
        _, h, d = box(cell.title, th["title_size"])
        m.title = (h, d)
        m.top += h + d + th["title_pad"]
    m.bar = None
    if cell.colorbar is not None:
        _, lo, hi = cell.colorbar
        ticks = _ticks.linear(lo, hi, _wanted("y", ph)) if lo < hi else [lo]
        labels, mult = _ticks.labels(ticks)
        wide = max((box(s, ts)[0] for s in labels), default=0.0)
        m.bar = {"ticks": ticks, "labels": labels, "mult": mult, "at": m.right + th["gap"]}
        m.right += th["gap"] + _BAR_W + 2.0 + tp + wide
        if mult:
            _, h, d = box(mult, ts)
            m.top = max(m.top, h + d + 2.0)
    # the legend: its size, and its margin when it stands outside the plot
    entries = _legend_entries(cell)
    loc = cell.legend["loc"]
    if loc == "best" and cell.placed is not None:
        loc = cell.placed  # where the layout found room (a corner, or outside)
    m.loc = loc
    if entries and loc is not None:
        m.legend = _legend_box(cell, entries, th, engine, pw, loc)
        lw, lh = m.legend["w"], m.legend["h"]
        if loc == "right":
            m.right += lw + th["gap"]
        elif loc == "top":
            m.top += lh + th["gap"] / 2
        elif loc == "bottom":
            m.bottom += lh + th["gap"] / 2
    return m


_HANDLE_W = 14.0
_BAR_W = 7.0


def _legend_box(cell, entries, th, engine, room, loc):
    """The size and the rows of a legend. Above or below the plot it takes as many columns as fit
    in the plot's width `room`."""
    size = th["legend_size"]
    outside = loc in ("top", "bottom")
    ncols = min(max(cell.legend["ncols"], len(entries)) if outside else cell.legend["ncols"], len(entries))
    boxes = [engine.layout([label])[0].box(size) for label, _ in entries]
    row_h = max(max(h + d for _, h, d in boxes), size) + 2.0
    while True:
        nrows = -(-len(entries) // ncols)
        col_w = [0.0] * ncols
        for i, (w, _, _) in enumerate(boxes):
            col_w[i // nrows] = max(col_w[i // nrows], _HANDLE_W + 3.0 + w)
        w = sum(col_w) + 8.0 * (ncols - 1)
        if not outside or w <= room or ncols == 1:
            break
        ncols -= 1
    ncols = -(-len(entries) // nrows)  # the columns in use
    col_w = col_w[:ncols]
    w = sum(col_w) + 8.0 * (ncols - 1)
    h = nrows * row_h
    title = cell.legend["title"]
    th_ = 0.0
    if title:
        tw, a, d = engine.layout([title])[0].box(size)
        th_ = a + d + 3.0
        w = max(w, tw)
    return {
        "w": w,
        "h": h + th_,
        "rows": nrows,
        "cols": ncols,
        "col_w": col_w,
        "row_h": row_h,
        "title_h": th_,
        "entries": entries,
        "boxes": boxes,
    }


def _share(fig, th):
    """The ranges shared by the cells of a grid made with sharex or sharey: True shares along
    the columns (x) or the rows (y), 'all' among all the cells. A shared axis shows its numbers
    on the outer cells only (the bottom of a column, the left of a row): a cell leaves its
    numbers out when a cell below it (x) or left of it (y) shows the same axis, which is one of
    the same shared range or of the same categories. Returns the groups [(0 for x or 1 for y,
    the indices of the cells of one shared range)], whose ticks the layout chooses once."""
    nr, nc = fig._nrows, fig._ncols
    cells = fig._cells
    shared = []
    for c in cells:
        c.span, c.inner = [None, None], [False, False]
    for k, how in enumerate(fig._share):
        if not how:
            continue
        axis = "x" if k == 0 else "y"
        if how == "all":
            groups = [list(range(nr * nc))]
        elif k == 0:
            groups = [[r * nc + c for r in range(nr)] for c in range(nc)]
        else:
            groups = [[r * nc + c for c in range(nc)] for r in range(nr)]
        for g in groups:
            members = [
                i
                for i in g
                if cells[i].frame and cells[i].layers and (cells[i].xcats if k == 0 else cells[i].ycats) is None
            ]
            if len(members) < 2:
                continue
            ranges = [_range(cells[i], axis, th) for i in members]
            span = (min(r[0] for r in ranges), max(r[1] for r in ranges))
            free = [i for i in members if (cells[i].xlim if k == 0 else cells[i].ylim) == [None, None]]
            for i in free:
                cells[i].span[k] = span
            if len(free) > 1:
                shared.append((k, free))

        def same(a, b):
            """Whether two cells show one axis: the same shared range, or the same categories."""
            ca, cb = (a.xcats, b.xcats) if k == 0 else (a.ycats, b.ycats)
            if ca is not None or cb is not None:
                return ca == cb and (a.xlim if k == 0 else a.ylim) == (b.xlim if k == 0 else b.ylim)
            return a.span[k] is not None and a.span[k] == b.span[k]

        drawn = [c.frame and bool(c.layers) for c in cells]
        for i, c in enumerate(cells):
            r, col = divmod(i, nc)
            if not drawn[i]:
                continue
            outer = range((r + 1) * nc + col, nr * nc, nc) if k == 0 else range(r * nc, i)
            c.inner[k] = any(drawn[j] and same(c, cells[j]) for j in outer)
    return shared


def _settle(fig):
    """The layers that depend on the axes they are drawn on (a hexbin's bins on log axes) made for
    them, and the colours of the layers that share one colour range (a facet's counts) set by
    the largest value of all of them."""
    found = []

    def walk(cell, xscale):
        for layer in cell.layers:
            if hasattr(layer, "settle"):
                layer.settle(xscale == "log", cell.yscale == "log")
                found.append((cell, layer))
        if cell.twin is not None:
            walk(cell.twin, xscale)  # (the x axis is the cell's)
        for inner, _ in cell.insets:
            walk(inner, inner.xscale)

    for cell in fig._cells:
        walk(cell, cell.xscale)
    tops = {}
    for _, layer in found:
        if layer.share is not None:
            tops[layer.share] = max(tops.get(layer.share, 0.0), layer.most)
    for cell, layer in found:
        layer.paint(cell, tops.get(layer.share) if layer.share is not None else None)


def _layout(fig, dpi=300.0):
    """The scene of a figure."""
    th = THEMES[fig._theme]
    engine = _text.engine(fig._font)
    engine.layout(_static_strings(fig))  # one TeX run for all that is new
    for cell in fig._cells:
        cell.placed = None
    _settle(fig)
    shared = _share(fig, THEMES[fig._theme])
    nr, nc = fig._nrows, fig._ncols
    W = fig._w * MM
    margin, gap = th["margin"], th["gap"]
    panel_h = 0.0
    if fig._panels:
        panel_h = max(engine.layout([rf"\textbf{{{p}}}"])[0].box(th["panel_size"])[1] for p in fig._panels if p) + 3.0
    cell_w = (W - 2 * margin - gap * (nc - 1)) / nc
    H = fig._h * MM if fig._h is not None else None

    def solve(pw, ph):
        """Margins of every column and row for plot sizes pw[c], ph[r]."""
        ms = [_measure(fig, fig._cells[r * nc + c], pw[c], ph[r], th, engine) for r in range(nr) for c in range(nc)]
        # The cells of a shared axis choose their ticks once: all take the fewest any of them asks
        # for (the narrowest plot's, or the one whose labels would touch).
        asked = {}
        for _ in range(3):
            again = set()
            for k, members in shared:
                counts = [ms[i].want[k] for i in members if ms[i].want[k] is not None]
                if not counts:
                    continue
                for i in members:
                    if ms[i].want[k] is not None and ms[i].want[k] != min(counts):
                        asked.setdefault(i, [None, None])[k] = min(counts)
                        again.add(i)
            if not again:
                break
            for i in again:
                ms[i] = _measure(fig, fig._cells[i], pw[i % nc], ph[i // nc], th, engine, asked[i])
        # The titles of a row stand on one baseline: each takes the row's tallest and deepest.
        for r in range(nr):
            row = [m for m in ms[r * nc : (r + 1) * nc] if m.title is not None]
            if row:
                h, d = max(m.title[0] for m in row), max(m.title[1] for m in row)
                for m in row:
                    m.top += h + d - sum(m.title)
                    m.title = (h, d)
        left = [max(ms[r * nc + c].left for r in range(nr)) for c in range(nc)]
        right = [max(ms[r * nc + c].right for r in range(nr)) for c in range(nc)]
        top = [max(ms[r * nc + c].top for c in range(nc)) + panel_h for r in range(nr)]
        bottom = [max(ms[r * nc + c].bottom for c in range(nc)) for r in range(nr)]
        return ms, left, right, top, bottom

    def shapes(pw):
        """The plot heights of the rows of a figure without a height: 4:3 where a row holds a 2D
        plot, else the tallest of its 3D boxes' projections."""
        out = []
        for r in range(nr):
            row = fig._cells[r * nc : (r + 1) * nc]
            spaces = [c.layers[0] for c in row if not c.frame and c.layers and hasattr(c.layers[0], "aspect")]
            flat = any(c.frame for c in row) or not spaces
            out.append(0.75 * pw[0] if flat else pw[0] * min(max(sp.aspect() for sp in spaces), 1.25))
        return out

    pw = [max(cell_w - 40.0, 10.0)] * nc
    ph = shapes(pw) if H is None else [max((H - 2 * margin - gap * (nr - 1)) / nr - 30.0, 10.0)] * nr
    for _ in range(3):  # the ticks depend on the plot size, the margins on the ticks
        ms, left, right, top, bottom = solve(pw, ph)
        pw = [max(cell_w - left[c] - right[c], 10.0) for c in range(nc)]
        if H is None:
            ph = shapes(pw)
        else:
            ch = (H - 2 * margin - gap * (nr - 1)) / nr
            ph = [max(ch - top[r] - bottom[r], 10.0) for r in range(nr)]
    ms, left, right, top, bottom = solve(pw, ph)
    # A legend left to the layout goes into a corner of the plot that the data leave empty; when
    # there is none it stands outside, and the plots are laid out again around it.
    # A legend that goes outside narrows its column or lowers its row, so the corners of the
    # other plots are looked at again at their new size (a legend only ever moves outwards).
    for _ in range(3):
        moved = False
        for r in range(nr):
            for c in range(nc):
                cell, m = fig._cells[r * nc + c], ms[r * nc + c]
                if not (cell.frame and m.legend and cell.legend["loc"] == "best") or cell.placed in ("top", "right"):
                    continue
                w, h = m.fit if m.fit is not None else (pw[c], ph[r])
                cell.placed = _place_legend(cell, m, w, h, th, fig._palette, engine)
                if cell.placed in ("top", "right"):
                    moved = True
                else:
                    m.loc = cell.placed  # a corner takes no room of the layout: the measure stands
        if not moved:
            break
        for _ in range(2):
            ms, left, right, top, bottom = solve(pw, ph)
            pw = [max(cell_w - left[c] - right[c], 10.0) for c in range(nc)]
            if H is None:
                ph = shapes(pw)
            else:
                ch = (H - 2 * margin - gap * (nr - 1)) / nr
                ph = [max(ch - top[r] - bottom[r], 10.0) for r in range(nr)]
        ms, left, right, top, bottom = solve(pw, ph)
    if H is None:
        H = 2 * margin + gap * (nr - 1) + sum(top[r] + ph[r] + bottom[r] for r in range(nr))
        cell_h = [top[r] + ph[r] + bottom[r] for r in range(nr)]
    else:
        cell_h = [(H - 2 * margin - gap * (nr - 1)) / nr] * nr

    scene = _scene.Scene(W, H, font=fig._font)
    scene.add(_scene.Path(*_scene.rect(0.0, 0.0, W, H), fill=WHITE))
    y0 = margin
    for r in range(nr):
        x0 = margin
        for c in range(nc):
            i = r * nc + c
            cell, m = fig._cells[i], ms[i]
            px, py = x0 + left[c], y0 + top[r]
            if fig._panels and fig._panels[i]:
                scene.add(
                    _scene.Text(rf"\textbf{{{fig._panels[i]}}}", x0, y0, size=th["panel_size"], ha="left", va="top")
                )
            cw, chh = pw[c], ph[r]
            band = py  # the top of the row's plot band: the titles of a row stand on one line
            if getattr(m, "fit", None) is not None:  # centred in the room the cell has
                fw, fh = m.fit
                px, py, cw, chh = px + (cw - fw) / 2, py + (chh - fh) / 2, fw, fh
            _draw_cell(scene, cell, m, px, py, cw, chh, th, fig._palette, engine, dpi, band)
            x0 += cell_w + gap
        y0 += cell_h[r] + gap
    return scene


class Ink:
    """Where the items of a plot rectangle put ink, on a grid of 2 bp cells (painted by the
    rasterizer, so fills, strokes, markers and pictures all count as they are drawn)."""

    K = 0.5

    def __init__(self, items, x, y, w, h):
        k = self.K
        self.x, self.y = x, y
        self.w, self.h = max(1, math.ceil(w * k)), max(1, math.ceil(h * k))
        canvas = sd.plot.Canvas(self.w, self.h)
        m, black = (k, 0.0, 0.0, k, -x * k, -y * k), (0.0, 0.0, 0.0, 1.0)
        for item in items:
            if isinstance(item, _scene.Image):
                canvas.fill(
                    b"MLLLZ", array("d", _scene.rect(item.x, item.y, item.w, item.h)[1]).tobytes(), m, black, ()
                )
            elif isinstance(item, _scene.Markers):
                canvas.markers(item.ops.encode(), item.xy.tobytes(), m, item.at.tobytes(), black, (), 0.0, ())
            elif isinstance(item, _scene.Path):
                ops, xy = item.ops.encode(), item.xy.tobytes()
                if item.fill is not None:
                    canvas.fill(ops, xy, m, black, ())
                if item.stroke is not None:
                    canvas.stroke(ops, xy, m, black, (), max(item.width * k, 1.0), 0, 1, [])
        self.alpha = canvas.rgba8()[3::4]

    def share(self, x, y, w, h):
        """The share of the rectangle (page coordinates) that holds ink; 1 outside the plot."""
        k = self.K
        c0, c1 = math.floor((x - self.x) * k), math.ceil((x + w - self.x) * k)
        r0, r1 = math.floor((y - self.y) * k), math.ceil((y + h - self.y) * k)
        if c0 < 0 or r0 < 0 or c1 > self.w or r1 > self.h or c1 <= c0 or r1 <= r0:
            return 1.0
        n = sum(c1 - c0 - self.alpha[r * self.w + c0 : r * self.w + c1].count(0) for r in range(r0, r1))
        return n / ((c1 - c0) * (r1 - r0))


_CORNERS = ("upper right", "upper left", "lower right", "lower left")


def _corner(name, x, y, w, h, bw, bh, inset=4.0):
    """The upper left point of a box of bw x bh in a corner of the plot rectangle."""
    return (x + w - bw - inset if "right" in name else x + inset, y + h - bh - inset if "lower" in name else y + inset)


def _place_legend(cell, m, w, h, th, palette, engine):
    """A corner of the plot the layers leave empty for the legend, else 'top' or 'right'."""
    trial = _scene.Scene(w, h)
    fr = Frame(trial, 0.0, 0.0, w, h, m.xa, m.ya, th, palette, engine)
    for layer in cell.layers:
        layer.draw(fr)
    if getattr(m, "ya2", None) is not None:
        fr2 = Frame(trial, 0.0, 0.0, w, h, m.xa, m.ya2, th, palette, engine)
        for layer in cell.twin.layers:
            layer.draw(fr2)
    ink = Ink(trial.items, 0.0, 0.0, w, h)
    lw, lh = m.legend["w"], m.legend["h"]
    best = None
    for name in _CORNERS:
        bx, by = _corner(name, 0.0, 0.0, w, h, lw, lh)
        share = ink.share(bx - 3.0, by - 3.0, lw + 6.0, lh + 6.0)
        if best is None or share < best[0]:
            best = (share, name)
    if best[0] <= 0.004:
        return best[1]
    # no room inside: above the plot when one row holds it, else at the right
    one_row = sum(_HANDLE_W + 3.0 + b[0] for b in m.legend["boxes"]) + 8.0 * (len(m.legend["boxes"]) - 1)
    return "top" if one_row <= w else "right"


def _line(scene, pts, th, color=INK, width=None, **kw):
    scene.add(
        _scene.Path(*_scene.polyline(pts), stroke=color, width=th["axis_width"] if width is None else width, **kw)
    )


def _draw_cell(scene, cell, m, x, y, w, h, th, palette, engine, dpi=300.0, band=None):
    ts, tl, tp = th["tick_size"], th["tick_len"], th["tick_pad"]
    if not cell.frame:
        fr = Frame(scene, x, y, w, h, None, None, th, palette, engine, dpi)
        for layer in cell.layers:
            layer.draw(fr)
        if getattr(m, "legend", None):
            _draw_legend(scene, cell, m, fr, th, engine)
        if cell.title:
            ty = (y if band is None else min(y, band)) - th["title_pad"] - m.title[1]
            scene.add(_scene.Text(cell.title, x + w / 2, ty, size=th["title_size"], ha="center", va="baseline"))
        return
    xa, ya = m.xa, m.ya
    fr = Frame(scene, x, y, w, h, xa, ya, th, palette, engine)
    if th["grid"]:
        for v in xa.ticks:
            px = fr.px(v)
            _line(scene, [(px, y), (px, y + h)], th, GRID_COLOR, 0.4)
        for v in ya.ticks:
            py = fr.py(v)
            _line(scene, [(x, py), (x + w, py)], th, GRID_COLOR, 0.4)
    first = len(scene.items)
    for layer in cell.layers:
        layer.draw(fr)
    fr2 = None
    if getattr(m, "ya2", None) is not None:
        fr2 = Frame(scene, x, y, w, h, xa, m.ya2, th, palette, engine)
        for layer in cell.twin.layers:
            layer.draw(fr2)
    drawn = scene.items[first:]
    # the axis lines, then ticks and their labels
    sp = th["spines"]
    aw = th["axis_width"]
    if "left" in sp and "bottom" in sp and len(sp) == 2:
        _line(scene, [(x, y), (x, y + h), (x + w, y + h)], th, cap="square")
    else:
        if len(sp) == 4:
            scene.add(_scene.Path(*_scene.rect(x, y, w, h), stroke=INK, width=aw))
        else:
            ends = {
                "left": [(x, y), (x, y + h)],
                "bottom": [(x, y + h), (x + w, y + h)],
                "right": [(x + w, y), (x + w, y + h)],
                "top": [(x, y), (x + w, y)],
            }
            for s in sp:
                _line(scene, ends[s], th, cap="square")
    for v, s in zip(xa.ticks, xa.labels):
        px = fr.px(v)
        _line(scene, [(px, y + h), (px, y + h + tl)], th)
        if not s:  # the numbers of a shared axis stand on another cell
            continue
        if m.xrot:
            scene.add(_scene.Text(s, px + 1.5, y + h + tl + tp, size=ts, ha="right", va="top", rotation=m.xrot))
        else:
            scene.add(_scene.Text(s, px, y + h + tl + tp + m.xtick_h, size=ts, ha="center", va="baseline"))
    for v in xa.minor:
        px = fr.px(v)
        _line(scene, [(px, y + h), (px, y + h + 0.6 * tl)], th, width=0.7 * aw)
    for v, s in zip(ya.ticks, ya.labels):
        py = fr.py(v)
        _line(scene, [(x, py), (x - tl, py)], th)
        if s:
            scene.add(_scene.Text(s, x - tl - tp, py, size=ts, ha="right", va="center"))
    for v in ya.minor:
        py = fr.py(v)
        _line(scene, [(x, py), (x - 0.6 * tl, py)], th, width=0.7 * aw)
    if fr2 is not None:  # the second y axis, at the right
        ya2, twin = m.ya2, cell.twin
        _line(scene, [(x + w, y), (x + w, y + h)], th, cap="square")
        for v, s in zip(ya2.ticks, ya2.labels):
            py = fr2.py(v)
            _line(scene, [(x + w, py), (x + w + tl, py)], th)
            scene.add(_scene.Text(s, x + w + tl + tp, py, size=ts, ha="left", va="center"))
        if ya2.mult:
            scene.add(_scene.Text(ya2.mult, x + w, y - 2.0, size=ts, ha="right", va="bottom"))
        if twin.ylabel:
            _, lh, ld = engine.layout([twin.ylabel])[0].box(th["size"])
            xx = x + w + tl + tp + m.ytick2_w + th["label_pad"] + lh
            scene.add(
                _scene.Text(twin.ylabel, xx, y + h / 2, size=th["size"], ha="center", va="baseline", rotation=90.0)
            )
    if ya.mult:
        scene.add(_scene.Text(ya.mult, x, y - 2.0, size=ts, ha="left", va="bottom"))
    if xa.mult:
        scene.add(_scene.Text(xa.mult, x + w + 2.0, y + h, size=ts, ha="left", va="center"))
    if cell.xlabel:
        yy = y + h + tl + tp + m.xtick_h + m.xtick_d + th["label_pad"]
        scene.add(_scene.Text(cell.xlabel, x + w / 2, yy, size=th["size"], ha="center", va="top"))
    if cell.ylabel:
        _, lh, ld = engine.layout([cell.ylabel])[0].box(th["size"])
        xx = x - tl - tp - m.ytick_w - th["label_pad"] - ld
        scene.add(_scene.Text(cell.ylabel, xx, y + h / 2, size=th["size"], ha="center", va="baseline", rotation=90.0))
    top = y
    if m.legend and m.loc == "top":
        top -= m.legend["h"] + th["gap"] / 2
    if ya.mult:
        top -= engine.layout([ya.mult])[0].box(ts)[1] + 2.0 if not cell.title else 0.0
    if cell.title:  # on the baseline the row's titles share (the layout's m.title)
        ty = top - th["title_pad"] - m.title[1]
        scene.add(_scene.Text(cell.title, x + w / 2, ty, size=th["title_size"], ha="center", va="baseline"))
    if cell.note:
        # in the corner the data and the legend leave emptiest
        nw, nh_, nd = engine.layout([cell.note])[0].box(ts)
        ink = Ink(drawn, x, y, w, h)
        taken = m.loc if m.legend else None
        _, name = min(
            (ink.share(*_corner(c, x, y, w, h, nw, nh_ + nd), nw, nh_ + nd), i)
            for i, c in enumerate(_CORNERS)
            if c != taken
        )
        nx, ny = _corner(_CORNERS[name], x, y, w, h, nw, nh_ + nd)
        scene.add(_scene.Text(cell.note, nx, ny, size=ts, ha="left", va="top"))
    if m.bar is not None:
        name, lo, hi = cell.colorbar
        bx = x + w + m.bar["at"]
        scene.add(_scene.Image(bx, y, _BAR_W, h, 1, 256, bytes(reversed_rows(_color.table(name))), smooth=True))
        scene.add(_scene.Path(*_scene.rect(bx, y, _BAR_W, h), stroke=INK, width=0.4))
        for v, s in zip(m.bar["ticks"], m.bar["labels"]):
            py = y + h - (v - lo) / (hi - lo) * h if hi > lo else y + h / 2
            _line(scene, [(bx + _BAR_W, py), (bx + _BAR_W + 2.0, py)], th, width=0.4)
            scene.add(_scene.Text(s, bx + _BAR_W + 2.0 + tp, py, size=ts, ha="left", va="center"))
        if m.bar["mult"]:
            scene.add(_scene.Text(m.bar["mult"], bx, y - 2.0, size=ts, ha="left", va="bottom"))
    if m.legend:
        _draw_legend(scene, cell, m, fr, th, engine)
    for inner, (rx, ry, rw, rh) in cell.insets:
        _draw_inset(scene, inner, x + rx * w, y + (1.0 - ry - rh) * h, rw * w, rh * h, th, palette, engine, dpi)


def _draw_inset(scene, cell, x, y, w, h, th, palette, engine, dpi):
    """A cell drawn in the rectangle (x, y, w, h) of the page, its ticks and labels inside it, on
    a white ground."""
    pw, ph = max(w - 30.0, 10.0), max(h - 20.0, 10.0)
    for _ in range(3):  # the margins depend on the plot size, the plot size on the margins
        m = _measure(None, cell, pw, ph, th, engine)
        pw, ph = max(w - m.left - m.right - 2.0, 10.0), max(h - m.top - m.bottom - 2.0, 10.0)
    m = _measure(None, cell, pw, ph, th, engine)
    scene.add(_scene.Path(*_scene.rect(x, y, w, h), fill=WHITE))
    px, py = x + m.left + 1.0, y + m.top + 1.0
    if getattr(m, "fit", None) is not None:
        fw, fh = m.fit
        px, py, pw, ph = px + (pw - fw) / 2, py + (ph - fh) / 2, fw, fh
    _draw_cell(scene, cell, m, px, py, pw, ph, th, palette, engine, dpi)


def _draw_legend(scene, cell, m, fr, th, engine):
    L = m.legend
    loc = m.loc
    x, y, w, h = fr.x, fr.y, fr.w, fr.h
    lw, lh = L["w"], L["h"]
    if loc == "best":  # a cell the layout did not place (no frame): the first corner
        loc = _CORNERS[0]
    if loc in _CORNERS:
        lx, ly = _corner(loc, x, y, w, h, lw, lh)
    elif loc == "right":
        lx, ly = x + w + m.right - lw, y + (h - lh) / 2
    elif loc == "top":
        lx, ly = x + (w - lw) / 2, y - lh - th["gap"] / 2
    else:
        lx, ly = x + (w - lw) / 2, y + h + m.bottom - lh
    size = th["legend_size"]
    if cell.legend["title"]:
        scene.add(_scene.Text(cell.legend["title"], lx, ly, size=size, ha="left", va="top"))
    ly += L["title_h"]
    cx = lx
    for c in range(L["cols"]):
        for r in range(L["rows"]):
            i = c * L["rows"] + r
            if i >= len(L["entries"]):
                break
            label, handle = L["entries"][i]
            cy = ly + r * L["row_h"] + L["row_h"] / 2
            handle(scene, cx, cy, _HANDLE_W, th)
            _, a, d = L["boxes"][i]
            scene.add(_scene.Text(label, cx + _HANDLE_W + 3.0, cy + (a - d) / 2, size=size, ha="left", va="baseline"))
        cx += L["col_w"][c] + 8.0


def reversed_rows(table):
    """A colour table (RGBA8, lowest value first) as a one-pixel-wide picture with the highest value on top."""
    return b"".join(table[i : i + 4] for i in range(len(table) - 4, -1, -4))


def interleave(X, Y):
    """Two sd arrays of equal length as array('d') of x, y pairs."""
    out = array("d")
    out.frombytes(bytes(memoryview(sd.reshape(sd.stack([X, Y], axis=1), (-1,)))))
    return out
