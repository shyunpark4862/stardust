"""The basic plots (docs/api/plot.md §3): line, scatter, bar, area, step, errorbar, hline, vline,
text, annotate, and grid, which gathers figures.

Each is a Figure method that adds layers to the figure's cell and returns the figure; the module
function of the same name makes a new figure from the figure options (width, height, font,
theme, palette) and calls the method.
"""

import functools
import inspect
import math

from .. import _calls
from .. import _core as sd
from .._core import _type_name
from ..optimize._common import _as_array, _check_bool, _check_choice
from ..stats._distribution import _real_option
from . import _color, _layers, _scene
from ._figure import DASHES, Cell, Figure, integer, real, string

FIGURE_OPTIONS = ("width", "height", "font", "theme", "palette")
_HA = ("left", "center", "right")
_VA = ("baseline", "bottom", "center", "top")
_ESCAPE = {
    "\\": r"\textbackslash{}",
    "_": r"\_",
    "%": r"\%",
    "&": r"\&",
    "#": r"\#",
    "$": r"\$",
    "{": r"\{",
    "}": r"\}",
    "^": r"\^{}",
    "~": r"\~{}",
}


def escape(s):
    """A string from the data (a group or a category) as LaTeX text."""
    return "".join(_ESCAPE.get(c, c) for c in str(s))


# ---- reading data arguments (docs/api/plot.md §1.2)


def _column(who, arg, v, data):
    if isinstance(v, str):
        if data is None:
            raise TypeError(f"{who}(): '{arg}' is a column name ({v!r}), which needs 'data'")
        try:
            return data[v]
        except KeyError:
            raise ValueError(f"{who}(): no column {v!r}") from None
    return v


def _check_data(who, data):
    from .. import frame

    if data is not None and not isinstance(data, frame.DataFrame):
        raise TypeError(f"{who}(): 'data' must be a DataFrame, not {_type_name(data)}")


def numbers_(who, arg, v, data=None):
    """A 1-dimensional float64 sd array from a data argument."""
    from .. import frame

    v = _column(who, arg, v, data)
    if isinstance(v, frame.Series):
        try:
            a = v.to_array(dtype=sd.float64, na_value=math.nan)
        except TypeError, ValueError:
            raise TypeError(f"{who}(): '{arg}' must be numeric, not a Series of {v.dtype}") from None
    else:
        a = _as_array(v, who, arg)
    if a.ndim != 1:
        raise ValueError(f"{who}(): '{arg}' must be 1-dimensional, not shape {tuple(a.shape)}")
    return a


def labels_or_numbers(who, arg, v, data=None):
    """('cat', [str, ...]) for strings (a category axis), else ('num', sd array)."""
    from .. import frame

    v = _column(who, arg, v, data)
    if isinstance(v, frame.Series) and v.dtype == sd.string:
        return "cat", ["" if s is None else str(s) for s in v.tolist()]
    if isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
        return "cat", list(v)
    return "num", numbers_(who, arg, v)


def same_length(who, **arrays):
    names = list(arrays)
    n0 = arrays[names[0]].shape[0] if hasattr(arrays[names[0]], "shape") else len(arrays[names[0]])
    for name in names[1:]:
        a = arrays[name]
        n = a.shape[0] if hasattr(a, "shape") else len(a)
        if n != n0:
            raise ValueError(f"{who}(): '{names[0]}' and '{name}' must have the same length: {n0} and {n}")
    return n0


def split(who, group, data, n):
    """The groups of a `group` argument: [(key, sd int array of row indices)] in order of first
    appearance, or None without one."""
    from .. import frame

    if group is None:
        return None
    g = _column(who, "group", group, data)
    if isinstance(g, frame.Series):
        keys = g.tolist()
    elif isinstance(g, sd.ndarray):
        keys = g.tolist()
    elif isinstance(g, (list, tuple)):
        keys = list(g)
    else:
        raise TypeError(f"{who}(): 'group' must be a column name or a sequence of labels, not {_type_name(g)}")
    if len(keys) != n:
        raise ValueError(f"{who}(): 'group' must have the data's length: {n}, not {len(keys)}")
    rows = {}
    for i, k in enumerate(keys):
        if k is None or (isinstance(k, float) and k != k):
            continue  # a missing label: the row belongs to no group
        rows.setdefault(k, []).append(i)
    return [(k, sd.asarray(idx)) for k, idx in rows.items()]


def _color_of(fig, who, color, alpha=1.0):
    """The colour of a new layer: the given one, else the next of the palette."""
    cell = fig._cell
    c = cell.next_color(fig._root._palette) if color is None else _color.parse(color, who, "color", fig._root._palette)
    return _color.with_alpha(c, alpha) if alpha != 1.0 else c


def _alpha(who, v):
    a = real(v, who, "alpha")
    if not 0.0 <= a <= 1.0:
        raise ValueError(f"{who}(): 'alpha' must be in [0, 1]")
    return a


def _style(who, v):
    return _check_choice(v, who, "style", tuple(DASHES))


def _opt_real(who, arg, v, positive=True):
    return None if v is None else real(v, who, arg, positive=positive)


def _label(who, v):
    return None if v is None else string(v, who, "label")


def _each(fig, who, n, group, data, color, label):
    """[(index array or None, colour source, label)] for a call with an optional `group`."""
    parts = split(who, group, data, n)
    if parts is None:
        return [(None, color, label)]
    if color is not None:
        raise ValueError(f"{who}(): 'color' and 'group' cannot both be given (a group takes the palette's colours)")
    if label is not None:
        raise ValueError(f"{who}(): 'label' and 'group' cannot both be given (a group is named by its value)")
    order = getattr(fig._cell, "order", None)  # a facet's cell: every group keeps the colour it has in the others
    if order:
        palette = fig._root._palette
        return [
            (idx, _color.cycle(palette, order.index(key)) if key in order else None, escape(key)) for key, idx in parts
        ]
    return [(idx, None, escape(key)) for key, idx in parts]


def _take(a, idx):
    return a if idx is None else sd.take(a, idx)


# ---- the methods


def line(self, x, y, /, *, data=None, group=None, color=None, width=None, style="-", alpha=1.0, label=None):
    who = "line"
    _check_data(who, data)
    X, Y = numbers_(who, "x", x, data), numbers_(who, "y", y, data)
    n = same_length(who, x=X, y=Y)
    width, style, a, label = _opt_real(who, "width", width), _style(who, style), _alpha(who, alpha), _label(who, label)
    for idx, col, lab in _each(self, who, n, group, data, color, label):
        self._cell.layers.append(
            _layers.Line(_take(X, idx), _take(Y, idx), _color_of(self, who, col, a), width, style, lab)
        )
    return self._touch()


def scatter(self, x, y, /, *, data=None, group=None, color=None, size=None, shape="o", alpha=1.0, label=None):
    who = "scatter"
    _check_data(who, data)
    X, Y = numbers_(who, "x", x, data), numbers_(who, "y", y, data)
    n = same_length(who, x=X, y=Y)
    size, a, label = _opt_real(who, "size", size), _alpha(who, alpha), _label(who, label)
    _check_choice(shape, who, "shape", _layers.MARKERS)
    for idx, col, lab in _each(self, who, n, group, data, color, label):
        self._cell.layers.append(
            _layers.Scatter(_take(X, idx), _take(Y, idx), _color_of(self, who, col, a), size, shape, lab)
        )
    return self._touch()


def _categories(cell, axis, cats):
    """Positions of category labels on a cell's category axis (new labels are appended)."""
    attr = "xcats" if axis == "x" else "ycats"
    have = getattr(cell, attr)
    if have is None:
        have = []
        setattr(cell, attr, have)
    pos = []
    for c in cats:
        e = escape(c)
        if e not in have:
            have.append(e)
        pos.append(float(have.index(e)))
    return sd.asarray(pos) if pos else sd.zeros(0)


def value_text(v, spec, who):
    """A number written on a plot (bar values, heatmap cells): spec True gives three significant
    digits; a format spec string ('.1f', '.0%', ',.0f') is Python's own."""
    if v != v:
        return ""
    if spec is True:
        s = f"{v:.3g}"
        if "e" in s:
            m, e = s.split("e")
            return f"${m}\\times10^{{{int(e)}}}$"
        return f"${s}$"
    try:
        s = format(v, spec)
    except ValueError:
        raise ValueError(f"{who}(): 'annotate' is not a format spec for numbers: {spec!r}") from None
    return "$" + s.replace("%", "\\%").replace(",", "{,}") + "$"


def _annotate(who, annotate):
    if isinstance(annotate, bool) or isinstance(annotate, str):
        return annotate
    raise TypeError(f"{who}(): 'annotate' must be a bool or a format spec string, not {_type_name(annotate)}")


def bar(
    self,
    x,
    height,
    /,
    *,
    data=None,
    group=None,
    stack=False,
    gap=0.1,
    err=None,
    annotate=False,
    color=None,
    width=0.8,
    horizontal=False,
    label=None,
):
    who = "bar"
    _check_data(who, data)
    kind, xv = labels_or_numbers(who, "x", x, data)
    H = numbers_(who, "height", height, data)
    n = same_length(who, x=xv, height=H)
    width, label = real(width, who, "width", positive=True), _label(who, label)
    gap = real(gap, who, "gap")
    if not 0.0 <= gap < 1.0:
        raise ValueError(f"{who}(): 'gap' must be in [0, 1)")
    _check_bool(horizontal, who, "horizontal")
    _check_bool(stack, who, "stack")
    annotate = _annotate(who, annotate)
    if err is not None:
        if stack:
            raise ValueError(f"{who}(): 'err' cannot be given for stacked bars")
        if isinstance(err, tuple):
            if len(err) != 2:
                raise ValueError(f"{who}(): 'err' as a tuple must be (below, above), not {len(err)} items")
            below, above = numbers_(who, "err[0]", err[0], data), numbers_(who, "err[1]", err[1], data)
        else:
            below = above = numbers_(who, "err", err, data)
        same_length(who, height=H, err=below)
        same_length(who, height=H, err=above)
        if bool(sd.any(below < 0.0)) or bool(sd.any(above < 0.0)):
            raise ValueError(f"{who}(): 'err' must be >= 0")
    cell = self._cell
    parts = _each(self, who, n, group, data, color, label)
    if kind == "cat":
        pos = _categories(cell, "y" if horizontal else "x", xv)
    else:
        if (cell.ycats if horizontal else cell.xcats) is not None:
            raise ValueError(f"{who}(): 'x' must be labels: the axis already has categories")
        pos = xv
    k = len(parts)
    up, down = {}, {}  # stacked: the running ends above and below zero at each position
    for j, (idx, col, lab) in enumerate(parts):
        P, V = _take(pos, idx), _take(H, idx)
        values = V.tolist()
        text = [value_text(v, annotate, who) for v in values] if annotate is not False else None
        if stack:
            bases = []
            for p, v in zip(P.tolist(), values):
                run = up if v >= 0 else down
                bases.append(run.get(p, 0.0))
                run[p] = run.get(p, 0.0) + (v if v == v else 0.0)
            cell.layers.append(
                _layers.Bars(
                    P, V, sd.asarray(bases), width, horizontal, _color_of(self, who, col), lab, None, text, True
                )
            )
            continue
        w = width / k
        off = (j - (k - 1) / 2) * w  # the groups' bars side by side around the position, a gap between them
        e = None
        if err is not None:
            e = (V - _take(below, idx), V + _take(above, idx))
        cell.layers.append(
            _layers.Bars(
                P + off, V, 0.0, w * (1.0 - gap) if k > 1 else w, horizontal, _color_of(self, who, col), lab, e, text
            )
        )
    return self._touch()


def area(self, x, y1, y2=0.0, /, *, data=None, color=None, alpha=0.3, label=None):
    who = "area"
    _check_data(who, data)
    X, A = numbers_(who, "x", x, data), numbers_(who, "y1", y1, data)
    n = same_length(who, x=X, y1=A)
    b = _real_option(y2, who, "y2")  # a number, numpy's scalars among them
    if b is not None:
        B = sd.full((n,), real(b, who, "y2"))
    else:
        B = numbers_(who, "y2", y2, data)
        same_length(who, x=X, y2=B)
    self._cell.layers.append(_layers.Area(X, A, B, _color_of(self, who, color, _alpha(who, alpha)), _label(who, label)))
    return self._touch()


def step(self, x, y, /, *, data=None, group=None, where="post", color=None, width=None, style="-", label=None):
    who = "step"
    _check_data(who, data)
    X, Y = numbers_(who, "x", x, data), numbers_(who, "y", y, data)
    n = same_length(who, x=X, y=Y)
    _check_choice(where, who, "where", ("post", "pre", "mid"))
    width, style, label = _opt_real(who, "width", width), _style(who, style), _label(who, label)
    for idx, col, lab in _each(self, who, n, group, data, color, label):
        xs, ys = _take(X, idx), _take(Y, idx)
        m = xs.shape[0]
        if m >= 2:
            if where == "mid":
                mid = (xs[:-1] + xs[1:]) * 0.5
                ex = sd.concat([xs[:1], sd.reshape(sd.stack([mid, mid], axis=1), (-1,)), xs[-1:]])
                ey = sd.reshape(sd.stack([ys, ys], axis=1), (-1,))
            elif where == "post":  # the value holds until the next x
                ex = sd.concat([xs[:1], sd.reshape(sd.stack([xs[1:], xs[1:]], axis=1), (-1,))])
                ey = sd.concat([sd.reshape(sd.stack([ys[:-1], ys[:-1]], axis=1), (-1,)), ys[-1:]])
            else:  # the value holds since the previous x
                ex = sd.concat([sd.reshape(sd.stack([xs[:-1], xs[:-1]], axis=1), (-1,)), xs[-1:]])
                ey = sd.concat([ys[:1], sd.reshape(sd.stack([ys[1:], ys[1:]], axis=1), (-1,))])
            xs, ys = ex, ey
        self._cell.layers.append(_layers.Line(xs, ys, _color_of(self, who, col), width, style, lab))
    return self._touch()


def _errors(who, arg, v, centre, data):
    """(lo, hi) arrays of an error argument around `centre`: a number, n numbers, or 2 x n."""
    if v is None:
        return None, None
    n = centre.shape[0]
    e = _real_option(v, who, arg)  # a number, numpy's scalars among them
    if e is not None:
        e = real(e, who, arg)
        return centre - e, centre + e
    if not isinstance(v, str):
        try:
            a = _as_array(v, who, arg)
        except TypeError:
            a = None
        if a is not None and a.ndim == 2:
            if tuple(a.shape) != (2, n):
                raise ValueError(
                    f"{who}(): '{arg}' must be a number, {n} numbers or shape (2, {n}), not shape {tuple(a.shape)}"
                )
            return centre - a[0], centre + a[1]
    e = numbers_(who, arg, v, data)
    if e.shape[0] != n:
        raise ValueError(f"{who}(): '{arg}' must be a number, {n} numbers or shape (2, {n}), not {e.shape[0]} numbers")
    return centre - e, centre + e


def errorbar(self, x, y, /, *, yerr=None, xerr=None, data=None, group=None, color=None, cap=2.0, label=None):
    who = "errorbar"
    _check_data(who, data)
    X, Y = numbers_(who, "x", x, data), numbers_(who, "y", y, data)
    n = same_length(who, x=X, y=Y)
    cap = real(cap, who, "cap")
    if cap < 0.0:
        raise ValueError(f"{who}(): 'cap' must be >= 0")
    ylo, yhi = _errors(who, "yerr", yerr, Y, data)
    xlo, xhi = _errors(who, "xerr", xerr, X, data)
    label = _label(who, label)
    for idx, col, lab in _each(self, who, n, group, data, color, label):
        t = lambda a: None if a is None else _take(a, idx)  # noqa: E731
        self._cell.layers.append(
            _layers.ErrorBars(
                _take(X, idx), _take(Y, idx), t(ylo), t(yhi), t(xlo), t(xhi), _color_of(self, who, col), cap, None, lab
            )
        )
    return self._touch()


def _rule(self, who, value, vertical, color, width, style, label):
    v = real(value, who, "x" if vertical else "y")
    c = _scene.INK if color is None else _color.parse(color, who, "color", self._root._palette)
    self._cell.layers.append(
        _layers.Rule(v, vertical, c, _opt_real(who, "width", width), _style(who, style), _label(who, label))
    )
    return self._touch()


def hline(self, y, /, *, color=None, width=None, style="-", label=None):
    return _rule(self, "hline", y, False, color, width, style, label)


def vline(self, x, /, *, color=None, width=None, style="-", label=None):
    return _rule(self, "vline", x, True, color, width, style, label)


def text_(self, x, y, s, /, *, ha="center", va="center", size=None, color=None, rotation=0.0):
    who = "text"
    c = _scene.INK if color is None else _color.parse(color, who, "color", self._root._palette)
    _check_choice(ha, who, "ha", _HA)
    _check_choice(va, who, "va", _VA)
    self._cell.layers.append(
        _layers.Label(
            string(s, who, "s"),
            real(x, who, "x"),
            real(y, who, "y"),
            ha,
            va,
            _opt_real(who, "size", size),
            c,
            real(rotation, who, "rotation"),
        )
    )
    return self._touch()


def _pair(who, arg, v):
    if not isinstance(v, (tuple, list)) or len(v) != 2:
        raise TypeError(f"{who}(): '{arg}' must be a pair of numbers (x, y), not {_type_name(v)}")
    return real(v[0], who, arg), real(v[1], who, arg)


def annotate(self, s, xy, xytext, /, *, arrow=True, size=None, color=None):
    who = "annotate"
    c = _scene.INK if color is None else _color.parse(color, who, "color", self._root._palette)
    target, at = _pair(who, "xy", xy), _pair(who, "xytext", xytext)
    _check_bool(arrow, who, "arrow")
    self._cell.layers.append(
        _layers.Label(
            string(s, who, "s"),
            at[0],
            at[1],
            "center",
            "center",
            _opt_real(who, "size", size),
            c,
            0.0,
            target if arrow else None,
        )
    )
    return self._touch()


# ---- module functions from the methods


def new_figure(who, options):
    """A new Figure from the figure options a module function was given."""
    for k in options:
        if k not in FIGURE_OPTIONS:
            raise TypeError(f"{who}(): unexpected keyword argument {k!r}")
    fig = object.__new__(Figure)
    return fig._init(
        who,
        options.get("width", 89.0),
        options.get("height"),
        1,
        1,
        options.get("font", "serif"),
        options.get("theme", "paper"),
        options.get("palette", "jco"),
    )


def module_function(name, method):
    """The module function of a Figure method: a new figure, then the method. A wrong call shape
    is reported as the other areas report it ("line(): unexpected keyword argument 'foo'")."""
    sig = inspect.signature(method)
    params = list(sig.parameters.values())[1:]
    public = inspect.Signature(params + [inspect.Parameter("figure", inspect.Parameter.VAR_KEYWORD)])
    names = {p.name for p in params if p.kind is inspect.Parameter.KEYWORD_ONLY}
    bound = _calls.method(name)(method)

    def f(*args, **kw):
        options = {k: kw.pop(k) for k in list(kw) if k in FIGURE_OPTIONS and k not in names}
        return bound(new_figure(name, options), *args, **kw)

    f.__name__ = f.__qualname__ = name
    f.__signature__ = public
    f.__doc__ = method.__doc__
    return f


def _share_arg(who, name, v):
    if v is False or v is True or v == "all":
        return v
    raise ValueError(f"{who}(): '{name}' must be False, True or 'all', not {v!r}")


def grid(figures, /, *, nrows=None, ncols=None, labels=None, sharex=False, sharey=False, width=None, height=None):
    """Gathers figures into the cells of a new one (docs/api/plot.md §2.1)."""
    who = "grid"
    if not isinstance(figures, (list, tuple)) or not figures or not all(isinstance(f, Figure) for f in figures):
        raise TypeError(f"{who}(): 'figures' must be a non-empty sequence of Figures, not {_type_name(figures)}")
    cells = []
    for f in figures:
        # copies: the figures given stay as they were, and changing the grid leaves them alone
        # (a view gives its cell, whole: of a twin() view too, both y axes)
        cells += [c.copy() for c in ([f._base] if f._root is not f else f._root._cells)]
    n = len(cells)
    if nrows is None and ncols is None:
        nc = min(n, 2) if n <= 4 else 3
        nr = -(-n // nc)
    elif nrows is None:
        nc = integer(ncols, who, "ncols", 1)
        nr = -(-n // nc)
    elif ncols is None:
        nr = integer(nrows, who, "nrows", 1)
        nc = -(-n // nr)
    else:
        nr, nc = integer(nrows, who, "nrows", 1), integer(ncols, who, "ncols", 1)
        if nr * nc < n:
            raise ValueError(f"{who}(): {nr} x {nc} cells cannot hold {n} figures")
    share = tuple(_share_arg(who, name, v) for name, v in (("sharex", sharex), ("sharey", sharey)))
    first = figures[0]._root
    out = object.__new__(Figure)
    out._init(who, first._w * nc if width is None else width, height, nr, nc, first._font, first._theme, first._palette)
    for i, c in enumerate(cells):
        out._cells[i] = c
    for i in range(n, nr * nc):
        empty = Cell()
        empty.frame = False
        out._cells[i] = empty
    out._share = share
    for f in figures:
        out._infos += f._root._infos
    if labels is not None:
        if labels in ("A", "a"):
            base = ord(labels)
            out._panels = [chr(base + i) if i < n else "" for i in range(nr * nc)]
        elif isinstance(labels, (list, tuple)) and all(isinstance(s, str) for s in labels) and len(labels) == n:
            out._panels = list(labels) + [""] * (nr * nc - n)
        else:
            raise ValueError(f"{who}(): 'labels' must be 'A', 'a' or {n} strings")
    return out


METHODS = {
    "line": line,
    "scatter": scatter,
    "bar": bar,
    "area": area,
    "step": step,
    "errorbar": errorbar,
    "hline": hline,
    "vline": vline,
    "text": text_,
    "annotate": annotate,
}


def flat_only(name, method):
    """A 2D plot method that refuses a cell holding a 3D plot (the other way round, the 3D
    methods refuse a cell with a 2D plot)."""

    @functools.wraps(method)
    def guarded(self, *args, **kw):
        if any(type(layer).__name__ == "Space" for layer in self._cell.layers):
            raise ValueError(f"{name}(): the cell already holds a 3D plot")
        return method(self, *args, **kw)

    return guarded


def install(methods=None, flat=True):
    """Attaches the methods to Figure and returns the module functions. `flat`: they draw 2D
    plots, which a 3D cell does not take."""
    out = {}
    for name, method in (METHODS if methods is None else methods).items():
        method.__name__ = name
        method.__qualname__ = f"Figure.{name}"
        setattr(Figure, name, flat_only(name, method) if flat else method)
        out[name] = module_function(name, method)  # (a new figure holds no 3D plot to refuse)
    return out
