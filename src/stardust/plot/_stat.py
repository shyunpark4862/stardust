"""Statistical plots and the plots of results (docs/api/plot.md §5, §6; docs/design/plot.md §5).

The numbers a plot shows are the computing areas' own: histogram bars are sd.histogram's
counts, a box is sd.quantile's quartiles, QQ points are a distribution's icdf, a fitted line is
the regression result's predict. A test's result is drawn as given (compare): the plot does not
choose or run a test.
"""

import math
import numbers
import operator
from array import array

from .. import _core as sd
from .._core import _type_name
from ..optimize._common import _check_bool, _check_choice
from . import _basic, _color, _field, _figure, _function, _layers, _scene
from ._basic import _alpha, _each, _label, _opt_real, _take, escape, numbers_, same_length, split
from ._figure import Cell, Figure, integer, real

INK = _scene.INK


def _nonan(a):
    return a[sd.isfinite(a)]


# ---- histogram, ecdf


class Hist(_layers.Layer):
    """Bars between bin edges, standing on `base` (zero; a pairs plot's diagonal stands them on
    the low end of its row's range). Filled as bars are: the colour's tint, the colour as a thin edge."""

    def __init__(self, edges, heights, color, label, base=0.0):
        self.edges, self.heights, self.color, self.label, self.base = edges, heights, color, label, base
        self.ceiling = None  # the top of the y range, when it is given with the base (a pairs plot's row)

    def extent(self, axis, positive):
        if axis == "x":
            return (self.edges[0], self.edges[-1], ())
        if self.ceiling is not None and not positive:
            return (self.base, self.ceiling, (self.base, self.ceiling))
        top = max(self.heights, default=0.0) + self.base
        return (self.base, top, (self.base,)) if not positive or self.base > 0.0 else None

    def draw(self, fr):
        ops, xy = [], array("d")
        y0 = fr.py(self.base)
        for a, b, h in zip(self.edges, self.edges[1:], self.heights):
            if h > 0.0:
                x0, x1, y1 = fr.px(a), fr.px(b), fr.py(self.base + h)
                ops.append("MLLLZ")
                xy.extend((x0, y0, x1, y0, x1, y1, x0, y1))
        if ops:
            fill = _color.fill_of(self.color)
            edge = (*self.color[:3], min(1.0, self.color[3] + 0.25))
            fr.scene.add(
                _scene.Path("".join(ops), xy, fill=fill, stroke=edge, width=fr.theme["edge_width"], clip=fr.clip)
            )

    def entries(self):
        return [(self.label, _layers._patch_handle(_color.fill_of(self.color), self.color))]


def histogram(
    self, x, /, *, data=None, group=None, bins=10, range=None, density=False, color=None, alpha=None, label=None
):
    who = "histogram"
    _basic._check_data(who, data)
    X = numbers_(who, "x", x, data)
    _check_bool(density, who, "density")
    label = _label(who, label)
    parts = _each(self, who, X.shape[0], group, data, color, label)
    a = (1.0 if len(parts) == 1 else 0.6) if alpha is None else _alpha(who, alpha)
    allx = _nonan(X)
    if range is None and allx.shape[0]:
        range = (float(sd.min(allx)), float(sd.max(allx)))  # one set of bins for every group
    for idx, col, lab in parts:
        v = _nonan(_take(X, idx))
        try:
            r = sd.histogram(v, bins=bins, range=range, density=density)
        except (TypeError, ValueError) as e:
            raise type(e)(str(e).replace("histogram():", f"{who}():", 1)) from None
        self._cell.layers.append(
            Hist(r.bin_edges.tolist(), [float(h) for h in r.hist.tolist()], _basic._color_of(self, who, col, a), lab)
        )
    return self._touch()


def ecdf(self, x, /, *, data=None, group=None, color=None, width=None, label=None):
    who = "ecdf"
    _basic._check_data(who, data)
    X = numbers_(who, "x", x, data)
    width, label = _opt_real(who, "width", width), _label(who, label)
    for idx, col, lab in _each(self, who, X.shape[0], group, data, color, label):
        v = sd.sort(_nonan(_take(X, idx)))
        n = v.shape[0]
        if n:
            # the proportion at or below each value: it rises at the value and holds to the next
            p = sd.arange(1.0, n + 1.0) / n
            xs = sd.reshape(sd.stack([v, v], axis=1), (-1,))
            ys = sd.concat([sd.zeros(1), sd.reshape(sd.stack([p[:-1], p[:-1]], axis=1), (-1,)), p[-1:]])
        else:
            xs = ys = sd.zeros(0)
        self._cell.layers.append(_layers.Line(xs, ys, _basic._color_of(self, who, col), width, "-", lab))
    return self._touch()


# ---- boxplot, strip


class Boxes(_layers.Layer):
    """Box-and-whisker glyphs: per box its position, the quartiles, the whisker ends and the
    points beyond them."""

    def __init__(self, stats, width, color, label, show_points):
        self.stats, self.width, self.color, self.label, self.show_points = stats, width, color, label, show_points

    def extent(self, axis, positive):
        if not self.stats:
            return None
        if axis == "x":
            return (
                min(s["pos"] for s in self.stats) - self.width / 2,
                max(s["pos"] for s in self.stats) + self.width / 2,
                (),
            )
        lo = min(min([s["lo"]] + s["out"] + s["pts"]) for s in self.stats)
        hi = max(max([s["hi"]] + s["out"] + s["pts"]) for s in self.stats)
        return (lo, hi, ())

    def draw(self, fr):
        half = self.width / 2
        boxes, bxy, lines, lxy, med, mxy, at = [], array("d"), [], array("d"), [], array("d"), array("d")
        for s in self.stats:
            x0, x1, xc = fr.px(s["pos"] - half), fr.px(s["pos"] + half), fr.px(s["pos"])
            q1, q2, q3, lo, hi = (fr.py(s[k]) for k in ("q1", "q2", "q3", "lo", "hi"))
            boxes.append("MLLLZ")
            bxy.extend((x0, q1, x1, q1, x1, q3, x0, q3))
            lines.append("MLML")
            lxy.extend((xc, q1, xc, lo, xc, q3, xc, hi))
            med.append("ML")
            mxy.extend((x0, q2, x1, q2))
            for v in s["pts"] if self.show_points else s["out"]:
                at.extend((xc, fr.py(v)))
        if not boxes:
            return
        w = fr.theme["axis_width"]
        fr.scene.add(_scene.Path("".join(lines), lxy, stroke=self.color, width=w, clip=fr.clip))
        fr.scene.add(
            _scene.Path(
                "".join(boxes), bxy, fill=_color.with_alpha(self.color, 0.25), stroke=self.color, width=w, clip=fr.clip
            )
        )
        fr.scene.add(_scene.Path("".join(med), mxy, stroke=self.color, width=2.2 * w, clip=fr.clip))
        if at:
            fr.scene.add(_layers.marker_item("o", 2.2, at, self.color, None, 0.0, None))

    def entries(self):
        return [(self.label, _layers._patch_handle(_color.with_alpha(self.color, 0.25), self.color))]

    def sample(self, fr):
        return [(fr.px(s["pos"]), fr.py(s["hi"])) for s in self.stats]


def box_stats(v):
    """The five numbers of a box: quartiles (sd.quantile, method 'linear'), the whiskers (the
    farthest values within 1.5 IQR of the box) and the values beyond."""
    v = sd.sort(_nonan(v))
    if v.shape[0] == 0:
        return None
    q1, q2, q3 = (float(q) for q in sd.quantile(v, sd.asarray([0.25, 0.5, 0.75])))
    iqr = q3 - q1
    vals = v.tolist()
    inside = [t for t in vals if q1 - 1.5 * iqr <= t <= q3 + 1.5 * iqr]
    lo, hi = min(inside, default=q1), max(inside, default=q3)
    return {
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "lo": min(lo, q1),
        "hi": max(hi, q3),
        "out": [t for t in vals if t < lo or t > hi],
        "pts": vals,
    }


def _by_category(self, who, x, y, data, group, color, label):
    """[(positions per category, index arrays per category, colour, label, offset, width share)]
    for a plot of values y over categories x (None: one category)."""
    _basic._check_data(who, data)
    Y = numbers_(who, "y", y, data)
    n = Y.shape[0]
    cell = self._cell
    if x is None:
        cats = [""] * n
    else:
        kind, cats = _basic.labels_or_numbers(who, "x", x, data)
        if kind != "cat":
            cats = [format(v, "g") for v in cats.tolist()]
        same_length(who, x=cats, y=Y)
    order = list(dict.fromkeys(cats))
    pos = dict(zip(order, _basic._categories(cell, "x", order).tolist())) if x is not None else {"": 0.0}
    if x is None and cell.xcats is None:
        cell.xcats = [""]
    parts = _each(self, who, n, group, data, color, label)
    out = []
    for j, (idx, col, lab) in enumerate(parts):
        rows = range(n) if idx is None else idx.tolist()
        by = {}
        for i in rows:
            by.setdefault(cats[i], []).append(i)
        out.append((Y, [(pos[c], sd.asarray(by[c])) for c in order if c in by], col, lab, j, len(parts)))
    return out


def boxplot(self, x, y, /, *, data=None, group=None, points=False, color=None, width=0.6):
    who = "boxplot"
    _check_bool(points, who, "points")
    width = real(width, who, "width", positive=True)
    for Y, cats, col, lab, j, k in _by_category(self, who, x, y, data, group, color, None):
        w = width / k
        off = (j - (k - 1) / 2) * w
        stats = []
        for p, idx in cats:
            s = box_stats(sd.take(Y, idx))
            if s is not None:
                s["pos"] = p + off
                stats.append(s)
        self._cell.layers.append(Boxes(stats, w * 0.9, _basic._color_of(self, who, col), lab, points))
    return self._touch()


def strip(self, x, y, /, *, data=None, group=None, jitter=0.2, rng=None, color=None, size=None, alpha=1.0):
    who = "strip"
    jitter = real(jitter, who, "jitter")
    if not 0.0 <= jitter <= 1.0:
        raise ValueError(f"{who}(): 'jitter' must be in [0, 1]")
    size, a = _opt_real(who, "size", size), _alpha(who, alpha)
    from .. import random as _random

    if rng is None:
        gen = _random.default_rng(0)  # a fixed seed: the same call is the same figure
    elif isinstance(rng, _random.Generator):
        gen = rng
    elif isinstance(rng, numbers.Integral) and not isinstance(rng, bool):  # (numpy's integers too)
        gen = _random.default_rng(operator.index(rng))
    else:
        raise TypeError(f"{who}(): 'rng' must be an sd.random Generator, an int seed or None, not {_type_name(rng)}")
    for Y, cats, col, lab, j, k in _by_category(self, who, x, y, data, group, color, None):
        off = (j - (k - 1) / 2) * (0.8 / k)
        xs, ys = [], []
        for p, idx in cats:
            m = idx.shape[0]
            xs.append((gen.random(m) - 0.5) * (jitter / k) + (p + off))
            ys.append(sd.take(Y, idx))
        X = sd.concat(xs) if xs else sd.zeros(0)
        V = sd.concat(ys) if ys else sd.zeros(0)
        self._cell.layers.append(_layers.Scatter(X, V, _basic._color_of(self, who, col, a), size, "o", lab))
    return self._touch()


# ---- qq, heatmap


def qq(self, x, /, *, dist=None, data=None, line=True, color=None):
    who = "qq"
    _basic._check_data(who, data)
    v = sd.sort(_nonan(numbers_(who, "x", x, data)))
    _check_bool(line, who, "line")
    n = v.shape[0]
    if n < 2:
        raise ValueError(f"{who}(): 'x' needs at least 2 finite values")
    if dist is None:
        from .. import stats

        dist = stats.Normal(mu=0.0, sigma=1.0)
    elif not hasattr(dist, "icdf"):
        raise TypeError(f"{who}(): 'dist' must be an sd.stats distribution, not {_type_name(dist)}")
    p = (sd.arange(1.0, n + 1.0) - 0.5) / n
    q = dist.icdf(p)
    c = _basic._color_of(self, who, color)
    if line:
        # through the quartiles of both, as R's qqline
        y1, y3 = (float(t) for t in sd.quantile(v, sd.asarray([0.25, 0.75])))
        x1, x3 = float(dist.icdf(0.25)), float(dist.icdf(0.75))
        if x3 > x1:
            k = (y3 - y1) / (x3 - x1)
            a, b = float(q[0]), float(q[-1])
            self._cell.layers.append(
                _layers.Line(
                    sd.asarray([a, b]),
                    sd.asarray([y1 + k * (a - x1), y1 + k * (b - x1)]),
                    (0.4, 0.4, 0.4, 1.0),
                    0.8,
                    "--",
                    None,
                )
            )
    self._cell.layers.append(_layers.Scatter(q, v, c, None, "o", None))
    cell = self._cell
    if cell.xlabel is None:
        cell.xlabel = "Theoretical quantiles"
    if cell.ylabel is None:
        cell.ylabel = "Sample quantiles"
    return self._touch()


class CellText(_layers.Layer):
    """A number in each cell of a heat map."""

    def __init__(self, items, size):
        self.items, self.size = items, size  # (string, x, y, colour)

    def draw(self, fr):
        for s, x, y, c in self.items:
            fr.scene.add(_scene.Text(s, fr.px(x), fr.py(y), size=self.size, ha="center", va="center", color=c))


def _strings(who, arg, v, n):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != n or not all(isinstance(s, str) for s in v):
        raise ValueError(f"{who}(): '{arg}' must be {n} strings")
    return list(v)


def heatmap(self, z, /, *, xlabels=None, ylabels=None, cmap=None, vmin=None, vmax=None, annotate=False, colorbar=True):
    who = "heatmap"
    from ..optimize._common import _as_array

    Z = _as_array(z, who, "z")
    if Z.ndim != 2 or Z.shape[0] == 0 or Z.shape[1] == 0:
        raise ValueError(f"{who}(): 'z' must be 2-dimensional and not empty, not shape {tuple(Z.shape)}")
    nr, nc = Z.shape
    xl, yl = _strings(who, "xlabels", xlabels, nc), _strings(who, "ylabels", ylabels, nr)
    name = _field._cmap(who, cmap)
    annotate = _basic._annotate(who, annotate)
    _check_bool(colorbar, who, "colorbar")
    r = _layers.finite_range(sd.reshape(Z, (-1,))) or (0.0, 1.0)
    lo = r[0] if vmin is None else real(vmin, who, "vmin")
    hi = r[1] if vmax is None else real(vmax, who, "vmax")
    if not lo < hi:
        if vmin is not None and vmax is not None:
            raise ValueError(f"{who}(): 'vmin' must be less than 'vmax'")
        lo, hi = lo - 0.5, lo + 0.5
    from .._core import plot as core

    rgba = core.colormap(bytes(memoryview(sd.reshape(Z + 0.0, (-1,)))), lo, hi, _color.table(name))
    cell = self._cell
    # row 0 of the matrix is the top row; the cells are centred on 0 .. n-1
    cell.layers.append(_field.Picture((-0.5, nc - 0.5, -0.5, nr - 0.5), nc, nr, rgba, False))
    cell.xticks = (list(range(nc)), xl if xl is not None else [f"${i}$" for i in range(nc)])
    cell.yticks = ([nr - 1 - i for i in range(nr)], yl if yl is not None else [f"${i}$" for i in range(nr)])
    if colorbar:
        cell.colorbar = (name, lo, hi)
    if annotate is not False:
        vals = Z.tolist()
        flat = [v for row in vals for v in row if math.isfinite(v)]
        # each with its own digits: a small value does not lengthen the rest
        text = [_basic.value_text(v, annotate, who) for v in flat]
        it = iter(text)
        items = []
        for i, row in enumerate(vals):
            for j, v in enumerate(row):
                if math.isfinite(v):
                    c = _color.cmap(name, (v - lo) / (hi - lo))
                    lum = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
                    items.append((next(it), float(j), float(nr - 1 - i), INK if lum > 0.55 else (1.0, 1.0, 1.0, 1.0)))
        layer = CellText(items, 6.0)
        layer.strings = lambda: [s for s, *_ in items]
        cell.layers.append(layer)
    return self._touch()


# ---- results: compare, fit, diagnostics, coefficients, distribution


def p_text(p, fmt):
    """A p-value as text: 'p = 0.012' with summary()'s digits, or stars; n/a when the test gave none (NaN)."""
    if p != p:
        return "n/a"
    if fmt == "stars":
        return "ns" if not p <= 0.05 else "*" * (1 + (p <= 0.01) + (p <= 0.001))
    if p < 0.001:
        return r"$p < 0.001$"
    return f"$p = {p:.3f}$"


class Bracket(_layers.Layer):
    """A bracket over two positions of a category axis with a text above it."""

    def __init__(self, x1, x2, level, s):
        self.x1, self.x2, self.level, self.s = x1, x2, level, s
        self.total = level + 1  # brackets in the cell; compare() sets it for all of them

    def strings(self):
        return [self.s]

    def draw(self, fr):
        # the bracket stands in the room the layout kept above the data (cell.headroom); level 0
        # is the lowest, just over the data, and later ones stand above it
        y = fr.y + 9.0 + 13.0 * (self.total - 1 - self.level)
        a, b = fr.px(self.x1), fr.px(self.x2)
        fr.scene.add(
            _scene.Path(
                *_scene.polyline([(a, y + 3.0), (a, y), (b, y), (b, y + 3.0)]),
                stroke=INK,
                width=fr.theme["axis_width"],
            )
        )
        fr.scene.add(_scene.Text(self.s, (a + b) / 2, y - 2.0, size=fr.theme["tick_size"], ha="center", va="bottom"))


def compare(self, result, /, *, groups=None, format="p", y=None):
    who = "compare"
    _check_choice(format, who, "format", ("p", "stars"))
    cell = self._cell
    cats = cell.xcats
    if not cats:
        raise ValueError(f"{who}(): the figure has no category axis (compare draws over boxplot, strip or bar)")
    p = getattr(result, "pvalue", None)
    if p is None:
        raise TypeError(f"{who}(): 'result' must be a test result with a pvalue, not {_type_name(result)}")
    if y is not None:
        raise ValueError(f"{who}(): 'y' is not supported yet; brackets stand above the data")
    level = getattr(cell, "headroom", 0)
    if isinstance(p, sd.ndarray) and p.ndim == 2:
        k = p.shape[0]
        if groups is not None:
            raise ValueError(
                f"{who}(): 'groups' cannot be given for a multiple comparison (its groups are the categories in order)"
            )
        if k != len(cats):
            raise ValueError(f"{who}(): the result compares {k} groups, the figure has {len(cats)} categories")
        pv = p.tolist()
        pairs = sorted((j - i, i, j) for i in range(k) for j in range(i + 1, k))  # near pairs lowest
        for _, i, j in pairs:
            cell.layers.append(Bracket(float(i), float(j), level, p_text(float(pv[i][j]), format)))
            level += 1
    elif isinstance(p, sd.ndarray) and p.ndim == 1 and type(result).__name__ == "DunnettResult":
        # each group against the control: the control is the first category
        if groups is not None:
            raise ValueError(
                f"{who}(): 'groups' cannot be given for Dunnett's test (the control is the first category)"
            )
        if p.shape[0] != len(cats) - 1:
            raise ValueError(
                f"{who}(): the result compares {p.shape[0]} groups with a control, "
                f"the figure has {len(cats)} categories"
            )
        for j, v in enumerate(p.tolist()):
            cell.layers.append(Bracket(0.0, float(j + 1), level, p_text(float(v), format)))
            level += 1
    else:
        try:
            value = float(p)
        except TypeError, ValueError:
            raise ValueError(
                f"{who}(): 'result' has a pvalue of shape {tuple(p.shape)}; a batch of tests cannot be drawn, give one"
            ) from None
        if groups is None:
            if len(cats) != 2:
                raise ValueError(
                    f"{who}(): 'groups' must name the two categories compared (the figure has {len(cats)})"
                )
            i, j = 0, 1
        else:
            if not isinstance(groups, (tuple, list)) or len(groups) != 2:
                raise TypeError(f"{who}(): 'groups' must be a pair of category labels, not {_type_name(groups)}")
            idx = []
            for g in groups:
                e = escape(g)
                if e not in cats:
                    raise ValueError(f"{who}(): no category {g!r}")
                idx.append(cats.index(e))
            i, j = sorted(idx)
        cell.layers.append(Bracket(float(i), float(j), level, p_text(value, format)))
        level += 1
    cell.headroom = level
    for layer in cell.layers:
        if isinstance(layer, Bracket):
            layer.total = level
    return self._touch()


def _simple(result, who):
    """(names, intercept, slope) of a regression result with one explanatory variable."""
    names = list(getattr(result, "param_names", []))
    params = getattr(result, "params", None)
    if params is None or not hasattr(result, "predict"):
        raise TypeError(f"{who}(): 'result' must be a regression result, not {_type_name(result)}")
    if params.ndim != 1:
        raise ValueError(f"{who}(): a batch of fits cannot be drawn; give one fit")
    if len(names) == 2 and names[0] == "const":
        return True, float(params[0]), float(params[1])
    if len(names) == 1:
        return False, 0.0, float(params[0])
    raise ValueError(f"{who}(): 'result' must have one explanatory variable; it has parameters {names}")


_LINKS = {
    "identity": r"\mu",
    "log": r"\log\mu",
    "logit": r"\mathrm{logit}\,\mu",
    "probit": r"\Phi^{-1}(\mu)",
    "inverse": r"1/\mu",
    "inverse_power": r"1/\mu",
    "cloglog": r"\log(-\log(1-\mu))",
    "sqrt": r"\sqrt{\mu}",
}


def _coef(v):
    """A coefficient with three significant digits, in TeX: 1.2 x 10^-5 written as such."""
    s = f"{v:.3g}"
    if "e" not in s:
        return s
    m, e = s.split("e")
    e = int(e)
    return rf"10^{{{e}}}" if m == "1" else rf"{m}\times10^{{{e}}}"


def fit(self, result, /, *, band=True, alpha=0.05, equation=True, color=None):
    who = "fit"
    _check_bool(band, who, "band")
    _check_bool(equation, who, "equation")
    alpha = real(alpha, who, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{who}(): 'alpha' must be in (0, 1)")
    const, b0, b1 = _simple(result, who)
    cell = self._cell
    e = None
    for layer in cell.layers:
        r = layer.extent("x", False)
        if r is not None:
            e = (r[0], r[1]) if e is None else (min(e[0], r[0]), max(e[1], r[1]))
    if e is None:
        raise ValueError(f"{who}(): the figure has no data to take the x range from")
    palette = self._root._palette
    c = (
        _color.parse(color, who, "color", palette)
        if color is not None
        else (cell.layers[-1].color if hasattr(cell.layers[-1], "color") else INK)
    )
    xs = sd.linspace(e[0], e[1], 101)
    X = sd.reshape(xs, (-1, 1))
    if band and hasattr(result, "predict_interval"):
        import inspect

        known = inspect.signature(result.predict_interval).parameters
        kw = {"alpha": alpha}
        if "kind" in known:
            kw["kind"] = "mean"  # the band of the mean, not of a new observation
        elif "which" in known:
            kw["which"] = "mean"
        iv = result.predict_interval(X, **kw)
        cell.layers.append(_layers.Area(xs, iv[:, 0], iv[:, 1], _color.with_alpha(c, 0.2), None))
    cell.layers.append(_layers.Line(xs, result.predict(X), c, None, "-", None))
    if equation:
        sign = "-" if b0 < 0 else "+"
        # a GLM's line is the mean on the scale of its link: the equation says so
        lhs = _LINKS.get(getattr(result, "link", "identity"), r"g(\mu)") if hasattr(result, "family") else "y"
        eq = f"${lhs} = {_coef(b1)}x {sign} {_coef(abs(b0))}$" if const else f"${lhs} = {_coef(b1)}x$"
        r2 = getattr(result, "rsquared", None)
        if r2 is not None:
            eq += f", $R^2 = {float(r2):.3f}$"
        cell.note = eq
    return self._touch()


def diagnostics(result, /, *, width=None, height=None, font="serif", theme="paper", palette="jco"):
    """The four diagnostic plots of a regression fit: residuals against fitted values, the QQ
    plot of the standardized residuals, their root against fitted values, and residuals against
    leverage."""
    who = "diagnostics"
    for name in ("fittedvalues", "resid", "resid_studentized_internal", "leverage"):
        if not hasattr(result, name):
            raise TypeError(
                f"{who}(): 'result' must be a linear regression result (it has no {name}), not {_type_name(result)}"
            )
    if result.resid.ndim != 1:
        raise ValueError(f"{who}(): a batch of fits cannot be drawn; give one fit")
    fig = object.__new__(Figure)
    fig._init(who, 183.0 if width is None else width, height, 2, 2, font, theme, palette)
    fv, rs, st, lev = result.fittedvalues, result.resid, result.resid_studentized_internal, result.leverage
    grey = (0.4, 0.4, 0.4, 1.0)
    a = fig[0, 0]
    a.scatter(fv, rs).hline(0.0, style="--", width=0.5, color=grey).xlabel("Fitted values").ylabel("Residuals").title(
        "Residuals vs fitted"
    )
    b = fig[0, 1]
    b.qq(st).ylabel("Standardized residuals").title("Normal Q--Q")
    c = fig[1, 0]
    c.scatter(fv, sd.sqrt(sd.abs(st))).xlabel("Fitted values").ylabel(
        r"$\sqrt{|\mathrm{standardized\ residuals}|}$"
    ).title("Scale--location")
    d = fig[1, 1]
    d.scatter(lev, st).hline(0.0, style="--", width=0.5, color=grey).xlabel("Leverage").ylabel(
        "Standardized residuals"
    ).title("Residuals vs leverage")
    return fig


def coefficients(self, result, /, *, alpha=0.05, intercept=False, color=None):
    who = "coefficients"
    alpha = real(alpha, who, "alpha")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"{who}(): 'alpha' must be in (0, 1)")
    _check_bool(intercept, who, "intercept")
    if not hasattr(result, "params"):
        raise TypeError(f"{who}(): 'result' must be a regression result, not {_type_name(result)}")
    params = result.params
    if params.ndim != 1:
        raise ValueError(f"{who}(): a batch of fits cannot be drawn; give one fit")
    names = list(getattr(result, "param_names", [f"x{i}" for i in range(params.shape[0])]))
    # a fit without intervals (ridge) shows its estimates alone
    ci = result.conf_int(alpha=alpha).tolist() if hasattr(result, "conf_int") else [[v, v] for v in params.tolist()]
    est = params.tolist()
    keep = [i for i, n in enumerate(names) if intercept or n != "const"]
    if not keep:
        raise ValueError(f"{who}(): the fit has no coefficient besides the intercept (intercept=True shows it)")
    cell = self._cell
    labels = [escape(names[i]) for i in keep]
    pos = _basic._categories(cell, "y", labels[::-1])  # the first coefficient on top
    c = _basic._color_of(self, who, color)
    y = sd.asarray(pos.tolist()[::-1])
    x = sd.asarray([est[i] for i in keep])
    lo, hi = sd.asarray([ci[i][0] for i in keep]), sd.asarray([ci[i][1] for i in keep])
    cell.layers.append(_layers.Rule(0.0, True, (0.4, 0.4, 0.4, 1.0), 0.5, "--", None))
    cell.layers.append(_layers.ErrorBars(x, y, None, None, lo, hi, c, 0.0, None, None))
    return self._touch()


def distribution(self, dist, /, *, kind="pdf", a=None, b=None, color=None, label=None):
    who = "distribution"
    _check_choice(kind, who, "kind", ("pdf", "cdf", "sf"))
    if not hasattr(dist, "icdf") or not hasattr(dist, "cdf"):
        raise TypeError(f"{who}(): 'dist' must be an sd.stats distribution, not {_type_name(dist)}")
    discrete = hasattr(dist, "pmf")
    try:
        lo, hi = (float(v) for v in dist.support())
        qa, qb = float(dist.icdf(0.001)), float(dist.icdf(0.999))
    except TypeError:
        raise ValueError(f"{who}(): a batch of distributions cannot be drawn; give one") from None
    a = (lo if math.isfinite(lo) and lo >= qa - 0.25 * (qb - qa) else qa) if a is None else real(a, who, "a")
    b = (hi if math.isfinite(hi) and hi <= qb + 0.25 * (qb - qa) else qb) if b is None else real(b, who, "b")
    if not a < b:
        raise ValueError(f"{who}(): 'a' must be less than 'b'")
    label = _label(who, label)
    cell = self._cell
    if discrete:
        ks = sd.arange(float(math.ceil(a)), float(math.floor(b)) + 1.0)
        v = dist.pmf(ks) if kind == "pdf" else dist.cdf(ks) if kind == "cdf" else dist.ccdf(ks)
        if kind == "pdf":
            cell.layers.append(_layers.Bars(ks, v, 0.0, 0.8, False, _basic._color_of(self, who, color), label))
        else:
            self.step(ks, v, where="post", color=color, label=label)
        if cell.xticks is None:  # values live on the integers: so do the ticks
            cell.xticks = (_integer_ticks(float(ks[0]), float(ks[-1])), None) if ks.shape[0] else None
        return self._touch()
    f = dist.pdf if kind == "pdf" else dist.cdf if kind == "cdf" else dist.ccdf
    ylim = None
    if kind == "pdf":
        # A density's peak is what the plot is about: the range reaches it, unless the density
        # runs away at an end of the range (a pole, as Gamma's with a < 1 at 0).
        t = sd.linspace(a, b, 2001)
        v = f(t)
        fin = sd.isfinite(v)
        if bool(sd.all(fin[1:-1])):
            inner = v[1:-1]
            top = float(sd.max(inner))
            ends = float(sd.max(sd.where(fin[[0, -1]], v[[0, -1]], 0.0)))
            k = int(sd.argmax(inner))
            if top > 0.0 and ends <= 1.05 * top and 10 < k < inner.shape[0] - 10:
                ylim = (0.0, 1.05 * top)
    held = list(cell.ylim)
    _function.function(self, f, a, b, ylim=ylim, color=color, label=label)
    if ylim is not None:
        # the curve asks for that range; the cell's own stays free, so that a taller layer of the
        # cell (the histogram under the density, another density) is not cut
        cell.ylim = held
        cell.layers[-1].reach = ylim
    return self


def _integer_ticks(lo, hi):
    """Ticks on the integers from lo to hi: a step of 1, 2 or 5 times a power of ten, the
    smallest that gives six ticks or fewer."""
    span, step, p = max(hi - lo, 1.0), 1, 1
    while span / step > 6:
        step = next(m * p for m in (1, 2, 5, 10) if span / (m * p) <= 6 or m == 10)
        p *= 10
    first = math.ceil(lo / step) * step
    return [float(v) for v in range(int(first), int(math.floor(hi)) + 1, step)]


def pairs(data, /, *, columns=None, group=None, width=None, height=None, font="serif", theme="paper", palette="jco"):
    """A matrix of plots of the numeric columns of a DataFrame: histograms on the diagonal,
    scatter plots elsewhere."""
    who = "pairs"
    from .. import frame

    if not isinstance(data, frame.DataFrame):
        raise TypeError(f"{who}(): 'data' must be a DataFrame, not {_type_name(data)}")
    if columns is None:
        columns = [c for c in data.columns if data[c].dtype in (sd.float64, sd.int64)]
        if isinstance(group, str) and group in columns:
            columns.remove(group)
    elif not isinstance(columns, (list, tuple)) or not all(isinstance(c, str) for c in columns):
        raise TypeError(f"{who}(): 'columns' must be a sequence of column names, not {_type_name(columns)}")
    k = len(columns)
    if k < 2:
        raise ValueError(f"{who}(): at least 2 numeric columns are needed, not {k}")
    cols = {c: numbers_(who, "columns", c, data) for c in columns}
    fig = object.__new__(Figure)
    fig._init(who, 183.0 if width is None else width, height, k, k, font, theme, palette)
    n = cols[columns[0]].shape[0]
    parts = split(who, group, data, n)
    for i, ci in enumerate(columns):
        for j, cj in enumerate(columns):
            cell = fig[i, j]
            if i == j:
                cell.histogram(
                    cols[ci], group=None if parts is None else group, data=data if isinstance(group, str) else None
                )
                # The bars span the row's range, so that the row's y ticks read as its variable
                # throughout: they stand on the low end of the range the row's scatter plots take
                # (the values with the theme's margin), the tallest reaches the largest value.
                r = _layers.finite_range(cols[ci])
                if r is not None and r[1] > r[0]:
                    hists = [layer for layer in cell._cell.layers if isinstance(layer, Hist)]
                    top = max((max(h.heights, default=0.0) for h in hists), default=0.0)
                    if top > 0.0:
                        d = (r[1] - r[0]) * _figure.THEMES[theme]["pad"]
                        for h in hists:
                            h.heights = [v * (r[1] - r[0] + d) / top for v in h.heights]
                            h.base, h.ceiling = r[0] - d, r[1] + d
            else:
                cell.scatter(
                    cols[cj],
                    cols[ci],
                    group=None if parts is None else group,
                    data=data if isinstance(group, str) else None,
                    size=2.0,
                )
            cell._cell.legend["loc"] = None
            if i == k - 1:
                cell.xlabel(escape(cj))
            if j == 0:
                cell.ylabel(escape(ci))
    if parts is not None:
        fig[0, k - 1]._cell.legend["loc"] = "best"  # a free corner of that cell, else beside it
    return fig


def _pairs_xy(who, x, y, data, range_):
    """The finite (x, y) pairs and the box they are counted in."""
    X, Y = numbers_(who, "x", x, data), numbers_(who, "y", y, data)
    same_length(who, x=X, y=Y)
    ok = sd.isfinite(X) & sd.isfinite(Y)
    X, Y = X[ok], Y[ok]
    if range_ is None:
        if X.shape[0] == 0:
            raise ValueError(f"{who}(): no finite (x, y) pair to count")
        box = [float(sd.min(X)), float(sd.max(X)), float(sd.min(Y)), float(sd.max(Y))]
    else:
        if (
            not isinstance(range_, (tuple, list))
            or len(range_) != 2
            or not all(isinstance(r, (tuple, list)) and len(r) == 2 for r in range_)
        ):
            raise TypeError(f"{who}(): 'range' must be ((x0, x1), (y0, y1)), not {_type_name(range_)}")
        box = [real(v, who, "range") for r in range_ for v in r]
        for k, name in ((0, "x"), (2, "y")):
            if box[k] > box[k + 1]:
                raise ValueError(
                    f"{who}(): 'range' must have {name}0 at most {name}1, not {box[k]:g} and {box[k + 1]:g}"
                )
        keep = (X >= box[0]) & (X <= box[1]) & (Y >= box[2]) & (Y <= box[3])
        X, Y = X[keep], Y[keep]
    for k in (0, 2):
        if not box[k] < box[k + 1]:
            d = 0.5 if box[k] == 0.0 else abs(box[k]) * 0.05
            box[k], box[k + 1] = box[k] - d, box[k + 1] + d
    return X, Y, box


def hist2d(self, x, y, /, *, data=None, bins=40, range=None, density=False, cmap=None, colorbar=True):
    """The counts of (x, y) pairs in a grid of bins, as a picture; empty bins are left blank."""
    who = "hist2d"
    _basic._check_data(who, data)
    _check_bool(density, who, "density")
    _check_bool(colorbar, who, "colorbar")
    name = _field._cmap(who, cmap)
    if isinstance(bins, (tuple, list)):
        if len(bins) != 2:
            raise ValueError(f"{who}(): 'bins' as a pair must hold 2 numbers, not {len(bins)}")
        nx, ny = integer(bins[0], who, "bins", 1), integer(bins[1], who, "bins", 1)
    else:
        nx = ny = integer(bins, who, "bins", 1)
    X, Y, (x0, x1, y0, y1) = _pairs_xy(who, x, y, data, range)
    i = sd.clip(sd.floor((X - x0) / (x1 - x0) * nx), min=0.0, max=nx - 1.0).astype(sd.int64)
    j = sd.clip(sd.floor((Y - y0) / (y1 - y0) * ny), min=0.0, max=ny - 1.0).astype(sd.int64)
    counts = sd.bincount(j * nx + i, minlength=nx * ny).astype(sd.float64)
    if density and X.shape[0]:
        counts = counts / (X.shape[0] * (x1 - x0) / nx * (y1 - y0) / ny)
    F = sd.reshape(sd.where(counts > 0.0, counts, math.nan), (ny, nx))
    hi = float(sd.max(counts)) if counts.shape[0] else 1.0
    lo = 0.0
    if not hi > lo:
        hi = 1.0
    cell = self._cell
    cell.layers.append(_field.Picture((x0, x1, y0, y1), nx, ny, _field._picture(F, lo, hi, _color.table(name)), False))
    if colorbar:
        cell.colorbar = (name, lo, hi)
    return self._touch()


class Hexes(_layers.Layer):
    """Hexagonal bins: centres and the half sizes of a hexagon in data units, a colour each."""

    def __init__(self, cx, cy, rx, ry, colors, box):
        self.cx, self.cy, self.rx, self.ry, self.colors, self.box = cx, cy, rx, ry, colors, box

    def extent(self, axis, positive):
        lo, hi = (self.box[0], self.box[1]) if axis == "x" else (self.box[2], self.box[3])
        if positive and not lo > 0.0:  # a log axis: from the lowest centre of a hexagon on it
            low = [v for v in (self.cx if axis == "x" else self.cy) if v > 0.0]
            if not low or not hi > 0.0:
                return None
            lo = min(low)
        return (lo, hi, ())

    def draw(self, fr):
        # pointy-top hexagons; those of one colour go in one path
        corners = [(math.cos(math.radians(90 + 60 * k)), math.sin(math.radians(90 + 60 * k))) for k in range(6)]
        groups = {}
        for x, y, c in zip(self.cx, self.cy, self.colors):
            groups.setdefault(c, []).append((x, y))
        # on a log axis a corner at zero or below is drawn off the low end of the axis
        fx = fr.xa.lo / 10.0 if fr.xa.scale == "log" else -math.inf
        fy = fr.ya.lo / 10.0 if fr.ya.scale == "log" else -math.inf
        for c, pts in groups.items():
            ops, xy = [], array("d")
            for x, y in pts:
                ops.append("MLLLLLZ")
                for ux, uy in corners:
                    xy.extend((fr.px(max(x + ux * self.rx, fx)), fr.py(max(y + uy * self.ry, fy))))
            # a hairline of the same colour closes the seams between neighbours
            fr.scene.add(_scene.Path("".join(ops), xy, fill=c, stroke=c, width=0.25, clip=fr.clip))


def hexbin(self, x, y, /, *, data=None, size=30, range=None, cmap=None, colorbar=True):
    """The counts of (x, y) pairs in hexagonal bins, `size` hexagons across; empty ones blank."""
    who = "hexbin"
    _basic._check_data(who, data)
    _check_bool(colorbar, who, "colorbar")
    name = _field._cmap(who, cmap)
    n = integer(size, who, "size", 2)
    X, Y, (x0, x1, y0, y1) = _pairs_xy(who, x, y, data, range)
    # the lattice in units where a plot of the default shape (4 : 3) shows the hexagons regular
    W, H = 4.0, 3.0
    u, v = (X - x0) / (x1 - x0) * W, (Y - y0) / (y1 - y0) * H
    dx = W / n
    dy = dx * math.sqrt(3.0)
    ia, ja = sd.round(u / dx), sd.round(v / dy)  # two rectangular lattices; the nearer centre wins
    ib, jb = sd.floor(u / dx), sd.floor(v / dy)
    da = (u - ia * dx) ** 2 + (v - ja * dy) ** 2
    db = (u - (ib + 0.5) * dx) ** 2 + (v - (jb + 0.5) * dy) ** 2
    first = da <= db
    na, nb = n + 2, int(H / dy) + 3
    key = sd.where(first, (ja + 1) * na + (ia + 1), na * nb + (jb + 1) * na + (ib + 1)).astype(sd.int64)
    counts = sd.bincount(key, minlength=2 * na * nb)
    full = sd.nonzero(counts > 0)[0]
    k = full.tolist()
    if not k:
        hi = 1.0
        cx = cy = []
        cols = []
    else:
        c = counts[full].astype(sd.float64).tolist()
        hi = max(c)
        cx, cy = [], []
        for key_ in k:
            second, rest = divmod(key_, na * nb)
            j_, i_ = divmod(rest, na)
            i_, j_ = i_ - 1, j_ - 1
            uu, vv = (i_ * dx, j_ * dy) if not second else ((i_ + 0.5) * dx, (j_ + 0.5) * dy)
            cx.append(x0 + uu / W * (x1 - x0))
            cy.append(y0 + vv / H * (y1 - y0))
        cols = [_color.cmap(name, v / hi) for v in c]
    rx = dx / math.sqrt(3.0) / W * (x1 - x0)  # the circumradius, as data units along each axis
    ry = dx / math.sqrt(3.0) / H * (y1 - y0)
    cell = self._cell
    cell.layers.append(Hexes(cx, cy, rx, ry, cols, (x0, x1, y0, y1)))
    if colorbar:
        cell.colorbar = (name, 0.0, hi)
    return self._touch()


METHODS = {
    "histogram": histogram,
    "hist2d": hist2d,
    "hexbin": hexbin,
    "ecdf": ecdf,
    "boxplot": boxplot,
    "strip": strip,
    "qq": qq,
    "heatmap": heatmap,
    "coefficients": coefficients,
    "distribution": distribution,
}
FIGURE_ONLY = {"compare": compare, "fit": fit}
_ = (Cell, integer)
