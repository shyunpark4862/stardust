"""Statistical plots and the plots of results (docs/api/plot.md §5, §6; docs/design/plot.md §5).

The numbers a plot shows are the computing areas' own: histogram bars are sd.histogram's
counts, a box is sd.quantile's quartiles, QQ points are a distribution's icdf, a fitted line is
the regression result's predict. A test's result is drawn as given (compare): the plot does not
choose or run a test.
"""

import math
from array import array

from .. import _core as sd
from .._core import _type_name
from ..optimize._common import _check_bool, _check_choice
from . import _basic, _color, _field, _function, _layers, _scene, _ticks
from ._basic import _alpha, _each, _label, _opt_real, _take, escape, numbers_, same_length, split
from ._figure import Cell, Figure, integer, real

INK = _scene.INK


def _nonan(a):
    return a[sd.isfinite(a)]


# ---- histogram, ecdf


class Hist(_layers.Layer):
    """Bars between bin edges, standing on zero."""

    def __init__(self, edges, heights, color, label):
        self.edges, self.heights, self.color, self.label = edges, heights, color, label

    def extent(self, axis, positive):
        if axis == "x":
            return (self.edges[0], self.edges[-1], ())
        top = max(self.heights, default=0.0)
        return (0.0, top, (0.0,)) if not positive else None

    def draw(self, fr):
        ops, xy = [], array("d")
        y0 = fr.py(0.0)
        for a, b, h in zip(self.edges, self.edges[1:], self.heights):
            if h > 0.0:
                x0, x1, y1 = fr.px(a), fr.px(b), fr.py(h)
                ops.append("MLLLZ")
                xy.extend((x0, y0, x1, y0, x1, y1, x0, y1))
        if ops:
            fr.scene.add(
                _scene.Path("".join(ops), xy, fill=self.color, stroke=(1.0, 1.0, 1.0, 1.0), width=0.3, clip=fr.clip)
            )

    def entries(self):
        return [(self.label, _layers._patch_handle(self.color))]


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
    elif isinstance(rng, int) and not isinstance(rng, bool):
        gen = _random.default_rng(rng)
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
    _check_bool(annotate, who, "annotate")
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
    if annotate:
        vals = Z.tolist()
        flat = [v for row in vals for v in row if math.isfinite(v)]
        text, _ = _ticks.labels([float(f"{v:.3g}") for v in flat]) if flat else ([], None)
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
    """A p-value as text: 'p = 0.012' with summary()'s digits, or stars."""
    if fmt == "stars":
        return "ns" if not p <= 0.05 else "*" * (1 + (p <= 0.01) + (p <= 0.001))
    if p < 0.001:
        return r"$p < 0.001$"
    return f"$p = {p:.3f}$"


class Bracket(_layers.Layer):
    """A bracket over two positions of a category axis with a text above it."""

    def __init__(self, x1, x2, level, s):
        self.x1, self.x2, self.level, self.s = x1, x2, level, s

    def strings(self):
        return [self.s]

    def draw(self, fr):
        # the bracket stands in the room the layout kept above the data (cell.headroom)
        y = fr.y + 9.0 + 13.0 * self.level
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
    else:
        try:
            value = float(p)
        except TypeError, ValueError:
            raise TypeError(
                f"{who}(): 'result' has a pvalue of shape {tuple(p.shape)}; one test or a k x k comparison is drawn"
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
        iv = result.predict_interval(X, alpha=alpha, kind="mean")
        cell.layers.append(_layers.Area(xs, iv[:, 0], iv[:, 1], _color.with_alpha(c, 0.2), None))
    cell.layers.append(_layers.Line(xs, result.predict(X), c, None, "-", None))
    if equation:
        sign = "-" if b0 < 0 else "+"
        eq = f"$y = {b1:.3g}x {sign} {abs(b0):.3g}$" if const else f"$y = {b1:.3g}x$"
        eq = eq.replace("e-0", r"\times10^{-").replace("e+0", r"\times10^{") if "e" in eq else eq
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
    if not hasattr(result, "params") or not hasattr(result, "conf_int"):
        raise TypeError(f"{who}(): 'result' must be a regression result, not {_type_name(result)}")
    params = result.params
    if params.ndim != 1:
        raise ValueError(f"{who}(): a batch of fits cannot be drawn; give one fit")
    names = list(getattr(result, "param_names", [f"x{i}" for i in range(params.shape[0])]))
    ci = result.conf_int(alpha=alpha).tolist()
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
    lo, hi = (float(v) for v in dist.support())
    qa, qb = float(dist.icdf(0.001)), float(dist.icdf(0.999))
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
        return self._touch()
    f = dist.pdf if kind == "pdf" else dist.cdf if kind == "cdf" else dist.ccdf
    return _function.function(self, f, a, b, color=color, label=label)


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
        fig[0, k - 1]._cell.legend["loc"] = "upper right"
    return fig


METHODS = {
    "histogram": histogram,
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
