"""Plots of functions of one variable (docs/api/plot.md §4.1, §4.2): function, parametric,
polar. The curve is sampled when the function is called (_sample.py): the samples close in on
bends, poles, jumps and the edges of the domain, the y range leaves out what runs away near a
pole, and a stretch that oscillates faster than the screen can show becomes a band.
"""

import collections
import math

from .. import _core as sd
from .._core import _type_name
from ..optimize._common import _check_bool
from . import _color, _layers, _sample
from ._basic import _label, _opt_real, _style
from ._figure import real, string

# What a function plot did, one record per curve (docs/api/plot.md §4.6): the samples used, the
# poles and jumps found (their positions to the sampler's resolution), the y range chosen and
# whether it leaves values out below or above, the stretches where the function is not finite,
# and those drawn as a band.
PlotInfo = collections.namedtuple("PlotInfo", "kind n poles jumps ylim clipped undefined oscillating")


def call(f, t, who, arg):
    """f(t) as a float64 sd array of t's shape."""
    out = f(t)
    if isinstance(out, sd.ndarray):
        if tuple(out.shape) != tuple(t.shape):
            raise ValueError(
                f"{who}(): '{arg}' must return an array of its argument's shape {tuple(t.shape)}, "
                f"not {tuple(out.shape)}"
            )
        if out.dtype == sd.complex128:
            raise TypeError(f"{who}(): '{arg}' must return real values, not complex128")
        return out if out.dtype == sd.float64 else out.astype(sd.float64)
    if isinstance(out, (int, float)) and not isinstance(out, bool):
        return sd.full(tuple(t.shape), float(out))
    raise TypeError(f"{who}(): '{arg}' must return an sd.ndarray (it is called with one), not {_type_name(out)}")


def _callable(f, who, arg):
    if not callable(f):
        raise TypeError(f"{who}(): '{arg}' must be callable, not {_type_name(f)}")
    return f


def _interval(who, a, b, na="a", nb="b"):
    a, b = real(a, who, na), real(b, who, nb)
    if not a < b:
        raise ValueError(f"{who}(): '{na}' must be less than '{nb}'")
    return a, b


class CurveLine(_layers.Line):
    """A sampled curve: a line whose y extent is held to the range the sampler chose."""

    def __init__(self, x, y, color, width, style, label, yclip=None):
        super().__init__(x, y, color, width, style, label)
        self.yclip = yclip

    def extent(self, axis, positive):
        e = super().extent(axis, positive)
        if e is None or axis == "x" or self.yclip is None:
            return e
        lo, hi = max(e[0], self.yclip[0]), min(e[1], self.yclip[1])
        return (lo, hi, ()) if lo <= hi else (self.yclip[0], self.yclip[1], ())


def _add_curve(fig, c, xr, yr, yclip, color, width, style, label, kind):
    """The layers of a sampled curve: the line (cut at its breaks), bands where it oscillates."""
    cell = fig._cell
    bands, inside = _sample.oscillation(c, xr, yr) if c.t.shape[0] > 4000 else ([], None)
    x, y = _sample.with_breaks(c)
    if bands:
        # the samples inside a band are left out of the line
        mask = sd.asarray(inside)
        cx, cy = c.x, sd.where(mask, math.nan, c.y)
        tmp = _sample.Curve()
        tmp.x, tmp.y, tmp.breaks = cx, cy, c.breaks
        x, y = _sample.with_breaks(tmp)
        for bx, lo, hi in bands:
            cell.layers.append(_layers.Area(bx, lo, hi, color, None))
    cell.layers.append(CurveLine(x, y, color, width, style, label, yclip))
    return [(float(b[0][0]), float(b[0][-1])) for b in bands]


def function(
    self,
    f,
    a,
    b,
    /,
    *,
    ylim=None,
    asymptotes=False,
    samples=None,
    refine=None,
    color=None,
    width=None,
    style="-",
    label=None,
    labels=None,
):
    who = "function"
    fs = list(f) if isinstance(f, (list, tuple)) else [f]
    if not fs:
        raise ValueError(f"{who}(): 'f' must be a function or a non-empty sequence of functions")
    for g in fs:
        _callable(g, who, "f")
    a, b = _interval(who, a, b)
    if ylim is not None:
        if not isinstance(ylim, (tuple, list)) or len(ylim) != 2:
            raise TypeError(f"{who}(): 'ylim' must be a pair (low, high), not {_type_name(ylim)}")
        ylim = _interval(who, ylim[0], ylim[1], "ylim[0]", "ylim[1]")
    _check_bool(asymptotes, who, "asymptotes")
    width, style, label = _opt_real(who, "width", width), _style(who, style), _label(who, label)
    if labels is not None:
        if not isinstance(labels, (list, tuple)) or len(labels) != len(fs):
            raise ValueError(f"{who}(): 'labels' must be {len(fs)} strings, one per function")
        labels = [string(s, who, "labels") for s in labels]
        if label is not None:
            raise ValueError(f"{who}(): 'label' and 'labels' cannot both be given")
    elif label is not None:
        if len(fs) > 1:
            raise ValueError(f"{who}(): several functions take 'labels', not 'label'")
        labels = [label]
    if color is not None and len(fs) > 1:
        raise ValueError(f"{who}(): 'color' cannot be given for several functions (they take the palette's colours)")
    palette = self._root._palette
    colors = [
        _color.parse(color, who, "color", palette) if color is not None else self._cell.next_color(palette) for _ in fs
    ]
    xr = (a, b)
    n, depth = _sample.knobs(who, samples, refine, _sample.N0 + 1, _sample.DEPTH, most=200000)
    t0 = _sample.first_grid(a, b, n - 1)
    firsts = [call(g, t0, who, "f") for g in fs]
    if ylim is not None:
        yr, clipped, yclip = ylim, (False, False), ylim
    else:
        rs = [_sample.robust_range(v) for v in firsts]
        lo, hi = min(r[0] for r in rs), max(r[1] for r in rs)
        clipped = (any(r[2] for r in rs), any(r[3] for r in rs))
        if not lo < hi:
            d = 0.5 if lo == 0.0 else abs(lo) * 0.05
            lo, hi = lo - d, hi + d
        yr = (lo, hi)
        # A side the first grid's values ran away on is cut there. The other sides may grow by
        # what the finer samples find (a peak between grid points), but not without bound: a pole
        # the grid stepped over must not take the axis.
        grow = 0.5 * (hi - lo)
        yclip = (lo if clipped[0] else lo - grow, hi if clipped[1] else hi + grow)
    poles_all = []
    for k, (g, y0) in enumerate(zip(fs, firsts)):
        c = _sample.refine(lambda t, g=g: (t, call(g, t, who, "f")), t0, t0, y0, xr, yr, depth)
        osc = _add_curve(self, c, xr, yr, yclip, colors[k], width, style, labels[k] if labels else None, who)
        poles_all += c.poles
        e = self._cell.layers[-1].extent("y", False)
        cut = (clipped[0] or (e is not None and e[0] <= yclip[0]), clipped[1] or (e is not None and e[1] >= yclip[1]))
        jumps = tuple(j for j in c.jumps if not any(x0 <= j <= x1 for x0, x1 in osc))
        self._root._infos.append(
            PlotInfo(
                who,
                int(c.t.shape[0]),
                tuple(c.poles),
                jumps,
                None if e is None else (e[0], e[1]),
                cut,
                tuple(c.undefined),
                tuple(osc),
            )
        )
    if ylim is not None and self._cell.ylim == [None, None]:
        self._cell.ylim = [ylim[0], ylim[1]]
    if asymptotes:
        for p in sorted(set(poles_all)):
            self._cell.layers.append(_layers.Rule(p, True, (0.5, 0.5, 0.5, 1.0), 0.4, "--", None))
    return self._touch()


def _plane(self, who, g, a, b, samples, refine, color, width, style, label):
    """A curve in the plane from g(t) -> (x, y): equal units on both axes."""
    width, style, label = _opt_real(who, "width", width), _style(who, style), _label(who, label)
    palette = self._root._palette
    c0 = _color.parse(color, who, "color", palette) if color is not None else self._cell.next_color(palette)
    n, depth = _sample.knobs(who, samples, refine, _sample.N0 + 1, _sample.DEPTH, most=200000)
    t0 = _sample.first_grid(a, b, n - 1)
    x0, y0 = g(t0)

    def full(v):
        r = _layers.finite_range(v)
        if r is None:
            return (0.0, 1.0)
        return r if r[0] < r[1] else (r[0] - 0.5, r[1] + 0.5)

    xr, yr = full(x0), full(y0)
    # one scale for both: the screen is the larger of the two extents, as the equal axes will be
    span = max(xr[1] - xr[0], (yr[1] - yr[0]) * _sample.W0 / _sample.H0)
    xr = (xr[0], xr[0] + span)
    yr = (yr[0], yr[0] + span * _sample.H0 / _sample.W0)
    c = _sample.refine(g, t0, x0, y0, xr, yr, depth)
    x, y = _sample.with_breaks(c)
    self._cell.layers.append(_layers.Line(x, y, c0, width, style, label))
    self._cell.aspect = 1.0
    self._root._infos.append(
        PlotInfo(who, int(c.t.shape[0]), tuple(c.poles), tuple(c.jumps), None, (False, False), tuple(c.undefined), ())
    )
    return self._touch()


def parametric(self, fx, fy, a, b, /, *, samples=None, refine=None, color=None, width=None, style="-", label=None):
    who = "parametric"
    _callable(fx, who, "fx")
    _callable(fy, who, "fy")
    a, b = _interval(who, a, b)
    return _plane(
        self,
        who,
        lambda t: (call(fx, t, who, "fx"), call(fy, t, who, "fy")),
        a,
        b,
        samples,
        refine,
        color,
        width,
        style,
        label,
    )


def polar(self, r, a, b, /, *, samples=None, refine=None, color=None, width=None, style="-", label=None):
    who = "polar"
    _callable(r, who, "r")
    a, b = _interval(who, a, b)

    def g(t):
        rho = call(r, t, who, "r")
        return rho * sd.cos(t), rho * sd.sin(t)

    return _plane(self, who, g, a, b, samples, refine, color, width, style, label)


METHODS = {"function": function, "parametric": parametric, "polar": polar}
