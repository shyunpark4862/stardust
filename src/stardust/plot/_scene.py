"""The scene: the list of shapes every output reads (docs/design/plot.md §3.1).

A Scene is a page of `width` x `height` bp (1/72 inch), origin at the upper left corner, y down,
with items in painting order:

    Path     a filled and/or stroked outline: ops over "MLCQZ" and the flat coordinates
    Markers  one small outline stamped at many points
    Image    a raster picture over a rectangle (a density, a 3D surface)
    Text   a string LaTeX typesets, anchored at a point (kept as a string: the golden form of a
           scene does not depend on the TeX installation)

The outputs (SVG, PDF and PNG) read a resolved scene, in which every Text has become
glyph outlines (resolve). to_dict is the exact, readable form the golden cases store.
"""

import math
from array import array

from . import _text

INK = (0.13, 0.13, 0.13, 1.0)  # axes, ticks and text by default: a soft black, as printed ink reads on white
_KAPPA = 4.0 * (math.sqrt(2.0) - 1.0) / 3.0  # a quarter circle as one cubic
_HA = ("left", "center", "right")
_VA = ("baseline", "bottom", "center", "top")


class Path:
    __slots__ = ("ops", "xy", "fill", "stroke", "width", "dash", "cap", "join", "clip", "rule")

    def __init__(
        self,
        ops,
        xy,
        *,
        fill=None,
        stroke=None,
        width=1.0,
        dash=(),
        cap="butt",
        join="miter",
        clip=None,
        rule="nonzero",
    ):
        self.ops, self.xy = ops, xy if isinstance(xy, array) else array("d", xy)
        self.fill, self.stroke, self.width = fill, stroke, float(width)
        self.dash, self.cap, self.join, self.clip = tuple(dash), cap, join, clip
        self.rule = rule  # the fill rule: 'nonzero' or 'evenodd'

    def to_dict(self):
        d = {"type": "path", "ops": self.ops, "xy": list(self.xy)}
        for k, default in (
            ("fill", None),
            ("stroke", None),
            ("width", 1.0),
            ("dash", ()),
            ("cap", "butt"),
            ("join", "miter"),
            ("clip", None),
            ("rule", "nonzero"),
        ):
            v = getattr(self, k)
            if v != default:
                d[k] = list(v) if isinstance(v, tuple) else v
        return d


class Markers:
    """One shape (ops, xy around the origin) stamped at the points `at` (x, y pairs). Each stamp
    is painted on its own, so translucent markers that overlap darken."""

    __slots__ = ("ops", "xy", "at", "fill", "stroke", "width", "clip")

    def __init__(self, ops, xy, at, *, fill=None, stroke=None, width=1.0, clip=None):
        self.ops, self.xy = ops, xy if isinstance(xy, array) else array("d", xy)
        self.at = at if isinstance(at, array) else array("d", at)
        self.fill, self.stroke, self.width, self.clip = fill, stroke, float(width), clip

    def to_dict(self):
        d = {"type": "markers", "ops": self.ops, "xy": list(self.xy), "at": list(self.at)}
        for k, default in (("fill", None), ("stroke", None), ("width", 1.0), ("clip", None)):
            v = getattr(self, k)
            if v != default:
                d[k] = list(v) if isinstance(v, tuple) else v
        return d


class Image:
    """A raster picture of iw x ih straight RGBA8 pixels over the rectangle (x, y, w, h). smooth:
    interpolated between the pixels; else each pixel is a flat square."""

    __slots__ = ("x", "y", "w", "h", "iw", "ih", "rgba", "smooth", "clip")

    def __init__(self, x, y, w, h, iw, ih, rgba, *, smooth=True, clip=None):
        self.x, self.y, self.w, self.h = float(x), float(y), float(w), float(h)
        self.iw, self.ih, self.rgba, self.smooth, self.clip = iw, ih, rgba, smooth, clip

    def to_dict(self):
        import hashlib

        d = {
            "type": "image",
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "size": [self.iw, self.ih],
            "sha256": hashlib.sha256(self.rgba).hexdigest(),
            "smooth": self.smooth,
        }
        if self.clip is not None:
            d["clip"] = list(self.clip)
        return d


class Text:
    __slots__ = ("s", "x", "y", "size", "ha", "va", "rotation", "color")

    def __init__(self, s, x, y, *, size=10.0, ha="left", va="baseline", rotation=0.0, color=INK):
        if ha not in _HA or va not in _VA:
            raise ValueError(f"bad text anchor {ha!r}, {va!r}")
        self.s, self.x, self.y, self.size = s, float(x), float(y), float(size)
        self.ha, self.va, self.rotation, self.color = ha, va, float(rotation), color

    def to_dict(self):
        return {
            "type": "text",
            "s": self.s,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "ha": self.ha,
            "va": self.va,
            "rotation": self.rotation,
            "color": list(self.color),
        }


class Glyphs:
    """A resolved Text: the glyphs [(font key, code, scale, x, y)] and rules [(x, y, w, h)] in
    the text's own frame (baseline's left end at the origin, y down), and the matrix that places
    that frame on the page."""

    __slots__ = ("glyphs", "rules", "matrix", "color")

    def __init__(self, glyphs, rules, matrix, color):
        self.glyphs, self.rules, self.matrix, self.color = glyphs, rules, matrix, color


class Scene:
    def __init__(self, width, height, *, font="serif"):
        self.width, self.height, self.font = float(width), float(height), font
        self.items = []

    def add(self, item):
        self.items.append(item)
        return item

    def to_dict(self):
        return {
            "width": self.width,
            "height": self.height,
            "font": self.font,
            "items": [i.to_dict() for i in self.items],
        }

    def strings(self):
        return [i.s for i in self.items if isinstance(i, Text)]

    def resolve(self):
        """(items with every Text replaced by Glyphs, the text engine). One batch typesets every
        string the cache does not hold."""
        engine = _text.engine(self.font)
        texts = [i for i in self.items if isinstance(i, Text)]
        layouts = dict(zip((t.s for t in texts), engine.layout([t.s for t in texts])))
        out = []
        for item in self.items:
            out.append(place(item, layouts[item.s]) if isinstance(item, Text) else item)
        return out, engine


def place(text, lay):
    """The Glyphs of a Text given its layout."""
    k0 = text.size / _text.UNIT_BP
    k = _text.SP_TO_BP * k0
    w, h, d = lay.width * k, lay.height * k, lay.depth * k
    dx = {"left": 0.0, "center": -w / 2, "right": -w}[text.ha]
    dy = {"baseline": 0.0, "bottom": -d, "center": (h - d) / 2, "top": h}[text.va]
    top = lay.height
    glyphs = [
        (lay.fonts[fi][0], code, lay.fonts[fi][1] / _text.TEN_PT * k0, hh * k, (v - top) * k)
        for fi, code, hh, v in lay.glyphs
    ]
    rules = [(hh * k, (v - top - ht) * k, ww * k, ht * k) for hh, v, ww, ht in lay.rules]
    a = math.radians(text.rotation)
    c, s = math.cos(a), math.sin(a)
    if text.rotation == 90.0:
        c, s = 0.0, 1.0
    # y is down: a counter-clockwise rotation by a is (c, -s; s, c) on (x, y)
    m = (c, -s, s, c, text.x + c * dx + s * dy, text.y - s * dx + c * dy)
    return Glyphs(glyphs, rules, m, text.color)


# ---- path builders


def polyline(points, *, closed=False):
    """(ops, xy) of straight segments through points [(x, y), ...]."""
    xy = [v for p in points for v in p]
    return "M" + "L" * (len(points) - 1) + ("Z" if closed else ""), xy


def rect(x, y, w, h):
    return "MLLLZ", [x, y, x + w, y, x + w, y + h, x, y + h]


def circle(cx, cy, r):
    k = _KAPPA * r
    return "MCCCCZ", [
        cx + r,
        cy,
        cx + r,
        cy + k,
        cx + k,
        cy + r,
        cx,
        cy + r,
        cx - k,
        cy + r,
        cx - r,
        cy + k,
        cx - r,
        cy,
        cx - r,
        cy - k,
        cx - k,
        cy - r,
        cx,
        cy - r,
        cx + k,
        cy - r,
        cx + r,
        cy - k,
        cx + r,
        cy,
    ]
