"""SVG output of a scene (docs/design/plot.md §3.6).

Glyph outlines are written once in <defs> and referenced; numbers are written with at most
four decimals of a bp, without trailing zeros, so the same scene is the same text.
"""

import base64
from array import array

from .._core import plot as _core_plot
from . import _png, _scene

_ARGS = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}


def num(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def path_data(ops, xy, digits=4):
    xy = xy if isinstance(xy, array) else array("d", xy)
    return _core_plot.path_text(ops.encode(), xy.tobytes(), digits, False)


def _paint(name, rgba):
    if rgba is None:
        return f' {name}="none"'
    r, g, b = (round(max(0.0, min(1.0, c)) * 255) for c in rgba[:3])
    a = rgba[3] if len(rgba) > 3 else 1.0
    out = f' {name}="#{r:02x}{g:02x}{b:02x}"'
    if a < 1.0:
        out += f' {name}-opacity="{num(a)}"'
    return out


def write(scene, dpi=300.0):
    """The SVG text of a scene. dpi: of the pictures that stand for very large marker layers."""
    items, engine = scene.resolve()
    items = _png.flatten_large(items, dpi)
    defs, body = [], []
    glyph_ids, clip_ids = {}, {}

    def clip_attr(clip):
        if clip is None:
            return ""
        cid = clip_ids.get(clip)
        if cid is None:
            cid = clip_ids[clip] = f"c{len(clip_ids)}"
            x, y, w, h = clip
            defs.append(
                f'<clipPath id="{cid}"><rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}"/></clipPath>'
            )
        return f' clip-path="url(#{cid})"'

    nmark = 0
    for item in items:
        if isinstance(item, _scene.Image):
            data = base64.b64encode(_png.encode(item.iw, item.ih, item.rgba, level=6)).decode()
            look = "" if item.smooth else ' image-rendering="pixelated"'
            body.append(
                f'<image x="{num(item.x)}" y="{num(item.y)}" width="{num(item.w)}" height="{num(item.h)}" '
                f'preserveAspectRatio="none"{look}{clip_attr(item.clip)} xlink:href="data:image/png;base64,{data}"/>'
            )
        elif isinstance(item, _scene.Markers):
            mid = f"m{nmark}"
            nmark += 1
            defs.append(f'<path id="{mid}" d="{path_data(item.ops, item.xy)}"/>')
            attr = _paint("fill", item.fill)
            if item.stroke is not None:
                attr += _paint("stroke", item.stroke) + f' stroke-width="{num(item.width)}"'
            at = item.at
            uses = "".join(
                f'<use xlink:href="#{mid}" x="{num(at[i])}" y="{num(at[i + 1])}"/>' for i in range(0, len(at), 2)
            )
            body.append(f"<g{attr}{clip_attr(item.clip)}>{uses}</g>")
        elif isinstance(item, _scene.Path):
            attr = _paint("fill", item.fill)
            if item.fill is not None and item.rule == "evenodd":
                attr += ' fill-rule="evenodd"'
            if item.stroke is not None:
                attr += _paint("stroke", item.stroke) + f' stroke-width="{num(item.width)}"'
                if item.cap != "butt":
                    attr += f' stroke-linecap="{item.cap}"'
                if item.join != "miter":
                    attr += f' stroke-linejoin="{item.join}"'
                elif "L" in item.ops[2:]:  # a corner: the miter limit of the PNG and the PDF (SVG's own is 4)
                    attr += ' stroke-miterlimit="10"'
                if item.dash:
                    attr += f' stroke-dasharray="{" ".join(num(v) for v in item.dash)}"'
            attr += clip_attr(item.clip)
            body.append(f'<path d="{path_data(item.ops, item.xy)}"{attr}/>')
        else:
            parts = []
            for key, code, scale, x, y in item.glyphs:
                gid = glyph_ids.get((key, code))
                if gid is None:
                    ops, xy = engine.outline(key, code)
                    if not ops:
                        continue
                    gid = glyph_ids[key, code] = f"g{len(glyph_ids)}"
                    defs.append(f'<path id="{gid}" d="{path_data(ops, xy, 6)}"/>')
                s = f"{scale:.6f}".rstrip("0").rstrip(".")
                parts.append(f'<use xlink:href="#{gid}" transform="matrix({s} 0 0 {s} {num(x)} {num(y)})"/>')
            for x, y, w, h in item.rules:
                parts.append(f'<rect x="{num(x)}" y="{num(y)}" width="{num(w)}" height="{num(h)}"/>')
            m = " ".join(f"{v:.6f}".rstrip("0").rstrip(".") or "0" for v in item.matrix)
            body.append(f'<g{_paint("fill", item.color)} transform="matrix({m})">' + "".join(parts) + "</g>")
    w, h = num(scene.width), num(scene.height)
    head = (
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}pt" height="{h}pt" viewBox="0 0 {w} {h}">\n'
    )
    return head + ("<defs>\n" + "\n".join(defs) + "\n</defs>\n" if defs else "") + "\n".join(body) + "\n</svg>\n"
