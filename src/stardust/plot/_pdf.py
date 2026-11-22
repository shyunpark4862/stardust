"""PDF output of a scene (docs/design/plot.md §3.6): one page of the scene's size, paths only.

Text is glyph outlines, each a form XObject defined once and placed by a matrix, so the file
needs no fonts. Nothing in the file depends on the time or the machine: the same scene is the
same bytes. Opacity uses graphics states; the page is flipped once so the scene's coordinates
(y down) are used as they are.
"""

import zlib
from array import array

from .._core import plot as _core_plot
from . import _png, _scene

_ARGS = {"M": 2, "L": 2, "Q": 4, "C": 6, "Z": 0}
_CAP = {"butt": 0, "round": 1, "square": 2}
_JOIN = {"miter": 0, "round": 1, "bevel": 2}


def _n(v, digits=4):
    s = f"{v:.{digits}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _path(ops, xy, digits=4):
    """The path construction operators of an outline (a quadratic becomes the same cubic)."""
    xy = xy if isinstance(xy, array) else array("d", xy)
    return _core_plot.path_text(ops.encode(), xy.tobytes(), digits, True)


def _bbox(xy):
    if not xy:
        return 0.0, 0.0, 0.0, 0.0
    return min(xy[0::2]), min(xy[1::2]), max(xy[0::2]), max(xy[1::2])


def write(scene, dpi=300.0):
    """The PDF file (bytes) of a scene. dpi: of the pictures that stand for very large marker layers."""
    items, engine = scene.resolve()
    items = _png.flatten_large(items, dpi)
    body = [f"1 0 0 -1 0 {_n(scene.height)} cm"]
    forms, alphas = {}, {}  # (font key, code) -> name; (kind, alpha) -> name

    def alpha(kind, a):
        key = (kind, round(a, 4))
        if key not in alphas:
            alphas[key] = f"A{len(alphas)}"
        return f"/{alphas[key]} gs"

    marks = []  # (name, ops, xy, paint operator) of the marker shapes
    images = []  # (name, Image)
    for item in items:
        if isinstance(item, _scene.Image):
            name = f"I{len(images)}"
            images.append((name, item))
            body.append("q")
            if item.clip is not None:
                x, y, w, h = item.clip
                body.append(f"{_n(x)} {_n(y)} {_n(w)} {_n(h)} re W n")
            # the page is flipped: the image's unit square is drawn top down
            body.append(f"{_n(item.w)} 0 0 {_n(-item.h)} {_n(item.x)} {_n(item.y + item.h)} cm /{name} Do Q")
        elif isinstance(item, _scene.Markers):
            fill, stroke = item.fill, item.stroke
            if fill is None and stroke is None:
                continue
            name = f"M{len(marks)}"
            marks.append(
                (
                    name,
                    item.ops,
                    item.xy,
                    "B" if fill is not None and stroke is not None else "f" if fill is not None else "S",
                    item.width,
                )
            )
            body.append("q")
            if item.clip is not None:
                x, y, w, h = item.clip
                body.append(f"{_n(x)} {_n(y)} {_n(w)} {_n(h)} re W n")
            if fill is not None:
                body.append(f"{_n(fill[0])} {_n(fill[1])} {_n(fill[2])} rg")
                if fill[3] < 1.0:
                    body.append(alpha("ca", fill[3]))
            if stroke is not None:
                body.append(f"{_n(stroke[0])} {_n(stroke[1])} {_n(stroke[2])} RG {_n(item.width)} w")
                if stroke[3] < 1.0:
                    body.append(alpha("CA", stroke[3]))
            at = item.at
            body.extend(f"q 1 0 0 1 {_n(at[i])} {_n(at[i + 1])} cm /{name} Do Q" for i in range(0, len(at), 2))
            body.append("Q")
        elif isinstance(item, _scene.Path):
            body.append("q")
            if item.clip is not None:
                x, y, w, h = item.clip
                body.append(f"{_n(x)} {_n(y)} {_n(w)} {_n(h)} re W n")
            fill, stroke = item.fill, item.stroke
            if fill is not None:
                body.append(f"{_n(fill[0])} {_n(fill[1])} {_n(fill[2])} rg")
                if fill[3] < 1.0:
                    body.append(alpha("ca", fill[3]))
            if stroke is not None:
                body.append(
                    f"{_n(stroke[0])} {_n(stroke[1])} {_n(stroke[2])} RG {_n(item.width)} w "
                    f"{_CAP[item.cap]} J {_JOIN[item.join]} j"
                )
                if stroke[3] < 1.0:
                    body.append(alpha("CA", stroke[3]))
                if item.dash:
                    body.append("[" + " ".join(_n(d) for d in item.dash) + "] 0 d")
            body.append(_path(item.ops, item.xy))
            star = "*" if item.rule == "evenodd" else ""
            body.append(
                "B" + star
                if fill is not None and stroke is not None
                else "f" + star
                if fill is not None
                else "S"
                if stroke is not None
                else "n"
            )
            body.append("Q")
        else:
            c = item.color
            body.append("q " + " ".join(_n(v, 6) for v in item.matrix) + f" cm {_n(c[0])} {_n(c[1])} {_n(c[2])} rg")
            if c[3] < 1.0:
                body.append(alpha("ca", c[3]))
            for key, code, scale, x, y in item.glyphs:
                name = forms.get((key, code))
                if name is None:
                    if not engine.outline(key, code)[0]:
                        continue
                    name = forms[key, code] = f"G{len(forms)}"
                body.append(f"q {_n(scale, 6)} 0 0 {_n(scale, 6)} {_n(x)} {_n(y)} cm /{name} Do Q")
            for x, y, w, h in item.rules:
                body.append(f"{_n(x)} {_n(y)} {_n(w)} {_n(h)} re f")
            body.append("Q")

    objects = []  # the bodies of objects 1..n

    def add(data):
        objects.append(data)
        return len(objects)

    def stream(head, data):
        packed = zlib.compress(data, 9)
        return f"<< {head} /Length {len(packed)} /Filter /FlateDecode >>\nstream\n".encode() + packed + b"\nendstream"

    catalog, pages, page = add(None), add(None), add(None)
    content = add(stream("", "\n".join(body).encode()))
    form_ids = {}
    for (key, code), name in forms.items():
        ops, xy = engine.outline(key, code)
        x0, y0, x1, y1 = _bbox(xy)
        head = f"/Type /XObject /Subtype /Form /BBox [{_n(x0 - 1)} {_n(y0 - 1)} {_n(x1 + 1)} {_n(y1 + 1)}]"
        form_ids[name] = add(stream(head, (_path(ops, xy, 6) + "\nf").encode()))
    for name, ops, xy, paint, width in marks:
        x0, y0, x1, y1 = _bbox(xy)
        g = width + 1.0
        head = f"/Type /XObject /Subtype /Form /BBox [{_n(x0 - g)} {_n(y0 - g)} {_n(x1 + g)} {_n(y1 + g)}]"
        form_ids[name] = add(stream(head, (_path(ops, xy) + "\n" + paint).encode()))
    for name, im in images:
        rgb = bytearray(3 * im.iw * im.ih)
        rgb[0::3], rgb[1::3], rgb[2::3] = im.rgba[0::4], im.rgba[1::4], im.rgba[2::4]
        rgb = bytes(rgb)
        alpha_ = im.rgba[3::4]
        smooth = " /Interpolate true" if im.smooth else ""
        base = f"/Type /XObject /Subtype /Image /Width {im.iw} /Height {im.ih} /BitsPerComponent 8{smooth}"
        if min(alpha_) < 255:
            mask = add(stream(base + " /ColorSpace /DeviceGray", bytes(alpha_)))
            form_ids[name] = add(stream(base + f" /ColorSpace /DeviceRGB /SMask {mask} 0 R", rgb))
        else:
            form_ids[name] = add(stream(base + " /ColorSpace /DeviceRGB", rgb))
    res = []
    if form_ids:
        res.append("/XObject << " + " ".join(f"/{n} {i} 0 R" for n, i in form_ids.items()) + " >>")
    if alphas:
        res.append(
            "/ExtGState << " + " ".join(f"/{n} << /{kind} {_n(a)} >>" for (kind, a), n in alphas.items()) + " >>"
        )
    objects[catalog - 1] = f"<< /Type /Catalog /Pages {pages} 0 R >>".encode()
    objects[pages - 1] = f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>".encode()
    objects[page - 1] = (
        f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 {_n(scene.width)} {_n(scene.height)}] "
        f"/Contents {content} 0 R /Resources << {' '.join(res)} >> >>"
    ).encode()

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, data in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + data + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root {catalog} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)
