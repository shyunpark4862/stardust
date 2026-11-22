"""PNG output of a scene: the C++ rasterizer (csrc/plot/raster.hpp) paints the resolved scene,
zlib packs the pixels (docs/design/plot.md §3.5).

pixels(scene, dpi) is what the golden cases hash: the picture itself, independent of zlib.
"""

import math
import struct
import zlib
from array import array

from .. import _core
from . import _scene, _svgpath

_Canvas = _core.plot.Canvas
_CAP = {"butt": 0, "round": 1, "square": 2}
_JOIN = {"miter": 0, "round": 1, "bevel": 1}
_RECT = b"MLLLZ"
MAX_PIXELS = 1 << 28


def size(scene, dpi):
    """(width, height) in pixels of a scene at `dpi`."""
    s = dpi / 72.0
    return max(1, round(scene.width * s)), max(1, round(scene.height * s))


def pixels(scene, dpi=300):
    """(width, height, straight 8-bit RGBA bytes, rows from the top) of a scene at `dpi`."""
    w, h = size(scene, dpi)
    if w > 65536 or h > 65536 or w * h > MAX_PIXELS:
        raise ValueError(f"the picture would be {w} x {h} pixels: too large")
    s = dpi / 72.0
    base = (s, 0.0, 0.0, s, 0.0, 0.0)
    items, engine = scene.resolve()
    canvas = _Canvas(w, h)
    for item in items:
        if isinstance(item, _scene.Image):
            clip = (
                ()
                if item.clip is None
                else (
                    item.clip[0] * s,
                    item.clip[1] * s,
                    (item.clip[0] + item.clip[2]) * s,
                    (item.clip[1] + item.clip[3]) * s,
                )
            )
            canvas.image(item.rgba, item.iw, item.ih, item.x * s, item.y * s, item.w * s, item.h * s, item.smooth, clip)
            continue
        if isinstance(item, (_scene.Path, _scene.Markers)):
            ops = item.ops.encode()
            xy = item.xy.tobytes()
            clip = (
                ()
                if item.clip is None
                else (
                    item.clip[0] * s,
                    item.clip[1] * s,
                    (item.clip[0] + item.clip[2]) * s,
                    (item.clip[1] + item.clip[3]) * s,
                )
            )
        if isinstance(item, _scene.Markers):
            canvas.markers(ops, xy, base, item.at.tobytes(), item.fill or (), item.stroke or (), item.width * s, clip)
        elif isinstance(item, _scene.Path):
            if item.fill is not None:
                canvas.fill(ops, xy, base, item.fill, clip)
            if item.stroke is not None:
                canvas.stroke(
                    ops,
                    xy,
                    base,
                    item.stroke,
                    clip,
                    item.width * s,
                    _CAP[item.cap],
                    _JOIN[item.join],
                    [d * s for d in item.dash],
                )
        else:
            m = _svgpath.compose(base, item.matrix)
            for key, code, scale, x, y in item.glyphs:
                ops, xy = engine.packed(key, code)
                if ops:
                    canvas.fill(ops, xy, _svgpath.compose(m, (scale, 0.0, 0.0, scale, x, y)), item.color, ())
            for x, y, rw, rh in item.rules:
                canvas.fill(
                    _RECT, array("d", (x, y, x + rw, y, x + rw, y + rh, x, y + rh)).tobytes(), m, item.color, ()
                )
    return w, h, canvas.rgba8()


RASTER_FROM = 20000  # stamps from which a marker layer is a picture in the vector outputs


def flatten_large(items, dpi):
    """The items with every marker layer of RASTER_FROM stamps or more replaced by its picture at
    `dpi` (SVG and PDF: a million small shapes make a file no viewer opens gladly)."""
    out = []
    for item in items:
        if not isinstance(item, _scene.Markers) or len(item.at) < 2 * RASTER_FROM:
            out.append(item)
            continue
        reach = max((abs(v) for v in item.xy), default=0.0) + item.width + 1.0
        xs, ys = item.at[0::2], item.at[1::2]
        x0, y0, x1, y1 = min(xs) - reach, min(ys) - reach, max(xs) + reach, max(ys) + reach
        if item.clip is not None:
            cx, cy, cw, ch = item.clip
            x0, y0, x1, y1 = max(x0, cx), max(y0, cy), min(x1, cx + cw), min(y1, cy + ch)
        s = dpi / 72.0
        iw, ih = max(1, math.ceil((x1 - x0) * s)), max(1, math.ceil((y1 - y0) * s))
        if not (x1 > x0 and y1 > y0) or iw * ih > MAX_PIXELS:
            out.append(item)
            continue
        canvas = _Canvas(iw, ih)
        canvas.markers(
            item.ops.encode(),
            item.xy.tobytes(),
            (s, 0.0, 0.0, s, -x0 * s, -y0 * s),
            item.at.tobytes(),
            item.fill or (),
            item.stroke or (),
            item.width * s,
            (),
        )
        out.append(_scene.Image(x0, y0, iw / s, ih / s, iw, ih, canvas.rgba8(), smooth=True, clip=item.clip))
    return out


def encode(w, h, rgba, *, dpi=None, level=6):
    """The PNG file of straight 8-bit RGBA pixels."""

    def chunk(tag, body):
        return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body))

    stride = 4 * w
    raw = bytearray((stride + 1) * h)  # every row with filter 0 in front
    view = memoryview(rgba)
    for y in range(h):
        raw[y * (stride + 1) + 1 : (y + 1) * (stride + 1)] = view[y * stride : (y + 1) * stride]
    out = [b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))]
    if dpi is not None:
        ppm = round(dpi / 0.0254)
        out.append(chunk(b"pHYs", struct.pack(">IIB", ppm, ppm, 1)))
    out.append(chunk(b"IDAT", zlib.compress(bytes(raw), level)))
    out.append(chunk(b"IEND", b""))
    return b"".join(out)


def write(scene, dpi=300, *, level=6):
    """The PNG file (bytes) of a scene."""
    w, h, rgba = pixels(scene, dpi)
    return encode(w, h, rgba, dpi=dpi, level=level)
