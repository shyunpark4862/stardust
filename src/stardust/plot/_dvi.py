"""A reader of TeX's DVI and XeTeX's XDV output and of TFM widths (docs/design/plot.md §3.2).

pages(data, tfm) gives, per page, the fonts used, the glyphs with their positions and the rules,
all in scaled points (sp, 65536 to a TeX point), origin at the page's upper left corner, y down.
TeX leaves the advance after a character to the reader, which takes it from the font's TFM
widths exactly as TeX does (`scaled`); native (OpenType) fonts of XDV carry their positions.
"""

import struct

_S = {1: ">b", 2: ">h", 4: ">i"}
_U = {1: ">B", 2: ">H", 4: ">I"}


class Font:
    __slots__ = ("key", "size", "native", "path", "index", "widths")

    def __init__(self, key, size, native=False, path=None, index=0):
        self.key, self.size, self.native, self.path, self.index = key, size, native, path, index
        self.widths = None


class Page:
    __slots__ = ("fonts", "glyphs", "rules")

    def __init__(self):
        self.fonts = []  # Font, in order of first use on this page
        self.glyphs = []  # (index into fonts, code or glyph id, h, v)
        self.rules = []  # (h, v of the bottom edge, width, height)


class _Reader:
    def __init__(self, data):
        self.d, self.i = data, 0

    def u(self, n):
        if n == 3:
            v = int.from_bytes(self.d[self.i : self.i + 3], "big")
        else:
            v = struct.unpack_from(_U[n], self.d, self.i)[0]
        self.i += n
        return v

    def s(self, n):
        if n == 3:
            v = int.from_bytes(self.d[self.i : self.i + 3], "big", signed=True)
        else:
            v = struct.unpack_from(_S[n], self.d, self.i)[0]
        self.i += n
        return v

    def take(self, n):
        b = self.d[self.i : self.i + n]
        self.i += n
        return b


def font_names(data):
    """The TFM font names a DVI file defines (to load their widths in one go)."""
    return {f.key for f in _walk(data, None, names_only=True)}


def pages(data, tfm):
    """The pages of a DVI or XDV file. tfm(name) gives the font's TFM widths (tfm_widths)."""
    return list(_walk(data, tfm, names_only=False))


def _walk(data, tfm, names_only):
    r = _Reader(data)
    defs = {}
    page = None
    h = v = w = x = y = z = 0
    stack = []
    cur = None
    slot = {}  # font number -> index in page.fonts

    def use(num):
        if num not in slot:
            slot[num] = len(page.fonts)
            page.fonts.append(defs[num])
        return slot[num]

    n = len(data)
    while r.i < n:
        op = r.u(1)
        if op <= 127 or 128 <= op <= 131 or 133 <= op <= 136:
            code = op if op <= 127 else r.u(op - 127 if op <= 131 else op - 132)
            f = defs[cur]
            if not names_only:
                page.glyphs.append((use(cur), code, h, v))
                if op <= 131:
                    if f.widths is None:
                        f.widths = tfm(f.key)
                    h += scaled(f.widths[code], f.size)
        elif op in (132, 137):
            a, b = r.s(4), r.s(4)
            if not names_only and a > 0 and b > 0:
                page.rules.append((h, v, b, a))
            if op == 132:
                h += b
        elif op == 138:
            pass
        elif op == 139:
            r.take(44)
            page = Page()
            slot = {}
            h = v = w = x = y = z = 0
            stack = []
        elif op == 140:
            if not names_only:
                yield page
            page = None
        elif op == 141:
            stack.append((h, v, w, x, y, z))
        elif op == 142:
            h, v, w, x, y, z = stack.pop()
        elif 143 <= op <= 146:
            h += r.s(op - 142)
        elif op == 147:
            h += w
        elif 148 <= op <= 151:
            w = r.s(op - 147)
            h += w
        elif op == 152:
            h += x
        elif 153 <= op <= 156:
            x = r.s(op - 152)
            h += x
        elif 157 <= op <= 160:
            v += r.s(op - 156)
        elif op == 161:
            v += y
        elif 162 <= op <= 165:
            y = r.s(op - 161)
            v += y
        elif op == 166:
            v += z
        elif 167 <= op <= 170:
            z = r.s(op - 166)
            v += z
        elif 171 <= op <= 234:
            cur = op - 171
        elif 235 <= op <= 238:
            cur = r.u(op - 234)
        elif 239 <= op <= 242:
            r.take(r.u(op - 238))
        elif 243 <= op <= 246:
            num = r.u(op - 242)
            r.take(4)
            size = r.u(4)
            r.take(4)
            a, l = r.u(1), r.u(1)
            name = r.take(a + l).decode("latin-1")
            if num not in defs:
                defs[num] = Font(name, size)
                if names_only:
                    yield defs[num]
        elif op == 247:
            r.take(13)
            r.take(r.u(1))
        elif op in (248, 249):
            return
        elif op in (250, 251):
            pass
        elif op == 252:
            num = r.s(4)
            size = r.u(4)
            flags = r.u(2)
            path = r.take(r.u(1)).decode("utf-8")
            index = r.u(4)
            for bit in (0x0200, 0x1000, 0x2000, 0x4000):
                if flags & bit:
                    r.take(4)
            if num not in defs:
                defs[num] = Font(f"{path.rsplit('/', 1)[-1]}#{index}", size, True, path, index)
        elif op in (253, 254):
            if op == 254:
                r.take(2 * r.u(2))
            adv = r.s(4)
            count = r.u(2)
            pos = [(r.s(4), r.s(4)) for _ in range(count)]
            ids = [r.u(2) for _ in range(count)]
            if not names_only:
                fi = use(cur)
                for (dx, dy), g in zip(pos, ids):
                    page.glyphs.append((fi, g, h + dx, v + dy))
            h += adv
        else:
            raise ValueError(f"unknown DVI opcode {op}")


def tfm_widths(data):
    """The 256 character widths of a TFM file as fix_words (raw 32-bit integers; 0 where the font
    has no such character)."""
    lf, lh, bc, ec, nw = struct.unpack_from(">5H", data, 0)
    info = 24 + 4 * lh
    table = info + 4 * (ec - bc + 1)
    words = struct.unpack_from(f">{nw}i", data, table)
    out = [0] * 256
    for c in range(bc, ec + 1):
        out[c] = words[data[info + 4 * (c - bc)]]
    return out


def scaled(fix, size):
    """TeX's product of a fix_word and a font size in sp (tex.web §571-575: the truncations are
    part of the result)."""
    z, alpha = size, 16
    while z >= 0o40000000:
        z //= 2
        alpha += alpha
    beta = 256 // alpha
    alpha *= z
    a, b, c, d = (fix >> 24) & 255, (fix >> 16) & 255, (fix >> 8) & 255, fix & 255
    sw = (((d * z) // 256 + c * z) // 256 + b * z) // beta
    return sw if a == 0 else sw - alpha
