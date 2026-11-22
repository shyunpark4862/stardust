"""SVG path data to absolute segments (the outlines dvisvgm writes; docs/design/plot.md §3.2).

parse(d) gives (ops, xy): ops a string over "MLCQZ", xy the flat coordinates each op consumes
(M and L two, Q four, C six, Z none), all absolute. The relative commands, H and V, the smooth
S and T, and repeated coordinate groups are resolved here; arcs do not occur in glyph outlines.
"""

import re

_TOKEN = re.compile(r"[MmLlHhVvCcSsQqTtZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_ARGS = {"M": 2, "L": 2, "H": 1, "V": 1, "C": 6, "S": 4, "Q": 4, "T": 2, "Z": 0}


def parse(d):
    ops, xy = [], []
    tokens = _TOKEN.findall(d)
    i, n = 0, len(tokens)
    x = y = sx = sy = 0.0  # current point, start of the subpath
    cx = cy = None  # the last control point, for S and T
    last = ""
    cmd = None
    while i < n:
        t = tokens[i]
        if t.isalpha():
            cmd = t
            i += 1
            if cmd in "Zz":
                ops.append("Z")
                x, y = sx, sy
                last, cx, cy = "Z", None, None
                continue
        elif cmd is None:
            raise ValueError("path data does not start with a command")
        up = cmd.upper()
        k = _ARGS[up]
        a = [float(v) for v in tokens[i : i + k]]
        if len(a) != k:
            raise ValueError("path data ends inside a command")
        i += k
        rel = cmd.islower()
        if up == "M":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            sx, sy = x, y
            ops.append("M")
            xy += (x, y)
            cmd = "l" if rel else "L"  # further pairs are line-tos
        elif up == "L":
            x, y = (x + a[0], y + a[1]) if rel else (a[0], a[1])
            ops.append("L")
            xy += (x, y)
        elif up == "H":
            x = x + a[0] if rel else a[0]
            ops.append("L")
            xy += (x, y)
        elif up == "V":
            y = y + a[0] if rel else a[0]
            ops.append("L")
            xy += (x, y)
        elif up in "CS":
            if up == "S":
                x1, y1 = (2 * x - cx, 2 * y - cy) if last == "C" else (x, y)
                rest = a
            else:
                x1, y1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                rest = a[2:]
            x2, y2, x3, y3 = (x + rest[0], y + rest[1], x + rest[2], y + rest[3]) if rel else rest
            ops.append("C")
            xy += (x1, y1, x2, y2, x3, y3)
            cx, cy, x, y = x2, y2, x3, y3
            last = "C"
            continue
        else:  # Q, T
            if up == "T":
                x1, y1 = (2 * x - cx, 2 * y - cy) if last == "Q" else (x, y)
                rest = a
            else:
                x1, y1 = (x + a[0], y + a[1]) if rel else (a[0], a[1])
                rest = a[2:]
            x2, y2 = (x + rest[0], y + rest[1]) if rel else rest
            ops.append("Q")
            xy += (x1, y1, x2, y2)
            cx, cy, x, y = x1, y1, x2, y2
            last = "Q"
            continue
        last, cx, cy = up, None, None
    return "".join(ops), xy


def matrix(text):
    """An SVG transform attribute as (a, b, c, d, e, f); None or '' is the identity."""
    m = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text or ""):
        v = [float(t) for t in re.split(r"[\s,]+", args.strip()) if t]
        if name == "matrix":
            t = tuple(v)
        elif name == "translate":
            t = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            t = (v[0], 0.0, 0.0, v[1] if len(v) > 1 else v[0], 0.0, 0.0)
        else:
            raise ValueError(f"unsupported transform {name!r}")
        m = compose(m, t)
    return m


def compose(m, t):
    """The matrix that applies t first, then m."""
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = t
    return (
        a * a2 + c * b2,
        b * a2 + d * b2,
        a * c2 + c * d2,
        b * c2 + d * d2,
        a * e2 + c * f2 + e,
        b * e2 + d * f2 + f,
    )


def apply(m, xy):
    a, b, c, d, e, f = m
    out = [0.0] * len(xy)
    for i in range(0, len(xy), 2):
        x, y = xy[i], xy[i + 1]
        out[i] = a * x + c * y + e
        out[i + 1] = b * x + d * y + f
    return out
