"""Colours of sd.plot: colour specifications, the categorical palettes and the continuous maps
(docs/api/plot.md §1.3, §2.1, §4.4). A colour inside the library is (r, g, b, a) in 0..1, sRGB."""

import numbers

from .._core import _type_name

NAMES = {
    "black": "#000000",
    "white": "#ffffff",
    "gray": "#808080",
    "grey": "#808080",
    "silver": "#c0c0c0",
    "red": "#ff0000",
    "green": "#008000",
    "blue": "#0000ff",
    "yellow": "#ffff00",
    "orange": "#ffa500",
    "purple": "#800080",
    "brown": "#a52a2a",
    "pink": "#ffc0cb",
    "cyan": "#00ffff",
    "magenta": "#ff00ff",
}

# Categorical palettes. jco is the default (the user's choice, 2026-11-22): the Journal of Clinical
# Oncology set of R's ggsci, a clear blue, a yellow, a grey and a brick red first. Lines and
# markers take the colour itself; filled shapes take its tint (fill_of) with the colour as a thin
# edge. tableau, deep and bright are the Tableau 10, seaborn "deep" and Paul Tol "bright" sets;
# okabe-ito is Okabe and Ito's colour-blind-safe set; the others are ggsci's journal palettes.
PALETTES = {
    "jco": (
        "#0073c2",
        "#efc000",
        "#868686",
        "#cd534c",
        "#7aa6dc",
        "#003c67",
        "#8f7700",
        "#3b3b3b",
        "#a73030",
        "#4a6990",
    ),
    "tableau": (
        "#4e79a7",
        "#f28e2b",
        "#e15759",
        "#76b7b2",
        "#59a14f",
        "#edc948",
        "#b07aa1",
        "#ff9da7",
        "#9c755f",
        "#bab0ac",
    ),
    "deep": (
        "#4c72b0",
        "#dd8452",
        "#55a868",
        "#c44e52",
        "#8172b3",
        "#937860",
        "#da8bc3",
        "#8c8c8c",
        "#ccb974",
        "#64b5cd",
    ),
    "bright": ("#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377", "#bbbbbb"),
    "okabe-ito": ("#0072b2", "#d55e00", "#009e73", "#e69f00", "#56b4e9", "#cc79a7", "#f0e442", "#000000"),
    "npg": (
        "#e64b35",
        "#4dbbd5",
        "#00a087",
        "#3c5488",
        "#f39b7f",
        "#8491b4",
        "#91d1c2",
        "#dc0000",
        "#7e6148",
        "#b09c85",
    ),
    "aaas": (
        "#3b4992",
        "#ee0000",
        "#008b45",
        "#631879",
        "#008280",
        "#bb0021",
        "#5f559b",
        "#a20056",
        "#808180",
        "#1b1919",
    ),
    "lancet": ("#00468b", "#ed0000", "#42b540", "#0099b4", "#925e9f", "#fdaf91", "#ad002a", "#adb6b6", "#1b1919"),
    "nejm": ("#bc3c29", "#0072b5", "#e18727", "#20854e", "#7876b1", "#6f99ad", "#ffdc91", "#ee4c97"),
}

# Continuous maps as evenly spaced anchors, interpolated linearly in sRGB: the perceptually
# uniform viridis, magma and cividis, and the diverging ColorBrewer RdBu (low values red).
CMAPS = {
    "viridis": (
        "#440154",
        "#482878",
        "#3e4989",
        "#31688e",
        "#26828e",
        "#1f9e89",
        "#35b779",
        "#6ece58",
        "#b5de2b",
        "#fde725",
    ),
    "magma": (
        "#000004",
        "#180f3d",
        "#440f76",
        "#721f81",
        "#9e2f7f",
        "#cd4071",
        "#f1605d",
        "#fd9668",
        "#feca8d",
        "#fcfdbf",
    ),
    "cividis": (
        "#00224e",
        "#123570",
        "#3b496c",
        "#575d6d",
        "#707173",
        "#8a8678",
        "#a59c74",
        "#c3b369",
        "#e1cc55",
        "#fee838",
    ),
    "rdbu": (
        "#67001f",
        "#b2182b",
        "#d6604d",
        "#f4a582",
        "#fddbc7",
        "#f7f7f7",
        "#d1e5f0",
        "#92c5de",
        "#4393c3",
        "#2166ac",
        "#053061",
    ),
}


def _hex(s):
    n = len(s) - 1
    try:
        v = int(s[1:], 16)
    except ValueError:
        return None
    if n == 6:
        return ((v >> 16) / 255.0, ((v >> 8) & 255) / 255.0, (v & 255) / 255.0, 1.0)
    if n == 8:
        return ((v >> 24) / 255.0, ((v >> 16) & 255) / 255.0, ((v >> 8) & 255) / 255.0, (v & 255) / 255.0)
    return None


def _spec():
    return "a colour name, '#rrggbb', '#rrggbbaa', 'C0' to 'C9' or 3 or 4 numbers in [0, 1]"


def parse(v, who, arg, palette):
    """(r, g, b, a) of a colour given by the user."""
    if isinstance(v, str):
        if v in NAMES:
            return _hex(NAMES[v])
        if v.startswith("#"):
            c = _hex(v)
            if c is not None:
                return c
        if len(v) == 2 and v[0] == "C" and v[1].isdigit():
            return cycle(palette, int(v[1]))
        raise ValueError(f"{who}(): '{arg}' must be {_spec()}, not {v!r}")
    if isinstance(v, (tuple, list)) and len(v) in (3, 4):
        if all(isinstance(c, numbers.Real) and not isinstance(c, bool) for c in v):
            c = tuple(float(x) for x in v)
            if all(0.0 <= x <= 1.0 for x in c):
                return c if len(c) == 4 else (*c, 1.0)
            raise ValueError(f"{who}(): '{arg}' must be {_spec()}; got a number outside [0, 1]")
    raise TypeError(f"{who}(): '{arg}' must be {_spec()}, not {_type_name(v)}")


def cycle(palette, i):
    """The i-th colour of a palette (it repeats)."""
    colours = PALETTES[palette]
    return _hex(colours[i % len(colours)])


def fill_of(c, t=0.42):
    """The tint a filled shape takes of a line colour: t of the way to white (alpha kept)."""
    return (c[0] + (1.0 - c[0]) * t, c[1] + (1.0 - c[1]) * t, c[2] + (1.0 - c[2]) * t, c[3])


def _linear(v):
    return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4


def _encode(v):
    v = min(1.0, max(0.0, v))
    return 12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055


def to_oklab(c):
    """(L, a, b) of an sRGB colour in Ottosson's OKLab, where equal steps look equal."""
    r, g, b = _linear(c[0]), _linear(c[1]), _linear(c[2])
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


def from_oklab(lab):
    L, a, b = lab
    l = (L + 0.3963377774 * a + 0.2158037573 * b) ** 3
    m = (L - 0.1055613458 * a - 0.0638541728 * b) ** 3
    s = (L - 0.0894841775 * a - 1.2914855480 * b) ** 3
    return (
        _encode(4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s),
        _encode(-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s),
        _encode(-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s),
        1.0,
    )


def ramp(low, high, n=11):
    """n colours from `low` to `high` through a pale cream, evenly spaced in OKLab: the map a
    surface takes of two palette colours (its low and its high side stay those colours, the
    middle is light, so the form reads under the light)."""
    a, b = to_oklab(low), to_oklab(high)
    mid = (0.93, 0.0, 0.055)  # a pale cream: the middle keeps some warmth and does not read as grey
    out = []
    for i in range(n):
        t = i / (n - 1)
        p, q, u = (a, mid, 2 * t) if t <= 0.5 else (mid, b, 2 * t - 1)
        u = u * u * (3 - 2 * u) if t <= 0.5 else 1 - (1 - u) * (1 - u) * (3 - 2 * (1 - u))  # eased around the middle
        out.append(from_oklab(tuple(p[k] + (q[k] - p[k]) * u for k in range(3))))
    return out


def with_alpha(c, a):
    return (c[0], c[1], c[2], c[3] * a)


def cmap(name, t):
    """(r, g, b, a) at t in [0, 1] of a continuous map (clamped; NaN is transparent)."""
    if not (t == t):
        return (0.0, 0.0, 0.0, 0.0)
    anchors = CMAPS[name]
    u = min(1.0, max(0.0, t)) * (len(anchors) - 1)
    i = min(int(u), len(anchors) - 2)
    f = u - i
    a, b = _hex(anchors[i]), _hex(anchors[i + 1])
    return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f, 1.0)


def table(name, n=256):
    """n colours of a map as bytes (RGBA8), for the rasterizer's lookups."""
    out = bytearray()
    for i in range(n):
        c = cmap(name, i / (n - 1))
        out += bytes(round(v * 255) for v in c)
    return bytes(out)
