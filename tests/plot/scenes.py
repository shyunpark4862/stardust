"""Hand-built scenes the plot tests, the golden cases and tools/plot_text.py share.

TEXT is the recorded text cache (layouts and glyph outlines of every string below, made by
`python tools/plot_text.py` with TeX Live): the tests read it frozen, so they need no TeX and
do not change when TeX does (docs/design/plot.md §8).
"""

import math
import pathlib

from stardust.plot import _scene as S

TEXT = pathlib.Path(__file__).parent / "data" / "text"
MM = 72.0 / 25.4
BLACK = (0.0, 0.0, 0.0, 1.0)
BLUE = (0.0, 0.447, 0.698, 1.0)
ORANGE = (0.835, 0.369, 0.0, 1.0)

# strings the text tests lay out besides those of the scenes
STRINGS = [
    r"Residuals",
    r"Fitted values",
    r"$x$",
    r"$\sin x$",
    r"$p < 0.001$",
    r"$\hat\beta_1 = 0.4217$",
    r"$\int_0^\infty e^{-x^2}\,dx = \frac{\sqrt\pi}{2}$",
    r"Group A",
    r"\textbf{A}",
    r"잔차",
    r"표준화 잔차",
    r"적합값 $\hat y$",
    "const",
]
NUMBERS = ["$0$", "$-1.5$", "$0.25$", "$1000$", "$-273.15$", "$10^{-3}$", "$10^{12}$", "$2^{-10}$"]


def curve(font):
    """A damped cosine with axes, ticks, a Hangul label, a rotated label and a formula."""
    w, h = 89 * MM, 66 * MM
    sc = S.Scene(w, h, font=font)
    x0, y0, x1, y1 = 38.0, 14.0, w - 8.0, h - 32.0
    sc.add(S.Path(*S.rect(0, 0, w, h), fill=(1.0, 1.0, 1.0, 1.0)))
    f = lambda t: math.exp(-t * t / 2) * math.cos(4 * t)  # noqa: E731
    pts = [(x0 + (x1 - x0) * i / 300, y1 - (y1 - y0) * (0.5 + 0.5 * f(-3 + 6 * i / 300))) for i in range(301)]
    sc.add(S.Path(*S.polyline(pts), stroke=BLUE, width=1.2, join="round", clip=(x0, y0, x1 - x0, y1 - y0)))
    for i in (60, 150, 240):
        sc.add(S.Path(*S.circle(*pts[i], 2.2), fill=ORANGE))
    sc.add(S.Path(*S.polyline([(x0, y0), (x0, y1), (x1, y1)]), stroke=BLACK, width=0.6, cap="square"))
    for i, v in enumerate((-3, -2, -1, 0, 1, 2, 3)):
        x = x0 + (x1 - x0) * i / 6
        sc.add(S.Path(*S.polyline([(x, y1), (x, y1 + 3)]), stroke=BLACK, width=0.6))
        sc.add(S.Text(f"${v}$", x, y1 + 5, size=7, ha="center", va="top"))
    for i, v in enumerate(("-1.0", "-0.5", "0.0", "0.5", "1.0")):
        y = y1 - (y1 - y0) * i / 4
        sc.add(S.Path(*S.polyline([(x0, y), (x0 - 3, y)]), stroke=BLACK, width=0.6))
        sc.add(S.Text(f"${v}$", x0 - 5, y, size=7, ha="right", va="center"))
    sc.add(S.Text(r"시간 $t$", (x0 + x1) / 2, h - 6, size=8, ha="center", va="bottom"))
    sc.add(S.Text(r"$e^{-t^2/2}\cos 4t$", 9, (y0 + y1) / 2, size=8, ha="center", va="center", rotation=90))
    sc.add(
        S.Text(
            r"$\displaystyle\int_{-\infty}^{\infty} e^{-t^2/2}\,dt=\sqrt{2\pi}$",
            x1 - 2,
            y0 + 2,
            size=7,
            ha="right",
            va="top",
        )
    )
    return sc


def anchors(font):
    """One string at every anchor and four rotations, with dashed guides and a translucent fill."""
    sc = S.Scene(200.0, 120.0, font=font)
    sc.add(S.Path(*S.rect(10, 10, 180, 100), fill=(0.9, 0.9, 0.9, 0.5), stroke=BLACK, width=0.5, dash=(3, 2)))
    for j, va in enumerate(("top", "center", "baseline", "bottom")):
        y = 25.0 + 22.0 * j
        sc.add(S.Path(*S.polyline([(10, y), (190, y)]), stroke=(0.5, 0.5, 0.5, 1.0), width=0.3))
        for i, ha in enumerate(("left", "center", "right")):
            sc.add(S.Text(r"$y = \frac{1}{2}gt^2$", 40.0 + 60.0 * i, y, size=8, ha=ha, va=va, color=BLUE))
    for k, rot in enumerate((0.0, 45.0, 90.0, 180.0)):
        sc.add(S.Text("Group A", 175.0, 30.0 + 20.0 * k, size=6, ha="center", va="center", rotation=rot))
    return sc


SCENES = {"curve": curve, "anchors": anchors}
