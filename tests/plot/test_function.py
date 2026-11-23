"""Function plots (docs/api/plot.md §4; docs/design/plot.md §4): the sampler finds bends, poles,
jumps and the edges of the domain, the y range leaves out what runs away, fast oscillation
becomes a band; level curves lie on their level; regions have their area; pictures have their
colours."""

import math
from array import array

import pytest

import stardust as sd
from stardust._core import plot as core
from stardust.plot import _color, _field, _figure, _function, _layers, _sample, _scene, _text

P = sd.plot


def measure(fig):
    return _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))


def finite(layer):
    return [(x, y) for x, y in zip(layer.x.tolist(), layer.y.tolist()) if math.isfinite(x) and math.isfinite(y)]


def test_a_smooth_function_is_followed_closely_with_few_samples():
    fig = P.function(sd.sin, 0, 2 * math.pi)
    (info,) = fig.info
    assert (
        info.kind == "function"
        and 257 <= info.n < 1200
        and info.poles == info.jumps == info.undefined == info.oscillating == ()
    )
    assert info.clipped == (False, False) and info.ylim == pytest.approx((-1.0, 1.0), abs=1e-4)
    pts = finite(fig._cell.layers[0])
    worst = max(abs(0.5 * (y0 + y1) - math.sin(0.5 * (x0 + x1))) for (x0, y0), (x1, y1) in zip(pts, pts[1:]))
    assert worst < 2e-4  # the chords stay within a hair of the curve
    assert pts[0][0] == 0.0 and pts[-1][0] == 2 * math.pi  # the ends exactly


def test_poles_are_found_cut_and_left_out_of_the_range():
    fig = P.function(sd.tan, -2 * math.pi, 2 * math.pi, asymptotes=True)
    (info,) = fig.info
    assert info.poles == pytest.approx([-1.5 * math.pi, -0.5 * math.pi, 0.5 * math.pi, 1.5 * math.pi], abs=2e-4)
    assert info.clipped == (True, True) and -15 < info.ylim[0] < -4 and 4 < info.ylim[1] < 15
    curve, *rules = fig._cell.layers
    assert sum(1 for y in curve.y.tolist() if y != y) == 4  # the line is cut at each pole
    assert [r.value for r in rules] == list(info.poles) and all(r.vertical and r.style == "--" for r in rules)
    m = measure(fig)
    assert -16 < m.ya.lo < -4 and 4 < m.ya.hi < 16  # the axis shows the bulk, not the poles
    gamma = P.function(sd.special.gamma, -4.2, 4.2).info[0]
    assert gamma.poles == pytest.approx([-4, -3, -2, -1, 0], abs=3e-4) and gamma.clipped[1] and gamma.ylim[1] < 30
    even = P.function(lambda x: 1 / (x * x), -1, 1).info[0]  # no sign change: no cut is needed, the range is still held
    assert even.clipped == (False, True) and even.ylim[1] < 1e3


def test_jumps_are_cut_without_a_riser():
    fig = P.function(sd.floor, -2.5, 2.5)
    (info,) = fig.info
    assert info.jumps == pytest.approx([-2, -1, 0, 1, 2], abs=2e-4) and info.poles == ()
    ys = fig._cell.layers[0].y.tolist()
    assert sum(1 for y in ys if y != y) == 5
    assert {y for y in ys if y == y} == {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0}  # no value between the steps
    steep = P.function(lambda x: sd.tanh(50 * x), -1, 1).info[0]
    assert steep.jumps == () and steep.n < 1500  # steep but continuous: resolved, not cut


def test_the_edge_of_the_domain_is_closed_in_on():
    fig = P.function([sd.sqrt, lambda x: sd.sqrt(1 - x * x)], -1.5, 3)
    root, arc = fig.info
    assert list(root.undefined[0]) == pytest.approx([-1.5, 0.0], abs=2e-4) and len(root.undefined) == 1
    assert [u for pair in arc.undefined for u in pair] == pytest.approx([-1.5, -1.0, 1.0, 3.0], abs=2e-4)
    pts = finite(fig._cell.layers[0])
    assert 0.0 <= pts[0][0] < 1e-4 and pts[0][1] < 0.011  # the curve starts at the axis
    apts = finite(fig._cell.layers[1])
    assert apts[0][1] < 0.02 and apts[-1][1] < 0.02


def test_growth_is_not_mistaken_for_a_pole():
    for f, a, b in ((sd.exp, 0, 10), (lambda x: x**3, -3, 3), (lambda x: x * x, -10, 10)):
        info = P.function(f, a, b).info[0]
        assert info.clipped == (False, False) and info.poles == ()
    assert P.function(sd.exp, 0, 10).info[0].ylim[1] == pytest.approx(math.exp(10))


def test_the_first_grid_is_not_in_step_with_a_periodic_function():
    n = _sample.N0
    info = P.function(lambda x: sd.sin(2 * math.pi * n * x), 0, 1).info[0]  # a period per grid step
    assert info.ylim[1] - info.ylim[0] > 1.5
    t = _sample.first_grid(2.0, 5.0).tolist()
    assert t[0] == 2.0 and t[-1] == 5.0 and all(a < b for a, b in zip(t, t[1:])) and len(t) == n + 1
    steps = {round((b - a) / (3.0 / n), 6) for a, b in zip(t, t[1:])}
    assert len(steps) >= 3 and max(steps) - min(steps) > 0.2  # uneven on purpose


def test_fast_oscillation_becomes_a_band():
    fig = P.function(lambda x: sd.sin(1 / x), -0.5, 0.5)
    (info,) = fig.info
    ((x0, x1),) = info.oscillating
    assert -0.03 < x0 < -0.002 and 0.002 < x1 < 0.03 and info.jumps == ()
    band, curve = fig._cell.layers
    assert isinstance(band, _layers.Area) and min(band.y1.tolist()) < -0.99 and max(band.y2.tolist()) > 0.99
    assert info.n <= _sample.CAP


def test_several_functions_share_the_range_and_take_labels():
    fig = P.function([sd.sin, sd.cos, lambda x: 0.5], 0, 6, labels=[r"$\sin\theta$", "Group A", "Residuals"])
    a, b, c = fig._cell.layers
    assert [la.label for la in (a, b, c)] == [r"$\sin\theta$", "Group A", "Residuals"] and a.color != b.color != c.color
    assert set(c.y.tolist()) == {0.5} and len(fig.info) == 3  # a function may return a number
    single = P.function(sd.sin, 0, 1, label="Group A", color="red", style="--", width=2)._cell.layers[0]
    assert (single.label, single.color, single.style, single.width) == ("Group A", (1.0, 0.0, 0.0, 1.0), "--", 2.0)
    fixed = P.function(sd.tan, -1.5, 1.5, ylim=(-2, 2))
    assert fixed._cell.ylim == [-2.0, 2.0] and fixed.info[0].ylim == pytest.approx((-2.0, 2.0))


def test_function_errors():
    cases = [
        (lambda: P.function(3, 0, 1), TypeError, "function(): 'f' must be callable, not int"),
        (
            lambda: P.function([], 0, 1),
            ValueError,
            "function(): 'f' must be a function or a non-empty sequence of functions",
        ),
        (lambda: P.function(sd.sin, 1, 1), ValueError, "function(): 'a' must be less than 'b'"),
        (lambda: P.function(sd.sin, 0, math.inf), ValueError, "function(): 'b' must be finite"),
        (lambda: P.function(sd.sin, 0, 1, ylim=3), TypeError, "function(): 'ylim' must be a pair (low, high), not int"),
        (
            lambda: P.function(sd.sin, 0, 1, ylim=(2, 1)),
            ValueError,
            "function(): 'ylim[0]' must be less than 'ylim[1]'",
        ),
        (
            lambda: P.function(sd.sin, 0, 1, asymptotes=1),
            TypeError,
            "function(): 'asymptotes' must be True or False, not int",
        ),
        (
            lambda: P.function(lambda x: x.tolist(), 0, 1),
            TypeError,
            "function(): 'f' must return an sd.ndarray (it is called with one), not list",
        ),
        (
            lambda: P.function(lambda x: x[:3], 0, 1),
            ValueError,
            "function(): 'f' must return an array of its argument's shape (257,), not (3,)",
        ),
        (
            lambda: P.function(lambda x: x + 1j, 0, 1),
            TypeError,
            "function(): 'f' must return real values, not complex128",
        ),
        (
            lambda: P.function([sd.sin, sd.cos], 0, 1, label="a"),
            ValueError,
            "function(): several functions take 'labels', not 'label'",
        ),
        (
            lambda: P.function([sd.sin, sd.cos], 0, 1, labels=["a"]),
            ValueError,
            "function(): 'labels' must be 2 strings, one per function",
        ),
        (
            lambda: P.function([sd.sin, sd.cos], 0, 1, color="red"),
            ValueError,
            "function(): 'color' cannot be given for several functions (they take the palette's colours)",
        ),
        (lambda: P.parametric(sd.sin, 3, 0, 1), TypeError, "parametric(): 'fy' must be callable, not int"),
        (lambda: P.polar(sd.sin, 2, 1), ValueError, "polar(): 'a' must be less than 'b'"),
    ]
    for call, kind, message in cases:
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message, message
    with pytest.raises(ZeroDivisionError):  # the function's own error passes
        P.function(lambda x: 1 / 0, 0, 1)


def test_parametric_and_polar_keep_the_plane_square():
    fig = P.parametric(sd.cos, sd.sin, 0, 2 * math.pi)
    pts = finite(fig._cell.layers[0])
    assert max(abs(math.hypot(x, y) - 1.0) for x, y in pts) < 1e-12 and fig._cell.aspect == 1.0
    worst = max(1.0 - math.hypot(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])) for a, b in zip(pts, pts[1:]))
    assert worst < 3e-4  # the chords are close to the circle
    sc = fig._root._build("t")
    axes = next(i for i in sc.items if isinstance(i, _scene.Path) and i.ops == "MLL" and i.cap == "square")
    w, h = axes.xy[4] - axes.xy[0], axes.xy[3] - axes.xy[1]
    assert w == pytest.approx(h, rel=1e-4)  # equal ranges: a square plot
    rose = P.polar(lambda t: sd.cos(4 * t), 0, 2 * math.pi)
    assert max(math.hypot(x, y) for x, y in finite(rose._cell.layers[0])) == pytest.approx(1.0, abs=1e-6)
    assert rose.info[0].kind == "polar" and rose.info[0].ylim is None


def test_a_plane_curve_that_is_a_straight_line_keeps_a_wide_cell():
    """polar(1 / cos t) is the line x = 1: its x values differ by rounding only, and with equal
    units the cell would have no width. The flat direction takes the other's span around the
    line, and a curve that is one point takes a unit both ways."""
    th = _figure.THEMES["paper"]
    fig = P.polar(lambda t: 1 / sd.cos(t), 0, 2 * math.pi)
    (x0, x1), (y0, y1) = (_figure._range(fig._cell, axis, th) for axis in "xy")
    assert x1 - x0 == pytest.approx(y1 - y0, rel=1e-9) and y1 - y0 > 10.0
    assert 0.5 * (x0 + x1) == pytest.approx(1.0, abs=1e-9)
    ticks = _figure._axis(fig._cell, "x", 150.0, th).ticks
    assert len(ticks) >= 3 and len(set(ticks)) == len(ticks)
    flat = P.parametric(lambda t: t, lambda t: 2 + 0 * t, 0, 4)  # the line y = 2
    (x0, x1), (y0, y1) = (_figure._range(flat._cell, axis, th) for axis in "xy")
    assert x1 - x0 == pytest.approx(y1 - y0, rel=1e-9) and 0.5 * (y0 + y1) == pytest.approx(2.0)
    dot = P.parametric(lambda t: 1 + 0 * t, lambda t: 2 + 0 * t, 0, 1)
    (x0, x1), (y0, y1) = (_figure._range(dot._cell, axis, th) for axis in "xy")
    assert x1 - x0 == pytest.approx(y1 - y0) and x0 < 1.0 < x1 and y0 < 2.0 < y1
    circle = P.parametric(sd.cos, sd.sin, 0, 2 * math.pi)  # nothing flat: the values' own range
    assert circle._cell.layers[0].held == ()


def test_level_curves_lie_on_their_level_and_close():
    fig = P.implicit(lambda x, y: x * x + y * y, -2, 2, -2, 2, level=1.0)
    layer = fig._cell.layers[0]
    pts = finite(layer)
    assert max(abs(x * x + y * y - 1.0) for x, y in pts) < 5e-6  # the secant step: far inside a cell's size (0.0125)
    assert pts[0] == pts[-1] and len(pts) > 300 and not any(v != v for v in layer.x.tolist())  # one closed curve
    assert layer.extent("x", False) == (-2.0, 2.0, (-2.0, 2.0)) and fig._cell.aspect == 1.0
    two = P.implicit(lambda x, y: x * x - 1, -2, 2, -2, 2)._cell.layers[0]
    assert sum(1 for v in two.x.tolist() if v != v) == 1  # two lines, a gap between them
    assert sorted({round(x, 9) for x in two.x.tolist() if x == x}) == [-1.0, 1.0]
    saddle = P.implicit(lambda x, y: x * x - y * y, -2, 2, -2, 2)._cell.layers[0]
    spts = finite(saddle)
    assert (
        max(abs(abs(x) - abs(y)) for x, y in spts) < 1e-3 and len(spts) > 600
    )  # the two diagonals (they cross in one cell)
    nothing = P.implicit(lambda x, y: x * x + y * y + 1, -1, 1, -1, 1)._cell.layers[0]
    assert nothing.x.shape[0] == 0 and nothing.extent("y", False)[:2] == (-1.0, 1.0)
    hole = P.implicit(lambda x, y: sd.where(x < 0, math.nan, y - x), -1, 1, -1, 1)._cell.layers[0]
    assert min(x for x, _ in finite(hole)) >= 0.0  # cells with a NaN corner are left out


def test_region_has_the_area_of_the_set():
    fig = P.region(lambda x, y: 1 - x * x - y * y, -1.5, 1.5, -1.5, 1.5, alpha=0.5, color="red", label="Group A")
    (fill,) = fig._cell.layers
    assert fill.color == (1.0, 0.0, 0.0, 0.5) and fill.label == "Group A"

    def areas(layer):
        out, i, xy = [], 0, layer.xy
        for sub in layer.ops.split("Z")[:-1]:
            n = len(sub)
            pts = [(xy[i + 2 * k], xy[i + 2 * k + 1]) for k in range(n)]
            i += 2 * n
            out.append(abs(0.5 * sum(p[0] * q[1] - q[0] * p[1] for p, q in zip(pts, pts[1:] + pts[:1]))))
        return out

    assert fill.rule == "evenodd" and fill.ops.count("M") == 1  # one outline
    assert areas(fill)[0] == pytest.approx(math.pi, rel=2e-5)
    assert P.region(lambda x, y: x - 9, -1, 1, -1, 1)._cell.layers[0].ops == ""  # empty
    whole = P.region(lambda x, y: x + 9, -1, 1, -1, 1)._cell.layers[0]
    assert whole.ops.count("M") == 1 and areas(whole)[0] == pytest.approx(4.0)  # the box itself
    ring = P.region(lambda x, y: sd.minimum(1 - x * x - y * y, x * x + y * y - 0.25), -1.5, 1.5, -1.5, 1.5)
    outer, inner = sorted(areas(ring._cell.layers[0]), reverse=True)  # even-odd: the inner loop is a hole
    assert outer == pytest.approx(math.pi, rel=1e-4) and inner == pytest.approx(math.pi / 4, rel=1e-4)
    holes = P.region(lambda x, y: sd.where(x > 0.5, math.nan, 1 - x * x - y * y), -1.5, 1.5, -1.5, 1.5)
    assert holes._cell.layers[0].rule == "nonzero"  # a hole in the domain opens the outline: cell by cell


def test_contours_levels_colours_and_labels():
    f = lambda x, y: x * x + y * y  # noqa: E731
    fig = P.contour(f, -2, 2, -2, 2, levels=[1, 2, 3])
    *lines, labels = fig._cell.layers
    assert len(lines) == 3 and [la.color for la in lines] == [_color.cmap(_color.CMAP, t) for t in (0.0, 0.5, 1.0)]
    assert [s for s, *_ in labels.items] == ["$1$", "$2$", "$3$"] and labels.strings() == ["$1$", "$2$", "$3$"]
    for line, level in zip(lines, (1, 2, 3)):
        assert max(abs(x * x + y * y - level) for x, y in finite(line)) < 1e-5
    auto = P.contour(f, -2, 2, -2, 2, levels=4, labels=False, color="black")._cell.layers
    assert 3 <= len(auto) <= 8 and all(la.color == (0.0, 0.0, 0.0, 1.0) for la in auto)
    filled = P.contour(f, -2, 2, -2, 2, levels=[1, 2, 3], filled=True, labels=False, cmap="magma")._cell.layers
    assert [type(la).__name__ for la in filled] == ["Fill"] * 4 + [
        "Line"
    ] * 3  # the base and a band per level, then the lines
    assert filled[0].ops == "MLLLZ" and filled[1].color != filled[2].color
    for call, kind, message in (
        (lambda: P.contour(f, -2, 2, -2, 2, levels=0), ValueError, "contour(): 'levels' must be at least 1"),
        (
            lambda: P.contour(f, -2, 2, -2, 2, levels=1.5),
            TypeError,
            "contour(): 'levels' must be an int or a sequence of numbers, not float",
        ),
        (lambda: P.contour(f, -2, 2, -2, 2, levels=[]), ValueError, "contour(): 'levels' is empty"),
        (
            lambda: P.contour(f, -2, 2, -2, 2, cmap="jet"),
            ValueError,
            "contour(): 'cmap' must be 'cool', 'viridis', 'magma', 'cividis', 'rdbu', 'spectral', 'coolwarm', "
            "'sunrise' or 'lagoon', not 'jet'",
        ),
        (
            lambda: P.contour(f, -2, 2, -2, 2, cmap="magma", color="red"),
            ValueError,
            "contour(): 'color' and 'cmap' cannot both be given",
        ),
        (
            lambda: P.contour(lambda x, y: x * math.nan, -2, 2, -2, 2),
            ValueError,
            "contour(): 'f' has no finite value on the grid",
        ),
        (lambda: P.implicit(lambda x, y: 1, 2, 1, 0, 1), ValueError, "implicit(): 'x0' must be less than 'x1'"),
        (
            lambda: P.region(lambda x, y: [1], 0, 1, 0, 1),
            TypeError,
            "region(): 'f' must return an sd.ndarray (it is called with two), not list",
        ),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message, message


def test_density_is_a_picture_with_a_colour_bar():
    fig = P.density(lambda x, y: x + 0 * y, 0, 1, 0, 2, cmap="magma")
    (pic,) = fig._cell.layers
    assert (pic.iw, pic.ih, pic.smooth, pic.box) == (_field.DX0, _field.DY0, True, (0.0, 1.0, 0.0, 2.0))
    table = _color.table("magma")
    row = pic.rgba[: 4 * pic.iw]
    assert row[:4] == table[:4] and row[-4:] == table[-4:]  # the lowest value to the highest across a row
    assert pic.rgba[: 4 * pic.iw] == pic.rgba[-4 * pic.iw :]  # constant along y
    lo, hi = fig._cell.colorbar[1:]
    assert lo == pytest.approx(0.5 / _field.DX0) and hi == pytest.approx(1 - 0.5 / _field.DX0)  # the pixel centres
    with_bar, without = measure(fig), measure(P.density(lambda x, y: x + 0 * y, 0, 1, 0, 2, colorbar=False))
    assert (
        with_bar.right > without.right + 15 and with_bar.xa.lo == 0.0 and with_bar.ya.hi == 2.0
    )  # no padding around a picture
    images = [i for i in fig._root._build("t").items if isinstance(i, _scene.Image)]
    assert [(i.iw, i.ih) for i in images] == [(_field.DX0, _field.DY0), (1, 256)]
    held = P.density(lambda x, y: x + 0 * y, 0, 1, 0, 2, vmin=0.25, vmax=0.5)._cell.layers[0].rgba
    assert (
        held[:4] == _color.table(_color.CMAP)[:4] and held[4 * 400 : 4 * 401] == _color.table(_color.CMAP)[-4:]
    )  # clamped to the ends
    with pytest.raises(ValueError, match=r"density\(\): 'vmin' must be less than 'vmax'"):
        P.density(lambda x, y: x, 0, 1, 0, 1, vmin=1, vmax=1)
    for fmt in (fig.to_svg(), fig.to_pdf(), fig.to_png(dpi=60)):
        assert len(fmt) > 2000
    assert "data:image/png;base64," in fig.to_svg() and b"/Subtype /Image" in fig.to_pdf()


def test_phase_portrait_colours():
    fig = P.phase(lambda z: z, -1, 1, -1, 1, contours=False)
    (pic,) = fig._cell.layers
    px = lambda i, j: tuple(pic.rgba[4 * (j * pic.iw + i) : 4 * (j * pic.iw + i) + 3])  # noqa: E731
    mid = pic.ih // 2
    assert px(pic.iw - 1, mid)[0] == 255 and px(pic.iw - 1, mid)[2] < 60  # positive reals: red
    assert px(0, mid)[1] > 200 and px(0, mid)[2] > 200 and px(0, mid)[0] < 60  # negative reals: cyan
    assert px(pic.iw // 2, 0)[1] > 200  # +i: between yellow and green
    assert fig._cell.aspect == 1.0
    banded = P.phase(lambda z: z, -1, 1, -1, 1)._cell.layers[0].rgba
    assert banded != pic.rgba
    with pytest.raises(
        TypeError, match=r"phase\(\): 'f' must return an sd.ndarray of its argument's shape \(360, 480\)"
    ):
        P.phase(lambda z: [1.0], -1, 1, -1, 1)


def test_core_colour_helpers():
    table = _color.table(_color.CMAP)
    v = array("d", [-1.0, 0.0, 0.5, 1.0, 2.0, math.nan, math.inf]).tobytes()
    out = core.colormap(v, 0.0, 1.0, table)
    assert out[:4] == out[4:8] == table[:4] and out[12:16] == out[16:20] == out[24:28] == table[-4:]
    assert out[8:12] == table[4 * 128 : 4 * 129] and out[20:24] == b"\0\0\0\0"
    assert core.colormap(v, 1.0, 1.0, table)[:4] == table[:4]  # an empty range: the first colour
    re_, im_ = (
        array("d", [1.0, 0.0, -1.0, 0.0, 0.0, math.inf, math.nan]),
        array("d", [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0]),
    )
    c = core.phase_colors(re_.tobytes(), im_.tobytes(), False)
    cols = [tuple(c[4 * i : 4 * i + 4]) for i in range(7)]
    assert (
        cols[0] == (255, 38, 38, 255)
        and cols[4] == (0, 0, 0, 255)
        and cols[5] == (255, 255, 255, 255)
        and cols[6] == (0, 0, 0, 0)
    )
    assert cols[2][1] == cols[2][2] == 255 and cols[1][1] == 255 and cols[3][2] == 255
    with pytest.raises(ValueError, match="re and im must be packed doubles of one length"):
        core.phase_colors(re_.tobytes(), im_.tobytes()[:8], False)


def test_images_in_the_rasterizer():
    red, blue = bytes([255, 0, 0, 255]), bytes([0, 0, 255, 255])
    c = core.Canvas(4, 2)
    c.image(red + blue, 2, 1, 0.0, 0.0, 4.0, 2.0, False, ())
    px = c.rgba8()
    assert px[:8] == red * 2 and px[8:16] == blue * 2  # flat squares
    c = core.Canvas(4, 1)
    c.image(red + blue, 2, 1, 0.0, 0.0, 4.0, 1.0, True, ())
    px = c.rgba8()
    assert px[0:4] == red and px[12:16] == blue and px[4] > px[8] > 0 and px[6] < px[10]  # a ramp between the centres
    c = core.Canvas(4, 2)
    c.image(red, 1, 1, 1.5, 0.0, 2.0, 2.0, False, (0.0, 1.0, 4.0, 2.0))
    assert [c.rgba8()[4 * i + 3] for i in range(8)] == [0, 0, 0, 0, 0, 128, 255, 128]  # partial pixels, clipped rows
    half = core.Canvas(1, 1)
    half.image(bytes([255, 255, 255, 128]), 1, 1, 0.0, 0.0, 1.0, 1.0, False, ())
    assert half.rgba8() == bytes([255, 255, 255, 128])
    with pytest.raises(ValueError, match="rgba must hold iw x ih pixels of 4 bytes"):
        c.image(red, 2, 1, 0.0, 0.0, 1.0, 1.0, False, ())
    big = bytes(range(256)) * 4 * 64  # the rows are painted on the pool: the same at any thread count
    old, outs = sd.get_num_threads(), []
    try:
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            cv = core.Canvas(700, 500)
            cv.image(big, 256, 64, 3.3, 2.2, 690.0, 490.0, True, ())
            outs.append(cv.rgba8())
    finally:
        sd.set_num_threads(old)
    assert outs[0] == outs[1] == outs[2]
    assert isinstance(_function.PlotInfo, type)


def test_a_curve_is_sampled_finer_near_itself():
    from stardust.plot import _field

    calls = []

    def circle(x, y):
        calls.append(x.size)
        return x * x + y * y - 1.0

    xs, ys, F = _field.grid(circle, -1.5, 1.5, -1.5, 1.5, _field.NX, _field.NY, "t")
    fx, fy, fine = _field.near(circle, xs, ys, F, 0.0, "t")
    assert len(fx) == _field.NX * _field.FINE + 1 and len(fy) == _field.NY * _field.FINE + 1
    assert fx[:: _field.FINE] == pytest.approx(xs, abs=1e-15) and calls[1] < 0.2 * fine.size  # the band only
    X, Y = sd.meshgrid(sd.asarray(fx), sd.asarray(fy), indexing="xy")
    true = circle(X, Y)
    assert bool(sd.all((fine > 0.0) == (true > 0.0)))  # the side of the level is right everywhere
    band = sd.abs(true) < 0.01
    assert bool(sd.all(fine[band] == true[band]))  # true values by the curve
    far = _field.near(circle, xs, ys, F, 50.0, "t")
    assert far[2] is F  # no curve: nothing sampled
    fig = sd.plot.implicit(lambda x, y: x * x - y * y, -1, 1, -1, 1)
    x, y = fig._cell.layers[0].x, fig._cell.layers[0].y
    ok = sd.isfinite(x)
    assert float(sd.max(sd.abs(sd.abs(x[ok]) - sd.abs(y[ok])))) < 1e-3  # both branches, through the crossing
    assert float(sd.min(sd.abs(x[ok]) + sd.abs(y[ok]))) < 2e-3


def test_level_curves_in_the_core_and_contours_by_a_pole():
    xs, ys = [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]
    F = sd.asarray([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # a peak in the middle
    cells = _field.Cells(lambda x, y: 0.0 * x, xs, ys, F, 0.5, "t", polish=False)
    X, Y = cells.arrays()
    assert X.tolist() == [1.0, 0.5, 1.0, 1.5, 1.0] and Y.tolist() == [0.5, 1.0, 1.5, 1.0, 0.5]  # a closed diamond
    assert cells.curves() == [[(1.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 1.0), (1.0, 0.5)]]
    ops, xy = cells.inside()
    assert ops == "MLLZ" * 4 and len(xy) == 24  # the corner triangles of the four cells
    grid = _field.Grid(xs, ys, F)
    assert _field.Cells(None, xs, ys, F, 2.0, "t", polish=False, grid=grid).arrays()[0].shape[0] == 0
    with pytest.raises(ValueError, match="the grid must hold"):
        core.Levels(b"\0" * 16, b"\0" * 16, b"\0" * 16)
    with pytest.raises(ValueError, match="points must hold"):
        grid.core.curves(0.5, b"")
    assert core.path_text(b"MLQZ", array("d", [1.0, -2.5, 0.00004, 1e-9, 3.0, 3.0, 6.0, 0.0]).tobytes(), 4, False) == (
        "M1 -2.5L0 0Q3 3 6 0Z"
    )
    assert core.path_text(b"MLZ", array("d", [1.25, 2.0, -0.5, 100000.123456]).tobytes(), 4, True) == (
        "1.25 2 m\n-0.5 100000.1235 l\nh"
    )
    with pytest.raises(ValueError, match="do not match"):
        core.path_text(b"ML", b"\0" * 8, 4, False)
    pole = P.contour(lambda x, y: 1 / (x * x + y * y), -1, 1, -1, 1)
    lines = [layer for layer in pole._cell.layers if isinstance(layer, _layers.Line)]
    assert len(lines) >= 5  # levels over the bulk of the values, not one lost near the pole's top
    busy = P.contour(lambda x, y: sd.sin(1 / (x * x + y * y + 0.02)), -1, 1, -1, 1, levels=6, filled=True)
    assert busy._cell.colorbar is not None and not any(
        isinstance(layer, _field.ContourLabels) for layer in busy._cell.layers
    )


def test_samples_and_refine_set_the_sampling():
    """Every function plot takes the two: the points of the first grid, and how often a piece of
    it may be divided again (0: the first grid as it is)."""
    assert P.function(sd.sin, 0, 6, samples=11, refine=0).info[0].n == 11
    assert P.function(sd.sin, 0, 6, samples=1001).info[0].n == 1001
    plain = P.function(sd.tan, -3, 3, refine=0).info[0]
    assert plain.n == 257 and plain.poles == () and len(P.function(sd.tan, -3, 3).info[0].poles) == 2
    coarse, deep = (P.function(lambda x: sd.sin(1 / x), 0.01, 1, refine=r).info[0].n for r in (2, 10))
    assert 257 < coarse < deep
    assert P.parametric(sd.cos, sd.sin, 0, 6, samples=50, refine=0).info[0].n == 50
    assert P.polar(lambda t: 1 + 0 * t, 0, 6, samples=50, refine=0).info[0].n == 50
    circle = lambda x, y: x * x + y * y - 1  # noqa: E731
    few = P.implicit(circle, -2, 2, -2, 2, samples=41, refine=0)._cell.layers[0].x.shape[0]
    assert few < P.implicit(circle, -2, 2, -2, 2, samples=41)._cell.layers[0].x.shape[0]
    assert P.region(circle, -2, 2, -2, 2, samples=(31, 21), refine=0) is not None
    assert P.contour(circle, -2, 2, -2, 2, samples=51, refine=0) is not None
    pic = P.density(lambda x, y: x * y, 0, 1, 0, 1, samples=(20, 10))._cell.layers[0]
    assert (pic.iw, pic.ih) == (20, 10) and P.phase(lambda z: z, -1, 1, -1, 1, samples=30)._cell.layers[0].iw == 30
    pole = lambda x, y: 1 / (x * x + y * y)  # noqa: E731
    sp = P.surface(pole, -1, 1, -1, 1, samples=(41, 61), refine=1)._cell.layers[0].surfaces[0]
    assert tuple(sp[0].shape) == (61, 41) and len(sp[6][1]) == 1  # one level of finer cells
    assert P.surface(pole, -1, 1, -1, 1, refine=0)._cell.layers[0].surfaces[0][6] is None
    assert (
        P.curve3d(sd.cos, sd.sin, lambda t: t, 0, 60, samples=100, refine=0)._cell.layers[0].curves[0][0].shape[0]
        == 100
    )
    for bad, kind, text in (
        (dict(samples=2), ValueError, "'samples' must be between 3 and 200000"),
        (dict(samples=2.5), TypeError, "'samples' must be an int, not float"),
        (dict(samples=(3, 4)), TypeError, "'samples' must be an int, not tuple"),
        (dict(refine=11), ValueError, "'refine' must be between 0 and 10"),
        (dict(refine=True), TypeError, "'refine' must be an int, not bool"),
    ):
        with pytest.raises(kind, match=text):
            P.function(sd.sin, 0, 1, **bad)
    with pytest.raises(ValueError, match=r"surface\(\): 'samples' as a pair must hold 2 numbers, not 3"):
        P.surface(pole, -1, 1, -1, 1, samples=(3, 4, 5))
    with pytest.raises(ValueError, match=r"implicit\(\): 'refine' must be between 0 and 1"):
        P.implicit(circle, -2, 2, -2, 2, refine=2)


def test_function_ranges_settle_on_the_curve():
    bump = lambda x: sd.where(sd.abs(x) < 1, sd.exp(-1 / (1 - sd.minimum(x * x, 0.999999))), 0.0)  # noqa: E731
    info = P.function(bump, -10, 10).info[0]
    assert info.ylim[1] == pytest.approx(math.exp(-1), rel=1e-3) and info.clipped == (False, False)
    log = P.function(sd.log, -1, 3).info[0]
    assert log.jumps == () and log.poles == ()  # the end of the domain is no jump
    spike = lambda x: sd.exp(-(((x - 0.5) / 1e-4) ** 2))  # noqa: E731
    assert P.function(spike, 0, 1, samples=20001, ylim=(0, 1.05)).info[0].clipped == (False, False)
    folium = P.parametric(lambda t: 3 * t / (1 + t**3), lambda t: 3 * t * t / (1 + t**3), -20, 20)
    lo, hi, _ = folium._cell.layers[0].extent("x", False)
    assert -1 < lo and 1.55 < hi < 2  # the whole loop (x up to 1.587), not the arms running off to a pole
    lo, hi, _ = folium._cell.layers[0].extent("y", False)
    assert -2.5 < lo and 1.55 < hi < 2.5
    assert P.phase(lambda z: 2.0, -1, 1, -1, 1)._cell.layers[0].iw == _field.DX0  # a constant, as elsewhere


def test_grid_values_take_the_place_of_a_function():
    x, y = sd.linspace(-2.0, 2.0, 41), sd.linspace(-1.0, 1.0, 21)
    X, Y = sd.meshgrid(x, y, indexing="xy")
    Z = X * X + Y * Y
    line = P.implicit(Z, -2, 2, -1, 1, level=1.0)._cell.layers[0]
    pts = [(a, b) for a, b in zip(line.x.tolist(), line.y.tolist()) if a == a]
    assert max(abs(a * a + b * b - 1.0) for a, b in pts) < 0.02  # linear interpolation on the grid
    pic = P.density(Z, -2, 2, -1, 1)._cell.layers[0]
    assert (pic.iw, pic.ih) == (41, 21)
    surf = P.surface(Z, -2, 2, -1, 1)._cell.layers[0].surfaces[0]
    assert tuple(surf[0].shape) == (21, 41) and surf[6] is None
    assert P.contour(Z, -2, 2, -1, 1, filled=True)._cell.colorbar is not None
    with pytest.raises(ValueError, match=r"contour\(\): 'samples' and 'refine' are for a function"):
        P.contour(Z, -2, 2, -1, 1, samples=11)
    with pytest.raises(TypeError, match=r"region\(\): 'f' must be callable or a 2-D array of values, not str"):
        P.region("Z", -2, 2, -1, 1)


def test_vector_fields_draw_arrows_and_even_streamlines():
    fig = P.vectorfield(lambda x, y: -y, lambda x, y: x, -2, 2, -2, 2, samples=(5, 4))
    arrows = fig._cell.layers[0]
    assert len(arrows.x) == 20 and fig._cell.aspect == 1.0 and fig._cell.colorbar is not None
    longest = max(math.hypot(u, v) for u, v in zip(arrows.u, arrows.v))
    assert longest == pytest.approx(0.9 * min(4 / 5, 4 / 4), rel=1e-9)  # the longest fills 90% of a cell
    black = P.vectorfield(lambda x, y: -y, lambda x, y: x, -2, 2, -2, 2, color="black")
    assert set(black._cell.layers[0].colors) == {(0.0, 0.0, 0.0, 1.0)} and black._cell.colorbar is None
    raw = array(
        "d",
        core.streamlines(
            array("d", [1.0] * 25).tobytes(), array("d", [0.0] * 25).tobytes(), 5, 5, 0.0, 1.0, 0.0, 1.0, 10, 10, 2.0
        ),
    )
    lines = [chunk for chunk in _split_nan(raw)]
    assert len(lines) == 10  # a uniform flow to the right: one line a row of cells
    for pts in lines:
        xs = pts[0::3]
        assert xs == sorted(xs) and xs[-1] - xs[0] > 0.8  # along the flow, across the box
    swirl = P.streamlines(lambda x, y: -y, lambda x, y: x, -1, 1, -1, 1, cells=10)
    assert swirl._cell.layers[0].xyz and swirl._root._build("t") is not None
    with pytest.raises(TypeError, match=r"streamlines\(\): 'fu' and 'fv' must both be functions or both"):
        P.streamlines(lambda x, y: x, [[1.0, 2.0], [3.0, 4.0]], 0, 1, 0, 1)
    with pytest.raises(ValueError, match=r"streamlines\(\): 'cells' must be between 2 and 400"):
        P.streamlines(lambda x, y: x, lambda x, y: y, 0, 1, 0, 1, cells=500)


def _split_nan(xyz):
    out, cur = [], []
    for i in range(0, len(xyz), 3):
        if xyz[i] != xyz[i]:
            out.append(cur)
            cur = []
        else:
            cur += xyz[i : i + 3]
    if cur:
        out.append(cur)
    return out


def test_vector_fields_of_values_name_their_arguments_and_keep_the_arrows_whole():
    x, y = sd.linspace(-2.0, 2.0, 15), sd.linspace(-1.0, 1.0, 8)
    X, Y = sd.meshgrid(x, y, indexing="xy")
    for plot in (P.vectorfield, P.streamlines):
        name = plot.__name__
        with pytest.raises(ValueError, match=rf"{name}\(\): 'samples' and 'refine' are for a function"):
            plot(-Y, X, -2, 2, -1, 1, samples=5)
        with pytest.raises(TypeError, match=rf"{name}\(\): 'fv' must be callable or a 2-D array of values, not int"):
            plot(lambda a, b: b, 3, -2, 2, -1, 1)
        with pytest.raises(ValueError, match=rf"{name}\(\): 'fu' as values must be a 2-D array of at least 2 x 2"):
            plot([[1.0, 2.0]], [[1.0, 2.0]], -2, 2, -1, 1)
    arrows = P.vectorfield(-Y, X, -2, 2, -1, 1)._cell.layers[0]
    lo, hi, sticky = arrows.extent("y", False)
    assert lo < -1.0 and hi > 1.0 and sticky == ()  # the arrows on the box's edge are not cut
    lo, hi, sticky = arrows.extent("x", False)
    assert lo < -2.0 and hi > 2.0 and sticky == ()
    inner = P.vectorfield(lambda a, b: -b, lambda a, b: a, -2, 2, -1, 1)._cell.layers[0]
    assert inner.extent("x", False) == (-2.0, 2.0, (-2.0, 2.0))  # at the cells' centres: within the box


def test_vector_field_samples_colour_key_and_colour_bar():
    def counts(**kw):
        layer = P.vectorfield(lambda a, b: b, lambda a, b: -sd.sin(a), -6.3, 6.3, -3, 3, **kw)._cell.layers[0]
        return len(set(layer.x)), len(set(layer.y))

    assert counts() == (21, 10) and counts(samples=12) == (12, 6) and counts(samples=(12, 5)) == (12, 5)
    u, v = (lambda a, b: -b), (lambda a, b: a)
    fig = P.vectorfield(u, v, -2, 2, -2, 2, label="arrows").streamlines(u, v, -2, 2, -2, 2, label="lines")
    assert fig._cell.layers[0].key == _color.cmap("cool", 0.5)  # the middle of the colour map, as streamlines
    h = lambda a, b: sd.sin(a) * sd.cos(b)  # noqa: E731
    bar = P.density(h, -3, 3, -3, 3)._cell.colorbar
    over = P.density(h, -3, 3, -3, 3).vectorfield(lambda a, b: 3 * a, lambda a, b: 3 * b, -3, 3, -3, 3)
    assert over._cell.colorbar == bar  # a cell has one bar: the density's stays
    alone = P.vectorfield(lambda a, b: 3 * a, lambda a, b: 3 * b, -3, 3, -3, 3)
    assert alone._cell.colorbar[2] > 1.0


def test_a_narrow_peak_found_between_the_first_samples_sets_the_range():
    spike = lambda x: sd.exp(-(((x - 0.5) / 1e-4) ** 2))  # noqa: E731
    info = P.function(spike, 0, 1).info[0]
    assert info.clipped == (False, False) and info.ylim[1] == pytest.approx(1.0, abs=0.05)
    flat = P.function(lambda x: 0.0 * x + 2.0, 0, 1).info[0]  # a level line keeps its range
    assert flat.clipped == (False, False) and flat.ylim == (2.0, 2.0)


def test_values_and_constant_returns_may_be_numpy():
    import numpy as np

    Z = np.ones((3, 3)) * np.arange(3)
    fig = P.contour(Z, 0, 1, 0, 1)
    assert any(isinstance(layer, _layers.Line) for layer in fig._cell.layers)
    assert P.density(np.asarray([[0.0, 1.0], [2.0, 3.0]]), 0, 1, 0, 1)._cell.layers[0].iw == 2
    const = P.function(lambda t: np.float32(2.0), 0, 1)._cell.layers[0]
    assert set(finite_y for _, finite_y in finite(const)) == {2.0}
    assert P.density(lambda a, b: np.int64(1), 0, 1, 0, 1)._cell.colorbar is not None
    assert P.function(sd.sin, 0, 1, samples=np.int64(11), refine=np.int32(0)).info[0].n == 11
    assert P.contour(lambda a, b: a * b, 0, 1, 0, 1, levels=np.int64(3))._cell.layers
    with pytest.raises(TypeError, match=r"function\(\): 'samples' must be an int, not numpy.bool"):
        P.function(sd.sin, 0, 1, samples=np.True_)


def test_a_saddle_cell_joins_its_inside_corners_as_the_curves_do():
    def inside(bl, br, tr, tl):
        F = sd.asarray([[bl, br], [tl, tr]])
        return _field.Cells(None, [0.0, 1.0], [0.0, 1.0], F, 0.0, "t", polish=False).inside()[0]

    # the mean of the corners on the inside corners' side: one hexagon; on the other side: two triangles
    assert inside(3.0, -1.0, 3.0, -1.0) == "MLLLLLZ" and inside(-1.0, 3.0, -1.0, 3.0) == "MLLLLLZ"
    assert inside(1.0, -3.0, 1.0, -3.0) == "MLLZMLLZ" and inside(-3.0, 1.0, -3.0, 1.0) == "MLLZMLLZ"
