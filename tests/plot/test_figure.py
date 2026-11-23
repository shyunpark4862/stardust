"""Figure: sizes, layout, axes and ticks, legends, cells and the outputs (docs/api/plot.md §2)."""

import math
import re

import numpy as np
import pytest

import stardust as sd
from stardust.plot import _figure, _scene, _text

from . import figures

P = sd.plot
MM = 72.0 / 25.4
NAMES = list(figures.FIGURES)


def scene_of(fig):
    return fig._root._build("test")


def text_boxes(scene):
    """(string, x0, y0, x1, y1) of every text of a scene (its box on the page; unrotated or 90)."""
    engine = _text.engine(scene.font)
    out = []
    for item in scene.items:
        if isinstance(item, _scene.Text):
            lay = engine.layout([item.s])[0]
            w, h, d = lay.box(item.size)
            a, b, c, dd, e, f = _scene.place(item, lay).matrix
            xs = [a * px + c * py + e for px in (0.0, w) for py in (-h, d)]
            ys = [b * px + dd * py + f for px in (0.0, w) for py in (-h, d)]
            out.append((item.s, min(xs), min(ys), max(xs), max(ys)))
    return out


def axis_lines(scene):
    """The plot rectangle of a one-cell 'paper' figure from its axis lines: (x, y, w, h)."""
    for item in scene.items:
        if isinstance(item, _scene.Path) and item.ops == "MLL" and item.cap == "square":
            x0, y0, x1, y1, x2, y2 = item.xy
            return x0, y0, x2 - x0, y1 - y0
    raise AssertionError("no axis lines")


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("font", _text.FAMILIES)
def test_every_text_is_inside_the_page_and_the_output_is_the_same_every_time(name, font):
    fig = figures.FIGURES[name](font)
    sc = scene_of(fig)
    assert sc.width == pytest.approx(fig.width * MM) and sc.height == pytest.approx(fig.height * MM)
    for s, x0, y0, x1, y1 in text_boxes(sc):
        assert -0.01 <= x0 and x1 <= sc.width + 0.01 and -0.01 <= y0 and y1 <= sc.height + 0.01, s
    assert fig.to_svg() == figures.FIGURES[name](font).to_svg()
    assert fig.to_pdf() == figures.FIGURES[name](font).to_pdf()
    assert fig.to_png(dpi=72) == figures.FIGURES[name](font).to_png(dpi=72)


def test_width_is_exact_and_the_default_height_gives_a_4_to_3_plot():
    fig = figures.lines("serif")
    assert fig.width == 89.0 and repr(fig) == "Figure(89 x auto mm, 1 axes, 3 layers)"
    svg = fig.to_svg()
    w, h = re.match(r'<svg [^>]*width="([\d.]+)pt" height="([\d.]+)pt"', svg).groups()
    assert float(w) == pytest.approx(89 * MM, abs=1e-4) and float(h) == pytest.approx(fig.height * MM, abs=1e-4)
    x, y, pw, ph = axis_lines(scene_of(fig))
    assert ph == pytest.approx(0.75 * pw)
    fig.size(120, 60)
    assert (fig.width, fig.height) == (120.0, 60.0) and repr(fig) == "Figure(120 x 60 mm, 1 axes, 3 layers)"
    sc = scene_of(fig)
    assert (sc.width, sc.height) == pytest.approx((120 * MM, 60 * MM))
    box = re.search(rb"/MediaBox \[0 0 (\S+) (\S+)\]", fig.to_pdf())
    assert float(box.group(1)) == pytest.approx(120 * MM, abs=1e-4) and float(box.group(2)) == pytest.approx(
        60 * MM, abs=1e-4
    )


def test_labels_do_not_overlap_the_plot_or_each_other():
    fig = figures.errors("serif")
    sc = scene_of(fig)
    x, y, w, h = axis_lines(sc)
    boxes = {s: b for s, *b in text_boxes(sc)}
    assert boxes["Counts"][2] < x and boxes["Dose (mg)"][1] > y + h  # outside the plot
    ticks = [b for s, *b in text_boxes(sc) if re.fullmatch(r"\$[\d.]+\$", s) and b[1] > y + h]
    ticks.sort()
    for a, b in zip(ticks, ticks[1:]):
        assert a[2] < b[0]  # the x tick labels do not touch
    assert boxes[r"$\times10^{6}$"][3] <= y  # the shared power stands above the axis
    ylabels = [b for s, *b in text_boxes(sc) if b[2] < x and s != "Counts"]
    assert max(b[2] for b in ylabels) < x and min(b[0] for b in ylabels) > boxes["Counts"][2]


def test_ticks_are_inside_the_limits_and_round():
    fig = P.line([0.0, 1.0, 2.0], [0.13, 0.5, 0.97])
    m = _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))
    assert m.ya.lo < 0.13 and m.ya.hi > 0.97  # padded
    assert all(m.ya.lo <= v <= m.ya.hi for v in m.ya.ticks)
    assert m.ya.labels == ["$0.2$", "$0.4$", "$0.6$", "$0.8$", "$1.0$"] and m.xa.labels == [
        "$0.0$",
        "$0.5$",
        "$1.0$",
        "$1.5$",
        "$2.0$",
    ]
    fig.ylim(0, 1).xlim(None, 4)
    m = _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))
    assert (m.ya.lo, m.ya.hi) == (0.0, 1.0) and m.xa.hi == 4.0 and m.xa.lo < 0.0
    assert m.ya.ticks[0] == 0.0 and m.ya.ticks[-1] == 1.0


def test_constant_and_empty_data_get_a_range():
    for fig in (
        P.line([1.0, 2.0], [5.0, 5.0]),
        P.line([], []),
        P.scatter([3.0], [0.0]),
        P.line([1.0, 2.0], [math.nan, math.nan]),
    ):
        sc = scene_of(fig)
        assert sc.height > 0 and len(text_boxes(sc)) >= 4


def test_log_axes_take_decades_and_drop_what_is_not_positive():
    fig = figures.loglog("serif")
    m = _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))
    assert m.xa.labels == ["$10^{0}$", "$10^{1}$", "$10^{2}$", "$10^{3}$", "$10^{4}$"] and len(m.xa.minor) > 20
    assert all(re.fullmatch(r"\$10\^\{-?\d+\}\$", s) for s in m.ya.labels)
    fig = P.line([-1.0, 0.0, 1.0, 10.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0]).xscale("log")
    path = next(i for i in scene_of(fig).items if isinstance(i, _scene.Path) and i.join == "round")
    assert path.ops == "MLL"  # the two points that are not positive are left out
    with pytest.raises(ValueError, match=r"xlim\(\): a log axis needs positive limits"):
        P.line([1.0, 2.0], [1.0, 2.0]).xscale("log").xlim(0, 5).to_svg()


def test_pi_ticks_and_given_ticks():
    fig = figures.steps("serif")
    m = _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))
    assert m.xa.labels == ["$0$", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"]
    assert m.xa.ticks == pytest.approx([k * math.pi / 2 for k in range(5)])
    fig = P.line([0.0, 10.0], [0.0, 1.0]).xticks([0, 5, 10, 99]).yticks([0, 1], labels=["Group A", "Residuals"])
    m = _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))
    assert m.xa.ticks == [0.0, 5.0, 10.0] and m.xa.labels == ["$0$", "$5$", "$10$"]  # 99 is outside the range
    assert m.ya.labels == ["Group A", "Residuals"]


def test_settings_return_the_figure_and_check_their_arguments():
    fig = P.Figure()
    assert (
        fig.title("Group A").xlabel("$x$").ylabel("$x$").xlim(0, 1).ylim(None, 2).xscale("log").yscale("linear") is fig
    )
    assert fig.xticks([1, 2]).yticks("pi").legend(loc=None).size(100) is fig
    cases = [
        (lambda: fig.title(3), TypeError, "title(): 'text' must be a str, not int"),
        (lambda: fig.xlim(2, 1), ValueError, "xlim(): 'low' must be less than 'high'"),
        (lambda: fig.ylim("a", 1), TypeError, "ylim(): 'low' must be an int or a float, not str"),
        (lambda: fig.xlim(0, math.inf), ValueError, "xlim(): 'high' must be finite"),
        (lambda: fig.xscale("symlog"), ValueError, "xscale(): 'kind' must be 'linear' or 'log', not 'symlog'"),
        (lambda: fig.xticks("tau"), ValueError, "xticks(): 'values' must be numbers or 'pi', not 'tau'"),
        (
            lambda: fig.xticks([1, 2], labels=["a"]),
            ValueError,
            "xticks(): 'values' and 'labels' must have the same length: 2 and 1",
        ),
        (lambda: fig.yticks(3), TypeError, "yticks(): 'values' must be a sequence of numbers or 'pi', not int"),
        (
            lambda: fig.legend(loc="middle"),
            ValueError,
            "legend(): 'loc' must be 'best', 'top', 'right', 'bottom', 'upper right', "
            "'upper left', 'lower left' or 'lower right', not 'middle'",
        ),
        (lambda: fig.legend(ncols=0), ValueError, "legend(): 'ncols' must be at least 1"),
        (lambda: fig.size(-1), ValueError, "size(): 'width' must be positive"),
        (lambda: P.Figure(font="mono"), ValueError, "Figure(): 'font' must be 'serif' or 'sans', not 'mono'"),
        (lambda: P.Figure(theme=None), TypeError, "Figure(): 'theme' must be 'paper', 'boxed' or 'grid', not NoneType"),
        (
            lambda: P.Figure(palette="rainbow"),
            ValueError,
            "Figure(): 'palette' must be 'jco', 'tableau', 'deep', 'bright', 'okabe-ito', 'npg', 'aaas', 'lancet' or "
            "'nejm', not 'rainbow'",
        ),
        (lambda: P.Figure(nrows=0), ValueError, "Figure(): 'nrows' must be at least 1"),
        (lambda: P.Figure(width=True), TypeError, "Figure(): 'width' must be an int or a float, not bool"),
    ]
    for call, kind, message in cases:
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_cells_are_views_of_one_figure():
    fig = P.Figure(nrows=2, ncols=3, width=183)
    assert fig.shape == (2, 3) and len(fig.axes) == 6
    assert fig[1, 2].line([0.0, 1.0], [0.0, 1.0]).title("Group A")._root is fig
    assert len(fig._cells[5].layers) == 1 and fig._cells[5].title == "Group A" and not fig._cells[0].layers
    fig[-2, 0].scatter([1.0], [1.0])
    assert len(fig._cells[0].layers) == 1
    fig.line([0.0, 1.0], [1.0, 0.0])  # without a cell: the first
    assert len(fig._cells[0].layers) == 2
    for key, message in (
        ((2, 0), "index 2 is out of bounds for a figure of 2 x 3 cells"),
        (1, "a figure of 2 x 3 cells takes two indices"),
        ((0, 1.0), "figure indices must be ints, not float"),
        ((0, 0, 0), "a figure takes one or two indices, not 3"),
    ):
        with pytest.raises(IndexError) as e:
            fig[key]
        assert str(e.value) == message
    row = P.Figure(ncols=3)
    assert row[2]._index == 2 and row[0, 1]._index == 1
    with pytest.raises(IndexError, match="a cell of a figure has no cells of its own"):
        row[1][0]
    assert scene_of(fig).width == pytest.approx(183 * MM)


def test_grid_gathers_figures_and_aligns_the_columns():
    fig = figures.panels("serif")
    assert fig.shape == (2, 2) and fig.width == 183.0
    sc = scene_of(fig)
    plots = [
        (i.xy[0], i.xy[1], i.xy[4], i.xy[3])
        for i in sc.items
        if isinstance(i, _scene.Path) and i.ops == "MLL" and i.cap == "square"
    ]
    assert len(plots) == 4
    (ax, ay, ax1, ay1), (bx, by, bx1, by1), (cx, cy, cx1, cy1), (dx, dy, dx1, dy1) = plots
    assert ax == cx and bx == dx and ax1 == cx1  # a column's plots share their left and right edges
    assert ay == by and ay1 == by1 and cy == dy  # a row's their top and bottom
    assert ay1 - ay == pytest.approx(0.75 * (ax1 - ax))
    panel = [b for s, *b in text_boxes(sc) if s.startswith(r"\textbf{")]
    assert len(panel) == 4 and panel[0][0] < ax and panel[1][0] > ax1
    assert P.grid([figures.lines("serif")] * 5).shape == (2, 3)
    assert P.grid([figures.lines("serif")] * 3, nrows=3).shape == (3, 1)
    for call, kind, message in (
        (lambda: P.grid([]), TypeError, "grid(): 'figures' must be a non-empty sequence of Figures, not list"),
        (lambda: P.grid([fig, fig], nrows=1, ncols=2), ValueError, "grid(): 1 x 2 cells cannot hold 8 figures"),
        (
            lambda: P.grid([figures.lines("serif")], labels="B"),
            ValueError,
            "grid(): 'labels' must be 'A', 'a' or 1 strings",
        ),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_legend_places_and_can_be_removed():
    fig = figures.groups("serif")
    sc = scene_of(fig)
    x, y, w, h = axis_lines(sc)
    boxes = {s: b for s, *b in text_boxes(sc)}
    assert set("ABC") <= set(boxes) and "Group" in boxes
    assert boxes["A"][1] < boxes["B"][1] < boxes["C"][1] and boxes["Group"][0] < x + w / 3  # upper left, in order
    assert "A" not in {s for s, *_ in text_boxes(scene_of(fig.legend(loc=None)))}
    top = scene_of(figures.bars("serif"))
    px, py, pw, ph = axis_lines(top)
    b = {s: b for s, *b in text_boxes(top)}
    assert b["2024"][3] < py and b["2025"][0] > b["2024"][2]  # above the plot, side by side
    best = figures.lines("serif")  # the curve leaves the corners free: the first candidate
    bb = {s: b for s, *b in text_boxes(scene_of(best))}
    assert bb[r"$e^{-t^2/2}$"][0] > axis_lines(scene_of(best))[0] + 0.5 * axis_lines(scene_of(best))[2]


def test_themes():
    base = len(scene_of(P.line([0.0, 1.0], [0.0, 1.0])).items)
    boxed = scene_of(P.line([0.0, 1.0], [0.0, 1.0], theme="boxed"))
    assert any(isinstance(i, _scene.Path) and i.ops == "MLLLZ" and i.stroke is not None for i in boxed.items)
    grid = scene_of(P.line([0.0, 1.0], [0.0, 1.0], theme="grid"))
    assert (
        len(grid.items) > base
        and sum(1 for i in grid.items if isinstance(i, _scene.Path) and i.stroke == _figure.GRID_COLOR) >= 6
    )


def test_save_writes_by_extension(tmp_path):
    fig = figures.bars("sans")
    assert fig.save(tmp_path / "a.pdf") is fig
    fig.save(str(tmp_path / "a.svg")).save(tmp_path / "a.PNG", dpi=80)
    assert (tmp_path / "a.pdf").read_bytes() == fig.to_pdf() and (tmp_path / "a.svg").read_text() == fig.to_svg()
    assert (tmp_path / "a.PNG").read_bytes() == fig.to_png(dpi=80)
    flat = fig._repr_mimebundle_()
    assert set(flat) == {"image/svg+xml", "text/plain"} and flat["image/svg+xml"] == fig.to_svg()
    hill = sd.plot.surface(lambda x, y: x * y, -1, 1, -1, 1)._repr_mimebundle_()
    assert set(hill) == {"image/png", "text/plain"} and hill["image/png"][:8] == b"\x89PNG\r\n\x1a\n"
    assert "image/png" in sd.plot.density(lambda x, y: x * y, 0, 1, 0, 1)._repr_mimebundle_()
    for call, kind, message in (
        (
            lambda: fig.save(tmp_path / "a.jpg"),
            ValueError,
            "save(): 'path' must end in '.pdf', '.svg' or '.png', not '.jpg'",
        ),
        (lambda: fig.save(3), TypeError, "save(): 'path' must be a str or a path, not int"),
        (lambda: fig.save(tmp_path / "a.png", dpi=0), ValueError, "save(): 'dpi' must be positive"),
        (lambda: fig.to_png(dpi="x"), TypeError, "to_png(): 'dpi' must be an int or a float, not str"),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message
    d = fig.scene()
    assert d["font"] == "sans" and {i["type"] for i in d["items"]} == {"path", "text"}


def test_a_change_drops_the_scene_and_a_typesetting_failure_names_the_method():
    fig = figures.lines("serif")
    a = fig.to_svg()
    assert fig._scene is not None
    fig.xlabel("Residuals")
    assert fig._scene is None and fig.to_svg() != a
    bad = P.line([0.0, 1.0], [0.0, 1.0]).xlabel("not recorded")  # the tests' cache is frozen
    with pytest.raises(_text.FrozenMiss):
        bad.to_svg()
    with pytest.raises(ValueError) as e:
        P.line([0.0, 1.0], [0.0, 1.0]).xlabel("50% of").save("/nonexistent/x.pdf")
    assert str(e.value) == r"save(): LaTeX could not typeset '50% of': an unescaped % (write \%)"


def test_inputs_numpy_and_frames_agree_with_lists():
    xs, ys = [0.0, 1.0, 2.0, 3.0], [1.0, 3.0, 2.0, 5.0]
    a = P.line(xs, ys).to_svg()
    assert P.line(np.array(xs), np.array(ys)).to_svg() == a
    assert P.line(sd.asarray(xs), tuple(ys)).to_svg() == a
    df = sd.frame.DataFrame({"t": xs, "v": ys, "k": [1, 3, 2, 5]})
    assert P.line("t", "v", data=df).to_svg() == a and P.line(df["t"], "k", data=df).to_svg() == a


def test_a_legend_goes_where_the_data_leave_room_or_outside():
    x = sd.linspace(0.0, 10.0, 200)
    free = P.line(x, sd.exp(-x), label="Group A")  # the upper right is empty
    sc = scene_of(free)
    assert free._cell.placed == "upper right"
    px, py, pw, ph = axis_lines(sc)
    box = next(b for s, *b in text_boxes(sc) if s == "Group A")
    assert px + pw / 2 < box[0] and box[2] < px + pw and py < box[1] < py + ph / 3
    full = P.Figure()
    for k in range(4):
        full.line(x, sd.sin(x * (1 + 0.3 * k) + k), label="Group A" if k == 0 else "Residuals" if k == 1 else None)
    sc = scene_of(full)
    assert full._cell.placed == "top"  # curves in every corner: above the plot, in one row
    px, py, pw, ph = axis_lines(sc)
    boxes = {s: b for s, *b in text_boxes(sc)}
    assert boxes["Group A"][3] < py and boxes["Residuals"][3] < py and boxes["Group A"][2] < boxes["Residuals"][0]
    assert scene_of(P.line(x, sd.exp(-x), label="Group A").legend(loc="lower left")) is not None
    held = P.line(x, sd.exp(-x), label="Group A").legend(loc="lower left")
    scene_of(held)
    assert held._cell.placed is None  # a place the user gave is kept
    ink = _figure.Ink([_scene.Path(*_scene.rect(10, 10, 20, 20), fill=(0, 0, 0, 1))], 0.0, 0.0, 100.0, 60.0)
    assert ink.share(10, 10, 20, 20) == 1.0 and ink.share(40, 10, 20, 20) == 0.0 and ink.share(20, 10, 20, 20) == 0.5
    assert ink.share(90, 50, 20, 20) == 1.0  # a rectangle that leaves the plot counts as taken


def test_x_labels_that_would_touch_are_thinned_or_turned():
    wide = P.line([1000.0, 1800.0], [0.0, 1.0], width=50)  # four-digit numbers
    th, e = _figure.THEMES["paper"], _text.engine("serif")
    roomy = _figure._measure(wide, wide._cell, 80.0, 60.0, th, e)
    m = _figure._measure(wide, wide._cell, 50.0, 60.0, th, e)  # a narrow plot: fewer ticks, none turned
    assert len(roomy.xa.ticks) == 4 and len(m.xa.ticks) == 2 and m.xrot == 0.0
    cats = P.bar(["Fitted values", "Residuals", "Theoretical quantiles", "Sample quantiles"], [1.0, 2.0, 3.0, 4.0])
    m = _figure._measure(cats, cats._cell, 150.0, 110.0, _figure.THEMES["paper"], _text.engine("serif"))
    flat = _figure._measure(
        P.bar(["x", "y"], [1.0, 2.0]),
        P.bar(["x", "y"], [1.0, 2.0])._cell,
        150.0,
        110.0,
        _figure.THEMES["paper"],
        _text.engine("serif"),
    )
    assert m.xrot == 40.0 and flat.xrot == 0.0 and m.bottom > flat.bottom + 20  # turned labels need room below
    narrow = scene_of(
        P.Figure(width=60).bar(
            ["Fitted values", "Residuals", "Theoretical quantiles", "Sample quantiles"], [1.0, 2.0, 3.0, 4.0]
        )
    )
    turned = [i for i in narrow.items if isinstance(i, _scene.Text) and i.rotation == 40.0]
    assert len(turned) == 4 and all(t.ha == "right" for t in turned)
    for s, x0, _y0, _x1, y1 in text_boxes(narrow):
        assert x0 >= -0.5 and y1 <= narrow.height + 0.5, s


def test_a_best_legend_is_drawn_in_the_corner_found():
    x = sd.linspace(0.0, 1.0, 50)
    fig = P.line(x, x, label="Group A")  # the rising line leaves the upper left free
    sc = scene_of(fig)
    assert fig._cell.placed == "upper left"
    px, py, pw, ph = axis_lines(sc)
    box = next(b for s, *b in text_boxes(sc) if s == "Group A")
    assert box[2] < px + pw / 2 and box[3] < py + ph / 2


def test_grid_copies_its_cells():
    a, b = P.line([0.0, 1.0], [0.0, 1.0]), P.line([0.0, 1.0], [1.0, 0.0])
    g = P.grid([a, b])
    g[0].title("Group A")
    assert a._cell.title is None  # the figure given stays as it was
    a.xlabel("Residuals")
    assert g[0]._cell.xlabel is None


def test_points_outside_the_range_and_log_labels():
    from stardust.plot import _ticks

    f = P.scatter([1.85, 3.0, 4.0, 5.1], [3.0, 1.9, 4.0, 3.0]).xlim(2, 5).ylim(2, 5)
    marks = [i for i in scene_of(f).items if isinstance(i, _scene.Markers)]
    assert sum(len(m.at) // 2 for m in marks) == 1  # only (4, 4) is inside
    assert _ticks.log_labels([1.0, 2.0, 4.0]) == ["$1$", "$2$", "$4$"]  # no mix of the two forms
    assert _ticks.log_labels([1.0, 10.0]) == ["$10^{0}$", "$10^{1}$"]


def test_the_notebook_gets_one_form():
    flat = P.line([0.0, 1.0], [0.0, 1.0])
    assert set(flat._repr_mimebundle_()) == {"image/svg+xml", "text/plain"}
    assert not hasattr(flat, "_repr_svg_") and not hasattr(flat, "_repr_png_")


def test_shared_axes_facets_twins_and_insets():
    x = sd.linspace(0.0, 10.0, 20)
    figs = [P.line(x, sd.sin(x) * k) for k in (1.0, 3.0)]
    g = P.grid(figs, ncols=2, sharey=True)
    scene_of(g)
    a, b = g._root._cells
    assert a.span[1] == b.span[1] and a.span[1][1] > 3.0 and b.inner[1] and not a.inner[1]
    with pytest.raises(ValueError, match=r"grid\(\): 'sharex' must be False, True or 'all'"):
        P.grid(figs, sharex="cols")
    df = sd.frame.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],
            "b": [2.0, 1.0, 4.0, 3.0],
            "g": ["Group A", "Residuals"] * 2,
            "s": ["u", "u", "v", "v"],
        }
    )
    f = P.facet(df, "s").scatter("a", "b", group="g")
    left, right = f.figure._root._cells
    assert [lay.color for lay in left.layers] == [lay.color for lay in right.layers]  # a group keeps its colour
    assert left.legend["loc"] is None and right.legend["loc"] == "best" and left.title == "u"
    assert f.xlabel("Fitted values") is f and left.xlabel == right.xlabel == "Fitted values"
    with pytest.raises(TypeError, match=r"scatter\(\): a facet's data is its own rows"):
        f.scatter("a", "b", data=df)
    twin = P.bar([1.0, 2.0], [3.0, 4.0])
    second = twin.twin().line([1.0, 2.0], [100.0, 300.0])
    cell = twin._cell
    assert cell.twin is second._cell and len(cell.twin.layers) == 1
    sc = scene_of(twin)
    px, py, pw, ph = axis_lines(sc)
    right_ticks = [i for i in sc.items if isinstance(i, _scene.Text) and i.ha == "left" and i.x > px + pw]
    assert len(right_ticks) >= 2  # the second axis's numbers, at the right
    with pytest.raises(ValueError, match=r"twin\(\): this is already the second y axis"):
        second.twin()
    host = P.line(x, sd.sin(x)).inset(P.line(x, sd.cos(x)), (0.6, 0.6, 0.35, 0.35))
    assert len(host._cell.insets) == 1 and scene_of(host) is not None
    with pytest.raises(ValueError, match=r"inset\(\): 'rect' must lie in the plot"):
        P.line(x, x).inset(P.line(x, x), (0.8, 0.8, 0.5, 0.5))


def _facet_frame():
    n = 40
    return sd.frame.DataFrame(
        {
            "site": ["u"] * 10 + ["v"] * 10 + ["w"] * 10 + ["x"] * 10,
            "sex": ["Group A", "Residuals"] * (n // 2),
            "length": [20.0 + (7 * i) % 23 for i in range(n)],
            "weight": [40.0 + (11 * i) % 31 for i in range(n)],
        }
    )


def test_facet_methods_without_data_draw_in_every_cell():
    g = P.facet(_facet_frame(), "site").scatter("length", "weight").hline(50.0).function(lambda t: 2 * t, 20, 40)
    assert len(g.axes) == 4  # the cells of the values; the grid's two empty places are not among them
    assert all(len(cell._cell.layers) == 3 for cell in g.axes)
    assert repr(g) == "Facets('site': 4 cells, 12 layers; 183 x auto mm)"
    with pytest.raises(AttributeError, match="Facets has no method 'twin'"):
        g.twin()
    with pytest.raises(AttributeError, match="Facets has no method 'inset'"):
        g.inset(P.line([0.0, 1.0], [0.0, 1.0]), (0.1, 0.1, 0.3, 0.3))


def _tick_labels(scene, y):
    """The strings of the texts at height y of a scene, left to right."""
    items = [i for i in scene.items if isinstance(i, _scene.Text) and abs(i.y - y) < 1e-6]
    return [i.s for i in sorted(items, key=lambda i: i.x)]


def test_shared_axes_choose_their_ticks_once_and_keep_the_numbers_they_do_not_share():
    for width in (None, 120.0):
        g = P.facet(_facet_frame(), "site", ncols=2, width=width).scatter("length", "weight", group="sex")
        sc = scene_of(g.figure)
        bottom = max(i.y for i in sc.items if isinstance(i, _scene.Text) and i.s.startswith("$"))  # the x numbers
        labels = _tick_labels(sc, bottom)
        assert len(labels) % 2 == 0 and labels[: len(labels) // 2] == labels[len(labels) // 2 :]  # one choice
    # no empty texts for the numbers left to the outer cells
    t = sd.linspace(0.0, 10.0, 60)
    shared = P.grid([P.line(t, sd.sin(t)), P.line(t + 5.0, sd.cos(t))], ncols=1, sharex=True)
    assert all(i.s for i in scene_of(shared).items if isinstance(i, _scene.Text))
    # a cell above another of other categories, or of a range of its own, keeps its numbers
    up = P.line([0.0, 10.0], [0.0, 1.0])
    cats = P.grid([P.bar(["a", "b"], [1.0, 2.0]), up], ncols=1, sharex=True)
    scene_of(cats)
    assert cats._root._cells[0].inner == [False, False]
    ranges = P.grid([up, P.line([0.0, 10.0], [0.0, 1.0]).xlim(0.0, 1.0)], ncols=1, sharex=True)
    scene_of(ranges)
    assert ranges._root._cells[0].inner == [False, False]
    same = P.grid([up, P.line([2.0, 12.0], [1.0, 0.0])], ncols=1, sharex=True)
    scene_of(same)
    assert same._root._cells[0].inner == [True, False]


def test_a_twin_view_sets_the_cell_and_gives_it_whole():
    f = P.line([0.0, 1.0], [0.0, 1.0])
    t = f.twin().line([0.0, 1.0], [3.0, 5.0]).title("A").xlabel("x").xlim(0.0, 5.0).legend(loc="top")
    t.ylabel("y").inset(P.line([0.0, 1.0], [1.0, 0.0]), (0.1, 0.1, 0.4, 0.4))
    cell = f._cell
    assert (cell.title, cell.xlabel, cell.xlim, cell.legend["loc"]) == ("A", "x", [0.0, 5.0], "top")
    assert cell.twin.ylabel == "y" and cell.ylabel is None
    assert len(cell.insets) == 1 and not cell.twin.insets
    texts = [i.s for i in scene_of(f).items if isinstance(i, _scene.Text)]
    assert "A" in texts and "x" in texts
    g = P.grid([t])
    (copy,) = g._root._cells
    assert len(copy.layers) == 1 and copy.twin is not None and len(copy.twin.layers) == 1
    with pytest.raises(ValueError, match=r"inset\(\): 'rect' must hold 4 numbers \(x, y, w, h\), not 3"):
        f.inset(P.line([0.0, 1.0], [0.0, 1.0]), (0.5, 0.5, 0.4))


def test_the_titles_of_a_row_stand_on_one_baseline():
    fig = P.Figure(width=150, height=45, ncols=3)
    for i, w in enumerate(["y", "a", "mid"]):  # with a descender and without
        fig[i].line([0.0, 1.0], [0.0, 1.0]).title(w)
    items = [i for i in scene_of(fig).items if isinstance(i, _scene.Text) and i.s in ("y", "a", "mid")]
    assert len(items) == 3 and {i.va for i in items} == {"baseline"} and len({i.y for i in items}) == 1
    # the title clears the plot by its depth: a descender does not touch the frame
    engine = _text.engine("serif")
    (p,) = [i for i in items if i.s == "y"]
    depth = engine.layout(["y"])[0].box(p.size)[2]
    assert depth > 0.0
    one = scene_of(P.line([0.0, 1.0], [0.0, 1.0]).title("y"))
    _, top, _, _ = axis_lines(one)
    (q,) = [i for i in one.items if isinstance(i, _scene.Text) and i.s == "y"]
    assert q.y + depth < top
