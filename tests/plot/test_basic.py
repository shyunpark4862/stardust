"""The basic plots (docs/api/plot.md §3): what each adds to the scene, groups and data frames,
missing values, and the errors."""

import math

import pytest

import stardust as sd
from stardust.plot import _figure, _layers, _scene, _text

P = sd.plot


def items(fig, kind=None):
    out = fig._root._build("test").items
    return out if kind is None else [i for i in out if isinstance(i, kind)]


def data_paths(fig):
    """The clipped paths: what the layers drew inside the plot."""
    return [i for i in items(fig, _scene.Path) if i.clip is not None]


def measure(fig):
    return _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))


def test_line_breaks_at_values_that_are_not_finite():
    fig = P.line([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, math.nan, 3.0, math.inf, 5.0])
    (path,) = data_paths(fig)
    assert path.ops == "MLMM" and len(path.xy) == 8  # runs of 2, 1 and 1 points
    assert path.stroke == (0.0, 0x73 / 255, 0xC2 / 255, 1.0) and path.width == 1.2 and path.join == "round"
    m = measure(fig)
    assert m.ya.lo < 1.0 and 5.0 < m.ya.hi < 6.0  # the infinity is no part of the range


def test_colours_cycle_through_the_palette_and_styles_dash():
    fig = (
        P.line([0.0, 1.0], [0.0, 1.0])
        .line([0.0, 1.0], [1.0, 0.0], style="--", width=2)
        .line([0.0, 1.0], [0.5, 0.5], color="#ff000080", style=":")
    )
    a, b, c = data_paths(fig)
    assert a.stroke[:3] == (0.0, 0x73 / 255, 0xC2 / 255) and b.stroke[:3] == (0xEF / 255, 0xC0 / 255, 0.0)
    assert (b.dash, b.width) == ((4.0, 2.0), 2.0) and c.stroke == (1.0, 0.0, 0.0, 128 / 255) and c.dash == (1.0, 1.6)
    npg = data_paths(P.line([0.0, 1.0], [0.0, 1.0], palette="npg", alpha=0.5))[0]
    assert npg.stroke == (0xE6 / 255, 0x4B / 255, 0x35 / 255, 0.5)
    assert data_paths(P.line([0.0, 1.0], [0.0, 1.0], color=(0.1, 0.2, 0.3)))[0].stroke == (0.1, 0.2, 0.3, 1.0)
    assert data_paths(P.Figure().line([0.0, 1.0], [0.0, 1.0], color="C3"))[0].stroke[:3] == (
        0xCD / 255,
        0x53 / 255,
        0x4C / 255,
    )


def test_scatter_stamps_one_shape_at_the_finite_points():
    fig = P.scatter([0.0, 1.0, math.nan, 3.0], [0.0, 1.0, 2.0, 3.0], shape="s", size=5)
    (m,) = items(fig, _scene.Markers)
    assert len(m.at) == 6 and m.ops == "MLLLZ" and m.fill is not None and m.clip is None
    assert max(m.xy) == pytest.approx(0.44 * 5) and (m.stroke, m.width) == ((1.0, 1.0, 1.0, 1.0), 0.5)  # a white rim
    many = P.scatter(list(range(4000)), list(range(4000)))
    assert items(many, _scene.Markers)[0].stroke is None  # a cloud of thousands goes without
    (plus,) = items(P.scatter([1.0], [1.0], shape="+"), _scene.Markers)
    assert plus.fill is None and plus.stroke is not None  # an open shape is stroked
    assert [i.ops for i in items(P.scatter([1.0], [1.0]), _scene.Markers)] == ["MCCCCZ"]


def test_group_splits_in_order_of_first_appearance():
    g = ["b", "a", "b", "a", None, "c"]
    fig = P.scatter([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], group=g)
    layers = fig._cell.layers
    assert [la.label for la in layers] == ["b", "a", "c"] and [la.x.tolist() for la in layers] == [
        [0.0, 2.0],
        [1.0, 3.0],
        [5.0],
    ]
    assert layers[0].color != layers[1].color != layers[2].color
    df = sd.frame.DataFrame({"x": [0.0, 1.0, 2.0], "y": [1.0, 2.0, 3.0], "who": ["x_1", "x_1", "50%"]})
    fig = P.line("x", "y", data=df, group="who")
    assert [la.label for la in fig._cell.layers] == [r"x\_1", r"50\%"]  # labels from data are escaped for LaTeX
    assert [la.label for la in P.line([1.0, 2.0], [1.0, 2.0], group=[1, 2])._cell.layers] == ["1", "2"]


def test_bars_on_a_category_axis_stand_on_zero():
    fig = P.bar(["Group A", "Residuals"], [2.0, 5.0])
    m = measure(fig)
    assert m.xa.labels == ["Group A", "Residuals"] and m.xa.ticks == [0.0, 1.0] and (m.xa.lo, m.xa.hi) == (-0.6, 1.6)
    assert m.ya.lo == 0.0 and m.ya.hi > 5.0  # no padding below the base
    (path,) = data_paths(fig)
    # a filled shape: the colour's tint with the colour as a thin edge
    assert path.ops == "MLLLZ" * 2 and path.stroke == (0.0, 0x73 / 255, 0xC2 / 255, 1.0) and path.width == 0.5
    assert path.fill == pytest.approx((0.42, 0.42 + 0.58 * 0x73 / 255, 0.42 + 0.58 * 0xC2 / 255, 1.0))
    x0, x1 = path.xy[0], path.xy[2]
    assert (x1 - x0) == pytest.approx(0.8 * (path.xy[8] - x0) / 1.0)  # 0.8 of the spacing
    neg = measure(P.bar(["Group A", "Residuals"], [-2.0, 5.0]))
    assert neg.ya.lo < -2.0 and neg.ya.hi > 5.0


def test_grouped_bars_stand_side_by_side_and_horizontal_bars_swap_the_axes():
    fig = P.bar(["Group A", "Residuals"] * 2, [1.0, 2.0, 3.0, 4.0], group=["x", "x", "y", "y"])
    a, b = fig._cell.layers
    assert (
        a.width == b.width == pytest.approx(0.4 * 0.9)
        and a.pos.tolist() == [-0.2, 0.8]
        and b.pos.tolist() == [0.2, 1.2]
    )
    h = P.bar(["Group A", "Residuals"], [1.0, 2.0], horizontal=True)
    m = measure(h)
    assert m.ya.labels == ["Group A", "Residuals"] and m.xa.lo == 0.0
    num = P.bar([1.0, 2.0, 4.0], [1.0, 2.0, 3.0], width=0.5)
    assert measure(num).xa.cats is None and num._cell.layers[0].width == 0.5
    with pytest.raises(ValueError, match=r"bar\(\): 'x' must be labels: the axis already has categories"):
        fig.bar([0.0], [1.0])


def test_area_step_errorbar_rules():
    fig = P.area([0.0, 1.0, 2.0, 3.0], [1.0, 2.0, math.nan, 4.0], [0.0, 0.5, 1.0, 1.5])
    (path,) = data_paths(fig)
    assert path.ops == "MLLLZ" and path.fill[3] == pytest.approx(
        0.3
    )  # one run of two points; the lone point is nothing
    assert measure(P.area([0.0, 1.0], [1.0, 2.0])).ya.lo < 0.0  # y2 defaults to 0 and is padded like data

    post = P.step([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])._cell.layers[0]
    assert post.x.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0] and post.y.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]
    pre = P.step([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], where="pre")._cell.layers[0]
    assert pre.x.tolist() == [0.0, 0.0, 1.0, 1.0, 2.0] and pre.y.tolist() == [1.0, 2.0, 2.0, 3.0, 3.0]
    mid = P.step([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], where="mid")._cell.layers[0]
    assert mid.x.tolist() == [0.0, 0.5, 0.5, 1.5, 1.5, 2.0] and mid.y.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    e = P.errorbar([1.0, 2.0], [10.0, 20.0], yerr=[1.0, 2.0], xerr=0.5)._cell.layers[0]
    assert (e.ylo.tolist(), e.yhi.tolist(), e.xlo.tolist(), e.xhi.tolist()) == (
        [9.0, 18.0],
        [11.0, 22.0],
        [0.5, 1.5],
        [1.5, 2.5],
    )
    two = P.errorbar([1.0, 2.0], [10.0, 20.0], yerr=[[1.0, 2.0], [3.0, 4.0]], cap=0)
    assert (two._cell.layers[0].ylo.tolist(), two._cell.layers[0].yhi.tolist()) == ([9.0, 18.0], [13.0, 24.0])
    bars = data_paths(two)[0]
    assert bars.ops == "MLML" and len(items(two, _scene.Markers)) == 1  # no caps; the points on top
    m = measure(two)
    assert m.ya.lo < 9.0 and m.ya.hi > 24.0  # the bars count for the range

    fig = P.line([0.0, 1.0], [0.0, 1.0]).hline(0.5, style="--").vline(3.0, color="red", label="Group A")
    h, v = data_paths(fig)[1:]
    x, y, w, hh = fig._root._build("t").items[1].clip
    assert (h.xy[0], h.xy[2]) == (x, x + w) and h.dash == (4.0, 2.0) and v.stroke == (1.0, 0.0, 0.0, 1.0)
    assert measure(fig).xa.hi > 3.0 and v.xy[1] == y  # a rule widens its own axis only


def test_text_and_annotate():
    fig = P.line([0.0, 10.0], [0.0, 10.0]).text(2, 8, "Group A", ha="left", rotation=45, size=6, color="blue")
    t = [i for i in items(fig, _scene.Text) if i.s == "Group A"][0]
    assert (t.ha, t.va, t.rotation, t.size, t.color) == ("left", "center", 45.0, 6.0, (0.0, 0.0, 1.0, 1.0))
    fig = P.line([0.0, 10.0], [0.0, 10.0]).annotate("Residuals", (8, 2), (3, 7))
    paths = items(fig, _scene.Path)
    k = next(i for i, p in enumerate(paths) if p.ops == "MLLZ" and p.fill is not None and p.clip is None)
    head, shaft = paths[k], paths[k - 1]
    assert shaft.ops == "ML" and shaft.stroke is not None  # the shaft, then the head
    tip = (head.xy[0], head.xy[1])
    target = [i for i in items(fig, _scene.Text) if i.s == "Residuals"][0]
    assert math.hypot(tip[0] - target.x, tip[1] - target.y) > 20  # the head is at the point, not at the text
    plain = P.line([0.0, 10.0], [0.0, 10.0]).annotate("Residuals", (8, 2), (3, 7), arrow=False)
    assert len(items(plain, _scene.Path)) == len(items(fig, _scene.Path)) - 2


def test_argument_errors():
    df = sd.frame.DataFrame({"x": [1.0, 2.0], "s": ["a", "b"]})
    cases = [
        (lambda: P.line([1.0, 2.0], [1.0]), ValueError, "line(): 'x' and 'y' must have the same length: 2 and 1"),
        (lambda: P.line("x", [1.0, 2.0]), TypeError, "line(): 'x' is a column name ('x'), which needs 'data'"),
        (lambda: P.line("zz", "x", data=df), ValueError, "line(): no column 'zz'"),
        (lambda: P.line("x", "s", data=df), TypeError, "line(): 'y' must be numeric, not a Series of string"),
        (lambda: P.line([1.0], [1.0], data=[1]), TypeError, "line(): 'data' must be a DataFrame, not list"),
        (lambda: P.line([[1.0, 2.0]], [1.0]), ValueError, "line(): 'x' must be 1-dimensional, not shape (1, 2)"),
        (
            lambda: P.line([1.0], [1.0], style="dashed"),
            ValueError,
            "line(): 'style' must be '-', '--', ':' or '-.', not 'dashed'",
        ),
        (lambda: P.line([1.0], [1.0], width=0), ValueError, "line(): 'width' must be positive"),
        (lambda: P.line([1.0], [1.0], alpha=1.5), ValueError, "line(): 'alpha' must be in [0, 1]"),
        (lambda: P.line([1.0], [1.0], label=3), TypeError, "line(): 'label' must be a str, not int"),
        (
            lambda: P.line([1.0], [1.0], color="reddish"),
            ValueError,
            "line(): 'color' must be a colour name, '#rrggbb', '#rrggbbaa', 'C0' to 'C9' or 3 or 4 numbers in [0, 1], "
            "not 'reddish'",
        ),
        (
            lambda: P.line([1.0], [1.0], color=(2.0, 0.0, 0.0)),
            ValueError,
            "line(): 'color' must be a colour name, '#rrggbb', '#rrggbbaa', 'C0' to 'C9' or 3 or 4 numbers in [0, 1]; "
            "got a number outside [0, 1]",
        ),
        (
            lambda: P.line([1.0], [1.0], color=5),
            TypeError,
            "line(): 'color' must be a colour name, '#rrggbb', '#rrggbbaa', 'C0' to 'C9' or 3 or 4 numbers in [0, 1], "
            "not int",
        ),
        (
            lambda: P.line([1.0, 2.0], [1.0, 2.0], group=["a"]),
            ValueError,
            "line(): 'group' must have the data's length: 2, not 1",
        ),
        (
            lambda: P.line([1.0], [1.0], group=["a"], color="red"),
            ValueError,
            "line(): 'color' and 'group' cannot both be given (a group takes the palette's colours)",
        ),
        (
            lambda: P.line([1.0], [1.0], group=["a"], label="x"),
            ValueError,
            "line(): 'label' and 'group' cannot both be given (a group is named by its value)",
        ),
        (
            lambda: P.line([1.0], [1.0], group=3),
            TypeError,
            "line(): 'group' must be a column name or a sequence of labels, not int",
        ),
        (
            lambda: P.scatter([1.0], [1.0], shape="*"),
            ValueError,
            "scatter(): 'shape' must be 'o', 's', '^', 'v', 'D', '+' or 'x', not '*'",
        ),
        (lambda: P.bar(["a"], [1.0], horizontal=1), TypeError, "bar(): 'horizontal' must be True or False, not int"),
        (
            lambda: P.step([1.0], [1.0], where="left"),
            ValueError,
            "step(): 'where' must be 'post', 'pre' or 'mid', not 'left'",
        ),
        (
            lambda: P.errorbar([1.0, 2.0], [1.0, 2.0], yerr=[1.0, 2.0, 3.0]),
            ValueError,
            "errorbar(): 'yerr' must be a number, 2 numbers or shape (2, 2), not 3 numbers",
        ),
        (
            lambda: P.errorbar([1.0, 2.0], [1.0, 2.0], yerr=[[1.0], [2.0]]),
            ValueError,
            "errorbar(): 'yerr' must be a number, 2 numbers or shape (2, 2), not shape (2, 1)",
        ),
        (lambda: P.errorbar([1.0], [1.0], cap=-1), ValueError, "errorbar(): 'cap' must not be negative"),
        (lambda: P.hline("a"), TypeError, "hline(): 'y' must be an int or a float, not str"),
        (lambda: P.text(1, 2, 3), TypeError, "text(): 's' must be a str, not int"),
        (
            lambda: P.text(1, 2, "a", ha="middle"),
            ValueError,
            "text(): 'ha' must be 'left', 'center' or 'right', not 'middle'",
        ),
        (
            lambda: P.annotate("a", (1, 2, 3), (1, 2)),
            TypeError,
            "annotate(): 'xy' must be a pair of numbers (x, y), not tuple",
        ),
        (lambda: P.line([1.0], [1.0], fontsize=3), TypeError, "line(): got an unexpected keyword argument 'fontsize'"),
        (lambda: P.line([1.0]), TypeError, "line(): missing a required argument: 'y'"),
        (lambda: P.line([1.0], [1.0], [2.0]), TypeError, "line(): too many positional arguments"),
        (lambda: P.line(x=[1.0], y=[1.0]), TypeError, "line(): missing a required positional-only argument: 'x'"),
        (lambda: P.line([1.0], [1.0], font="mono"), ValueError, "line(): 'font' must be 'serif' or 'sans', not 'mono'"),
    ]
    for call, kind, message in cases:
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message, message


def test_module_function_is_the_method_on_a_new_figure():
    a = P.scatter([0.0, 1.0], [1.0, 2.0], shape="D", width=120, height=80, font="sans", theme="boxed", palette="npg")
    b = P.Figure(width=120, height=80, font="sans", theme="boxed", palette="npg").scatter(
        [0.0, 1.0], [1.0, 2.0], shape="D"
    )
    assert a.to_svg() == b.to_svg() and (a.width, a.height) == (120.0, 80.0)
    assert P.line.__name__ == "line" and P.Figure.line.__qualname__ == "Figure.line"
    assert isinstance(_layers.Layer().entries(), list)


def test_a_long_line_is_thinned_and_a_large_scatter_is_a_picture_in_the_vector_files():
    n = 50_000
    x = sd.linspace(0.0, 100.0, n)
    y = sd.sin(x) + sd.cos(x * 977.0) * 0.5  # noise faster than any pixel
    fig = P.line(x, y)
    (path,) = data_paths(fig)
    assert len(path.ops) < n // 3 and path.ops[0] == "M" and set(path.ops[1:]) == {"L"}  # at most four points a column
    xs, ys = path.xy[0::2], path.xy[1::2]
    full = fig._root._build("t")
    frame = full.items[1].clip
    assert min(xs) == pytest.approx(frame[0] + frame[2] * 0.04 / 1.08, abs=0.5)  # the same extent as the data
    assert max(ys) - min(ys) == pytest.approx(frame[3] / 1.08, rel=0.02)
    short = P.line(x[:5000], y[:5000])
    assert 3000 < len(data_paths(short)[0].ops) < 5000  # visible wiggles kept, points on a straight run left out
    smooth = P.line(x[:5000], sd.sin(x[:5000]))
    assert len(data_paths(smooth)[0].ops) < 400  # a smooth curve: a few points a bp

    from stardust.plot import _png

    m = 2 * _png.RASTER_FROM
    big = P.scatter(sd.linspace(0.0, 1.0, m), sd.sin(sd.linspace(0.0, 50.0, m)), size=1.0, alpha=0.5)
    svg, pdf = big.to_svg(), big.to_pdf()
    assert svg.count("<use ") < 100 and "data:image/png;base64," in svg and len(svg) < 400_000
    assert b"/Subtype /Image" in pdf and pdf.count(b" Do") < 100 and len(pdf) < 300_000
    small = P.scatter(sd.linspace(0.0, 1.0, 500), sd.sin(sd.linspace(0.0, 50.0, 500))).to_svg()
    assert small.count("<use ") >= 500 and "data:image" not in small  # a small one stays vector
    assert len(items(big, _scene.Markers)[0].at) == 2 * m  # the scene itself keeps the points


def test_bar_gaps_stacks_errors_and_values():
    from stardust.plot import _basic

    x, g = ["Q1", "Q2"] * 2, ["Group A"] * 2 + ["Residuals"] * 2
    h = [2.0, 3.0, 4.0, -1.0]
    side = P.bar(x, h, group=g)
    a, b = side._cell.layers
    assert a.width == pytest.approx(0.4 * 0.9) and float(b.pos[0] - a.pos[0]) == pytest.approx(0.4)  # a gap between
    assert P.bar(x, h, group=g, gap=0.0)._cell.layers[0].width == pytest.approx(0.4)
    stacked = P.bar(x, h, group=g, stack=True)
    low, high = stacked._cell.layers
    assert high.base.tolist() == [2.0, 0.0] and high.pos.tolist() == low.pos.tolist()  # on top; a negative goes down
    assert stacked._cell.layers[1].extent("y", False)[:2] == (-1.0, 6.0)
    err = P.bar(["Q1", "Q2"], [2.0, 3.0], err=([0.5, 0.2], [1.0, 0.3]))
    lo, hi = err._cell.layers[0].err
    assert lo.tolist() == [1.5, 2.8] and hi.tolist() == [3.0, 3.3]
    sym = P.bar(["Q1", "Q2"], [2.0, 3.0], err=[0.5, 0.2])._cell.layers[0].err
    assert sym[0].tolist() == [1.5, 2.8] and sym[1].tolist() == [2.5, 3.2]
    assert (
        _basic.value_text(0.1234, ".1%", "bar") == r"$12.3\%$"
        and _basic.value_text(12345.0, ",.0f", "bar") == "$12{,}345$"
    )
    assert _basic.value_text(1.2e-5, True, "bar") == r"$1.2\times10^{-5}$"
    marked = P.bar(["Q1", "Q2"], [2.0, 3.0], annotate=True)._cell.layers[0]
    assert marked.text == ["$2$", "$3$"] and marked.strings() == ["$2$", "$3$"]
    for kw, kind, text in (
        (dict(gap=1.0), ValueError, "'gap' must be in [0, 1)"),
        (dict(stack=True, err=[1.0, 1.0]), ValueError, "'err' cannot be given for stacked bars"),
        (dict(err=[-1.0, 1.0]), ValueError, "'err' must not be negative"),
        (dict(annotate=3), TypeError, "'annotate' must be a bool or a format spec string, not int"),
        (dict(annotate="q"), ValueError, "'annotate' is not a format spec for numbers: 'q'"),
    ):
        with pytest.raises(kind, match=text.replace("(", r"\(").replace(")", r"\)").replace("[", r"\[")):
            P.bar(["Q1", "Q2"], [2.0, 3.0], **kw)
