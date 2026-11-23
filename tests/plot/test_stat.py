"""Statistical plots and the plots of results (docs/api/plot.md §5, §6): the numbers drawn are
the computing areas' own."""

import math

import pytest

import stardust as sd
from stardust.plot import _figure, _layers, _scene, _stat, _text

from . import figures

P = sd.plot


def measure(fig):
    return _figure._measure(fig, fig._cell, 200.0, 150.0, _figure.THEMES["paper"], _text.engine("serif"))


DATA = [1.0, 2.0, 2.5, 3.0, 9.0, 4.0, 4.5, math.nan, 5.0, 6.0]


def test_histogram_bars_are_sd_histogram():
    fig = P.histogram(DATA, bins=4)
    (h,) = fig._cell.layers
    ref = sd.histogram(sd.asarray([v for v in DATA if v == v]), bins=4)
    assert h.edges == ref.bin_edges.tolist() and h.heights == [float(c) for c in ref.hist.tolist()]
    m = measure(fig)
    assert m.ya.lo == 0.0 and m.ya.hi > max(h.heights)  # the bars stand on zero
    dens = P.histogram(DATA, bins=4, density=True, range=(0, 10))._cell.layers[0]
    ref = sd.histogram(sd.asarray([v for v in DATA if v == v]), bins=4, range=(0, 10), density=True)
    assert dens.heights == ref.hist.tolist() and dens.edges == [0.0, 2.5, 5.0, 7.5, 10.0]
    two = P.histogram(DATA, group=["a"] * 5 + ["b"] * 5, bins=3)._cell.layers
    assert (
        two[0].edges == two[1].edges and two[0].color[3] == 0.6 and [la.label for la in two] == ["a", "b"]
    )  # shared bins
    with pytest.raises(ValueError, match=r"^histogram\(\): "):
        P.histogram(DATA, bins=0)


def test_ecdf_rises_at_each_value():
    layer = P.ecdf([3.0, 1.0, 2.0, math.nan])._cell.layers[0]
    assert layer.x.tolist() == [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]
    assert layer.y.tolist() == pytest.approx([0.0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0])


def test_box_is_the_quartiles_and_whiskers_within_one_and_a_half_iqr():
    v = sd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0, math.nan])
    s = _stat.box_stats(v)
    q = sd.quantile(sd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 30.0]), sd.asarray([0.25, 0.5, 0.75])).tolist()
    assert [s["q1"], s["q2"], s["q3"]] == q == [3.0, 5.0, 7.0]
    assert (s["lo"], s["hi"], s["out"]) == (1.0, 8.0, [30.0])  # 30 is beyond 7 + 1.5 * 4
    assert _stat.box_stats(sd.asarray([math.nan])) is None
    fig = P.boxplot(["Group A"] * 5 + ["Residuals"] * 4, [1.0, 2.0, 3.0, 4.0, 50.0, 2.0, 4.0, 6.0, 8.0])
    (boxes,) = fig._cell.layers
    assert [s["pos"] for s in boxes.stats] == [0.0, 1.0] and fig._cell.xcats == ["Group A", "Residuals"]
    paths = [i for i in fig._root._build("t").items if isinstance(i, _scene.Path) and i.clip is not None]
    assert [p.ops for p in paths] == ["MLML" * 2, "MLLLZ" * 2, "ML" * 2]  # whiskers, boxes, medians
    marks = [i for i in fig._root._build("t").items if isinstance(i, _scene.Markers)]
    assert len(marks) == 1 and len(marks[0].at) == 2  # the one outlier
    every = P.boxplot(["Group A"] * 5, [1.0, 2.0, 3.0, 4.0, 50.0], points=True)
    assert len([i for i in every._root._build("t").items if isinstance(i, _scene.Markers)][0].at) == 10
    single = P.boxplot(None, [1.0, 2.0, 3.0])
    assert single._cell.xcats == [""] and single._cell.layers[0].stats[0]["q2"] == 2.0
    dodge = P.boxplot(
        ["a", "a", "b", "b"] * 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], group=["x"] * 4 + ["y"] * 4
    )._cell.layers
    assert [s["pos"] for s in dodge[0].stats] == [-0.15, 0.85] and [s["pos"] for s in dodge[1].stats] == [0.15, 1.15]


def test_strip_scatter_is_the_same_every_time_and_takes_a_generator():
    g, v = ["a", "b"] * 10, [float(i) for i in range(20)]
    a, b = P.strip(g, v)._cell.layers[0], P.strip(g, v)._cell.layers[0]
    assert a.x.tolist() == b.x.tolist() and a.y.tolist() == [float(i) for i in range(0, 20, 2)] + [
        float(i) for i in range(1, 20, 2)
    ]
    xs = a.x.tolist()
    assert all(-0.1 <= x <= 0.1 for x in xs[:10]) and all(0.9 <= x <= 1.1 for x in xs[10:]) and len(set(xs)) == 20
    seeded = P.strip(g, v, rng=5)._cell.layers[0].x.tolist()
    assert seeded == P.strip(g, v, rng=sd.random.default_rng(5))._cell.layers[0].x.tolist() != xs
    assert set(P.strip(g, v, jitter=0)._cell.layers[0].x.tolist()) == {0.0, 1.0}
    for call, kind, message in (
        (lambda: P.strip(g, v, jitter=2), ValueError, "strip(): 'jitter' must be in [0, 1]"),
        (
            lambda: P.strip(g, v, rng="x"),
            TypeError,
            "strip(): 'rng' must be an sd.random Generator, an int seed or None, not str",
        ),
        (lambda: P.boxplot(g, v[:3]), ValueError, "boxplot(): 'x' and 'y' must have the same length: 20 and 3"),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_qq_points_are_the_distributions_icdf():
    x = [3.0, 1.0, 2.0, 5.0, 4.0]
    fig = P.qq(x)
    line, pts = fig._cell.layers
    N = sd.stats.Normal(mu=0.0, sigma=1.0)
    assert pts.y.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert pts.x.tolist() == N.icdf(sd.asarray([0.1, 0.3, 0.5, 0.7, 0.9])).tolist()
    # the line passes through the quartiles of both
    (x0, x1), (y0, y1) = line.x.tolist(), line.y.tolist()
    k = (y1 - y0) / (x1 - x0)
    assert y0 + k * (float(N.icdf(0.25)) - x0) == pytest.approx(2.0) and y0 + k * (
        float(N.icdf(0.75)) - x0
    ) == pytest.approx(4.0)
    assert (fig._cell.xlabel, fig._cell.ylabel) == ("Theoretical quantiles", "Sample quantiles")
    e = P.qq(x, dist=sd.stats.Exponential(scale=2.0), line=False)._cell.layers
    assert (
        len(e) == 1
        and e[0].x.tolist() == sd.stats.Exponential(scale=2.0).icdf(sd.asarray([0.1, 0.3, 0.5, 0.7, 0.9])).tolist()
    )
    with pytest.raises(ValueError, match=r"qq\(\): 'x' needs at least 2 finite values"):
        P.qq([1.0, math.nan])
    with pytest.raises(TypeError, match=r"qq\(\): 'dist' must be an sd.stats distribution, not int"):
        P.qq(x, dist=3)


def test_heatmap_cells_labels_and_numbers():
    fig = P.heatmap([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], xlabels=["Group A", "x", "y"], annotate=True, cmap="magma")
    pic, text = fig._cell.layers
    assert (pic.iw, pic.ih, pic.smooth, pic.box) == (3, 2, False, (-0.5, 2.5, -0.5, 1.5))
    assert pic.rgba[:4] == bytes([0, 0, 4, 255]) and pic.rgba[-4:] == bytes(
        [252, 253, 191, 255]
    )  # the first row is the top one
    assert [(s, x, y) for s, x, y, _ in text.items] == [
        ("$0$", 0.0, 1.0),
        ("$1$", 1.0, 1.0),
        ("$2$", 2.0, 1.0),
        ("$3$", 0.0, 0.0),
        ("$4$", 1.0, 0.0),
        ("$5$", 2.0, 0.0),
    ]
    assert text.items[0][3] == (1.0, 1.0, 1.0, 1.0) and text.items[-1][3] == _scene.INK  # white on dark, black on light
    m = measure(fig)
    assert (
        m.xa.labels == ["Group A", "x", "y"]
        and (m.ya.ticks, m.ya.labels) == ([1.0, 0.0], ["$0$", "$1$"])
        and (m.xa.lo, m.xa.hi) == (-0.5, 2.5)
    )
    assert fig._cell.colorbar == ("magma", 0.0, 5.0)
    for call, kind, message in (
        (
            lambda: P.heatmap([1.0, 2.0]),
            ValueError,
            "heatmap(): 'z' must be 2-dimensional and not empty, not shape (2,)",
        ),
        (lambda: P.heatmap([[1.0, 2.0]], xlabels=["a"]), ValueError, "heatmap(): 'xlabels' must be 2 strings"),
        (lambda: P.heatmap([[1.0, 2.0]], vmin=3, vmax=3), ValueError, "heatmap(): 'vmin' must be less than 'vmax'"),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_compare_draws_the_given_result():
    a, b, c = sd.asarray([1.0, 2.0, 3.0, 4.0]), sd.asarray([2.0, 3.5, 4.1, 6.0]), sd.asarray([5.0, 6.5, 7.1, 9.0])
    g = ["control"] * 4 + ["low dose"] * 4 + ["high dose"] * 4
    fig = P.boxplot(g, sd.concat([a, b, c]))
    t = sd.htest.ttest_ind(a, c)
    fig.compare(t, groups=("control", "high dose"))
    br = fig._cell.layers[-1]
    assert (br.x1, br.x2, br.level) == (0.0, 2.0, 0) and br.s == f"$p = {t.pvalue:.3f}$" and fig._cell.headroom == 1
    fig.compare(sd.htest.ttest_ind(a, b), groups=("low dose", "control"), format="stars")
    assert (fig._cell.layers[-1].x1, fig._cell.layers[-1].x2, fig._cell.layers[-1].level, fig._cell.layers[-1].s) == (
        0.0,
        1.0,
        1,
        "ns",
    )
    multi = P.boxplot(g, sd.concat([a, b, c])).compare(sd.htest.tukey_hsd(a, b, c))
    brs = [la for la in multi._cell.layers if isinstance(la, _stat.Bracket)]
    pv = sd.htest.tukey_hsd(a, b, c).pvalue.tolist()
    assert [(br.x1, br.x2) for br in brs] == [(0.0, 1.0), (1.0, 2.0), (0.0, 2.0)]  # neighbours lowest
    assert [br.s for br in brs] == [
        _stat.p_text(pv[0][1], "p"),
        _stat.p_text(pv[1][2], "p"),
        _stat.p_text(pv[0][2], "p"),
    ]
    m = measure(multi)
    assert m.ya.hi > 9.0 + 3.0 and max(m.ya.ticks) <= 9.5  # room above the data, no ticks beside the brackets
    assert [_stat.p_text(p, "stars") for p in (0.2, 0.05, 0.01, 0.001)] == ["ns", "*", "**", "***"]
    assert _stat.p_text(0.0004, "p") == "$p < 0.001$" and _stat.p_text(0.0499, "p") == "$p = 0.050$"
    two = P.boxplot(["a", "a", "b", "b"], [1.0, 2.0, 3.0, 4.0])
    assert two.compare(t)._cell.layers[-1].x2 == 1.0  # two categories need no names
    for call, kind, message in (
        (
            lambda: P.line([0.0, 1.0], [0.0, 1.0]).compare(t),
            ValueError,
            "compare(): the figure has no category axis (compare draws over boxplot, strip or bar)",
        ),
        (lambda: fig.compare(3), TypeError, "compare(): 'result' must be a test result with a pvalue, not int"),
        (
            lambda: fig.compare(t),
            ValueError,
            "compare(): 'groups' must name the two categories compared (the figure has 3)",
        ),
        (lambda: fig.compare(t, groups=("control", "zz")), ValueError, "compare(): no category 'zz'"),
        (lambda: fig.compare(t, format="x"), ValueError, "compare(): 'format' must be 'p' or 'stars', not 'x'"),
        (
            lambda: two.compare(sd.htest.tukey_hsd(a, b, c)),
            ValueError,
            "compare(): the result compares 3 groups, the figure has 2 categories",
        ),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_fit_draws_the_results_own_line_and_band():
    x = sd.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    y = sd.asarray([1.1, 1.9, 3.2, 3.9, 5.3, 5.8])
    res = sd.regression.ols(y, sd.reshape(x, (-1, 1)))
    fig = P.scatter(x, y).fit(res)
    pts, band, line = fig._cell.layers
    assert (line.x[0], line.x[-1]) == (1.0, 6.0) and line.color == pts.color
    assert line.y.tolist() == res.predict(sd.reshape(line.x, (-1, 1))).tolist()
    iv = res.predict_interval(sd.reshape(band.x, (-1, 1)), alpha=0.05, kind="mean")
    assert (
        band.y1.tolist() == iv[:, 0].tolist()
        and band.y2.tolist() == iv[:, 1].tolist()
        and band.color[3] == pytest.approx(0.2)
    )
    assert (
        fig._cell.note
        == f"$y = {float(res.params[1]):.3g}x + {float(res.params[0]):.3g}$, $R^2 = {float(res.rsquared):.3f}$"
    )
    bare = P.scatter(x, y).fit(res, band=False, equation=False, color="red")
    assert (
        len(bare._cell.layers) == 2 and bare._cell.note is None and bare._cell.layers[1].color == (1.0, 0.0, 0.0, 1.0)
    )
    two = sd.regression.ols(y, sd.stack([x, x * x], axis=1))
    for call, kind, message in (
        (
            lambda: P.scatter(x, y).fit(two),
            ValueError,
            "fit(): 'result' must have one explanatory variable; it has parameters ['const', 'x1', 'x2']",
        ),
        (lambda: P.scatter(x, y).fit(3), TypeError, "fit(): 'result' must be a regression result, not int"),
        (lambda: P.Figure().fit(res), ValueError, "fit(): the figure has no data to take the x range from"),
        (lambda: P.scatter(x, y).fit(res, alpha=1), ValueError, "fit(): 'alpha' must be in (0, 1)"),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


def test_diagnostics_coefficients_distribution_pairs():
    x = sd.linspace(0.0, 10.0, 30)
    X = sd.stack([x, sd.sin(x)], axis=1)
    y = 1.8 * x - 2.0 * sd.sin(x) + sd.cos(7 * x)
    res = sd.regression.ols(y, X)
    d = P.diagnostics(res)
    assert d.shape == (2, 2) and d.width == 183.0
    assert (
        d._cells[0].layers[0].y.tolist() == res.resid.tolist()
        and d._cells[0].layers[0].x.tolist() == res.fittedvalues.tolist()
    )
    assert (
        d._cells[3].layers[0].x.tolist() == res.leverage.tolist()
        and d._cells[3].layers[0].y.tolist() == res.resid_studentized_internal.tolist()
    )
    assert d._cells[2].layers[0].y.tolist() == sd.sqrt(sd.abs(res.resid_studentized_internal)).tolist()
    c = P.coefficients(res)
    rule, bars = c._cell.layers
    ci = res.conf_int().tolist()
    assert c._cell.ycats == ["x2", "x1"] and bars.y.tolist() == [1.0, 0.0]  # the first coefficient on top
    assert (
        bars.x.tolist() == res.params.tolist()[1:]
        and bars.xlo.tolist() == [ci[1][0], ci[2][0]]
        and bars.xhi.tolist() == [ci[1][1], ci[2][1]]
    )
    assert rule.vertical and rule.value == 0.0
    assert P.coefficients(res, intercept=True, alpha=0.1)._cell.ycats == ["x2", "x1", "const"]

    g = sd.stats.Gamma(a=2.0, scale=1.0)
    fig = P.distribution(g)
    assert fig.info[0].kind == "function" and fig._cell.layers[0].x[0] == 0.0  # the support's finite end
    xs, ys = fig._cell.layers[0].x, fig._cell.layers[0].y
    assert ys.tolist() == g.pdf(xs).tolist()
    b = sd.stats.Binomial(n=20, p=0.3)
    bars = P.distribution(b)._cell.layers[0]
    assert isinstance(bars, _layers.Bars) and bars.value.tolist() == b.pmf(bars.pos).tolist()
    assert P.distribution(b, kind="cdf")._cell.layers[0].y.tolist()[-1] == pytest.approx(
        float(b.cdf(float(bars.pos[-1])))
    )
    assert P.distribution(g, kind="sf", a=1, b=3)._cell.layers[0].x[-1] == 3.0

    df = sd.frame.DataFrame(
        {"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 1.0, 4.0, 3.0], "c": [5, 6, 7, 8], "who": ["x", "x", "y", "y"]}
    )
    p = P.pairs(df, group="who")
    assert p.shape == (3, 3) and [type(la).__name__ for la in p._cells[0].layers] == ["Hist", "Hist"]
    assert p._cells[1].layers[0].x.tolist() == [2.0, 1.0] and p._cells[1].layers[0].y.tolist() == [
        1.0,
        2.0,
    ]  # row a, column b: x is b
    assert (p._cells[6].ylabel, p._cells[6].xlabel, p._cells[8].xlabel, p._cells[4].ylabel) == ("c", "a", "c", None)
    assert P.pairs(df, columns=["a", "c"]).shape == (2, 2)
    for call, kind, message in (
        (
            lambda: P.diagnostics(3),
            TypeError,
            "diagnostics(): 'result' must be a linear regression result (it has no fittedvalues), not int",
        ),
        (lambda: P.coefficients(3), TypeError, "coefficients(): 'result' must be a regression result, not int"),
        (lambda: P.distribution(3), TypeError, "distribution(): 'dist' must be an sd.stats distribution, not int"),
        (
            lambda: P.distribution(g, kind="pmf"),
            ValueError,
            "distribution(): 'kind' must be 'pdf', 'cdf' or 'sf', not 'pmf'",
        ),
        (lambda: P.pairs([1, 2]), TypeError, "pairs(): 'data' must be a DataFrame, not list"),
        (lambda: P.pairs(df, columns=["a"]), ValueError, "pairs(): at least 2 numeric columns are needed, not 1"),
    ):
        with pytest.raises(kind) as e:
            call()
        assert str(e.value) == message


@pytest.mark.parametrize("name", ["boxes", "hists", "regress", "quantiles", "matrix", "dists", "checks"])
def test_the_statistical_figures_draw_in_every_format(name):
    fig = figures.FIGURES[name]("serif")
    assert len(fig.to_svg()) > 3000 and fig.to_pdf().startswith(b"%PDF") and fig.to_png(dpi=72)[:4] == b"\x89PNG"


def test_fit_equations_intervals_and_links():
    from stardust.plot import _stat

    assert [_stat._coef(v) for v in (1.2e-5, 1.86e-10, 0.154, 1e6)] == [
        r"1.2\times10^{-5}",
        r"1.86\times10^{-10}",
        "0.154",
        "10^{6}",
    ]
    x = sd.linspace(0.0, 10.0, 30)
    tiny = 1.2e-5 * x + 3e-6 * sd.sin(7 * x)
    note = P.scatter(x, tiny).fit(sd.regression.ols(tiny, sd.reshape(x, (-1, 1))))._cell.note
    assert note.startswith(r"$y = 1.2\times10^{-5}x + ") and note.count("{") == note.count("}")
    y = 1.8 * x + 3 + sd.cos(7 * x)
    for name in ("rlm", "quantreg"):  # their predict_interval takes no 'kind'
        assert len(P.scatter(x, y).fit(getattr(sd.regression, name)(y, x))._cell.layers) == 3
    counts = sd.round(sd.exp(0.2 * x))
    note = P.scatter(x, counts).fit(sd.regression.glm(counts, x, family="poisson"))._cell.note
    assert note.startswith(r"$\log\mu = ")  # the line is the mean; the equation is on the link's scale
    ridge = P.coefficients(sd.regression.ridge(y, x, alpha=1.0))  # no intervals: points alone
    assert ridge._cell.layers[-1].xlo.tolist() == ridge._cell.layers[-1].xhi.tolist()


def test_compare_stacks_near_pairs_lowest_and_reads_dunnett():
    from stardust.plot import _stat

    a, b, c = sd.asarray([1.0, 2, 3, 4]), sd.asarray([2.0, 3.5, 4.1, 6]), sd.asarray([5.0, 6.5, 7.1, 9])
    g = ["A"] * 4 + ["B"] * 4 + ["C"] * 4
    fig = P.boxplot(g, sd.concat([a, b, c])).compare(sd.htest.tukey_hsd(a, b, c))
    brs = [br for br in fig._cell.layers if isinstance(br, _stat.Bracket)]
    assert {br.total for br in brs} == {3} and fig._cell.headroom == 3
    widest = max(brs, key=lambda br: br.x2 - br.x1)
    assert widest.level == 2  # neighbours first, at level 0: the lowest, just over the data
    page = [9.0 + 13.0 * (br.total - 1 - br.level) for br in brs]  # draw()'s height below the plot's top
    assert page[brs.index(widest)] == min(page)  # the widest stands highest
    dun = P.boxplot(g, sd.concat([a, b, c])).compare(sd.htest.dunnett(b, c, control=a))
    assert [(br.x1, br.x2) for br in dun._cell.layers if isinstance(br, _stat.Bracket)] == [(0.0, 1.0), (0.0, 2.0)]
    batch = sd.htest.ttest_ind(sd.asarray([[1.0, 2, 3, 4], [2.0, 3, 4, 5]]), sd.asarray([[2.5, 3, 4, 6]] * 2), axis=1)
    with pytest.raises(ValueError, match="a batch of tests cannot be drawn"):
        P.boxplot(["a"] * 4 + ["b"] * 4, [1.0] * 8).compare(batch)
    assert _stat.p_text(math.nan, "stars") == _stat.p_text(math.nan, "p") == "n/a"


def test_distribution_reaches_the_peak_and_counts_on_integers():
    info = P.distribution(sd.stats.LogNormal(mu=0.0, sigma=1.0)).info[0]
    assert info.clipped == (False, False) and info.ylim[1] == pytest.approx(0.6577, abs=1e-3)
    ticks = P.distribution(sd.stats.Binomial(n=20, p=0.3))._cell.xticks[0]
    assert all(t == int(t) for t in ticks)
    with pytest.raises(ValueError, match=r"distribution\(\): a batch of distributions cannot be drawn"):
        P.distribution(sd.stats.Normal(mu=sd.asarray([0.0, 1.0]), sigma=1.0))


def test_hist2d_and_hexbin_count_the_pairs():
    x = [0.1, 0.2, 0.9, 0.95, 0.5]
    y = [0.1, 0.15, 0.9, 0.85, 0.5]
    h = P.hist2d(x, y, bins=2, range=((0.0, 1.0), (0.0, 1.0)))
    pic = h._cell.layers[0]
    assert (pic.iw, pic.ih, pic.smooth) == (2, 2, False) and h._cell.colorbar[1:] == (0.0, 3.0)
    rgba = pic.rgba  # rows top down: the upper left bin is empty, so blank
    assert rgba[3] == 0 and rgba[4 * 1 + 3] == 255
    hx = P.hexbin(x, y, size=4)
    hexes = hx._cell.layers[0]
    assert len(hexes.cx) == len(hexes.colors) >= 3 and hx._cell.colorbar[2] >= 2.0
    with pytest.raises(ValueError, match=r"hist2d\(\): 'bins' as a pair must hold 2 numbers"):
        P.hist2d(x, y, bins=(2, 3, 4))
    with pytest.raises(TypeError, match=r"hexbin\(\): 'range' must be \(\(x0, x1\), \(y0, y1\)\)"):
        P.hexbin(x, y, range=(0, 1))


def test_hist2d_and_hexbin_take_a_range_in_order_and_stand_on_a_log_axis():
    x, y = [0.1, 0.2, 0.9], [0.1, 0.2, 0.9]
    for plot in (P.hist2d, P.hexbin):
        with pytest.raises(ValueError, match=rf"{plot.__name__}\(\): 'range' must have x0 at most x1, not 1 and 0"):
            plot(x, y, range=((1.0, 0.0), (0.0, 1.0)))
        with pytest.raises(ValueError, match=rf"{plot.__name__}\(\): 'range' must have y0 at most y1"):
            plot(x, y, range=((0.0, 1.0), (1.0, 0.5)))
    # two bins of values, 1..50.5 and 50.5..100: on a log axis the edge between them stands at 50.5
    h = P.hist2d([1.0, 2.0, 3.0, 100.0], [0.0, 0.0, 0.0, 0.0], bins=(2, 1)).xscale("log")
    sc = h._root._build("test")
    img = max((i for i in sc.items if isinstance(i, _scene.Image)), key=lambda i: i.w)  # (not the colour bar's)
    row = [img.rgba[4 * k : 4 * k + 4] for k in range(img.iw)]
    edge = next(k for k in range(img.iw) if row[k] != row[0])
    lo, hi = h._cell.layers[0].extent("x", True)[:2]
    assert lo == 1.0 and hi == 100.0
    assert edge / img.iw == pytest.approx(math.log10(50.5) / 2.0, abs=2.0 / img.iw)
    # hexbin on log axes bins the logarithms: every hexagon has one size on the page
    xs = [10.0 ** (k / 7.0) for k in range(-14, 15)]
    ys = [10.0 ** ((k * 5 % 29) / 9.0) for k in range(-14, 15)]
    hx = P.hexbin(xs + [0.0, -1.0], ys + [1.0, 1.0], size=5).xscale("log").yscale("log")
    sizes = []
    for item in hx._root._build("test").items:
        if isinstance(item, _scene.Path) and item.fill is not None and item.ops.startswith("MLLLLLZ"):
            for k in range(0, len(item.xy), 12):
                px, py = item.xy[k : k + 12 : 2], item.xy[k + 1 : k + 12 : 2]
                sizes.append((max(px) - min(px), max(py) - min(py)))
    assert len(sizes) >= 5 and all(math.isfinite(a) and math.isfinite(b) for a, b in sizes)
    w, h = sizes[0]
    assert all(a == pytest.approx(w, rel=1e-6) and b == pytest.approx(h, rel=1e-6) for a, b in sizes)
    layer = hx._cell.layers[0]
    assert sum(layer.counts) == len(xs)  # the pairs at zero and below are not counted on a log axis
    assert min(layer.cx) > 0.0 and min(layer.cy) > 0.0
    lin = P.hexbin(xs, ys, size=5)._cell.layers[0]
    assert lin.scales == (False, False) and sum(lin.counts) == len(xs)


def test_a_facet_counts_on_one_grid_with_one_colour_range():
    df = sd.frame.DataFrame(
        {
            "s": ["u"] * 6 + ["v"] * 3,
            "x": [0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 2.0, 2.5, 3.0],
            "y": [0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 4.0, 4.5, 5.0],
        }
    )
    for name in ("hexbin", "hist2d"):
        g = getattr(P.facet(df, "s"), name)("x", "y")
        u, v = (cell._cell for cell in g.axes)
        a, b = u.layers[0], v.layers[0]
        assert a.box == b.box == (0.0, 3.0, 0.0, 5.0)  # the box of all the rows
        assert u.colorbar is None and v.colorbar is not None  # one bar, on the last cell
        g.figure._root._build("test")
        assert a.top == b.top == a.most == 3.0 and b.most == 1.0  # the largest count of all
        assert v.colorbar[1:] == (0.0, 3.0) and u.colorbar is None
    own = P.facet(df, "s").hexbin("x", "y", range=((0.0, 4.0), (0.0, 6.0)))
    assert all(cell._cell.layers[0].box == (0.0, 4.0, 0.0, 6.0) for cell in own.axes)


def test_a_density_over_a_taller_layer_leaves_the_range_to_it():
    xs = sd.stats.Normal(mu=170.0, sigma=8.0).sample(200, rng=sd.random.default_rng(1))
    tallest = float(sd.max(sd.histogram(xs, bins=15, density=True).hist))
    fig = P.histogram(xs, bins=15, density=True).distribution(sd.stats.Normal(mu=170.0, sigma=8.0))
    th = _figure.THEMES["paper"]
    assert fig._cell.ylim == [None, None] and _figure._range(fig._cell, "y", th)[1] >= tallest
    two = P.distribution(sd.stats.StudentT(df=1.0), a=-5, b=5).distribution(
        sd.stats.Normal(mu=0.0, sigma=1.0), a=-5, b=5
    )
    assert _figure._range(two._cell, "y", th)[1] >= 1.0 / math.sqrt(2.0 * math.pi)
    alone = P.distribution(sd.stats.Normal(mu=0.0, sigma=1.0))
    assert _figure._range(alone._cell, "y", th)[1] == pytest.approx(1.05 / math.sqrt(2.0 * math.pi), rel=1e-2)


def test_the_diagonal_of_pairs_reads_on_its_rows_axis():
    rng = sd.random.default_rng(11)
    N = sd.stats.Normal(mu=0.0, sigma=1.0)
    z1, z2 = N.sample(60, rng=rng), N.sample(60, rng=rng)
    df = sd.frame.DataFrame({"height": 163 + 6 * z1, "weight": 56 + 3.6 * z1 + 5 * z2})
    fig = P.pairs(df)
    th = _figure.THEMES["paper"]
    diag, beside = fig._root._cells[0], fig._root._cells[1]
    assert _figure._range(diag, "y", th) == pytest.approx(_figure._range(beside, "y", th), rel=1e-12)
