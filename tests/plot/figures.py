"""Figures the plot tests, the golden cases and tools/plot_text.py share: each entry of FIGURES
makes a Figure from fixed data (no random state beyond sd.random's seeded generator), given the
font family. Their strings are in the recorded text cache (scenes.TEXT)."""

import math

import stardust as sd

P = sd.plot


def lines(font):
    x = sd.linspace(-3.0, 3.0, 200)
    fig = P.line(x, sd.exp(-x * x / 2) * sd.cos(4 * x), label=r"$e^{-t^2/2}\cos 4t$", font=font)
    fig.line(x, sd.exp(-x * x / 2), style="--", label=r"$e^{-t^2/2}$").line(
        x, -sd.exp(-x * x / 2), style="--", color="C1"
    )
    return fig.xlabel(r"시간 $t$").ylabel("진폭").title("감쇠 진동")


def groups(font):
    xs = sd.asarray([i % 20 / 2.0 for i in range(60)])
    slope = sd.asarray([0.5] * 20 + [1.0] * 20 + [1.6] * 20)
    wiggle = sd.asarray([math.sin(1.7 * i) for i in range(60)])
    g = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    fig = P.scatter(xs, xs * slope + wiggle, group=g, alpha=0.8, font=font)
    return fig.xlabel(r"$x$").ylabel(r"$y$").legend(loc="upper left", title="Group")


def bars(font):
    doses = ["control", "low", "mid", "high"] * 2
    fig = P.bar(doses, [3.1, 4.2, 6.5, 7.9, 2.2, 3.9, 5.1, 8.8], group=["2024"] * 4 + ["2025"] * 4, font=font)
    return fig.ylabel("Response (a.u.)").legend(loc="top")


def errors(font):
    x = sd.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
    y = sd.asarray([2.1e5, 3.9e5, 6.2e5, 7.8e5, 1.01e6])
    fig = P.errorbar(x, y, yerr=[3e4, 5e4, 4e4, 8e4, 6e4], label="measured", font=font)
    fig.line(x, 2e5 * x, color="gray", style=":", label="model").area(x, 1.8e5 * x, 2.2e5 * x, color="gray", alpha=0.2)
    fig.hline(5e5, style="--", width=0.5).annotate("threshold", (2.5, 5e5), (1.6, 8.5e5))
    return fig.xlabel("Dose (mg)").ylabel("Counts")


def loglog(font):
    x = sd.asarray([10.0 ** (k / 10.0) for k in range(41)])
    fig = P.line(x, x**-1.5, label=r"$x^{-3/2}$", font=font).line(x, 3 * x**-0.5, label=r"$3x^{-1/2}$")
    return fig.xscale("log").yscale("log").xlabel(r"$x$").ylabel(r"$f(x)$")


def steps(font):
    t = sd.linspace(0.0, 2 * math.pi, 100)
    k = sd.linspace(0.0, 2 * math.pi, 13)
    fig = P.line(t, sd.sin(t), label=r"$\sin\theta$", font=font).step(k, sd.sin(k), where="mid", label="step")
    return fig.xticks("pi").xlabel(r"$\theta$").text(4.7, 0.6, r"$\sin\theta$", color="C0").vline(math.pi, style=":")


def panels(font):
    return P.grid([lines(font), groups(font), bars(font), errors(font)], ncols=2, labels="A", width=183)


def poles(font):
    fig = P.function(sd.tan, -2 * math.pi, 2 * math.pi, asymptotes=True, font=font)
    return fig.xticks("pi").title(r"$\tan x$")


def domain(font):
    fs = [sd.sqrt, lambda x: sd.log(x), lambda x: sd.sqrt(1 - x * x), sd.floor]
    return P.function(fs, -1.5, 3, labels=[r"$\sqrt{x}$", r"$\ln x$", r"$\sqrt{1-x^2}$", "floor"], font=font)


def wiggle(font):
    return P.function(lambda x: sd.sin(1 / x), -0.5, 0.5, font=font).title(r"$\sin(1/x)$")


def curves(font):
    fig = P.implicit(
        lambda x, y: (x * x + y * y - 1) ** 3 - x * x * y**3, -1.5, 1.5, -1.3, 1.5, color="C1", label="heart", font=font
    )
    fig.polar(lambda t: 0.4 * sd.cos(4 * t), 0, 2 * math.pi, label="rose").region(
        lambda x, y: 0.09 - x * x - (y + 0.9) ** 2, -1.5, 1.5, -1.3, 1.5
    )
    return fig.legend(loc="lower left")


def levels(font):
    return P.contour(lambda x, y: sd.sin(x) * sd.cos(y) + 0.1 * x, -3, 3, -2, 2, levels=8, font=font)


def bands(font):
    f = lambda x, y: (1 - x) ** 2 + 100 * (y - x * x) ** 2  # noqa: E731
    return P.contour(
        f, -2, 2, -1, 3, levels=[1, 5, 20, 50, 100, 300, 800], filled=True, labels=False, cmap="magma", font=font
    )


def heat(font):
    f = lambda x, y: sd.exp(-(x * x + y * y) / 2) * sd.cos(3 * x)  # noqa: E731
    return P.density(f, -3, 3, -2.2, 2.2, cmap="rdbu", font=font).xlabel(r"$x$").ylabel(r"$y$")


def portrait(font):
    return P.phase(lambda z: (z * z - 1) * (z - 2 - 1j) ** 2 / (z * z + 2 + 2j), -3, 3, -3, 3, font=font)


FIGURES = {
    "poles": poles,
    "domain": domain,
    "wiggle": wiggle,
    "curves": curves,
    "levels": levels,
    "bands": bands,
    "heat": heat,
    "portrait": portrait,
    "lines": lines,
    "groups": groups,
    "bars": bars,
    "errors": errors,
    "loglog": loglog,
    "steps": steps,
    "panels": panels,
}


def _draws(n, mu, s, seed):
    """n fixed values shaped like a normal sample (the inverse cdf on an even grid, shuffled by a
    fixed rule), so the figures need no random generator."""
    N = sd.stats.Normal(mu=0.0, sigma=1.0)
    p = [((i * 37 + seed * 11) % n + 0.5) / n for i in range(n)]
    return N.icdf(sd.asarray(p)) * s + mu


def _three():
    a, b, c = _draws(30, 5.0, 1.0, 1), _draws(30, 6.1, 1.3, 2), _draws(30, 7.9, 0.9, 3)
    return a, b, c, sd.concat([a, b, c]), ["control"] * 30 + ["low dose"] * 30 + ["high dose"] * 30


def boxes(font):
    a, b, c, vals, grp = _three()
    fig = P.boxplot(grp, vals, font=font).strip(grp, vals, alpha=0.5, size=2.2, color="gray").ylabel("Response")
    return fig.compare(sd.htest.tukey_hsd(a, b, c))


def hists(font):
    *_, vals, grp = _three()
    return P.histogram(vals, group=grp, bins=12, font=font).xlabel("Response").ylabel("Count").legend(loc="upper right")


def regress(font):
    x = sd.linspace(0.0, 10.0, 40)
    y = 1.8 * x + 3.0 + _draws(40, 0.0, 2.0, 5)
    res = sd.regression.ols(y, sd.reshape(x, (-1, 1)))
    return P.scatter(x, y, alpha=0.8, font=font).fit(res).xlabel(r"$x$").ylabel(r"$y$")


def quantiles(font):
    *_, vals, grp = _three()
    return P.grid(
        [P.qq(_draws(60, 0.0, 1.0, 7) ** 3, font=font), P.ecdf(vals, group=grp, font=font).xlabel("Response")],
        ncols=2,
        width=160,
    )


def matrix(font):
    corr = [[1.0, 0.82, -0.31, 0.05], [0.82, 1.0, -0.12, 0.4], [-0.31, -0.12, 1.0, -0.66], [0.05, 0.4, -0.66, 1.0]]
    names = ["Group A", "x", "y", "z"]
    return P.heatmap(corr, xlabels=names, ylabels=names, cmap="rdbu", vmin=-1, vmax=1, annotate=True, font=font)


def dists(font):
    return P.grid(
        [
            P.distribution(sd.stats.Gamma(a=2.0, scale=1.0), label="pdf", font=font),
            P.distribution(sd.stats.Binomial(n=20, p=0.3), font=font),
        ],
        ncols=2,
        width=160,
    )


def checks(font):
    x = sd.linspace(0.0, 10.0, 40)
    X = sd.stack([x, sd.sin(x)], axis=1)
    y = 1.8 * x - 2.0 * sd.sin(x) + _draws(40, 0.0, 1.0, 9)
    res = sd.regression.ols(y, X)
    return P.grid([P.diagnostics(res, font=font), P.coefficients(res, font=font)], ncols=3, width=183)


def hill(font):
    f = lambda x, y: sd.sin(sd.sqrt(x * x + y * y) * 3) / (sd.sqrt(x * x + y * y) * 3 + 1e-12)  # noqa: E731
    return P.surface(f, -3, 3, -3, 3, font=font).xlabel(r"$x$").ylabel(r"$y$").zlabel(r"$z$")


def crossing(font):
    fig = P.surface(lambda x, y: x * x - y * y, -1, 1, -1, 1, cmap="rdbu", font=font)
    return fig.surface(lambda x, y: 0 * x + 0.2, -1, 1, -1, 1, color=(0.6, 0.6, 0.6, 1.0), mesh=False)


def solids(font):
    torus = P.parametric_surface(
        lambda u, v: (2 + sd.cos(v)) * sd.cos(u),
        lambda u, v: (2 + sd.cos(v)) * sd.sin(u),
        lambda u, v: sd.sin(v),
        0,
        2 * math.pi,
        0,
        2 * math.pi,
        color="C0",
        elevation=35,
        font=font,
    )
    helix = P.curve3d(sd.cos, sd.sin, lambda t: t / 6, 0, 8 * math.pi, width=1.0, perspective=False, font=font)
    helix.scatter3d([0.0, 0.5, -0.5], [0.0, 0.5, -0.5], [1.0, 2.0, 3.0], color="C1", size=5)
    return P.grid([torus, helix], ncols=2, width=160)


FIGURES.update({"hill": hill, "crossing": crossing, "solids": solids})
FIGURES.update(
    {
        "boxes": boxes,
        "hists": hists,
        "regress": regress,
        "quantiles": quantiles,
        "matrix": matrix,
        "dists": dists,
        "checks": checks,
    }
)
