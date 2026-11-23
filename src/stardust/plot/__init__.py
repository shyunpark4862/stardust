"""stardust.plot: figures of mathematical functions and publication figures
(docs/api/plot.md, docs/design/plot.md).

Usage: ``import stardust as sd`` then ``fig = sd.plot.line(x, y)``, ``fig.xlabel(r"$x$")``,
``fig.save("fig.pdf")``. Every plotting function returns a Figure; there is no current figure
and no window. The same name is a Figure method that adds to an existing figure. Sizes are mm,
every string is typeset by LaTeX (TeX Live is needed when a figure is drawn), and the outputs
are PDF, SVG and PNG from the library's own renderer.

The public functions come milestone by milestone (docs/design/plot.md §9): PL3 and PL4 brought
Figure, grid and the basic plots (line, scatter, bar, area, step, errorbar, hline, vline, text,
annotate).
"""

from . import _basic
from ._basic import grid
from ._figure import Figure

_functions = _basic.install()
line = _functions["line"]
scatter = _functions["scatter"]
bar = _functions["bar"]
area = _functions["area"]
step = _functions["step"]
errorbar = _functions["errorbar"]
hline = _functions["hline"]
vline = _functions["vline"]
text = _functions["text"]
annotate = _functions["annotate"]

from . import _function  # noqa: E402  (it adds to the methods above)
from ._function import PlotInfo  # noqa: E402, F401

_functions = _basic.install(_function.METHODS)
function = _functions["function"]
parametric = _functions["parametric"]
polar = _functions["polar"]

from . import _field  # noqa: E402

_functions = _basic.install(_field.METHODS)
implicit = _functions["implicit"]
region = _functions["region"]
contour = _functions["contour"]
density = _functions["density"]
phase = _functions["phase"]
vectorfield = _functions["vectorfield"]
streamlines = _functions["streamlines"]

from . import _stat  # noqa: E402
from ._facet import Facets, facet  # noqa: E402, F401  (Facets: the type facet() returns)
from ._stat import diagnostics, pairs  # noqa: E402

_functions = _basic.install(_stat.METHODS)
histogram = _functions["histogram"]
hist2d = _functions["hist2d"]
hexbin = _functions["hexbin"]
ecdf = _functions["ecdf"]
boxplot = _functions["boxplot"]
strip = _functions["strip"]
qq = _functions["qq"]
heatmap = _functions["heatmap"]
coefficients = _functions["coefficients"]
distribution = _functions["distribution"]
from . import _surface  # noqa: E402

_functions = _basic.install(_surface.METHODS, flat=False)
surface = _functions["surface"]
parametric_surface = _functions["parametric_surface"]
curve3d = _functions["curve3d"]
scatter3d = _functions["scatter3d"]
for _name, _method in _stat.FIGURE_ONLY.items():  # they draw over an existing figure only
    _method.__name__, _method.__qualname__ = _name, f"Figure.{_name}"
    setattr(Figure, _name, _basic.flat_only(_name, _method))
del _name, _method

from .. import _calls  # noqa: E402

_calls.keyword_init(Figure)  # "Figure(): unexpected keyword argument 'foo'"
grid = _calls.function(grid)
facet = _calls.function(facet)
pairs = _calls.function(pairs)
diagnostics = _calls.function(diagnostics)
del _calls

__all__ = [
    "Figure",
    "annotate",
    "area",
    "bar",
    "boxplot",
    "coefficients",
    "contour",
    "curve3d",
    "density",
    "diagnostics",
    "distribution",
    "ecdf",
    "errorbar",
    "facet",
    "function",
    "grid",
    "heatmap",
    "hexbin",
    "hist2d",
    "histogram",
    "hline",
    "implicit",
    "line",
    "pairs",
    "parametric",
    "parametric_surface",
    "phase",
    "polar",
    "qq",
    "region",
    "scatter",
    "scatter3d",
    "step",
    "streamlines",
    "strip",
    "surface",
    "text",
    "vectorfield",
    "vline",
]
