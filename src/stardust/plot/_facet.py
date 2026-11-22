"""Facets: one cell per value of a column, the same plot in each (docs/design/plot.md PL52).

    sd.plot.facet(df, "species").scatter("length", "width").xlabel("Length")

A Facets object stands for a grid of cells, one per value of the column `by` (sorted when they
are numbers, else in order of first appearance), each titled with its value. A plotting method
called on it draws in every cell with that cell's rows of the DataFrame as `data`; the arguments
name columns. The cells share their axes (sharex, sharey as grid takes them), a group keeps one
colour in all the cells, and the legend stands in the last cell only. xlabel goes to the bottom
cells, ylabel to the left ones; the other settings to all. The outputs are the figure's.
"""

from .. import _core as sd
from .._core import _type_name
from . import _basic
from ._basic import escape, split
from ._figure import Figure, integer

_SETTINGS = ("xlim", "ylim", "xscale", "yscale", "xticks", "yticks")
_OUTPUTS = ("save", "to_svg", "to_png", "to_pdf", "scene", "size", "_repr_mimebundle_")


class Facets:
    def __init__(self, fig, frames, keys, by):
        self.figure, self._frames, self._keys, self._by = fig, frames, keys, by

    def __repr__(self):
        return f"Facets({self._by!r}: {len(self._keys)} cells; {self.figure!r})"

    def _cells(self):
        return [self.figure.axes[i] for i in range(len(self._keys))]

    def __getattr__(self, name):
        fig = self.__dict__.get("figure")
        if fig is None:
            raise AttributeError(name)
        if name in _OUTPUTS or name in ("info", "width", "height", "shape", "axes"):
            return getattr(fig, name)
        if name in _SETTINGS or name == "legend":

            def setting(*args, **kw):
                for cell in self._cells():
                    getattr(cell, name)(*args, **kw)
                return self

            return setting
        method = getattr(Figure, name, None)
        if method is None or name.startswith("_") or name in ("title", "zlabel", "compare", "fit"):
            raise AttributeError(f"Facets has no method {name!r}")

        def draw(*args, **kw):
            if "data" in kw:
                raise TypeError(f"{name}(): a facet's data is its own rows; 'data' cannot be given")
            group = kw.get("group")
            order = self._orders.get(group) if isinstance(group, str) else None
            for cell, rows in zip(self._cells(), self._frames):
                cell._cell.order = order  # the colours of the groups, the same in every cell
                try:
                    getattr(cell, name)(*args, data=rows, **kw)
                finally:
                    cell._cell.order = None
            self._legend_once()
            return self

        draw.__name__ = name
        return draw

    def _legend_once(self):
        cells = self._cells()
        for cell in cells[:-1]:
            if cell._cell.legend["loc"] == "best":
                cell._cell.legend = dict(cell._cell.legend, loc=None)

    def xlabel(self, text, /):
        nr, nc = self.figure.shape
        n = len(self._keys)
        for i, cell in enumerate(self._cells()):
            if i + nc >= n:  # no cell below it
                cell.xlabel(text)
        return self

    def ylabel(self, text, /):
        nc = self.figure.shape[1]
        for i, cell in enumerate(self._cells()):
            if i % nc == 0:
                cell.ylabel(text)
        return self


def facet(
    data,
    by,
    /,
    *,
    ncols=None,
    sharex=True,
    sharey=True,
    width=None,
    height=None,
    font="serif",
    theme="paper",
    palette="jco",
):
    """A grid of cells, one per value of the column `by` of a DataFrame (module docstring)."""
    who = "facet"
    from .. import frame

    if not isinstance(data, frame.DataFrame):
        raise TypeError(f"{who}(): 'data' must be a DataFrame, not {_type_name(data)}")
    if not isinstance(by, str):
        raise TypeError(f"{who}(): 'by' must be a column name, not {_type_name(by)}")
    n = data.shape[0]
    parts = split(who, by, data, n)
    keys = [k for k, _ in parts]
    if all(isinstance(k, (int, float)) and not isinstance(k, bool) for k in keys):
        parts = sorted(parts, key=lambda p: p[0])
        keys = [k for k, _ in parts]
    k = len(keys)
    if k == 0:
        raise ValueError(f"{who}(): the column {by!r} has no value")
    nc = min(k, 3) if ncols is None else integer(ncols, who, "ncols", 1)
    nr = -(-k // nc)
    share = tuple(_basic._share_arg(who, name, v) for name, v in (("sharex", sharex), ("sharey", sharey)))
    fig = object.__new__(Figure)
    fig._init(who, 183.0 if width is None else width, height, nr, nc, font, theme, palette)
    fig._share = tuple("all" if s is True else s for s in share)
    frames = []
    column = data[by]
    for key, cell in zip(keys, fig._cells):
        frames.append(data[column == key])
        cell.title = escape(key)
    for cell in fig._cells[k:]:
        cell.frame = False
    # a group keeps its colour in every cell: the order of all the groups of each column is kept
    # by the cells (Cell.order) when a plotting method names one
    order = {}
    for column in data.columns:
        if data[column].dtype == sd.string:
            order[column] = [key for key, _ in split(who, column, data, n)]
    out = Facets(fig, frames, keys, by)
    out._orders = order
    return out
