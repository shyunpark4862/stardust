"""The text summary() of every result object of sd.regression, sd.htest and sd.optimize: one
look for all (API.md §2.3; decisions G189-G193 in docs/design/regression.md, H74-H76 in
docs/design/htest.md, O119 in docs/design/optimize.md).

The look is statsmodels' OLS summary, the regression box (G13):

    title, centered in 78 columns
    ==============================================================================
    Label:                           value  Label:                           value
    ...   (the header: two columns of (label, value) cells 38 wide, values right-aligned)
    ==============================================================================
                     coef    std err  ...   (a table: the stub column of names, left-aligned
    ------------------------------------    and at least 10 wide, then the columns right-
    name           1.2345      0.067        aligned, 11 wide or one more than their longest
    ==============================================================================   cell)
    (more tables, a footer of pairs, each closed by a '=' rule)

    Notes:
    [1] ...

A table wider than 78 widens the rules and the notes' wrapping; the title and the pairs stay
at 78. Numbers follow statsmodels: forg (prec decimals, or prec significant digits outside
[1e-4, 1e4)) in the tables, 3 decimals for a p-value in a table, '#.4g' for a statistic and
'#.3g' for a p-value in the header. A table of more than MAX_ROWS rows (a batch's lanes, a long
x) shows EDGE rows at each end with a row of '...' between.
"""

import math
import textwrap

WIDTH = 78
HALF = (WIDTH - 2) // 2  # a header cell: label and value
CELL = 11  # a table column's least width
STUB = 9  # the stub column's least width, one more is added
MAX_ROWS = 20  # a longer table is cut ...
EDGE = 5  # ... to this many rows at each end

# ---- numbers


def forg(x, prec=3):
    """statsmodels' forg: fixed with prec decimals, or prec significant digits outside [1e-4, 1e4)."""
    x = float(x)
    if math.isfinite(x) and (abs(x) >= 1e4 or abs(x) < 1e-4) and x != 0:
        return f"{x:.{prec}g}"
    return f"{x:.{prec}f}"


def count(v):
    """A count (nobs, df_resid): its integer, 'nan' or 'inf'."""
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    return str(int(v))


def df(v):
    """Degrees of freedom: an integer as one (47), else '#.4g' (45.32, inf, nan)."""
    v = float(v)
    if math.isfinite(v) and v == math.floor(v) and abs(v) < 1e15:
        return str(int(v))
    return f"{v:#.4g}"


def stat(v):
    """A statistic in the header (the regression box's F-statistic)."""
    return f"{float(v):#.4g}"


def prob(v):
    """A p-value in the header (the regression box's Prob (F-statistic))."""
    return f"{float(v):#.3g}"


def pvalue(v):
    """A p-value in a table (the coefficient table's P>|t|)."""
    return f"{float(v):#.3f}"


def flag(v):
    return "True" if v else "False"


def quantile_label(q):
    """0.025, 0.975, 0.05, 0.95 ...: the interval's quantiles as statsmodels prints them."""
    return f"{q:.3f}" if abs(q * 1000 - round(q * 1000)) < 1e-9 else f"{q:g}"


def interval_columns(alpha, alternative="two-sided"):
    """The column names of a 1 - alpha interval: [0.025 0.975] (two-sided), [0.000 0.950]
    ('less': the lower end is -inf) or [0.050 1.000] ('greater')."""
    lo, hi = {"two-sided": (alpha / 2, 1 - alpha / 2), "less": (0.0, 1 - alpha), "greater": (alpha, 1.0)}[alternative]
    return ["[" + quantile_label(lo), quantile_label(hi) + "]"]


def sizes(n):
    """No. Observations of one or more samples: '25' or '25, 24' (up to 4 samples), else the
    total."""
    if isinstance(n, (tuple, list)):
        if len(n) <= 4:
            return ", ".join(count(v) for v in n)
        return count(sum(n))
    return count(n)


# ---- the header


def pairs(left, right):
    """Two columns of (label, value) cells, each 38 wide, values right-aligned."""
    rows = []
    for i in range(max(len(left), len(right))):
        cells = []
        for side in (left, right):
            if i < len(side):
                label, value = side[i]
                cells.append(label + value.rjust(HALF - len(label)))
            else:
                cells.append(" " * HALF)
        rows.append((cells[0] + "  " + cells[1]).rstrip())
    return rows


def wide(label, text):
    """A (label, value) row across the whole header (a long message): right-aligned when it fits
    in 78 columns, else wrapped."""
    if len(label) + 1 + len(text) <= WIDTH:
        return [label + text.rjust(WIDTH - len(label))]
    return textwrap.wrap(f"{label} {text}", WIDTH, subsequent_indent="    ")


# ---- tables


class Table:
    """columns: the column names; rows: (stub, cells) pairs or None for the '...' row; left: the
    columns aligned left (names, terms); stub_head: the stub column's name (blank by default)."""

    __slots__ = ("columns", "rows", "left", "stub_head")

    def __init__(self, columns, rows, *, left=(), stub_head=""):
        self.columns = list(columns)
        self.rows = list(rows)
        self.left = set(left)
        self.stub_head = stub_head

    def _widths(self):
        stubs = [len(self.stub_head)] + [len(r[0]) for r in self.rows if r is not None]
        wname = max(stubs + [STUB]) + 1
        widths = []
        for j, c in enumerate(self.columns):
            w = max([CELL, len(c) + 1] + [len(r[1][j]) + 1 for r in self.rows if r is not None])
            widths.append(w + (1 if j in self.left else 0))
        return wname, widths

    def width(self):
        wname, widths = self._widths()
        return wname + sum(widths)

    def lines(self):
        wname, widths = self._widths()

        def line(stub, cells):
            out = stub.ljust(wname)
            for j, (c, w) in enumerate(zip(cells, widths)):
                out += ("  " + c.ljust(w - 2)) if j in self.left else c.rjust(w)
            return out.rstrip()

        out = [line(self.stub_head, self.columns)]
        body = []
        for r in self.rows:
            body.append(line("...", ["..."] * len(self.columns)) if r is None else line(r[0], r[1]))
        return out, body


def render(title, *, header=None, wide_rows=(), tables=(), footer=None, notes=()):
    """The box: header (left, right) pairs and wide rows, tables, footer (left, right) pairs
    and notes."""
    width = max([WIDTH] + [t.width() for t in tables])
    rule = "=" * width
    lines = [title.center(WIDTH).rstrip(), rule]
    if header is not None and (header[0] or header[1]):
        lines += pairs(*header)
    for label, text in wide_rows:
        lines += wide(label, text)
    if len(lines) > 2:
        lines.append(rule)
    for t in tables:
        head, body = t.lines()
        lines += head
        lines.append("-" * width)
        lines += body
        lines.append(rule)
    if footer is not None:
        lines += pairs(*footer)
        lines.append(rule)
    if notes:
        lines.append("")
        lines.append("Notes:")
        for i, t in enumerate(notes):
            lines += textwrap.wrap(f"[{i + 1}] {t}", width, subsequent_indent="    ")
    return "\n".join(lines)


# ---- long tables and batches


def shown(n):
    """The row indices a table of n rows shows: all of them, or EDGE at each end with None (the
    '...' row) between."""
    if n <= MAX_ROWS:
        return list(range(n))
    return list(range(EDGE)) + [None] + list(range(n - EDGE, n))


def lane_label(i, shape):
    """The index of lane i (C order) in a batch of `shape`: '[3]', '[1, 2]'."""
    idx = []
    for s in reversed(shape):
        idx.append(i % s if s else 0)
        i = i // s if s else 0
    return "[" + ", ".join(str(v) for v in reversed(idx)) + "]"


def lanes(shape):
    return math.prod(shape)


def flat(v, n):
    """A per-lane field (an sd array of the batch's shape, or a number the lanes share) as a
    function of the lane."""
    from . import _core as _sd

    if isinstance(v, _sd.ndarray):
        f = _sd.reshape(v, (n,)) if v.ndim != 1 else v
        return lambda i: f[i]
    if isinstance(v, list):  # a batch's per-lane names, nested in the batch's shape
        items = _flatten(v)
        return lambda i: items[i]
    return lambda i: v


def _flatten(v):
    if isinstance(v, list):
        return [e for x in v for e in _flatten(x)]
    return [v]


def shared(v, n):
    """The value every lane of a batch has for a per-lane field (NaNs count as equal), else None.
    The rule of every batch summary (API.md §2.3, regression G195): a degrees of freedom the
    lanes share goes in the header, one that varies is a column of the table."""
    from . import _core as _sd

    if n == 0:
        return None
    if isinstance(v, _sd.ndarray):
        vals = _sd.reshape(v, (n,)).tolist() if v.size == n else None
    elif isinstance(v, list):
        vals = _flatten(v)
    else:
        return v
    if not vals:
        return None
    first = vals[0]
    for x in vals[1:]:
        if not (x == first or (x != x and first != first)):
            return None
    return first


def lane_rows(shape, columns):
    """The rows of a batch's table, one per lane (cut by shown()): columns is a list of
    (per-lane getter, formatter)."""
    n = lanes(shape)
    rows = []
    for i in shown(n):
        if i is None:
            rows.append(None)
            continue
        rows.append((lane_label(i, shape), [fmt(get(i)) for get, fmt in columns]))
    return rows


def shape_text(shape):
    return "(" + ", ".join(str(s) for s in shape) + ("," if len(shape) == 1 else "") + ")"


# ---- tests: the header of their fields, the interval, a batch's lanes

_P_FIELDS = frozenset(("pvalue", "lm_pvalue", "f_pvalue"))
_DF_FIELDS = frozenset(("df", "df_num", "df_denom"))
_LABELS = {
    "statistic": "Statistic:",
    "pvalue": "P-value:",
    "df": "Df:",
    "df_num": "Df Num:",
    "df_denom": "Df Denom:",
    "fvalue": "F-statistic:",
    "f_pvalue": "F p-value:",
    "lm": "LM statistic:",
    "lm_pvalue": "LM p-value:",
    "zstatistic": "Z-statistic:",
    "statistic_location": "Statistic Location:",
    "statistic_sign": "Statistic Sign:",
}


def _label(k):
    return _LABELS.get(k, k.replace("_", " ").capitalize() + ":")


def _head(k, v):
    """A scalar field in the header: a p-value '#.3g', degrees of freedom as an integer when
    whole, an int as itself, another number '#.4g'."""
    if isinstance(v, bool):
        return flag(v)
    if isinstance(v, int):
        return str(v)
    if k in _P_FIELDS:
        return prob(v)
    if k in _DF_FIELDS:
        return df(v)
    return stat(v)


def _cell_fn(k):
    """A per-lane field in a table: p-values with 3 decimals, degrees of freedom as integers when
    whole, ints and flags as themselves, other numbers forg."""

    def cell(v):
        if isinstance(v, bool):
            return flag(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, str):
            return v
        if k in _P_FIELDS:
            return pvalue(v)
        if k in _DF_FIELDS:
            return df(v)
        return forg(v)

    return cell


def _count_cell(v):
    """An element of an array field (a table of counts): a whole number as an integer."""
    if isinstance(v, bool):
        return flag(v)
    if isinstance(v, int):
        return str(v)
    return df(v) if math.isfinite(v) and v == math.floor(v) else forg(v)


def _array_table(k, v):
    """An array field of one test (a contingency table): rows [i], columns [j] (a 1-D array is
    one column named k), long ones cut at both ends."""
    if v.ndim == 1:
        rows = [None if i is None else (f"[{i}]", [_count_cell(v[i])]) for i in shown(v.shape[0])]
        return Table([k], rows)
    if v.ndim != 2:
        return None
    cols = shown(v.shape[1])
    head = ["..." if j is None else f"[{j}]" for j in cols]
    rows = []
    for i in shown(v.shape[0]):
        if i is None:
            rows.append(None)
            continue
        rows.append((f"[{i}]", ["..." if j is None else _count_cell(v[i, j]) for j in cols]))
    return Table(head, rows, stub_head=k)


def _split(items):
    """The header's descriptors in two columns when no number fills the left one (a batch)."""
    h = (len(items) + 1) // 2
    return items[:h], items[h:]


def test_box(title, fields, batch, info, interval=None, tables=()):
    """The summary of a test result (htest H74-H76, regression G191): fields its (name, value)
    pairs in order: numbers go to the header's left column, strings (the method, the alternative
    field) to the right one with info's alternative, sample sizes ('nobs') and extra 'pairs';
    the arrays of one test (a contingency table) become tables. A batch (its shape) gives one
    row per lane of the per-lane fields. interval (low, high, alpha, label): the 1 - alpha
    interval, a row of its own (or two columns of a batch's table). tables: the names of the
    fields that are tables whatever their size (one test's table, a batch's left out:
    docs/api/htest.md §1.5); another array field is per lane when it has one value per lane."""
    from . import _core as _sd

    n = lanes(batch) if batch else 1
    alt = info.get("alternative", "two-sided")
    left, right, columns, arrays = [], [], [], []
    for k, v in fields:
        if isinstance(v, str):
            right.append((_label(k), v))
        elif isinstance(v, list):  # a batch's per-lane method names (htest H48)
            if n == 0:  # an empty batch: no lane to name (its table has no rows)
                continue
            names = flat(v, n)
            first = names(0)
            if all(names(i) == first for i in range(n)):
                right.append((_label(k), str(first)))
            else:
                columns.append((k, names, str))
        elif isinstance(v, _sd.ndarray):
            if k in tables:
                t = None if batch else _array_table(k, v)
                if t is not None:
                    arrays.append(t)
            elif v.size == n:
                get = flat(v, n)
                same = shared(v, n) if batch and k in _DF_FIELDS else None
                if same is not None:  # degrees of freedom the lanes share: the header (G195)
                    left.append((_label(k), _head(k, same)))
                elif batch:
                    columns.append((k, get, _cell_fn(k)))
                else:  # one test kept as an array (keepdims=True)
                    left.append((_label(k), _head(k, get(0))))
            elif not batch:
                t = _array_table(k, v)
                if t is not None:
                    arrays.append(t)
            # a batch's per-lane arrays (its contingency tables) are left to the fields
        elif isinstance(v, (bool, int, float)):
            left.append((_label(k), _head(k, v)))
    desc = []
    if batch:
        desc.append(("Batch shape:", shape_text(batch)))
    if "alternative" in info:
        desc.append(("Alternative:", alt))
    desc += right
    if "nobs" in info:
        nobs = info["nobs"]
        desc.append(("No. Observations:", sizes(nobs)))
        if isinstance(nobs, (tuple, list)) and len(nobs) > 4:
            desc.append(("No. Samples:", str(len(nobs))))
    desc += info.get("pairs", [])
    if not left:
        left, desc = _split(desc)
    tables = []
    if batch:
        if interval is not None:
            lo, hi, alpha, _ = interval
            names = interval_columns(alpha, alt)
            columns.append((names[0], flat(lo, n), forg))
            columns.append((names[1], flat(hi, n), forg))
        if columns:
            rows = lane_rows(batch, [(get, fmt) for _, get, fmt in columns])
            tables.append(Table([c[0] for c in columns], rows))
    else:
        if interval is not None:
            lo, hi, alpha, label = interval
            lo, hi = flat(lo, 1)(0), flat(hi, 1)(0)
            tables.append(Table(interval_columns(alpha, alt), [(label, [forg(lo), forg(hi)])]))
        tables += arrays
    return render(title, header=(left, desc), tables=tables, notes=info.get("notes", ()))


def slopes_box(r, title):
    """theilslopes' and siegelslopes' summary (regression G192): the method and the number of
    points, then slope and intercept (with Sen's interval of the slope); a batch one row per
    regression."""
    from . import _core as _sd

    info = getattr(r, "_info", {})
    ci = "low_slope" in r._fields
    alpha = info.get("alpha")
    shape = tuple(r.slope.shape) if isinstance(r.slope, _sd.ndarray) else ()
    left = [("Method:", info["method"])] if "method" in info else []
    right = []
    if shape:
        right.append(("Batch shape:", shape_text(shape)))
    if "nobs" in info:
        right.append(("No. Observations:", count(info["nobs"])))
    bounds = interval_columns(alpha) if alpha is not None else ["low", "high"]
    if not shape:
        rows = [("slope", [forg(r.slope, 4)] + ([forg(r.low_slope), forg(r.high_slope)] if ci else []))]
        rows.append(("intercept", [forg(r.intercept, 4)] + (["", ""] if ci else [])))
        table = Table(["coef"] + (bounds if ci else []), rows)
    else:
        n = lanes(shape)
        cols = [(flat(r.slope, n), lambda v: forg(v, 4)), (flat(r.intercept, n), lambda v: forg(v, 4))]
        names = ["slope", "intercept"]
        if ci:
            cols += [(flat(r.low_slope, n), forg), (flat(r.high_slope, n), forg)]
            names += bounds  # the slope's: the intercept has none
        table = Table(names, lane_rows(shape, cols))
    return render(title, header=(left, right), tables=[table])


# ---- one-line reprs


def compact(v):
    """A field's value in a one-line repr: a number with 6 significant digits, a short 1-D array
    as [v, v, ...] (at most 6 values), a longer one as its shape."""
    from . import _core as _sd

    if isinstance(v, bool) or isinstance(v, (int, str)) or v is None:
        return repr(v)
    if isinstance(v, float):
        return f"{v:.6g}" if math.isfinite(v) else repr(v)
    if isinstance(v, _sd.ndarray):
        if v.ndim == 1 and v.shape[0] <= 6:
            return "[" + ", ".join(compact(e) for e in v.tolist()) + "]"
        return f"<array of shape {shape_text(tuple(v.shape))}>"
    return None


def one_line(name, fields):
    """The one-line repr of a result (API.md §2.3, regression G194): Name(field=value, ...) with
    each value compact(); fields are (name, value) pairs, a value compact() cannot write left out."""
    parts = []
    for k, v in fields:
        text = compact(v)
        if text is not None:
            parts.append(f"{k}={text}")
    return f"{name}(" + ", ".join(parts) + ")"
