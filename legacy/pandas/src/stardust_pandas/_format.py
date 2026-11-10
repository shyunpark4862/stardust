"""repr of Series and DataFrame (API.md §9).

A port of the pandas 3.1.1 repr path (pandas/io/formats/format.py and
string.py) with its default display options, minus the terminal-width
wrapping. Comments name the pandas function each piece comes from, so the
two can be read side by side.
"""

import re

from ._na import NA

MAX_ROWS = 60  # display.max_rows: more rows than this are truncated ...
MIN_ROWS = 10  # display.min_rows: ... to this many, half from each end
MAX_COLS = 20  # display.max_columns (outside a terminal)
MAX_COLWIDTH = 50  # display.max_colwidth
PRECISION = 6  # display.precision
MAX_SEQ_ITEMS = 100  # display.max_seq_items, for "Empty DataFrame" listings

_NUMBER_WITH_DECIMAL = re.compile(r"^\s*[+-]?[0-9]+\.[0-9]*$")
_ESCAPES = {"\t": r"\t", "\r": r"\r", "\n": r"\n"}


def _escape(text):  # printing._as_escaped_string with escape_chars ("\t", "\r", "\n")
    for c, rep in _ESCAPES.items():
        text = text.replace(c, rep)
    return text


# --- values -------------------------------------------------------------------


def _trim_zeros_float(strs):  # _trim_zeros_float: drop trailing zeros equally from all decimals
    is_number = [_NUMBER_WITH_DECIMAL.match(x) is not None for x in strs]
    if not any(is_number):
        return strs
    trimmed = list(strs)
    while all(x.endswith("0") for x, num in zip(trimmed, is_number) if num):
        trimmed = [x[:-1] if num else x for x, num in zip(trimmed, is_number)]
    return [x + "0" if num and x.endswith(".") else x for x, num in zip(trimmed, is_number)]


def _format_floats(values):  # FloatArrayFormatter.get_result_as_array, fixed width, leading space
    def format_all(spec):
        out = []
        for v in values:
            if v is NA:
                out.append("<NA>")  # NA is formatted like NaN, with its own text
            elif v != v:
                out.append("NaN")
            else:
                out.append(f"{v: .{PRECISION}{spec}}")
        return _trim_zeros_float(out)

    out = format_all("f")
    too_long = bool(out) and max(len(x) for x in out) > PRECISION + 6
    magnitudes = [abs(v) for v in values if v is not NA and v == v]
    has_large = any(a > 1e6 for a in magnitudes)
    has_small = any(0 < a < 10**-PRECISION for a in magnitudes)
    if has_small or (too_long and has_large):  # switch the whole column to exponent notation
        out = format_all("e")
    return out


def _format_values(values, dtype):
    """Each value as text, with pandas' leading space, before justification."""
    if dtype == "float64":
        return _format_floats(values)
    if dtype == "int64":  # _IntArrayFormatter
        return [" <NA>" if v is NA else f"{v: d}" for v in values]
    # _GenericArrayFormatter (bool, str): a leading space, then the text
    return [" <NA>" if v is NA else " " + _escape(str(v)) for v in values]


def _make_fixed_width(strings, justify, minimum=None):  # _make_fixed_width
    if not strings:
        return strings, 0
    max_len = max(len(x) for x in strings)
    if minimum is not None:
        max_len = max(minimum, max_len)
    max_len = min(max_len, MAX_COLWIDTH)
    keep = max(max_len - 3, 0)
    placeholder = "..."[: max_len - keep]
    strings = [x[:keep] + placeholder if len(x) > max_len else x for x in strings]
    return _justify(strings, max_len, justify), max_len


def _justify(strings, width, mode):
    if mode == "left":
        return [x.ljust(width) for x in strings]
    if mode == "center":
        return [x.center(width) for x in strings]
    return [x.rjust(width) for x in strings]


def _adjoin(space, *lists):  # printing.adjoin
    lengths = [max(map(len, x)) + space for x in lists[:-1]]
    lengths.append(max(map(len, lists[-1])))
    rows = max(map(len, lists))
    padded = [[" " * w] * (rows - len(lst)) + _justify(lst, w, "left") for lst, w in zip(lists, lengths)]
    return "\n".join("".join(parts) for parts in zip(*padded))


def _trim_front(strings):  # indexes.base.trim_front
    if not strings:
        return strings
    lead = min(len(x) - len(x.lstrip()) for x in strings)
    return [x[lead:] for x in strings] if lead > 0 else strings


# --- labels -------------------------------------------------------------------


def _format_labels(index, positions, truncated):
    """Row labels at `positions`, left-justified to one width."""
    labels = [index._range[p] for p in positions]
    if not truncated:  # RangeIndex._format_with_header
        width = max(len(str(index._range[0])), len(str(index._range[-1])))
        return [f"{x:<{width}}" for x in labels]
    # A truncated frame's index is no longer a range (the middle is cut out),
    # so pandas formats it as an int64 Index: Index._format_with_header.
    strings, _ = _make_fixed_width([f"{x: d}" for x in labels], "left")
    return _trim_front(strings)


def _pprint_seq(items):  # printing._pprint_seq, for the "Empty DataFrame" lines (no escaping)
    shown = [str(x) for x in items[:MAX_SEQ_ITEMS]]
    body = ", ".join(shown)
    if len(items) > MAX_SEQ_ITEMS:
        body += ", ..."
    return f"[{body}]"


def _row_positions(n):
    """The rows to show, and where the cut is (None when nothing is cut)."""
    if n <= MAX_ROWS:
        return list(range(n)), None
    half = min(MIN_ROWS, MAX_ROWS) // 2
    return list(range(half)) + list(range(n - half, n)), half


def _values_at(col, positions):
    if not positions:
        return []
    if positions[-1] - positions[0] + 1 == len(positions):
        return col.slice(positions[0], len(positions)).to_list(NA)
    half = len(positions) // 2  # head and tail
    return col.slice(positions[0], half).to_list(NA) + col.slice(positions[half], half).to_list(NA)


# --- Series -------------------------------------------------------------------


def series_repr(s):  # SeriesFormatter.to_string
    col, n = s._col, len(s._col)
    positions, cut = _row_positions(n)
    footer = []
    if s.name is not None:
        footer.append(f"Name: {_escape(s.name)}")
    if cut is not None:
        footer.append(f"Length: {n}")
    footer.append(f"dtype: {col.dtype}")
    footer = ", ".join(footer)
    if n == 0:
        return f"Series([], {footer})"

    labels = _format_labels(s.index, positions, cut is not None)
    values, _ = _make_fixed_width(_format_values(_values_at(col, positions), col.dtype), "right")
    if cut is not None:
        width = len(values[cut - 1])
        values.insert(cut, _justify(["..." if width > 3 else ".."], width, "center")[0])
        labels.insert(cut, "")
    return _adjoin(3, labels, values) + "\n" + footer


# --- DataFrame ----------------------------------------------------------------


def frame_repr(df):  # StringFormatter.to_string over DataFrameFormatter
    names = list(df._cols)
    n = len(df)
    dimensions = f"\n\n[{n} rows x {len(names)} columns]"
    truncated = n > MAX_ROWS or len(names) > MAX_COLS  # show_dimensions="truncate"
    if n == 0 or not names:
        text = f"Empty DataFrame\nColumns: {_pprint_seq(names)}\nIndex: {_pprint_seq(df.index.to_list())}"
        return text + dimensions if truncated else text

    col_cut = None
    if len(names) > MAX_COLS:  # _truncate_horizontally
        col_cut = MAX_COLS // 2
        names = names[:col_cut] + names[-col_cut:]
    positions, row_cut = _row_positions(n)  # _truncate_vertically

    # The index column: a blank header line, then the labels.
    labels, _ = _make_fixed_width(_format_labels(df.index, positions, row_cut is not None), "left")
    strcols = [[""] + labels]
    # The column labels as one Index: Index._format_flat trims their common leading space.
    headers = _trim_front([" " + _escape(name) for name in names])
    for name, header in zip(names, headers):  # _get_strcols_without_index
        col = df._cols[name]
        if col.dtype != "string":  # numeric (and bool) column labels get a leading space
            header = " " + header
        values, width = _make_fixed_width(
            _format_values(_values_at(col, positions), col.dtype), "right", minimum=len(header)
        )
        width = max(width, len(header))
        strcols.append(_justify([header], width, "right") + values)

    if col_cut is not None:  # _insert_dot_separator_horizontal
        strcols.insert(col_cut + 1, [" ..."] * len(strcols[0]))
    if row_cut is not None:  # _insert_dot_separator_vertical
        for i, col in enumerate(strcols):
            width = len(col[row_cut])
            is_dot_col = col_cut is not None and i == col_cut + 1
            dots = "..." if width > 3 or is_dot_col else ".."
            if i == 0:
                mode = "left"
            else:
                mode = "right"
                if is_dot_col:
                    width = 4
            col.insert(row_cut + 1, _justify([dots], width, mode)[0])

    text = _adjoin(1, *strcols)
    return text + dimensions if truncated else text
