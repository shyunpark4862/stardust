"""Gower's distance for mixed data (docs/api/cluster.md §6; docs/design/cluster.md §2.6; decisions CL38 and those of
milestone CL5 in §1.1): R cluster's daisy(metric = "gower").

The distance of two rows is sum_j w_j delta_j / sum_j w_j over the variables j that both rows have (a missing value
leaves the variable out of that pair, and the weights are those of the variables left): an interval variable's
delta is |x - y| / (its range over the rows; 1 when it is 0), a nominal variable's and a symmetric binary one's is
[x != y], and an asymmetric binary variable leaves out the pairs where both are 0. Columns of numbers are interval,
of strings nominal, of bools symmetric binary (R makes a logical column asymmetric, with a warning); `types` names
the columns to read otherwise: 'asymm', 'symm' (values 0/1 or bools), 'ordratio' (the dense ranks of the values,
then interval: R's ordered factor is its integer codes as an interval column) or 'logratio' (log10 of positive
values, then interval). The result is the condensed vector of the pairs (i < j, scipy's order), ready for
linkage(y), kmedoids(..., metric='precomputed') and the other functions that take distances.
"""

import math

from .. import _core as _sd
from .._calls import function
from ._common import _array, check_alloc, contiguous, real, type_name

_TYPES = ("asymm", "symm", "ordratio", "logratio")
_INTERVAL, _NOMINAL, _SYMM, _ASYMM = 0, 1, 2, 3


def _column_values(col, kind, func, name):
    """(float values with NaN for missing, the column's type code)."""
    if kind == "string":
        vals = col.tolist()
        miss = [v is None or (not isinstance(v, str)) for v in vals]
        uniq = sorted({v for v, m in zip(vals, miss) if not m})
        code = {u: float(i) for i, u in enumerate(uniq)}
        return [math.nan if m else code[v] for v, m in zip(vals, miss)], uniq
    try:
        a = col.to_array(dtype=_sd.float64, na_value=math.nan)
    except TypeError, ValueError:
        raise TypeError(f"{func}(): column {name!r} must hold numbers, bools or strings") from None
    return a.tolist(), None


def _dense_ranks(vals):
    uniq = sorted({v for v in vals if not (isinstance(v, float) and math.isnan(v))})
    rank = {u: float(i + 1) for i, u in enumerate(uniq)}
    return [math.nan if isinstance(v, float) and math.isnan(v) else rank[v] for v in vals]


@function
def gower_distance(data, /, *, types=None, weights=None):
    """Gower's distance of the rows of mixed data (R cluster's daisy(metric='gower')): a condensed (n (n - 1) / 2,)
    float64 vector in scipy's pair order; NaN for a pair with no variable in common.

    data: a frame DataFrame (number columns interval, string columns nominal, bool columns symmetric binary;
    missing values left out pair by pair) or a 2-D array of numbers (every column interval). types: a dict from
    column (its name; for an array its position) to 'asymm', 'symm', 'ordratio' or 'logratio'. weights: one
    weight >= 0 per column (default 1 each).
    """
    func = "gower_distance"
    cols, names = [], []
    if type(data).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(data, _frame.DataFrame):
            raise TypeError(
                f"{func}(): 'data' must be a frame DataFrame or a 2-D array of numbers, not {type_name(data)}"
            )
        for c in data.columns:
            s = data[c]
            kind = str(s.dtype)
            vals, levels = _column_values(s, kind, func, c)
            base = {"string": _NOMINAL, "bool": _SYMM}.get(kind, _INTERVAL)
            cols.append([vals, base, kind])
            names.append(c)
    else:
        a = _array(data, func, "data")
        if a.ndim != 2:
            raise ValueError(f"{func}(): an array 'data' must be 2-D, not {a.ndim}-D")
        n, p = a.shape
        for j in range(p):
            cols.append([contiguous(a[:, j]).tolist(), _INTERVAL, "float64"])
            names.append(j)
    p = len(cols)
    n = len(cols[0][0]) if p else 0
    if p == 0:
        raise ValueError(f"{func}(): 'data' must have at least 1 column")
    if n == 0:
        raise ValueError(f"{func}(): 'data' must have at least 1 row")
    if types is not None:
        if not isinstance(types, dict):
            raise TypeError(f"{func}(): 'types' must be a dict or None, not {type_name(types)}")
        for col, t in types.items():
            if col not in names:
                raise ValueError(f"{func}(): 'types' names a column {col!r} that data does not have")
            if not isinstance(t, str):
                raise TypeError(
                    f"{func}(): the types must be 'asymm', 'symm', 'ordratio' or 'logratio', not {type_name(t)}"
                )
            if t not in _TYPES:
                raise ValueError(f"{func}(): the types must be 'asymm', 'symm', 'ordratio' or 'logratio', not {t!r}")
            entry = cols[names.index(col)]
            vals, _, kind = entry
            if t in ("asymm", "symm"):
                if kind == "string" or any(not (math.isnan(v) or v in (0.0, 1.0)) for v in vals):
                    raise ValueError(
                        f"{func}(): the binary column {col!r} must hold 0, 1 (or bools) and missing values"
                    )
                entry[1] = _ASYMM if t == "asymm" else _SYMM
            elif t == "ordratio":
                if kind == "string":
                    sv = data[col].tolist()
                    entry[0] = _dense_ranks([math.nan if (v is None or not isinstance(v, str)) else v for v in sv])
                else:
                    entry[0] = _dense_ranks(vals)
                entry[1] = _INTERVAL
            else:
                if kind == "string":
                    raise ValueError(f"{func}(): the logratio column {col!r} must hold numbers")
                if any(v <= 0 for v in vals if not math.isnan(v)):
                    raise ValueError(f"{func}(): the logratio column {col!r} must hold values > 0")
                entry[0] = [v if math.isnan(v) else math.log10(v) for v in vals]
                entry[1] = _INTERVAL
    if weights is None:
        w = [1.0] * p
    else:
        wa = _array(weights, func, "weights")
        if wa.ndim != 1 or wa.shape[0] != p:
            raise ValueError(f"{func}(): 'weights' must have one value per column ({p})")
        w = [real(v, func, "weights") for v in wa.tolist()]
        if any(not (v >= 0 and math.isfinite(v)) for v in w):
            raise ValueError(f"{func}(): 'weights' must be finite and >= 0")
    V = _sd.empty((n, p), dtype=_sd.float64)
    tcode, scale = [], []
    for j, (vals, t, _) in enumerate(cols):
        col = _sd.asarray(vals, dtype=_sd.float64) if vals else _sd.empty((0,), dtype=_sd.float64)
        if bool(_sd.any(_sd.isinf(col))):
            raise ValueError(f"{func}(): column {names[j]!r} contains infinity")
        V[:, j] = col
        tcode.append(t)
        rng_ = 1.0
        if t == _INTERVAL:
            present = [v for v in vals if not math.isnan(v)]
            if present:
                r = max(present) - min(present)
                rng_ = r if r > 0 else 1.0
        scale.append(rng_)
    check_alloc(8 * (n * (n - 1) // 2), func)
    return _sd.cluster.gower(
        contiguous(V), _sd.asarray(tcode, dtype=_sd.int64), _sd.asarray(scale, dtype=_sd.float64),
        _sd.asarray(w, dtype=_sd.float64)
    )  # fmt: skip
