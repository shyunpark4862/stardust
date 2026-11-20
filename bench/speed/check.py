"""The results check of a smoke run: does each comparator compute what sd computes?

A smoke run (python -m bench.speed run --smoke) calls every case once on each side. The
worker turns sd's result and each checked comparator's result into flat float64 or
complex128 arrays (normalise) and compares them value by value (compare):

- two values agree when both are NaN, equal, or within rtol of the larger magnitude
  (plus ATOL, so two results that underflow differently near 0 still agree);
- a value only one side gives as NaN is not a disagreement: the NaN shares of both sides
  are reported, and the tables note a comparator whose share is above sd's (nan_note);
- the check fails when fewer than `agree` of the values both sides give agree, or the
  shapes differ, or a result cannot be compared.

`agree` is a share, not all: the special and stats inputs lean on the hard corners of a
function, where a comparator may lose its accuracy; a comparator that computes another
function (a wrong argument order, a rounded argument) disagrees almost everywhere.

Which comparators are checked: the first Python one, and every R and Boost one; never one
with `differs` set. A case's Check changes the normalisers, the tolerance, or skips the
check with a reason.
"""

from __future__ import annotations

import numpy as np

# A comparator whose NaN share is this much above sd's gets a note in the tables.
NAN_NOTE_MIN = 0.01
# Values this close agree whatever their size: products and powers that underflow to
# subnormals or to 0 in another order (prod over 10^6 numbers below 1: 0.0 against 5e-324).
ATOL = 1e-300


def nan_share(value) -> float | None:
    """The share of NaN in a numeric result, or None when it is not numeric."""
    try:
        a = flat(normalise(value))
    except Exception:  # noqa: BLE001  (a result we cannot read has no share)
        return None
    if a is None or a.size == 0:
        return None
    return float(np.isnan(a).mean())


# The field of a solver's or a fit's result that holds the answer, in the order tried: R's
# fits (coefficients; before x, which an R fit holds as its model matrix), sd's and scipy's
# (x, root, params, bracket, statistic, coef_) and R's solvers (par, root, minimum).
FIELDS = ("coefficients", "x", "root", "par", "minimum", "params", "bracket", "statistic", "coef_")


def normalise(x, depth: int = 0):
    """A result as numpy arrays (or a list of them): arrays, scalars, frames and series,
    the answer field of solver and fit results (FIELDS), tuples and named tuples. Anything
    else becomes ("opaque", type name)."""
    if depth > 4:
        return ("opaque", type(x).__name__)
    if x is None:
        return ("opaque", "None")
    if isinstance(x, (bool, int, float, complex, np.number, np.bool_)):
        return np.asarray(x)
    if isinstance(x, (str, bytes)):
        return ("opaque", "str")
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, dict):
        for k in FIELDS:
            if k in x:
                return normalise(x[k], depth + 1)
        return ("opaque", "dict")
    if hasattr(x, "to_numpy") and hasattr(x, "index"):  # frames and series (pandas, sd.frame)
        try:
            return frame_columns(x)
        except Exception:  # noqa: BLE001
            return ("opaque", type(x).__name__)
    for attr in FIELDS:
        named = hasattr(x, "_fields")
        if (named and attr in x._fields) or (not named and not isinstance(x, (tuple, list)) and hasattr(x, attr)):
            try:
                return normalise(getattr(x, attr), depth + 1)
            except Exception:  # noqa: BLE001
                pass
    if hasattr(x, "_fields"):  # a named tuple
        return [normalise(getattr(x, f), depth + 1) for f in x._fields]
    if isinstance(x, (tuple, list)):
        try:
            a = np.asarray(x)
            if a.dtype.kind in "biufc":
                return a
        except Exception:  # noqa: BLE001
            pass
        return [normalise(e, depth + 1) for e in x]
    try:
        a = np.asarray(x)
    except Exception:  # noqa: BLE001
        return ("opaque", type(x).__name__)
    return a if a.dtype != object else ("opaque", type(x).__name__)


def frame_columns(x) -> list:
    """A frame or series as its columns, each a float64 array: strings by a hash of their
    text (crc32), NA as NaN. A named index (a pandas groupby or value_counts result) becomes
    the leading columns, as sd.frame returns its keys as columns; an unnamed one (a sort's
    permutation) is left out."""
    idx = x.index
    names = list(getattr(idx, "names", None) or [getattr(idx, "name", None)])
    if any(n is not None for n in names) and hasattr(x, "reset_index"):
        x = x.reset_index()
    if hasattr(x, "columns"):
        cols = [x[c] for c in list(x.columns)]
    else:
        cols = [x]
    return [_column(c) for c in cols]


def _column(c) -> np.ndarray:
    a = None
    for kw in ({"dtype": np.float64, "na_value": np.nan}, {"na_value": np.nan}, {"na_value": None}, {}):
        try:  # pandas takes a numpy dtype; sd.frame takes na_value with its own dtype
            a = np.asarray(c.to_numpy(**kw))
            break
        except TypeError, ValueError:
            continue
    if a is None:
        raise ValueError("a column without a numpy form")
    if a.dtype.kind in "biuf":
        return a.astype(np.float64)
    if a.dtype.kind == "M":
        return a.astype("int64").astype(np.float64)
    import zlib

    out = np.empty(a.size)
    for i, v in enumerate(a.ravel()):
        if v is None or (isinstance(v, float) and v != v) or v is getattr(np, "nan", None):
            out[i] = np.nan
        elif isinstance(v, (bool, int, float, np.number)):
            out[i] = float(v)
        else:
            try:
                import pandas as pd

                if v is pd.NA:
                    out[i] = np.nan
                    continue
            except ImportError:
                pass
            out[i] = float(zlib.crc32(str(v).encode()))
    return out


def flat(v) -> np.ndarray | None:
    """A normalised result as one flat float64 or complex128 array (None: not numeric)."""
    if isinstance(v, list):
        parts = [flat(e) for e in v]
        if any(p is None for p in parts):
            return None
        return np.concatenate(parts) if parts else np.zeros(0)
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "biuf":
            return v.astype(np.float64).ravel()
        if v.dtype.kind == "c":
            return v.astype(np.complex128).ravel()
        if v.dtype.kind == "M":
            return v.astype("int64").astype(np.float64).ravel()
    return None


def describe(v) -> str:
    if isinstance(v, np.ndarray):
        return f"{v.dtype}{v.shape}"
    if isinstance(v, list):
        return "[" + ", ".join(describe(e) for e in v[:4]) + (", ..." if len(v) > 4 else "") + "]"
    if isinstance(v, tuple) and v and v[0] == "opaque":
        return v[1]
    return type(v).__name__


def compare(a, b, *, rtol: float = 1e-6, atol: float | None = None, agree: float = 0.5) -> dict:
    """Compare two normalised results. Returns {"ok", "text", "n", "nan_a", "nan_b",
    "agree"}: ok is False when the check fails, text says why or how well they agree."""
    fa, fb = flat(a), flat(b)
    out = {"ok": False, "n": 0, "nan_a": None, "nan_b": None, "agree": None}
    if fa is None or fb is None:
        out["text"] = f"cannot compare {describe(a)} with {describe(b)}"
        return out
    if fa.shape != fb.shape:
        out["text"] = f"{fa.size} values against {fb.size} ({describe(a)}, {describe(b)})"
        return out
    out["n"] = n = fa.size
    if n == 0:
        out.update(ok=True, text="empty")
        return out
    if np.iscomplexobj(fa) or np.iscomplexobj(fb):
        fa, fb = fa.astype(np.complex128), fb.astype(np.complex128)
    na, nb = np.isnan(fa), np.isnan(fb)
    out["nan_a"], out["nan_b"] = float(na.mean()), float(nb.mean())
    both = ~na & ~nb
    with np.errstate(all="ignore"):
        tol = rtol * np.maximum(np.abs(fa), np.abs(fb)) + (ATOL if atol is None else atol)
        close = (fa == fb) | (np.abs(fa - fb) <= tol)
    m = int(both.sum())
    share = float(close[both].mean()) if m else 1.0
    out["agree"] = share
    out["ok"] = share >= agree
    text = f"{share:.1%} of {m} agree"
    if out["nan_a"] != out["nan_b"]:
        text += f"; NaN {out['nan_a']:.1%} (sd) vs {out['nan_b']:.1%}"
    if not out["ok"]:
        i = int(np.flatnonzero(both & ~close)[0])
        text += f"; first [{i}] {fa[i]!r} vs {fb[i]!r}"
    out["text"] = text
    return out


def nan_note(label: str, nan_ref: float | None, nan_sd: float | None) -> str:
    """The table note for a comparator that gives NaN more often than sd ("" when not)."""
    if nan_ref is None or nan_sd is None or nan_ref - nan_sd < NAN_NOTE_MIN:
        return ""
    return f"{label}: NaN {nan_ref:.0%}" + (f" (sd {nan_sd:.0%})" if nan_sd >= 0.005 else "")
