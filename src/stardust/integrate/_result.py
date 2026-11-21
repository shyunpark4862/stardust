"""The result object of sd.integrate (docs/api/integrate.md §1.3; decisions C9, C10 and C13 in
docs/design/integrate.md §1.1).

One class for every function, on the base it shares with sd.optimize's OptimizeResult and
sd.differentiate's DifferentiateResult (stardust._results._FieldResult): a function gives the
fields it has and reading another raises AttributeError, every field is read-only, an array
field is a read-only array given without a copy (O103). The value is `integral` everywhere
(`sum` for a series), the error estimate `error`, the count `nfev` (C13).

The quad family's results (quad, dblquad, tplquad, nquad, quad_vec) unpack into (integral,
error) like scipy's (y, abserr) tuples (C10, htest H7): a, b = r, r[0], r[1] and len(r) == 2.
The function marks that when it makes the result (_make(unpack=True)); the other results
(tanhsinh, nsum, cubature) do not unpack and are read by name.

summary() is the shared box (API.md §2.3, C20): the value, its error estimate, convergence,
status and counts in the header, the message, an array value (quad_vec, cubature) one row per
element, and quad's and quad_vec's intervals and cubature's regions as tables (largest error
first); a batch (tanhsinh, nsum) one row per element. A complex quad (complex_func=True) gives its
two real quads as the fields real and imag (C57).
"""

import math

from .. import _calls, _core
from .. import _summary as _S
from .._results import _FieldResult

# The field names, in the order repr and docs/api/integrate.md §1.3 list them.
FIELDS = (
    "integral",
    "sum",
    "error",
    "converged",
    "status",
    "message",
    "nfev",
    "max_level",
    "intervals",
    "interval_integrals",
    "interval_errors",
    "n_intervals",
    "cycle_integrals",
    "cycle_errors",
    "cycle_status",
    "n_cycles",
    "subdivisions",
    "regions",
    "region_integrals",
    "region_errors",
    "n_regions",
    "real",
    "imag",
)
_REPR = ("integral", "sum", "error", "converged", "nfev")
_PAIR = ("integral", "error")


class IntegrateResult(_FieldResult):
    """The result of an sd.integrate function: integral (sum for nsum), error, converged,
    status, nfev and the function's other fields (docs/api/integrate.md §1.3)."""

    __slots__ = ("_unpack",)
    _FIELDS = FIELDS
    _INT = _FieldResult._INT | {"max_level", "n_intervals", "cycle_status", "n_cycles", "subdivisions", "n_regions"}

    def __init__(self, *, batch=(), **fields):
        self._set(batch, fields, True)  # the caller's arrays are copied once
        self._unpack = False

    @classmethod
    def _make(cls, *, batch=(), unpack=False, **fields):
        """The functions' constructor (no copy of their new arrays); unpack: the quad family's
        results, which unpack into (integral, error) (C10)."""
        r = super()._make(batch=batch, **fields)
        r._unpack = bool(unpack) and all(k in r._fields for k in _PAIR)
        return r

    @classmethod
    def _quad(cls, fields):
        """The quad family's fast constructor: fields a dict of the function's own values in the
        order of FIELDS (Python numbers, strings, results and _Lazy arrays), a result that unpacks."""
        r = cls.__new__(cls)
        r._fields = fields
        r._batch = ()
        r._unpack = True
        return r

    def _pair(self):
        if not self._unpack:
            raise TypeError(
                "this IntegrateResult does not unpack (only the quad family's results do): read its fields by name"
            )
        return (self.integral, self.error)

    def __iter__(self):
        return iter(self._pair())

    def __len__(self):
        return len(self._pair())

    def __getitem__(self, i):
        return self._pair()[i]

    def __bool__(self):  # a result is a value, true whether it unpacks or not (len() may raise)
        return True

    @_calls.method("summary")
    def summary(self):
        """The result as a text table (C20): the value, its error estimate, convergence, status and
        counts, the message, an array value's elements, quad's and quad_vec's intervals and
        cubature's regions; a batch one row per element."""
        return _summary(self)

    def __repr__(self):
        if not self._batch and any(isinstance(self._fields.get(k), complex) for k in _REPR):
            parts = [f"{k}={_complex_text(v, 6)}" if isinstance(v, complex) else f"{k}={_S.compact(v)}"
                     for k in _REPR if k in self._fields for v in (self._fields[k],)]  # fmt: skip
            return f"{type(self).__name__}(" + ", ".join(parts) + ")"
        return self._fields_repr(_REPR)


def _complex_text(v, digits):
    """A complex value as (re+imj) with `digits` significant digits in each part."""

    def part(x):
        return f"{x:.{digits}g}" if math.isfinite(x) else repr(x)

    im = part(v.imag)
    return f"({part(v.real)}{'' if im.startswith('-') else '+'}{im}j)"


def _value_text(v, spec):
    if isinstance(v, complex):
        return _complex_text(v, 6 if spec == "#.6g" else 4)
    if isinstance(v, _core.ndarray):  # an array value: its table follows
        return f"shape {_S.shape_text(tuple(v.shape))}"
    return f"{float(v):{spec}}"


def _point(v):
    """A region's corner as (x, y, ...) with 6 significant digits (at most 6 values)."""
    if v.shape[0] > 6:
        return f"<{v.shape[0]} values>"
    return "(" + ", ".join(f"{t:.6g}" for t in v.tolist()) + ")"


def _values_table(value, error):
    """An array value (and error estimate of the same shape) as a table, one row per element."""
    shape = tuple(value.shape)
    n = _S.lanes(shape)
    vals = value.reshape((n,)).tolist()
    errs = error.reshape((n,)).tolist() if isinstance(error, _core.ndarray) else None
    rows = []
    for i in _S.shown(n):
        if i is None:
            rows.append(None)
            continue
        rows.append((_S.lane_label(i, shape), [_S.forg(vals[i], 6)] + ([_S.forg(errs[i])] if errs else [])))
    return _S.Table(["integral"] + (["error"] if errs else []), rows)


_TITLE = "Integration Results"
# the columns of a batch's table: field, formatter
_COLUMNS = (
    ("integral", lambda v: _S.forg(v, 6)),
    ("sum", lambda v: _S.forg(v, 6)),
    ("error", _S.forg),
    ("converged", _S.flag),
    ("status", str),
    ("nfev", str),
    ("max_level", str),
)
_COUNTS = (("nfev", "Function Evals:"), ("n_intervals", "Intervals:"), ("n_cycles", "Cycles:"),
           ("max_level", "Max Level:"), ("subdivisions", "Subdivisions:"),
           ("n_regions", "Regions:"))  # fmt: skip


def _summary(r):
    f = r._fields
    left, right = [], []
    if r._batch:
        n = _S.lanes(r._batch)
        left.append(("Batch shape:", _S.shape_text(r._batch)))
        if "converged" in f:
            conv = f["converged"]
            done = int(_core.count_nonzero(conv)) if isinstance(conv, _core.ndarray) else int(bool(conv)) * n
            right.append(("Converged:", f"{done} of {n}"))
        cols = [(k, _S.flat(f[k], n), fmt) for k, fmt in _COLUMNS if k in f]
        table = _S.Table([c[0] for c in cols], _S.lane_rows(r._batch, [(g, fm) for _, g, fm in cols]))
        return _S.render(_TITLE, header=(left, right), tables=[table])
    for k, label in (("converged", "Converged:"), ("status", "Status:")):
        if k in f:
            left.append((label, str(f[k])))
    for k, label in (("integral", "Integral:"), ("sum", "Sum:")):
        if k in f:
            right.append((label, _value_text(f[k], "#.6g")))
    if "error" in f:
        right.append(("Error estimate:", _value_text(f["error"], "#.4g")))
    tables = []
    if isinstance(f.get("integral"), _core.ndarray):  # quad_vec's and cubature's array values
        tables.append(_values_table(f["integral"], f.get("error")))
    for k, label in _COUNTS:
        if k in f:
            right.append((label, str(f[k])))
    wide = [("Message:", " ".join(str(f["message"]).split()))] if "message" in f else []
    if "intervals" in f and f["n_intervals"]:
        iv, vi, ve = r.intervals, r.interval_integrals, r.interval_errors
        one = vi.ndim == 1  # quad_vec's array values: the integrals are in the field, not the table
        rows = []
        for i in _S.shown(int(f["n_intervals"])):
            if i is None:
                rows.append(None)
                continue
            cells = [_S.forg(iv[i, 0], 6), _S.forg(iv[i, 1], 6)] + ([_S.forg(vi[i], 6)] if one else [])
            rows.append((f"[{i}]", cells + [_S.forg(ve[i])]))
        tables.append(_S.Table(["a", "b"] + (["integral"] if one else []) + ["error"], rows))
    if "regions" in f and f["n_regions"]:  # cubature's regions (corners a, b)
        rg, vi, ve = r.regions, r.region_integrals, r.region_errors
        one = vi.ndim == 1
        rows = []
        for i in _S.shown(int(f["n_regions"])):
            if i is None:
                rows.append(None)
                continue
            cells = [_point(rg[i, 0]), _point(rg[i, 1])]
            if one:
                cells += [_S.forg(vi[i], 6), _S.forg(ve[i])]
            else:
                cells.append(_S.forg(float(_core.max(_core.abs(ve[i])))))
            rows.append((f"[{i}]", cells))
        tables.append(_S.Table(["a", "b"] + (["integral", "error"] if one else ["max error"]), rows, left={0, 1}))
    if "cycle_integrals" in f and f["n_cycles"]:  # QAWF's cycles, in order
        ci, ce, cs = r.cycle_integrals, r.cycle_errors, r.cycle_status
        rows = [None if i is None else (f"[{i}]", [_S.forg(ci[i], 6), _S.forg(ce[i]), str(cs[i])])
                for i in _S.shown(int(f["n_cycles"]))]  # fmt: skip
        tables.append(_S.Table(["integral", "error", "status"], rows))
    parts = [(k, f[k]) for k in ("real", "imag") if k in f]  # a complex quad's two real quads
    if parts:
        pf = parts[0][1]._fields
        n = "n_intervals" if "n_intervals" in pf else "n_cycles"
        rows = [(k, [_S.forg(p.integral, 6), _S.forg(p.error), str(p.status), str(p.nfev), str(p._fields[n])])
                for k, p in parts]  # fmt: skip
        tables.append(_S.Table(["integral", "error", "status", "nfev", n[2:]], rows))
    return _S.render(_TITLE, header=(left, right), wide_rows=wide, tables=tables)
