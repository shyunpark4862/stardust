"""The result objects of sd.htest (docs/api/htest.md §1.5; decisions H6, H7, H10, H12, H16, H18,
H19 and H21 in docs/design/htest.md §1.1).

One class per test, named <Function>Result (TtestIndResult, PearsonrResult), on the common base
_TestResult. The classes are not exported (H6), except the one class two areas return for the
same test (H18: Jarque-Bera's, public as sd.regression.JarqueBeraResult). The rules are those
of the other areas' results (regression G98, G120; optimize's OptimizeResult; API.md §2.2): every
field is a read-only property, an array field is a read-only array given without a copy (H21),
and a field the test does not give for this call (wilcoxon's zstatistic outside the normal
approximation) raises AttributeError, so hasattr() tells.

A test with one sample (or one table) gives Python numbers (counts as int, df as float); a batch
gives arrays of the batch shape. A class unpacks like scipy's result exactly where scipy's does
(H7: _UNPACK, usually statistic and pvalue; chi2_contingency four); elsewhere it does not
iterate. Where scipy's result has a confidence interval, the class has regression's method
conf_int(*, alpha=0.05) (H16, regression G99: _CI): a new (..., 2) array of the lower and upper
ends of the 1 - alpha interval.

A subclass lists its fields and the rest is built here:

    class TtestIndResult(_TestResult):
        __slots__ = ()
        _FIELDS = ("statistic", "pvalue", "df")
        _UNPACK = ("statistic", "pvalue")
        _CI = True

and the test returns TtestIndResult._make(statistic=..., pvalue=..., df=..., ci=fn), fn(alpha)
giving the 1 - alpha interval's (low, high).
"""

import math

from .. import _core as _sd
from . import _common


def _own(v, copy=False):
    """A field as the result keeps and gives it: an sd array made read-only in place (first
    copied when copy: an array the caller may still hold); a tuple's arrays too."""
    if isinstance(v, _sd.ndarray):
        if copy:
            v = v.copy()
        v._set_readonly()
    elif isinstance(v, tuple) and not hasattr(v, "_fields"):
        v = tuple(_own(e, copy) for e in v)
    elif isinstance(v, tuple):  # a namedtuple field keeps its type
        v = type(v)(*(_own(e, copy) for e in v))
    return v


def _field(name):
    def get(self):
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}' (the test does not give it for this call)"
            ) from None

    get.__name__ = name
    return property(get, doc=f"The test's '{name}' (read-only; an array field is a read-only array).")


def _iter(self):
    return iter(tuple(getattr(self, f) for f in self._UNPACK))


def _len(self):
    return len(self._UNPACK)


def _getitem(self, i):
    return tuple(getattr(self, f) for f in self._UNPACK)[i]


def conf_int(self, *, alpha=0.05):
    """The 1 - alpha confidence interval (alpha keyword-only, between 0 and 1; regression's
    conf_int, G99): a new (..., 2) float64 array of the lower and upper ends, (2,) for one test.
    A new array at each call, writable as regression's conf_int() (G120)."""
    a = _common.alpha(alpha)
    low, high = self._ci(a)
    if isinstance(low, _sd.ndarray) or isinstance(high, _sd.ndarray):
        return _sd.stack([_sd.asarray(low) + 0.0, _sd.asarray(high) + 0.0], axis=-1)
    return _sd.asarray([float(low), float(high)])


class _TestResult:
    """The base of every sd.htest result class (docs/api/htest.md §1.5)."""

    __slots__ = ("_values", "_batch", "_ci")
    _FIELDS = ("statistic", "pvalue")  # every field, in the order repr and the documents list them
    _UNPACK = ()  # the fields a tuple unpacking gives, scipy's order (H7); () does not unpack
    _CI = False  # has conf_int() (H16)

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        for f in cls._FIELDS:
            if not isinstance(cls.__dict__.get(f), property):
                setattr(cls, f, _field(f))
        unknown = set(cls._UNPACK) - set(cls._FIELDS)
        if unknown:
            raise TypeError(f"{cls.__name__}: _UNPACK names fields it does not have: {sorted(unknown)}")
        if cls._UNPACK:
            cls.__iter__, cls.__len__, cls.__getitem__ = _iter, _len, _getitem
        if cls._CI:
            cls.conf_int = conf_int

    def __init__(self, *, batch=(), ci=None, **fields):
        """The caller's arrays are copied once (the result owns its fields)."""
        self._set(batch, ci, fields, True)

    @classmethod
    def _make(cls, *, batch=(), ci=None, **fields):
        """The tests' constructor: their arrays are new and held by nobody else, so the result
        takes them without a copy. ci(c) gives the interval's (low, high) at level c."""
        r = cls.__new__(cls)
        r._set(batch, ci, fields, False)
        return r

    def _set(self, batch, ci, fields, copy):
        name = type(self).__name__
        unknown = sorted(set(fields) - set(self._FIELDS))
        if unknown:
            raise TypeError(f"{name}(): unexpected keyword argument {unknown[0]!r}")
        if self._CI and ci is None:
            raise TypeError(f"{name}(): 'ci' is required (the class has conf_int())")
        self._values = {k: _own(fields[k], copy) for k in self._FIELDS if k in fields}
        self._batch = tuple(batch)
        self._ci = ci

    def __repr__(self):
        name = type(self).__name__
        if self._batch:
            return f"{name}(batch shape {self._batch})"
        parts = []
        for k in self._FIELDS:
            if k not in self._values:
                continue
            v = self._values[k]
            if isinstance(v, bool) or isinstance(v, int) or isinstance(v, str):
                parts.append(f"{k}={v!r}")
            elif isinstance(v, float):
                parts.append(f"{k}={_num(v)}")
        return f"{name}(" + ", ".join(parts) + ")"


def _num(v):
    return f"{v:.6g}" if math.isfinite(v) else repr(v)
