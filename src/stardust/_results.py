"""Named result tuples of every area, in one module (DESIGN.md §9.7). Each is
named after its function: <FunctionName>Result in CamelCase, with the Array
API's spellings (UniqueAllResult, EighResult, QRResult, SVDResult, ...).

Field names follow the Array API where it has the function
(docs/api/array.md §8.3, docs/api/linalg.md §1.6)."""

from collections import namedtuple

from . import _calls, _core
from . import _summary as _S

# array: unique_* (built by csrc/array/ops/sort.cpp, which looks them up by name)
UniqueAllResult = namedtuple("UniqueAllResult", ["values", "indices", "inverse_indices", "counts"])
UniqueCountsResult = namedtuple("UniqueCountsResult", ["values", "counts"])
UniqueInverseResult = namedtuple("UniqueInverseResult", ["values", "inverse_indices"])

# array: finfo / iinfo (built by csrc/array/ops/dtype_info.cpp, by name); the Array API's attributes
FinfoResult = namedtuple("FinfoResult", ["bits", "eps", "max", "min", "smallest_normal", "dtype"])
IinfoResult = namedtuple("IinfoResult", ["bits", "max", "min", "dtype"])
# array: histogram (csrc/array/ops/histogram.cpp, by name); numpy's (hist, bin_edges) order
HistogramResult = namedtuple("HistogramResult", ["hist", "bin_edges"])

# linalg (built by the C++ core, which receives them in LINALG)
SlogdetResult = namedtuple("SlogdetResult", ["sign", "logabsdet"])
LUResult = namedtuple("LUResult", ["L", "U", "perm"])
QRResult = namedtuple("QRResult", ["Q", "R"])
QRPivotedResult = namedtuple("QRPivotedResult", ["Q", "R", "perm"])
LstsqResult = namedtuple("LstsqResult", ["solution", "rank", "singular_values"])
EighResult = namedtuple("EighResult", ["eigenvalues", "eigenvectors"])
SVDResult = namedtuple("SVDResult", ["U", "S", "Vh"])
EigResult = namedtuple("EigResult", ["eigenvalues", "eigenvectors"])
SchurResult = namedtuple("SchurResult", ["T", "Z"])
HessenbergResult = namedtuple("HessenbergResult", ["H", "Q"])
LDLResult = namedtuple("LDLResult", ["L", "D", "perm"])
BandwidthResult = namedtuple("BandwidthResult", ["lower", "upper"])

# special: airy (built by csrc/bind/special_bind.cpp, which looks it up by name)
AiryResult = namedtuple("AiryResult", ["ai", "aip", "bi", "bip"])

# regression: the linear models' diagnostics (docs/api/regression.md §2; statsmodels' het_breuschpagan
# fields and order). Jarque-Bera's result is sd.htest's class (G165, htest H18, H39). The three
# tuples below have summary() (G191, G192): what it shows beyond the fields is in _info, set by
# the function that makes the tuple (no slots, so the tuple has a __dict__ for it).


def _tuple_repr(t, first):
    """The one-line repr of a regression tuple (API.md §2.3, regression G194): its fields as the
    test results' (stardust._summary.one_line), a batch (first, its first field, an array) its shape."""
    name = type(t).__name__
    shape = tuple(first.shape) if isinstance(first, _core.ndarray) else ()
    if shape:
        return f"{name}(batch shape {shape})"
    return _S.one_line(name, [(k, float(v)) for k, v in zip(t._fields, t)])


class BreuschPaganResult(namedtuple("BreuschPaganResult", ["lm", "lm_pvalue", "fvalue", "f_pvalue"])):
    def __repr__(self):
        return _tuple_repr(self, self.lm)

    @_calls.method("summary")
    def summary(self):
        """The test as a text table (G191): LM and F statistics with their p-values."""
        from . import _summary

        info = getattr(self, "_info", {})
        batch = tuple(self.lm.shape) if hasattr(self.lm, "shape") else ()
        title = info.get("title", "Breusch-Pagan test for heteroscedasticity")
        return _summary.test_box(title, list(zip(self._fields, self)), batch, info)


# regression: theilslopes, siegelslopes (scipy.stats' fields and order, docs/api/regression.md §5)
class TheilslopesResult(namedtuple("TheilslopesResult", ["slope", "intercept", "low_slope", "high_slope"])):
    def __repr__(self):
        return _tuple_repr(self, self.slope)

    @_calls.method("summary")
    def summary(self):
        """The estimate as a text table (G192): slope and intercept, the slope's interval."""
        from . import _summary

        return _summary.slopes_box(self, "Theil-Sen Regression Results")


class SiegelslopesResult(namedtuple("SiegelslopesResult", ["slope", "intercept"])):
    def __repr__(self):
        return _tuple_repr(self, self.slope)

    @_calls.method("summary")
    def summary(self):
        """The estimate as a text table (G192): slope and intercept."""
        from . import _summary

        return _summary.slopes_box(self, "Siegel Repeated Medians Regression Results")


# The order the C++ core expects in _core.linalg._setup (csrc/linalg/pyrefs.hpp, enum Result).
LINALG = (
    SlogdetResult,
    LUResult,
    QRResult,
    QRPivotedResult,
    LstsqResult,
    EighResult,
    SVDResult,
    EigResult,
    SchurResult,
    HessenbergResult,
    LDLResult,
    BandwidthResult,
)

# stats: describe, mode (scipy.stats' fields and order, docs/api/stats.md §4)
DescribeResult = namedtuple("DescribeResult", ["nobs", "minmax", "mean", "variance", "skewness", "kurtosis"])
ModeResult = namedtuple("ModeResult", ["mode", "count"])
# stats: MultivariateNormal.cdf and MultivariateT.cdf with return_error=True (docs/api/stats.md §3)
CdfResult = namedtuple("CdfResult", ["value", "error"])


# ---- the field results: sd.optimize's OptimizeResult, sd.integrate's IntegrateResult and
# sd.differentiate's DifferentiateResult (optimize O4, O103; integrate C9). A function gives the
# fields it has; reading another raises AttributeError (so hasattr() tells). Every field is a
# read-only property and an array field a read-only array, the same at each access without a
# copy (API.md §2.2). The subclass lists its fields (_FIELDS, in the order repr and the documents
# use), says why an absent field is absent (_ABSENT), and defines its own __init__ (keyword-only;
# its call-shape errors name the class, _calls.keyword_init) and __repr__. The public
# constructor copies what the caller gave (an sd.ndarray, a numpy array, a list of numbers: an
# sd.ndarray of its own) and checks the kinds of the shared fields (integrate C121).


class _Lazy:
    """A field computed when first read: make() gives its value."""

    __slots__ = ("make",)

    def __init__(self, make):
        self.make = make


def _own(v):
    """An array field (or a tuple of them) made read-only, as the result keeps and gives it."""
    if isinstance(v, _core.ndarray):
        v._set_readonly()
    elif isinstance(v, tuple):
        v = tuple(_own(e) for e in v)
    return v


def _type_text(v):
    """The type's name as the errors give it: numpy.ndarray, sd.ndarray as ndarray."""
    t = type(v)
    return (
        t.__name__ if t.__module__ in ("builtins", "stardust._core") else f"{t.__module__.split('.')[0]}.{t.__name__}"
    )


def _as_array(v):
    """A numpy array or a list of numbers as a new sd.ndarray (a 0-d numpy array as its Python
    number), or None when v is neither."""
    if isinstance(v, list) or (type(v).__module__ == "numpy" and type(v).__name__ == "ndarray"):
        if getattr(v, "ndim", 1) == 0:
            return v.item()
        try:
            a = _core.asarray(v)
        except TypeError, ValueError, OverflowError:
            return False
        return a.copy() if not isinstance(v, list) else a  # asarray shares a numpy array's memory
    return None


def _given(cls, name, v):
    """A field the caller of the public constructor gave: copied into a read-only array of the
    result's own when it is an array or a list of numbers, its kind checked."""
    a = _as_array(v)
    if a is False:
        raise TypeError(f"{cls.__name__}(): '{name}' must be a number or an array of numbers, not {_type_text(v)}")
    if a is not None:
        v = a
    elif isinstance(v, _core.ndarray):
        v = v.copy()
    elif isinstance(v, tuple):
        return tuple(_given(cls, name, e) for e in v)
    arr = isinstance(v, _core.ndarray)
    if name in cls._TEXT:
        ok, want = isinstance(v, str), "a str"
    elif name in cls._BOOL:
        ok, want = isinstance(v, bool) or (arr and _core.isdtype(v.dtype, "bool")), "a bool or an array of bools"
    elif name in cls._INT:
        ok = (isinstance(v, int) and not isinstance(v, bool)) or (arr and _core.isdtype(v.dtype, "integral"))
        want = "an int or an array of ints"
    else:
        ok = not isinstance(v, (str, bytes, bool, dict, set))
        want = "a number or an array of numbers"
    if not ok:
        raise TypeError(f"{cls.__name__}(): '{name}' must be {want}, not {_type_text(v)}")
    return v


def _result_field(name):
    def get(self):
        try:
            v = self._fields[name]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}' ({self._ABSENT})") from None
        if isinstance(v, _Lazy):
            v = self._fields[name] = _own(v.make())
        return v

    get.__name__ = name
    return property(get, doc=f"The result's '{name}' (read-only; arrays are read-only too).")


class _FieldResult:
    """The base of the field results (the comment above)."""

    __slots__ = ("_fields", "_batch")
    _FIELDS = ()
    _ABSENT = "the function does not give it"
    # the kinds the public constructor checks (any other field: a number, an array, a tuple of
    # them, a result, None)
    _TEXT = frozenset(("message", "method"))
    _BOOL = frozenset(("converged",))
    _INT = frozenset(("status", "n_iter", "nfev", "njev", "nhev"))

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls._FIELD_SET = frozenset(cls._FIELDS)
        for name in cls._FIELDS:
            if name not in cls.__dict__:
                setattr(cls, name, _result_field(name))
        _calls.keyword_init(cls)  # "IntegrateResult(): takes ... positional arguments but 1 was given"

    @classmethod
    def _make(cls, *, batch=(), **fields):
        """The functions' constructor: their arrays are new and held by nobody else, so the
        result takes them without a copy."""
        r = cls.__new__(cls)
        r._set(batch, fields, False)
        return r

    def _set(self, batch, fields, copy):
        unknown = sorted(set(fields) - self._FIELD_SET)
        if unknown:
            raise TypeError(f"{type(self).__name__}(): unexpected keyword argument {unknown[0]!r}")
        stored = {}
        cls = type(self)
        for k in self._FIELDS:  # a fixed order, whatever the caller's
            if k in fields:
                v = _given(cls, k, fields[k]) if copy else fields[k]  # copy: the public constructor
                stored[k] = _own(v)  # the result owns its arrays, read-only
        self._fields = stored
        self._batch = tuple(batch)

    def _batch_repr(self):
        """The one-line repr of a batch: Name(batch shape (k,), j of k converged)."""
        name = type(self).__name__
        conv = self._fields.get("converged")
        if conv is None:
            return f"{name}(batch shape {self._batch})"
        n = 1
        for s in self._batch:
            n *= s
        done = int(_core.count_nonzero(conv)) if isinstance(conv, _core.ndarray) else int(bool(conv)) * n
        return f"{name}(batch shape {self._batch}, {done} of {n} converged)"

    def _fields_repr(self, keys):
        """The one-line repr of one result (API.md §2.3, stardust._summary.one_line): Name(k=v, ...)
        over the keys it has, numbers to 6 significant digits and short arrays as [v, ...]."""
        if self._batch:
            return self._batch_repr()
        return _S.one_line(type(self).__name__, [(k, self._fields[k]) for k in keys if k in self._fields])
