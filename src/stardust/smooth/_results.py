"""The common base of sd.smooth's regression-like results (docs/api/smooth.md §1.3, docs/design/smooth.md §2.1,
§2.7; decision SM33): LowessResult, LoessResult, SmoothingSplineResult, PSplineResult and KernelRegressionResult.

A result holds its fields as given by the fitting function: arrays become read-only (API.md §2.2), a field
given as a callable is computed when first read and kept (a lazy field), and no field can be set afterwards.
summary() is the regression box of stardust._summary with a header only (API.md §2.3): the method, nobs, the
effective degrees of freedom, the scale and the subclass's tuning and criterion rows; repr is one line of
the subclass's main fields.

A subclass sets
    TITLE     the box's title ("Lowess Results")
    METHOD    the header's method name ("LOWESS")
    TUNING    (label, field) rows of the header's right column: the tuning parameters and the criterion
    REPR      the fields of the one-line repr
"""

from .. import _core as _sd
from .. import _summary

_MISSING = object()


class SmootherResult:
    """Base class: SmootherResult(**fields). Not a public constructor (docs/api/smooth.md §7)."""

    TITLE = "Smoother Results"
    METHOD = ""
    TUNING = ()
    REPR = ("nobs", "df")

    def __init__(self, **fields):
        values = {}
        for k, v in fields.items():
            if isinstance(v, _sd.ndarray):
                v._set_readonly()
            values[k] = v
        object.__setattr__(self, "_fields", values)

    def __getattr__(self, name):
        fields = object.__getattribute__(self, "_fields")
        v = fields.get(name, _MISSING)
        if v is _MISSING:
            raise AttributeError(f"{type(self).__name__} has no field {name!r}")
        if callable(v) and not isinstance(v, _sd.ndarray):
            v = v()
            if isinstance(v, _sd.ndarray):
                v._set_readonly()
            fields[name] = v
        return v

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__}: the fields are read-only")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__}: the fields are read-only")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(object.__getattribute__(self, "_fields")))

    def _cell(self, field):
        v = getattr(self, field)
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, str):
            return v
        if isinstance(v, int):
            return str(v)
        if isinstance(v, _sd.ndarray):
            return _summary.compact(v)
        return _summary.stat(v)

    def summary(self):
        """The regression box with a header only: the method, the sample, the fit's size and scale on the left,
        the tuning and criterion on the right."""
        left = [
            ("Method:", self.METHOD),
            ("No. Observations:", _summary.count(self.nobs)),
            ("Df (effective):", _summary.df(self.df)),
            ("Scale:", _summary.stat(self.scale)),
        ]
        right = [(label, self._cell(field)) for label, field in self.TUNING]
        return _summary.render(self.TITLE, header=(left, right))

    def __repr__(self):
        return _summary.one_line(type(self).__name__, [(k, getattr(self, k)) for k in self.REPR])
