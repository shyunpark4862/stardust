"""The distribution classes' common machinery (docs/api/stats.md §2, docs/design/stats.md §2-§3).

A distribution holds its parameters as given (Python numbers or sd.ndarrays) and calls
the native elementwise functions stardust._core.stats.<name>_<method>, whose inputs are
(params..., x) or (params...): the parameters broadcast with the input at each call.
Parameters outside their domain give NaN (T8), never an error.
"""

from .. import _core as _sd_core


_k = _sd_core.stats

# Methods with an input (and out=), and the parameter-only ones (T9, T22).
_INPUT_METHODS = ("pdf", "logpdf", "cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")
_PARAM_METHODS = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy",
                  "support_lo", "support_hi")


def _check_param(name, value):
    """A parameter is a real Python number or a non-complex sd.ndarray (T19)."""
    if isinstance(value, _sd_core.ndarray):
        if value.dtype == _sd_core.complex128:
            raise TypeError(f"'{name}' must be real, not a complex128 array")
        return
    if isinstance(value, (bool, int, float)):
        return
    raise TypeError(f"'{name}' must be a real number or an sd.ndarray, not {type(value).__name__}")


def _parameter(i, name):
    def get(self):
        return self._p[i]
    get.__name__ = name
    return property(get, doc=f"The parameter {name}, as given (read-only).")


class ContinuousDistribution:
    """Base of the continuous distributions. Subclasses set _name (the native prefix) and
    _params (the parameter names in native order) and define __init__ with keyword-only
    parameters that calls _set."""

    __slots__ = ("_p",)
    _name = ""
    _params = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ops = {m: getattr(_k, f"{cls._name}_{m}") for m in _INPUT_METHODS + _PARAM_METHODS}
        cls._ops = ops
        for i, p in enumerate(cls._params):  # read-only attributes (T19)
            setattr(cls, p, _parameter(i, p))

    def _set(self, *values):
        arrays = []
        for name, v in zip(self._params, values):
            _check_param(name, v)
            if isinstance(v, _sd_core.ndarray):
                arrays.append(v)
        if len(arrays) > 1:
            _sd_core.broadcast_arrays(*arrays)  # parameters that cannot broadcast fail here (views only)
        self._p = values

    def __repr__(self):
        args = ", ".join(f"{n}={v!r}" for n, v in zip(self._params, self._p))
        return f"{type(self).__name__}({args})"

    # ---- methods with an input: elementwise, broadcast with the parameters, out= (T22)

    def pdf(self, x, /, *, out=None):
        """The probability density at x."""
        return self._ops["pdf"](*self._p, x, out=out)

    def logpdf(self, x, /, *, out=None):
        """log pdf(x), without underflow."""
        return self._ops["logpdf"](*self._p, x, out=out)

    def cdf(self, x, /, *, out=None):
        """P(X <= x)."""
        return self._ops["cdf"](*self._p, x, out=out)

    def logcdf(self, x, /, *, out=None):
        """log P(X <= x), without underflow."""
        return self._ops["logcdf"](*self._p, x, out=out)

    def ccdf(self, x, /, *, out=None):
        """P(X > x), computed directly (accurate in the upper tail, not 1 - cdf)."""
        return self._ops["ccdf"](*self._p, x, out=out)

    def logccdf(self, x, /, *, out=None):
        """log P(X > x), without underflow."""
        return self._ops["logccdf"](*self._p, x, out=out)

    def icdf(self, p, /, *, out=None):
        """The x with cdf(x) = p (the inverse of cdf)."""
        return self._ops["icdf"](*self._p, p, out=out)

    def iccdf(self, p, /, *, out=None):
        """The x with ccdf(x) = p (the inverse of ccdf)."""
        return self._ops["iccdf"](*self._p, p, out=out)

    def ilogcdf(self, logp, /, *, out=None):
        """The x with logcdf(x) = logp."""
        return self._ops["ilogcdf"](*self._p, logp, out=out)

    def ilogccdf(self, logp, /, *, out=None):
        """The x with logccdf(x) = logp."""
        return self._ops["ilogccdf"](*self._p, logp, out=out)

    # ---- parameter-only methods: the parameters' broadcast shape (a float for scalars)

    def mean(self):
        """The mean (NaN where it is undefined, inf where infinite)."""
        return self._ops["mean"](*self._p)

    def variance(self):
        """The variance."""
        return self._ops["variance"](*self._p)

    def standard_deviation(self):
        """The standard deviation."""
        return self._ops["standard_deviation"](*self._p)

    def skewness(self):
        """The skewness."""
        return self._ops["skewness"](*self._p)

    def kurtosis(self):
        """The excess kurtosis (0 for the normal distribution; T23)."""
        return self._ops["kurtosis"](*self._p)

    def median(self):
        """The median."""
        return self._ops["median"](*self._p)

    def mode(self):
        """The mode: the point of the largest density, even at an end of the support where the
        density diverges; NaN where no single point is the largest (T21)."""
        return self._ops["mode"](*self._p)

    def entropy(self):
        """The differential entropy, in nats."""
        return self._ops["entropy"](*self._p)

    def support(self):
        """(lo, hi): the ends of the support."""
        return self._ops["support_lo"](*self._p), self._ops["support_hi"](*self._p)
