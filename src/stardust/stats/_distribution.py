"""The distribution classes' common machinery (docs/api/stats.md §2, docs/design/stats.md §2-§3).

A distribution holds its parameters as given (Python numbers or sd.ndarrays) and calls
the native elementwise functions stardust._core.stats.<name>_<method>, whose inputs are
(params..., x) or (params...): the parameters broadcast with the input at each call.
Parameters outside their domain give NaN (T8), never an error.
"""

from .. import _core as _sd_core
from .. import random as _random


_k = _sd_core.stats

# Methods with an input (and out=) besides the density, and the parameter-only ones (T9, T22).
_INPUT_METHODS = ("cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")
_PARAM_METHODS = ("mean", "variance", "standard_deviation", "skewness", "kurtosis", "median", "mode", "entropy",
                  "support_lo", "support_hi")


_MAX_DOUBLE_INT = int(1.7976931348623157e308)  # ints above it round to inf or raise in float()


def _check_param(name, value):
    """A parameter is a real Python number or a non-complex sd.ndarray (T19)."""
    if isinstance(value, _sd_core.ndarray):
        if value.dtype == _sd_core.complex128:
            raise TypeError(f"'{name}' must be real, not a complex128 array")
        return
    if isinstance(value, (bool, int, float)):
        if isinstance(value, int) and abs(value) > _MAX_DOUBLE_INT:  # float(value) would overflow
            raise OverflowError(f"'{name}' is out of range")
        return
    raise TypeError(f"'{name}' must be a real number or an sd.ndarray, not {type(value).__name__}")


def _parameter(i, name):
    def get(self):
        return self._p[i]
    get.__name__ = name
    return property(get, doc=f"The parameter {name}, as given (read-only).")


class _Distribution:
    """What the continuous and discrete distributions share. A distribution class sets _name
    (the native prefix) and _params (the parameter names in native order) and defines
    __init__ with keyword-only parameters that calls _set."""

    __slots__ = ("_p", "_scalar")
    _name = ""
    _params = ()
    _density = ()  # the density's two methods: pdf, logpdf or pmf, logpmf

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls._name:  # the bases ContinuousDistribution and DiscreteDistribution
            return
        ops = {m: getattr(_k, f"{cls._name}_{m}") for m in cls._density + _INPUT_METHODS + _PARAM_METHODS}
        ops["sample"] = getattr(_k, f"{cls._name}_sample")
        ops["sample1"] = getattr(_k, f"{cls._name}_sample1")
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
        self._scalar = None if arrays else tuple(float(v) for v in values)  # for one-sample calls

    def __repr__(self):
        args = ", ".join(f"{n}={v!r}" for n, v in zip(self._params, self._p))
        return f"{type(self).__name__}({args})"

    # ---- methods with an input: elementwise, broadcast with the parameters, out= (T22)

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

    # ---- samples (T7)

    def sample(self, shape=(), /, *, rng):
        """Random samples, an array of shape `shape` + the parameters' broadcast shape (a float
        when both are empty). `rng` is an sd.random.Generator (sd.random.default_rng(seed)); the
        values depend on its state only, not on the thread count (docs/api/stats.md §2.5)."""
        if not isinstance(rng, _random.Generator):
            raise TypeError(f"sample(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), "
                            f"not {type(rng).__name__}")
        if shape == () and self._scalar is not None:  # one float: no arrays on the way
            key, pos = rng._reserve(1)
            return self._ops["sample1"](key[0], key[1], pos, *self._scalar)
        shp = _random._shape(shape, "sample")
        arrays = [v for v in self._p if isinstance(v, _sd_core.ndarray)]
        pshape = tuple(_sd_core.broadcast_arrays(*arrays)[0].shape) if arrays else ()
        out_shape = shp + pshape
        n = _random._size(out_shape)
        ps = []
        for v in self._p:
            if isinstance(v, _sd_core.ndarray):
                if v.dtype != _sd_core.float64:
                    v = v.astype(_sd_core.float64)
                ps.append(_sd_core.broadcast_to(v, pshape))
            else:
                ps.append(_sd_core.asarray([float(v)]))
        key, pos = rng._reserve(n)
        out = self._ops["sample"](key[0], key[1], pos, out_shape if out_shape else (1,), *ps)
        return float(out[0]) if not out_shape else out


class ContinuousDistribution(_Distribution):
    """Base of the continuous distributions."""

    __slots__ = ()
    _density = ("pdf", "logpdf")

    def pdf(self, x, /, *, out=None):
        """The probability density at x."""
        return self._ops["pdf"](*self._p, x, out=out)

    def logpdf(self, x, /, *, out=None):
        """log pdf(x), without underflow."""
        return self._ops["logpdf"](*self._p, x, out=out)


class DiscreteDistribution(_Distribution):
    """Base of the discrete distributions (T25): pmf and logpmf in place of the density; the
    cdfs at a non-integer x are those at floor(x); icdf(p) is the smallest k with
    cdf(k) >= p, iccdf(p) the smallest k with ccdf(k) <= p."""

    __slots__ = ()
    _density = ("pmf", "logpmf")

    def pmf(self, k, /, *, out=None):
        """P(X = k): 0 where k is not an integer of the support."""
        return self._ops["pmf"](*self._p, k, out=out)

    def logpmf(self, k, /, *, out=None):
        """log pmf(k), without underflow."""
        return self._ops["logpmf"](*self._p, k, out=out)
