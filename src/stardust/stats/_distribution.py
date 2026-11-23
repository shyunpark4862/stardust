"""The distribution classes' common machinery (docs/api/stats.md §2, docs/design/stats.md §2-§3).

A distribution holds its parameters as given (Python numbers or sd.ndarrays) and calls
the native elementwise functions stardust._core.stats.<name>_<method>, whose inputs are
(params..., x) or (params...): the parameters broadcast with the input at each call.
Parameters outside their domain give NaN (T8), never an error.
"""

import numbers
import operator

from .. import _calls
from .. import _core as _sd_core
from .. import random as _random

_Generator = _random.Generator

_k = _sd_core.stats

# Methods with an input (and out=) besides the density, and the parameter-only ones (T9, T22).
_INPUT_METHODS = ("cdf", "logcdf", "ccdf", "logccdf", "icdf", "iccdf", "ilogcdf", "ilogccdf")
_PARAM_METHODS = (
    "mean",
    "variance",
    "standard_deviation",
    "skewness",
    "kurtosis",
    "median",
    "mode",
    "entropy",
    "support_lo",
    "support_hi",
)


_MAX_DOUBLE_INT = int(1.7976931348623157e308)  # ints above it round to inf or raise in float()


_RENAMED = (TypeError, ValueError, OverflowError)


_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1


def _int_input(x, method):
    """A Python int input beyond int64: OverflowError as in sd.special (the array area's rule
    for an int operand, docs/api/array.md §1.5), whatever the parameters' types."""
    if not _INT64_MIN <= x <= _INT64_MAX:
        raise OverflowError(f"{method}(): Python int too large to convert to int64")


def _renamed(e, method, arg, value=None):
    """The kernel's error (prefixed with its own name, e.g. normal_cdf()) as the method's:
    "cdf(): ...". A complex input, refused by the kernels' type table, is said as such."""
    msg = str(e)
    head, sep, rest = msg.partition("(): ")
    if sep and head.isidentifier():
        msg = rest
    if arg is not None and isinstance(e, TypeError):
        if msg.startswith("not supported for") and "complex128" in msg:
            msg = f"'{arg}' must be real, not {'complex' if isinstance(value, complex) else 'a complex128 array'}"
        else:  # the kernels' table takes complex too; the methods do not
            msg = msg.replace(
                "must be an sd.ndarray or a bool, int, float or complex,", "must be an sd.ndarray or a real number,"
            )
    return type(e)(f"{method}(): {msg}")


def _real_scalar(v):
    """A real scalar as a Python number, or None when v is not one: bool, int and float as
    given (numpy's float64, a float, as float), numpy's other floating and integer scalars as
    float, numpy's bool as bool (API.md §2: numpy integers are ints)."""
    if isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if type(v) is float else float(v)
    if type(v).__module__ == "numpy":
        if type(v).__name__ in ("bool", "bool_"):
            return bool(v)
        if isinstance(v, numbers.Integral):
            return float(operator.index(v))  # (int64 and uint64: always inside the doubles)
        if isinstance(v, numbers.Real):
            return float(v)
    return None


def _check_param(cls, name, value):
    """A parameter is a real number or a non-complex sd.ndarray (T19), returned as kept (a
    numpy scalar as a Python number: _real_scalar); errors name the constructor, "Normal():
    ..." (API.md §2.1)."""
    if isinstance(value, _sd_core.ndarray):
        if value.dtype == _sd_core.complex128:
            raise TypeError(f"{cls}(): '{name}' must be real, not a complex128 array")
        return value
    v = _real_scalar(value)
    if v is None:
        raise TypeError(f"{cls}(): '{name}' must be a real number or an sd.ndarray, not {_sd_core._type_name(value)}")
    if isinstance(v, int) and abs(v) > _MAX_DOUBLE_INT:  # float(v) would overflow
        raise OverflowError(f"{cls}(): '{name}' is out of range")
    return v


def _kept(v):
    """A parameter as the distribution keeps it (user decision 2026-11-20, D-5): an sd.ndarray
    copied once and read-only (the caller's array may change later; the attribute and every
    method see the copy), a real scalar as _real_scalar's Python number, anything else as
    given (for the constructor's own type check)."""
    if isinstance(v, _sd_core.ndarray):
        c = v.copy()
        c._set_readonly()
        return c
    r = _real_scalar(v)
    return v if r is None else r


def _readonly(v):
    """A summary's array result read-only (user decision 2026-11-20, D-4); a Python float as is."""
    if isinstance(v, _sd_core.ndarray):
        v._set_readonly()
    return v


def _real_option(v, func, arg):
    """An int-or-float option as a float (API.md §2: numpy integers are ints, bool and numpy's
    bool are not; numpy's floating scalars of any width are floats, docs/api/array.md §1.4; an
    int beyond the doubles is OverflowError "'arg' is out of range"), or None when v is not a
    real number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, float):  # (numpy's float64 too)
        return float(v)
    try:
        i = operator.index(v)
    except TypeError:
        # numpy.float32, numpy.float16, numpy.longdouble, as the kernels' real_value takes them
        # (numpy's bool is not a Real)
        if type(v).__module__ == "numpy" and isinstance(v, numbers.Real):
            return float(v)
        return None
    try:
        return float(i)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def _public_methods(cls):
    """Every method of cls named by itself in Python's call errors: its __qualname__ the
    method's name alone ("cdf() got an unexpected keyword argument 'foo'"), not the private
    base class that defines it. The methods keep Python's own form, not the constructors'
    "Normal(): ..." (user decision 2026-11-20, D-29 (b): a wrapper costs ~50 ns on calls of a
    few hundred ns; API.md §2.1)."""
    for name, f in list(vars(cls).items()):
        if callable(f) and hasattr(f, "__qualname__") and not name.startswith("__"):
            f.__qualname__ = name


def _parameter(i, name):
    def get(self):
        return self._p[i]

    get.__name__ = name
    return property(get, doc=f"The parameter {name}, as given (read-only).")


_INPUT_ARG = {"icdf": "p", "iccdf": "p", "ilogcdf": "logp", "ilogccdf": "logp", "pmf": "k", "logpmf": "k"}


def _fast_input(name, op, nparams, slow):
    """The input method `name` of a distribution class with a shorter way for the commonest call,
    one Python float and no out= (a scalar call took twice the native function's time in the
    wrapper): the parameters unpacked by arity and the native op called directly. Every other
    call is the base class's method `slow`, which checks the input (a Python int beyond int64)
    and passes out=."""
    ps = ", ".join(f"p{i}" for i in range(nparams))
    a = _INPUT_ARG.get(name, "x")  # the input's name, as the base method's signature has it
    src = (
        f"def {name}(self, {a}, /, *, out=None):\n"
        f"    if out is None and type({a}) is float:\n"
        f"        {ps}, = self._p\n"
        f"        try:\n"
        f"            return op({ps}, {a})\n"
        f"        except _RENAMED as e:\n"
        f"            raise _renamed(e, name, arg, {a}) from None\n"
        f"    return slow(self, {a}, out=out)\n"
    )
    env = {"op": op, "slow": slow, "name": name, "arg": a, "_RENAMED": _RENAMED, "_renamed": _renamed}
    exec(src, env)
    f = env[name]
    f.__qualname__ = name
    f.__doc__ = slow.__doc__
    f.__module__ = slow.__module__
    return f


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
        ops["sample1s"] = getattr(_k, f"{cls._name}_sample1s")
        cls._ops = ops
        cls._s1 = staticmethod(ops["sample1s"])  # sample()'s scalar call, one attribute away
        for i, p in enumerate(cls._params):  # read-only attributes (T19)
            setattr(cls, p, _parameter(i, p))
        for m in cls._density + _INPUT_METHODS:
            setattr(cls, m, _fast_input(m, ops[m], len(cls._params), getattr(cls, m)))
        _calls.keyword_init(cls)

    def __init__(self, *args, **kwargs):
        # (every distribution defines its own __init__: only the bases arrive here)
        raise TypeError(
            f"{type(self).__name__}(): a base class for isinstance checks, not a distribution "
            "(make one of its subclasses, e.g. sd.stats.Normal)"
        )

    def _set(self, *values):
        for v in values:  # the common case, Python floats and ints, without the checks below
            t = type(v)
            if t is not float and (t is not int or not -_MAX_DOUBLE_INT <= v <= _MAX_DOUBLE_INT):
                break
        else:
            self._p = values
            self._scalar = tuple(map(float, values))
            return
        cls = type(self).__name__
        arrays = []
        values = tuple(_kept(_check_param(cls, name, v)) for name, v in zip(self._params, values))
        for v in values:
            if isinstance(v, _sd_core.ndarray):
                arrays.append(v)
        if len(arrays) > 1:
            try:  # parameters that cannot broadcast fail here (views only)
                _sd_core.broadcast_arrays(*arrays)
            except ValueError as e:
                raise ValueError(f"{cls}(): {str(e).partition('(): ')[2] or e}") from None
        self._p = values
        self._scalar = None if arrays else tuple(float(v) for v in values)  # for one-sample calls

    def __repr__(self):
        args = ", ".join(f"{n}={v!r}" for n, v in zip(self._params, self._p))
        return f"{type(self).__name__}({args})"

    # ---- methods with an input: elementwise, broadcast with the parameters, out= (T22)

    def cdf(self, x, /, *, out=None):
        """P(X <= x)."""
        if type(x) is int:
            _int_input(x, "cdf")
        try:
            return self._ops["cdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "cdf", "x", x) from None

    def logcdf(self, x, /, *, out=None):
        """log P(X <= x), without underflow."""
        if type(x) is int:
            _int_input(x, "logcdf")
        try:
            return self._ops["logcdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "logcdf", "x", x) from None

    def ccdf(self, x, /, *, out=None):
        """P(X > x), computed directly (accurate in the upper tail, not 1 - cdf)."""
        if type(x) is int:
            _int_input(x, "ccdf")
        try:
            return self._ops["ccdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "ccdf", "x", x) from None

    def logccdf(self, x, /, *, out=None):
        """log P(X > x), without underflow."""
        if type(x) is int:
            _int_input(x, "logccdf")
        try:
            return self._ops["logccdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "logccdf", "x", x) from None

    def icdf(self, p, /, *, out=None):
        """The x with cdf(x) = p (the inverse of cdf)."""
        if type(p) is int:
            _int_input(p, "icdf")
        try:
            return self._ops["icdf"](*self._p, p, out=out)
        except _RENAMED as e:
            raise _renamed(e, "icdf", "p", p) from None

    def iccdf(self, p, /, *, out=None):
        """The x with ccdf(x) = p (the inverse of ccdf)."""
        if type(p) is int:
            _int_input(p, "iccdf")
        try:
            return self._ops["iccdf"](*self._p, p, out=out)
        except _RENAMED as e:
            raise _renamed(e, "iccdf", "p", p) from None

    def ilogcdf(self, logp, /, *, out=None):
        """The x with logcdf(x) = logp."""
        if type(logp) is int:
            _int_input(logp, "ilogcdf")
        try:
            return self._ops["ilogcdf"](*self._p, logp, out=out)
        except _RENAMED as e:
            raise _renamed(e, "ilogcdf", "logp", logp) from None

    def ilogccdf(self, logp, /, *, out=None):
        """The x with logccdf(x) = logp."""
        if type(logp) is int:
            _int_input(logp, "ilogccdf")
        try:
            return self._ops["ilogccdf"](*self._p, logp, out=out)
        except _RENAMED as e:
            raise _renamed(e, "ilogccdf", "logp", logp) from None

    # ---- parameter-only methods: the parameters' broadcast shape (a float for scalars)

    def mean(self):
        """The mean (NaN where it is undefined, inf where infinite)."""
        try:
            return _readonly(self._ops["mean"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "mean", None) from None

    def variance(self):
        """The variance."""
        try:
            return _readonly(self._ops["variance"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "variance", None) from None

    def standard_deviation(self):
        """The standard deviation."""
        try:
            return _readonly(self._ops["standard_deviation"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "standard_deviation", None) from None

    def skewness(self):
        """The skewness."""
        try:
            return _readonly(self._ops["skewness"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "skewness", None) from None

    def kurtosis(self):
        """The excess kurtosis (0 for the normal distribution; T23)."""
        try:
            return _readonly(self._ops["kurtosis"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "kurtosis", None) from None

    def median(self):
        """The median."""
        try:
            return _readonly(self._ops["median"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "median", None) from None

    def mode(self):
        """The mode: the point of the largest density, even at an end of the support where the
        density diverges; NaN where no single point is the largest (T21)."""
        try:
            return _readonly(self._ops["mode"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "mode", None) from None

    def entropy(self):
        """The entropy, in nats: differential for a continuous distribution, Shannon's for a
        discrete one."""
        try:
            return _readonly(self._ops["entropy"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "entropy", None) from None

    def support(self):
        """(lo, hi): the ends of the support."""
        try:
            return _readonly(self._ops["support_lo"](*self._p)), _readonly(self._ops["support_hi"](*self._p))
        except _RENAMED as e:
            raise _renamed(e, "support", None) from None

    # ---- samples (T7)

    def sample(self, shape=None, /, *, rng):
        """Random samples: one per parameter set for shape=None (a float for scalar parameters),
        else an array of shape `shape` + the parameters' broadcast shape. `rng` is an
        sd.random.Generator (sd.random.default_rng(seed)); the values depend on its state only,
        not on the thread count (docs/api/stats.md §2.5)."""
        if shape is None and type(rng) is _Generator and self._scalar is not None:
            # one float: no arrays on the way, the stream element reserved natively
            return self._s1(rng._st, self._scalar)
        if not isinstance(rng, _random.Generator):
            raise TypeError(
                f"sample(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), "
                f"not {_sd_core._type_name(rng)}"
            )
        if shape is None and self._scalar is not None:
            return self._ops["sample1s"](rng._st, self._scalar)
        shp = _random._sample_shape(shape)
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
            elif pshape:  # a scalar beside array parameters: the same broadcast shape (the kernel
                # takes the parameter sets' count from the first parameter)
                ps.append(_sd_core.broadcast_to(_sd_core.asarray([float(v)]), pshape))
            else:
                ps.append(_sd_core.asarray([float(v)]))
        key, pos = rng._take(n, "sample")
        out = self._ops["sample"](key[0], key[1], pos, out_shape if out_shape else (1,), *ps)
        return float(out[0]) if not out_shape else out


class ContinuousDistribution(_Distribution):
    """Base of the continuous distributions."""

    __slots__ = ()
    _density = ("pdf", "logpdf")

    def pdf(self, x, /, *, out=None):
        """The probability density at x."""
        if type(x) is int:
            _int_input(x, "pdf")
        try:
            return self._ops["pdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "pdf", "x", x) from None

    def logpdf(self, x, /, *, out=None):
        """log pdf(x), without underflow."""
        if type(x) is int:
            _int_input(x, "logpdf")
        try:
            return self._ops["logpdf"](*self._p, x, out=out)
        except _RENAMED as e:
            raise _renamed(e, "logpdf", "x", x) from None


class DiscreteDistribution(_Distribution):
    """Base of the discrete distributions (T25): pmf and logpmf in place of the density; the
    cdfs at a non-integer x are those at floor(x); icdf(p) is the smallest k with
    cdf(k) >= p, iccdf(p) the smallest k with ccdf(k) <= p."""

    __slots__ = ()
    _density = ("pmf", "logpmf")

    def pmf(self, k, /, *, out=None):
        """P(X = k): 0 where k is not an integer of the support."""
        if type(k) is int:
            _int_input(k, "pmf")
        try:
            return self._ops["pmf"](*self._p, k, out=out)
        except _RENAMED as e:
            raise _renamed(e, "pmf", "k", k) from None

    def logpmf(self, k, /, *, out=None):
        """log pmf(k), without underflow."""
        if type(k) is int:
            _int_input(k, "logpmf")
        try:
            return self._ops["logpmf"](*self._p, k, out=out)
        except _RENAMED as e:
            raise _renamed(e, "logpmf", "k", k) from None


# Python's own call errors (a missing rng=, an unknown keyword such as mean(out=...)) name the
# method alone (API.md §2.1).
for _cls in (_Distribution, ContinuousDistribution, DiscreteDistribution):
    _public_methods(_cls)
