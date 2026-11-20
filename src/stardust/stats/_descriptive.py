"""Descriptive statistics (T8, docs/api/stats.md §4, docs/design/stats.md §5; decisions T28).

scipy.stats' functions with stardust's conventions: the input is positional and the options
keyword-only; axis=None (the whole array) by default as sd.mean and sd.var; the degrees of
freedom are `correction` (sd.var's name for numpy's ddof); nan_policy is scipy's three
('propagate' a NaN to its lane's result, 'omit' it, 'raise' ValueError). Results are float64
(Python floats where the result has no axes); describe and mode return scipy's namedtuples.

The reductions move the reduced axes last and run lane by lane in the native kernels
stardust._core.stats.desc_* (compensated sums, two-pass moments).
"""

import math
import operator

from .. import _calls
from .. import _core as _sd_core
from .._results import DescribeResult, ModeResult
from ._distribution import _readonly, _real_option, _real_scalar

_k = _sd_core.stats

_POLICIES = ("propagate", "omit", "raise")
_INT64_MIN, _INT64_MAX = -(2**63), 2**63 - 1
_EPS = 2.0**-52


def _type_name(v):
    return _sd_core._type_name(v)  # (numpy's types as numpy.bool: API.md §2.1)


_TAKES = "an sd.ndarray, a sequence of numbers or a frame Series or DataFrame"


def _scalar_value(x):
    """x as a Python number when it is a real scalar (Python's, numpy's, or a 0-d numpy array
    of a real type), else None."""
    v = _real_scalar(x)
    if v is None and type(x).__module__ == "numpy" and getattr(x, "ndim", None) == 0:
        if getattr(getattr(x, "dtype", None), "kind", "") in "biuf":
            v = _real_scalar(x[()])
    return v


def _as_array(x, name, arg="x"):
    """x as a float64 sd.ndarray (real scalars as one element, sequences through sd.asarray,
    whose shape errors pass through under this function's name)."""
    sd = _sd_core
    if isinstance(x, sd.ndarray):
        a = x
    elif (v := _scalar_value(x)) is not None:
        a = sd.asarray([float(v)])
    elif type(x).__module__.startswith("stardust.frame"):
        # a frame Series or DataFrame (user decision 2026-11-20, D-30: data-taking functions take
        # frame columns, as sd.regression's): missing values as NaN, for nan_policy
        from .. import frame as _frame

        if not isinstance(x, (_frame.Series, _frame.DataFrame)):
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {_type_name(x)}")
        try:
            a = x.to_array(dtype=sd.float64, na_value=math.nan)
        except TypeError, ValueError:  # a string column is a ValueError of to_array
            raise TypeError(f"{name}(): '{arg}' must have numeric columns") from None
    else:
        try:
            a = sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" not in msg:  # a ragged sequence, ...
                raise ValueError(f"{name}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {_type_name(x)}") from e
    if a.dtype == sd.complex128:
        raise TypeError(f"{name}(): '{arg}' must be real, not a complex128 array")
    return a if a.dtype == sd.float64 else a.astype(sd.float64)


def _base(base, name):
    """base=: None (nats) or a positive number; base 1 divides by log 1 = 0 (inf, as scipy)."""
    if base is None:
        return
    if _real_option(base, name, "base") is None:
        raise TypeError(f"{name}(): 'base' must be a positive number or None, not {_type_name(base)}")
    if not base > 0:
        raise ValueError(f"{name}(): 'base' must be a positive number or None")


def _choice(v, name, arg, choices):
    """A string option, by the message form of every area (API.md §2.1): not a str is a
    TypeError ("must be 'a', 'b' or 'c', not NoneType"), another str a ValueError ("..., not 'x'")."""
    listed = ", ".join(f"'{c}'" for c in choices[:-1]) + f" or '{choices[-1]}'"
    if not isinstance(v, str):
        raise TypeError(f"{name}(): '{arg}' must be {listed}, not {_type_name(v)}")
    if v not in choices:
        raise ValueError(f"{name}(): '{arg}' must be {listed}, not {v!r}")
    return v


def _policy(nan_policy, name):
    return _choice(nan_policy, name, "nan_policy", _POLICIES)


def _check_nan(a, nan_policy, name):
    if nan_policy == "raise" and a.size and bool(_sd_core.any(_sd_core.isnan(a))):
        raise ValueError(f"{name}(): the input contains NaN (nan_policy='raise')")


def _bool(v, name, arg):
    if v is not True and v is not False:
        raise TypeError(f"{name}(): '{arg}' must be True or False, not {_type_name(v)}")
    return v


def _axes(axis, ndim, name):
    """The reduced axes, sorted (None: all)."""
    if axis is None:
        return tuple(range(ndim))
    # an int, or a tuple or list of ints, as the array area's reductions (docs/api/array.md §1.3)
    items = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
    out = []
    for ax in items:
        if isinstance(ax, bool):
            raise TypeError(f"{name}(): 'axis' must be an int or a tuple or list of ints, not bool")
        try:
            i = operator.index(ax)
        except TypeError:
            raise TypeError(
                f"{name}(): 'axis' must be an int or a tuple or list of ints, not {_type_name(ax)}"
            ) from None
        if not _INT64_MIN <= i <= _INT64_MAX:  # (as the array area's axis arguments, API.md §2)
            raise OverflowError(f"{name}(): 'axis' is out of range")
        if not -ndim <= i < ndim:
            raise ValueError(f"{name}(): axis {i} is out of bounds for an array of dimension {ndim}")
        out.append(i % ndim)
    if len(set(out)) != len(out):
        raise ValueError(f"{name}(): repeated axis")
    return tuple(sorted(out))


class _Lanes:
    """a with the reduced axes moved last, as a (lanes, n) float64 array, and the shapes to put
    a per-lane result back."""

    def __init__(self, a, axis, name):
        sd = _sd_core
        red = _axes(axis, a.ndim, name)
        kept = tuple(d for d in range(a.ndim) if d not in red)
        self.ndim = a.ndim
        self.red = red
        self.perm = kept + red
        self.kept_shape = tuple(a.shape[d] for d in kept)
        self.n = math.prod(a.shape[d] for d in red)
        lanes = math.prod(self.kept_shape)
        t = sd.permute_dims(a, self.perm) if self.perm != tuple(range(a.ndim)) else a
        self.moved_shape = tuple(t.shape)
        self.rows = sd.reshape(t, (lanes, self.n))
        self.keep_shape = tuple(1 if d in red else a.shape[d] for d in range(a.ndim))

    def out(self, v, keepdims=False):
        """A per-lane vector in the result's shape (a Python number for no axes)."""
        shape = self.keep_shape if keepdims else self.kept_shape
        if not shape:
            x = v[0]
            return int(x) if isinstance(x, int) else float(x)
        return _sd_core.reshape(v, shape)

    def back(self, rows):
        """A (lanes, n) result in the input's shape (the axes moved back)."""
        sd = _sd_core
        t = sd.reshape(rows, self.moved_shape)
        if self.perm == tuple(range(self.ndim)):
            return t
        inv = [0] * self.ndim
        for i, d in enumerate(self.perm):
            inv[d] = i
        return sd.permute_dims(t, tuple(inv))


def _float(v):
    return v.astype(_sd_core.float64)


_M2_SMALL = 2.0**-500  # below it m2^2 leaves the normal doubles (and the squares that made m2 may have)
_BIG = 2.0**1000


def _moments(a, axis, nan_policy, name, kmax=4, minmax=False):
    """(lanes, (n, mean, m2, m3, m4, lo, hi, mean_lo), unit): the kernel's moments per lane.
    `unit` is None, or a per-lane (u, f): the central moments of that lane are then those of
    the lane divided by u f (m_k = m_k' (u f)^k), and the mean is the lane's own.

    The kernel sums powers of x - mean: for data spread beyond ~1e77 (m4) or ~1e154 (m2), or
    below their reciprocals, the powers overflow or underflow (sem of 1e-200-sized data came
    out 0, the skewness of 1e150-sized data NaN). Such lanes (a moment that is not finite, or
    m2 below 2^-500, with a finite mean) are computed again from (x - min) / u, u the power of
    two below half their range: an exact rescaling, and central moments do not move with the
    shift. Lanes reaching 2^1000 are halved first (x / 2 - min / 2: the range itself can
    overflow), which f = 2 records."""
    sd = _sd_core
    lanes = _Lanes(a, axis, name)
    _check_nan(a, nan_policy, name)
    omit = nan_policy == "omit"
    res = _k.desc_moments(lanes.rows, omit, kmax, minmax)
    n, mean, m2 = res[0], res[1], res[2]
    top = res[kmax]
    if lanes.n < 2 or m2.shape[0] == 0:  # (no lanes: a reduction to an empty shape)
        return lanes, res, None
    if m2.shape[0] == 1:  # one lane (axis=None): the test on Python floats, ~0.2 us
        if float(m2[0]) >= _M2_SMALL and math.isfinite(float(top[0])):
            return lanes, res, None
    else:
        # every lane fine (the usual data): two reductions say so. A NaN or an infinity in
        # either makes its reduction NaN or infinite, and the lanes are then looked at one by one.
        if float(sd.min(m2)) >= _M2_SMALL and math.isfinite(float(sd.sum(top))):
            return lanes, res, None
    fine = (m2 >= _M2_SMALL) & sd.isfinite(top)
    need = sd.isfinite(mean) & ~fine & (_float(n) >= 2.0)
    if not bool(sd.any(need)):
        return lanes, res, None
    rows = lanes.rows
    lo = (sd.nanmin if omit else sd.min)(rows, axis=1)
    hi = (sd.nanmax if omit else sd.max)(rows, axis=1)
    big = sd.maximum(sd.abs(lo), sd.abs(hi)) >= _BIG
    half = sd.where(big, 0.5, 1.0)
    span = hi * half - lo * half  # the range, or half of it
    need = need & (span > 0.0) & sd.isfinite(span)
    if not bool(sd.any(need)):
        return lanes, res, None
    u = sd.where(need, 2.0 ** sd.floor(sd.log2(sd.where(need, span, 1.0)) - sd.where(big, 0.0, 1.0)), 1.0)
    u = sd.where(u > 0.0, u, 5e-324)  # a range of one subnormal step
    f = sd.where(need & big, 2.0, 1.0)
    shift = sd.where(need, lo, 0.0)
    col = (rows.shape[0], 1)
    hcol = sd.reshape(sd.where(need, half, 1.0), col)
    y = (rows * hcol - sd.reshape(shift, col) * hcol) / sd.reshape(u, col)
    res2 = _k.desc_moments(y, omit, kmax, False)
    out = list(res)
    for k in range(2, kmax + 1):
        out[k] = sd.where(need, res2[k], res[k])
    return lanes, tuple(out), (u, f)


def _in_unit(v, unit, k=1):
    """v (of the scaled lane, a k-th power of its unit) in the data's own unit."""
    if unit is None:
        return v
    for _ in range(k):
        v = v * unit[0] * unit[1]
    return v


def _mean_in_unit(mean, unit):
    """The mean in the unit of the scaled moments (for the test against a constant lane)."""
    return mean if unit is None else mean / unit[0] / unit[1]


def _skew_of(n, mean, m2, m3, bias):
    sd = _sd_core
    with_nan = sd.where(m2 <= (_EPS * mean) * (_EPS * mean), float("nan"), m3 / (m2 * sd.sqrt(m2)))
    if bias:
        return with_nan
    nf = _float(n)
    corr = sd.sqrt((nf - 1.0) * nf) / (nf - 2.0) * with_nan
    return sd.where(nf > 2.0, corr, with_nan)


def _kurt_of(n, mean, m2, m4, fisher, bias):
    sd = _sd_core
    vals = sd.where(m2 <= (_EPS * mean) * (_EPS * mean), float("nan"), m4 / (m2 * m2))
    if not bias:
        nf = _float(n)
        corr = 1.0 / (nf - 2.0) / (nf - 3.0) * ((nf * nf - 1.0) * vals - 3.0 * (nf - 1.0) * (nf - 1.0)) + 3.0
        vals = sd.where(nf > 3.0, corr, vals)
    return vals - 3.0 if fisher else vals


def _var_of(n, m2, correction):
    sd = _sd_core
    nf = _float(n)
    den = nf - correction
    return sd.where(den > 0.0, m2 * nf / den, float("nan"))


def _correction(v, name):
    c = _real_option(v, name, "correction")
    if c is None:
        raise TypeError(f"{name}(): 'correction' must be an int or a float, not {_type_name(v)}")
    return c


# ---- the functions ------------------------------------------------------------------------


def skew(x, /, *, axis=None, bias=True, nan_policy="propagate", keepdims=False):
    """The sample skewness m3 / m2^(3/2) (bias=False: the adjusted Fisher-Pearson coefficient,
    for n > 2); NaN where the sample is constant."""
    name = "skew"
    _policy(nan_policy, name), _bool(bias, name, "bias"), _bool(keepdims, name, "keepdims")
    lanes, (n, mean, m2, m3, _, _, _, _), unit = _moments(_as_array(x, name), axis, nan_policy, name, 3)
    return lanes.out(_skew_of(n, _mean_in_unit(mean, unit), m2, m3, bias), keepdims)


def kurtosis(x, /, *, axis=None, fisher=True, bias=True, nan_policy="propagate", keepdims=False):
    """The sample kurtosis m4 / m2^2, minus 3 with fisher=True (the excess kurtosis, as
    X.kurtosis(): T23); bias=False corrects it (n > 3). NaN where the sample is constant."""
    name = "kurtosis"
    _policy(nan_policy, name), _bool(fisher, name, "fisher"), _bool(bias, name, "bias")
    _bool(keepdims, name, "keepdims")
    lanes, (n, mean, m2, _, m4, _, _, _), unit = _moments(_as_array(x, name), axis, nan_policy, name)
    return lanes.out(_kurt_of(n, _mean_in_unit(mean, unit), m2, m4, fisher, bias), keepdims)


def describe(x, /, *, axis=None, correction=1, bias=True, nan_policy="propagate"):
    """DescribeResult(nobs, minmax, mean, variance, skewness, kurtosis): the count, (min, max),
    the mean, the variance with `correction` (1: the sample variance), skew and kurtosis
    (excess) with `bias`. An empty lane is nobs 0 with NaNs, as every descriptive statistic
    (user decision 2026-11-20, D-32; scipy raises)."""
    name = "describe"
    _policy(nan_policy, name), _bool(bias, name, "bias")
    corr = _correction(correction, name)
    a = _as_array(x, name)
    lanes, (n, mean, m2, m3, m4, lo, hi, _), unit = _moments(a, axis, nan_policy, name, 4, True)
    mean_u = _mean_in_unit(mean, unit)
    if nan_policy == "omit":
        nobs = lanes.out(n)
    else:
        nobs = lanes.n
    ro = _readonly  # result objects' arrays are read-only (user decision 2026-11-20, D-4)
    return DescribeResult(
        ro(nobs),
        (ro(lanes.out(lo)), ro(lanes.out(hi))),
        ro(lanes.out(mean)),
        ro(lanes.out(_in_unit(_var_of(n, m2, corr), unit, 2))),
        ro(lanes.out(_skew_of(n, mean_u, m2, m3, bias))),
        ro(lanes.out(_kurt_of(n, mean_u, m2, m4, True, bias))),
    )


def _orders(order, name):
    if order is None:  # (None takes the same check as any wrong value, API.md §2.1)
        raise TypeError(f"{name}(): 'order' must be an int or a sequence of ints, not NoneType")
    # one number (numpy integers too, API.md §2; an int beyond the doubles is "'order' is out
    # of range") or a sequence of them
    scalar = isinstance(order, bool) or _real_option(order, name, "order") is not None
    try:
        items = [order] if scalar else list(order.tolist() if isinstance(order, _sd_core.ndarray) else order)
    except TypeError:
        raise TypeError(f"{name}(): 'order' must be an int or a sequence of ints, not {_type_name(order)}") from None
    if not items:
        raise ValueError(f"{name}(): 'order' must not be empty")
    out = []
    for o in items:
        v = _real_option(o, name, "order")
        if v is None:
            raise TypeError(f"{name}(): 'order' must be an int or a sequence of ints, not {_type_name(o)}")
        if not (math.isfinite(v) and v == math.floor(v)):
            raise ValueError(f"{name}(): every 'order' must be integral")
        out.append(v)
    return out, scalar


def moment(x, /, *, order, center=None, axis=None, nan_policy="propagate", keepdims=False):
    """The `order`-th moment about `center`, mean((x - center)^order); center=None is the
    mean (the central moment: order 1 gives 0). `order` an int or a sequence of them (the
    results stacked on a new first axis); `center` a number or an array of the result's shape."""
    name = "moment"
    sd = _sd_core
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    orders, scalar = _orders(order, name)
    a = _as_array(x, name)
    lanes = _Lanes(a, axis, name)
    _check_nan(a, nan_policy, name)
    unit = None
    centers = None
    if center is not None:
        cv = float(center) if isinstance(center, bool) else _real_option(center, name, "center")
        c = _as_array(center, name, "center") if cv is None else None
        lanes_count = lanes.rows.shape[0]
        if c is None:
            centers = sd.full((lanes_count,), cv)
        else:
            try:
                centers = (
                    sd.reshape(sd.broadcast_to(c, lanes.kept_shape), (lanes_count,))
                    if lanes.kept_shape
                    else sd.reshape(c, (1,))
                )
            except ValueError:
                raise ValueError(
                    f"{name}(): 'center' cannot be broadcast to the result's shape {lanes.kept_shape}"
                ) from None
    omit = nan_policy == "omit"
    central = None
    if centers is None and any(o in (2.0, 3.0, 4.0) for o in orders):  # the moments kernel (parallel blocks)
        _, central, unit = _moments(a, axis, nan_policy, name, int(max(o for o in orders if o <= 4.0)))

    def one(o):
        if central is not None and o in (2.0, 3.0, 4.0):
            return _in_unit(central[int(o)], unit, int(o))
        return _k.desc_moment(lanes.rows, o, centers, omit)

    res = [lanes.out(one(o), keepdims) for o in orders]
    if scalar:
        return res[0]
    if isinstance(res[0], float):
        return sd.asarray(res)
    return sd.stack(res)


def sem(x, /, *, axis=None, correction=1, nan_policy="propagate", keepdims=False):
    """The standard error of the mean, std(correction) / sqrt(n)."""
    name = "sem"
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    corr = _correction(correction, name)
    lanes, (n, _, m2, _, _, _, _, _), unit = _moments(_as_array(x, name), axis, nan_policy, name, 2)
    return lanes.out(_in_unit(_sd_core.sqrt(_var_of(n, m2, corr) / _float(n)), unit), keepdims)


def zscore(x, /, *, axis=None, correction=0, nan_policy="propagate"):
    """(x - mean) / std along the axes (`correction` for the std; omit: the NaNs stay NaN and
    are left out of the mean and std). NaN where the lane is constant."""
    name = "zscore"
    sd = _sd_core
    _policy(nan_policy, name)
    corr = _correction(correction, name)
    a = _as_array(x, name)
    lanes, (n, mean, m2, _, _, _, _, mean_lo), unit = _moments(a, axis, nan_policy, name, 2)
    std = _in_unit(sd.sqrt(_var_of(n, m2, corr)), unit)
    std = sd.where(std <= sd.abs(_EPS * mean), float("nan"), std)
    shape = lanes.keep_shape
    # (x - mean) - mean_lo: the mean's rounding would shift every z by it / std
    z = ((a - sd.reshape(mean, shape)) - sd.reshape(mean_lo, shape)) / sd.reshape(std, shape)
    return float(z[0]) if _scalar_value(x) is not None else z  # a scalar's shape: a Python float


_RANK_METHODS = {"average": 0, "min": 1, "max": 2, "dense": 3, "ordinal": 4}


def rankdata(x, /, *, method="average", axis=None, nan_policy="propagate"):
    """Ranks from 1 along the axis (axis=None: of the flattened array, a 1-D result as scipy).
    method: 'average' (ties share their mean rank), 'min', 'max', 'dense' (consecutive) or
    'ordinal' (ties by position). float64. A lane with a NaN is all NaN; with omit its NaNs
    are NaN and the rest ranked among themselves."""
    name = "rankdata"
    sd = _sd_core
    _policy(nan_policy, name)
    _choice(method, name, "method", tuple(_RANK_METHODS))
    a = _as_array(x, name)
    if axis is None:
        a = sd.reshape(a, (a.size,))
        axis = 0
    else:
        # one axis (T37): the message names what this function takes, not the reductions' tuples
        try:
            if isinstance(axis, bool):
                raise TypeError
            operator.index(axis)
        except TypeError:
            raise TypeError(f"rankdata(): 'axis' must be an int or None, not {_type_name(axis)}") from None
    _check_nan(a, nan_policy, name)
    lanes = _Lanes(a, axis, name)
    r = lanes.back(_k.desc_rank(lanes.rows, _RANK_METHODS[method], nan_policy == "omit"))
    return float(r[0]) if _scalar_value(x) is not None else r  # a scalar's shape: a Python float


def mode(x, /, *, axis=None, nan_policy="propagate", keepdims=False):
    """ModeResult(mode, count): the most frequent value (the smallest of ties) and its count.
    With propagate NaN counts as a value (scipy); an empty lane is (NaN, 0)."""
    name = "mode"
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    a = _as_array(x, name)
    lanes = _Lanes(a, axis, name)
    _check_nan(a, nan_policy, name)
    m, c = _k.desc_mode(lanes.rows, nan_policy == "omit")
    return ModeResult(_readonly(lanes.out(m, keepdims)), _readonly(lanes.out(c, keepdims)))  # (D-4)


_NORMAL_IQR = 1.3489795003921634  # 2 sqrt(2) erfinv(1/2): the standard normal's IQR


def iqr(x, /, *, axis=None, q=(0.25, 0.75), scale=1.0, method="linear", nan_policy="propagate", keepdims=False):
    """quantile(x, q[1]) - quantile(x, q[0]) divided by `scale` (a number, or 'normal' for the
    standard normal's IQR, 1.349...: a consistent estimate of sigma). q as fractions and
    `method` as sd.quantile's."""
    name = "iqr"
    sd = _sd_core
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    what = "iqr(): 'q' must be a pair of numbers in [0, 1]"
    try:
        items = list(q)
    except TypeError:
        raise TypeError(f"{what}, not {_type_name(q)}") from None
    if len(items) != 2:
        raise ValueError(what)
    qs = []
    for v in items:  # (numbers as every number option: numpy integers too, not bool or str)
        f = _real_option(v, name, "q")
        if f is None:
            raise TypeError(f"{what}, not {_type_name(v)}")
        if not 0.0 <= f <= 1.0:
            raise ValueError(what)
        qs.append(f)
    q0, q1 = sorted(qs)  # (scipy sorts its rng: the range is never negative)
    if isinstance(scale, str):
        if scale != "normal":
            raise ValueError(f"iqr(): 'scale' must be a number or 'normal', not {scale!r}")
        scale = _NORMAL_IQR
    else:
        sc = _real_option(scale, name, "scale")
        if sc is None:
            raise TypeError(f"iqr(): 'scale' must be a number or 'normal', not {_type_name(scale)}")
        scale = sc
    a = _as_array(x, name)
    red = _axes(axis, a.ndim, name) if axis is not None else None  # (this function's own messages)
    _check_nan(a, nan_policy, name)
    f = sd.nanquantile if nan_policy == "omit" else sd.quantile
    try:  # both quantiles in one selection (q's axis first)
        both = f(a, sd.asarray([q0, q1]), axis=red, method=method, keepdims=keepdims)
    except (TypeError, ValueError) as e:  # method=: the quantile's check, under this function's name
        head, sep, rest = str(e).partition("(): ")
        raise type(e)(f"iqr(): {rest}" if sep else str(e)) from None
    lo, hi = both[0], both[1]
    # IEEE division (scale = 0 gives inf, or NaN for a zero range, as scipy)
    return sd.divide(hi - lo, float(scale))


def entropy(pk, qk=None, /, *, base=None, axis=None, nan_policy="propagate", keepdims=False):
    """The Shannon entropy -sum p log p of pk normalized along the axes, or with qk the relative
    entropy (Kullback-Leibler divergence) sum p log(p / q); in nats, or in `base`. omit drops
    the NaN positions (of either)."""
    name = "entropy"
    sd = _sd_core
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    _base(base, name)
    p = _as_array(pk, name, "pk")
    q = _as_array(qk, name, "qk") if qk is not None else None
    if q is not None:
        try:
            p, q = sd.broadcast_arrays(p, q)
        except ValueError:
            raise ValueError("entropy(): 'pk' and 'qk' could not be broadcast together") from None
    red = _axes(axis, p.ndim, name)
    for v in (p, q):
        if v is not None:
            _check_nan(v, nan_policy, name)
    if nan_policy == "omit":
        bad = sd.isnan(p) if q is None else (sd.isnan(p) | sd.isnan(q))
        p = sd.where(bad, 0.0, p)
        if q is not None:
            q = sd.where(bad, 0.0, q)
    p = p / sd.sum(p, axis=red, keepdims=True)
    if q is None:
        vec = sd.special.entr(p)
    else:
        q = q / sd.sum(q, axis=red, keepdims=True)
        vec = sd.special.rel_entr(p, q)
    s = sd.sum(vec, axis=red, keepdims=keepdims)
    # base 1: log(base) = 0, IEEE division (inf, or NaN for a zero entropy, as scipy)
    return sd.divide(s, math.log(base)) if base is not None else s


_DE_METHODS = {"vasicek": 0, "van es": 1, "ebrahimi": 2, "correa": 3, "auto": -1}


def differential_entropy(
    x, /, *, window_length=None, base=None, axis=None, method="auto", nan_policy="propagate", keepdims=False
):
    """An estimate of the differential entropy of the distribution x is a sample of, from its
    spacings (scipy's methods: 'vasicek', 'van es', 'ebrahimi', 'correa'; 'auto' picks van Es
    for n <= 10, Ebrahimi for n <= 1000, else Vasicek). window_length m: default
    floor(sqrt(n) + 1/2); a lane with 2m >= n (or m < 1) gives NaN."""
    name = "differential_entropy"
    _policy(nan_policy, name), _bool(keepdims, name, "keepdims")
    sd = _sd_core
    _choice(method, name, "method", tuple(_DE_METHODS))  # exact names, as every string option
    if window_length is None:
        m = 0
    else:
        if isinstance(window_length, bool):
            raise TypeError("differential_entropy(): 'window_length' must be an int or None, not bool")
        try:
            m = operator.index(window_length)
        except TypeError:
            raise TypeError(
                f"differential_entropy(): 'window_length' must be an int or None, not {_type_name(window_length)}"
            ) from None
        if not _INT64_MIN <= m <= _INT64_MAX:
            raise OverflowError("differential_entropy(): 'window_length' is out of range")
        if m < 1:
            raise ValueError("differential_entropy(): 'window_length' must be >= 1")
    _base(base, name)
    a = _as_array(x, name)
    lanes = _Lanes(a, axis, name)
    _check_nan(a, nan_policy, name)
    v = _k.desc_diffent(lanes.rows, m, _DE_METHODS[method], nan_policy == "omit")
    if base is not None:
        v = sd.divide(v, math.log(base))
    return lanes.out(v, keepdims)


# a wrong call shape in the library's form, "moment(): missing 1 required keyword-only argument:
# 'order'" (API.md §2.1)
describe = _calls.function(describe)
skew = _calls.function(skew)
kurtosis = _calls.function(kurtosis)
moment = _calls.function(moment)
zscore = _calls.function(zscore)
rankdata = _calls.function(rankdata)
sem = _calls.function(sem)
iqr = _calls.function(iqr)
mode = _calls.function(mode)
entropy = _calls.function(entropy)
differential_entropy = _calls.function(differential_entropy)
