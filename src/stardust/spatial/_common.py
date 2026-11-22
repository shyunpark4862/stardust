"""The argument checks and messages sd.spatial's functions share (docs/api/spatial.md §1; decisions SPT12,
SPT13, SPT15, SPT23, SPT24 in docs/design/spatial.md §1.1).

- Data (API.md §5, SPT13): an sd.ndarray, a sequence of numbers, a numpy array or a frame DataFrame
  (missing values as NaN), computed in float64; bool and int64 become float64, except for hamming and
  jaccard, which compare bool, int64 and float64 as they are (SPT23). complex128 is a TypeError.
- metric (SPT15): keyword only, one of the fourteen lowercase names (no aliases).
- The options of a metric (p, w, V, VI) are given only to the metric that uses them: another metric
  raises TypeError ("pdist(): 'p' applies only to metric='minkowski'").
- w (SPT24): 1-D, one weight per column, finite and nonnegative; V: the same, finite and positive; VI:
  (n, n).
"""

import math

from .. import _core as _sd
from ..stats._descriptive import _choice
from ..stats._distribution import _real_option

_core = _sd.spatial

METRICS = (
    "euclidean", "sqeuclidean", "cityblock", "chebyshev", "minkowski", "seuclidean", "mahalanobis", "cosine",
    "correlation", "canberra", "braycurtis", "hamming", "jaccard", "haversine",
)  # fmt: skip
COUNTING = ("hamming", "jaccard")  # compare the input's dtype as it is (SPT23)
WEIGHTLESS = ("seuclidean", "mahalanobis", "haversine")  # take no w (scipy's set, and haversine)

_TAKES = "an sd.ndarray, a sequence of numbers, a numpy array or a frame DataFrame"


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def data(x, name, arg, *, keep=False):
    """x as an sd.ndarray of at least one dimension: float64, or with `keep` (hamming, jaccard) bool,
    int64 or float64 as it is."""
    sd = _sd
    if isinstance(x, sd.ndarray):
        a = x
    elif type(x).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(x, (_frame.Series, _frame.DataFrame)):
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = None
            if keep:  # bool and int64 columns as they are (a missing value needs float64's NaN)
                try:
                    a = x.to_array()
                except TypeError, ValueError:
                    a = None
            if a is None or a.dtype == sd.string:
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
            raise TypeError(f"{name}(): '{arg}' must be {_TAKES}, not {type_name(x)}") from None
    if a.dtype == sd.complex128:
        raise TypeError(f"{name}(): '{arg}' must be real, not a complex128 array")
    if keep:
        return a
    return a if a.dtype == sd.float64 else a.astype(sd.float64)


def common_dtype(a, b):
    """hamming and jaccard on two dtypes: float64 if either is, else int64 (bool with int64)."""
    sd = _sd
    if a.dtype == b.dtype:
        return a, b
    t = sd.float64 if sd.float64 in (a.dtype, b.dtype) else sd.int64
    return (a if a.dtype == t else a.astype(t)), (b if b.dtype == t else b.astype(t))


def metric(v, name):
    return _choice(v, name, "metric", METRICS)


def flag(v, name, arg):
    if v is not True and v is not False:
        raise TypeError(f"{name}(): '{arg}' must be True or False, not {type_name(v)}")
    return v


def p_value(p, name):
    """minkowski's p: a positive real number or inf (p < 1 is a quasi-metric, taken as scipy does)."""
    r = _real_option(p, name, "p")
    if r is None and type(p).__module__ == "numpy" and getattr(getattr(p, "dtype", None), "kind", "") == "f":
        r = float(p)  # numpy.float32 and float16 too, as the C++ real arguments (API.md §2)
    if r is None:
        raise TypeError(f"{name}(): 'p' must be an int or a float, not {type_name(p)}")
    if not r > 0:  # NaN too
        raise ValueError(f"{name}(): 'p' must be a positive number or inf, not {p!r}")
    return r


def applies(name, metric_name, p, w, V, VI):
    """TypeError for an option the metric does not use."""
    if p is not None and metric_name != "minkowski":
        raise TypeError(f"{name}(): 'p' applies only to metric='minkowski'")
    if w is not None and metric_name in WEIGHTLESS:
        raise TypeError(f"{name}(): 'w' does not apply to metric='{metric_name}'")
    if V is not None and metric_name != "seuclidean":
        raise TypeError(f"{name}(): 'V' applies only to metric='seuclidean'")
    if VI is not None and metric_name != "mahalanobis":
        raise TypeError(f"{name}(): 'VI' applies only to metric='mahalanobis'")


def vector(v, name, arg, n, *, positive=False):
    """w (finite, nonnegative) or V (finite, positive): 1-D float64 of length n."""
    sd = _sd
    a = data(v, name, arg)
    if a.ndim != 1:
        raise ValueError(f"{name}(): '{arg}' must have 1 dimension, not {a.ndim}")
    if a.shape[0] != n:
        raise ValueError(f"{name}(): '{arg}' must have {n} entries (one per column), not {a.shape[0]}")
    if a.size:
        if positive:
            ok = bool(sd.all(sd.isfinite(a) & (a > 0.0)))
        else:
            ok = bool(sd.all(sd.isfinite(a) & (a >= 0.0)))
        if not ok:
            raise ValueError(f"{name}(): '{arg}' must be finite and {'positive' if positive else 'nonnegative'}")
    return a


def matrix_vi(v, name, n):
    a = data(v, name, "VI")
    if tuple(a.shape) != (n, n):
        raise ValueError(f"{name}(): 'VI' must have shape {(n, n)}, not {tuple(a.shape)}")
    return a


def kernel_metric(metric_name, p):
    """The kernel a metric runs: minkowski's p = 1, 2 and inf are cityblock, euclidean and chebyshev, bit
    for bit."""
    if metric_name == "minkowski":
        if p == 1.0:
            return "cityblock"
        if p == 2.0:
            return "euclidean"
        if p == math.inf:
            return "chebyshev"
    return metric_name


def check_alloc(nbytes, name):
    """MemoryError before an output larger than this machine's physical memory (API.md §2)."""
    if nbytes > _sd._physical_memory():
        raise MemoryError(f"{name}(): cannot allocate {nbytes} bytes: more than this machine's physical memory")


def broadcast(a, b, name, ua, ub):
    """The broadcast of two batch shapes (numpy's rule)."""
    n = max(len(a), len(b))
    a2, b2 = (1,) * (n - len(a)) + tuple(a), (1,) * (n - len(b)) + tuple(b)
    out = []
    for x, y in zip(a2, b2):
        if x != y and x != 1 and y != 1:
            raise ValueError(f"{name}(): shapes could not be broadcast together: {tuple(ua)} {tuple(ub)}")
        out.append(y if x == 1 else x)
    return tuple(out)
