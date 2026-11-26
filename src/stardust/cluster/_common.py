"""The argument checks and messages sd.cluster's functions share (docs/api/cluster.md §1.1, §1.2, §1.4; decisions
CL5, CL8, CL11, CL12, CL41 in docs/design/cluster.md §1.1).

- Data (API.md §5): an sd.ndarray, a sequence of numbers, a numpy array, a frame DataFrame (the columns are the
  variables) or a Series (one variable), computed in float64; bool and int64 become float64; complex is a
  TypeError. Points are rows: X is (n, p), a 1-D X is (n, 1).
- nan_policy (CL11): 'raise' (default; ValueError) or 'omit' (the rows with a NaN left out: their label is -1,
  nobs counts the rest, `omitted` marks them). +-inf is always a ValueError.
- weights: 1-D, one per point, finite and >= 0 with at least one positive; a NaN weight makes its row a NaN row
  (observations; with precomputed distances it is a ValueError, CL73).
- metric (CL12): sd.spatial's names, and 'precomputed' for the functions defined by distances: an (n, n) square
  (symmetric bit for bit, zero diagonal, no negative value, no NaN) or a 1-D condensed vector of length
  n (n - 1) / 2 (finite, >= 0).
- rng (CL8, CL41): an sd.random.Generator, required by the functions that draw; a call takes two 64-bit words
  from it (key()), the key of Philox4x64-10, and draws element r 2^40 + t for the t-th number of its r-th start,
  sample or reference set.
- Options are keyword only; numbers follow the regression and smooth areas' forms (an int or a float,
  operator.index for ints, OverflowError 'out of range').
"""

import math
import numbers
import operator

from .. import _core as _sd
from .. import random as _random

_TAKES = "an sd.ndarray, a sequence of numbers, a numpy array or a frame DataFrame or Series"
POLICIES = ("raise", "omit")
METRICS = (
    "euclidean",
    "sqeuclidean",
    "cityblock",
    "chebyshev",
    "minkowski",
    "seuclidean",
    "mahalanobis",
    "cosine",
    "correlation",
    "canberra",
    "braycurtis",
    "hamming",
    "jaccard",
    "haversine",
)  # fmt: skip  (sd.spatial's, docs/api/spatial.md)
START_BITS = 40  # element r 2^40 + t: the t-th number of start (sample, reference set) r (CL41)


def type_name(v):
    return _sd._type_name(v)  # numpy's types keep their module (API.md §2.1)


def contiguous(a):
    """a as a C-ordered array (a view when it already is one)."""
    if a._flags()["c_contiguous"]:
        return a
    out = _sd.empty(tuple(a.shape), dtype=a.dtype)
    out[...] = a
    return out


def _is_numpy_scalar(x):
    return type(x).__module__ == "numpy" and getattr(x, "ndim", None) == 0


def _array(x, func, arg):
    """x as a real float64 sd.ndarray of at least one dimension."""
    if isinstance(x, numbers.Number) or _is_numpy_scalar(x):
        raise ValueError(f"{func}(): '{arg}' must be 1-D or 2-D, not 0-D")
    if isinstance(x, _sd.ndarray):
        a = x
    elif type(x).__module__.startswith("stardust.frame"):
        from .. import frame as _frame

        if not isinstance(x, (_frame.Series, _frame.DataFrame)):
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = x.to_array(dtype=_sd.float64, na_value=math.nan)
        except TypeError, ValueError:
            raise TypeError(f"{func}(): '{arg}' must have numeric columns") from None
    else:
        if x is None or isinstance(x, (str, bytes)):
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}")
        try:
            a = _sd.asarray(x)
        except (TypeError, ValueError) as e:
            msg = str(e)
            if isinstance(e, ValueError) and "0-dimensional" in msg:
                raise ValueError(f"{func}(): '{arg}' must be 1-D or 2-D, not 0-D") from None
            if isinstance(e, ValueError):  # a ragged sequence, ...
                raise ValueError(f"{func}(): {msg.partition('(): ')[2] or msg}") from None
            raise TypeError(f"{func}(): '{arg}' must be {_TAKES}, not {type_name(x)}") from None
    if a.dtype == _sd.complex128:
        raise TypeError(f"{func}(): '{arg}' must be real, not complex128")
    if a.dtype == _sd.string:
        raise TypeError(f"{func}(): '{arg}' must be numeric, not string")
    return a if a.dtype == _sd.float64 else a.astype(_sd.float64)


def points(x, func, arg="X"):
    """The points of x: a C-contiguous (n, p) float64 sd.ndarray with n >= 1 and p >= 1 (1-D x is (n, 1))."""
    a = _array(x, func, arg)
    if a.ndim == 1:
        a = _sd.reshape(a, (a.shape[0], 1))
    elif a.ndim != 2:
        raise ValueError(f"{func}(): '{arg}' must be 1-D or 2-D, not {a.ndim}-D")
    n, p = a.shape
    if n == 0:
        raise ValueError(f"{func}(): '{arg}' must have at least 1 point")
    if p == 0:
        raise ValueError(f"{func}(): '{arg}' must have at least 1 column")
    return contiguous(a)


def policy(v, func):
    return choice(v, func, "nan_policy", POLICIES)


def weights(w, func, n):
    """The checked weights (a C-contiguous (n,) float64 array) or None; NaN is left for rows() to judge."""
    if w is None:
        return None
    a = _array(w, func, "weights")
    if a.ndim != 1:
        raise ValueError(f"{func}(): 'weights' must be 1-D, not {a.ndim}-D")
    if a.shape[0] != n:
        raise ValueError(f"{func}(): 'weights' must have one value per point ({n}), not {a.shape[0]}")
    if bool(_sd.any(_sd.isinf(a))):
        raise ValueError(f"{func}(): 'weights' contains infinity")
    if bool(_sd.any(a < 0)):
        raise ValueError(f"{func}(): 'weights' must be >= 0")
    return contiguous(a)


def rows(X, func, nan_policy, w=None, arg="X"):
    """The rows nan_policy keeps: (X, w, omitted) with X (and w) the kept rows and omitted the (n,) bool mask
    of the rows left out (None when none was). +-inf is a ValueError; a NaN (in X or in w) is one under
    'raise'; 'omit' leaves the row out and needs at least one row left. nan_policy None: a function without the
    argument (a NaN is a ValueError naming arg)."""
    bare = nan_policy is None
    nan_policy = "raise" if bare else policy(nan_policy, func)
    if bool(_sd.any(_sd.isinf(X))):
        raise ValueError(f"{func}(): '{arg}' contains infinity")
    bad = _sd.any(_sd.isnan(X), axis=1)
    if w is not None:
        bad = _sd.logical_or(bad, _sd.isnan(w))
    if not bool(_sd.any(bad)):
        if w is not None and not bool(_sd.any(w > 0)):
            raise ValueError(f"{func}(): at least one weight must be positive")
        return X, w, None
    if bare:
        raise ValueError(f"{func}(): '{arg}' contains NaN")
    if nan_policy == "raise":
        raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
    keep = _sd.logical_not(bad)
    X = contiguous(X[keep])
    w = contiguous(w[keep]) if w is not None else None
    if X.shape[0] == 0:
        raise ValueError(f"{func}(): no point is left without NaN (nan_policy='omit')")
    if w is not None and not bool(_sd.any(w > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    return X, w, bad


def labels_back(labels, omitted):
    """The labels of the kept rows at their places in an (n,) int64 array of -1 (nan_policy='omit')."""
    if omitted is None:
        return labels
    out = _sd.full((omitted.shape[0],), -1, dtype=_sd.int64)
    out[_sd.logical_not(omitted)] = labels
    return out


def real(v, func, arg):
    """v (an int or a float; numpy's too) as a float."""
    if isinstance(v, bool) or not isinstance(v, numbers.Real):
        raise TypeError(f"{func}(): '{arg}' must be an int or a float, not {type_name(v)}")
    try:
        return float(v)
    except OverflowError:
        raise OverflowError(f"{func}(): '{arg}' is out of range") from None


def integer(v, func, arg):
    if isinstance(v, bool) or (type(v).__module__ == "numpy" and type(v).__name__ == "bool"):
        raise TypeError(f"{func}(): '{arg}' must be an int, not {type_name(v)}")
    try:
        k = operator.index(v)
    except TypeError:
        raise TypeError(f"{func}(): '{arg}' must be an int, not {type_name(v)}") from None
    if not -(2**63) <= k < 2**63:
        raise OverflowError(f"{func}(): '{arg}' is out of range")
    return k


def positive_int(v, func, arg):
    k = integer(v, func, arg)
    if k < 1:
        raise ValueError(f"{func}(): '{arg}' must be >= 1, not {k}")
    return k


def choice(v, func, arg, choices):
    listed = (
        ", ".join(f"'{c}'" for c in choices[:-1]) + f" or '{choices[-1]}'" if len(choices) > 1 else f"'{choices[0]}'"
    )
    if not isinstance(v, str):
        raise TypeError(f"{func}(): '{arg}' must be {listed}, not {type_name(v)}")
    if v not in choices:
        raise ValueError(f"{func}(): '{arg}' must be {listed}, not {v!r}")
    return v


def metric(v, func, precomputed=True):
    """A metric name of sd.spatial, or 'precomputed' when the function takes distances (CL12); the message is
    sd.spatial's (choice())."""
    return choice(v, func, "metric", METRICS + (("precomputed",) if precomputed else ()))


def metric_columns(metric_name, p, func):
    """sd.spatial's column rule of a metric, under the calling function's name: haversine takes (latitude,
    longitude) pairs."""
    if metric_name == "haversine" and p != 2:
        raise ValueError(f"{func}(): metric='haversine' needs 2 columns (latitude and longitude in radians), not {p}")


def new_points(X_new, func, p):
    """The new points of a predict-like method: (m, p) like the fit's 'X' (regression's messages); NaN and
    infinity are ValueErrors (a label needs a place)."""
    Xn = points(X_new, func, "X_new")
    if Xn.shape[1] != p:
        cols = f"{p} column" + ("" if p == 1 else "s")
        raise ValueError(f"{func}(): 'X_new' must have {cols} like the fit's 'X', not {Xn.shape[1]}")
    if bool(_sd.any(_sd.isnan(Xn))):
        raise ValueError(f"{func}(): 'X_new' contains NaN")
    if bool(_sd.any(_sd.isinf(Xn))):
        raise ValueError(f"{func}(): 'X_new' contains infinity")
    return Xn


def condensed_size(m, func, arg):
    """n of a condensed vector of length m = n (n - 1) / 2 (n >= 2), or ValueError."""
    n = (1 + math.isqrt(1 + 8 * m)) // 2
    if n * (n - 1) // 2 != m or m == 0:
        raise ValueError(
            f"{func}(): a 1-D '{arg}' must be a condensed distance vector of length n (n - 1) / 2 for some n >= 2, "
            f"not {m}"
        )
    return n


def distances(D, func, arg="X"):
    """Precomputed distances (CL12): (the condensed (n (n - 1) / 2,) float64 vector, n). D is an (n, n) square,
    symmetric bit for bit with a zero diagonal, or 1-D condensed; finite and >= 0 either way."""
    a = _array(D, func, arg)
    if a.ndim == 1:
        n = condensed_size(a.shape[0], func, arg)
        d = contiguous(a)
    elif a.ndim == 2:
        n, m = a.shape
        if n != m:
            raise ValueError(f"{func}(): a 2-D '{arg}' with metric='precomputed' must be square, not {n} x {m}")
        if n < 2:
            raise ValueError(f"{func}(): '{arg}' must have at least 2 points with metric='precomputed'")
        if bool(_sd.any(_sd.isnan(a))):
            raise ValueError(f"{func}(): the distances '{arg}' contain NaN")
        if not bool(_sd.all(a == a.T)):
            raise ValueError(f"{func}(): the distance matrix '{arg}' must be symmetric")
        if not bool(_sd.all(_sd.reshape(a, (n * n,))[:: n + 1] == 0)):
            raise ValueError(f"{func}(): the distance matrix '{arg}' must have a zero diagonal")
        from .. import spatial as _spatial

        d = contiguous(_spatial.squareform(a, checks=False))
    else:
        raise ValueError(f"{func}(): '{arg}' must be 1-D or 2-D, not {a.ndim}-D")
    if bool(_sd.any(_sd.isnan(d))):
        raise ValueError(f"{func}(): the distances '{arg}' contain NaN")
    if bool(_sd.any(_sd.isinf(d))):
        raise ValueError(f"{func}(): the distances '{arg}' contain infinity")
    if bool(_sd.any(d < 0)):
        raise ValueError(f"{func}(): the distances '{arg}' must be >= 0")
    return d, n


def check_rng(rng, func):
    if not isinstance(rng, _random.Generator):
        raise TypeError(
            f"{func}(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {type_name(rng)}"
        )
    return rng


def no_rng(rng, func, where):
    """ValueError for an rng given to a method that draws nothing (CL8): where names the methods that draw
    (regression's rlm form: 'rng' is only used with method='MM')."""
    if rng is not None:
        raise ValueError(f"{func}(): 'rng' is only used with {where}")


def key(rng):
    """The call's two 64-bit words from rng: the key of its Philox4x64-10 stream (CL41; regression's
    rlm(method='MM') takes its key the same way). The generator always moves by exactly two words."""
    return [int(v) for v in rng.integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,)).tolist()]


def check_alloc(nbytes, func):
    """MemoryError before an allocation larger than this machine's physical memory (API.md §2)."""
    if nbytes > _sd._physical_memory():
        raise MemoryError(f"{func}(): cannot allocate {nbytes} bytes: more than this machine's physical memory")
