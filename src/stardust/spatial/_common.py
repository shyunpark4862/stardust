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


_METRIC_SET = frozenset(METRICS)


def metric(v, name):
    if type(v) is str and v in _METRIC_SET:  # the common case without _choice's checks (small calls, SPT6)
        return v
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
            raise ValueError(f"{name}(): '{arg}' must be finite and {'positive' if positive else '>= 0'}")
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


# ---- the neighbour searches' options (KDTree.query, kneighbors; docs/api/spatial.md §4, SPT55, SPT58) ----


def real(v, name, arg):
    """An int-or-float option as a float (numpy's floats too); TypeError for anything else."""
    r = _real_option(v, name, arg)
    if r is None and type(v).__module__ == "numpy" and getattr(getattr(v, "dtype", None), "kind", "") == "f":
        r = float(v)
    if r is None:
        raise TypeError(f"{name}(): '{arg}' must be an int or a float, not {type_name(v)}")
    return r


def tree_p(p, name):
    """The Minkowski p of a tree search: at least 1, or inf (scipy's rule; NaN is a ValueError)."""
    r = real(p, name, "p")
    if not r >= 1.0:
        raise ValueError(f"{name}(): 'p' must be a number of at least 1 or inf, not {p!r}")
    return r


def upper_bound(v, name):
    """distance_upper_bound: nonnegative or inf (SPT19; scipy squares it, so -1 acts as 1 there)."""
    r = real(v, name, "distance_upper_bound")
    if not r >= 0.0:
        raise ValueError(f"{name}(): 'distance_upper_bound' must be a number >= 0 or inf, not {v!r}")
    return r


def _int(v):
    from ..sparse._common import is_int

    return is_int(v)


_INT64_MAX = 2**63 - 1


def k_ranks(k, name):
    """k checked: an int k >= 1 as itself (the ranks 1 .. k), or a sequence of ints >= 1 (a list, a
    tuple or a 1-D integer array; any order, repeats allowed: scipy's meaning of the ranks to return) as
    a list. bool and floats are TypeErrors; 0, negatives and an empty sequence ValueErrors (SPT55; scipy
    takes k=True and k=2.0, and k=[0] kills its process)."""
    import operator

    what = "an int or a sequence of ints"
    if _int(k):
        k = operator.index(k)
        if k < 1:
            raise ValueError(f"{name}(): 'k' must be at least 1, not {k}")
        if k > _INT64_MAX:
            raise OverflowError(f"{name}(): 'k' is out of range")
        return k
    if isinstance(k, (list, tuple)):
        items = list(k)
    elif isinstance(k, _sd.ndarray) or type(k).__module__ == "numpy":
        if getattr(k, "ndim", 0) != 1:
            raise TypeError(f"{name}(): 'k' must be {what}, not a {getattr(k, 'ndim', 0)}-D array")
        items = k.tolist()
    else:
        raise TypeError(f"{name}(): 'k' must be {what}, not {type_name(k)}")
    if not items:
        raise ValueError(f"{name}(): 'k' must not be an empty sequence")
    for v in items:
        if not _int(v):
            raise TypeError(f"{name}(): 'k' must be {what}, not a sequence holding {type_name(v)}")
    items = [operator.index(v) for v in items]
    bad = next((v for v in items if v < 1), None)
    if bad is not None:
        raise ValueError(f"{name}(): every entry of 'k' must be at least 1, not {bad}")
    if max(items) > _INT64_MAX:
        raise OverflowError(f"{name}(): 'k' is out of range")
    return items


# ---- kneighbors and radius_neighbors (SPT2; docs/api/spatial.md §1.2, §3, §4) ----


def finite(a, name, arg):
    """ValueError unless every entry of the data is finite (SPT12: the neighbour searches); bool and int64
    data always are."""
    if a.dtype == _sd.float64 and a.size and not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{name}(): '{arg}' must be finite")


def radius(r, name, rows):
    """The radius of radius_neighbors (query_ball_point's rule): a number >= 0 or inf, or an array
    of them that broadcasts to (rows,); NaN and negatives are ValueErrors (scipy compares r squared, so -1
    acts as 1 there). As a float64 (rows,) array."""
    sd = _sd
    if isinstance(r, (int, float)) or type(r).__module__ == "numpy" and getattr(r, "ndim", 1) == 0:
        v = real(r, name, "r")
        if not v >= 0.0:
            raise ValueError(f"{name}(): 'r' must be a number >= 0 or inf, not {r!r}")
        return sd.full((rows,), v)
    try:
        a = data(r, name, "r")
    except TypeError:
        raise TypeError(f"{name}(): 'r' must be a number or an array of numbers, not {type_name(r)}") from None
    if a.ndim > 1 or a.ndim == 1 and a.shape[0] not in (1, rows):
        raise ValueError(f"{name}(): 'r' of shape {tuple(a.shape)} does not broadcast to the {rows} rows of 'XA'")
    if a.size and not bool(sd.all(a >= 0.0)):  # NaN too
        raise ValueError(f"{name}(): every entry of 'r' must be a number >= 0 or inf")
    return sd.broadcast_to(a, (rows,)).copy()  # contiguous, the caller's array untouched


# ---- KDTree's radius, pair and count queries (SPT4; docs/api/spatial.md §4, decisions SPT63-) ----


def scalar_radius(r, name, arg="r"):
    """A radius of one value (query_ball_tree, query_pairs, sparse_distance_matrix): a nonnegative number or
    inf; NaN and negatives are ValueErrors (scipy compares r squared, so -1 acts as 1 there)."""
    if type(r).__module__ == "numpy" and type(r).__name__ == "ndarray" and r.ndim == 0:
        r = r[()]  # numpy's 0-d array: its scalar
    v = real(r, name, arg)
    if not v >= 0.0:
        raise ValueError(f"{name}(): '{arg}' must be a number >= 0 or inf, not {r!r}")
    return v


def ball_radius(r, name, batch):
    """query_ball_point's r: a number >= 0 or inf, or an array of them that broadcasts to the batch
    shape of x (numpy's rule, the batch shape unchanged); as a float64 array of prod(batch) entries in C
    order."""
    sd = _sd
    rows = math.prod(batch)
    if isinstance(r, (int, float)) or type(r).__module__ == "numpy" and getattr(r, "ndim", 1) == 0:
        return sd.full((rows,), scalar_radius(r, name))
    try:
        a = data(r, name, "r")
    except TypeError:
        raise TypeError(f"{name}(): 'r' must be a number or an array of numbers, not {type_name(r)}") from None
    shape = tuple(a.shape)
    ok = len(shape) <= len(batch) and all(s in (1, b) for s, b in zip(shape[::-1], batch[::-1]))
    if not ok:
        raise ValueError(f"{name}(): 'r' of shape {shape} does not broadcast to the batch shape {tuple(batch)} of 'x'")
    if a.size and not bool(sd.all(a >= 0.0)):  # NaN too
        raise ValueError(f"{name}(): every entry of 'r' must be a number >= 0 or inf")
    return sd.reshape(sd.broadcast_to(a, tuple(batch)).copy(), (rows,))


def tree_other(other, name, m):
    """The other KDTree of query_ball_tree, count_neighbors and sparse_distance_matrix: a KDTree of the same
    m (scipy's rule)."""
    from ._kdtree import KDTree

    if not isinstance(other, KDTree):
        raise TypeError(f"{name}(): 'other' must be a KDTree, not {type_name(other)}")
    if other.m != m:
        raise ValueError(f"{name}(): 'other' must have the same m as this tree ({m}), not {other.m}")
    return other


# ---- the geometry classes (docs/design/spatial_geometry.md §2.1; SPT98, SPT101, SPT149) ----


def geometry_points(x, name, *, check_finite=True):
    """The points of Delaunay and ConvexHull: an (n, d) float64 sd.ndarray, d = 2 or 3, finite when check_finite
    (ConvexHull checks finiteness in its native scan, with the same message). Not copied here: the native
    code makes its own C-contiguous copy (the points attribute, SPT135)."""
    a = data(x, name, "points")
    if a.ndim != 2:
        raise ValueError(f"{name}(): 'points' must have 2 dimensions, not {a.ndim}")
    if a.shape[1] not in (2, 3):
        raise ValueError(f"{name}(): 'points' must have 2 or 3 columns, not {a.shape[1]}")
    if check_finite:
        finite(a, name, "points")
    return a
