"""The errors of pdist, cdist and the distance functions (docs/api/spatial.md §1.2, §1.4, §2; API.md §2,
§2.1): every wrong argument, its type and its message, which starts with the function's name. scipy's
silent cases (p <= 0 in pdist, NaN or inf weights, nonpositive V, positional metric) are errors here."""

import math
import re

import numpy as np
import pytest

import stardust as sd

S = sd.spatial
X = np.arange(12.0).reshape(4, 3)


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def test_metric():
    names = ("'euclidean', 'sqeuclidean', 'cityblock', 'chebyshev', 'minkowski', 'seuclidean', 'mahalanobis', "
             "'cosine', 'correlation', 'canberra', 'braycurtis', 'hamming', 'jaccard' or 'haversine'")  # fmt: skip
    with raises(ValueError, f"pdist(): 'metric' must be {names}, not 'e'"):
        S.pdist(X, metric="e")  # no aliases
    with raises(ValueError, f"cdist(): 'metric' must be {names}, not 'Euclidean'"):
        S.cdist(X, X, metric="Euclidean")
    with raises(TypeError, f"pdist(): 'metric' must be {names}, not NoneType"):
        S.pdist(X, metric=None)
    with raises(TypeError, "pdist(): takes 1 positional argument but 2 were given"):
        S.pdist(X, "cityblock")  # SPT15: keyword only
    with raises(TypeError, "pdist(): unexpected keyword argument 'workers'"):
        S.pdist(X, workers=2)
    with raises(TypeError, "cdist(): unexpected keyword argument 'out'"):
        S.cdist(X, X, out=None)


def test_default_V_with_a_zero_variance():
    """A constant column makes the default V zero there: refused as a given V with a zero is, not NaN distances."""
    const = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
    with raises(ValueError, "pdist(): 'V' must be finite and positive"):
        S.pdist(const, metric="seuclidean", V=[1.0, 0.0])
    with raises(ValueError, "pdist(): the variance of a column of the data is 0, so the default 'V' is not positive; "
                            "give 'V'"):  # fmt: skip
        S.pdist(const, metric="seuclidean")
    for f, name in ((S.cdist, "cdist"), (S.kneighbors, "kneighbors"), (S.radius_neighbors, "radius_neighbors")):
        args = (const, const, 1.0) if f is S.radius_neighbors else (const, const)
        with raises(ValueError, f"{name}(): the variance of a column of the data is 0, so the default 'V' is not "
                                "positive; give 'V'"):  # fmt: skip
            f(*args, metric="seuclidean")
    assert np.all(np.isfinite(np.asarray(S.pdist(const, metric="seuclidean", V=[1.0, 1.0]))))
    # 0.1 repeated: the computed variance is a rounding residue (2.9e-34), not 0; the column is still constant
    tenth = np.array([[1.0, 0.1], [2.0, 0.1], [4.0, 0.1]])
    assert float(sd.var(sd.asarray(tenth), axis=0, correction=1)[1]) > 0.0
    for X in (tenth, np.full((7, 2), 0.1), np.column_stack([np.arange(1000.0), np.full(1000, 0.3)])):
        with raises(ValueError, "pdist(): the variance of a column of the data is 0, so the default 'V' is not "
                                "positive; give 'V'"):  # fmt: skip
            S.pdist(X, metric="seuclidean")
    with raises(ValueError, "cdist(): the variance of a column of the data is 0, so the default 'V' is not "
                            "positive; give 'V'"):  # fmt: skip
        S.cdist(tenth[:2], tenth[2:], metric="seuclidean")
    # one row has no pairs, and a NaN column is not constant: no error
    assert S.pdist(tenth[:1], metric="seuclidean").shape == (0,)
    nan = np.array([[1.0, math.nan], [2.0, math.nan], [4.0, math.nan]])
    assert np.all(np.isnan(np.asarray(S.pdist(nan, metric="seuclidean"))))


def test_options_of_another_metric():
    with raises(TypeError, "pdist(): 'p' applies only to metric='minkowski'"):
        S.pdist(X, p=3)
    with raises(TypeError, "cdist(): 'w' does not apply to metric='seuclidean'"):
        S.cdist(X, X, metric="seuclidean", w=np.ones(3))
    with raises(TypeError, "pdist(): 'w' does not apply to metric='haversine'"):
        S.pdist(X[:, :2], metric="haversine", w=np.ones(2))
    with raises(TypeError, "pdist(): 'V' applies only to metric='seuclidean'"):
        S.pdist(X, V=np.ones(3))
    with raises(TypeError, "pdist(): 'VI' applies only to metric='mahalanobis'"):
        S.pdist(X, metric="seuclidean", VI=np.eye(3))


def test_p():
    with raises(ValueError, "pdist(): 'p' must be a positive number or inf, not 0"):
        S.pdist(X, metric="minkowski", p=0)
    with raises(ValueError, "pdist(): 'p' must be a positive number or inf, not -1.0"):
        S.pdist(X, metric="minkowski", p=-1.0)
    with raises(ValueError, "minkowski(): 'p' must be a positive number or inf, not nan"):
        S.minkowski([0.0], [1.0], p=math.nan)
    with raises(TypeError, "minkowski(): 'p' must be an int or a float, not bool"):
        S.minkowski([0.0], [1.0], p=True)
    with raises(TypeError, "cdist(): 'p' must be an int or a float, not str"):
        S.cdist(X, X, metric="minkowski", p="3")
    assert S.minkowski([0.0, 0.0], [3.0, 4.0], p=0.5) == pytest.approx((3**0.5 + 4**0.5) ** 2)  # p < 1: taken
    assert S.minkowski([0.0, 0.0], [3.0, 4.0], p=np.float32(3.0)) == S.minkowski([0.0, 0.0], [3.0, 4.0], p=3)


def test_weights():
    with raises(ValueError, "pdist(): 'w' must have 3 entries (one per column), not 2"):
        S.pdist(X, w=[1.0, 2.0])
    with raises(ValueError, "pdist(): 'w' must have 1 dimension, not 2"):
        S.pdist(X, w=np.ones((1, 3)))
    for bad in (math.nan, math.inf, -1.0):
        with raises(ValueError, "euclidean(): 'w' must be finite and nonnegative"):
            S.euclidean([0.0, 1.0], [1.0, 1.0], w=[bad, 1.0])
    with raises(TypeError, "cityblock(): 'w' must be real, not a complex128 array"):
        S.cityblock([0.0], [1.0], w=[1j])


def test_V_and_VI():
    for bad in (0.0, -1.0, math.nan, math.inf):
        with raises(ValueError, "seuclidean(): 'V' must be finite and positive"):
            S.seuclidean([0.0, 1.0], [1.0, 1.0], [bad, 1.0])
    with raises(ValueError, "pdist(): 'V' must have 3 entries (one per column), not 1"):
        S.pdist(X, metric="seuclidean", V=[1.0])
    with raises(ValueError, "mahalanobis(): 'VI' must have shape (2, 2), not (3, 3)"):
        S.mahalanobis([0.0, 1.0], [1.0, 1.0], np.eye(3))
    with raises(ValueError, "cdist(): 'VI' must have shape (3, 3), not (3,)"):
        S.cdist(X, X, metric="mahalanobis", VI=np.ones(3))
    with raises(ValueError, "pdist(): the default 'VI' needs more observations than columns, not 3 observations "
                            "of 3 columns; give 'VI'"):  # fmt: skip
        S.pdist(X[:3], metric="mahalanobis")
    with raises(sd.linalg.LinAlgError, "pdist(): the covariance of the data is singular, so the default 'VI' does "
                                       "not exist; give 'VI'"):  # fmt: skip
        S.pdist(X, metric="mahalanobis")  # collinear columns
    with raises(TypeError, "mahalanobis(): missing 1 required positional argument: 'VI'"):
        S.mahalanobis([0.0], [1.0])


def test_data():
    with raises(ValueError, "pdist(): 'X' must have 2 dimensions, not 1"):
        S.pdist([1.0, 2.0])
    with raises(ValueError, "cdist(): 'XB' must have 2 dimensions, not 3"):
        S.cdist(X, np.zeros((1, 1, 3)))
    with raises(ValueError, "cdist(): 'XA' and 'XB' must have the same number of columns, not 3 and 2"):
        S.cdist(X, X[:, :2])
    with raises(TypeError, "pdist(): 'X' must be real, not a complex128 array"):
        S.pdist(np.array([[1j, 0.0], [0.0, 1.0]]))  # [scipy: ValueError]
    with raises(TypeError, "pdist(): 'X' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame "
                           "DataFrame, not dict"):  # fmt: skip
        S.pdist({"a": 1})
    with raises(TypeError, "euclidean(): 'u' must be an sd.ndarray, a sequence of numbers, a numpy array or a "
                           "frame DataFrame, not float"):  # fmt: skip
        S.euclidean(1.0, 2.0)
    with raises(TypeError, "pdist(): 'X' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame "
                           "DataFrame, not list"):  # fmt: skip
        S.pdist([["a", "b"]])
    with raises(ValueError, "pdist(): metric='haversine' needs 2 columns (latitude and longitude in radians), not 3"):
        S.pdist(X, metric="haversine")
    with raises(ValueError, "haversine(): 'u' and 'v' must have 2 entries on their last axis (latitude and "
                            "longitude in radians), not 3"):  # fmt: skip
        S.haversine([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])


def test_distance_function_shapes():
    with raises(ValueError, "euclidean(): 'u' and 'v' must have the same length on their last axis, not 3 and 2"):
        S.euclidean([1.0, 2.0, 3.0], [1.0, 2.0])
    with raises(ValueError, "cosine(): shapes could not be broadcast together: (2, 3) (4, 3)"):
        S.cosine(np.ones((2, 3)), np.ones((4, 3)))
    with raises(TypeError, "correlation(): 'centered' must be True or False, not int"):
        S.correlation([1.0, 2.0], [2.0, 1.0], centered=1)
    with raises(TypeError, "euclidean(): takes 2 positional arguments but 3 were given"):
        S.euclidean([1.0], [2.0], [1.0])


def test_outputs_beyond_physical_memory():
    with raises(MemoryError, f"pdist(): cannot allocate {8 * (2**22 * (2**22 - 1) // 2)} bytes: more than this "
                             "machine's physical memory"):  # fmt: skip
        S.pdist(np.zeros((2**22, 0)))
    with raises(MemoryError, f"cdist(): cannot allocate {8 * 2**42} bytes: more than this machine's physical memory"):
        S.cdist(np.zeros((2**21, 0)), np.zeros((2**21, 0)))
