"""The errors of kneighbors and radius_neighbors (docs/api/spatial.md §1.2, §1.4, §3, §4; API.md §2, §2.1):
every wrong argument, its type and its message, which starts with the function's name. scipy's and
sklearn's silent cases (negative or NaN radius and upper bound, k=True, k=2.0) and their NaN data are
errors here (SPT12, SPT19)."""

import math
import re

import numpy as np
import pytest

import stardust as sd

S = sd.spatial
X = np.arange(12.0).reshape(4, 3) % 5


def raises(exc, msg):
    return pytest.raises(exc, match="^" + re.escape(msg) + "$")


def test_call_shape():
    with raises(TypeError, "kneighbors(): takes 2 positional arguments but 3 were given"):
        S.kneighbors(X, X, 3)  # k is keyword only (SPT15)
    with raises(TypeError, "kneighbors(): unexpected keyword argument 'workers'"):
        S.kneighbors(X, X, workers=2)
    with raises(TypeError, "radius_neighbors(): unexpected keyword argument 'return_sorted'"):
        S.radius_neighbors(X, X, 1.0, return_sorted=True)
    with raises(TypeError, "radius_neighbors(): missing 1 required positional argument: 'r'"):
        S.radius_neighbors(X, X)


def test_k():
    with raises(ValueError, "kneighbors(): 'k' must be at least 1, not 0"):
        S.kneighbors(X, X, k=0)
    with raises(TypeError, "kneighbors(): 'k' must be an int or a sequence of ints, not bool"):
        S.kneighbors(X, X, k=True)
    with raises(TypeError, "kneighbors(): 'k' must be an int or a sequence of ints, not float"):
        S.kneighbors(X, X, k=2.0)
    with raises(ValueError, "kneighbors(): 'k' must not be an empty sequence"):
        S.kneighbors(X, X, k=[])
    with raises(ValueError, "kneighbors(): every entry of 'k' must be at least 1, not 0"):
        S.kneighbors(X, X, k=[1, 0])  # scipy's k=[0] kills its process
    with raises(MemoryError, "kneighbors(): cannot allocate 640000000000000 bytes: more than this machine's physical "
                             "memory"):  # fmt: skip
        S.kneighbors(X, X, k=10**13)


def test_upper_bound():
    with raises(ValueError, "kneighbors(): 'distance_upper_bound' must be a number >= 0 or inf, not -1"):
        S.kneighbors(X, X, distance_upper_bound=-1)  # scipy squares it: -1 acts as 1
    with raises(ValueError, "kneighbors(): 'distance_upper_bound' must be a number >= 0 or inf, not nan"):
        S.kneighbors(X, X, distance_upper_bound=math.nan)
    with raises(TypeError, "kneighbors(): 'distance_upper_bound' must be an int or a float, not str"):
        S.kneighbors(X, X, distance_upper_bound="1")


def test_radius():
    with raises(ValueError, "radius_neighbors(): 'r' must be a number >= 0 or inf, not -1"):
        S.radius_neighbors(X, X, -1)
    with raises(ValueError, "radius_neighbors(): 'r' must be a number >= 0 or inf, not nan"):
        S.radius_neighbors(X, X, math.nan)
    with raises(ValueError, "radius_neighbors(): every entry of 'r' must be a number >= 0 or inf"):
        S.radius_neighbors(X, X, [1.0, -1.0, 1.0, 1.0])
    with raises(ValueError, "radius_neighbors(): 'r' of shape (2,) does not broadcast to the 4 rows of 'XA'"):
        S.radius_neighbors(X, X, [1.0, 2.0])
    with raises(TypeError, "radius_neighbors(): 'r' must be an int or a float, not bool"):
        S.radius_neighbors(X, X, True)
    with raises(TypeError, "radius_neighbors(): 'r' must be a number or an array of numbers, not str"):
        S.radius_neighbors(X, X, "1")
    with raises(TypeError, "radius_neighbors(): 'return_length' must be True or False, not int"):
        S.radius_neighbors(X, X, 1.0, return_length=1)


def test_data():
    with raises(ValueError, "kneighbors(): 'XA' must be finite"):
        S.kneighbors([[math.nan, 0.0, 0.0]], X)  # sklearn: "Input X contains NaN."
    with raises(ValueError, "kneighbors(): 'XB' must be finite"):
        S.kneighbors(X, [[math.inf, 0.0, 0.0]])
    with raises(ValueError, "radius_neighbors(): 'XB' must be finite"):
        S.radius_neighbors(X, [[0.0, -math.inf, 0.0]], 1.0, metric="cityblock")
    with raises(ValueError, "kneighbors(): 'XA' must have 2 dimensions, not 1"):
        S.kneighbors(X[0], X)
    with raises(ValueError, "kneighbors(): 'XA' and 'XB' must have the same number of columns, not 3 and 2"):
        S.kneighbors(X, X[:, :2])
    with raises(TypeError, "radius_neighbors(): 'XB' must be real, not a complex128 array"):
        S.radius_neighbors(X, [[1j, 0.0, 0.0]], 1.0)


def test_metric_options():
    with raises(TypeError, "radius_neighbors(): 'p' applies only to metric='minkowski'"):
        S.radius_neighbors(X, X, 1.0, p=3)
    with raises(TypeError, "kneighbors(): 'w' does not apply to metric='haversine'"):
        S.kneighbors(X[:, :2], X[:, :2], metric="haversine", w=[1.0, 1.0])
    with raises(ValueError, "kneighbors(): 'w' must be finite and >= 0"):
        S.kneighbors(X, X, w=[1.0, math.nan, 1.0])
    with raises(ValueError, "kneighbors(): metric='haversine' needs 2 columns (latitude and longitude in radians), "
                            "not 3"):  # fmt: skip
        S.kneighbors(X, X, metric="haversine")
    with pytest.raises(ValueError, match=r"^kneighbors\(\): 'metric' must be 'euclidean', .* not 'e'$"):
        S.kneighbors(X, X, metric="e")
