"""sd.spatial's references and their contract (docs/design/spatial.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy) will both judge them.

tests/spatial/ref/<family>.npz (tools/spatial_ref.py; layout: tests/reffile.py) holds one point per
pair or query: the inputs "set" (the id in tests/spatial/points.py, or "edge/<id>", "haversine/<id>"),
"pair" (the pair's or query's number in its set), "digest" (points.digest of the point's own inputs)
and the family's inputs, and per method the reference hi + lo (a double-double):

    distances  u, v, w, V, VI; methods euclidean, sqeuclidean, cityblock, chebyshev,
               minkowski_p0.5 / p1.5 / p3 / p7.5, seuclidean, mahalanobis (n <= 16), cosine,
               correlation, canberra, braycurtis and the weighted w_<metric>
    counting   u, v (int64 in the file; input dtype: 'bool' or 'int64', the point's own), w; methods
               hamming, jaccard, w_hamming, w_jaccard
    haversine  u, v ((latitude, longitude) in radians); method haversine
    neighbors  query; methods p1, p2, pinf: the distance to every data point of the set, in order

tests/spatial/data holds the other libraries' records (tools/spatial_probe.py: scipy.json,
sklearn.json, r.json), each with its own "_provenance".

The contract (draft until SPT1 measures it; docs/design/spatial.md §4), in units of its bound (1 or
less is within it): the depth of a pair's sum is m = min(n, 32) + ceil(n / 32) - 1 (the tile order,
decision SPT26); chebyshev is exact; euclidean, seuclidean and the weighted forms |d - ref| <=
(gamma_{m+3} / 2 + u) ref; sqeuclidean and cityblock |s - ref| <= gamma_{m+2} ref; minkowski
(2u + (gamma_m + c_pow) / p + c_pow) ref with c_pow = 2u (SPT0: libm's pow within 0.504 ULP where
the result is normal).
"""

from __future__ import annotations

import json
import math
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("distances", "counting", "haversine", "neighbors")
U = 2.0**-53
TILE_BLOCK = 32  # the tile order's block of terms (csrc/spatial/pair_kernel.hpp, SPT26)
C_POW = 2 * U


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def depth(n: int) -> int:
    """The rounding depth of a pair's sum of n terms in the tile order."""
    if n == 0:
        return 0
    return min(n, TILE_BLOCK) + math.ceil(n / TILE_BLOCK) - 1


def gamma(k: int) -> float:
    return k * U / (1 - k * U)


def bound(metric: str, n: int, value: float, p: float | None = None) -> float:
    """The contract's absolute bound for a distance of n terms with true value `value`."""
    m = depth(n)
    v = abs(value)
    if metric == "chebyshev":
        return 0.5 * math.ulp(v) if v else 0.0
    if metric in ("euclidean", "seuclidean", "w_euclidean"):
        return (gamma(m + 3) / 2 + U) * v
    if metric in ("sqeuclidean", "cityblock", "w_sqeuclidean", "w_cityblock"):
        return gamma(m + 2) * v
    if metric.startswith(("minkowski", "w_minkowski")):
        return (2 * U + (gamma(m) + C_POW) / p + C_POW) * v
    raise KeyError(metric)


def units(metric: str, n: int, got: float, hi: float, lo: float, p: float | None = None) -> float:
    """|got - ref| in units of the contract's bound (0 when both are exactly equal)."""
    err = abs((got - hi) - lo)
    if err == 0:
        return 0.0
    b = bound(metric, n, hi, p)
    return math.inf if b == 0 else err / b
