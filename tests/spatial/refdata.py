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

The contract (docs/design/spatial.md §4, measured in SPT1), in units of its bound (1 or less is within
it): the depth of a pair's sum is m = min(n, 32) + ceil(n / 32) - 1 (the tile order, decision SPT26), u =
2^-53, and every bound adds ETA = 2^-1074 (the smallest subnormal: a subnormal result is off by up to half
of it when correctly rounded, and the double-double reference holds nothing finer):

    chebyshev, hamming, jaccard     correctly rounded (half an ULP)
    euclidean, w_euclidean          (gamma_{m+3} / 2 + u) d
    seuclidean                      (gamma_{m+4} / 2 + u) d (the rounding of 1 / V too)
    sqeuclidean, cityblock          gamma_{m+2} s; w_sqeuclidean gamma_{m+3} s
    minkowski                       (4u + (gamma_m + 2 c_pow) / p + c_pow) d, c_pow = gamma_2 (SPT30, SPT80)
    canberra                        gamma_{m+3} d; braycurtis gamma_{2m+3} d
    cosine                          u (3 sqrt(d) + (2m + 8) d) + (m + 4)^2 u^2 (SPT29: normalized rows)
    correlation                     cosine's + 2 (rho_x^2 + rho_y^2), rho = gamma_{2m+2} max|x_i| sqrt(sum w)
                                    / ||x_c||_w (the rounding of the mean, second order in it)
    mahalanobis                     min(E / d, sqrt(E)) + u d, E = gamma_{2m+3} |z|^T |VI| |z|
    w_hamming, w_jaccard            gamma_{2m+1} d
    haversine                       C_HAVERSINE u d (SPT31)
"""

from __future__ import annotations

import functools
import json
import math
import pathlib

import numpy as np

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("distances", "counting", "haversine", "neighbors")
U = 2.0**-53
TILE_BLOCK = 32  # the tile order's block of terms (csrc/spatial/metric.hpp, SPT26)
C_POW = 2 * U / (1 - 2 * U)  # gamma_2: libm pow within one ULP (SPT0), and p = 3's (a a) a (SPT80)
ETA = 2.0**-1074
C_HAVERSINE = 20.0  # SPT31


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


@functools.cache
def ref(family: str) -> reffile.Ref:
    """The family's file (read once; callers must not change it)."""
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


def _half_ulp(v: float) -> float:
    return 0.5 * math.ulp(v) if v else 0.0


def _rho(x, w) -> float:
    """correlation's relative shift of a row by the rounding of its mean (see the module docstring)."""
    x = np.asarray(x, dtype=np.float64)
    w = np.ones(x.size) if w is None else np.asarray(w, dtype=np.float64)
    keep = w > 0
    sw = math.fsum(w)
    if not keep.any() or sw == 0:
        return math.inf
    top = float(np.abs(x).max())
    if top > 0 and math.isfinite(top):
        x = np.ldexp(x, -math.frexp(top)[1])  # rho does not depend on the scale; no overflow below
    mu = math.fsum(w * x) / sw
    r = x - mu
    r = r - math.fsum(w * r) / sw  # a second pass: the centered row to the last digits that matter here
    nc = math.sqrt(math.fsum(w * r * r))
    if nc == 0:
        return math.inf
    return gamma(2 * depth(x.size) + 2) * float(np.abs(x[keep]).max()) * math.sqrt(sw) / nc


def bound(metric: str, value: float, *, n: int, p: float | None = None, u=None, v=None, w=None, VI=None) -> float:
    """The contract's absolute bound for a distance of n terms with true value `value` (the method names
    of tests/spatial/ref: euclidean, w_euclidean, minkowski_p3, ...; correlation and mahalanobis need the
    pair's inputs u, v (and w, VI))."""
    m = depth(n)
    d = abs(value)
    base = metric[2:] if metric.startswith("w_") else metric
    weighted = metric.startswith("w_")
    if base.startswith("minkowski"):
        p = float(base[len("minkowski_p") :]) if p is None else p
        base = "minkowski"
    if base in ("chebyshev",) or (base in ("hamming", "jaccard") and not weighted):
        return _half_ulp(d) + ETA
    if base in ("hamming", "jaccard"):
        return gamma(2 * m + 1) * d + ETA
    if base == "euclidean":
        return (gamma(m + 3) / 2 + U) * d + ETA
    if base == "seuclidean":
        return (gamma(m + 4) / 2 + U) * d + ETA
    if base == "sqeuclidean":
        return gamma(m + (3 if weighted else 2)) * d + ETA
    if base == "cityblock":
        return gamma(m + 2) * d + ETA
    if base == "minkowski":
        return (4 * U + (gamma(m) + 2 * C_POW) / p + C_POW) * d + ETA
    if base == "canberra":
        return gamma(m + 3) * d + ETA
    if base == "braycurtis":
        return gamma(2 * m + 3) * d + ETA
    if base in ("cosine", "correlation"):
        b = U * (3 * math.sqrt(d) + (2 * m + 8) * d) + (m + 4) ** 2 * U * U + ETA
        if base == "correlation":
            b += 2 * (_rho(u, w) ** 2 + _rho(v, w) ** 2)
        return b
    if base == "mahalanobis":
        z = np.abs(np.asarray(u, dtype=np.float64) - np.asarray(v, dtype=np.float64))
        top = float(z.max()) if z.size else 0.0
        e = -math.frexp(top)[1] if top > 0 and math.isfinite(top) else 0  # scaled: no underflow in E
        zs, ds = np.ldexp(z, e), math.ldexp(d, e)
        E = gamma(2 * m + 3) * float(zs @ np.abs(np.asarray(VI, dtype=np.float64)) @ zs)
        return math.ldexp((min(E / ds, math.sqrt(E)) if ds else math.sqrt(E)) + U * ds, -e) + ETA
    if base == "haversine":
        return C_HAVERSINE * U * d + ETA
    raise KeyError(metric)


def units(metric: str, got: float, hi: float, lo: float, **inputs) -> float:
    """|got - ref| in units of the contract's bound (0 when both are exactly equal; inf for a NaN
    against a number). inputs: n, and p, u, v, w, VI where bound() needs them."""
    if math.isnan(got) or math.isnan(hi):
        return 0.0 if math.isnan(got) and math.isnan(hi) else math.inf
    if math.isinf(hi) or math.isinf(got):
        return 0.0 if got == hi else math.inf
    err = abs((got - hi) - lo)
    if err == 0:
        return 0.0
    b = bound(metric, hi, **inputs)
    return math.inf if b == 0 else err / b


def call(mod, method: str, u, v, w=None, V=None, VI=None):
    """The distance `method` (a method name of tests/spatial/ref) of u and v through a module with
    scipy's distance-function names: sd.spatial or scipy.spatial.distance."""
    weighted = method.startswith("w_")
    base = method[2:] if weighted else method
    kw = {"w": w} if weighted else {}
    if base.startswith("minkowski_p"):
        return mod.minkowski(u, v, p=float(base[len("minkowski_p") :]), **kw)
    if base == "seuclidean":
        return mod.seuclidean(u, v, V)
    if base == "mahalanobis":
        return mod.mahalanobis(u, v, VI)
    return getattr(mod, base)(u, v, **kw)


def inputs(family: str, k: int) -> dict:
    """Point k's inputs as the distance functions take them (counting: u and v in their own dtype)."""
    r = ref(family)
    out = {name: r.inputs[name][k] for name in ("u", "v", "w", "V", "VI") if name in r.inputs}
    if family == "counting":
        dt = str(r.inputs["dtype"][k])
        out["u"], out["v"] = out["u"].astype(dt), out["v"].astype(dt)
    return out


def point_units(family: str, method: str, k: int, got: float, j: int) -> float:
    """units() of a result at point k (value j of the method) with the inputs bound() needs."""
    r = ref(family)
    m = r.methods[method]
    x = inputs(family, k)
    w = x.get("w") if method.startswith("w_") else None
    return units(method, float(got), float(m.hi[j]), float(m.lo[j]), n=len(x["u"]), u=x["u"], v=x["v"], w=w,
                 VI=x.get("VI"))  # fmt: skip
