"""sd.interpolate's references and their contract (docs/design/interpolate.md §4, §7), as the tests and the
accuracy measurement (bench/accuracy) will both judge them.

tests/interpolate/ref/<family>.npz (tools/interpolate_ref.py; layout: tests/reffile.py) holds one point per
evaluation point, node, coefficient or root, with the inputs "set" (the id in tests/interpolate/samples.py,
"lerp/<id>" and "curve/<id>" in the linear family), "index" (the point's number in its set) and
"digest" (the data's digest), the family's own inputs, and per method the reference hi + lo:

    linear      x0, x1, a, b, x; method lerp (the exact a + (x - x0) / (x1 - x0) (b - a))
    slopes      x (the node); methods natural, clamped, not_a_knot, periodic, pchip, akima, makima
    bspline     grouped by curve (group input "set" and the knots t_<method>); methods k0 .. k5,
                k3_natural, k3_clamped, k3_periodic, k5_periodic (the coefficients), and over the groups
                ainv_<method> (the exact infinity norm of the inverse of the system's matrix)
    curves      x; the slopes' and the B-splines' methods: the exact interpolant's value
    polynomial  x; methods value, d1, d2, lebesgue, abs_terms, fh3
    grid        x (the point (d,)); methods linear, cubic, quintic, pchip
    rbf         x (the point (d,)); one method per kernel, thin_plate_spline_s0.1 (smoothing 0.1) and
                thin_plate_spline_k8 (neighbors = 8)
    roots       piece, multiplicity; method root

tests/interpolate/data holds the other libraries' records (tools/interpolate_probe.py: scipy.json,
numpy.json, r.json), each with its own "_provenance".

The contract of the linear kernel (docs/design/interpolate.md §3.2 proves it, §4): with
t = (x - x0) / (x1 - x0) exact and y = a + t (b - a) exact, the computed value is within
gamma_6 (|a| + |b|) + 3 eta of y for 0 <= t <= 1 and within gamma_10 max(1, |t|) (|a| + |b|) + 3 eta outside
(decision IP40), eta = 2^-1075 (the products that underflow); at the nodes it is exact. 3 eta = 1.5 2^-1074 is
not a double; the bound uses the double above it, 2^-1073 (ETA3). The other contracts are drafts until their
milestones prove them (§4).
"""

from __future__ import annotations

import json
import math
import pathlib

from .. import reffile

REF = pathlib.Path(__file__).parent / "ref"
DATA = pathlib.Path(__file__).parent / "data"
FAMILIES = ("bspline", "curves", "grid", "linear", "polynomial", "rbf", "roots", "slopes")
U = 2.0**-53
ETA3 = 2.0**-1073  # 3 eta rounded up to a double (eta = 2^-1075 itself rounds to 0)


def names() -> list[str]:
    return sorted(p.stem for p in REF.glob("*.npz"))


def ref(family: str) -> reffile.Ref:
    return reffile.read(REF / f"{family}.npz")


def data(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def gamma(k: int) -> float:
    return k * U / (1 - k * U)


def linear_bound(a: float, b: float, t: float) -> float:
    """The linear kernel's contract: the absolute bound at t (the exact t of the point)."""
    s = abs(a) + abs(b)
    if 0.0 <= t <= 1.0:
        return gamma(6) * s + ETA3
    return gamma(10) * max(1.0, abs(t)) * s + ETA3


def linear_units(got: float, hi: float, lo: float, a: float, b: float, t: float) -> float:
    """|got - ref| in units of the linear contract (0 when both are exactly equal)."""
    err = abs((got - hi) - lo)
    if err == 0:
        return 0.0
    return err / linear_bound(a, b, t)


def isclose_dd(got: float, hi: float, lo: float) -> bool:
    """got is the reference rounded to a double (hi), or the reference is exactly a double."""
    return got == hi or (math.isnan(got) and math.isnan(hi))
