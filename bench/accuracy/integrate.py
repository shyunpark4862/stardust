"""sd.integrate's accuracy sweeps (docs/design/integrate.md §4, §7.5): one per function family on
the reference families of tests/integrate/ref (tools/integrate_ref.py).

Since milestone C3:
  samples     the sample rules on the sample sets of tests/integrate/problems.py against each
              rule's formula computed exactly from the same doubles (ref/samples.json). Contract:
              the error within c eps S, S the terms' size (sum |d_i| (|y_i| + |y_i+1|)), c =
              ceil(log2 n) + 10 for a sum (the pairwise tree), k + 10 for the k-th running value
              (a scan); bound 1. The relative error is recorded per rule.
  fixed_quad  n-point Gauss-Legendre (n = 5, 20, 100) on the collection's finite smooth integrals
              without a weight, against the Arb integral (ref/integrals.json): recorded only (the
              rule's own error, not a contract). Its nodes and weights are correctly rounded
              (tests/integrate/test_fixed.py) and its sum is within the pairwise bound.
scipy.integrate 1.18.1 is the comparator: the same formulas (its sums in numpy's order) and its
roots_legendre.

The quad family (C4, C5), tanhsinh and nsum (C6), quad_vec and cubature (C7) join with their
milestones, measuring the relative error and the honesty of the error estimate.
"""

from __future__ import annotations

import math

import numpy as np

from .common import Custom, compare_libs, region

EPS = 2.0**-52
RULES = ("trapezoid", "simpson", "romb", "cumulative_trapezoid", "cumulative_simpson")


def _lib(lib):
    if lib == "sd":
        import stardust as sd

        return sd.integrate, sd.asarray, lambda v: np.asarray(v.tolist() if isinstance(v, sd.ndarray) else v)
    if lib == "scipy":
        import scipy.integrate as si

        return si, np.asarray, np.asarray
    return None


def _samples(lib="sd"):
    m = _lib(lib)
    if m is None:
        return None
    I, wrap, back = m
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.samples()
    out = {}
    for rule in RULES:
        where, rel, contract = [], [], []
        for sid, (y, x, dx) in P.samples().items():
            e = ref[sid]
            if rule not in e:
                continue
            kw = {"x": wrap(x)} if x is not None else {"dx": dx}
            got = back(getattr(I, rule)(wrap(y), **kw)).astype(float)
            r = np.asarray(e[rule], dtype=float)
            err = np.abs((got - r[..., 0]) - r[..., 1])
            d = np.abs(np.diff(x)) if x is not None else np.full(y.shape[-1] - 1, abs(dx))
            a = np.abs(y)
            scale = np.maximum(np.cumsum(d * (a[..., 1:] + a[..., :-1]), axis=-1), 1e-300)
            if rule.startswith("cumulative"):
                bound = (np.arange(1, err.shape[-1] + 1) + 10) * EPS * scale
            else:
                bound = (math.ceil(math.log2(max(y.shape[-1], 2))) + 10) * EPS * scale[..., -1]
            where.append(sid)
            rel.append(float(np.max(err / np.maximum(np.abs(r[..., 0]), 1e-300))))
            contract.append(float(np.max(err / bound)))
        out[rule] = region(where, rel=np.array(rel), contract=np.array(contract), bound=1.0)
    return out


def _fixed_quad(lib="sd"):
    m = _lib(lib)
    if m is None:
        return None
    I, wrap, _ = m
    from tests.integrate import problems as P
    from tests.integrate import refdata

    ref = refdata.integrals()
    out = {}
    for n in (5, 20, 100):
        where, rel = [], []
        for p in P.INTEGRALS:
            if not (p.numeric and p.weight is None and math.isfinite(p.a) and math.isfinite(p.b)):
                continue
            with np.errstate(all="ignore"):
                v = I.fixed_quad(lambda t, p=p: wrap(p.f_np(np.asarray(t.tolist() if hasattr(t, "tolist") else t))),
                                 p.a, p.b, n=n)  # fmt: skip
            v = float(v[0] if isinstance(v, tuple) else v)
            hi, lo = ref[p.id]["value"]
            err = abs((v - hi) - lo)
            where.append(p.id)
            rel.append(err / abs(hi) if hi else err)
        out[f"n={n}"] = region(where, rel=np.array(rel))
    return out


def sweeps():
    specs = [("samples", _samples), ("fixed_quad", _fixed_quad)]
    return [Custom("integrate", name, fn, compare_libs(fn, ("scipy",), seconds=120)) for name, fn in specs]
