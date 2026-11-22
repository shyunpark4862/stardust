"""sd.interpolate's accuracy sweeps: one per reference family of tests/interpolate/ref
(tools/interpolate_ref.py) once the functions it measures are public.

IP1 brings

  linear       sd.interp on linear.npz (exact a + t (b - a), rounded once) in units of the linear contract
               (tests/interpolate/refdata.linear_bound; docs/design/interpolate.md §4): region `curves`, the
               points inside [x[0], x[-1]] of every real curve of tests/interpolate/samples.py (sd.interp on the
               whole curve), and region `intervals`, the points inside the kernel intervals of samples.lerps()
               (sd.interp on the interval's two nodes: the doubles next to the ends, opposite signs, nearly equal
               values, signed zeros, subnormal and overflowing widths). Outside the domain sd.interp is numpy's
               clamp, not the line, so those points are not part of the sweep. Comparators, live on the same
               points: numpy 2.5.4's interp, scipy 1.18.1's make_interp_spline(k=1) and R 4.6.2's
               approx(rule = 2).

The plan for the others (docs/design/interpolate.md §7.5), each judged through tests/interpolate/refdata.py with
the end-to-end error (against curves.npz) as a column of its own:

  coefficients IP2, IP3: the slopes (slopes.npz) and B-spline coefficients (bspline.npz, scaled by the
               ainv_<method> norms) against the exact ones; comparator scipy
  evaluation   IP2, IP3: the piecewise polynomials and B-splines evaluated from their own coefficients
               (exact rational evaluation at test time) and end to end (curves.npz); comparators scipy and
               R's splinefun (natural, periodic)
  polynomial   IP4: Barycentric, Krogh and Floater-Hormann on polynomial.npz (the Lebesgue function and
               sum |l_j y_j| are the contract's scale)
  aaa          IP4: AAA's residual on the samples and its value against Arb evaluation of the computed
               rational
  grid         IP5: RegularGridInterpolator's six methods on grid.npz (the 1-D linear also joins `linear`)
  rbf          IP6: RBFInterpolator on rbf.npz (eight kernels, smoothing, neighbors)
  roots        IP2: PPoly.roots on roots.npz (the roots found and missed)
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .common import Custom, compare_libs, region, ulps


def _sets():
    """[(region, set id, nodes x, values y, points inside)] of linear.npz's sets."""
    from tests.interpolate import samples as S

    out = []
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        out.append(("curves", f"curve/{cid}", c.x, c.y, c.xe))
    for L in S.lerps():
        if S.lerp_finite(L):
            out.append(("intervals", f"lerp/{L.id}", np.array([L.x0, L.x1]), np.array([L.a, L.b]), L.x))
    return out


def _values(lib, sets):
    """lib's values at every set's points, one array per set (None: lib not run here)."""
    if lib == "sd":
        import stardust as sd

        return [np.asarray(sd.interp(sd.asarray(q), sd.asarray(x), sd.asarray(y))) for _, _, x, y, q in sets]
    if lib == "numpy":
        return [np.interp(q, x, y) for _, _, x, y, q in sets]
    if lib == "scipy":
        import scipy.interpolate as si

        return [si.make_interp_spline(x, y, k=1)(q) for _, _, x, y, q in sets]
    if lib == "R":
        from . import rsession

        expr = (
            "{ ox <- c(0, cumsum(NX)); oq <- c(0, cumsum(NQ)); unlist(lapply(seq_along(NX), function(k) { "
            "i <- (ox[k] + 1):ox[k + 1]; j <- (oq[k] + 1):oq[k + 1]; approx(X[i], Y[i], xout = Q[j], rule = 2)$y })) }"
        )
        flat = rsession.evaluate_named(
            expr,
            {"X": np.concatenate([s[2] for s in sets]), "Y": np.concatenate([s[3] for s in sets]),
             "Q": np.concatenate([s[4] for s in sets]), "NX": np.array([s[2].size for s in sets], dtype=float),
             "NQ": np.array([s[4].size for s in sets], dtype=float)},
        )  # fmt: skip
        out, k = [], 0
        for s in sets:
            out.append(flat[k : k + s[4].size])
            k += s[4].size
        return out
    return None


def _linear_regions(lib):
    from tests.interpolate import refdata

    r = refdata.ref("linear")
    m = r.methods["lerp"]
    sets_of = r.inputs["set"][m.index]
    sets = _sets()
    vals = _values(lib, sets)
    if vals is None:
        return None
    acc = {}
    for (reg, sid, x, _, q), got in zip(sets, vals):
        pos = np.flatnonzero(sets_of == sid)
        idx = m.index[pos]
        order = np.argsort(r.inputs["index"][idx], kind="stable")
        pos, idx = pos[order], idx[order]
        inside = (q >= x[0]) & (q <= x[-1])
        e = acc.setdefault(reg, {"where": [], "contract": [], "ulp": [], "rel": [], "excluded": 0})
        for p, k, g in zip(pos[inside], idx[inside], np.asarray(got, dtype=float)[inside]):
            x0, x1, a, b, v = (float(r.inputs[n][k]) for n in ("x0", "x1", "a", "b", "x"))
            hi, lo = float(m.hi[p]), float(m.lo[p])
            if not np.isfinite(g):  # a comparator's overflow (numpy's slope on subnormal spacing)
                e["excluded"] += 1
                continue
            t = float((Fraction(v) - Fraction(x0)) / (Fraction(x1) - Fraction(x0)))
            e["where"].append(f"{sid}#{int(r.inputs['index'][k])}")
            e["contract"].append(refdata.linear_units(float(g), hi, lo, a, b, t))
            e["ulp"].append(float(ulps(g, hi)))
            with np.errstate(all="ignore"):
                e["rel"].append(abs((g - hi) - lo) / abs(hi) if hi else abs(g - hi))
    out = {}
    for reg, e in acc.items():
        out[reg] = region(e["where"], ulp=np.array(e["ulp"]), rel=np.array(e["rel"]), contract=np.array(e["contract"]),
                          bound=1.0, excluded=e["excluded"] if lib != "sd" else None)  # fmt: skip
    return out


def sweeps():
    import stardust as sd

    if not hasattr(sd, "interp"):
        return []
    return [Custom("interpolate", "linear", lambda: _linear_regions("sd"),
                   compare_libs(_linear_regions, ("numpy", "scipy", "R"), seconds=120.0))]  # fmt: skip
