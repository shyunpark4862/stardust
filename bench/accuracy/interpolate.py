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

IP2 brings (judged through tests/interpolate/refdata.py, the functions the tests use)

  coefficients the node slopes of CubicSpline (natural, clamped, not_a_knot, periodic), PchipInterpolator and
               Akima1DInterpolator (akima, makima) against slopes.npz in units of their contracts (the spline's
               componentwise gamma_12 |A^-1|(|A||s| + |b|), PCHIP's 40 u max|D|, Akima's bound with its
               conditioning), one region per method, at the nodes but the last (the coefficients' c1 there, which
               is the slope itself for every library); the real curves whose divided differences are doubles and
               whose coefficients do not overflow. Comparator: scipy (its coefficients' c1).
  evaluation   the same objects evaluated (nu = 0) at every curve's evaluation points, the end pieces extended, in
               units of the evaluation contract (ppoly_bound) against the exact value of the object's own
               coefficients; `e2e_rel` is the end-to-end relative error against curves.npz (the exact data's exact
               interpolant), relative to max(|value|, max|y|). Comparators: scipy (its own coefficients, its own
               evaluation) and R's splinefun (natural, periodic; end to end only, inside [x[0], x[-1]]: R's
               natural spline extrapolates linearly).
  roots        PPoly.roots(extrapolate=False, discontinuity=False) on roots.npz: per reference root the distance to
               the nearest returned root in ulps (`missed`: the roots farther than 2^-20 relative), and the
               returned roots that are none of the references (`extra`). Comparator: scipy.

IP3 brings (judged through tests/interpolate/refdata.py)

  coefficients also the B-spline coefficients of make_interp_spline (k2 .. k5, k3_natural, k3_clamped,
               k3_periodic, k5_periodic; one region per method; k0 and k1 copy y) against bspline.npz in units of
               the construction contract 2 gamma_{8k+8} |A^-1| |A| |c| (refdata.collocation_bound); the real curves
               the method allows (subnormal/x's natural and clamped overflow: left out). Comparator: scipy.
  evaluation   also the B-splines (the same methods and k1, nu = 0, the end pieces extended) in units of the
               evaluation contract (refdata.bspline_bound; k1: the linear kernel's) against the exact value of the
               object's own coefficients, and `e2e_rel` against curves.npz. Comparator: scipy.

The plan for the others (docs/design/interpolate.md §7.5):

  polynomial   IP4: Barycentric, Krogh and Floater-Hormann on polynomial.npz (the Lebesgue function and
               sum |l_j y_j| are the contract's scale)
  aaa          IP4: AAA's residual on the samples and its value against Arb evaluation of the computed
               rational
  grid         IP5: RegularGridInterpolator's six methods on grid.npz (the 1-D linear also joins `linear`)
  rbf          IP6: RBFInterpolator on rbf.npz (eight kernels, smoothing, neighbors)
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


CUBIC = ("natural", "clamped", "not_a_knot", "periodic", "pchip", "akima", "makima")
_BC = {"natural": ((2, 0.0), (2, 0.0)), "clamped": ((1, 0.0), (1, 0.0)), "not_a_knot": ("not-a-knot", "not-a-knot"),
       "periodic": ("periodic", "periodic")}  # fmt: skip


def _cubic_curves():
    from tests.interpolate import samples as S

    out = []
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        with np.errstate(all="ignore"):
            if not np.all(np.isfinite(np.diff(c.y) / np.diff(c.x))) or cid == "large/1e300":
                continue
        out.append(c)
    return out


def _cubic_object(lib, meth, c):
    """lib's interpolant of meth on curve c (None: not made for it), the end pieces extended."""
    if meth == "periodic" and "periodic" not in c.tags:
        return None
    if lib == "sd":
        import stardust as sd

        mod = sd.interpolate
    else:
        import scipy.interpolate as mod
    if meth in _BC:
        bc = {"natural": "natural", "clamped": "clamped", "not_a_knot": "not-a-knot", "periodic": "periodic"}[meth]
        return mod.CubicSpline(c.x, c.y, bc_type=bc)
    if meth == "pchip":
        return mod.PchipInterpolator(c.x, c.y)
    return mod.Akima1DInterpolator(c.x, c.y, method=meth, extrapolate=True)


def _slope_bound(meth, c, s):
    from tests.interpolate import refdata

    if meth in _BC:
        return refdata.spline_bound(c.x, c.y, _BC[meth], s)
    if meth == "pchip":
        return refdata.pchip_bound(c.x, c.y)
    return refdata.akima_bound(c.x, c.y, meth == "makima")


def _bspline_object(lib, method, c):
    """lib's make_interp_spline of a bspline.npz method on curve c (None: the method does not apply)."""
    from tests.interpolate import refdata

    k, bc = refdata.bspline_args(method)
    if (bc == "periodic" and "periodic" not in c.tags) or (bc in ("natural", "clamped") and c.id == "subnormal/x"):
        return None
    if c.n < (2 if bc in ("natural", "clamped") else k + 1):
        return None
    if lib == "sd":
        import stardust as sd

        return sd.interpolate.make_interp_spline(c.x, c.y, k=k, bc_type=bc)
    import scipy.interpolate as si

    try:
        return si.make_interp_spline(c.x, c.y, k=k, bc_type=bc)
    except ValueError:  # subnormal/x for k >= 2: scipy's collocation matrix overflows (its weights' reciprocals)
        return None


def _bspline_coefficient_regions(lib):
    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    ref = refdata.ref("bspline")
    out = {}
    for method in refdata.BSPLINE_METHODS[2:]:
        m = ref.methods[method]
        k, bc = refdata.bspline_args(method)
        where, con, ul, rel = [], [], [], []
        for cid in S.real_curve_ids():
            c = S.curve(cid)
            pos = np.flatnonzero(ref.inputs["set"][m.index] == cid)
            obj = _bspline_object(lib, method, c) if pos.size else None
            if obj is None:
                continue
            order = np.argsort(ref.inputs["index"][m.index[pos]], kind="stable")
            hi, lo = m.hi[pos[order]], m.lo[pos[order]]
            got = np.asarray(obj.c, dtype=float)
            bound = refdata.collocation_bound(c.x, np.asarray(obj.t), k, bc, got)
            for i, g in enumerate(got):
                err = abs((g - hi[i]) - lo[i])
                where.append(f"{cid}#{i}")
                con.append(0.0 if err == 0 else err / bound[i])
                ul.append(float(ulps(g, hi[i])))
                rel.append(err / abs(hi[i]) if hi[i] else err)
        out[method] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(con), bound=1.0)
    return out


def _bspline_evaluation_regions(lib):
    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    curves = refdata.ref("curves")
    out = {}
    for method in refdata.BSPLINE_METHODS[1:]:
        m = curves.methods[method]
        k, _ = refdata.bspline_args(method)
        where, con, ul, e2e = [], [], [], []
        for cid in S.real_curve_ids():
            c = S.curve(cid)
            obj = _bspline_object(lib, method, c)
            pos = np.flatnonzero(curves.inputs["set"][m.index] == cid)
            if obj is None or not pos.size:
                continue
            order = np.argsort(curves.inputs["index"][m.index[pos]], kind="stable")
            exact = m.hi[pos[order]]
            scale = np.max(np.abs(c.y))
            t, cc = np.asarray(obj.t), np.asarray(obj.c)
            vals = np.asarray(obj(c.xe, extrapolate=True))
            for i, (v, g) in enumerate(zip(c.xe, vals)):
                want = refdata.bspline_exact(t, cc, k, v)
                if k == 1:  # the linear kernel's contract on the interval of v
                    j = refdata.knot_interval(t, 1, v)
                    a, b, x0, x1 = float(cc[j - 1]), float(cc[j]), float(t[j]), float(t[j + 1])
                    tt = float((Fraction(float(v)) - Fraction(x0)) / (Fraction(x1) - Fraction(x0)))
                    bound = refdata.linear_bound(a, b, tt)
                else:
                    bound = refdata.bspline_bound(t, cc, k, v)
                if not np.isfinite(bound):
                    continue
                err = abs(Fraction(float(g)) - want) if np.isfinite(g) else np.inf
                where.append(f"{cid}#{i}")
                con.append(0.0 if err == 0 else float(err) / bound)
                ul.append(float(ulps(g, float(want))))
                e2e.append(abs(g - exact[i]) / max(abs(exact[i]), scale))
        out[method] = region(where, contract=np.array(con), bound=1.0, ulp=np.array(ul), e2e_rel=np.array(e2e))
    return out


def _coefficient_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    from tests.interpolate import refdata

    ref = refdata.ref("slopes")
    out = {}
    for meth in CUBIC:
        m = ref.methods[meth]
        where, con, ul, rel = [], [], [], []
        for c in _cubic_curves():
            obj = _cubic_object(lib, meth, c)
            if obj is None:
                continue
            got = np.asarray(obj.c)[2]  # the slopes at the nodes but the last
            pos = np.flatnonzero(ref.inputs["set"][m.index] == c.id)
            order = np.argsort(ref.inputs["index"][m.index[pos]], kind="stable")
            hi, lo = m.hi[pos[order]], m.lo[pos[order]]
            bound = _slope_bound(meth, c, np.append(got, hi[-1]))
            for i, g in enumerate(got):
                err = abs((g - hi[i]) - lo[i])
                where.append(f"{c.id}#{i}")
                con.append(0.0 if err == 0 else err / bound[i])
                ul.append(float(ulps(g, hi[i])))
                rel.append(err / abs(hi[i]) if hi[i] else err)
        out[meth] = region(where, ulp=np.array(ul), rel=np.array(rel), contract=np.array(con), bound=1.0)
    if _has("BSpline"):
        out.update(_bspline_coefficient_regions(lib))
    return out


def _has(name):
    import stardust as sd

    return name in sd.interpolate.__all__


def _exact_piece(c, x, i, v):
    k = c.shape[0] - 1
    s = Fraction(float(v)) - Fraction(float(x[i]))
    total = Fraction(0)
    for j in range(k + 1):
        total = total * s + Fraction(float(c[j, i]))
    return total, float(s)


def _evaluation_regions(lib):
    from tests.interpolate import refdata

    if lib not in ("sd", "scipy", "R"):
        return None
    curves = refdata.ref("curves")
    out = {}
    for meth in CUBIC:
        if lib == "R" and meth not in ("natural", "periodic"):
            continue
        m = curves.methods[meth]
        where, con, ul, e2e = [], [], [], []
        for c in _cubic_curves():
            if meth == "periodic" and "periodic" not in c.tags:
                continue
            pos = np.flatnonzero(curves.inputs["set"][m.index] == c.id)
            order = np.argsort(curves.inputs["index"][m.index[pos]], kind="stable")
            exact = m.hi[pos[order]]
            scale = np.max(np.abs(c.y))
            if lib == "R":
                from . import rsession

                vals = rsession.evaluate_named(
                    f"splinefun(X, Y, method = '{'natural' if meth == 'natural' else 'periodic'}')(Q)",
                    {"X": c.x, "Y": c.y, "Q": c.xe},
                )
                coeffs = None
            else:
                obj = _cubic_object(lib, meth, c)
                vals = np.asarray(obj(c.xe, extrapolate=True))
                coeffs = np.asarray(obj.c)
            for k, (v, g) in enumerate(zip(c.xe, vals)):
                if coeffs is None and not c.x[0] <= v <= c.x[-1]:
                    continue  # R's natural spline extrapolates linearly: inside only
                where.append(f"{c.id}#{k}")
                e2e.append(abs(g - exact[k]) / max(abs(exact[k]), scale))
                if coeffs is None:
                    continue
                i = int(min(max(np.searchsorted(c.x, v, side="right") - 1, 0), len(c.x) - 2))
                want, s = _exact_piece(coeffs, c.x, i, v)
                err = abs(Fraction(float(g)) - want)
                con.append(0.0 if err == 0 else float(err) / refdata.ppoly_bound(coeffs[:, i], s))
                ul.append(float(ulps(g, float(want))))
        kw = {"contract": np.array(con), "bound": 1.0, "ulp": np.array(ul)} if con else {}
        out[meth] = region(where, e2e_rel=np.array(e2e), **kw)
    if lib != "R" and _has("BSpline"):
        out.update(_bspline_evaluation_regions(lib))
    return out


def _roots_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    ref = refdata.ref("roots")
    m = ref.methods["root"]
    where, dist, missed, extra = [], [], [], []
    for pid in S.PIECES_IDS:
        P = S.pieces(pid)
        if lib == "sd":
            import stardust as sd

            got = np.asarray(sd.interpolate.PPoly(P.c, P.x).roots(extrapolate=False, discontinuity=False))
        else:
            import scipy.interpolate as si

            got = si.PPoly(P.c, P.x).roots(extrapolate=False, discontinuity=False)
        got = got[np.isfinite(got)]
        pos = np.flatnonzero(ref.inputs["set"][m.index] == pid)
        refs = m.hi[pos]
        for k, r in enumerate(refs):
            d = np.min(np.abs(got - r)) if got.size else np.inf
            where.append(f"{pid}#{k}")
            dist.append(d / np.spacing(abs(r)))
            missed.append(1.0 if d > 2.0**-20 * max(1.0, abs(r)) else 0.0)
        for r in got:
            if not refs.size or np.min(np.abs(refs - r)) > 2.0**-20 * max(1.0, abs(r)):
                extra.append(1.0)
    return {"pieces": region(where, ulp=np.array(dist), missed=np.array([sum(missed)]),
                             extra=np.array([float(len(extra))]))}  # fmt: skip


def sweeps():
    import stardust as sd

    out = []
    if hasattr(sd, "interp"):
        out.append(Custom("interpolate", "linear", lambda: _linear_regions("sd"),
                          compare_libs(_linear_regions, ("numpy", "scipy", "R"), seconds=120.0)))  # fmt: skip
    if "CubicSpline" in sd.interpolate.__all__:
        out.append(Custom("interpolate", "coefficients", lambda: _coefficient_regions("sd"),
                          compare_libs(_coefficient_regions, ("scipy",), seconds=120.0)))  # fmt: skip
        out.append(Custom("interpolate", "evaluation", lambda: _evaluation_regions("sd"),
                          compare_libs(_evaluation_regions, ("scipy", "R"), seconds=300.0)))  # fmt: skip
        out.append(Custom("interpolate", "roots", lambda: _roots_regions("sd"),
                          compare_libs(_roots_regions, ("scipy",), seconds=120.0)))  # fmt: skip
    return out
