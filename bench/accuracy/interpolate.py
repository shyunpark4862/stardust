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

IP4 brings (judged through tests/interpolate/refdata.py)

  polynomial   BarycentricInterpolator, KroghInterpolator and FloaterHormannInterpolator (d = 3) on polynomial.npz,
               end to end against the exact interpolant of the float data: regions barycentric, barycentric_d1,
               barycentric_d2 (refdata.barycentric_bound, its scale the Lebesgue function and sum |l_j y_j|; the
               derivatives with the nodal values' bound), krogh, krogh_d1, krogh_d2 (refdata.krogh_bound with the
               table's bound) and floater_hormann (refdata.rational_bound with the weights' gamma_{3d}), with
               `e2e_rel` relative to max(|value|, max|y|). Comparator: scipy (its objects, the same units).
  aaa          AAA on the sample sets of tests/interpolate/samples.py: `contract`, its value at the samples,
               their midpoints and points beyond against the computed rational evaluated exactly (Fractions;
               Arb for complex data) in units of refdata.rational_bound, and `residual`, max |r(z_i) - f_i| over
               the samples in units of rtol max|f| (at most 1 when converged). Comparator: scipy (its own
               rational and evaluation).

IP6 brings (judged through tests/interpolate/refdata.py)

  rbf          RBFInterpolator on rbf.npz, one region per method (the eight kernels, thin_plate_spline_s0.1 with
               smoothing 0.1, thin_plate_spline_k8 with neighbors = 8): `contract`, the value at each point against
               the computed coefficients' function evaluated exactly (Arb) in units of refdata.rbf_eval_bound;
               `solve`, the solve's componentwise backward error in units of refdata.rbf_backward_bound (the computed
               system, its sytrf factors), with `growth` (|| |L||D||L^T| || / ||A||) per set; `residual`, max
               |f(y_i) + smoothing_i a_i - d_i| / max|d| at the data points; `e2e_rel` against the exact interpolant
               of the float data relative to max|ref| (the system's conditioning decides it). The neighbors region
               has e2e_rel alone (each set is the global interpolant of its points to the bit, tests/interpolate/
               test_rbf.py). Comparator: scipy (its coefficients with its own system and scipy.linalg.ldl's factors).

IP5 brings (judged through tests/interpolate/refdata.py)

  grid         RegularGridInterpolator on grid.npz (the exact interpolant of the float data), one region per method:
               linear and slinear in units of the stacked linear contract (refdata.grid_linear_exact), cubic and
               quintic end to end in units of the tensor spline's evaluation contract with the coefficients'
               construction bound carried through (refdata.tensor_bound with refdata.tensor_collocation_bound),
               pchip end to end (no contract: `e2e_rel` and ULP); `e2e_rel` relative to max(|ref|, max|values|).
               nearest picks a node's value (no rounding) and has no region. Comparator: scipy 1.18.1 with its
               defaults (cubic and quintic through its iterative solve), in the same units (sd's bound as the scale).
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


def _poly_object(lib, kind, P):
    if lib == "sd":
        import stardust as sd

        IP = sd.interpolate
        kw = {}
    else:
        import scipy.interpolate as IP

        kw = {"rng": 0} if kind == "barycentric" else {}
    if kind == "barycentric":
        return IP.BarycentricInterpolator(P.xi, P.yi, **kw)
    if kind == "krogh":
        return IP.KroghInterpolator(P.xi, P.yi)
    return IP.FloaterHormannInterpolator(P.xi, P.yi, d=3)


def _poly_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    import math
    import warnings

    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    ref = refdata.ref("polynomial")

    def column(method, pid):
        m = ref.methods[method]
        pos = np.flatnonzero(ref.inputs["set"][m.index] == pid)
        order = np.argsort(ref.inputs["index"][m.index[pos]], kind="stable")
        return m.hi[pos[order]], m.lo[pos[order]]

    acc = {}

    def add(reg, where, err, b, exact, scale):
        r = acc.setdefault(reg, ([], [], []))
        r[0].append(where)
        r[1].append(0.0 if err == 0 else float(err) / b)
        r[2].append(float(err) / max(abs(exact), scale))

    for pid in S.POLY_IDS:
        P = S.poly(pid)
        n = P.xi.size
        X, Y = [Fraction(float(v)) for v in P.xi], [Fraction(float(v)) for v in P.yi]
        scale = float(np.max(np.abs(P.yi)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            k = _poly_object(lib, "krogh", P)
            tb = refdata.krogh_table_bound(X, Y)
            c_hat = np.asarray(k.c)[:-1, 0] if lib == "scipy" else np.asarray(k._c)[:, 0]
            for m, meth in ((0, "value"), (1, "d1"), (2, "d2")):
                got = np.asarray(k(P.x) if m == 0 else k.derivative(P.x, der=m))
                hi, lo = column(meth, pid)
                for j, v in enumerate(P.x):
                    err = abs(Fraction(float(got[j])) - Fraction(float(hi[j])) - Fraction(float(lo[j])))
                    b = refdata.krogh_bound(X, c_hat, Fraction(float(v)), m, tb)
                    add("krogh" + ("" if m == 0 else f"_d{m}"), f"{pid}#{j}", err, b, hi[j], scale)
        if not P.distinct:
            continue
        b_obj = _poly_object(lib, "barycentric", P)
        D = refdata.differentiation_matrix(X)
        v_hat, bound, vf = Y, [0.0] * n, P.yi.copy()
        lam, _ = column("lebesgue", pid)
        for m, meth in ((0, "value"), (1, "d1"), (2, "d2")):
            if m:
                bound = refdata.rational_nodal_bound(D, v_hat, bound)
                if lib == "sd":
                    import stardust as sd

                    nodal = sd._core.interpolate.bary_node_derivative(b_obj.xi, b_obj.wi, sd.asarray(vf.reshape(-1, 1)))
                    vf = np.asarray(nodal)[:, 0]
                else:  # scipy's nodal values: its D (from its weights) applied once more
                    vf = _scipy_dmatrix(b_obj) @ vf
                v_hat = [Fraction(float(t)) for t in vf]
            got = np.asarray(b_obj(P.x) if m == 0 else b_obj.derivative(P.x, der=m))
            hi, lo = column(meth, pid)
            for j, v in enumerate(P.x):
                err = abs(Fraction(float(got[j])) - Fraction(float(hi[j])) - Fraction(float(lo[j])))
                V = Fraction(float(v))
                ell = refdata.lagrange_basis(X, V)
                q = sum((e * t for e, t in zip(ell, v_hat)), Fraction(0))
                at = float(sum((abs(e * t) for e, t in zip(ell, v_hat)), Fraction(0)))
                b = refdata.barycentric_bound(n, at, float(lam[j]), float(q))
                b += sum(abs(float(e)) * t for e, t in zip(ell, bound))
                add("barycentric" + ("" if m == 0 else f"_d{m}"), f"{pid}#{j}", err, 1.01 * b, hi[j], scale)
        f_obj = _poly_object(lib, "fh", P)
        got = np.asarray(f_obj(P.x))
        hi, lo = column("fh3", pid)
        Xs = sorted(X)
        Ys = [Y[X.index(t)] for t in Xs]
        W = _fh_weights(Xs, 3)
        for j, v in enumerate(P.x):
            V = Fraction(float(v))
            err = abs(Fraction(float(got[j])) - Fraction(float(hi[j])) - Fraction(float(lo[j])))
            if V in Xs:
                b = math.inf
            else:
                lam_ = [w / (V - z) for z, w in zip(Xs, W)]
                den = sum(lam_, Fraction(0))
                r = sum((t * f for t, f in zip(lam_, Ys)), Fraction(0)) / den
                sf = sum((abs(t * f) for t, f in zip(lam_, Ys)), Fraction(0))
                ss = sum((abs(t) for t in lam_), Fraction(0))
                b = refdata.rational_bound(n, float(sf), float(ss), float(den), float(r), n + 12, n + 11, n + 10)
            add("floater_hormann", f"{pid}#{j}", err, b, hi[j], scale)
    return {reg: region(w, contract=np.array(c), bound=1.0, e2e_rel=np.array(e)) for reg, (w, c, e) in acc.items()}


def _scipy_dmatrix(b):
    """scipy's differentiation matrix of a BarycentricInterpolator (its cached c_ij)."""
    if b._diff_cij is None:
        b.derivative(b.xi[:1])
    return b._diff_cij


def _fh_weights(X, d):
    n = len(X)
    w = []
    for k in range(n):
        acc = Fraction(0)
        for i in range(max(k - d, 0), min(k + 1, n - d)):
            prod = Fraction(1)
            for j in range(i, i + d + 1):
                if j != k:
                    prod *= abs(X[k] - X[j])
            acc += 1 / prod
        w.append(acc if (k - d) % 2 == 0 else -acc)
    return w


def _aaa_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    from tests.interpolate import refdata
    from tests.interpolate import samples as S
    from tests.interpolate.test_aaa import _arb_parts, _exact_parts, _points

    out = {}
    for aid in S.AAA_IDS:
        s = S.aaa(aid)
        if lib == "sd":
            import stardust as sd

            a = sd.interpolate.AAA(s.z, s.f)
        else:
            import scipy.interpolate as si

            a = si.AAA(s.z, s.f)
        zj, fj, wj = (np.asarray(v) for v in (a.support_points, a.support_values, a.weights))
        m = zj.size
        keep = np.isfinite(s.f)
        q = _points(s.z[keep])
        got = np.asarray(a(q))
        real = not np.iscomplexobj(zj)
        if real:
            Z, W, F = ([Fraction(float(v)) for v in arr] for arr in (zj, wj, fj))
        where, con = [], []
        for k, v in enumerate(q):
            if v in zj:
                continue
            if real:
                r, sf, ss, den = _exact_parts(Z, W, F, Fraction(float(v)))
                err = float(abs(Fraction(float(got[k])) - r))
                b = refdata.rational_bound(m, float(sf), float(ss), float(den), float(r), m + 3, m + 2, m + 1)
            else:
                from flint import acb

                r, sf, ss, den = _arb_parts(zj, wj, fj, v)
                err = float(abs(acb(complex(got[k])) - r).upper())
                b = refdata.rational_bound(m, float(sf.upper()), float(ss.upper()), float(abs(den).lower()),
                                           float(abs(r).upper()), 3 * m + 16, 2 * m + 16, 2 * m + 8)  # fmt: skip
            where.append(f"{aid}#{k}")
            con.append(err / b)
        res = np.max(np.abs(np.asarray(a(s.z[keep])) - s.f[keep])) / ((2.0**-52) ** 0.75 * np.max(np.abs(s.f[keep])))
        out[aid.split("/")[1]] = region(where, contract=np.array(con), bound=1.0, residual=np.array([res]))
    return out


def _rbf_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    ref = refdata.ref("rbf")
    sets = np.asarray(ref.inputs["set"])
    out = {}
    for method, M in ref.methods.items():
        kernel, eps, sm, nb = refdata.rbf_options(method)
        where, con, solve, growth, resid, e2e = [], [], [], [], [], []
        for sid in sorted(set(sets[M.index])):
            sc = S.scatter(sid)
            pos = np.flatnonzero(sets[M.index] == sid)
            pos = pos[np.argsort(np.asarray(ref.inputs["index"])[M.index[pos]], kind="stable")]
            if lib == "sd":
                import stardust as sd

                r = sd.interpolate.RBFInterpolator(sc.y, sc.d, kernel=kernel, epsilon=eps, smoothing=sm, neighbors=nb)
            else:
                import scipy.interpolate as si

                r = si.RBFInterpolator(sc.y, sc.d, kernel=kernel, epsilon=eps, smoothing=sm, neighbors=nb)
            got = np.asarray(r(sc.xi))
            scale = float(np.max(np.abs(M.hi[pos])))
            e2e += list(np.abs((got - M.hi[pos]) - M.lo[pos]) / scale)
            where += [f"{sid}#{k}" for k in range(len(pos))]
            if nb is not None:
                continue
            if lib == "sd":
                import stardust as sd

                A = np.asarray(sd._core.interpolate.rbf_matrix(sd.asarray(sc.y), r._smoothing, r._kid, eps, r._powers))
                L, D, _ = (np.asarray(v) for v in sd.linalg.ldl(sd.asarray(A)))
                c = np.asarray(r._coeffs)
                pw, sh, scl = np.asarray(r._powers), np.asarray(r._shift), np.asarray(r._scale)
            else:
                import scipy.linalg as sl
                from scipy.interpolate import _rbfinterp_np as rnp

                A, _, sh, scl = rnp._build_system(r.y, r.d, r.smoothing, kernel, eps, r.powers, np)
                L, D, _ = sl.ldl(A)
                c, pw = np.asarray(r._coeffs), np.asarray(r.powers)
            units = float(
                np.max(refdata.rbf_residual(A, c, sc.d.reshape(-1, 1)) / refdata.rbf_backward_bound(A, c, L, D))
            )
            g = refdata.rbf_growth(A, L, D)
            fit = np.asarray(r(sc.y))
            res = float(np.max(np.abs(fit + sm * c[: len(sc.y), 0] - sc.d)) / np.max(np.abs(sc.d)))
            for k, x in enumerate(sc.xi):
                f, phis, monos, rs = refdata.rbf_exact(kernel, eps, sc.y, x, c[:, 0], pw, sh, scl)
                b = refdata.rbf_eval_bound(kernel, sc.y.shape[1], c[:, 0], phis, monos, rs, pw)
                con.append(abs(float(got[k]) - float(f.mid())) / b)
                solve.append(units)
                growth.append(g)
                resid.append(res)
        if nb is not None:
            out[method] = region(where, e2e_rel=np.array(e2e))
        else:
            out[method] = region(where, contract=np.array(con), bound=1.0, solve=np.array(solve),
                                 growth=np.array(growth), residual=np.array(resid), e2e_rel=np.array(e2e))  # fmt: skip
    return out


def _grid_regions(lib):
    if lib not in ("sd", "scipy"):
        return None
    from tests.interpolate import refdata
    from tests.interpolate import samples as S

    if lib == "sd":
        import stardust as sd

        RGI = sd.interpolate.RegularGridInterpolator
    else:
        import scipy.interpolate as si

        RGI = si.RegularGridInterpolator
    import stardust as sd

    ref = refdata.ref("grid")
    acc = {}
    for meth, refm in (("linear", "linear"), ("slinear", "linear"), ("cubic", "cubic"), ("quintic", "quintic"),
                       ("pchip", "pchip")):  # fmt: skip
        m = ref.methods[refm]
        sets = ref.inputs["set"][m.index]
        for gid in dict.fromkeys(sets.tolist()):
            G = S.grid(gid)
            if meth == "slinear" and any(p.size < 2 for p in G.points):
                continue
            got = np.asarray(RGI(G.points, G.values, method=meth)(G.xi))
            mine = sd.interpolate.RegularGridInterpolator(G.points, G.values, method=meth)
            pts = [np.asarray(p) for p in mine.grid]
            vals = np.asarray(mine.values)
            scale = float(np.max(np.abs(G.values)))
            if meth in ("cubic", "quintic"):
                k = 3 if meth == "cubic" else 5
                knots, c2 = mine._spline(k, "accuracy")
                knots = [np.asarray(t) for t in knots]
                c = np.asarray(c2).reshape(vals.shape)
                cb = refdata.tensor_collocation_bound(pts, knots, k, vals)
            e = acc.setdefault(meth, {"where": [], "contract": [], "ulp": [], "e2e": []})
            for p in np.flatnonzero(sets == gid):
                i = int(ref.inputs["index"][m.index[p]])
                hi, lo = float(m.hi[p]), float(m.lo[p])
                err = abs(Fraction(float(got[i])) - Fraction(hi) - Fraction(lo))
                if meth in ("linear", "slinear"):
                    b = refdata.grid_linear_exact(pts, vals, G.xi[i])[1]
                elif meth == "pchip":
                    b = None
                else:
                    b = refdata.tensor_bound(knots, c, k, G.xi[i], None, c_bound=cb)
                e["where"].append(f"{gid}#{i}")
                if b is not None:
                    e["contract"].append(0.0 if err == 0 else float(err) / b)
                e["ulp"].append(float(ulps(got[i], hi)))
                e["e2e"].append(float(err) / max(abs(hi), scale))
    out = {}
    for meth, e in acc.items():
        kw = {"contract": np.array(e["contract"]), "bound": 1.0} if e["contract"] else {}
        out[meth] = region(e["where"], ulp=np.array(e["ulp"]), e2e_rel=np.array(e["e2e"]), **kw)
    return out


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
    if "AAA" in sd.interpolate.__all__:
        out.append(Custom("interpolate", "polynomial", lambda: _poly_regions("sd"),
                          compare_libs(_poly_regions, ("scipy",), seconds=300.0)))  # fmt: skip
        out.append(Custom("interpolate", "aaa", lambda: _aaa_regions("sd"),
                          compare_libs(_aaa_regions, ("scipy",), seconds=120.0)))  # fmt: skip
    if "RegularGridInterpolator" in sd.interpolate.__all__:
        out.append(Custom("interpolate", "grid", lambda: _grid_regions("sd"),
                          compare_libs(_grid_regions, ("scipy",), seconds=120.0)))  # fmt: skip
    if "RBFInterpolator" in sd.interpolate.__all__:
        out.append(Custom("interpolate", "rbf", lambda: _rbf_regions("sd"),
                          compare_libs(_rbf_regions, ("scipy",), seconds=120.0)))  # fmt: skip
    return out
