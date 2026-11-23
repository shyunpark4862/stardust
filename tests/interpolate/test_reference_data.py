"""The reference and comparison data of sd.interpolate (docs/design/interpolate.md §7): every file belongs to
the test data as they are (digests), the references pass checks of their own (exact rationals, two
independent constructions of the same spline, the properties of PCHIP, Akima and the polynomial
interpolant), and the recorded scipy 1.18.1, numpy 2.5.4 and R 4.6.2 results agree with the references and
show the behaviours the design departs from. The recorded files are compared, not the installed
libraries, so the tests do not change with them."""

import math
from fractions import Fraction

import numpy as np
import pytest

from . import refdata
from . import samples as S

REFS = {f: refdata.ref(f) for f in refdata.FAMILIES}
SCIPY = refdata.data("scipy")
NUMPY = refdata.data("numpy")
R = refdata.data("r")


def fr(v) -> Fraction:
    return Fraction(float(v))


def at(family, method, sid):
    """(positions in the method's arrays, point numbers) of the set's points with that method."""
    r = REFS[family]
    m = r.methods[method]
    sets = r.inputs["set"][m.index]
    pos = np.flatnonzero(sets == sid)
    return pos, m.index[pos]


def value(rec):
    v = rec["value"]
    if isinstance(v, dict) and "array" in v:
        return np.array(v["array"], dtype=float)
    return v


# ---- the files belong to the data -----------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == sorted(refdata.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/interpolate_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f
        for m in r.methods.values():
            assert m.dropped.size == 0


def _object(sid):
    kind = sid.split("/", 1)[0]
    if kind == "lerp":
        return next(L for L in S.lerps() if L.id == sid[5:])
    if kind == "curve":
        return S.curve(sid[6:])
    getter = {"grid": S.grid, "scatter": S.scatter, "tri": S.tri_set, "roots": S.pieces, "poly": S.poly}.get(kind)
    return getter(sid) if getter else S.curve(sid)


def test_points_are_the_generators_points():
    for f, r in REFS.items():
        for sid, dig in set(zip(r.inputs["set"].tolist(), r.inputs["digest"].tolist())):
            assert _object(sid).digest == dig, (f, sid)
    lin = REFS["linear"]
    for k in range(lin.points):
        sid, i = str(lin.inputs["set"][k]), int(lin.inputs["index"][k])
        o = _object(sid)
        assert lin.inputs["x"][k] == (o.x[i] if sid.startswith("lerp/") else o.xe[i]), sid
    for f in ("curves", "polynomial"):
        r = REFS[f]
        for k in range(0, r.points, 7):
            o = _object(str(r.inputs["set"][k]))
            want = o.xe if f == "curves" else o.x
            assert r.inputs["x"][k] == want[int(r.inputs["index"][k])]


def test_every_data_set_is_covered():
    covered = {f: set(REFS[f].inputs["set"].tolist()) for f in REFS}
    assert covered["slopes"] == covered["curves"] == set(S.real_curve_ids())
    assert {f"curve/{c}" for c in S.real_curve_ids()} <= covered["linear"]
    assert {f"lerp/{L.id}" for L in S.lerps() if S.lerp_finite(L)} <= covered["linear"]
    assert covered["polynomial"] == set(S.POLY_IDS)
    assert covered["roots"] == set(S.PIECES_IDS) - {"roots/breakpoint"}  # constant pieces have no root
    assert covered["grid"] == {g for g in S.GRID_IDS if not ({"nan", "batch"} & S.grid(g).tags)}
    assert covered["rbf"] == set(S.SCATTER_IDS) - {"scatter/duplicate2d"}
    assert covered["scattered"] == set(S.tri_ids())


# ---- the references check themselves --------------------------------------------------------


def test_linear_references_are_the_exact_values_rounded_once():
    r = REFS["linear"]
    m = r.methods["lerp"]
    for j, k in enumerate(m.index):
        x0, x1, a, b, x = (fr(r.inputs[n][k]) for n in ("x0", "x1", "a", "b", "x"))
        exact = a + (x - x0) / (x1 - x0) * (b - a)
        assert m.hi[j] == float(exact), k
        assert abs(fr(m.hi[j]) + fr(m.lo[j]) - exact) <= abs(exact) * Fraction(1, 2**104) + Fraction(1, 2**1074)


@pytest.mark.parametrize(("spline", "bspline"), [("natural", "k3_natural"), ("clamped", "k3_clamped"),
                                                 ("not_a_knot", "k3"), ("periodic", "k3_periodic")])  # fmt: skip
def test_two_constructions_of_one_spline_agree(spline, bspline):
    """The slope systems of CubicSpline and the B-spline collocation give the same function (the
    interpolating cubic spline with those end conditions is unique): equal references at every point."""
    r = REFS["curves"]
    a, b = r.methods[spline], r.methods[bspline]
    common = sorted(set(a.index) & set(b.index))
    assert len(common) > 100
    ia = {k: j for j, k in enumerate(a.index)}
    ib = {k: j for j, k in enumerate(b.index)}
    for k in common:
        ha, la, hb, lb = a.hi[ia[k]], a.lo[ia[k]], b.hi[ib[k]], b.lo[ib[k]]
        if spline == "periodic":
            # scipy's periodic knots outside [x[0], x[-1]] are x[0] - dx and x[-1] + dx in doubles, not exact
            # shifts by the period, so the B-spline is periodic only to rounding
            assert abs(ha - hb) <= 1e-13 * np.max(np.abs(S.curve(str(r.inputs["set"][k])).y)), k
            continue
        assert ha == hb and abs(la - lb) <= 2.0**-104 * abs(ha) + 2.0**-1074, (spline, r.inputs["set"][k])


def test_degree_0_and_1_coefficients_are_the_data():
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        for meth in ("k0", "k1"):
            pos, _ = at("bspline", meth, cid)
            assert np.array_equal(REFS["bspline"].methods[meth].hi[pos], c.y), (cid, meth)


def test_knots_are_scipys():
    r = REFS["bspline"]
    seen = 0
    for g, cid in enumerate(r.group_inputs["set"].tolist()):
        for meth in ("k2", "k3", "k4", "k5", "k3_natural", "k3_clamped", "k3_periodic", "k5_periodic"):
            mine = r.group_inputs[f"t_{meth}"][g]
            rec = SCIPY[f"knots/{cid}/{meth}"]
            # no reference (too few nodes, 'periodic' on a curve not made for it) or no scipy result (its
            # solve fails on subnormal/x)
            if mine.size == 0 or "error" in rec:
                continue
            assert np.array_equal(mine, value(rec)), (cid, meth)
            seen += 1
    assert seen > 100


def test_pchip_slopes_keep_monotone_data_monotone():
    """Fritsch-Carlson: on monotone data every slope is 0 or has the data's sign and is at most 3 times
    the smaller neighbouring secant (exact references)."""
    for cid in S.curve_ids("monotone"):
        c = S.curve(cid)
        pos, _ = at("slopes", "pchip", cid)
        s = REFS["slopes"].methods["pchip"].hi[pos]
        d = np.diff(c.y) / np.diff(c.x)
        sign = np.sign(d[d != 0][0])
        assert np.all((s == 0) | (np.sign(s) == sign)), cid
        for i in range(1, c.n - 1):
            assert abs(s[i]) <= 3 * min(abs(d[i - 1]), abs(d[i])) * (1 + 1e-15), (cid, i)


def test_akima_flat_runs_are_flat():
    c = S.curve("flat/akima")
    for meth in ("akima", "makima"):
        pos, _ = at("slopes", meth, c.id)
        s = REFS["slopes"].methods[meth].hi[pos]
        flat = [i for i in range(1, c.n - 1) if c.y[i - 1] == c.y[i] == c.y[i + 1]]
        assert flat and all(s[i] == 0 for i in flat), meth


def test_polynomial_references_interpolate():
    r = REFS["polynomial"]
    for pid in S.POLY_IDS:
        P = S.poly(pid)
        pos, idx = at("polynomial", "value", pid)
        xs = r.inputs["x"][idx]
        vals = r.methods["value"].hi[pos]
        if P.distinct:
            for xi, yi in zip(P.xi, P.yi):
                hit = np.flatnonzero(xs == xi)
                assert hit.size == 1 and vals[hit[0]] == yi, pid
        if P.distinct:
            lpos, _ = at("polynomial", "lebesgue", pid)
            leb = r.methods["lebesgue"].hi[lpos]
            assert np.all(leb >= 1.0) and np.all(leb[np.isin(xs, P.xi)] == 1.0), pid
    pos, _ = at("polynomial", "lebesgue", "poly/uniform30")
    assert REFS["polynomial"].methods["lebesgue"].hi[pos].max() > 1e5  # Runge's nodes: the bound grows


def test_hermite_reference_meets_its_derivative_data():
    r = REFS["polynomial"]
    P = S.poly("poly/hermite")
    for meth, at_x, want in (("value", 0.0, 1.0), ("d1", 0.0, 2.0), ("value", 1.0, 0.0), ("d1", 1.0, -1.0),
                             ("d2", 1.0, 3.0), ("value", 2.0, 1.0)):  # fmt: skip
        pos, idx = at("polynomial", meth, P.id)
        k = np.flatnonzero(r.inputs["x"][idx] == at_x)
        assert k.size == 1 and r.methods[meth].hi[pos[k[0]]] == want, (meth, at_x)


def test_grid_references_meet_the_values_at_the_nodes():
    r = REFS["grid"]
    for gid in set(r.inputs["set"].tolist()):
        G = S.grid(gid)
        for meth in r.methods:
            pos, idx = at("grid", meth, gid)
            for p, k in zip(pos, idx):
                q = r.inputs["x"][k]
                ii = [np.flatnonzero(ax == v) for ax, v in zip(G.points, q)]
                if all(i.size for i in ii):
                    assert r.methods[meth].hi[p] == G.values[tuple(int(i[0]) for i in ii)], (gid, meth)


def _poly_at(coeffs, s):
    v = Fraction(0)
    for c in coeffs:
        v = v * s + c
    return v


def test_scattered_references_meet_the_data():
    """At a data point the linear and the Clough-Tocher references are the value itself; the linear scale bounds the
    value; every set of 2-D points whose gradients are finite has its Clough-Tocher points, with its gradients."""
    r = REFS["scattered"]
    lin, ct = r.methods["linear"], r.methods["clough_tocher"]
    for k in range(r.points):
        sc = S.tri_set(str(r.inputs["set"][k]))
        i = int(r.inputs["index"][k])
        assert np.array_equal(r.inputs["x"][k], sc.xi[i])
        if i >= 64:  # the data point i - 64
            want = float(sc.d[i - 64])
            assert lin.hi[k] == want and lin.lo[k] == 0.0
            j = np.flatnonzero(ct.index == k)
            assert j.size == 0 or (ct.hi[j[0]] == want and ct.lo[j[0]] == 0.0)
    scale = r.methods["linear_scale"]
    assert np.all(np.abs(lin.hi) <= scale.hi * (1 + 2**-50))
    with_ct = set(r.inputs["set"][ct.index].tolist())
    for g, sid in enumerate(r.group_inputs["set"].tolist()):
        grad = r.group_inputs["grad"][g]
        assert (grad.shape[0] > 0) == (sid in with_ct), sid
        if grad.shape[0]:
            assert grad.shape == (S.tri_set(sid).y.shape[0], 2) and np.all(np.isfinite(grad))


def test_roots_are_roots():
    r = REFS["roots"]
    m = r.methods["root"]
    for j in range(r.points):
        P = S.pieces(str(r.inputs["set"][j]))
        i = int(r.inputs["piece"][j])
        coeffs = [fr(v) for v in P.c[:, i]]
        x0 = fr(P.x[i])
        root = fr(m.hi[j]) + fr(m.lo[j])
        lo, hi = root - abs(root) * Fraction(1, 2**100) - Fraction(1, 2**200), root + abs(root) * Fraction(1, 2**100)
        vlo, vhi = _poly_at(coeffs, lo - x0), _poly_at(coeffs, hi - x0)
        if int(r.inputs["multiplicity"][j]) % 2:
            assert vlo == 0 or vhi == 0 or (vlo > 0) != (vhi > 0), j
        else:
            assert abs(_poly_at(coeffs, root - x0)) < Fraction(1, 2**150), j


def test_rbf_references_agree_with_scipy_where_well_conditioned():
    r = REFS["rbf"]
    for sid in S.SCATTER_IDS:
        rec = SCIPY[f"{sid}/thin_plate_spline"]
        pos, _ = at("rbf", "thin_plate_spline", sid)
        if not pos.size:
            continue
        got = value(rec)
        assert np.max(np.abs(got - r.methods["thin_plate_spline"].hi[pos])) < 1e-12, sid


# ---- the records agree with the references --------------------------------------------------


def _curve_ref(method, cid):
    pos, idx = at("curves", method, cid)
    return REFS["curves"].methods[method].hi[pos], REFS["curves"].inputs["x"][idx]


# spacing/1e10 is ill-conditioned; scipy overflows on large/1e300 (its cubic coefficients pass 1e308 where
# the spacing is small) and on subnormal/x (the slopes overflow): test_scipy_and_numpy_overflow
WELL_CONDITIONED = [c for c in S.real_curve_ids() if c not in ("spacing/1e10", "subnormal/x", "large/1e300")]


SCIPY_METHODS = ("natural", "clamped", "not_a_knot", "periodic", "pchip", "akima", "makima", "k0", "k1", "k2", "k3",
                 "k4", "k5", "k3_natural", "k3_clamped", "k3_periodic")  # fmt: skip


@pytest.mark.parametrize("method", SCIPY_METHODS)
def test_scipy_values_are_near_the_references(method):
    seen = 0
    for cid in WELL_CONDITIONED:
        rec = SCIPY.get(f"curve/{cid}/{method}")
        if rec is None or "error" in rec:
            continue
        ref, x = _curve_ref(method, cid)
        c = S.curve(cid)
        inside = (x >= c.x[0]) & (x <= c.x[-1])
        got = value(rec)
        scale = np.maximum(np.abs(ref), np.max(np.abs(c.y)))
        err = np.abs(got - ref) / scale
        assert np.max(err[inside]) < 1e-9, (cid, method, float(np.max(err[inside])))
        seen += 1
    assert seen


def test_numpy_and_r_linear_values_are_within_the_linear_contract():
    r = REFS["linear"]
    m = r.methods["lerp"]
    worst = {"numpy": 0.0, "R": 0.0}
    for cid in S.real_curve_ids():
        if cid == "subnormal/x":  # numpy's slope overflows there: test_scipy_and_numpy_overflow
            continue
        c = S.curve(cid)
        pos, idx = at("linear", "lerp", f"curve/{cid}")
        x0, x1, a, b, x = (r.inputs[n][idx] for n in ("x0", "x1", "a", "b", "x"))
        inside = (x >= c.x[0]) & (x <= c.x[-1])
        for lib, got in (("numpy", NUMPY[f"interp/{cid}"]["values"]), ("R", R[f"approx2/{cid}"]["values"])):
            got = np.array(got)
            for p, k in zip(np.flatnonzero(inside), pos[inside]):
                t = float((fr(x[p]) - fr(x0[p])) / (fr(x1[p]) - fr(x0[p])))
                worst[lib] = max(worst[lib], refdata.linear_units(got[p], m.hi[k], m.lo[k], a[p], b[p], t))
    assert worst["numpy"] <= 1 and worst["R"] <= 1, worst


def test_r_splines_are_near_the_references():
    for meth, mine in (("natural", "natural"), ("periodic", "periodic")):
        for cid in WELL_CONDITIONED:
            rec = R.get(f"{meth}/{cid}")
            if rec is None:
                continue
            ref, x = _curve_ref(mine, cid)
            c = S.curve(cid)
            inside = (x >= c.x[0]) & (x <= c.x[-1])
            err = np.abs(np.array(rec["values"]) - ref) / np.max(np.abs(c.y))
            assert np.max(err[inside]) < 1e-9, (meth, cid)
    # R's default end condition (fmm) is not not-a-knot
    ref, _ = _curve_ref("not_a_knot", "uniform/6")
    assert np.max(np.abs(np.array(R["fmm/uniform/6"]["values"]) - ref)) > 1e-3


def test_r_approx_rule_1_is_nan_outside():
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        v = np.array(R[f"approx1/{cid}"]["values"])
        outside = (c.xe < c.x[0]) | (c.xe > c.x[-1])
        assert np.all(np.isnan(v[outside])) and not np.any(np.isnan(v[~outside])), cid


# ---- the records show the behaviours the design departs from ----------------------------------


def test_numpy_interp_behaviours():
    assert value(NUMPY["unsorted"]).tolist() == [2.5, 7.5, 27.5]  # silently wrong (§1.3)
    assert value(NUMPY["repeated"]).tolist() == [0.5, 5.0, 5.5, 6.0]  # a step: the right value at the node
    assert value(NUMPY["nonfinite/inf_1"]).tolist() == [math.inf, math.inf, math.inf, 1.0]
    assert np.isnan(value(NUMPY["nonfinite/-inf_inf"])[1:3]).all()
    assert NUMPY["period0"]["error"] == "ValueError"
    assert np.isnan(value(NUMPY["period_nan"])).all()
    assert "warnings" in NUMPY["period_inf"]
    assert value(NUMPY["positional_left_right"]).tolist() == [-7.0, 9.0]
    assert NUMPY["scalar_type"]["type"] == "float64"
    assert NUMPY["signed_zeros"]["signs"][0][1] == 1.0  # -0 ends give +0 inside (§3.2)
    f = NUMPY["formula"]
    assert f["equal_fma"] == f["points"] > f["equal_unfused"]  # the wheel computes fma(s, x - x_j, f_j)
    rn = NUMPY["right_node"]
    assert rn["out_of_range_left_of_x1"] > 0 and rn["x1_exact"] == rn["intervals"]


def test_scipy_behaviours():
    assert value(SCIPY["extrapolate/Akima"]) is False
    assert value(SCIPY["extrapolate/CubicSpline_periodic"]) == "periodic"
    assert value(SCIPY["extrapolate/make_interp_spline_periodic"]) == "periodic"
    for name in ("CubicSpline", "Pchip", "Akima", "make_interp_spline", "CubicHermiteSpline"):
        assert SCIPY[f"nan_y/{name}"]["error"] == "ValueError", name
    assert np.isnan(value(SCIPY["nan_y/RBF"])).all()  # sd: ValueError (IP19)
    assert SCIPY["complex/Pchip"]["error"] == SCIPY["complex/Akima"]["error"] == "ValueError"
    assert "ComplexWarning" in SCIPY["complex_x/CubicSpline"]["warnings"][0]  # sd: TypeError (§2.1)
    assert SCIPY["periodic/tolerance_1e-14"]["error"] == "ValueError"
    assert "value" in SCIPY["periodic/tolerance_1e-16"]
    assert "value" in SCIPY["periodic/make_interp_spline_1e-6"]  # rtol 1e-5 there (decision IP28)
    assert SCIPY["nu_negative/BSpline"]["error"] == "NotImplementedError"
    assert SCIPY["k0/values"]["value"]["array"][4] == -1.0 + 1.0  # at x[-1]: y[-1] (= 0.0 here)
    assert value(SCIPY["design_matrix"])[:2] == [12, 8]  # stored zeros
    assert SCIPY["rgi/nan_xi_bounds"]["error"] == "ValueError"  # sd: NaN (IP24)
    assert np.isnan(value(SCIPY["rgi/nan_xi_fill"])[0])
    assert value(SCIPY["rgi/cubic_quadratic"])[0] != 25.0  # the iterative solver's error (IP23)
    assert value(SCIPY["rgi/nearest_tie"]).tolist() == [20.0, 30.0]
    bits = SCIPY["bits/linear_paths"]
    assert bits["rgi_linear_eq_np_interp"] < bits["points"] and bits["rgi_slinear_eq_linear"] < bits["points"]
    assert SCIPY["bits/k1_nodes"]["not_exact"] > 0
    assert "value" in SCIPY["rbf/duplicate"] and "value" in SCIPY["rbf/negative_smoothing"]  # sd: errors (IP25)
    assert "warnings" in SCIPY["rbf/low_degree"] and "warnings" not in SCIPY["rbf/degree_minus1"]
    assert np.max(np.abs(value(SCIPY["rbf/collinear"]))) > 1e9  # a singular system solved without a word
    assert value(SCIPY["barycentric/rng_values_equal"]) is False  # IP21
    assert "warnings" in SCIPY["krogh/high_degree40"]  # IP22


def test_scipy_misses_the_double_roots():
    assert value(SCIPY["roots_inside/roots/tangent"]).size == 0
    pos, _ = at("roots", "root", "roots/tangent")
    assert REFS["roots"].inputs["multiplicity"][REFS["roots"].methods["root"].index[pos]].tolist() == [2, 2]
    close = value(SCIPY["roots_inside/roots/close"])
    pos, _ = at("roots", "root", "roots/close")
    ref = REFS["roots"].methods["root"].hi[pos]
    assert np.max(np.abs(close - ref)) > 1e-9  # the closed-form cubic loses half the digits


def test_scipy_grid_splines_carry_the_solver_error():
    r = REFS["grid"]
    for gid in ("grid/2d", "grid/3d"):
        pos, _ = at("grid", "cubic", gid)
        ref = r.methods["cubic"].hi[pos]
        assert np.max(np.abs(value(SCIPY[f"{gid}/cubic"]) - ref)) > 1e-8
        assert np.max(np.abs(value(SCIPY[f"{gid}/cubic_legacy"]) - ref)) < 1e-13


def test_scipy_and_numpy_overflow():
    """Where sd's design keeps finite values (§3.2: no slope; §4) the comparators overflow: numpy's slope
    (b - a) / h on subnormal spacing, scipy's spline coefficients on large/1e300 and its knots on
    subnormal/x."""
    c = S.curve("subnormal/x")
    got = np.array(NUMPY["interp/subnormal/x"]["values"])
    inside = (c.xe > c.x[0]) & (c.xe < c.x[-1]) & ~np.isin(c.xe, c.x)
    assert not np.all(np.isfinite(got[inside]))
    assert SCIPY["curve/subnormal/x/natural"]["error"] == "ValueError"
    c = S.curve("large/1e300")
    ref, x = _curve_ref("clamped", c.id)
    got = value(SCIPY["curve/large/1e300/clamped"])
    inside = (x >= c.x[0]) & (x <= c.x[-1])
    assert np.all(np.isfinite(ref[inside])) and not np.all(np.isfinite(got[inside]))
