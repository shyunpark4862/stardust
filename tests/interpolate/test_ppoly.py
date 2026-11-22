"""PPoly (docs/design/interpolate.md §2.4, §3.5, §4; milestone IP2): the evaluation contract against the exact value
of the computed coefficients (exact rationals), the derivative, antiderivative and integral against exact
arithmetic on the same coefficients, extend, decreasing breakpoints, the outside rules, the end-to-end values
against the exact interpolants (tests/interpolate/ref/curves.npz) and the records of scipy 1.18.1 and R 4.6.2."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
CURVES = refdata.ref("curves")
SCIPY = refdata.data("scipy")
R = refdata.data("r")


def fr(v):
    return Fraction(float(v))


def _interval(x, v):
    """scipy's piece of v (right-open, the last closed, clamped to the end pieces) for increasing x."""
    return int(min(max(np.searchsorted(x, v, side="right") - 1, 0), len(x) - 2))


def _exact_value(c, x, i, v, nu=0):
    """The exact value of piece i's derivative of order nu at v, from the computed coefficients c (k + 1, m)."""
    k = c.shape[0] - 1
    s = fr(v) - fr(x[i])
    total = Fraction(0)
    for j in range(k + 1):
        d = k - j
        if d < nu:
            continue
        total += fr(c[j, i]) * math.perm(d, nu) * s ** (d - nu)
    return total, float(s)


def _objects(c):
    out = {}
    if "periodic" in c.tags:
        out["periodic"] = IP.CubicSpline(c.x, c.y, bc_type="periodic")
    try:
        out["not_a_knot"] = IP.CubicSpline(c.x, c.y)
        out["natural"] = IP.CubicSpline(c.x, c.y, bc_type="natural")
        out["clamped"] = IP.CubicSpline(c.x, c.y, bc_type="clamped")
        out["pchip"] = IP.PchipInterpolator(c.x, c.y)
        out["akima"] = IP.Akima1DInterpolator(c.x, c.y, extrapolate=True)
        out["makima"] = IP.Akima1DInterpolator(c.x, c.y, method="makima", extrapolate=True)
    except ValueError:  # IP36 (large/1e300, subnormal/x): test_cubic
        pass
    return out


def test_evaluation_contract():
    worst, seen = 0.0, 0
    for cid in S.real_curve_ids():
        c = S.curve(cid)
        for meth, obj in _objects(c).items():
            cc = np.asarray(obj.c)
            for nu in (0, 1, 2, 3):
                got = np.asarray(obj(c.xe, nu=nu, extrapolate=True))
                for v, g in zip(c.xe, got):
                    i = _interval(c.x, v)
                    want, s = _exact_value(cc, c.x, i, v, nu)
                    err = abs(fr(g) - want)
                    if err == 0:
                        continue
                    b = refdata.ppoly_bound(cc[:, i], s, nu)
                    u = float(err) / b
                    assert u <= 1, (cid, meth, nu, v, u)
                    worst, seen = max(worst, u), seen + 1
    assert seen and worst > 0


def test_end_to_end_values_are_near_the_exact_interpolants():
    """The accuracy table's end-to-end column (the exact data's exact interpolant): within 1e-12 of the data's scale
    on the well-conditioned curves."""
    for cid in S.real_curve_ids():
        if cid in ("spacing/1e10", "large/1e300", "subnormal/x"):
            continue
        c = S.curve(cid)
        for meth, obj in _objects(c).items():
            m = CURVES.methods[meth]
            pos = np.flatnonzero(CURVES.inputs["set"][m.index] == cid)
            idx = m.index[pos]
            order = np.argsort(CURVES.inputs["index"][idx], kind="stable")
            ref = m.hi[pos[order]]
            got = np.asarray(obj(c.xe, extrapolate=True))
            scale = np.maximum(np.max(np.abs(c.y)), np.abs(ref))
            assert np.all(np.abs(got - ref) <= 1e-12 * scale), (cid, meth)


def test_scipy_records_and_r_are_near_ours():
    """scipy's and R's recorded values on the curves agree with sd's as both agree with the exact values."""
    for cid in S.real_curve_ids():
        if cid in ("spacing/1e10", "large/1e300", "subnormal/x"):
            continue
        c = S.curve(cid)
        objs = _objects(c)
        scale = np.max(np.abs(c.y))
        inside = (c.xe >= c.x[0]) & (c.xe <= c.x[-1])
        for meth, obj in objs.items():
            rec = SCIPY.get(f"curve/{cid}/{meth}")
            if rec is None or "error" in rec:
                continue
            got = np.asarray(obj(c.xe))
            theirs = np.array(rec["value"]["array"], dtype=float)
            ok = inside | ~np.isnan(theirs)
            tol = 1e-10 * np.maximum(scale, np.abs(theirs[ok]))
            assert np.all(np.abs(got[ok] - theirs[ok]) <= tol), (cid, meth)
        for meth in ("natural", "periodic"):
            rec = R.get(f"{meth}/{cid}")
            if rec is None or meth not in objs:
                continue
            got = np.asarray(objs[meth](c.xe))
            assert np.max(np.abs(got[inside] - np.array(rec["values"])[inside])) <= 1e-10 * scale, (cid, meth)


def _exact_antiderivative(c, x, nu=1):
    """The exact antiderivative's pieces (lists of Fractions, highest degree first) of the computed c."""
    k1, m = c.shape
    pieces = []
    prev = None
    for i in range(m):
        coeffs = [fr(c[j, i]) / math.prod(range(k1 - j, k1 - j + nu)) for j in range(k1)] + [Fraction(0)] * nu
        if i:
            h = fr(x[i]) - fr(x[i - 1])
            deg = len(prev) - 1
            for q in range(nu - 1, -1, -1):
                coeffs[deg - q] = sum(prev[deg - d] * math.comb(d, q) * h ** (d - q) for d in range(q, deg + 1))
        pieces.append(coeffs)
        prev = coeffs
    return pieces


def test_derivative_and_antiderivative():
    c = S.curve("random/50")
    p = IP.CubicSpline(c.x, c.y)
    cc = np.asarray(p.c)
    d = p.derivative()
    assert type(d) is IP.CubicSpline and d.extrapolate is True and np.asarray(d.c).shape == (3, 49)
    assert np.array_equal(np.asarray(d.c), cc[:3] * np.array([3.0, 2.0, 1.0])[:, None])
    assert np.array_equal(np.asarray(p.derivative(nu=4).c), np.zeros((1, 49)))
    assert np.array_equal(np.asarray(p.derivative(nu=-1).c), np.asarray(p.antiderivative().c))
    assert np.asarray(p.derivative(nu=0).c) is not None
    for nu in (1, 2):
        a = p.antiderivative(nu=nu)
        ac = np.asarray(a.c)
        exact = _exact_antiderivative(cc, c.x, nu)
        assert ac.shape == (4 + nu, 49)
        for i, coeffs in enumerate(exact):
            for j, want in enumerate(coeffs):
                # each coefficient: one division (the shifted ones) or a prefix sum of i Horner sums (the constants)
                tol = refdata.gamma(i + 12) * sum(abs(float(cf)) for cf in coeffs) * max(1, i) * 4 + 1e-300
                assert abs(float(fr(ac[j, i]) - want)) <= tol, (nu, i, j)
        assert float(a(c.x[0])) == 0.0
    per = IP.CubicSpline(*_periodic(), bc_type="periodic")
    assert per.antiderivative().extrapolate is False  # scipy's
    assert per.derivative().extrapolate == "periodic"


def _periodic():
    c = S.curve("periodic/sin16")
    return c.x, c.y


def test_integrate_against_exact_rationals():
    for cid in ("uniform/50", "random/50", "chebyshev/21", "spacing/1e10"):
        c = S.curve(cid)
        p = IP.CubicSpline(c.x, c.y)
        cc = np.asarray(p.c)
        anti = _exact_antiderivative(cc, c.x)
        g = np.random.default_rng(3)
        lo, hi = c.x[0] - 0.1 * (c.x[-1] - c.x[0]), c.x[-1] + 0.1 * (c.x[-1] - c.x[0])
        for a, b in list(zip(g.uniform(lo, hi, 20), g.uniform(lo, hi, 20))) + [(c.x[0], c.x[-1]), (c.x[3], c.x[3])]:

            def F(v):
                i = _interval(c.x, v)
                s = fr(v) - fr(c.x[i])
                return sum(cf * s ** (len(anti[i]) - 1 - j) for j, cf in enumerate(anti[i]))

            want = F(b) - F(a)
            got = p.integrate(a, b)
            scale = sum(abs(float(cf)) * abs(float(c.x[-1] - c.x[0])) ** 4 for cf in cc.ravel()) + abs(float(want))
            assert abs(fr(got) - want) <= Fraction(refdata.gamma(len(c.x) + 16) * scale), (cid, a, b)
            assert p.integrate(b, a) == -got
    p = IP.CubicSpline(*_periodic(), bc_type="periodic")
    one = p.integrate(p.x[0], p.x[-1])
    assert abs(p.integrate(-1.3, 7.9) - (p.integrate(-1.3, p.x[-1]) + p.integrate(p.x[0], 7.9 - 2 * math.pi))) < 1e-12
    assert abs(p.integrate(0.0, 4 * math.pi) - 2 * one) < 1e-14
    assert math.isnan(IP.PPoly([[1.0]], [0.0, 1.0], extrapolate=False).integrate(-1, 0.5))
    assert math.isnan(IP.PPoly([[1.0]], [0.0, 1.0]).integrate(math.nan, 0.5))  # [scipy differs]: ValueError


@LIVE
def test_integrate_and_calls_follow_scipy():
    g = np.random.default_rng(11)
    c = g.standard_normal((4, 6, 2))
    x = np.cumsum(g.uniform(0.5, 1.5, 7))
    mine, theirs = IP.PPoly(c, x), si.PPoly(c, x)
    for a, b in ((-1.0, 3.0), (2.5, 9.0), (8.0, 1.0), (x[0], x[-1])):
        assert np.allclose(np.asarray(mine.integrate(a, b)), theirs.integrate(a, b), rtol=1e-13, atol=1e-13)
    q = g.uniform(x[0] - 1, x[-1] + 1, 40)
    for ext in (True, False, "periodic"):
        for nu in (0, 1, 3, 5):
            assert np.allclose(np.asarray(mine(q, nu=nu, extrapolate=ext)), theirs(q, nu=nu, extrapolate=ext),
                               rtol=1e-13, atol=1e-12, equal_nan=True), (ext, nu)  # fmt: skip
        mi, th = IP.PPoly(c, x, extrapolate=ext), si.PPoly(c, x, extrapolate=ext)
        assert np.allclose(np.asarray(mi.integrate(-0.3, 11.2)), th.integrate(-0.3, 11.2), rtol=1e-12, atol=1e-12,
                           equal_nan=True), ext  # fmt: skip
    # decreasing breakpoints: scipy's mirrored interval rule, integrals in the usual orientation
    xd = x[::-1].copy()
    mine, theirs = IP.PPoly(c, xd), si.PPoly(c, xd)
    qq = np.concatenate([q, xd])
    assert np.allclose(np.asarray(mine(qq)), theirs(qq), rtol=1e-13, atol=1e-12)
    for a, b in ((-1.0, 3.0), (2.5, 9.0), (8.0, 1.0)):
        assert np.allclose(np.asarray(mine.integrate(a, b)), theirs.integrate(a, b), rtol=1e-12, atol=1e-12)
    # axis: c (..., k + 1, m, ...) with k + 1 at axis
    c4 = g.standard_normal((2, 4, 6, 3))
    mine, theirs = IP.PPoly(c4, x, axis=1), si.PPoly(c4, x, axis=1)
    assert np.asarray(mine.c).shape == theirs.c.shape
    assert np.allclose(np.asarray(mine(q.reshape(5, 8))), theirs(q.reshape(5, 8)), rtol=1e-13, atol=1e-12)


def test_decreasing_breakpoints_rule():
    p = IP.PPoly([[1.0, 2.0, 3.0]], [3.0, 2.0, 1.0, 0.0])
    assert np.asarray(p([3.0, 2.5, 2.0, 1.0, 0.0, -1.0, 4.0])).tolist() == [1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 1.0]
    assert p.integrate(0.5, 2.5) == 4.0 and p.integrate(-1.0, 4.0) == 10.0


def test_extend():
    p = IP.PPoly([[1.0, 2.0, 3.0]], [0.0, 1.0, 2.0, 3.0])
    p.extend([[5.0, 6.0]], [4.0, 5.0])
    assert np.asarray(p.x).tolist() == [0, 1, 2, 3, 4, 5] and np.asarray(p.c).tolist() == [[1, 2, 3, 5, 6]]
    p.extend([[7.0], [-1.0]], [-1.0])
    assert np.asarray(p.c).tolist() == [[7, 0, 0, 0, 0, 0], [-1, 1, 2, 3, 5, 6]]
    with pytest.raises(ValueError):
        p.c[0, 0] = 1.0
    with pytest.raises(ValueError, match=r"^extend\(\): 'x' must lie beyond either end"):
        p.extend([[1.0]], [2.5])
    with pytest.raises(ValueError, match=r"^extend\(\): 'x' must lie beyond either end"):
        p.extend([[1.0]], [5.0])  # [scipy differs]: a repeated breakpoint
    p.extend([[1j]], [6.0])
    assert np.asarray(p.c).dtype == np.complex128


def test_scalars_shapes_and_nan_points():
    p = IP.CubicSpline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 2.0])
    assert isinstance(p(0.5), float) and isinstance(p.integrate(0, 1), float)
    assert math.isnan(p(math.nan))
    assert np.asarray(p([[0.5, 1.5]])).shape == (1, 2)
    assert p(5.0, nu=7) == 0.0 and math.isnan(p(math.nan, nu=7))
    q = IP.CubicSpline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 2.0], extrapolate=False)
    assert math.isnan(q(5.0, nu=7)) and q(3.0) == 2.0


def test_r_natural_and_periodic_records_are_within_the_references():
    """R's splinefun natural and periodic: sd and R both near the exact interpolant (the comparison of IP2's row)."""
    worst = {"sd": 0.0, "R": 0.0}
    for meth in ("natural", "periodic"):
        for cid in S.real_curve_ids():
            rec = R.get(f"{meth}/{cid}")
            if rec is None or cid in ("spacing/1e10", "large/1e300", "subnormal/x"):
                continue
            c = S.curve(cid)
            m = CURVES.methods[meth]
            pos = np.flatnonzero(CURVES.inputs["set"][m.index] == cid)
            order = np.argsort(CURVES.inputs["index"][m.index[pos]], kind="stable")
            ref = m.hi[pos[order]]
            inside = (c.xe >= c.x[0]) & (c.xe <= c.x[-1])
            scale = np.max(np.abs(c.y))
            mine = np.asarray(IP.CubicSpline(c.x, c.y, bc_type=meth)(c.xe))
            worst["sd"] = max(worst["sd"], float(np.max(np.abs(mine - ref)[inside]) / scale))
            worst["R"] = max(worst["R"], float(np.max(np.abs(np.array(rec["values"]) - ref)[inside]) / scale))
    assert worst["sd"] < 1e-13 and worst["R"] < 1e-9, worst
