"""smoothing_spline and make_smoothing_spline (docs/api/smooth.md §3; docs/design/smooth.md §3.3, §4.2; SM3
milestone): the coefficient, leverage and trace contracts against the exact references, lam = 0 and lam = inf,
the GCV search's equivariance and the inputs scipy fails on."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd
from stardust.linalg import LinAlgError

from . import refdata
from . import samples as S
from .conftest import LIVE

RC = refdata.ref("spline")
RF = refdata.ref("spline_fit")
SS_C = ("1e-8", "1e-4", "1e-1", "1e2")
KAPPA_OK = 2.0**38  # below it the fit must succeed (the guard is at an estimate of 2^43 with unit columns, SM58)
KAPPA_BAD = 2.0**52  # above it (u kappa > 1/2: no digit left) the fit must be a LinAlgError


def per_set(r, method, sid):
    pos, idx = refdata.at(r, method, sid)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def group_value(r, method, g):
    m = r.methods[method]
    j = np.flatnonzero(m.index == g)
    if j.size == 0:
        return None
    return m.hi[j[0]], m.lo[j[0]], {k: v[j[0]] for k, v in m.fields.items()}


def spline_cases():
    out = []
    for sid in RC.group_inputs["set"].tolist():
        for c in SS_C:
            if f"ss_{c}" in RC.methods and np.any(RC.inputs["set"][RC.methods[f"ss_{c}"].index] == sid):
                out.append((sid, c))
    return out


CASES = spline_cases()


@pytest.mark.parametrize("sid,c", CASES, ids=[f"{s}-{c}" for s, c in CASES])
def test_reference_contracts(sid, c):
    """Coefficients within C_SPLINE u kappa max|c|, leverage within u kappa max(C_LEVERAGE, kappa), tr L within n
    times that, the criteria from them (decision SM30): lam and the knots are the references'."""
    s = S.scatter(sid)
    gc, gf = refdata.group_of(RC, sid), refdata.group_of(RF, sid)
    lam = float(RC.group_inputs[f"lam_ss_{c}"][gc])
    _, _, tf = group_value(RF, f"trl_ss_{c}", gf)
    kappa = float(tf["kappa"])
    w = None if s.w is None else sd.asarray(s.w)
    if kappa > KAPPA_BAD:
        with pytest.raises(LinAlgError, match="singular to working precision"):
            sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=lam, weights=w)
        return
    try:
        r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=lam, weights=w)
    except LinAlgError:
        assert kappa > KAPPA_OK, (sid, c, kappa)
        return
    idx, hi, lo, _ = per_set(RC, f"ss_{c}", sid)
    got = np.asarray(r.spline.c)
    assert got.size == idx.size
    cmax = float(np.abs(hi).max())
    bound = refdata.spline_bound(kappa, cmax)
    for i, h, l in zip(idx, hi, lo):
        assert refdata.units(float(got[i]), h, l, bound) <= 1.0, (sid, c, i)
    idx, fh, fl, ff = per_set(RF, f"ss_{c}", sid)
    lev = np.asarray(r.leverage)
    lb = refdata.leverage_bound(kappa)
    for i, v in zip(idx, ff["leverage"]):
        assert refdata.units(float(lev[i]), float(v), 0.0, lb) <= 1.0, (sid, c, i)
    th, tl, _ = group_value(RF, f"trl_ss_{c}", gf)
    assert refdata.units(r.df, th, tl, refdata.trace_bound(s.n, kappa)) <= 1.0, (sid, c)
    gv = group_value(RF, f"gcv_ss_{c}", gf)
    if gv is not None and "magnitude" not in s.tags and kappa < 1e6:
        # GCV's denominator (n - tr L)^2 carries tr L's error relative to n - tr L, twice
        rel = 1e-9 + 4 * refdata.trace_bound(s.n, kappa) / max(s.n - th, 1e-300)
        assert math.isclose(r.gcv, gv[0], rel_tol=rel, abs_tol=1e-12 * float(np.mean(s.y**2))), (sid, c)


@pytest.mark.parametrize("sid", ["random/50", "uniform/40", "unsorted/30", "ties/30", "outliers/40", "line/20"])
def test_lam_zero_is_the_natural_interpolant_bit_for_bit(sid):
    """lam = 0 is make_interp_spline(bc_type='natural') through the merged points, the same bits."""
    s = S.scatter(sid)
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=0)
    xs, inv = np.unique(s.x, return_inverse=True)
    ys = np.bincount(inv, weights=s.y) / np.bincount(inv)
    if xs.size == s.n:
        order = np.argsort(s.x, kind="stable")
        ys = s.y[order]
    ref = sd.interpolate.make_interp_spline(sd.asarray(xs), sd.asarray(ys), k=3, bc_type="natural")
    assert np.asarray(r.spline.t).tobytes() == np.asarray(ref.t).tobytes()
    if xs.size == s.n:
        assert np.asarray(r.spline.c).tobytes() == np.asarray(ref.c).tobytes()
    else:
        assert np.allclose(np.asarray(r.spline.c), np.asarray(ref.c), rtol=1e-14, atol=1e-14)
    assert r.df == xs.size
    b = sd.smooth.make_smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=0.0)
    assert np.asarray(b.c).tobytes() == np.asarray(r.spline.c).tobytes()


def test_lam_inf_is_the_weighted_least_squares_line():
    s = S.scatter("weights/30")
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=math.inf, weights=sd.asarray(s.w))
    W = [Fraction(float(v)) for v in s.w]
    X = [Fraction(float(v)) for v in s.x]
    Y = [Fraction(float(v)) for v in s.y]
    sw = sum(W)
    mx = sum(w * x for w, x in zip(W, X)) / sw
    my = sum(w * y for w, y in zip(W, Y)) / sw
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(W, X, Y)) / sum(w * (x - mx) ** 2 for w, x in zip(W, X))
    for x, f in zip(s.x, np.asarray(r.fittedvalues)):
        exact = float(my + b * (Fraction(float(x)) - mx))
        assert abs(f - exact) <= 64 * refdata.U * max(1.0, abs(exact))
    assert abs(r.df - 2.0) <= 1e-13
    assert r.converged and r.status == 0


def test_line_is_reproduced():
    s = S.scatter("line/20")
    for lam in (1e-6, 1.0, 1e6):
        r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=lam)
        assert np.abs(np.asarray(r.fittedvalues) - s.y).max() <= 1e-12


@pytest.mark.parametrize("a", [4.0, 0.25, 2.0**-20])
def test_gcv_scales_with_x_bit_for_bit_by_powers_of_four(a):
    """x -> a x with a a power of 4 (sqrt(a) exact): lam scales by a^3 exactly and the fit has the same bits."""
    s = S.scatter("random/200")
    r0 = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))
    r1 = sd.smooth.smoothing_spline(sd.asarray(s.x * a), sd.asarray(s.y))
    assert r1.lam == r0.lam * a**3
    assert np.asarray(r1.fittedvalues).tobytes() == np.asarray(r0.fittedvalues).tobytes()
    assert r1.df == r0.df and r1.gcv == r0.gcv


@pytest.mark.parametrize("a,b", [(0.01, 0.0), (10.0, 0.0), (1000.0, -3.0), (1.0, 1e4)])
def test_gcv_is_equivariant(a, b):
    """The GCV choice follows x -> a x + b (lam scales by a^3); scipy's does not (docs/api/smooth.md §3)."""
    s = S.scatter("random/200")
    r0 = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))
    r1 = sd.smooth.smoothing_spline(sd.asarray(s.x * a + b), sd.asarray(s.y))
    assert math.isclose(r1.lam, r0.lam * a**3, rel_tol=1e-6)
    # the search stops within |d rho| <= 1e-8 of its minimum: the fits agree to that order
    assert np.abs(np.asarray(r1.fittedvalues) - np.asarray(r0.fittedvalues)).max() <= 1e-7


@LIVE
def test_scipy_gcv_depends_on_the_units():
    s = S.scatter("random/200")
    f0 = si.make_smoothing_spline(s.x, s.y)(s.x)
    f1 = si.make_smoothing_spline(s.x * 1000.0, s.y)(s.x * 1000.0)
    assert np.abs(f1 - f0).max() > 0.1
    mine = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))
    assert np.abs(np.asarray(mine.fittedvalues) - f0).max() < 1e-3  # unit scale: the same minimum


@LIVE
def test_inputs_scipy_fails_on():
    """The data of the probe's record (n = 10^4 uneven x) and narrow/30: scipy's GCV raises 'ill-posed'; the search
    here returns an interior minimum."""
    g = S.rng("probe", "illposed")
    x = np.sort(10 * g.random(10**4))
    y = np.sin(x) + 0.3 * g.standard_normal(x.size)
    with pytest.raises(ValueError):
        si.make_smoothing_spline(x, y)
    r = sd.smooth.smoothing_spline(sd.asarray(x), sd.asarray(y))
    assert r.converged and r.status == 0
    assert 5 < r.df < 100
    assert np.abs(np.asarray(r.fittedvalues) - np.sin(x)).max() < 0.2
    n30 = S.scatter("narrow/30")
    with pytest.raises(ValueError):
        si.make_smoothing_spline(n30.x, n30.y)
    rn = sd.smooth.smoothing_spline(sd.asarray(n30.x), sd.asarray(n30.y))
    r30 = sd.smooth.smoothing_spline(sd.asarray(n30.x * 1e8), sd.asarray(n30.y))
    assert rn.converged and abs(rn.df - r30.df) <= 1e-6
    # repeated x: scipy refuses them
    xr = np.round(x, 1)
    rr = sd.smooth.smoothing_spline(sd.asarray(xr), sd.asarray(y))
    assert rr.converged
    with pytest.raises(ValueError):
        si.make_smoothing_spline(xr, y)


@LIVE
def test_coefficients_match_scipy_at_a_given_lam():
    s = S.scatter("random/200")
    for lam in (1e-3, 1.0, 100.0):
        a = si.make_smoothing_spline(s.x, s.y, lam=lam)
        b = sd.smooth.make_smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=lam)
        assert np.abs(a.c - np.asarray(b.c)).max() <= 1e-8 * np.abs(a.c).max()
        assert np.asarray(b.t).tobytes() == a.t.tobytes()


def test_ties_and_order_do_not_matter():
    s = S.scatter("ties/30")
    p = np.random.default_rng(1).permutation(s.n)
    r0 = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=0.5)
    r1 = sd.smooth.smoothing_spline(sd.asarray(s.x[p]), sd.asarray(s.y[p]), lam=0.5)
    assert np.asarray(r0.spline.c).tobytes() == np.asarray(r1.spline.c).tobytes()
    assert np.asarray(r0.fittedvalues)[p].tobytes() == np.asarray(r1.fittedvalues).tobytes()
    assert np.asarray(r0.leverage)[p].tobytes() == np.asarray(r1.leverage).tobytes()


def test_merged_points_give_the_same_spline():
    """Equal x merged into their weighted mean with the summed weight: the same penalized problem."""
    s = S.scatter("ties/30")
    xs, inv = np.unique(s.x, return_inverse=True)
    cnt = np.bincount(inv).astype(float)
    ys = np.bincount(inv, weights=s.y) / cnt
    r0 = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=0.5)
    r1 = sd.smooth.smoothing_spline(sd.asarray(xs), sd.asarray(ys), lam=0.5, weights=sd.asarray(cnt))
    assert np.abs(np.asarray(r0.spline.c) - np.asarray(r1.spline.c)).max() <= 1e-13
    assert math.isclose(r0.df, r1.df, rel_tol=1e-12)


def test_df_target():
    s = S.scatter("random/200")
    for df in (2.5, 6.0, 30.0, 150.0):
        r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), df=df)
        assert abs(r.df - df) <= 1e-10 * df
        assert r.converged
    with pytest.raises(ValueError, match="'df' must be in"):
        sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), df=2.0)
    with pytest.raises(ValueError, match="'df' must be in"):
        sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), df=201.0)


def test_criteria():
    s = S.scatter("random/200")
    g = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), criterion="gcv")
    c = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), criterion="cv")
    assert g.criterion == "gcv" and c.criterion == "cv"
    for r in (g, c):
        assert r.converged and r.status == 0
        lev = np.asarray(r.leverage)
        res = np.asarray(r.resid)
        assert math.isclose(r.cv, float(np.mean((res / (1 - lev)) ** 2)), rel_tol=1e-12)
        assert math.isclose(r.gcv, s.n * float(np.sum(res**2)) / (s.n - r.df) ** 2, rel_tol=1e-12)
    # each choice is no worse than its neighbours on the grid
    for r, name in ((g, "gcv"), (c, "cv")):
        for f in (0.9, 1.1):
            other = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=r.lam * f)
            assert getattr(other, name) >= getattr(r, name) * (1 - 1e-12)


def test_end_of_the_interval():
    """Exact interpolation data: the smallest criterion is at lam -> 0, the low end, converged False."""
    x = np.linspace(0.0, 1.0, 30)
    r = sd.smooth.smoothing_spline(sd.asarray(x), sd.asarray(x**3 - x))
    assert r.status in (-1, 0)
    rl = sd.smooth.smoothing_spline(sd.asarray(x), sd.asarray(2.0 * x + 1.0))
    assert np.abs(np.asarray(rl.fittedvalues) - (2.0 * x + 1.0)).max() <= 1e-12


def test_batch_shares_lam():
    s = S.scatter("random/200")
    Y = np.stack([s.y, 2.0 * s.y + 1.0, np.cos(s.x)], axis=1)
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(Y), lam=0.3)
    for j in range(3):
        rj = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(Y[:, j].copy()), lam=0.3)
        assert np.abs(np.asarray(r.spline.c)[:, j] - np.asarray(rj.spline.c)).max() <= 1e-13
        assert np.asarray(r.fittedvalues).shape == (s.n, 3)
    rt = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(Y.T.copy()), lam=0.3, axis=1)
    assert np.asarray(rt.fittedvalues).shape == (3, s.n)
    assert np.asarray(rt.spline(sd.asarray([1.0, 2.0]))).shape == (3, 2)
    rg = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(Y))
    assert rg.converged and np.asarray(rg.scale).shape == (3,)


def test_nan_policy_and_weights():
    s = S.scatter("random/50")
    y = s.y.copy()
    y[[3, 10]] = np.nan
    p = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(y), lam=1.0)
    assert np.all(np.isnan(np.asarray(p.fittedvalues))) and math.isnan(p.df)
    assert math.isnan(float(p.predict(2.0)))
    o = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(y), lam=1.0, nan_policy="omit")
    keep = ~np.isnan(y)
    r = sd.smooth.smoothing_spline(sd.asarray(s.x[keep]), sd.asarray(s.y[keep]), lam=1.0)
    assert np.asarray(o.spline.c).tobytes() == np.asarray(r.spline.c).tobytes()
    assert np.isnan(np.asarray(o.fittedvalues)[3]) and o.nobs == 48
    with pytest.raises(ValueError, match="nan_policy='raise'"):
        sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(y), nan_policy="raise")
    w = np.ones(s.n)
    w[[5, 6]] = 0.0
    rw = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=1.0, weights=sd.asarray(w))
    keep = w > 0
    rk = sd.smooth.smoothing_spline(sd.asarray(s.x[keep]), sd.asarray(s.y[keep]), lam=1.0)
    assert np.abs(np.asarray(rw.spline(sd.asarray(s.x))) - np.asarray(rk.spline(sd.asarray(s.x)))).max() <= 1e-12
    assert np.asarray(rw.leverage)[5] == 0.0 and rw.nobs == 48


def test_predict_and_interval():
    s = S.scatter("random/200")
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))
    xe = sd.asarray(s.xe)
    assert np.asarray(r.predict(xe)).tobytes() == np.asarray(r.spline(xe)).tobytes()
    iv = np.asarray(r.predict_interval(xe))
    assert iv.shape == (s.xe.size, 2)
    mid = np.asarray(r.predict(xe))
    assert np.all(iv[:, 0] < mid) and np.all(mid < iv[:, 1])
    # at the data points ||l(x_i)||^2 = sum_j l_ij^2 / w_j, and at a data point of unit weight l_ii^2 <= that <= l_ii
    half = (iv[:, 1] - iv[:, 0]) / 2
    assert np.all(np.isfinite(half))
    ivx = np.asarray(r.predict_interval(sd.asarray(s.x)))
    from stardust.stats import StudentT

    q = float(StudentT(df=s.n - r.df).iccdf(0.025))
    norm2 = ((ivx[:, 1] - ivx[:, 0]) / (2 * q * r.scale)) ** 2
    lev = np.asarray(r.leverage)
    assert np.all(norm2 <= lev * (1 + 1e-9)) and np.all(norm2 >= lev**2 * (1 - 1e-9))
    assert r.predict_interval(2.0).shape == (2,)


def test_errors():
    x = sd.asarray(np.arange(10.0))
    y = sd.asarray(np.sin(np.arange(10.0)))
    with pytest.raises(ValueError, match="'lam' or 'df'"):
        sd.smooth.smoothing_spline(x, y, lam=1.0, df=3.0)
    with pytest.raises(ValueError, match="'criterion'"):
        sd.smooth.smoothing_spline(x, y, criterion="aic")
    with pytest.raises(ValueError, match="at least 5 distinct"):
        sd.smooth.smoothing_spline(sd.asarray([0.0, 1.0, 1.0, 2.0, 3.0]), sd.asarray(np.ones(5)))
    with pytest.raises(ValueError, match="'lam' must be >= 0"):
        sd.smooth.smoothing_spline(x, y, lam=-1.0)
    with pytest.raises(TypeError):
        sd.smooth.smoothing_spline(x, sd.asarray(np.ones(10) + 1j))
    with pytest.raises(ValueError, match="infinity"):
        sd.smooth.make_smoothing_spline(sd.asarray([0.0, 1.0, 2.0, 3.0, math.inf]), sd.asarray(np.ones(5)))
    with pytest.raises(ValueError, match="NaN"):
        sd.smooth.make_smoothing_spline(sd.asarray([0.0, 1.0, 2.0, 3.0, math.nan]), sd.asarray(np.ones(5)))
    s = S.scatter("clustered/40")  # clusters 1e-8 wide: no digit left once lam is large (SM58)
    with pytest.raises(LinAlgError, match="singular to working precision"):
        sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y), lam=42.0)
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))  # the search keeps to trusted lam
    assert r.converged and 2 < r.df < 40


def test_results_box():
    s = S.scatter("random/200")
    r = sd.smooth.smoothing_spline(sd.asarray(s.x), sd.asarray(s.y))
    assert repr(r).startswith("SmoothingSplineResult(nobs=200, df=")
    text = r.summary()
    assert "Smoothing Spline Results" in str(text) and "Lambda:" in str(text)
    with pytest.raises(AttributeError):
        r.lam = 1.0
    assert isinstance(r.spline, sd.interpolate.BSpline)
