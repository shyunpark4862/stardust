"""pspline (docs/api/smooth.md §3; docs/design/smooth.md §3.3, §4.2; SM3 milestone): the contracts against the
exact references, the limits lam = 0 (the least-squares spline) and lam = inf (a polynomial), the search."""

import math

import numpy as np
import pytest

import stardust as sd
from stardust.linalg import LinAlgError

from . import refdata
from . import samples as S
from .test_smoothing_spline import group_value, per_set

RC = refdata.ref("spline")
RF = refdata.ref("spline_fit")
PS = ("1e-2", "1", "1e3")


def cases():
    out = []
    for sid in RC.group_inputs["set"].tolist():
        for c in PS:
            if np.any(RC.inputs["set"][RC.methods[f"ps_{c}"].index] == sid):
                out.append((sid, c))
    return out


CASES = cases()


@pytest.mark.parametrize("sid,c", CASES, ids=[f"{s}-{c}" for s, c in CASES])
def test_reference_contracts(sid, c):
    s = S.scatter(sid)
    gc, gf = refdata.group_of(RC, sid), refdata.group_of(RF, sid)
    nseg = int(RC.group_inputs["n_segments"][gc])
    _, _, tf = group_value(RF, f"trl_ps_{c}", gf)
    kappa = float(tf["kappa"])
    w = None if s.w is None else sd.asarray(s.w)
    try:
        r = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), lam=float(c), n_segments=nseg, weights=w)
    except LinAlgError:
        assert kappa > 2.0**38, (sid, c, kappa)
        return
    t_ref = np.asarray(RC.group_inputs["t_ps"][gc])
    assert np.asarray(r.spline.t).tobytes() == t_ref.tobytes()
    idx, hi, lo, _ = per_set(RC, f"ps_{c}", sid)
    got = np.asarray(r.spline.c)
    bound = refdata.spline_bound(kappa, float(np.abs(hi).max()))
    for i, h, l in zip(idx, hi, lo):
        assert refdata.units(float(got[i]), h, l, bound) <= 1.0, (sid, c, i)
    idx, _, _, ff = per_set(RF, f"ps_{c}", sid)
    lev = np.asarray(r.leverage)
    lb = refdata.leverage_bound(kappa)
    for i, v in zip(idx, ff["leverage"]):
        assert refdata.units(float(lev[i]), float(v), 0.0, lb) <= 1.0, (sid, c, i)
    th, tl, _ = group_value(RF, f"trl_ps_{c}", gf)
    assert refdata.units(r.df, th, tl, refdata.trace_bound(s.n, kappa)) <= 1.0


def test_lam_zero_is_the_least_squares_spline():
    s = S.scatter("random/200")
    r = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), lam=0, n_segments=12)
    b = sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), r.spline.t)
    assert np.asarray(r.spline.c).tobytes() == np.asarray(b.c).tobytes()
    with pytest.raises(LinAlgError, match="least-squares spline"):
        sd.smooth.pspline(sd.asarray(s.x[:20]), sd.asarray(s.y[:20]), lam=0, n_segments=40)


@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_lam_inf_is_a_polynomial(d):
    s = S.scatter("random/200")
    r = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), lam=math.inf, diff_order=d)
    coef = np.polyfit(s.x, s.y, d - 1)
    assert np.abs(np.asarray(r.fittedvalues) - np.polyval(coef, s.x)).max() <= 1e-10
    assert abs(r.df - d) <= 1e-10


def test_search_and_df():
    s = S.scatter("random/200")
    r = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y))
    assert r.converged and r.n_segments == 40 and r.k == 3 and r.diff_order == 2
    for f in (0.9, 1.1):
        o = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), lam=r.lam * f)
        assert o.gcv >= r.gcv * (1 - 1e-12)
    rd = sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), df=8.0)
    assert abs(rd.df - 8.0) <= 8e-10
    with pytest.raises(ValueError, match="'df' must be in"):
        sd.smooth.pspline(sd.asarray(s.x), sd.asarray(s.y), df=44.0)


def test_large_n_uses_few_knots():
    g = np.random.default_rng(2)
    x = np.sort(10.0 * g.random(100_000))
    y = np.sin(x) + 0.3 * g.standard_normal(x.size)
    r = sd.smooth.pspline(sd.asarray(x), sd.asarray(y))
    assert r.converged and np.asarray(r.spline.c).size == 43
    assert np.abs(np.asarray(r.fittedvalues) - np.sin(x)).max() < 0.05


def test_arguments():
    s = S.scatter("random/50")
    x, y = sd.asarray(s.x), sd.asarray(s.y)
    r = sd.smooth.pspline(x, y, xb=-1.0, xe=11.0, n_segments=5, k=2, diff_order=3, lam=1.0)
    assert float(r.spline.t[2]) == -1.0 and r.spline.k == 2
    with pytest.raises(ValueError, match="'diff_order' must be in"):
        sd.smooth.pspline(x, y, k=1, diff_order=3)
    with pytest.raises(ValueError, match="within"):
        sd.smooth.pspline(x, y, xb=1.0)
    with pytest.raises(ValueError, match="'n_segments' must be >= 1"):
        sd.smooth.pspline(x, y, n_segments=0)
    assert repr(r).startswith("PSplineResult(nobs=50")
    assert "P-Spline Results" in str(r.summary())
