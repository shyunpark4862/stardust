"""make_lsq_spline (docs/api/smooth.md §3; docs/design/smooth.md §3.3, §4.2; SM3 milestone): the coefficient
contract against the exact references, scipy's weights (sum (w r)^2), the Schoenberg-Whitney check."""

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd
from stardust.linalg import LinAlgError

from . import refdata
from . import samples as S
from .conftest import LIVE
from .test_smoothing_spline import group_value, per_set

RC = refdata.ref("spline")
RF = refdata.ref("spline_fit")
SETS = [sid for sid in RC.group_inputs["set"].tolist() if np.any(RC.inputs["set"][RC.methods["lsq"].index] == sid)]


@pytest.mark.parametrize("sid", SETS)
def test_reference_contract(sid):
    s = S.scatter(sid)
    gc, gf = refdata.group_of(RC, sid), refdata.group_of(RF, sid)
    t = np.asarray(RC.group_inputs["t_lsq"][gc])
    _, _, tf = group_value(RF, "trl_lsq", gf)
    kappa = float(tf["kappa"])
    w = None if s.w is None else sd.asarray(s.w)
    b = sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), sd.asarray(t), w=w)
    idx, hi, lo, _ = per_set(RC, "lsq", sid)
    got = np.asarray(b.c)
    bound = refdata.spline_bound(kappa, float(np.abs(hi).max()))
    for i, h, l in zip(idx, hi, lo):
        assert refdata.units(float(got[i]), h, l, bound) <= 1.0, (sid, i)


@LIVE
def test_matches_scipy():
    s = S.scatter("random/200")
    t = np.r_[[s.x[0]] * 4, s.x[10:-10:20], [s.x[-1]] * 4]
    w = 0.5 + np.random.default_rng(0).random(s.n)
    for k in (1, 3, 5):
        tk = np.r_[[s.x[0]] * (k + 1), s.x[10:-10:20], [s.x[-1]] * (k + 1)]
        a = si.make_lsq_spline(s.x, s.y, tk, k=k, w=w)
        b = sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), sd.asarray(tk), k=k, w=sd.asarray(w))
        assert np.abs(a.c - np.asarray(b.c)).max() <= 1e-12 * np.abs(a.c).max()
    Y = np.stack([s.y, np.cos(s.x)], axis=1)
    a = si.make_lsq_spline(s.x, Y, t)
    b = sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(Y), sd.asarray(t))
    assert np.abs(a.c - np.asarray(b.c)).max() <= 1e-12


def test_schoenberg_whitney():
    x = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.95, 1.0])
    t = np.r_[[0.0] * 4, 0.4, 0.5, 0.6, [1.0] * 4]
    with pytest.raises(LinAlgError, match="Schoenberg-Whitney"):
        sd.smooth.make_lsq_spline(sd.asarray(x), sd.asarray(np.sin(x)), sd.asarray(t))
    w = np.ones(7)
    x2 = np.array([0.0, 0.3, 0.45, 0.55, 0.7, 0.9, 1.0])
    sd.smooth.make_lsq_spline(sd.asarray(x2), sd.asarray(np.sin(x2)), sd.asarray(t))
    w[3] = 0.0  # a zero weight takes its point out of the check
    with pytest.raises(LinAlgError, match="Schoenberg-Whitney"):
        sd.smooth.make_lsq_spline(sd.asarray(x2), sd.asarray(np.sin(x2)), sd.asarray(t), w=sd.asarray(w))


def test_unsorted_and_errors():
    s = S.scatter("unsorted/30")
    xs = np.sort(s.x)
    t = np.r_[[xs[0]] * 4, xs[5:-5:5], [xs[-1]] * 4]
    b = sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), sd.asarray(t))
    o = np.argsort(s.x)
    b2 = sd.smooth.make_lsq_spline(sd.asarray(s.x[o]), sd.asarray(s.y[o]), sd.asarray(t))
    assert np.asarray(b.c).tobytes() == np.asarray(b2.c).tobytes()
    with pytest.raises(ValueError, match="within the knots"):
        sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), sd.asarray(t[4:]))
    with pytest.raises(ValueError, match="nondecreasing"):
        sd.smooth.make_lsq_spline(sd.asarray(s.x), sd.asarray(s.y), sd.asarray(t[::-1].copy()))
