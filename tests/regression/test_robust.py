"""G4: rlm (M and MM), theilslopes, siegelslopes (docs/api/regression.md §5,
docs/design/regression.md §3.6).

Accuracy against Arb (tests/regression/ref/robust.json: the estimating equations solved and
certified on the solution's pieces), statsmodels' RLM at a tight tolerance, R's MASS::rlm and
robustbase::lmrob (tests/regression/ref/robust_r.json, made by tools/gen_robust_r.py; R is
not needed here), MM's breakdown, determinism, batches, threads, NaN policies, edge cases and
errors; the slopes against scipy bit for bit and against exact rational medians."""

import math
from fractions import Fraction
import itertools
import json
import pathlib
import warnings

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from . import refdata_robust

R = sd.regression
K = sd._core.regression
REFS = refdata_robust.load()
RREF = json.loads((pathlib.Path(__file__).parent / "ref" / "robust_r.json").read_text())


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def dd(v):
    return np.array([p[0] for p in v]) + np.array([p[1] for p in v])


def norm_err(got, ref):
    hi = np.array([v[0] for v in ref])
    lo = np.array([v[1] for v in ref])
    return np.linalg.norm((a(got) - hi) - lo) / np.linalg.norm(hi + lo)


def rel(got, ref):
    got, ref = a(got), np.asarray(ref, dtype=float)
    return float(np.max(np.abs(got - ref) / np.abs(ref)))


def contaminated(n=200, p=3, frac=0.1, seed=0, shift=10.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, p)) * rng.uniform(0.5, 2.0, p) + 1.0
    y = X @ np.linspace(1.0, -1.0, p) + 0.5 + rng.standard_normal(n)
    m = int(frac * n)
    y[:m] += shift
    return y, X


def leverage(n=300, frac=0.35, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1.0, 2.0]) + 0.5 + 0.5 * rng.standard_normal(n)
    m = int(frac * n)
    X[:m] = rng.normal(8.0, 0.5, (m, 2))
    y[:m] = rng.normal(-20.0, 1.0, m)
    return y, X


# ---- Arb references

@pytest.mark.parametrize("name", list(REFS))
def test_against_arb(name):
    p = REFS[name]
    r = refdata_robust.fit(p)
    ref = p["ref"]
    assert r.converged
    assert norm_err(r.params, ref["params"]) <= 4e-16
    assert abs(r.scale - sum(ref["scale"])) <= 4e-15 * sum(ref["scale"])
    assert rel(r.bse, dd(ref["bse"])) <= 4e-15
    assert np.max(np.abs(a(r.weights) - dd(ref["weights"]))) <= 4e-15
    if p["method"] == "MM":
        # the S estimate against its certified fixed point (regref.s_estimate, rounded): the
        # S iteration's Newton steps (G82) reach it to rounding (measured 9.4e-16, 2.2e-16)
        assert rel(r._raw[8], p["init_params"]) <= 2e-15
        assert r.scale == pytest.approx(p["s_scale"], rel=1e-15)


# ---- statsmodels

def _sm_norm(norm):
    from statsmodels.robust import norms

    return {"huber": norms.HuberT(), "bisquare": norms.TukeyBiweight(), "hampel": norms.Hampel()}[norm]


@pytest.mark.parametrize("norm", ["huber", "bisquare", "hampel"])
@pytest.mark.parametrize("scale_est", ["mad", "huber"])
@pytest.mark.parametrize("seed", [0, 1])
def test_against_statsmodels(norm, scale_est, seed):
    import statsmodels.api as sm
    from statsmodels.robust import scale

    y, X = contaminated(n=150 + 7 * seed, seed=seed)
    r = R.rlm(y, X, norm=norm, scale_est=scale_est)
    est = "mad" if scale_est == "mad" else scale.HuberScale(d=2.5, tol=1e-15, maxiter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = sm.RLM(y, sm.add_constant(X), M=_sm_norm(norm)).fit(scale_est=est, tol=1e-15, maxiter=1000)
        d = sm.RLM(y, sm.add_constant(X), M=_sm_norm(norm)).fit(scale_est=est if scale_est == "mad" else
                                                                 scale.HuberScale())
    assert r.converged
    assert np.max(np.abs(a(r.params) - s.params)) <= 1e-13 * np.max(np.abs(s.params))
    assert r.scale == pytest.approx(s.scale, rel=1e-13)
    assert rel(r.bse, s.bse) <= 1e-13
    assert np.max(np.abs(a(r.weights) - s.weights)) <= 1e-12
    assert a(r.resid) == pytest.approx(s.resid, abs=1e-12)
    # statsmodels' defaults stop earlier (tol 1e-8 on the objective's absolute change)
    assert np.max(np.abs(a(r.params) - d.params)) <= 1e-5 * np.max(np.abs(s.params))
    # t(df_resid) where statsmodels uses the normal (G57)
    t = a(r.params) / a(r.bse)
    assert a(r.pvalues) == pytest.approx(2 * st.t.sf(np.abs(t), r.df_resid), rel=1e-12)
    assert a(r.tvalues) == pytest.approx(s.tvalues, rel=1e-12)
    ci = a(r.conf_int(alpha=0.1))
    q = st.t.ppf(0.95, r.df_resid)
    assert ci[:, 1] == pytest.approx(a(r.params) + q * a(r.bse), rel=1e-12)
    assert r.nobs == 150 + 7 * seed and r.df_model == 3 and r.df_resid == r.nobs - 4


def test_summary():
    r = R.rlm(*contaminated(), norm="bisquare")
    txt = r.summary()
    for s in ("Robust linear Model Regression Results", "TukeyBiweight", "Scale Est.:", "mad", "H1", "P>|t|",
              "No. Iterations:"):
        assert s in txt
    assert "x3" in txt and "const" in txt
    mm = R.rlm(*leverage(), method="MM", rng=sd.random.default_rng(1)).summary()
    assert "avar1" in mm and "MM" in mm
    assert "summary(): takes one regression" in _err(lambda: R.rlm(sd.asarray(np.ones((2, 30))),
                                                                   sd.asarray(np.arange(30.0))).summary())
    assert repr(r).startswith("RLMResult(method='M', norm='bisquare'")


# ---- R: MASS::rlm and robustbase::lmrob

R_PROBLEMS = [k for k in RREF if not k.startswith("_")]


def _kappa_factor(r, norm):
    """MASS's summary.rlm uses var(psi') with n - 1 where H1 (statsmodels) uses n."""
    z = a(r.resid) / r.scale
    dp = _sm_norm(norm).psi_deriv(z)
    n, p, m = len(z), r.rank, dp.mean()
    return (1 + p / n * dp.var(ddof=1) / m**2) / (1 + p / n * dp.var() / m**2)


@pytest.mark.parametrize("name", R_PROBLEMS)
@pytest.mark.parametrize("norm", ["huber", "bisquare", "hampel"])
def test_against_mass(name, norm):
    pr = RREF[name]
    Rv = pr["R"]
    # proposal 2 with k2 = 2.5 is scale_est='huber' (statsmodels' HuberScale(d=2.5)): the same fit
    r = R.rlm(pr["y"], pr["X"], norm=norm, scale_est="huber")
    tag = f"mass_{norm}_huber"
    assert Rv[tag + "_conv"][0] == 1
    assert rel(r.params, Rv[tag + "_coef"]) <= 1e-13 or np.max(np.abs(a(r.params) - Rv[tag + "_coef"])) <= 1e-13
    assert r.scale == pytest.approx(Rv[tag + "_scale"][0], rel=1e-13)
    assert rel(a(r.bse) * _kappa_factor(r, norm), Rv[tag + "_se"]) <= 1e-12
    # MASS's MAD divides by 0.6745 where sd divides by Phi^-1(3/4) = 0.674489750...: a scale
    # 1.5e-5 apart and coefficients about 1e-6 apart
    r = R.rlm(pr["y"], pr["X"], norm=norm, scale_est="mad")
    tag = f"mass_{norm}_mad"
    assert r.scale == pytest.approx(Rv[tag + "_scale"][0], rel=3e-5)
    assert np.max(np.abs(a(r.params) - Rv[tag + "_coef"])) <= 1e-5 * np.max(np.abs(Rv[tag + "_coef"]))


@pytest.mark.parametrize("name", R_PROBLEMS)
def test_against_lmrob(name):
    pr = RREF[name]
    Rv = pr["R"]
    r = R.rlm(pr["y"], pr["X"], method="MM", rng=sd.random.default_rng(pr["seed"]))
    assert r.converged
    # the S start the R values were made from: the certified fixed point near sd's S estimate
    # (regref.s_estimate, rounded), which sd reaches to rounding (G82; measured 3.8e-16, 0)
    assert rel(r._raw[8], pr["S_params"]) <= 2e-15
    assert r.scale == pytest.approx(pr["S_scale"], rel=1e-15)
    # robustbase's M step and .vcov.avar1 from the same start (rel.tol 1e-15)
    assert Rv["mstep_conv"][0] == 1
    assert np.max(np.abs(a(r.params) - Rv["mstep_coef"])) <= 1e-14 * np.max(np.abs(Rv["mstep_coef"]))
    # robustbase inverts X^T Psi' X itself (kappa(X)^2 u: 2e-13 on the stack loss, whose Arb
    # reference sd matches to 2e-15)
    assert rel(r.bse, Rv["mstep_se"]) <= 1e-12
    # lmrob itself (its own subsamples; its S refinement stops at refine.tol 1e-7)
    assert np.max(np.abs(a(r.params) - Rv["lmrob_coef"])) <= 1e-6 * np.max(np.abs(Rv["lmrob_coef"]))
    assert rel(r.bse, Rv["lmrob_se"]) <= 1e-5
    assert r.scale == pytest.approx(Rv["lmrob_scale"][0], rel=2e-5)


# ---- MM: breakdown, determinism

@pytest.mark.parametrize("frac", [0.3, 0.4])
def test_mm_breakdown(frac):
    y, X = leverage(n=400, frac=frac, seed=3)
    truth = np.array([0.5, 1.0, 2.0])
    m = R.rlm(y, X, norm="huber")
    mm = R.rlm(y, X, method="MM", rng=sd.random.default_rng(0))
    ls = R.ols(y, X)
    assert np.max(np.abs(a(m.params) - truth)) > 1.0      # M from least squares is pulled away
    assert np.max(np.abs(a(ls.params) - truth)) > 1.0
    assert np.max(np.abs(a(mm.params) - truth)) < 0.15    # MM finds the clean line
    w = a(mm.weights)
    nbad = int(frac * 400)
    assert np.all(w[:nbad] == 0.0) and np.mean(w[nbad:]) > 0.8


def test_mm_rng_determinism():
    y, X = leverage(frac=0.2)
    r1 = R.rlm(y, X, method="MM", rng=sd.random.default_rng(42))
    r2 = R.rlm(y, X, method="MM", rng=sd.random.default_rng(42))
    assert r1.params.tolist() == r2.params.tolist() and r1.bse.tolist() == r2.bse.tolist()
    assert r1.weights.tolist() == r2.weights.tolist() and r1.scale == r2.scale
    # two words per regression from the generator
    g = sd.random.default_rng(42)
    R.rlm(y, X, method="MM", rng=g)
    h = sd.random.default_rng(42)
    h.integers(-2**63, 2**63 - 1, endpoint=True, shape=(2,))
    assert g.state == h.state
    # another seed: other subsets, the same estimate to the S refinement's precision
    r3 = R.rlm(y, X, method="MM", rng=sd.random.default_rng(7))
    assert a(r3.params) == pytest.approx(a(r1.params), abs=1e-10)


@pytest.mark.parametrize("frac", [0.25, 0.4])
def test_mm_large_n(frac):
    """Beyond 2000 rows, robustbase's large-n fast-S (G58): five groups of 400 drawn rows,
    merged, then all the rows. The S estimate is the one the small-n search on all rows finds
    (kernel option large_n=False), and MM still finds the clean line."""
    y, X = leverage(n=6000, frac=frac, seed=4)
    ys, Xs = sd.asarray(y), sd.asarray(X)
    r = R.rlm(ys, Xs, method="MM", rng=sd.random.default_rng(5))
    keys = sd.random.default_rng(5).integers(-2**63, 2**63 - 1, endpoint=True, shape=(2,))
    small = K.rlm_fit("rlm", ys, Xs, keys, True, -1.0, 1, 1, 0.0, 0.0, 4.685061, 2, 50, 1e-8, 500, False)
    assert r.scale == pytest.approx(float(small[6][2]), rel=1e-10)
    assert a(r.params) == pytest.approx(a(small[0]), abs=1e-9)
    assert np.max(np.abs(a(r.params) - [0.5, 1.0, 2.0])) < 0.05
    assert np.all(a(r.weights)[: int(frac * 6000)] == 0.0)
    # batch == loop, determinism
    Y = np.stack([y, y + 0.01])
    b = R.rlm(Y, X, method="MM", rng=sd.random.default_rng(6))
    g = sd.random.default_rng(6)
    for i in range(2):
        o = R.rlm(Y[i], X, method="MM", rng=g)
        assert o.params.tolist() == b.params[i].tolist() and o.bse.tolist() == b.bse[i].tolist()


def test_batch_equals_loop():
    rng = np.random.default_rng(5)
    y, X = contaminated(n=120, seed=5)
    Y = np.stack([y, y + rng.standard_normal(120), y[::-1].copy()])
    for kw in (dict(norm="hampel", scale_est="huber"), dict(norm="bisquare")):
        b = R.rlm(Y, X, **kw)
        for i in range(3):
            o = R.rlm(Y[i], X, **kw)
            assert o.params.tolist() == b.params[i].tolist()
            assert o.bse.tolist() == b.bse[i].tolist()
            assert o.weights.tolist() == b.weights[i].tolist()
            assert o.scale == float(b.scale[i]) and o.n_iter == int(b.n_iter[i])
    b = R.rlm(Y, X, method="MM", rng=sd.random.default_rng(9))
    g = sd.random.default_rng(9)
    for i in range(3):
        o = R.rlm(Y[i], X, method="MM", rng=g)
        assert o.params.tolist() == b.params[i].tolist() and o.bse.tolist() == b.bse[i].tolist()
    assert b.params.shape == (3, 4) and b.cov_params().shape == (3, 4, 4) and b.converged.shape == (3,)


def test_threads_bit_identical():
    y, X = contaminated(n=70000, p=4, seed=2)
    yl, Xl = leverage(n=3000, frac=0.2)
    old = sd.get_num_threads()
    out = []
    try:
        for t in (1, 10):
            sd.set_num_threads(t)
            m = R.rlm(y, X, norm="bisquare")
            h = R.rlm(y, X, scale_est="huber")
            mm = R.rlm(yl, Xl, method="MM", rng=sd.random.default_rng(3))
            out.append([v.tolist() for v in (m.params, m.bse, m.weights, h.params, h.bse, mm.params, mm.bse,
                                             mm.weights)])
    finally:
        sd.set_num_threads(old)
    assert out[0] == out[1]


# ---- inputs, edge cases

def test_nan_policies_and_frames():
    y, X = contaminated(n=80, seed=4)
    yn = y.copy()
    yn[5] = np.nan
    r = R.rlm(yn, X)
    assert math.isnan(r.scale) and np.all(np.isnan(a(r.params))) and r.rank == 0
    assert math.isnan(r.nobs)
    keep = np.arange(80) != 5
    o = R.rlm(yn, X, nan_policy="omit")
    f = R.rlm(y[keep], X[keep])
    assert o.params.tolist() == f.params.tolist() and o.nobs == 79
    om = R.rlm(yn, X, method="MM", nan_policy="omit", rng=sd.random.default_rng(1))
    fm = R.rlm(y[keep], X[keep], method="MM", rng=sd.random.default_rng(1))
    assert om.params.tolist() == fm.params.tolist()
    with pytest.raises(ValueError, match=r"rlm\(\): the input contains NaN \(nan_policy='raise'\)"):
        R.rlm(yn, X, nan_policy="raise")
    with pytest.raises(ValueError, match="nan_policy='omit' takes one regression"):
        R.rlm(np.stack([yn, y]), X, nan_policy="omit")
    yi = y.copy()
    yi[3] = np.inf
    with pytest.raises(ValueError, match=r"rlm\(\): 'y' contains infinity"):
        R.rlm(yi, X)
    Xi = X.copy()
    Xi[3, 1] = -np.inf
    with pytest.raises(ValueError, match=r"rlm\(\): 'X' contains infinity"):
        R.rlm(y, Xi, method="MM", rng=sd.random.default_rng(1))
    df = sd.frame.DataFrame({"a": X[:, 0].tolist(), "b": X[:, 1].tolist(), "c": X[:, 2].tolist()})
    r = R.rlm(sd.frame.Series(y.tolist(), name="y"), df)
    assert r.param_names == ["const", "a", "b", "c"]
    assert r.params.tolist() == R.rlm(y, X).params.tolist()
    assert R.rlm(y, X[:, 0]).param_names == ["const", "x1"]


def test_aliased_and_intercept():
    y, X = contaminated(n=90, seed=6)
    Xd = np.column_stack([X[:, 0], 2.0 * X[:, 0], X[:, 1:]])
    for kw in (dict(), dict(norm="bisquare"), dict(method="MM", rng=sd.random.default_rng(2))):
        r = R.rlm(y, Xd, **kw)
        if "rng" in kw:
            kw = dict(kw, rng=sd.random.default_rng(2))
        f = R.rlm(y, X, **kw)
        p = a(r.params)
        assert math.isnan(p[2]) and math.isnan(a(r.bse)[2])
        assert np.delete(p, 2).tolist() == a(f.params).tolist()
        assert np.delete(a(r.bse), 2).tolist() == a(f.bse).tolist()
        assert r.rank == 4 and r.df_model == 3
        assert r.predict(Xd).tolist() == r.fittedvalues.tolist()
    r = R.rlm(y, X, intercept=False)
    assert r.params.shape == (3,) and r.df_model == 3 and r.param_names == ["x1", "x2", "x3"]
    assert r.predict(X[:5]).tolist() == r.fittedvalues[:5].tolist()


def test_exact_fit_and_iteration_cap():
    rng = np.random.default_rng(8)
    X = rng.integers(-5, 6, (40, 2)).astype(float)
    y = 1.0 + X @ [2.0, -1.0]
    y[:10] += rng.integers(20, 40, 10)   # 30 of 40 rows exactly on the line
    for r in (R.rlm(y, X, norm="bisquare"), R.rlm(y, X, method="MM", rng=sd.random.default_rng(0))):
        assert r.scale == 0.0 and r.converged
        assert r.params.tolist() == [1.0, 2.0, -1.0]
        assert np.all(a(r.bse) == 0.0) and np.all(a(r.pvalues) == 0.0)
        assert np.all(a(r.weights)[:10] == 0.0) and np.all(a(r.weights)[10:] == 1.0)
    y2, X2 = contaminated(seed=9)
    c = R.rlm(y2, X2, norm="bisquare", max_iter=1)
    assert not c.converged and c.n_iter == 1
    assert R.rlm(y2, X2, norm="bisquare", max_iter=2).n_iter == 2


def _check_exact_or_root(r, y, x, line, norm, c):
    """A converged fit is the exact fit of the line (scale 0, covariance 0, weights 1 on the
    line and 0 off it) or a root of the estimating equations sum psi(r_i/s) x_i = 0."""
    p = a(r.params)
    X = np.column_stack([np.ones(len(y)), x])
    if r.scale == 0.0:
        assert np.max(np.abs(p - line)) <= 1e-15 * (1 + np.max(np.abs(line)))
        assert np.all(a(r.bse) == 0.0)
        on = np.abs(y - X @ line) < 1e-9   # the rows off the line are 5 or more off
        assert a(r.weights).tolist() == on.astype(float).tolist()
        return
    z = (y - X @ p) / r.scale
    q = _sm_norm(norm).psi(z) if c is None else sd_psi_bisquare(z, c)
    F = X.T @ q
    assert np.linalg.norm(F) <= 1e-8 * np.sum(np.abs(q)[:, None] * np.abs(X))


def sd_psi_bisquare(z, c):
    u = z / c
    return np.where(np.abs(z) <= c, z * (1 - u * u) ** 2, 0.0)


def test_exact_fit_to_rounding():
    """Most rows on a line whose residuals the IRLS only approaches: the scale falls to the
    rounding level without reaching 0, and the iterations stop there as at an exact fit (the
    rows within (k + 16) u (|y_i| + ||x_i||_1 ||beta||_inf) count as 0). Before, the scaled
    residuals of rounding noise sent a Newton step anywhere: x = 1..10, y = 1 + 2x with
    y[3] = 50, y[7] = -20 gave [1.971, 1.820], scale 0.936, converged=True (statsmodels
    gives [1, 2])."""
    x = np.arange(1.0, 11.0)
    for line in ([1.0, 2.0], [0.1, 0.3]):     # exact in double, and rounded
        for out in ({3: 50.0, 7: -20.0}, {3: 50.0}):
            y = line[0] + line[1] * x
            for i, v in out.items():
                y[i] = v
            kws = [dict(norm=nm) for nm in ("huber", "hampel", "bisquare")]
            kws += [dict(method="MM", rng=sd.random.default_rng(s)) for s in range(3)]
            kws += [dict(max_iter=m) for m in (36, 38, 40)]
            if len(out) == 1:   # with two of ten off the line proposal 2 has a root away from it
                kws += [dict(norm=nm, scale_est="huber") for nm in ("huber", "hampel", "bisquare")]
            for kw in kws:
                r = R.rlm(y, x, **kw)
                assert r.converged and r.scale == 0.0, kw
                assert np.max(np.abs(a(r.params) - line)) <= 1e-15, kw
                assert np.all(a(r.bse) == 0.0)
                assert a(r.weights).tolist() == [0.0 if i in out else 1.0 for i in range(10)]
                if kw.get("method") == "MM":
                    assert r.n_iter == 0
    # lines with fewer than half the rows off: every converged fit is the line or a root
    rng = np.random.default_rng(11)
    for trial in range(60):
        n = int(rng.choice([8, 11, 15, 20, 31]))
        x = [np.arange(1.0, n + 1), rng.integers(-10, 11, n).astype(float), np.round(rng.normal(size=n), 3)][trial % 3]
        line = np.array([1.0, 2.0] if trial % 2 else [0.1, 0.3])
        y = line[0] + line[1] * x
        m = int(rng.integers(1, max(2, (n - 1) // 2)))
        idx = rng.choice(n, m, replace=False)
        y[idx] += rng.choice([-1, 1], m) * rng.uniform(5, 60, m)
        y_line = line[0] + line[1] * x
        for norm in ("huber", "hampel", "bisquare"):
            for est in ("mad", "huber"):
                r = R.rlm(y, x, norm=norm, scale_est=est)
                if r.converged:
                    _check_exact_or_root(r, y, x, line, norm, None)
        r = R.rlm(y, x, method="MM", rng=sd.random.default_rng(trial))
        assert r.converged
        if r.scale == 0.0:
            assert np.max(np.abs(a(r.params) - line)) <= 1e-15 * 3
            assert a(r.weights).tolist() == (np.abs(y - y_line) < 1e-9).astype(float).tolist()
        else:
            _check_exact_or_root(r, y, x, line, "bisquare", 4.685061)


def test_tuning_constants():
    import statsmodels.api as sm
    from statsmodels.robust import norms

    y, X = contaminated(seed=10)
    for norm, c, M in (("huber", 2.0, norms.HuberT(t=2.0)), ("bisquare", 3.0, norms.TukeyBiweight(c=3.0)),
                       ("hampel", (1.5, 3.5, 8.0), norms.Hampel(1.5, 3.5, 8.0))):
        r = R.rlm(y, X, norm=norm, c=c)
        s = sm.RLM(y, sm.add_constant(X), M=M).fit(tol=1e-15, maxiter=1000)
        assert np.max(np.abs(a(r.params) - s.params)) <= 1e-13 * np.max(np.abs(s.params))
        assert rel(r.bse, s.bse) <= 1e-13
    d = R.rlm(y, X, norm="hampel")
    assert d.params.tolist() == R.rlm(y, X, norm="hampel", c=[2, 4, 8]).params.tolist()
    assert R.rlm(y, X).params.tolist() == R.rlm(y, X, c=1.345).params.tolist()
    m1 = R.rlm(*leverage(frac=0.2), method="MM", rng=sd.random.default_rng(0))
    m2 = R.rlm(*leverage(frac=0.2), method="MM", rng=sd.random.default_rng(0), c=4.685061)
    assert m1.params.tolist() == m2.params.tolist()


def _err(f):
    try:
        f()
    except (TypeError, ValueError) as e:
        return str(e)
    return ""


def test_errors():
    y, X = contaminated(n=30, seed=0)
    g = sd.random.default_rng(0)
    cases = [
        (TypeError, "rlm(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not NoneType",
         lambda: R.rlm(y, X, method="MM")),
        (TypeError, "rlm(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not int",
         lambda: R.rlm(y, X, method="MM", rng=3)),
        (ValueError, "rlm(): 'rng' is only used with method='MM'", lambda: R.rlm(y, X, rng=g)),
        (ValueError, "rlm(): 'method' must be 'M' or 'MM', not 'S'", lambda: R.rlm(y, X, method="S")),
        (ValueError, "rlm(): 'norm' must be 'huber', 'bisquare' or 'hampel', not 'andrew'",
         lambda: R.rlm(y, X, norm="andrew")),
        (ValueError, "rlm(): method='MM' takes the bisquare norm, not 'huber'",
         lambda: R.rlm(y, X, method="MM", norm="huber", rng=g)),
        (ValueError, "rlm(): 'scale_est' is only used with method='M'",
         lambda: R.rlm(y, X, method="MM", scale_est="mad", rng=g)),
        (ValueError, "rlm(): 'scale_est' must be 'mad' or 'huber', not 'iqr'", lambda: R.rlm(y, X, scale_est="iqr")),
        (ValueError, "rlm(): 'c' must be None or a finite number > 0 for the huber norm", lambda: R.rlm(y, X, c=-1.0)),
        (ValueError, "rlm(): 'c' must be None or a finite number > 0 for the bisquare norm",
         lambda: R.rlm(y, X, norm="bisquare", c=math.inf)),
        (TypeError, "rlm(): 'c' must be None or a finite number > 0 for the huber norm, not str",
         lambda: R.rlm(y, X, c="a")),
        (TypeError, "rlm(): 'c' must be None or a finite number > 0 for the huber norm, not tuple",
         lambda: R.rlm(y, X, c=(1, 2))),
        (ValueError, "rlm(): 'c' must be three numbers 0 < a <= b < c for the hampel norm",
         lambda: R.rlm(y, X, norm="hampel", c=(2, 8, 4))),
        (TypeError, "rlm(): 'c' must be three numbers 0 < a <= b < c for the hampel norm, not float",
         lambda: R.rlm(y, X, norm="hampel", c=2.0)),
        (ValueError, "rlm(): 'max_iter' must be >= 1", lambda: R.rlm(y, X, max_iter=0)),
        (ValueError, "rlm(): 'tol' must be a finite number > 0", lambda: R.rlm(y, X, tol=0.0)),
        (ValueError, "rlm(): 'X' must have 30 rows like 'y', not 29", lambda: R.rlm(y, X[:29])),
        (ValueError, "rlm(): the model has no parameters", lambda: R.rlm(y, np.zeros((30, 0)), intercept=False)),
        (TypeError, "rlm(): 'X' must be real, not a complex128 array", lambda: R.rlm(y, sd.asarray(X) * 1j)),
        (ValueError, "rlm(): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'drop'", lambda: R.rlm(y, X, nan_policy="drop")),
    ]
    for exc, msg, f in cases:
        with pytest.raises(exc) as e:
            f()
        assert msg in str(e.value), (msg, str(e.value))


# ---- Theil-Sen and Siegel

def _same(u, v):
    return all((float(p) == float(q)) or (math.isnan(p) and math.isnan(q)) for p, q in zip(u, v))


def _points(n, seed, kind):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = 2.0 * x + rng.standard_normal(n)
    if kind == "ties":
        x[: n // 4] = x[n // 4]
        y[n // 3: n // 2] = y[0]
    elif kind == "grid":
        x = np.round(x, 1)
        y = np.round(y, 1)
    elif kind == "flat":
        y[: n // 2] = 1.0
    return y, x


@pytest.mark.parametrize("n", [2, 3, 4, 11, 60, 301, 1000])
@pytest.mark.parametrize("kind", ["plain", "ties", "grid", "flat"])
def test_slopes_against_scipy(n, kind):
    y, x = _points(n, n, kind)
    for method in ("separate", "joint"):
        for alpha in (0.95, 0.1, 0.5, 0.999):
            s = st.theilslopes(y, x, alpha=alpha, method=method)
            o = R.theilslopes(y, x, alpha=alpha, method=method)
            assert _same(o, [v + 0.0 for v in s]), (method, alpha, o, s)
    for method in ("hierarchical", "separate"):
        s = st.siegelslopes(y, x, method=method)
        o = R.siegelslopes(y, x, method=method)
        assert _same(o, [v + 0.0 for v in s]), (method, o, s)
    assert _same(R.theilslopes(y), st.theilslopes(y)) and _same(R.siegelslopes(y), st.siegelslopes(y))


@pytest.mark.parametrize("n,direct,collect", [(200, 0, 10**9), (300, 0, 100), (700, 0, 40), (3000, 0, 5000),
                                              (2500, 10**5, 3000)])
def test_theil_passes(n, direct, collect):
    """The pass algorithm (beyond 2^23 slopes) on small problems: forced by its limits."""
    z = float(sd.special.ndtri(0.025))
    for kind in ("plain", "ties", "grid", "flat"):
        y, x = _points(n, n + 1, kind)
        s = st.theilslopes(y, x)
        v = K.slopes("t", sd.asarray(y), sd.asarray(x), 0, False, z, direct, collect).tolist()
        assert _same(v, [u + 0.0 for u in s]), (kind, v, s)


def _frac_median(vals):
    vals = sorted(vals)
    m = len(vals)
    return vals[m // 2] if m % 2 else (vals[m // 2 - 1] + vals[m // 2]) / 2


def test_slopes_exact_rational():
    """The medians of the exact rational slopes (of the double inputs), rounded once: the
    double slopes' medians agree with them to a few units in the last place."""
    rng = np.random.default_rng(12)
    for n in (5, 8, 13, 20):
        x = rng.standard_normal(n)
        y = 3.0 * x + rng.standard_normal(n)
        X = [Fraction(v) for v in x]
        Y = [Fraction(v) for v in y]
        sl = [(Y[i] - Y[j]) / (X[i] - X[j]) for i, j in itertools.combinations(range(n), 2)]
        exact = float(_frac_median(sl))
        got = R.theilslopes(y, x).slope
        assert abs(got - exact) <= 4 * np.spacing(abs(exact))
        rows = [_frac_median([(Y[i] - Y[j]) / (X[i] - X[j]) for j in range(n) if j != i]) for i in range(n)]
        exact = float(_frac_median(rows))
        assert abs(R.siegelslopes(y, x).slope - exact) <= 4 * np.spacing(abs(exact))
    # small integers: the slopes are exact in double, so is the median
    x = np.array([0.0, 1, 2, 4, 7, 8])
    y = np.array([1.0, 3, 2, 9, 4, 12])
    X, Y = [Fraction(v) for v in x], [Fraction(v) for v in y]
    sl = [(Y[i] - Y[j]) / (X[i] - X[j]) for i, j in itertools.combinations(range(6), 2)]
    assert R.theilslopes(y, x).slope == float(_frac_median(sl))


def test_slopes_batch_nan_errors():
    y, x = _points(50, 3, "plain")
    Y = np.stack([y, y[::-1].copy(), 2 * y])
    b = R.theilslopes(Y, x)
    for i in range(3):
        o = R.theilslopes(Y[i], x)
        assert _same([float(v[i]) for v in b], o)
    b = R.siegelslopes(Y, x, method="separate")
    assert _same([float(v[1]) for v in b], R.siegelslopes(Y[1], x, method="separate"))
    yn = y.copy()
    yn[4] = np.nan
    assert all(math.isnan(v) for v in R.theilslopes(yn, x))
    assert all(math.isnan(v) for v in R.siegelslopes(yn, x))
    keep = ~np.isnan(yn)
    assert _same(R.theilslopes(yn, x, nan_policy="omit"), R.theilslopes(y[keep], x[keep]))
    assert _same(R.theilslopes(yn, x, nan_policy="omit"), st.theilslopes(yn, x, nan_policy="omit"))
    assert _same(R.siegelslopes(yn, x, nan_policy="omit"), st.siegelslopes(yn, x, nan_policy="omit"))
    with pytest.raises(ValueError, match=r"theilslopes\(\): the input contains NaN"):
        R.theilslopes(yn, x, nan_policy="raise")
    with pytest.raises(ValueError, match=r"siegelslopes\(\): 'y' contains infinity"):
        R.siegelslopes(np.r_[y[:-1], np.inf], x)
    with pytest.raises(ValueError, match="length at least 2"):
        R.theilslopes([1.0], [2.0])
    with pytest.raises(ValueError, match=r"theilslopes\(\): 'method' must be 'joint' or 'separate', not 'x'"):
        R.theilslopes(y, x, method="x")
    with pytest.raises(ValueError, match=r"siegelslopes\(\): 'method' must be 'hierarchical' or 'separate'"):
        R.siegelslopes(y, x, method="joint")
    with pytest.raises(ValueError, match=r"theilslopes\(\): 'alpha' must be between 0 and 1"):
        R.theilslopes(y, x, alpha=1.0)
    with pytest.raises(ValueError, match=r"'x' must have 50 values like 'y', not 49"):
        R.theilslopes(y, x[:49])
    # all x equal: no slope (scipy: NaN)
    assert all(math.isnan(v) for v in R.theilslopes(y[:5], np.ones(5)))
    assert all(math.isnan(v) for v in R.siegelslopes(y[:5], np.ones(5)))
    r = R.theilslopes(y, x)
    assert isinstance(r, sd.regression.TheilslopesResult) and isinstance(r.slope, float)
    assert r._fields == ("slope", "intercept", "low_slope", "high_slope")
    assert R.siegelslopes(y, x)._fields == ("slope", "intercept")


def test_slopes_threads_bit_identical():
    y, x = _points(5000, 4, "grid")
    old = sd.get_num_threads()
    out = []
    try:
        for t in (1, 10):
            sd.set_num_threads(t)
            out.append((tuple(R.theilslopes(y, x)), tuple(R.siegelslopes(y, x, method="separate"))))
    finally:
        sd.set_num_threads(old)
    assert out[0] == out[1]
    assert _same(out[0][0], st.theilslopes(y, x))


def test_mscale_gradient_scale_free():
    """The S polish's Newton step differentiates sum rho~(r_i/s) = T: grad s = -sum psi x /
    sum psi z, no factor s (it had one: wrong for s != 1, though near the fixed point, where
    the sums vanish, the step hardly saw it). MM is equivariant in y: the S iterations and the
    scaled results are the same for y, 1000 y and y / 1000."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((300, 2))
    y = X @ [1.0, -2.0] + 0.5 + rng.standard_normal(300)
    y[:60] += 10
    fits = [R.rlm(y * s, X, method="MM", rng=sd.random.default_rng(0)) for s in (1.0, 1000.0, 1e-3)]
    its = [(f.n_iter, f.converged) for f in fits]
    assert its[0] == its[1] == its[2] and its[0][1]
    for f, s in zip(fits[1:], (1000.0, 1e-3)):
        assert np.max(np.abs(a(f.params) / s - a(fits[0].params))) <= 1e-13
        assert abs(f.scale / s - fits[0].scale) <= 1e-13 * fits[0].scale


def test_slopes_omit_keeps_positions():
    """G83 ([scipy differs]): with x=None, nan_policy='omit' drops the point but the others keep
    their positions 0, 1, 3, 4 (scipy renumbers them 0 .. 3: slope 1.4167)."""
    y = [1.0, 2.0, math.nan, 4.0, 5.0]
    for f in (R.theilslopes, R.siegelslopes):
        r = f(y, nan_policy="omit")
        assert r.slope == 1.0 and r.intercept == 1.0
        assert f(y, [0.0, 1, 2, 3, 4], nan_policy="omit").slope == 1.0
