"""sd.regression.nls (docs/api/regression.md §7, docs/design/regression.md §3.7): the NIST
StRD nonlinear problems (certified values, and Arb references of the same double data),
R's nls on fixed data (tests/regression/ref/nls_r.json, tools/gen_nls_r.py), bounds against
scipy's trust region reflective method, weights, NaN, non-finite trial points, the user's
Jacobian, aliased parameters, the stopping and the errors."""

import json
import math
import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy.optimize import curve_fit

import stardust as sd
from stardust.regression import _nls

from . import nist_nls as N

ROOT = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import regref  # noqa: E402

R_REF = json.loads((pathlib.Path(__file__).parent / "ref" / "nls_r.json").read_text())

# fits from NIST's start 1 that do not reach the certified minimum: (problem, jac given)
NOT_REACHED = {("BoxBOD", False): "another stationary point (b2 large, the Jacobian singular)",
               ("Bennett5", False): "max_nfev", ("Bennett5", True): "max_nfev", ("MGH09", True): "max_nfev"}

_ARB = {}


def arb_ref(name):
    if name not in _ARB:
        d = N.load(name)
        xn, yn = N.data(d, np)
        f, jac = N.model(name, regref.ARB)
        _ARB[name] = regref.nls(f, jac, xn.tolist(), yn.tolist(), [float(v) for v in d["params"]])
    return _ARB[name]


def rel(a, b):
    return max(abs(u - v) / abs(v) for u, v in zip(a, b))


def decay_data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 5.0, n)
    y = 3.0 * np.exp(-0.7 * x) + 0.5 + 0.03 * rng.standard_normal(n)
    return x, y


def decay(x, a, b, c):
    return a * sd.exp(-b * x) + c


def decay_np(x, a, b, c):
    return a * np.exp(-b * x) + c


def decay_jac(x, a, b, c):
    e = sd.exp(-b * x)
    return sd.stack([e, -a * x * e, x * 0.0 + 1.0], axis=1)


# ---- NIST StRD

@pytest.mark.parametrize("name", N.NAMES)
@pytest.mark.parametrize("start", ["start1", "start2"])
@pytest.mark.parametrize("use_jac", [False, True], ids=["differences", "jac"])
def test_nist(name, start, use_jac):
    d = N.load(name)
    x, y = N.data(d, sd)
    f, jac = N.model(name, sd)
    r = sd.regression.nls(f, x, y, p0=[float(v) for v in d[start]], jac=jac if use_jac else None)
    if start == "start1" and (name, use_jac) in NOT_REACHED:
        assert not r.converged or not all(math.isfinite(v) for v in r.bse.tolist())
        return
    assert r.converged
    params, bse = r.params.tolist(), r.bse.tolist()
    # the certified values (11 digits, of the decimal data)
    assert min(N.lre(v, c) for v, c in zip(params, d["params"])) >= (10.0 if use_jac else 9.0)
    if name != "Lanczos1":
        assert min(N.lre(v, c) for v, c in zip(bse, d["bse"])) >= (10.0 if use_jac else 8.5)
    # Arb, the same double data: the minimizer to rounding (with jac) or to the differences'
    # accuracy; Lanczos1's residuals are at the rounding of f, so its bse has ~4 digits
    ref = arb_ref(name)
    assert rel(params, ref["params"]) <= (1e-12 if use_jac else 1e-9)
    assert rel(bse, ref["bse"]) <= (1e-3 if name == "Lanczos1" else (1e-10 if use_jac else 3e-9))
    # the sum of squares: f's rounding moves it by about 2 u ||f|| ||r||
    ynorm = math.sqrt(sum(v * v for v in y.tolist()))
    assert abs(r.ssr - ref["ssr"]) <= (1e-12 if use_jac else 1e-9) * ref["ssr"] + 8 * 2.0 ** -52 * ynorm * math.sqrt(
        ref["ssr"])
    assert r.nobs == len(d["y"]) and r.df_resid == len(d["y"]) - len(params) and r.rank == len(params)


def test_nist_from_the_certified_values_well_conditioned_to_rounding():
    for name in ("Misra1a", "Chwirut2", "DanWood", "Gauss1", "ENSO", "Eckerle4", "Rat43"):
        d = N.load(name)
        x, y = N.data(d, sd)
        f, jac = N.model(name, sd)
        r = sd.regression.nls(f, x, y, p0=[float(v) for v in d["params"]], jac=jac)
        assert rel(r.params.tolist(), arb_ref(name)["params"]) <= 1e-14, name


# ---- R's nls

R_MODELS = {
    "decay": decay,
    "puromycin": lambda x, Vm, K: Vm * x / (K + x),
    "twod": lambda x, a, b, c: a * x[:, 0] ** b * sd.exp(c * x[:, 1]),
    "logistic": lambda x, Asym, xmid, scal: Asym / (1 + sd.exp((xmid - x) / scal)),
}


@pytest.mark.parametrize("name", [k for k in R_REF if not k.startswith("_")])
def test_against_r(name):
    p = R_REF[name]
    f = R_MODELS[name.split("_")[0]]
    kw = {}
    if p["w"] is not None:
        kw["weights"] = p["w"]
    if name == "decay_bounded":
        kw["bounds"] = ([0, 0, 0.6], [10, 10, 10])
    r = sd.regression.nls(f, p["x"], p["y"], p0=p["start"], **kw)
    R = p["R"]
    # R stops at its relative-offset test (tol 1e-10, 1e-8 for twod; port's own for the bounds)
    tol = {"twod": 1e-9, "decay_bounded": 1e-7}.get(name, 3e-11)
    assert r.converged
    assert r.param_names == p["names"]
    assert rel(r.params.tolist(), R["coef"]) <= tol
    assert rel(r.bse.tolist(), R["se"]) <= 10 * tol
    assert rel(r.tvalues.tolist(), R["t"]) <= 10 * tol
    assert rel(r.pvalues.tolist(), R["p"]) <= 1e4 * tol
    assert abs(math.sqrt(r.scale) - R["sigma"][0]) <= 1e-13 * R["sigma"][0]
    assert abs(r.ssr - R["deviance"][0]) <= 1e-13 * R["deviance"][0]
    assert [r.df_model, r.df_resid] == R["df"]
    assert np.allclose(r.fittedvalues.tolist(), R["fitted"], rtol=0, atol=10 * tol * max(map(abs, R["fitted"])))
    assert np.allclose(r.resid.tolist(), R["resid"], rtol=0, atol=10 * tol * max(map(abs, R["fitted"])))
    # R's logLik.nls is ours (weighted: + 1/2 sum log w over the rows used, as wls); its AIC and BIC
    # count sigma as a parameter (one more than rank, as R's AIC of lm)
    assert abs(r.llf - R["loglik"][0]) <= 1e-12 * max(1.0, abs(R["loglik"][0]))
    assert abs(r.aic + 2 - R["aic"][0]) <= 1e-12 * max(1.0, abs(R["aic"][0]))
    assert abs(r.bic + math.log(r.nobs) - R["bic"][0]) <= 1e-12 * max(1.0, abs(R["bic"][0]))


def test_llf_aic_bic_are_the_linear_models():
    # a model linear in its parameters: nls's llf, aic, bic are wls's (zero weights excluded)
    rng = np.random.default_rng(6)
    xn = rng.uniform(0, 3, 50)
    yn = 1.0 + 2.0 * xn + 0.3 * rng.standard_normal(50)
    w = rng.uniform(0.5, 2.0, 50)
    w[[4, 20]] = 0.0
    f = lambda x, a, b: a + b * x
    for kw, lin in (({}, sd.regression.ols(yn, xn)), ({"weights": w}, sd.regression.wls(yn, xn, weights=w))):
        r = sd.regression.nls(f, xn, yn, p0=[0.0, 1.0], **kw)
        for name in ("llf", "aic", "bic", "ssr", "scale", "nobs", "df_resid"):
            assert abs(getattr(r, name) - getattr(lin, name)) <= 1e-13 * abs(getattr(lin, name)), name


# ---- bounds (trust region reflective) against scipy

@pytest.mark.parametrize("bounds, p0", [(([0, 0, 0.5], [10, 10, 10]), [1.0, 1.0, 1.0]),
                                        (([0, 0.9, -1], [2.8, 10, 10]), [1.0, 1.0, 1.0]),
                                        (([-np.inf, 0, -np.inf], [np.inf, 0.5, np.inf]), [1.0, 0.4, 1.0])])
def test_bounds_against_scipy(bounds, p0):
    rng = np.random.default_rng(1)
    xn = np.linspace(0, 5, 40)
    yn = 3.0 * np.exp(-0.7 * xn) + 0.2 - 0.05 * xn + 0.03 * rng.standard_normal(40)
    r = sd.regression.nls(decay, xn, yn, p0=p0, bounds=bounds)
    assert r.method == "trf" and r.converged
    popt, pcov = curve_fit(decay_np, xn, yn, p0=p0, bounds=bounds)
    ssr_scipy = float(np.sum((decay_np(xn, *popt) - yn) ** 2))
    assert r.ssr <= ssr_scipy * (1 + 1e-12)
    assert rel(r.params.tolist(), popt) <= 2e-5
    assert rel(r.bse.tolist(), np.sqrt(np.diag(pcov))) <= 2e-5
    lb, ub = bounds
    params = r.params.tolist()
    assert all(a <= v <= b for v, a, b in zip(params, lb, ub))
    # the active bounds are hit exactly
    assert r._active
    for j in r._active:
        assert params[j] in (lb[j], ub[j])
    assert "At a bound" in r.summary()


def test_inactive_bounds_give_the_unconstrained_minimum():
    xn, yn = decay_data()
    free = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0])
    boxed = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], bounds=([-10, -10, -10], [10, 10, 10]))
    trf = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], method="trf")
    assert boxed.method == "trf" and boxed._active == []
    for r in (boxed, trf):
        assert rel(r.params.tolist(), free.params.tolist()) <= 1e-13
        assert rel(r.bse.tolist(), free.bse.tolist()) <= 1e-11


def test_scalar_bounds_and_p0_on_a_bound():
    xn, yn = decay_data()
    r = sd.regression.nls(decay, xn, yn, p0=[1.0, 0.0, 1.0], bounds=(0, 10))
    assert r.converged and rel(r.params.tolist(), sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0]).params.tolist()) <= 1e-12


# ---- weights (G72)

def test_unit_weights_are_the_unweighted_fit():
    xn, yn = decay_data()
    a = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0])
    b = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=np.ones(len(yn)))
    assert a.params.tolist() == b.params.tolist() and a.bse.tolist() == b.bse.tolist()


def test_zero_weights_are_the_rows_left_out():
    xn, yn = decay_data()
    rng = np.random.default_rng(4)
    w = rng.uniform(0.5, 2.0, len(yn))
    w[[2, 11, 30]] = 0.0
    keep = w > 0
    a = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=w)
    b = sd.regression.nls(decay, xn[keep], yn[keep], p0=[1.0, 1.0, 1.0], weights=w[keep])
    assert rel(a.params.tolist(), b.params.tolist()) <= 1e-15
    assert rel(a.bse.tolist(), b.bse.tolist()) <= 1e-13
    assert a.nobs == keep.sum() and a.df_resid == keep.sum() - 3
    assert a.fittedvalues.shape == (len(yn),) and a.resid.shape == (len(yn),)
    assert np.allclose(np.asarray(a.fittedvalues.tolist()), decay_np(xn, *a.params.tolist()), rtol=1e-15, atol=0)
    # a model that is not finite in a zero weight row
    f = lambda x, a, b, c: decay(x, a, b, c) + sd.where(x == x[2], math.inf, 0.0)
    c = sd.regression.nls(f, xn, yn, p0=[1.0, 1.0, 1.0], weights=w)
    assert rel(c.params.tolist(), a.params.tolist()) <= 1e-15


def test_weights_are_scipy_sigma():
    xn, yn = decay_data()
    sigma = np.linspace(0.5, 2.0, len(yn))
    r = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=1 / sigma ** 2)
    popt, pcov = curve_fit(decay_np, xn, yn, p0=[1.0, 1.0, 1.0], sigma=sigma)
    assert rel(r.params.tolist(), popt) <= 1e-6
    assert rel(r.bse.tolist(), np.sqrt(np.diag(pcov))) <= 1e-6


def test_weight_errors():
    xn, yn = decay_data()
    w = np.ones(len(yn))
    with pytest.raises(ValueError, match=r"^nls\(\): 'weights' must be >= 0$"):
        sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=np.where(np.arange(len(w)) == 3, -1.0, w))
    with pytest.raises(ValueError, match=r"^nls\(\): 'weights' contains infinity$"):
        sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=np.where(np.arange(len(w)) == 3, np.inf, w))
    with pytest.raises(ValueError, match=r"^nls\(\): 'weights' must have 40 values like 'y', not shape \(39,\)$"):
        sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], weights=w[:-1])


# ---- NaN (G75)

def test_nan_policy():
    xn, yn = decay_data()
    y2 = yn.copy()
    y2[5] = np.nan
    x2 = np.column_stack([xn, np.zeros_like(xn)])
    x2[9, 1] = np.nan
    calls = []

    def f(x, a, b, c):
        calls.append(1)
        return decay(x, a, b, c)
    r = sd.regression.nls(f, xn, y2, p0=[1.0, 1.0, 1.0])
    assert not calls
    assert all(math.isnan(v) for v in r.params.tolist() + r.bse.tolist() + r.fittedvalues.tolist())
    assert math.isnan(r.nobs) and math.isnan(r.ssr) and r.rank == 0 and not r.converged and r.nfev == 0
    assert math.isnan(r.llf) and math.isnan(r.aic) and math.isnan(r.bic)
    assert all(math.isnan(v) for v in r.predict(xn[:4]).tolist())
    o = sd.regression.nls(decay, xn, y2, p0=[1.0, 1.0, 1.0], nan_policy="omit")
    keep = ~np.isnan(y2)
    s = sd.regression.nls(decay, xn[keep], yn[keep], p0=[1.0, 1.0, 1.0])
    assert o.params.tolist() == s.params.tolist() and o.nobs == 39 and o.fittedvalues.shape == (39,)
    f2 = lambda x, a, b, c: decay(x[:, 0], a, b, c)
    o2 = sd.regression.nls(f2, x2, yn, p0=[1.0, 1.0, 1.0], nan_policy="omit")
    keep = ~np.isnan(x2[:, 1])
    s2 = sd.regression.nls(f2, x2[keep], yn[keep], p0=[1.0, 1.0, 1.0])
    assert o2.params.tolist() == s2.params.tolist()
    with pytest.raises(ValueError, match=r"^nls\(\): the input contains NaN \(nan_policy='raise'\)$"):
        sd.regression.nls(decay, xn, y2, p0=[1.0, 1.0, 1.0], nan_policy="raise")
    with pytest.raises(ValueError, match=r"^nls\(\): 'y' contains infinity$"):
        sd.regression.nls(decay, xn, np.where(np.arange(40) == 1, np.inf, yn), p0=[1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match=r"^nls\(\): 'x' contains infinity$"):
        sd.regression.nls(decay, np.where(np.arange(40) == 1, np.inf, xn), yn, p0=[1.0, 1.0, 1.0])


# ---- non-finite values (G74)

def test_non_finite_trial_points_are_rejected():
    # log(b - x) is NaN for b < 4: steps from b = 4.05 that cross it are rejected
    rng = np.random.default_rng(3)
    xn = np.linspace(0.0, 4.0, 30)
    yn = 2.0 * np.log(5.0 - xn) + 0.02 * rng.standard_normal(30)
    f = lambda x, a, b: a * sd.log(b - x)
    fn = lambda x, a, b: a * np.log(b - x)
    good = sd.regression.nls(f, xn, yn, p0=[2.0, 5.0])
    for method in ("lm", "trf"):
        r = sd.regression.nls(f, xn, yn, p0=[0.5, 4.05], method=method)
        assert r.converged
        assert rel(r.params.tolist(), good.params.tolist()) <= 1e-12
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, _ = curve_fit(fn, xn, yn, p0=[2.0, 5.0])
    assert rel(good.params.tolist(), popt) <= 1e-6


def test_non_finite_at_p0_is_an_error():
    xn, yn = decay_data()
    with pytest.raises(ValueError, match=r"^nls\(\): 'f' is not finite at 'p0'"):
        sd.regression.nls(lambda x, a, b: a * sd.log(b - x), xn, yn, p0=[1.0, 1.0])
    with pytest.raises(ValueError, match=r"^nls\(\): 'jac' is not finite"):
        sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], jac=lambda x, a, b, c: decay_jac(x, a, b, c) / 0.0)


# ---- the Jacobian (G70)

def test_jac_and_differences_agree():
    xn, yn = decay_data()
    a = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0])
    b = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], jac=decay_jac)
    assert rel(a.params.tolist(), b.params.tolist()) <= 1e-12
    assert rel(a.bse.tolist(), b.bse.tolist()) <= 1e-10
    assert b.njev >= b.n_iter and b.nfev < a.nfev
    # a sequence and a 1-parameter (n,) Jacobian are taken
    c = sd.regression.nls(lambda x, a: a * x, xn, 2 * xn + 0.1 * np.sin(xn), p0=[1.0], jac=lambda x, a: x)
    assert abs(c.params.tolist()[0] - np.dot(xn, 2 * xn + 0.1 * np.sin(xn)) / np.dot(xn, xn)) <= 1e-15
    with pytest.raises(ValueError, match=r"^nls\(\): 'jac' must return shape \(40, 3\), not \(40, 2\)$"):
        sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0], jac=lambda x, a, b, c: decay_jac(x, a, b, c)[:, :2])


def test_aliased_parameters():
    xn = np.linspace(0.0, 1.0, 20)
    yn = 2 * xn + 0.01 * np.sin(7 * xn)
    f = lambda x, a, b: a * b * x
    jac = lambda x, a, b: sd.stack([b * x, a * x], axis=1)
    for kw in ({"jac": jac}, {}):
        r = sd.regression.nls(f, xn, yn, p0=[1.0, 1.0], **kw)
        assert r.rank == 1 and r._raw[5].tolist() == [True, False] and r.df_resid == 19 and r.df_model == 1
        assert all(math.isfinite(v) for v in r.params.tolist())
        bse = r.bse.tolist()
        assert math.isfinite(bse[0]) and math.isnan(bse[1]) and math.isnan(r.pvalues.tolist()[1])
        assert abs(r.params.tolist()[0] * r.params.tolist()[1] - np.dot(xn, yn) / np.dot(xn, xn)) <= 1e-14
        assert "Aliased" in r.summary()


# ---- the stopping (G73, G32)

def test_max_nfev_returns_the_last_iterate():
    d = N.load("MGH09")
    x, y = N.data(d, sd)
    f, _ = N.model("MGH09", sd)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = sd.regression.nls(f, x, y, p0=[float(v) for v in d["start1"]], max_nfev=50)
    assert not r.converged and r._count_stat(_nls._S_NPOLISH) == 0
    assert all(math.isfinite(v) for v in r.params.tolist())
    assert "max_nfev" in r.summary()


def test_the_polish_reaches_rounding_from_loose_tolerances():
    d = N.load("ENSO")
    x, y = N.data(d, sd)
    f, jac = N.model("ENSO", sd)
    r = sd.regression.nls(f, x, y, p0=[float(v) for v in d["start2"]], jac=jac, ftol=1e-4, xtol=1e-4, gtol=1e-4)
    assert r.converged and rel(r.params.tolist(), arb_ref("ENSO")["params"]) <= 1e-13


def test_same_bits_with_1_and_10_threads():
    rng = np.random.default_rng(8)
    xn = rng.uniform(0, 5, 200_000)
    yn = 3.0 * np.exp(-0.7 * xn) + 0.5 + 0.03 * rng.standard_normal(len(xn))
    before = sd.get_num_threads()
    out = []
    try:
        for th in (1, 10):
            sd.set_num_threads(th)
            r = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0])
            out.append((r.params.tolist(), r.bse.tolist(), r.nfev))
    finally:
        sd.set_num_threads(before)
    assert out[0] == out[1]


# ---- the result

def test_result_fields():
    xn, yn = decay_data()
    r = sd.regression.nls(decay, xn, yn, p0=[1.0, 1.0, 1.0])
    assert isinstance(r, sd.regression.NLSResult)
    assert r.param_names == ["a", "b", "c"] and r.method == "lm" and r.cov_type == "nonrobust"
    assert type(r.nobs) is float and type(r.rank) is int and type(r.converged) is bool
    assert type(r.n_iter) is int and type(r.nfev) is int and type(r.njev) is int
    assert r.params.shape == (3,) and r.cov_params().shape == (3, 3) and r.conf_int().shape == (3, 2)
    assert abs(r.scale - r.ssr / 37) <= 1e-16 * r.scale and r.df_model == 3
    assert abs(r.llf + 20 * (math.log(2 * math.pi) + math.log(r.ssr / 40) + 1)) <= 1e-13 * abs(r.llf)
    assert r.aic == -2 * r.llf + 6 and r.bic == -2 * r.llf + math.log(40) * 3
    C = np.asarray(r.cov_params().tolist())
    assert np.array_equal(C, C.T)
    bse = np.asarray(r.bse.tolist())
    assert np.allclose(np.sqrt(np.diag(C)), bse, rtol=1e-15)
    t = float(sd.stats.StudentT(df=37).iccdf(0.025))
    ci = np.asarray(r.conf_int().tolist())
    assert np.allclose(ci[:, 1] - ci[:, 0], 2 * t * bse, rtol=1e-14)
    p = np.asarray(r.params.tolist())
    assert np.allclose(np.asarray(r.pvalues.tolist()), 2 * np.asarray(
        sd.stats.StudentT(df=37).ccdf(sd.asarray(np.abs(p / bse))).tolist()), rtol=1e-15)
    assert np.allclose(np.asarray(r.resid.tolist()), yn - decay_np(xn, *p), rtol=0, atol=1e-15)
    xs = np.array([0.5, 1.5, 7.0])
    assert r.predict(xs).tolist() == decay(sd.asarray(xs), *p.tolist()).tolist()
    text = r.summary()
    assert "Nonlinear Least Squares Results" in text and "Levenberg-Marquardt" in text and "5-point" in text
    assert repr(r).startswith("NLSResult(method='lm'")
    with pytest.raises(ValueError, match=r"^predict\(\): 'x_new' must be 1-dimensional like the fit's 'x', "
                                         r"not shape \(3, 2\)$"):
        r.predict(np.ones((3, 2)))
    with pytest.raises(ValueError, match=r"^summary\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        r.summary(alpha=1.5)


def test_parameter_names_and_x_forms():
    xn, yn = decay_data()
    r = sd.regression.nls(lambda x, *t: t[0] * sd.exp(-t[1] * x) + t[2], list(xn), list(yn), p0=[1, 1, 1])
    assert r.param_names == ["b1", "b2", "b3"]
    x2 = np.column_stack([xn, xn ** 2])
    f2 = lambda x, a, b, c: a * sd.exp(-b * x[:, 0]) + c * x[:, 1] * 0.0 + c
    r2 = sd.regression.nls(f2, x2, yn, p0=[1.0, 1.0, 1.0])
    assert rel(r2.params.tolist(), r.params.tolist()) <= 1e-12
    assert r2.predict(x2[:3]).shape == (3,)
    df = sd.frame.DataFrame({"x": xn.tolist()})
    r3 = sd.regression.nls(lambda x, a, b, c: decay(x[:, 0], a, b, c), df, yn, p0=[1.0, 1.0, 1.0])
    assert rel(r3.params.tolist(), r.params.tolist()) <= 1e-12


def test_the_model_gets_an_sd_array_and_floats():
    xn, yn = decay_data()
    seen = []

    def f(x, a, b, c):
        seen.append((type(x), x.dtype, type(a), type(b), type(c)))
        return decay(x, a, b, c).tolist()  # a sequence is taken
    sd.regression.nls(f, xn.astype(np.int64) * 1.0, yn, p0=[1, 1, 1])
    assert set(seen) == {(sd.ndarray, sd.float64, float, float, float)}


# ---- errors

def test_errors():
    xn, yn = decay_data()
    p0 = [1.0, 1.0, 1.0]
    nls = sd.regression.nls
    with pytest.raises(TypeError, match=r"^nls\(\): 'f' must be callable, not int$"):
        nls(3, xn, yn, p0=p0)
    with pytest.raises(TypeError, match=r"^nls\(\): 'jac' must be None or callable, not str$"):
        nls(decay, xn, yn, p0=p0, jac="2-point")
    with pytest.raises(TypeError):
        nls(decay, xn, yn)
    with pytest.raises(ValueError, match=r"^nls\(\): 'p0' must have at least one value$"):
        nls(decay, xn, yn, p0=[])
    with pytest.raises(ValueError, match=r"^nls\(\): 'p0' must be finite$"):
        nls(decay, xn, yn, p0=[1.0, math.nan, 1.0])
    with pytest.raises(ValueError, match=r"^nls\(\): 'method' must be 'lm' or 'trf', not 'dogbox'$"):
        nls(decay, xn, yn, p0=p0, method="dogbox")
    with pytest.raises(ValueError, match=r"^nls\(\): method='lm' takes no bounds \(method='trf' does\)$"):
        nls(decay, xn, yn, p0=p0, method="lm", bounds=(0, 10))
    with pytest.raises(ValueError, match=r"^nls\(\): each lower bound must be less than its upper bound$"):
        nls(decay, xn, yn, p0=p0, bounds=([0, 0, 2], [10, 10, 2]))
    with pytest.raises(ValueError, match=r"^nls\(\): 'p0' must be within 'bounds'$"):
        nls(decay, xn, yn, p0=p0, bounds=(2, 10))
    with pytest.raises(ValueError, match=r"^nls\(\): 'bounds' must have 3 values like 'p0', not 2$"):
        nls(decay, xn, yn, p0=p0, bounds=([0, 0], [10, 10]))
    with pytest.raises(TypeError, match=r"^nls\(\): 'bounds' must be None or a pair \(lower, upper\), not int$"):
        nls(decay, xn, yn, p0=p0, bounds=3)
    with pytest.raises(ValueError, match=r"^nls\(\): 'y' must be 1-dimensional \(nls fits one regression, not a "
                                         r"batch\)$"):
        nls(decay, xn, np.stack([yn, yn]), p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'x' must have 40 rows like 'y', not 39$"):
        nls(decay, xn[:-1], yn, p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'x' must be 1-dimensional \(n,\) or 2-dimensional \(n, d\)$"):
        nls(decay, np.ones((40, 2, 2)), yn, p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'f' must return 40 values \(one per row of 'x'\), not shape "
                                         r"\(39,\)$"):
        nls(lambda x, a, b, c: decay(x, a, b, c)[:-1], xn, yn, p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'f' must return 40 values \(one per row of 'x'\), not a scalar$"):
        nls(lambda x, a, b, c: a, xn, yn, p0=p0)
    with pytest.raises(TypeError, match=r"^nls\(\): 'f' must return an sd.ndarray or a sequence of numbers, not str$"):
        nls(lambda x, a, b, c: "x", xn, yn, p0=p0)
    with pytest.raises(TypeError, match=r"^nls\(\): 'f' must return real values, not a complex128 array$"):
        nls(lambda x, a, b, c: decay(x, a, b, c) * 1j, xn, yn, p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'ftol' must be a finite number >= 0$"):
        nls(decay, xn, yn, p0=p0, ftol=-1.0)
    with pytest.raises(TypeError, match=r"^nls\(\): 'xtol' must be an int or a float, not NoneType$"):
        nls(decay, xn, yn, p0=p0, xtol=None)
    with pytest.raises(ValueError, match=r"^nls\(\): at least one of 'ftol', 'xtol' and 'gtol' must be >= 2.2e-16"):
        nls(decay, xn, yn, p0=p0, ftol=0, xtol=0, gtol=0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'max_nfev' must be >= 1$"):
        nls(decay, xn, yn, p0=p0, max_nfev=0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'p0' has 3 parameters, more than the 2 observations used$"):
        nls(decay, xn[:2], yn[:2], p0=p0)
    with pytest.raises(ValueError, match=r"^nls\(\): 'nan_policy' must be 'propagate', 'omit' or 'raise', "
                                         r"not 'x'$"):
        nls(decay, xn, yn, p0=p0, nan_policy="x")
    with pytest.raises(ZeroDivisionError):  # the model's own errors pass through
        nls(lambda x, a, b, c: 1 / 0, xn, yn, p0=p0)


def test_noise_column_is_aliased():
    """G88: at k ~ 100, exp(-k t) is below c's ulp on every row with t > 0, so the difference
    column of k is rounding noise (constant over those rows: c's direction). It is set to 0 and
    k is the aliased parameter; before, c was dropped in its place and k's bse was 1e15."""
    t = np.linspace(0, 10, 25)
    rng = np.random.default_rng(0)
    y = 2.0 * np.exp(-0.5 * t) + 0.3 + 0.01 * rng.standard_normal(25)

    def f(x, a, k, c):
        return a * sd.exp(-k * x) + c

    r = sd.regression.nls(f, t, y, p0=[1.0, 100.0, 0.0])
    bse = np.asarray(r.bse.tolist())
    assert r.rank == 2 and math.isnan(bse[1]) and np.all(np.isfinite(bse[[0, 2]]))
    p = np.asarray(r.params.tolist())
    # a and c are the fit of the first row and the mean of the rest
    assert abs(p[2] - np.mean(y[1:])) <= 1e-12 and abs(p[0] + p[2] - y[0]) <= 1e-12
    # an identified problem keeps every column
    good = sd.regression.nls(f, t, y, p0=[1.0, 1.0, 0.0])
    assert good.rank == 3 and np.all(np.isfinite(np.asarray(good.bse.tolist())))


def test_overflowing_ssr_is_not_converged():
    """G89: p0 = [1, -50, 0] grows to e^500; the xtol test passes far from a solution (scipy's lm
    stops there too and reports success), the sum of squares overflows: converged=False."""
    t = np.linspace(0, 10, 25)
    rng = np.random.default_rng(0)
    y = 2.0 * np.exp(-0.5 * t) + 0.3 + 0.01 * rng.standard_normal(25)
    r = sd.regression.nls(lambda x, a, k, c: a * sd.exp(-k * x) + c, t, y, p0=[1.0, -50.0, 0.0])
    assert math.isinf(r.ssr) and not r.converged
    assert "overflows" in r.summary()
