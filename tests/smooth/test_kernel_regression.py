"""kernel_regression and bw_dpill (SM4 milestone; docs/design/smooth.md §3.4, §4.4; decisions SM17, SM35, SM80-SM94):
the reference contract on kreg.npz (Nadaraya-Watson and local linear, far points where every double weight
underflows included), the fitted values, leverages and leave-one-out values against their identities, the CV and
AICc criteria equal to statsmodels' functions at the same bandwidth (AICc of the local line: statsmodels' trace is
the local constant's, decision SM89), the chosen bandwidth's criterion not above statsmodels' choice, dpill against
KernSmooth's binned dpill (the recorded values), the limits, the equivariance, the input's order, the threads and
the inputs and errors."""

import functools
import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S
from .conftest import LIVE

A = sd.asarray
KR = sd.smooth.kernel_regression
REF = refdata.ref("kreg")
SM = refdata.data("statsmodels")
R = refdata.data("r")


def _np(a):
    return np.asarray(a.tolist() if hasattr(a, "tolist") else a, dtype=float)


def _set(sid):
    """(X (n, p), y, w or None, Xe (m, p), one_d) of a kreg reference set."""
    if sid in S.scatter_ids():
        s = S.scatter(sid)
        return s.x[:, None], s.y, s.w, s.xe[:, None], True
    s = S.multi(sid)
    return s.X, s.y, s.w, s.Xe, False


def kappa(X, w, x0, bw):
    """The local line's cancellation at x0 (the intercept's variance inflation a [M^-1]_00 = 1 + a zb^T C^-1 zb in
    x0-centred units, kappa_1 of lowess for one variable); 1 for the local constant."""
    z = (X - x0) / bw
    D = np.sum(z * z, axis=1)
    pos = np.ones(len(X), bool) if w is None else w > 0
    v = np.where(pos, (1.0 if w is None else w) * np.exp(-(D - D[pos].min()) / 2), 0.0)
    a = v.sum()
    zb = (v[:, None] * z).sum(axis=0) / a
    o = np.argsort(-v, kind="stable")  # C = R^T R from the weighted centred rows, heaviest first (graded rows)
    R = np.linalg.qr((np.sqrt(v)[:, None] * (z - zb))[o], mode="r")
    d = np.abs(np.diag(R))
    if not np.all(d > 0):
        return math.inf
    g = np.linalg.solve(R.T, zb)
    return 1.0 + a * float(g @ g)


# ---- the reference contract (§4.4) ---------------------------------------------------------------------------


@functools.cache
def _contract_units():
    out = {}
    for g, sid in enumerate(REF.group_inputs["set"].tolist()):
        X, y, w, Xe, one_d = _set(sid)
        far = np.asarray(REF.group_inputs["far"][g], dtype=float)
        for tag in ("a", "b"):
            bw = np.asarray(REF.group_inputs[f"bw_{tag}"][g], dtype=float)
            for name, deg in (("nw", 0), ("ll", 1)):
                method = f"{name}_{tag}"
                m = REF.methods[method]
                r = KR(A(X[:, 0] if one_d else X), A(y), degree=deg, bw=float(bw[0]) if one_d else A(bw),
                       weights=None if w is None else A(w))  # fmt: skip
                for kind in (1, 2):
                    pos, idx = refdata.at(REF, method, sid, kind)
                    for p, i in zip(pos, idx):
                        x0 = Xe[i] if kind == 1 else far
                        got = r.predict(float(x0[0])) if one_d else r.predict(A(x0))
                        k = 1.0 if deg == 0 else kappa(X, w, x0, bw)
                        bound = refdata.kreg_bound(float(m.fields["scale"][p]), k)
                        u = refdata.units(float(got), float(m.hi[p]), float(m.lo[p]), bound)
                        rel = abs(float(got) - float(m.hi[p])) / float(m.fields["scale"][p])
                        out[(sid, method, kind, int(i))] = (u, rel)
    return out


def test_reference_contract():
    worst = max(_contract_units().items(), key=lambda kv: kv[1][0])
    assert worst[1][0] <= 1.0, worst


def test_far_points_keep_the_ratio():
    # every double weight underflows at the far points: the result is the ratio's limit, close in scale units
    far = {k: v for k, v in _contract_units().items() if k[2] == 2}
    assert len(far) == 4 * len(REF.group_inputs["set"])
    assert max(v[1] for v in far.values()) <= 1e-9, max(far.items(), key=lambda kv: kv[1][1])


def test_line_is_reproduced():
    s = S.scatter("line/20")
    for bw in (0.05, 0.5, 5.0):
        r = KR(A(s.x), A(s.y), bw=bw)
        assert np.allclose(_np(r.fittedvalues), s.y, rtol=0, atol=1e-13)
        assert np.allclose(_np(r.gradient(A(s.xe))), 2.0, rtol=0, atol=1e-12)


# ---- identities of the fit ----------------------------------------------------------------------------------


@pytest.mark.parametrize("degree", [0, 1])
@pytest.mark.parametrize("sid", ["random/50", "weights/30", "ties/30", "clustered/40", "outliers/40"])
def test_fit_identities(sid, degree):
    s = S.scatter(sid)
    w = None if s.w is None else A(s.w)
    r = KR(A(s.x), A(s.y), degree=degree, weights=w)
    f, lev = _np(r.fittedvalues), _np(r.leverage)
    pos = np.ones(s.n, bool) if s.w is None else s.w > 0
    assert np.all((lev[pos] >= 0) & (lev[pos] <= 1 + 1e-15)) and np.all(lev[~pos] == 0)
    assert math.isclose(r.df, float(lev[pos].sum()), rel_tol=1e-13)
    wv = np.ones(s.n) if s.w is None else s.w
    rss = float(np.sum(wv * (s.y - f) ** 2))
    assert math.isclose(r.scale, math.sqrt(rss / (r.nobs - r.df)), rel_tol=1e-12)
    # the fitted values are the estimator at the data points
    assert np.allclose(_np(r.predict(A(s.x))), f, rtol=0, atol=1e-12 * max(1.0, np.abs(s.y).max()))
    # CV = sum w ((y - f) / (1 - L))^2 / nobs (the leave-one-out identity)
    cv = float(np.sum((wv * ((s.y - f) / (1 - lev)) ** 2)[pos])) / r.nobs
    assert math.isclose(r.criterion_value, cv, rel_tol=1e-9)
    assert r.criterion == "cv_ls" and r.status in (-1, 0, 1)


def test_leave_one_out_is_the_refit():
    s = S.scatter("random/30")
    for degree in (0, 1):
        r = KR(A(s.x), A(s.y), degree=degree, bw=0.7)
        f, lev = _np(r.fittedvalues), _np(r.leverage)
        for i in (0, 7, 29):
            keep = np.arange(s.n) != i
            loo = KR(A(s.x[keep]), A(s.y[keep]), degree=degree, bw=0.7).predict(float(s.x[i]))
            assert math.isclose(loo, f[i] - lev[i] * (s.y[i] - f[i]) / (1 - lev[i]), rel_tol=1e-12, abs_tol=1e-14)


def test_limits():
    s = S.scatter("random/50")
    big = KR(A(s.x), A(s.y), degree=0, bw=1e9)
    assert np.allclose(_np(big.fittedvalues), s.y.mean(), rtol=1e-12)
    ll = KR(A(s.x), A(s.y), degree=1, bw=1e7)
    b1, b0 = np.polyfit(s.x, s.y, 1)
    assert np.allclose(_np(ll.fittedvalues), b0 + b1 * s.x, rtol=0, atol=1e-9)
    tiny = KR(A(s.x), A(s.y), degree=0, bw=1e-6)
    assert np.array_equal(_np(tiny.fittedvalues), s.y) and np.all(_np(tiny.leverage) == 1.0)
    # far from the data the local constant is the nearest point's value, not 0/0
    assert tiny.predict(-1e3) == s.y[np.argmin(s.x)]


def test_power_of_two_scaling_and_order():
    s = S.scatter("random/50")
    for degree in (0, 1):
        r = KR(A(s.x), A(s.y), degree=degree)
        r2 = KR(A(4.0 * s.x), A(s.y), degree=degree)
        assert r2.bw == 4.0 * r.bw
        assert bits_equal(r.fittedvalues, r2.fittedvalues) and bits_equal(r.leverage, r2.leverage)
        p = np.random.default_rng(3).permutation(s.n)
        rp = KR(A(s.x[p]), A(s.y[p]), degree=degree)
        assert rp.bw == r.bw and np.array_equal(_np(rp.fittedvalues), _np(r.fittedvalues)[p])
    m = S.multi("uniform2/60")
    p = np.random.default_rng(4).permutation(m.n)
    r = KR(A(m.X), A(m.y), bw=A([0.4, 0.5]))
    rp = KR(A(m.X[p]), A(m.y[p]), bw=A([0.4, 0.5]))
    assert np.array_equal(_np(rp.fittedvalues), _np(r.fittedvalues)[p])


# ---- the criteria against statsmodels (decision SM89) --------------------------------------------------------


def _sm_reg(x, y, reg, h):
    import warnings

    import statsmodels.api as sm

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k = sm.nonparametric.KernelReg(y, x, "c" * (1 if x.ndim == 1 else x.shape[1]), reg_type=reg, bw=list(h))
    return k


@LIVE
@pytest.mark.parametrize("sid", ["random/50", "outliers/40", "uniform2/60"])
def test_criteria_equal_statsmodels(sid):
    import warnings

    from stardust.smooth._kreg import _criterion

    X, y, _, _, one_d = _set(sid)
    x = X[:, 0] if one_d else X
    for h0 in (0.3, 0.8, 2.0):
        h = np.full(X.shape[1], h0)
        bw = h0 if one_d else A(h)
        for deg, reg in ((0, "lc"), (1, "ll")):
            k = _sm_reg(x, y, reg, h)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_sm = float(np.ravel(k.cv_loo(h, k.est[reg]))[0])
                aic_sm = float(np.asarray(k.aic_hurvich(h, k.est[reg])).ravel()[0])
            cv = _criterion(A(x), A(y), bw, degree=deg, criterion="cv_ls")
            assert math.isclose(cv, cv_sm, rel_tol=1e-12), (sid, h0, deg, cv, cv_sm)
            aicc = _criterion(A(x), A(y), bw, degree=deg, criterion="aicc")
            if deg == 0:
                assert math.isclose(aicc, aic_sm, rel_tol=1e-12, abs_tol=1e-13), (sid, h0, aicc, aic_sm)
            else:  # statsmodels' AIC of the local line takes the local constant's trace (decision SM89)
                r1 = KR(A(x), A(y), degree=1, bw=bw)
                r0 = KR(A(x), A(y), degree=0, bw=bw)
                n = len(y)
                rss = float(np.sum(_np(r1.resid) ** 2))
                theirs = math.log(rss / n) + (1 + r0.df / n) / (1 - (r0.df + 2) / n)
                assert math.isclose(theirs, aic_sm, rel_tol=1e-12, abs_tol=1e-13), (sid, h0, theirs, aic_sm)
                ours = math.log(rss / n) + (1 + r1.df / n) / (1 - (r1.df + 2) / n)
                assert math.isclose(aicc, ours, rel_tol=1e-12, abs_tol=1e-13)


def test_chosen_criterion_not_above_statsmodels_choice():
    # tests/smooth/data/statsmodels.json: KernelReg's bandwidths on random/200 (fmin from h0 = 1.06 sigma n^-1/5)
    from stardust.smooth._kreg import _criterion

    s = S.scatter("random/200")
    for reg, deg in (("ll", 1), ("lc", 0)):
        for rule, crit in (("cv_ls", "cv_ls"), ("aic", "aicc")):
            rec = SM[f"kernel_reg_bw/random/200/{reg}/{rule}"]
            assert rec["digest"] == s.digest
            h_sm = float(rec["value"]["array"][0])
            if deg == 1 and crit == "aicc":
                continue  # statsmodels minimizes another function there (decision SM89)
            r = KR(A(s.x), A(s.y), degree=deg, bw=crit)
            theirs = _criterion(A(s.x), A(s.y), h_sm, degree=deg, criterion=crit)
            assert r.criterion_value <= theirs * (1 + 1e-12) if crit == "cv_ls" else r.criterion_value <= theirs + 1e-12
            assert r.converged and r.status == 0
            assert math.isclose(r.bw, h_sm, rel_tol=2e-3), (reg, rule, r.bw, h_sm)


def test_search_ends_and_multivariate_search():
    g = np.random.default_rng(8)
    x = np.sort(10.0 * g.random(80))
    r = KR(A(x), A(g.standard_normal(80)), degree=0)  # noise about a constant: the largest bandwidth
    assert r.status == 1 and not r.converged and math.isclose(r.bw, 64 * 1.06 * np.std(x) * 80**-0.2, rel_tol=1e-12)
    m = S.multi("uniform2/60")
    r = KR(A(m.X), A(m.y))
    assert r.bw.shape == (2,) and r.status == 0 and r.converged
    from stardust.smooth._kreg import _criterion

    for f in (0.97, 1.03):
        assert r.criterion_value <= _criterion(A(m.X), A(m.y), A(_np(r.bw) * np.array([f, 1.0]))) + 1e-12


# ---- dpill (decision SM90) -----------------------------------------------------------------------------------


@pytest.mark.parametrize("sid", ["uniform/40", "random/30", "random/50", "random/200"])
def test_dpill_near_kernsmooth(sid):
    s = S.scatter(sid)
    rec = R[f"dpill/{sid}"]
    assert rec["digest"] == s.digest
    h = sd.smooth.bw_dpill(A(s.x), A(s.y))
    assert math.isclose(h, rec["values"][0], rel_tol=0.05), (sid, h, rec["values"][0])
    r = KR(A(s.x), A(s.y), bw="dpill")
    assert r.bw == h and r.criterion == "dpill" and math.isnan(r.criterion_value) and r.converged


def test_dpill_scales_with_x():
    s = S.scatter("random/200")
    h = sd.smooth.bw_dpill(A(s.x), A(s.y))
    assert math.isclose(sd.smooth.bw_dpill(A(8.0 * s.x), A(s.y)), 8.0 * h, rel_tol=1e-12)
    assert math.isclose(sd.smooth.bw_dpill(A(s.x), A(3.0 * s.y)), h, rel_tol=1e-12)


def test_dpill_errors():
    with pytest.raises(ValueError, match="at least 6 points"):
        sd.smooth.bw_dpill(A([0.0, 1.0, 2.0]), A([1.0, 2.0, 0.0]))
    with pytest.raises(ValueError, match="positive spread"):
        sd.smooth.bw_dpill(A(np.ones(30)), A(np.arange(30.0)))
    with pytest.raises(ValueError, match="finite"):
        sd.smooth.bw_dpill(A([0.0, math.nan] + list(range(10))), A(np.arange(12.0)))
    with pytest.raises(ValueError, match="cannot be estimated"):
        sd.smooth.bw_dpill(A(np.arange(40.0)), A(np.zeros(40)))
    s = S.multi("uniform2/60")
    with pytest.raises(ValueError, match="one variable and degree=1"):
        KR(A(s.X), A(s.y), bw="dpill")
    with pytest.raises(ValueError, match="no weights"):
        KR(A(np.arange(30.0)), A(np.sin(np.arange(30.0))), bw="dpill", weights=A(np.ones(30)))


# ---- results, inputs and errors -----------------------------------------------------------------------------


def test_result_fields_and_predict_shapes():
    s = S.scatter("random/50")
    r = KR(A(s.x), A(s.y), bw=0.6)
    assert isinstance(r, sd.smooth.KernelRegressionResult)
    assert r.bw == 0.6 and r.degree == 1 and r.kernel == "gaussian" and r.criterion is None
    assert math.isnan(r.criterion_value) and r.status == 0 and r.converged
    assert isinstance(r.predict(2.5), float) and isinstance(r.gradient(2.5), float)
    assert r.predict(A(np.ones((2, 3)))).shape == (2, 3)
    iv = r.predict_interval(A(s.xe))
    assert iv.shape == (s.xe.size, 2) and np.all(_np(iv[:, 0]) < _np(iv[:, 1]))
    assert r.predict_interval(1.0).shape == (2,)
    assert "Kernel Regression Results" in r.summary() and "Bandwidth rule:" in r.summary()
    assert repr(r).startswith("KernelRegressionResult(nobs=50, df=")
    with pytest.raises(AttributeError):
        r.bw = 1.0
    with pytest.raises(ValueError, match="no gradient"):
        KR(A(s.x), A(s.y), degree=0, bw=0.6).gradient(1.0)
    m = S.multi("uniform3/60")
    rm = KR(A(m.X), A(m.y), bw=0.5)
    assert rm.bw.shape == (3,) and rm.predict(A(m.Xe)).shape == (m.Xe.shape[0],)
    assert isinstance(rm.predict(A(m.Xe[0])), float) and rm.gradient(A(m.Xe[0])).shape == (3,)
    assert rm.gradient(A(m.Xe)).shape == (m.Xe.shape[0], 3)
    with pytest.raises(ValueError, match=r"\(m, 3\)"):
        rm.predict(A(np.ones((2, 2))))


def test_gradient_matches_statsmodels():
    s = S.scatter("random/50")
    k = _sm_reg(s.x, s.y, "ll", [0.6])
    mean, mfx = k.fit(s.xe)
    r = KR(A(s.x), A(s.y), bw=0.6)
    assert np.allclose(_np(r.predict(A(s.xe))), mean, rtol=0, atol=1e-12)
    assert np.allclose(_np(r.gradient(A(s.xe))), mfx[:, 0], rtol=0, atol=1e-11)


def test_nan_policy():
    s = S.scatter("random/30")
    x, y = s.x.copy(), s.y.copy()
    x[4] = math.nan
    r = KR(A(x), A(y), bw=0.6)
    assert np.all(np.isnan(_np(r.fittedvalues))) and math.isnan(r.df) and math.isnan(r.predict(1.0))
    o = KR(A(x), A(y), bw=0.6, nan_policy="omit")
    keep = np.arange(30) != 4
    ref = KR(A(x[keep]), A(y[keep]), bw=0.6)
    assert math.isnan(_np(o.fittedvalues)[4]) and np.array_equal(_np(o.fittedvalues)[keep], _np(ref.fittedvalues))
    assert o.nobs == 29
    with pytest.raises(ValueError, match="nan_policy='raise'"):
        KR(A(x), A(y), nan_policy="raise")
    with pytest.raises(ValueError, match="infinity"):
        KR(A([0.0, math.inf, 1.0]), A([1.0, 2.0, 3.0]))


def test_weights_zero_points():
    s = S.scatter("weights/30")
    r = KR(A(s.x), A(s.y), bw=0.6, weights=A(s.w))
    keep = s.w > 0
    ref = KR(A(s.x[keep]), A(s.y[keep]), bw=0.6, weights=A(s.w[keep]))
    assert np.array_equal(_np(r.fittedvalues)[keep], _np(ref.fittedvalues)) and r.nobs == 27
    assert np.array_equal(_np(r.fittedvalues)[~keep], _np(ref.predict(A(s.x[~keep]))))


def test_inputs_and_errors():
    x = np.linspace(0.0, 5.0, 25)
    y = np.sin(x)
    a = KR(A(x), A(y), bw=0.5)
    assert np.array_equal(_np(KR(x.tolist(), y.tolist(), bw=0.5).fittedvalues), _np(a.fittedvalues))
    assert np.array_equal(_np(KR(A(x[:, None]), A(y), bw=A([0.5])).fittedvalues), _np(a.fittedvalues))
    df = sd.frame.DataFrame({"u": x, "v": x[::-1].copy()})
    rd = KR(df, A(y), bw=0.5)
    assert rd.bw.shape == (2,)
    for kw, err, msg in (
        ({"degree": 2}, ValueError, "'degree' must be 0 or 1"),
        ({"degree": 1.0}, TypeError, "'degree' must be an int"),
        ({"bw": -1.0}, ValueError, "must be a finite number > 0"),
        ({"bw": "scott"}, ValueError, "'bw' must be"),
        ({"bw": A([0.5, 0.5])}, ValueError, "have 1 values"),
        ({"kernel": "epanechnikov"}, ValueError, "'kernel' must be 'gaussian'"),
        ({"nan_policy": "drop"}, ValueError, "'nan_policy'"),
        ({"weights": A(-np.ones(25))}, ValueError, ">= 0"),
        ({"weights": A(np.zeros(25))}, ValueError, "at least one weight must be positive"),
    ):
        with pytest.raises(err, match=msg):
            KR(A(x), A(y), **kw)
    with pytest.raises(ValueError, match="same number of points"):
        KR(A(x), A(y[:-1]))
    with pytest.raises(ValueError, match="at least 2 points"):
        KR(A([1.0]), A([2.0]))
    with pytest.raises(ValueError, match="positive spread"):
        KR(A(np.full(10, 2.0)), A(np.arange(10.0)))
    with pytest.raises(TypeError, match="unexpected keyword"):
        KR(A(x), A(y), bandwidth=1.0)
    one = KR(A([1.0]), A([2.0]), bw=1.0)
    assert one.fittedvalues[0] == 2.0 and one.leverage[0] == 1.0 and math.isnan(one.scale)
    same = KR(A(np.full(10, 2.0)), A(np.arange(10.0)), bw=1.0)  # one x: the local mean
    assert np.allclose(_np(same.fittedvalues), 4.5)


# ---- threads (decision SM3) ----------------------------------------------------------------------------------


@pytest.mark.usefixtures("restore_threads")
def test_thread_counts():
    g = np.random.default_rng(41)
    x = np.sort(10.0 * g.random(1500))
    y = np.sin(x) + 0.3 * g.standard_normal(x.size)
    X = 2.0 * g.random((400, 2))
    Y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.2 * g.standard_normal(400)
    q = np.linspace(-1.0, 11.0, 3000)

    def results():
        r = KR(A(x), A(y), bw="aicc")
        n0 = KR(A(x), A(y), degree=0, bw=0.2)
        m = KR(A(X), A(Y), bw=A([0.3, 0.4]))
        return [r.fittedvalues, r.leverage, r.bw, r.criterion_value, r.predict_interval(A(q)), r.gradient(A(q)),
                n0.fittedvalues, n0.predict(A(q)), m.fittedvalues, m.leverage, m.predict(A(X * 1.1))]  # fmt: skip

    sd.set_num_threads(1)
    base = results()
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        for k, (a, b) in enumerate(zip(base, results())):
            assert bits_equal(a, b), (t, k)
