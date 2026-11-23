"""lowess and LowessResult (docs/api/smooth.md §2, docs/design/smooth.md §2.2, §3.1, §4.1; decisions SM8, SM15, SM27,
SM36-SM49): the reference contract of the fitted values, predict and the leverages (tests/smooth/ref/local.npz), the
robust iterations against the exact fit with sd's robust weights, delta's plan, the results' independence of the
input order, ties and power-of-two scalings, statsmodels 0.15.0 and R 4.6.2 where they compute the same estimator and
the documented differences where they do not, the input rules and the result object."""

import math

import numpy as np
import pytest
from flint import fmpq

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S
from .conftest import LIVE

LOCAL = refdata.ref("local")
SM_REC = refdata.data("statsmodels")
R_REC = refdata.data("r")
U = refdata.U
FRACS = (("lowess_f0.3", 0.3, "0.3"), ("lowess_f0.67", 2 / 3, "0.67"), ("lowess_f1", 1.0, "1"))


def A(v):
    return sd.asarray(np.asarray(v, dtype=float))


def N(a):
    return np.asarray(a.tolist() if isinstance(a, sd.ndarray) else a, dtype=float)


def per_set(method, sid, kind):
    r = LOCAL
    pos, idx = refdata.at(r, method, sid, kind)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def fit(s, frac, **kw):
    w = None if s.w is None else A(s.w)
    return sd.smooth.lowess(A(s.x), A(s.y), frac=frac, weights=w, **kw)


# ---- the exact local line (the contract's reference for a robust fit) ------------------------------------------


def Q(v):
    return fmpq(*float(v).as_integer_ratio())


def exact_line(xs, ys, pw, x0, k):
    """(value, scale = sum |l_j||y_j|, kappa_1, npos) of the exact weighted local line at x0 from the k nearest
    points (exact distances), prior weights pw, or None with fewer than two positive weights."""
    X, Y, W, x0 = [Q(v) for v in xs], [Q(v) for v in ys], [Q(v) for v in pw], Q(x0)
    d = [abs(x - x0) for x in X]
    h = sorted(d)[k - 1]
    if h == 0:
        idx = [j for j in range(len(X)) if d[j] == 0 and W[j] > 0]
        if len(idx) < 2:
            return None
        a = sum((W[j] for j in idx), fmpq(0))
        return sum((W[j] * Y[j] for j in idx), fmpq(0)) / a, float(sum(W[j] * abs(Y[j]) for j in idx) / a), 1.0, 2
    v = {}
    for j in range(len(X)):
        if d[j] < h and W[j] > 0:
            r = d[j] / h
            v[j] = (1 - r * r * r) ** 3 * W[j]
    if len(v) < 2:
        return None
    u = {j: (X[j] - x0) / h for j in v}
    a = sum(v.values(), fmpq(0))
    b = sum((v[j] * u[j] for j in v), fmpq(0))
    c = sum((v[j] * u[j] ** 2 for j in v), fmpq(0))
    det = a * c - b * b
    if det == 0:
        row = {j: v[j] / a for j in v}
        kappa = 1.0
    else:
        row = {j: v[j] * (c - b * u[j]) / det for j in v}
        kappa = float(a * c / det)
    value = sum((row[j] * Y[j] for j in v), fmpq(0))
    scale = float(sum((abs(row[j]) * abs(Y[j]) for j in v), fmpq(0)))
    return value, scale, kappa, len(v)


def exact_units(got, ref):
    value, scale, kappa, _ = ref
    err = abs(Q(got) - value) if math.isfinite(got) else math.inf
    bound = refdata.local_bound(scale, kappa)
    return 0.0 if err == 0 else float(err) / bound


# ---- the reference contract -------------------------------------------------------------------------------------


def test_fitted_values_predict_and_leverage_meet_the_contract():
    """|got - ref| <= C_LOCAL kappa_1 u sum |l_j||y_j| at every data point (it = 0) and evaluation point; the
    leverage within C_LOCAL kappa_1 u (the same form with sum |l_j| >= 1 for the hat row); points with fewer than two
    positive weights give y (fit) and NaN (predict), as statsmodels."""
    worst = {"fit": 0.0, "predict": 0.0, "leverage": 0.0}
    seen = 0
    for sid in sorted(set(LOCAL.inputs["set"].tolist())):
        s = S.scatter(sid)
        for method, frac, _ in FRACS:
            r = fit(s, frac, it=0)
            got_fit, got_lev = N(r.fittedvalues), N(r.leverage)
            got_pred = N(r.predict(A(s.xe)))
            for kind, got in ((0, got_fit), (1, got_pred)):
                idx, hi, lo, f = per_set(method, sid, kind)
                for j, i in enumerate(idx):
                    if not (f["npos"][j] >= 2) or math.isnan(hi[j]):
                        if kind == 0:
                            assert got[i] == s.y[i], (sid, method, i)
                        else:
                            assert math.isnan(got[i]), (sid, method, i)
                        continue
                    u = refdata.units(got[i], hi[j], lo[j], refdata.local_bound(f["scale"][j], f["kappa"][j]))
                    key = "fit" if kind == 0 else "predict"
                    worst[key] = max(worst[key], u)
                    assert u <= 1, (sid, method, kind, i, u)
                    if kind == 0:
                        lb = refdata.C_LOCAL * f["kappa"][j] * U  # the hat row has sum |l_j| >= 1
                        lu = 0.0 if got_lev[i] == f["leverage"][j] else abs(got_lev[i] - f["leverage"][j]) / lb
                        worst["leverage"] = max(worst["leverage"], lu)
                        assert lu <= 1, (sid, method, i, lu)
                    seen += 1
    assert seen > 2000


ROBUST_SETS = ("random/50", "outliers/40", "clustered/40", "weights/30", "halfzero/21", "ties/30", "shifted/30")


@pytest.mark.parametrize("sid", ROBUST_SETS)
def test_robust_fit_is_the_exact_fit_with_its_robust_weights(sid):
    """docs/design/smooth.md §4.1: the result of the last iteration is the exact local line with the robust weights
    sd used (its robust_weights times the weights), within the contract; the leverages sum to df."""
    s = S.scatter(sid)
    for it in (1, 3):
        r = fit(s, 0.3, it=it)
        rho = N(r.robust_weights)
        pw = rho * (s.w if s.w is not None else 1.0)
        got = N(r.fittedvalues)
        for i in range(s.n):
            ref = exact_line(s.x, s.y, pw, s.x[i], r.k)
            if ref is None:
                assert got[i] == s.y[i]
                continue
            assert exact_units(got[i], ref) <= 1, (sid, it, i)
        pred = N(r.predict(A(s.xe)))
        for i, x0 in enumerate(s.xe):
            ref = exact_line(s.x, s.y, pw, x0, r.k)
            if ref is None:
                assert math.isnan(pred[i])
                continue
            assert exact_units(pred[i], ref) <= 1, (sid, it, i)
        lev = N(r.leverage)
        assert r.df == pytest.approx(lev[(s.w if s.w is not None else np.ones(s.n)) > 0].sum(), rel=1e-12)


def test_robust_weights_are_statsmodels_bisquare():
    """rho = (1 - (r / 6m)^2)^2 with m the median |r| of the previous fit, 0 beyond 6m; m = 0 gives 1 at the zero
    residuals and 0 elsewhere (halfzero/21: 14 of 21 points on a line)."""
    s = S.scatter("outliers/40")
    r0 = fit(s, 0.3, it=0)
    r1 = fit(s, 0.3, it=1)
    res = np.abs(N(r0.resid))
    m = np.median(res)
    u = np.minimum(res / (6 * m), 1.0)
    assert np.array_equal(N(r1.robust_weights), (1 - u * u) ** 2)
    assert np.array_equal(N(r0.robust_weights), np.ones(s.n))
    h = S.scatter("halfzero/21")
    rh = fit(h, 1.0, it=1)
    r0 = fit(h, 1.0, it=0)
    if np.median(np.abs(N(r0.resid))) == 0:
        assert set(N(rh.robust_weights).tolist()) <= {0.0, 1.0}


# ---- statsmodels and R ------------------------------------------------------------------------------------------


def rec_value(rec):
    return np.array(rec["value"]["array"], dtype=float)


def test_statsmodels_records_where_x_has_a_unit_scale():
    """statsmodels' lowess computes the same estimator (it = 0 and 3, delta 0 and 1% of the range) on the sets of
    unit scale; it departs through its absolute 1e-12 guards only (the narrow and subnormal sets)."""
    worst = {}
    for sid in S.scatter_ids(tag="plain") + ["outliers/40", "step/40", "unsorted/30"]:
        s = S.scatter(sid)
        span = float(np.ptp(s.x))
        for _, frac, tag in FRACS:
            for it in (0, 3):
                for dtag, delta in (("0", 0.0), ("r", 0.01 * span)):
                    rec = SM_REC.get(f"lowess/{sid}/f{tag}/it{it}/delta{dtag}")
                    if rec is None:
                        continue
                    want = rec_value(rec)
                    got = N(fit(s, frac, it=it, delta=delta).fittedvalues)
                    err = float(np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want))))
                    worst[(it, dtag)] = max(worst.get((it, dtag), 0.0), err)
    assert worst[(0, "0")] < 1e-13 and worst[(0, "r")] < 1e-13, worst
    assert worst[(3, "0")] < 1e-11 and worst[(3, "r")] < 1e-11, worst


def test_statsmodels_xvals_is_predict():
    for sid in S.scatter_ids(tag="plain"):
        s = S.scatter(sid)
        for _, frac, tag in FRACS:
            want = rec_value(SM_REC[f"lowess_xvals/{sid}/f{tag}"])
            got = N(fit(s, frac, it=0).predict(A(s.xe)))
            ok = ~np.isnan(want)
            assert np.allclose(got[ok], want[ok], rtol=0, atol=1e-13), (sid, tag)


def test_r_differs_by_its_weight_cut_only():
    """R cuts the tricube weights at 0.001 h and 0.999 h (and the robust ones likewise): relative differences of
    order 1e-9, below 1e-7 (docs/api/smooth.md §2)."""
    worst = 0.0
    for sid in S.scatter_ids(tag="plain") + ["outliers/40", "step/40", "unsorted/30"]:
        s = S.scatter(sid)
        for _, frac, tag in FRACS:
            for it in (0, 3):
                want = np.array(R_REC[f"lowess/{sid}/f{tag}/it{it}/delta0"]["values"])
                got = N(fit(s, frac, it=it).fittedvalues)
                worst = max(worst, float(np.max(np.abs(got - want)) / max(1.0, np.max(np.abs(want)))))
    assert 1e-12 < worst < 1e-7, worst


def test_clustered_x_is_where_statsmodels_and_r_give_up_the_line():
    """Clusters 1e-8 wide, 2.5 apart: a window's weighted spread of x is below statsmodels' absolute 1e-12 and R's
    0.001 range, so both fall back to a local mean; sd fits the line its definition asks for (decision SM15), which
    the exact reference holds within the contract (test above)."""
    s = S.scatter("clustered/40")
    got = N(fit(s, 0.3, it=0).fittedvalues)
    sm = rec_value(SM_REC["lowess/clustered/40/f0.3/it0/delta0"])
    r = np.array(R_REC["lowess/clustered/40/f0.3/it0/delta0"]["values"])
    assert np.max(np.abs(got - sm)) > 1e-3 and np.max(np.abs(got - r)) > 1e-3
    idx, hi, lo, f = per_set("lowess_f0.3", "clustered/40", 0)
    ok = f["npos"] >= 2
    assert np.max(np.abs(got[idx[ok]] - hi[ok])) < 1e-12


def test_r_default_delta_is_one_percent_of_the_range():
    s = S.scatter("random/200")
    want = np.array(R_REC["lowess_default/random/200"]["values"])
    got = N(fit(s, 2 / 3, delta=0.01 * float(np.ptp(s.x))).fittedvalues)
    assert np.max(np.abs(got - want)) < 1e-7
    assert np.max(np.abs(N(fit(s, 2 / 3).fittedvalues) - want)) > 1e-7  # sd's default fits every point


def test_unit_of_x_does_not_matter():
    """statsmodels' absolute 1e-12 guard makes its fit depend on the unit of x (0.34 at x * 1e-8 on random/200,
    the record); sd's degenerate rule has no scale, so the fits agree within the contract, and are the same bits
    for a power-of-two unit."""
    s = S.scatter("random/200")
    assert SM_REC["rule/lowess_depends_on_the_unit_of_x"]["1e-08"] > 0.1
    base = N(sd.smooth.lowess(A(s.x), A(s.y), frac=0.1, it=0).fittedvalues)
    for f in (1e-8, 1e-4, 1e4, 1e8):
        got = N(sd.smooth.lowess(A(s.x * f), A(s.y), frac=0.1, it=0).fittedvalues)
        assert np.max(np.abs(got - base)) < 1e-12
    for p in (-1000, -60, 3, 900):
        a = sd.smooth.lowess(A(s.x * 2.0**p), A(s.y), frac=0.3, it=2, delta=0.05 * 2.0**p)
        b = sd.smooth.lowess(A(s.x), A(s.y), frac=0.3, it=2, delta=0.05)
        assert bits_equal(a.fittedvalues, b.fittedvalues) and bits_equal(a.leverage, b.leverage), p


def test_all_x_equal_and_ties():
    """Every x equal: the weighted mean of all points (h = 0; statsmodels divides by 0 and warns). Tied x: the
    points at one x get one value."""
    t = S.scatter("allsame/8")
    r = sd.smooth.lowess(A(t.x), A(t.y), it=0)
    assert np.allclose(N(r.fittedvalues), np.mean(t.y), rtol=0, atol=1e-15)
    assert SM_REC["rule/lowess_all_x_equal"]["warnings"]
    s = S.scatter("ties/30")
    got = N(sd.smooth.lowess(A(s.x), A(s.y), frac=0.3, it=0).fittedvalues)
    for v in np.unique(s.x):
        assert len(set(got[s.x == v].tolist())) == 1


# ---- delta ------------------------------------------------------------------------------------------------------


def test_delta_interpolates_between_fits():
    """A delta beyond the range: statsmodels' plan fits the first point, the next to last and the last, and
    interpolates linearly between the first two (its update_indices)."""
    s = S.scatter("random/200")
    o = np.argsort(s.x)
    full = N(fit(s, 0.3, it=0).fittedvalues)[o]
    fx = N(fit(s, 0.3, it=0, delta=100.0).fittedvalues)[o]
    xs = s.x[o]
    assert fx[0] == full[0] and fx[-2] == full[-2] and fx[-1] == full[-1]
    a = (xs[1:-2] - xs[0]) / (xs[-2] - xs[0])
    assert np.array_equal(fx[1:-2], a * full[-2] + (1 - a) * full[0])
    tiny = fit(s, 0.3, it=2, delta=1e-12)
    assert bits_equal(tiny.fittedvalues, fit(s, 0.3, it=2).fittedvalues)
    assert bits_equal(tiny.leverage, fit(s, 0.3, it=2).leverage)


def test_delta_leverage_is_the_interpolated_hat_diagonal():
    s = S.scatter("random/50")
    r = fit(s, 0.67, it=0, delta=1.0)
    # column i of the smoother: fit of y = e_i, read at point i
    lev = np.empty(s.n)
    for i in range(s.n):
        e = np.zeros(s.n)
        e[i] = 1.0
        lev[i] = N(sd.smooth.lowess(A(s.x), A(e), frac=0.67, it=0, delta=1.0).fittedvalues)[i]
    assert np.allclose(N(r.leverage), lev, rtol=0, atol=1e-14)


# ---- order, ties and threads ------------------------------------------------------------------------------------


def test_result_is_a_function_of_the_data():
    """The same points in another order give the same bits in that order (decision SM36), weights included."""
    g = np.random.default_rng(7)
    for sid in ("random/200", "ties/30", "weights/30", "clustered/40"):
        s = S.scatter(sid)
        r = fit(s, 0.3, it=3, delta=0.02)
        p = g.permutation(s.n)
        w = None if s.w is None else A(s.w[p])
        q = sd.smooth.lowess(A(s.x[p]), A(s.y[p]), frac=0.3, it=3, delta=0.02, weights=w)
        for name in ("fittedvalues", "leverage", "robust_weights", "resid"):
            assert bits_equal(getattr(q, name), getattr(r, name)[sd.asarray(p)]), (sid, name)
        assert q.df == r.df and q.scale == r.scale
        assert bits_equal(q.predict(A(s.xe)), r.predict(A(s.xe)))


def test_window_ties_at_the_radius_do_not_matter():
    """Two points at distance h on either side of x0: whichever the window takes has weight 0, so the fit is the
    exact local line of the points inside (x = 0..4, k = 4 at x0 = 2: 0 and 4 tie at h = 2)."""
    x = np.arange(5.0)
    y = np.array([0.3, -1.0, 2.0, 0.5, 7.0])
    r = sd.smooth.lowess(A(x), A(y), frac=0.8, it=0)
    assert r.k == 4
    got = r.predict(2.0)
    ref = exact_line(x, y, np.ones(5), 2.0, 4)
    assert ref[3] == 3 and exact_units(got, ref) <= 1
    y2 = y.copy()
    y2[[0, 4]] = (-50.0, 90.0)  # the tied points' values change nothing
    assert sd.smooth.lowess(A(x), A(y2), frac=0.8, it=0).predict(2.0) == got


def test_thread_counts_give_the_same_bytes(restore_threads):
    g = np.random.default_rng(3)
    n = 6000
    x = np.sort(g.random(n)) * 10
    x[100:140] = x[100]  # ties
    y = np.sin(x) + 0.3 * g.standard_normal(n)
    w = 0.5 + g.random(n)
    w[::17] = 0.0
    q = g.random(5000) * 12 - 1
    out = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        res = []
        for kw in ({}, {"weights": A(w)}, {"delta": 0.05}, {"frac": 0.1, "it": 1}):
            r = sd.smooth.lowess(A(x), A(y), **kw)
            res += [r.fittedvalues, r.leverage, r.robust_weights, sd.asarray([r.df, r.scale])]
            res += [r.predict(A(q)), r.predict_interval(A(q))]
        out[t] = res
    for t in (3, 10):
        for k, (a, b) in enumerate(zip(out[1], out[t])):
            assert bits_equal(a, b), (t, k)


# ---- inputs, results ---------------------------------------------------------------------------------------------


def test_small_samples():
    for n in (1, 2, 3, 5):
        s = S.scatter(f"tiny/{n}")
        r = sd.smooth.lowess(A(s.x), A(s.y))
        assert r.k == min(n, 2) if n < 3 else r.k == max(2, int(2 / 3 * n + 1e-10))
        assert N(r.fittedvalues).shape == (n,)
        if n <= 2:
            assert np.array_equal(N(r.fittedvalues), s.y)  # one positive weight in every window
            assert np.isnan(N(r.predict(A(s.xe)))).all()


def test_exact_line_is_reproduced():
    s = S.scatter("line/20")
    for it in (0, 3):
        r = sd.smooth.lowess(A(s.x), A(s.y), frac=0.3, it=it)
        assert np.max(np.abs(N(r.fittedvalues) - s.y)) <= 1e-14 * np.max(np.abs(s.y))
        assert abs(r.predict(1.3) - 3.6) < 1e-13


def test_nan_policy():
    s = S.scatter("random/50")
    y = s.y.copy()
    y[[4, 30]] = np.nan
    r = sd.smooth.lowess(A(s.x), A(y))
    for name in ("fittedvalues", "resid", "leverage", "robust_weights"):
        assert np.isnan(N(getattr(r, name))).all()
    assert math.isnan(r.df) and math.isnan(r.scale) and math.isnan(r.predict(5.0))
    assert np.isnan(N(r.predict_interval(A([1.0, 2.0])))).all()
    o = sd.smooth.lowess(A(s.x), A(y), nan_policy="omit")
    keep = ~np.isnan(y)
    ref = sd.smooth.lowess(A(s.x[keep]), A(y[keep]))
    got = N(o.fittedvalues)
    assert np.isnan(got[~keep]).all() and np.array_equal(got[keep], N(ref.fittedvalues))
    assert o.nobs == 48 and o.df == ref.df
    with pytest.raises(ValueError, match=r"^lowess\(\): the input contains NaN \(nan_policy='raise'\)$"):
        sd.smooth.lowess(A(s.x), A(y), nan_policy="raise")
    xi = s.x.copy()
    xi[0] = np.inf
    for pol in ("propagate", "omit", "raise"):
        with pytest.raises(ValueError, match=r"^lowess\(\): 'x' contains infinity$"):
            sd.smooth.lowess(A(xi), A(s.y), nan_policy=pol)


def test_errors():
    x, y = A(np.arange(5.0)), A(np.arange(5.0) ** 2)
    L = sd.smooth.lowess
    cases = [
        (TypeError, r"^lowess\(\): 'frac' must be an int or a float, not str$", dict(frac="a")),
        (TypeError, r"^lowess\(\): 'frac' must be an int or a float, not bool$", dict(frac=True)),
        (ValueError, r"^lowess\(\): 'frac' must be in \[0, 1\]$", dict(frac=1.5)),
        (ValueError, r"^lowess\(\): 'frac' must be in \[0, 1\]$", dict(frac=math.nan)),
        (TypeError, r"^lowess\(\): 'it' must be an int, not float$", dict(it=1.0)),
        (ValueError, r"^lowess\(\): 'it' must be >= 0$", dict(it=-1)),
        (ValueError, r"^lowess\(\): 'delta' must be a finite number >= 0$", dict(delta=-0.1)),
        (ValueError, r"^lowess\(\): 'delta' must be a finite number >= 0$", dict(delta=math.inf)),
        (ValueError, r"^lowess\(\): 'nan_policy' must be 'propagate', 'omit' or 'raise', not 'drop'$",
         dict(nan_policy="drop")),
        (ValueError, r"^lowess\(\): 'weights' must be >= 0$", dict(weights=A([1.0, -1, 1, 1, 1]))),
        (ValueError, r"^lowess\(\): at least one weight must be positive$", dict(weights=A(np.zeros(5)))),
        (ValueError, r"^lowess\(\): 'weights' must have one value per point \(5\), not 2$", dict(weights=A([1.0, 2]))),
    ]  # fmt: skip
    for exc, msg, kw in cases:
        with pytest.raises(exc, match=msg):
            L(x, y, **kw)
    with pytest.raises(ValueError, match=r"^lowess\(\): 'x' and 'y' must have the same length, not 5 and 4$"):
        L(x, A([1.0, 2, 3, 4]))
    with pytest.raises(ValueError, match=r"^lowess\(\): 'x' must be 1-D, not 2-D$"):
        L(A(np.ones((2, 2))), A(np.ones(2)))
    with pytest.raises(TypeError, match=r"^lowess\(\): 'y' must be real, not complex128$"):
        L(x, sd.asarray([1j, 2, 3, 4, 5]))
    with pytest.raises(ValueError, match=r"^lowess\(\): 'x' and 'y' must have at least 1 point$"):
        L(A([]), A([]))
    with pytest.raises(TypeError, match=r"^lowess\(\): takes 2 positional arguments but 3 were given$"):
        L(x, y, 0.5)
    with pytest.raises(TypeError, match=r"^lowess\(\): unexpected keyword argument 'xvals'$"):
        L(x, y, xvals=x)
    with pytest.raises(ValueError, match=r"^lowess\(\): the range of 'x' overflows$"):
        L(A([-1.5e308, 0.0, 1.5e308]), A([1.0, 2.0, 3.0]))
    r = L(x, y)
    with pytest.raises(ValueError, match=r"^predict_interval\(\): 'alpha' must be between 0 and 1 \(exclusive\)$"):
        r.predict_interval(x, alpha=1.0)
    with pytest.raises(TypeError, match=r"^predict\(\): 'x_new' must be real, not complex$"):
        r.predict(1j)


def test_inputs_of_every_kind():
    s = S.scatter("random/30")
    base = sd.smooth.lowess(A(s.x), A(s.y))
    for xx, yy in ((s.x, s.y), (s.x.tolist(), s.y.tolist()), (sd.frame.Series(s.x), sd.frame.Series(s.y))):
        r = sd.smooth.lowess(xx, yy)
        assert bits_equal(r.fittedvalues, base.fittedvalues)
    ints = sd.smooth.lowess(np.arange(10), np.arange(10) % 3)
    assert ints.fittedvalues.dtype == sd.float64


def test_result_fields_and_shapes():
    s = S.scatter("random/50")
    r = sd.smooth.lowess(A(s.x), A(s.y))
    assert isinstance(r, sd.smooth.LowessResult)
    assert (r.frac, r.it, r.delta, r.k, r.nobs) == (2 / 3, 3, 0.0, 33, 50)
    assert bits_equal(r.resid, A(s.y) - r.fittedvalues)
    assert 2 < r.df < 10 and r.scale > 0
    assert r.scale == pytest.approx(math.sqrt(float(np.sum(N(r.resid) ** 2)) / (50 - r.df)), rel=1e-14)
    with pytest.raises(ValueError, match="read-only"):
        r.fittedvalues[0] = 1.0
    with pytest.raises(AttributeError, match="read-only"):
        r.frac = 0.5
    assert isinstance(r.predict(5.0), float)
    assert N(r.predict(A(np.ones((2, 3))))).shape == (2, 3)
    pi = N(r.predict_interval(A(np.linspace(0, 10, 7))))
    assert pi.shape == (7, 2) and np.all(pi[:, 0] < pi[:, 1])
    assert N(r.predict_interval(5.0)).shape == (2,)
    mid = N(r.predict(A(np.linspace(0, 10, 7))))
    assert np.allclose(pi.mean(axis=1), mid, rtol=0, atol=1e-14)
    assert repr(r) == f"LowessResult(nobs=50, df={r.df:.6g}, frac=0.666667, it=3)"
    text = r.summary()
    assert "LOWESS" in text and "Window points:" in text and "33" in text


def test_predict_interval_is_the_hat_row_norm():
    """half width = t(nobs - df) scale ||l||, ||l||^2 = sum l_j^2 / w_j (the hat row of the local line at x0)."""
    s = S.scatter("weights/30")
    r = fit(s, 0.67, it=0)
    x0 = 4.2
    # the hat row: the fit of y = e_j at x0
    lrow = np.empty(s.n)
    for j in range(s.n):
        e = np.zeros(s.n)
        e[j] = 1.0
        lrow[j] = sd.smooth.lowess(A(s.x), A(e), frac=0.67, it=0, weights=A(s.w)).predict(x0)
    pos = s.w > 0
    norm = math.sqrt(float(np.sum(lrow[pos] ** 2 / s.w[pos])))
    import scipy.stats

    q = scipy.stats.t.isf(0.025, r.nobs - r.df)
    lo, hi = N(r.predict_interval(x0))
    assert (hi - lo) / 2 == pytest.approx(q * r.scale * norm, rel=1e-10)


@LIVE
def test_live_statsmodels_on_a_large_sample():
    from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

    g = np.random.default_rng(11)
    x = np.sort(g.random(3000)) * 10
    y = np.sin(x) + 0.3 * g.standard_normal(3000)
    for frac, it, delta in ((2 / 3, 3, 0.0), (0.1, 0, 0.0), (0.3, 2, 0.1)):
        want = sm_lowess(y, x, frac=frac, it=it, delta=delta, return_sorted=False)
        got = N(sd.smooth.lowess(A(x), A(y), frac=frac, it=it, delta=delta).fittedvalues)
        assert np.max(np.abs(got - want)) < 1e-11, (frac, it, delta)
