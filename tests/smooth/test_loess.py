"""loess and LoessResult (docs/api/smooth.md §2, docs/design/smooth.md §2.2, §3.2, §4.1; decisions SM10, SM16, SM28,
SM29, SM95-SM111): the reference contract of the fitted values, predict and the leverages in one variable
(tests/smooth/ref/local.npz) and several (local_nd.npz), R 4.6.2's direct computation (surface='direct',
statistics='exact') where it computes the same estimator and the documented differences where it does not, the bits of
lowess at p = 1 and degree 1, the statistics against the smoother matrix built column by column, R's robust weights,
the independence of the input order, the thread count and the neighbour search's path, rank deficiency, the input
rules and the result object."""

import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S

LOCAL = refdata.ref("local")
LOCAL_ND = refdata.ref("local_nd")
R_REC = refdata.data("r")
U = refdata.U
_core = sd._core.smooth


def A(v):
    return sd.asarray(np.asarray(v, dtype=float))


def N(a):
    return np.asarray(a.tolist() if isinstance(a, sd.ndarray) else a, dtype=float)


def per_set(r, method, sid, kind):
    pos, idx = refdata.at(r, method, sid, kind)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def tuning(method):
    """(degree, span) of a reference method loess_d<degree>_s<span>."""
    return int(method[7]), float(method.split("_s")[1])


def smoother_matrix(X, y_len, **kw):
    """L of a gaussian loess, column by column (the fit is linear in y), and the fits' result for y = e_0."""
    cols = []
    for j in range(y_len):
        e = np.zeros(y_len)
        e[j] = 1.0
        cols.append(N(sd.smooth.loess(A(X), A(e), **kw).fittedvalues))
    return np.stack(cols, axis=1)


# ---- the reference contract ---------------------------------------------------------------------------------------

# The leverage at p = 1, degree 1 is lowess's kernel (decision SM46): on clustered/40 with q = 30 the window's far
# edge sits 1e-8 inside the radius, so the rounded differences x - x0 move those weights by 1e-7 relative and the
# hat diagonal by up to 1.73 C_LOCAL kappa u (docs/design/smooth.md §4.7; the same bits as lowess(frac=0.75)).
LOWESS_KERNEL_LEVERAGE = {("clustered/40", "loess_d1_s0.75"): 2.0}
# The contract's windows (decision SM107): above kappa = 2^36 (a quadratic extrapolated from clusters 1e-8 wide) the
# one rounding of each coordinate x - x0 moves the exact fit of the rounded coordinates by about kappa^2 u, more than
# kappa u sum |l_j||y_j|; there the error is held to kappa^2 u sum |l_j||y_j|.
KAPPA_MAX = 2.0**36


def test_one_variable_meets_the_contract():
    """|got - ref| <= C_LOCAL kappa u sum |l_j||y_j| at every data point and evaluation point of local.npz (every
    loess method: degree 0, 1, 2, span 0.5, 0.75, 1.5), the leverage within C_LOCAL kappa u; where no point of the
    window has a positive weight the reference has no value."""
    worst = {"fit": 0.0, "predict": 0.0, "leverage": 0.0}
    seen = beyond = 0
    methods = [m for m in LOCAL.methods if m.startswith("loess")]
    for sid in sorted(set(LOCAL.inputs["set"].tolist())):
        s = S.scatter(sid)
        w = None if s.w is None else A(s.w)
        for method in methods:
            degree, span = tuning(method)
            q = min(s.n, math.floor(s.n * span + 1e-5))
            if q < degree + 1:
                with pytest.raises(ValueError, match="'span' is too small"):
                    sd.smooth.loess(A(s.x), A(s.y), span=span, degree=degree, weights=w)
                continue
            r = sd.smooth.loess(A(s.x), A(s.y), span=span, degree=degree, weights=w)
            got = {0: N(r.fittedvalues), 1: N(r.predict(A(s.xe)))}
            lev = N(r.leverage)
            for kind in (0, 1):
                idx, hi, lo, f = per_set(LOCAL, method, sid, kind)
                for j, i in enumerate(idx):
                    if math.isnan(hi[j]):
                        continue
                    if f["kappa"][j] > KAPPA_MAX:  # the coordinates' rounding, times kappa^2 (decision SM107)
                        err = abs(got[kind][i] - hi[j])
                        assert err <= f["kappa"][j] ** 2 * U * f["scale"][j], (sid, method, kind, i)
                        beyond += 1
                        continue
                    u = refdata.units(got[kind][i], hi[j], lo[j], refdata.local_bound(f["scale"][j], f["kappa"][j]))
                    key = "fit" if kind == 0 else "predict"
                    worst[key] = max(worst[key], u)
                    assert u <= 1, (sid, method, kind, i, u)
                    if kind == 0:
                        lb = refdata.C_LOCAL * f["kappa"][j] * U
                        lu = 0.0 if lev[i] == f["leverage"][j] else abs(lev[i] - f["leverage"][j]) / lb
                        limit = LOWESS_KERNEL_LEVERAGE.get((sid, method), 1.0)
                        assert lu <= limit, (sid, method, i, lu)
                        if limit == 1.0:
                            worst["leverage"] = max(worst["leverage"], lu)
                    seen += 1
    assert seen > 3000 and 0 < beyond < 200, (seen, beyond)


def test_several_variables_meet_the_contract():
    """local_nd.npz (normalize off; the windows of full rank): the same bound with kappa the weighted design's
    2-norm condition."""
    seen = 0
    for sid in sorted(set(LOCAL_ND.inputs["set"].tolist())):
        s = S.multi(sid)
        for method in LOCAL_ND.methods:
            degree, span = tuning(method)
            r = sd.smooth.loess(A(s.X), A(s.y), span=span, degree=degree, normalize=False)
            got = {0: N(r.fittedvalues), 1: N(r.predict(A(s.Xe)))}
            lev = N(r.leverage)
            for kind in (0, 1):
                idx, hi, lo, f = per_set(LOCAL_ND, method, sid, kind)
                for j, i in enumerate(idx):
                    if math.isnan(hi[j]):
                        continue
                    u = refdata.units(got[kind][i], hi[j], lo[j], refdata.local_bound(f["scale"][j], f["kappa"][j]))
                    assert u <= 1, (sid, method, kind, i, u)
                    if kind == 0:
                        lb = refdata.C_LOCAL * f["kappa"][j] * U
                        assert abs(lev[i] - f["leverage"][j]) <= lb, (sid, method, i)
                    seen += 1
    assert seen > 800, seen


# ---- R's direct computation ---------------------------------------------------------------------------------------


def r_records(kind):
    """(key, set id, degree, span, family or normalize, values) of R's loess records (tests/smooth/data/r.json)."""
    for key, rec in R_REC.items():
        parts = key.split("/")
        if parts[0] != "loess":
            continue
        vals = np.array(rec["values"], dtype=float)
        if np.isnan(vals).all():  # R stopped (its weights at y near 1e+-300)
            continue
        sid = f"{parts[1]}/{parts[2]}"
        degree, span = int(parts[3][1]), float(parts[3].split("_s")[1])
        if kind == 1 and parts[4] in ("gaussian", "symmetric"):
            yield key, sid, degree, span, parts[4], vals
        if kind == 2 and parts[4].startswith("normalize"):
            yield key, sid, degree, span, parts[4] == "normalizeTRUE", vals


def close(a, b, scale, tol):
    return np.all(np.abs(np.asarray(a) - np.asarray(b)) <= tol * scale)


def test_r_direct_gaussian_is_the_same_estimator():
    """R's loess(surface='direct', statistics='exact'), gaussian: the fitted values, predict, trace.hat (df), enp,
    one.delta, two.delta and s within 1e-13 of the fit's scale (the same local least squares, rounded differently),
    outside clustered/40, whose extrapolated windows (kappa up to 1e13) both round like the exact fit of rounded
    coordinates (below)."""
    seen = 0
    for key, sid, degree, span, family, vals in r_records(1):
        if family != "gaussian" or sid == "clustered/40":
            continue
        s = S.scatter(sid)
        r = sd.smooth.loess(A(s.x), A(s.y), span=span, degree=degree)
        n = s.n
        sc = max(np.abs(vals[5 : 5 + n]).max(), 1e-300)
        assert close(N(r.fittedvalues), vals[5 : 5 + n], sc, 1e-13), key
        pe = vals[5 + n :]
        got = N(r.predict(A(s.xe)))
        assert np.array_equal(np.isnan(got), np.isnan(pe)) and close(got[~np.isnan(pe)], pe[~np.isnan(pe)], sc, 1e-13)
        enp, d1, d2, rs, trace = vals[:5]
        for g, ref in ((r.enp, enp), (r.one_delta, d1), (r.two_delta, d2), (r.df, trace)):
            assert abs(g - ref) <= 1e-12 * max(1.0, abs(ref)), (key, g, ref)
        ys = max(np.abs(s.y).max(), 1e-300)
        assert r.scale == rs or abs(r.scale - rs) <= 1e-12 * rs + 1e-14 * ys, (key, r.scale, rs)
        seen += 1
    assert seen >= 60


def test_r_direct_symmetric_fits_are_the_same():
    """family='symmetric' with R's robust weights (lowesw: cuts at 0.001 and 0.999 of 6 median |r|): the same fitted
    values and predict as R. Its statistics differ by design: R keeps trace.hat, delta1 and delta2 of the first
    (gaussian) fit and takes s from pseudovalues; sd's are those of the last fit's smoother (decision SM101)."""
    seen = differ = 0
    for key, sid, degree, span, family, vals in r_records(1):
        if family != "symmetric" or sid == "clustered/40":
            continue
        s = S.scatter(sid)
        r = sd.smooth.loess(A(s.x), A(s.y), span=span, degree=degree, family="symmetric")
        n = s.n
        sc = max(np.abs(vals[5 : 5 + n]).max(), 1e-300)
        assert close(N(r.fittedvalues), vals[5 : 5 + n], sc, 1e-12), key
        pe = vals[5 + n :]
        got = N(r.predict(A(s.xe)))
        assert close(got[~np.isnan(pe)], pe[~np.isnan(pe)], sc, 1e-12), key
        differ += abs(r.df - vals[4]) > 1e-6
        seen += 1
    assert seen >= 60 and differ >= 10


def test_r_direct_several_variables():
    """R's records with two and three variables, normalize TRUE and FALSE: the fitted values and the statistics."""
    seen = 0
    for key, sid, degree, span, normalize, vals in r_records(2):
        s = S.multi(sid)
        r = sd.smooth.loess(A(s.X), A(s.y), span=span, degree=degree, normalize=normalize)
        n = s.n
        sc = np.abs(vals[5 : 5 + n]).max()
        # R's pseudo-inverse and sd's dropped columns: both interpolate the line
        if sid == "lowrank2/40" and degree > 0:
            assert r.n_rank_deficient > 0
            continue
        assert close(N(r.fittedvalues), vals[5 : 5 + n], sc, 1e-12), key
        for g, ref in ((r.enp, vals[0]), (r.one_delta, vals[1]), (r.two_delta, vals[2]), (r.df, vals[4])):
            assert abs(g - ref) <= 1e-11 * max(1.0, abs(ref)), (key, g, ref)
        seen += 1
    assert seen >= 24


def test_extrapolation_from_clusters_rounds_like_r():
    """clustered/40 (four clusters 1e-8 wide), a quadratic extrapolated at the evaluation points (kappa 5e11-9e12,
    beyond 2^36): the one rounding of each coordinate x - x0 moves the exact fit by about kappa^2 u, and R's direct
    computation is off by the same order (decision SM107); sd stays within kappa^2 u sum |l_j||y_j|."""
    s = S.scatter("clustered/40")
    vals = np.array(R_REC["loess/clustered/40/d2_s0.5/gaussian"]["values"])
    got = N(sd.smooth.loess(A(s.x), A(s.y), span=0.5, degree=2).predict(A(s.xe)))
    pe = vals[5 + s.n :]
    idx, hi, _, f = per_set(LOCAL, "loess_d2_s0.5", "clustered/40", 1)
    assert np.all(f["kappa"] > KAPPA_MAX)
    bound = refdata.C_LOCAL * f["kappa"] * U * f["scale"]
    sd_units, r_units = np.abs(got[idx] - hi) / bound, np.abs(pe[idx] - hi) / bound
    assert 1 < sd_units.max() < 10 and 1 < r_units.max() < 10
    assert np.all(np.abs(got[idx] - hi) <= f["kappa"] ** 2 * U * f["scale"])


# ---- lowess's bits, the statistics, R's robust weights -----------------------------------------------------------


def test_one_variable_degree_one_is_lowess():
    """p = 1, degree 1, gaussian: lowess's kernel, the same bits as lowess(frac=span, it=0) wherever the window has
    the same points (q = k) and two positive weights (decision SM106)."""
    seen = 0
    for sid in S.scatter_ids():
        s = S.scatter(sid)
        w = None if s.w is None else A(s.w)
        for span in (0.3, 0.5, 2 / 3, 0.75, 1.0):
            q = min(s.n, math.floor(s.n * span + 1e-5))
            k = min(s.n, max(2, int(span * s.n + 1e-10)))
            if q != k or q < 2:
                continue
            a = sd.smooth.loess(A(s.x), A(s.y), span=span, degree=1, weights=w)
            b = sd.smooth.lowess(A(s.x), A(s.y), frac=span, it=0, weights=w)
            fa, fb = N(a.fittedvalues), N(b.fittedvalues)
            keep = np.ones(s.n, bool) if s.w is None else s.w > 0
            assert bits_equal(A(fa[keep]), A(fb[keep])), (sid, span)
            assert bits_equal(a.leverage, b.leverage), (sid, span)
            assert a.df == b.df
            pa, pb = N(a.predict(A(s.xe))), N(b.predict(A(s.xe)))
            both = ~np.isnan(pb)
            assert bits_equal(A(pa[both]), A(pb[both])), (sid, span)
            seen += 1
    assert seen >= 40


@pytest.mark.parametrize(
    ("kind", "sid", "kw"),
    [
        ("scatter", "random/30", {"degree": 2}),
        ("scatter", "weights/30", {"degree": 2, "span": 0.5}),
        ("scatter", "ties/30", {"degree": 0}),
        ("scatter", "random/30", {"degree": 1, "span": 1.5}),
        ("multi", "uniform2/60", {"degree": 2}),
        ("multi", "scales2/50", {"degree": 1, "span": 0.5}),
    ],
)
def test_statistics_are_those_of_the_smoother_matrix(kind, sid, kw):
    """df = tr L, leverage = diag L, one_delta = tr((I - L)^T (I - L)), two_delta = tr(((I - L)^T (I - L))^2),
    enp = one_delta + 2 df - nobs (R's), scale = sqrt(sum w r^2 / one_delta), over the points of positive weight;
    predict_interval's ||l|| is the row of the smoother at the point (sum l_j^2 / w_j)."""
    s = S.scatter(sid) if kind == "scatter" else S.multi(sid)
    X = s.x if kind == "scatter" else s.X
    Xe = s.xe if kind == "scatter" else s.Xe
    w = getattr(s, "w", None)
    kw = dict(kw, weights=None if w is None else A(w))
    r = sd.smooth.loess(A(X), A(s.y), **kw)
    L = smoother_matrix(X, s.n, **kw)
    pos = np.ones(s.n, bool) if w is None else w > 0
    Lp = L[np.ix_(pos, pos)]
    I = np.eye(pos.sum())
    B = (I - Lp).T @ (I - Lp)
    lev = N(r.leverage)
    assert np.allclose(lev[pos], np.diag(L)[pos], rtol=0, atol=1e-13)
    assert np.all(lev[~pos] == 0)
    assert abs(r.df - np.trace(Lp)) <= 1e-12 * s.n
    assert abs(r.one_delta - np.trace(B)) <= 1e-12 * s.n
    assert abs(r.two_delta - np.trace(B @ B)) <= 1e-11 * s.n
    assert abs(r.enp - (r.one_delta + 2 * r.df - r.nobs)) <= 1e-12 * s.n
    rr = (s.y - N(r.fittedvalues))[pos]
    wp = np.ones(pos.sum()) if w is None else w[pos]
    assert math.isclose(r.scale, math.sqrt(np.sum(wp * rr * rr) / r.one_delta), rel_tol=1e-13)
    # the interval's row at new points: predict's linearity in y
    rows = []
    for j in range(s.n):
        e = np.zeros(s.n)
        e[j] = 1.0
        rows.append(N(sd.smooth.loess(A(X), A(e), **kw).predict(A(Xe))))
    Le = np.stack(rows, axis=1)[:, pos]
    norm = np.sqrt(np.sum(Le * Le / wp, axis=1))
    iv = N(r.predict_interval(A(Xe), alpha=0.1))
    half = (iv[:, 1] - iv[:, 0]) / 2
    from scipy.stats import t as student

    tq = student.ppf(0.95, r.one_delta**2 / r.two_delta)
    assert np.allclose(half, tq * r.scale * norm, rtol=1e-11, atol=0)
    assert np.allclose((iv[:, 0] + iv[:, 1]) / 2, N(r.predict(A(Xe))), rtol=1e-14, atol=1e-14)


def lowesw(r):
    """R's lowesw (decision SM100)."""
    a = np.abs(r)
    srt = np.sort(a)
    c = len(a)
    cmad = 3.0 * (srt[c // 2] + srt[c // 2 - 1]) if c % 2 == 0 else 6.0 * srt[c // 2]
    if cmad < np.finfo(float).tiny:
        return np.ones_like(a)
    u = a / cmad
    return np.where(cmad * 0.999 < a, 0.0, np.where(cmad * 0.001 < a, (1 - u * u) ** 2, 1.0))


@pytest.mark.parametrize("sid", ["outliers/40", "halfzero/21", "random/50", "step/40"])
def test_symmetric_is_r_robust_weights_of_the_previous_fit(sid):
    s = S.scatter(sid)
    for it in (2, 4):
        last = sd.smooth.loess(A(s.x), A(s.y), family="symmetric", iterations=it)
        prev = sd.smooth.loess(A(s.x), A(s.y), family="symmetric", iterations=it - 1)
        assert np.allclose(N(last.robust_weights), lowesw(N(prev.resid)), rtol=0, atol=1e-12)
    g = sd.smooth.loess(A(s.x), A(s.y))
    assert np.all(N(g.robust_weights) == 1.0)
    one = sd.smooth.loess(A(s.x), A(s.y), family="symmetric", iterations=1)
    assert bits_equal(one.fittedvalues, g.fittedvalues)


# ---- determinism ---------------------------------------------------------------------------------------------------


def big(n=3000, p=2, seed=5):
    g = np.random.default_rng(seed)
    X = g.random((n, p)) * 10 if p > 1 else np.sort(g.random(n) * 10)
    y = np.sin(X[:, 0] if p > 1 else X) + 0.3 * g.standard_normal(n)
    w = 0.5 + g.random(n)
    w[::17] = 0.0
    return X, y, w


def test_thread_counts_give_the_same_bytes(restore_threads):
    X1, y1, w1 = big(4000, 1)
    X2, y2, w2 = big(2500, 2)
    g = np.random.default_rng(9)
    q1, q2 = g.random(3000) * 12 - 1, g.random((1500, 2)) * 12 - 1
    out = {}
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        res = []
        for X, y, _w, q, kws in (
            (X1, y1, w1, q1, ({"degree": 2}, {"degree": 1, "family": "symmetric", "weights": A(w1)}, {"span": 0.1})),
            (X2, y2, w2, q2, ({"degree": 2}, {"degree": 1, "span": 0.05, "weights": A(w2)}, {"family": "symmetric"})),
        ):
            for kw in kws:
                r = sd.smooth.loess(A(X), A(y), **kw)
                res += [r.fittedvalues, r.leverage, r.robust_weights, A([r.df, r.scale, r.one_delta])]
                res += [r.predict(A(q)), r.predict_interval(A(q))]
        r = sd.smooth.loess(A(X2[:700]), A(y2[:700]), degree=1)
        res.append(A([r.two_delta]))
        out[t] = res
    for t in (3, 10):
        for k, (a, b) in enumerate(zip(out[1], out[t])):
            assert bits_equal(a, b), (t, k)


def test_tree_and_scan_find_the_same_window():
    """p >= 2: the KD tree's k nearest (small spans) and the scan of every distance give the same radius and the
    same window, so the same bits (decision SM96)."""
    for X, y, w in (big(1500, 2), big(800, 3, seed=7)):
        n = len(y)
        Xa = A(X)
        for span in (0.02, 0.1, 0.6):
            q = math.floor(n * span + 1e-5)
            outs = [_core.loess_fit(Xa, A(y), A(w), 1, q, 1.0, 1, False, path) for path in (0, 1)]
            for a, b in zip(outs[0], outs[1]):
                if isinstance(a, sd.ndarray):
                    assert bits_equal(a, b), span
                else:
                    assert a == b or (a is None and b is None)
    s = S.multi("lattice2/49")  # many equal distances
    outs = [_core.loess_fit(A(s.X), A(s.y), None, 2, 9, 1.0, 1, False, path) for path in (0, 1)]
    assert bits_equal(outs[0][0], outs[1][0])


def test_scan_radius_beyond_the_direct_selection():
    """Above 2048 points the scan takes the q-th distance from a sampled bracket (decision SM138): the same radius
    and window as the KD tree's k nearest, so the same bits, also with many equal distances."""
    X, y, w = big(5000, 2, seed=9)
    Xa = A(X)
    for span in (0.003, 0.05, 0.5, 1.0):
        q = math.floor(len(y) * span + 1e-5)
        outs = [_core.loess_fit(Xa, A(y), A(w), 1, q, 1.0, 1, False, path) for path in (0, 1)]
        assert bits_equal(outs[0][0], outs[1][0]) and bits_equal(outs[0][1], outs[1][1]), span
    g = np.random.default_rng(3)
    L = A(np.round(g.random((4000, 2)) * 20.0))  # a lattice: ties at the radius
    yl = A(g.standard_normal(4000))
    outs = [_core.loess_fit(L, yl, None, 1, 37, 1.0, 1, False, path) for path in (0, 1)]
    assert bits_equal(outs[0][0], outs[1][0])


def test_result_is_a_function_of_the_data():
    g = np.random.default_rng(4)
    for X, y, _ in (big(300, 1), big(300, 2), big(200, 3)):
        perm = g.permutation(len(y))
        for kw in ({"degree": 2}, {"degree": 1, "family": "symmetric"}, {"degree": 0, "span": 0.3}):
            a = sd.smooth.loess(A(X), A(y), **kw)
            b = sd.smooth.loess(A(X[perm]), A(y[perm]), **kw)
            assert bits_equal(A(N(a.fittedvalues)[perm]), b.fittedvalues)
            assert bits_equal(A(N(a.leverage)[perm]), b.leverage)
            assert a.df == b.df and a.one_delta == b.one_delta


def test_power_of_two_scaling_gives_the_same_bits():
    for X, y, _ in (big(200, 1), big(200, 2)):
        a = sd.smooth.loess(A(X), A(y), normalize=False)
        b = sd.smooth.loess(A(X * 2.0**-40), A(y), normalize=False)
        assert bits_equal(a.fittedvalues, b.fittedvalues) and a.df == b.df
    s = S.scatter("subnormal/12")
    r = sd.smooth.loess(A(s.x), A(s.y), span=1.5, degree=1)
    t = sd.smooth.loess(A(s.x * 2.0**1000), A(s.y), span=1.5, degree=1)
    assert bits_equal(r.fittedvalues, t.fittedvalues)


# ---- degenerate windows ----------------------------------------------------------------------------------------------


def test_rank_deficient_windows_lower_the_fit():
    s = S.multi("lowrank2/40")  # every point on one line
    r = sd.smooth.loess(A(s.X), A(s.y), degree=2, normalize=False)
    assert r.n_rank_deficient == s.n and np.all(np.isfinite(N(r.fittedvalues)))
    u = s.X[:, 0]  # along the line the fit is loess in one variable of degree 2 with the same window
    one = sd.smooth.loess(A(u), A(s.y), degree=2)
    assert np.allclose(N(r.fittedvalues), N(one.fittedvalues), rtol=0, atol=1e-9)
    t = S.scatter("ties/30")  # x integers: windows with two distinct x drop to a line
    r = sd.smooth.loess(A(t.x), A(t.y), degree=2, span=0.2)
    assert r.n_rank_deficient > 0 and np.all(np.isfinite(N(r.fittedvalues)))
    a = S.scatter("allsame/8")
    r = sd.smooth.loess(A(a.x), A(a.y), degree=2)
    assert np.allclose(N(r.fittedvalues), a.y.mean(), rtol=1e-15) and r.n_rank_deficient == 8
    assert r.predict(2.5) == r.fittedvalues[0] and math.isnan(r.predict(3.0))  # every point at the radius


def test_exact_polynomials_are_reproduced():
    x = np.arange(20) / 8.0
    for degree, y in ((0, np.full(20, 0.75)), (1, 2 * x + 1), (2, x * x - 3 * x + 2)):
        r = sd.smooth.loess(A(x), A(y), degree=degree)
        assert np.allclose(N(r.fittedvalues), y, rtol=0, atol=1e-13)
    g = np.random.default_rng(2)
    X = g.random((60, 3))
    y = 1 + X @ np.array([2.0, -1.0, 0.5]) + X[:, 0] * X[:, 2]
    r = sd.smooth.loess(A(X), A(y), degree=2, span=0.5)
    assert np.allclose(N(r.fittedvalues), y, rtol=0, atol=1e-11)


def test_span_and_windows():
    s = S.scatter("random/30")
    r = sd.smooth.loess(A(s.x), A(s.y), span=2.0)
    assert r.q == 30 and r.span == 2.0
    assert sd.smooth.loess(A(s.x), A(s.y), span=0.1, degree=2).q == 3
    with pytest.raises(ValueError, match="fewer than the 3 terms"):
        sd.smooth.loess(A(s.x), A(s.y), span=0.06)
    with pytest.raises(ValueError, match="fewer than the 6 terms"):
        m = S.multi("uniform2/60")
        sd.smooth.loess(A(m.X), A(m.y), span=0.05)


# ---- inputs and results ---------------------------------------------------------------------------------------------


def test_nan_policy():
    s = S.scatter("random/30")
    x, y = s.x.copy(), s.y.copy()
    x[4] = np.nan
    r = sd.smooth.loess(A(x), A(y))
    assert np.all(np.isnan(N(r.fittedvalues))) and math.isnan(r.df) and math.isnan(r.predict(1.0))
    assert np.all(np.isnan(N(r.predict_interval(A([1.0, 2.0])))))
    o = sd.smooth.loess(A(x), A(y), nan_policy="omit")
    ref = sd.smooth.loess(A(np.delete(x, 4)), A(np.delete(y, 4)))
    got = N(o.fittedvalues)
    assert math.isnan(got[4]) and bits_equal(A(np.delete(got, 4)), ref.fittedvalues) and o.nobs == 29
    with pytest.raises(ValueError, match="NaN"):
        sd.smooth.loess(A(x), A(y), nan_policy="raise")
    m = S.multi("uniform2/60")
    X = m.X.copy()
    X[3, 1] = np.nan
    o = sd.smooth.loess(A(X), A(m.y), nan_policy="omit")
    assert math.isnan(N(o.fittedvalues)[3]) and o.nobs == 59


def test_errors():
    x, y = A(np.arange(12.0)), A(np.sin(np.arange(12.0)))
    for kw, exc in (
        ({"span": 0.0}, ValueError),
        ({"span": math.inf}, ValueError),
        ({"span": "a"}, TypeError),
        ({"degree": 3}, ValueError),
        ({"degree": 1.0}, TypeError),
        ({"family": "robust"}, ValueError),
        ({"iterations": 0}, ValueError),
        ({"normalize": 1}, TypeError),
        ({"nan_policy": "drop"}, ValueError),
        ({"weights": A(-np.ones(12))}, ValueError),
    ):
        with pytest.raises(exc):
            sd.smooth.loess(x, y, **kw)
    with pytest.raises(ValueError, match="same number"):
        sd.smooth.loess(A(np.arange(5.0)), A(np.arange(4.0)))
    with pytest.raises(ValueError, match="infinity"):
        sd.smooth.loess(A([0.0, math.inf, 2.0, 3.0]), A([1.0, 2.0, 3.0, 4.0]))
    with pytest.raises(ValueError, match="overflows"):
        sd.smooth.loess(A([-1.5e308, 1.5e308, 0.0, 1.0]), A([1.0, 2.0, 3.0, 4.0]), degree=0)
    with pytest.raises(ValueError, match="no spread"):
        X = np.c_[np.arange(20.0), np.r_[0.0, np.ones(18), 2.0]]
        sd.smooth.loess(A(X), A(np.arange(20.0)))
    m = S.multi("uniform2/60")
    r = sd.smooth.loess(A(m.X), A(m.y))
    with pytest.raises(ValueError, match="2 columns"):
        r.predict(1.0)
    with pytest.raises(ValueError):
        r.predict(A(np.ones((3, 3))))
    with pytest.raises(ValueError, match="alpha"):
        r.predict_interval(A(m.Xe), alpha=1.5)


def test_inputs_of_every_kind():
    s = S.scatter("random/30")
    ref = sd.smooth.loess(A(s.x), A(s.y))
    for x, y in ((s.x, s.y), (s.x.tolist(), s.y.tolist()), (sd.frame.Series(s.x), sd.frame.Series(s.y))):
        assert bits_equal(sd.smooth.loess(x, y).fittedvalues, ref.fittedvalues)
    m = S.multi("uniform2/60")
    ref = sd.smooth.loess(A(m.X), A(m.y))
    df = sd.frame.DataFrame({"a": m.X[:, 0], "b": m.X[:, 1]})
    assert bits_equal(sd.smooth.loess(df, m.y).fittedvalues, ref.fittedvalues)
    col = sd.smooth.loess(A(s.x[:, None]), A(s.y))  # (n, 1) is one variable given as 2-D
    assert bits_equal(col.fittedvalues, sd.smooth.loess(A(s.x), A(s.y)).fittedvalues)
    assert N(col.predict(A([[1.0], [2.0]]))).shape == (2,)


def test_result_fields_and_shapes():
    s = S.scatter("weights/30")
    r = sd.smooth.loess(A(s.x), A(s.y), weights=A(s.w), family="symmetric")
    assert r.nobs == 27 and r.q == 22 and r.degree == 2 and r.family == "symmetric" and r.span == 0.75
    assert isinstance(r.n_rank_deficient, int)
    assert N(r.fittedvalues).shape == N(r.resid).shape == N(r.leverage).shape == N(r.robust_weights).shape == (30,)
    assert isinstance(r.predict(2.0), float)
    assert N(r.predict(A(np.ones((2, 3))))).shape == (2, 3)
    assert N(r.predict_interval(A(np.ones((2, 3))))).shape == (2, 3, 2)
    assert repr(r).startswith("LoessResult(nobs=27, df=")
    text = r.summary()
    assert "Loess Results" in text and "Equivalent parameters" in text and "symmetric" in text
    with pytest.raises(AttributeError):
        r.span = 1.0
    m = S.multi("uniform3/60")
    r = sd.smooth.loess(A(m.X), A(m.y))
    assert N(r.predict(A(m.Xe))).shape == (8,) and isinstance(r.predict(A(m.Xe[0])), float)


def test_two_delta_beyond_4096_points_is_not_computed():
    """delta2 needs the dense smoother matrix (O(n^2) memory, O(n^2 q) work): above 4096 points of positive weight it
    is NaN and predict_interval takes the normal quantile (decision SM7)."""
    X, y, _ = big(4200, 1)
    r = sd.smooth.loess(A(X), A(y), span=0.05)
    assert math.isnan(r.two_delta)
    iv = N(r.predict_interval(A([3.0]), alpha=0.05))[0]
    _, norm = _core.loess_predict(*_stored(r), A([[3.0]]), True, -1)
    half = 1.959963984540054 * r.scale * math.sqrt(N(norm)[0])
    assert math.isclose((iv[1] - iv[0]) / 2, half, rel_tol=1e-14)


def _stored(r):
    Xs, ys, ws, rho, divisor, widen = object.__getattribute__(r, "_data")
    return Xs, ys, ws, rho, r.degree, r.q, widen, divisor


@pytest.mark.parametrize(
    "x, kw, rtol",
    [
        ([0.0, 1.0], {"degree": 0}, 1e-14),
        ([0.0, 1.0], {"degree": 0, "weights": [1.0, 2.0]}, 1e-14),
        ([0.0, 1.0, 2.0], {"degree": 1}, 1e-4),
        ([0.0, 1.0, 2.0, 3.0], {"degree": 2}, 1e-4),
    ],
)
def test_one_delta_rows_keep_the_small_entries(x, kw, rtol):
    """span just above 1 gives the far points weights near 2.7e-11: every hat row is e_i plus entries far below
    ulp(1), and one_delta (about 1e-20) is their squares. Each row's sum over j != i is added on its own (decision
    SM143); taken as sum l_j^2 - l_ii^2 it would vanish below ulp(l_ii^2) and one_delta would lose half or more."""
    from fractions import Fraction

    kw = dict(kw, span=1 + 2e-4)
    if "weights" in kw:
        kw["weights"] = A(kw["weights"])
    n = len(x)
    r = sd.smooth.loess(A(x), A([float(i % 2) for i in range(n)]), **kw)
    L = smoother_matrix(x, n, **kw)
    exact = sum(
        (1 - Fraction(L[i, i])) ** 2 + sum(Fraction(L[i, j]) ** 2 for j in range(n) if j != i) for i in range(n)
    )
    assert 0 < float(exact) < 1e-18
    assert math.isclose(r.one_delta, float(exact), rel_tol=rtol)
