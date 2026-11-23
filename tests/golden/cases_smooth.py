"""Golden cases for sd.smooth (tools/golden.py has the contract).

The cases come with the milestones that make the names public (docs/design/smooth.md §8). lowess (SM1
milestone): the sets of tests/smooth/samples.py's kinds rebuilt from literals and L.rng (a grid, random x,
clusters, ties, one x, unsorted, narrow, wide and shifted ranges, subnormal spacing, 1 to 5 points, a line,
outliers, half the residuals zero, values near 1e300, weights with zeros) times frac and it, predict, delta,
predict_interval, every nan_policy, the inputs' kinds, the wrong calls and three cases of 20000 points.

The plan (docs/design/smooth.md §7.5): every public function on the representatives of tests/smooth/samples.py
(a uniform grid, random x, clusters, repeated and unsorted x, the narrow, wide and shifted ranges, outliers,
exact polynomials, weights with zeros; the multivariate sets; the densities with their far tails; the
Savitzky-Golay signals) times each option's values (frac, it, delta; span, degree, family; lam, df, criterion;
bw rules and numbers; kernels, bounds; window, order, deriv, mode), NaN under each nan_policy, the wrong calls
(their errors are results), and cases large enough to split into tasks at 3 and 10 threads (one worker
below 1 GB). Results are the same at 1, 3 and 10 threads (decision SM3), so no case is thread dependent.
"""

import math

NAN, INF = math.nan, math.inf

# Public names no case calls (tools/golden.py): the reason each needs none.
EXCLUDED = {
    "LowessResult": "a result class: the cases of the function that returns it store it whole",
    "LoessResult": "a result class: the cases of the function that returns it store it whole",
    "SmoothingSplineResult": "a result class: the cases of the function that returns it store it whole",
    "PSplineResult": "a result class: the cases of the function that returns it store it whole",
    "KernelRegressionResult": "a result class: the cases of the function that returns it store it whole",
    "KDEResult": "a result class: the cases of the function that returns it store it whole",
    "DensityResult": "a result class: the cases of the function that returns it store it whole",
}


def _scatters(L):
    """lowess's data, rebuilt from literals and L.rng: (name, x, y, weights or None)."""
    np = L.np
    r = L.rng(20261123)
    out = []
    xg = np.linspace(0.0, 10.0, 40)
    out.append(("uniform40", xg, np.sin(xg) + 0.3 * r.standard_normal(40), None))
    xr = np.sort(10.0 * r.random(200))
    yr = np.sin(xr) + 0.3 * r.standard_normal(200)
    out.append(("random200", xr, yr, None))
    xc = np.sort((np.arange(4.0)[:, None] * 2.5 + 1e-8 * r.random((4, 10))).ravel())
    out.append(("clustered40", xc, np.sin(xc) + 0.3 * r.standard_normal(40), None))
    xt = np.round(10.0 * r.random(30))
    out.append(("ties30", xt, np.sin(xt) + 0.3 * r.standard_normal(30), None))
    out.append(("allsame8", np.full(8, 2.5), r.standard_normal(8), None))
    p = r.permutation(200)
    out.append(("unsorted200", xr[p], yr[p], None))
    xs = np.sort(10.0 * r.random(30))
    ys = np.sin(xs) + 0.3 * r.standard_normal(30)
    out += [("narrow30", xs * 1e-8, ys, None), ("wide30", xs * 1e8, ys, None), ("shifted30", xs + 1e8, ys, None)]
    out.append(("subnormal12", np.arange(1.0, 13.0) * 2.0**-1070, r.standard_normal(12), None))
    for n in (1, 2, 3, 5):
        out.append((f"tiny{n}", np.arange(n) * 1.5 + 0.25, r.standard_normal(n), None))
    xd = np.arange(20) / 8.0
    out.append(("line20", xd, 2.0 * xd + 1.0, None))
    yo = np.sin(xr[:40] * 5) + 0.3 * r.standard_normal(40)
    yo[[5, 17, 31]] += (25.0, -40.0, 30.0)
    out.append(("outliers40", xr[::5], yo, None))
    xh = np.arange(21) / 4.0
    yh = 0.5 * xh - 1.0
    yh[::3] += r.standard_normal(7)
    out.append(("halfzero21", xh, yh, None))
    out.append(("big20", xs[:20], ys[:20] * 1e300, None))
    w = 0.5 + 1.5 * r.random(30)
    w[[3, 4, 20]] = 0.0
    out.append(("weights30", xs, ys, w))
    return out


def _lowess_cases(L):
    sd, np = L.sd, L.np
    a = L.arr
    lowess = sd.smooth.lowess
    for name, x, y, w in _scatters(L):
        span = float(np.ptp(x)) if x.size else 0.0
        xe = np.concatenate([np.linspace(x.min() - 0.1 * span - 0.5, x.max() + 0.1 * span + 0.5, 9), x[:3]])
        for frac in (0.3, 2 / 3, 1.0):
            for it in (0, 3):

                def fit(x=x, y=y, w=w, frac=frac, it=it):
                    return lowess(a(x), a(y), frac=frac, it=it, weights=None if w is None else a(w))

                yield f"lowess {name} frac={frac:.3g} it={it}", fit
                yield f"lowess {name} frac={frac:.3g} it={it} predict", lambda fit=fit, xe=xe: fit().predict(a(xe))
        for tag, delta in (("1% range", 0.01 * span), ("25% range", 0.25 * span), ("1e300", 1e300)):
            yield (
                f"lowess {name} delta={tag}",
                lambda x=x, y=y, w=w, delta=delta: lowess(a(x), a(y), delta=delta, weights=None if w is None else a(w)),
            )
        yield (
            f"lowess {name} predict_interval",
            lambda x=x, y=y, w=w, xe=xe: lowess(a(x), a(y), weights=None if w is None else a(w)).predict_interval(
                a(xe), alpha=0.1
            ),
        )
    r = L.rng(7)
    xn = np.sort(10.0 * r.random(50))
    yn = np.sin(xn) + 0.3 * r.standard_normal(50)
    yn[[3, 20]] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield f"lowess nan_policy={pol}", lambda pol=pol: lowess(a(xn), a(yn), nan_policy=pol)
        yield (
            f"lowess nan_policy={pol} predict",
            lambda pol=pol: lowess(a(xn), a(yn), nan_policy="omit").predict(a([NAN, 1.0, 5.0, 20.0])),
        )
    yield "lowess scalar predict", lambda: lowess(a(xn[:30]), a(np.cos(xn[:30]))).predict(2.5)
    yield "lowess list input", lambda: lowess(xn[:30].tolist(), np.cos(xn[:30]).tolist(), frac=0.5)
    yield "lowess int input", lambda: lowess(np.arange(12), np.arange(12) % 4, it=1)
    for kw in (
        {"frac": 1.5}, {"frac": -0.1}, {"frac": "a"}, {"frac": True}, {"it": -1}, {"it": 2.0}, {"delta": -1.0},
        {"delta": INF}, {"nan_policy": "drop"}, {"weights": a(np.zeros(12))}, {"weights": a(-np.ones(12))},
        {"weights": a(np.ones(3))}, {"xvals": a(np.ones(3))},
    ):  # fmt: skip
        yield (
            f"lowess error {sorted(kw)[0]}={kw[sorted(kw)[0]]!r}",
            lambda kw=kw: lowess(a(np.arange(12.0)), a(np.arange(12.0)), **kw),
        )
    yield "lowess error lengths", lambda: lowess(a(np.arange(5.0)), a(np.arange(4.0)))
    yield "lowess error inf", lambda: lowess(a([0.0, INF, 2.0]), a([1.0, 2.0, 3.0]))
    yield "lowess error empty", lambda: lowess(a(np.zeros(0)), a(np.zeros(0)))
    yield "lowess error 2-D", lambda: lowess(a(np.ones((3, 2))), a(np.ones(3)))
    yield "lowess error overflow", lambda: lowess(a([-1.5e308, 1.5e308]), a([1.0, 2.0]))
    # large enough to split into many tasks at 3 and 10 threads
    rb = L.rng(11)
    xb = np.sort(10.0 * rb.random(20000))
    yb = np.sin(xb) + 0.3 * rb.standard_normal(20000)
    wb = 0.5 + rb.random(20000)
    wb[::13] = 0.0
    yield "lowess large", lambda: lowess(a(xb), a(yb), frac=0.2)
    yield "lowess large weights delta", lambda: lowess(a(xb), a(yb), frac=0.3, weights=a(wb), delta=0.002)
    yield (
        "lowess large predict_interval",
        lambda: lowess(a(xb), a(yb), frac=0.1, it=1).predict_interval(a(rb.random(20000) * 12 - 1)),
    )


def _savgol_cases(L):
    """savgol_coeffs and savgol_filter (SM6 milestone): the coefficient cases of tests/smooth/samples.py and
    orders up to the window, both orders and delta; the filter on a cubic, noise, 1e15 plus a walk, a complex
    signal, a signal as long as the window, a NaN, a batch along both axes and a strided 3-D view, every mode with
    even and odd windows and deriv > polyorder; arrays large enough to split into tasks at 3 and 10 threads; the
    wrong calls."""
    sd, np = L.sd, L.np
    a = L.arr
    co, fi = sd.smooth.savgol_coeffs, sd.smooth.savgol_filter
    for n, d, deriv, pos in (
        (5, 2, 0, None), (7, 3, 0, None), (7, 3, 1, None), (7, 3, 2, None), (7, 3, 3, None), (7, 3, 4, None),
        (11, 3, 0, 0), (11, 3, 0, 10), (11, 3, 1, 2), (10, 3, 0, None), (10, 3, 1, None), (4, 3, 0, None),
        (1, 0, 0, None), (3, 2, 0, None), (25, 5, 2, None), (51, 45, 0, None), (101, 10, 3, None), (301, 30, 0, None),
        (64, 63, 0, None), (201, 150, 0, 0), (201, 200, 1, 100), (1001, 12, 1, None),
    ):  # fmt: skip
        for use in ("conv", "dot"):
            yield (
                f"savgol_coeffs {n},{d} deriv={deriv} pos={pos} use={use}",
                lambda n=n, d=d, deriv=deriv, pos=pos, use=use: co(n, d, deriv=deriv, pos=pos, use=use),
            )
    for delta in (0.1, 2.0**-120, -3.0):
        yield f"savgol_coeffs delta={delta!r}", lambda delta=delta: co(11, 4, deriv=3, delta=delta)
    r = L.rng(20261123)
    k = np.arange(40.0)
    signals = [
        ("cubic40", 0.125 * k**3 - 2.0 * k * k + 3.0 * k - 7.0),
        ("noisy100", np.sin(np.arange(100) / 7.0) + 0.2 * r.standard_normal(100)),
        ("cancel50", 1e15 + np.cumsum(r.standard_normal(50))),
        ("complex40", r.standard_normal(40) + 1j * r.standard_normal(40)),
        ("short7", r.standard_normal(7)),
    ]
    xn = np.sin(np.arange(30) / 3.0)
    xn[[2, 15]] = NAN
    signals.append(("nan30", xn))
    for name, x in signals:
        for mode in ("interp", "mirror", "nearest", "constant", "wrap"):
            for N, d, deriv in ((5, 2, 0), (7, 3, 0), (7, 3, 1), (10, 3, 0), (11, 5, 2), (7, 3, 4)):
                if mode == "interp" and N > x.size:
                    continue
                yield (
                    f"savgol_filter {name} {N},{d} deriv={deriv} {mode}",
                    lambda x=x, N=N, d=d, deriv=deriv, mode=mode: fi(a(x), N, d, deriv=deriv, mode=mode, cval=0.5),
                )
    yield "savgol_filter short7 window 9 mirror", lambda: fi(a(signals[4][1]), 9, 3, mode="mirror")
    yield "savgol_filter short7 window 15 wrap", lambda: fi(a(signals[4][1]), 15, 2, mode="wrap")
    yield "savgol_filter delta", lambda: fi(a(signals[1][1]), 9, 4, deriv=2, delta=0.25)
    b = r.standard_normal((6, 30))
    for axis in (0, 1, -1):
        for mode in ("interp", "mirror"):
            yield (
                f"savgol_filter batch axis={axis} {mode}",
                lambda axis=axis, mode=mode: fi(a(b), 5, 2, axis=axis, mode=mode),
            )
    c = r.standard_normal((5, 24, 6))
    yield "savgol_filter strided 3-D axis=1", lambda: fi(a(c)[:, ::2, ::-1], 5, 2, axis=1, deriv=1)
    yield "savgol_filter list input", lambda: fi(list(range(12)), 5, 2)
    yield "savgol_filter int input", lambda: fi(np.arange(12), 5, 2, mode="nearest")
    yield "savgol_filter bool input", lambda: fi(np.array([True, False, True, True, False, False]), 3, 1)
    yield "savgol_filter empty axis", lambda: fi(a(np.zeros((3, 0))), 5, 2, mode="wrap")
    # large enough to split into many tasks at 3 and 10 threads
    rb = L.rng(11)
    big = rb.standard_normal(200001)
    yield "savgol_filter large 101,3", lambda: fi(a(big), 101, 3)
    yield "savgol_filter large 11,3 mirror deriv=1", lambda: fi(a(big), 11, 3, mode="mirror", deriv=1)
    m2 = rb.standard_normal((600, 500))
    yield "savgol_filter large axis=0", lambda: fi(a(m2), 11, 3, axis=0)
    yield "savgol_filter large axis=1 wrap", lambda: fi(a(m2), 25, 5, mode="wrap")
    m3 = rb.standard_normal((40, 30, 50)) + 1j * rb.standard_normal((40, 30, 50))
    yield "savgol_filter large complex axis=1", lambda: fi(a(m3), 9, 2, axis=1)
    for call, f in (
        ("polyorder", lambda: co(5, 5)), ("window", lambda: co(0, 0)), ("deriv", lambda: co(5, 2, deriv=-1)),
        ("pos", lambda: co(5, 2, pos=5)), ("pos float", lambda: co(5, 2, pos=1.5)), ("use", lambda: co(5, 2, use="x")),
        ("delta", lambda: co(5, 2, delta=0.0)), ("interp long", lambda: fi(a(np.ones(5)), 7, 3)),
        ("mode", lambda: fi(a(np.ones(9)), 7, 3, mode="reflect")), ("axis", lambda: fi(a(np.ones(9)), 7, 3, axis=1)),
        ("scalar", lambda: fi(3.0, 1, 0)), ("string", lambda: fi("abc", 1, 0)),
    ):  # fmt: skip
        yield f"savgol error {call}", f


def _sspl(r):
    """A smoothing or P-spline result's numbers (the spline as its knots and coefficients)."""
    return (r.fittedvalues, r.resid, r.leverage, r.df, r.scale, r.nobs, r.lam, r.gcv, r.cv, r.status, r.converged,
            r.spline.t, r.spline.c, r.spline.k, r.spline.axis, repr(r))  # fmt: skip


def _bsp(b):
    return (b.t, b.c, b.k, b.extrapolate, b.axis)


def _prep(b):
    return (_bsp(b[0]), b[1])


PER = "periodic"


def _spline_cases(L):
    """The splines (SM3 milestone): smoothing_spline over lowess's sets times lam (given, 0, inf), df, criterion,
    predict and predict_interval, the weights and every nan_policy; pspline's segments, degrees and orders;
    make_smoothing_spline, make_lsq_spline; make_splrep, make_splprep and generate_knots at several s and periodic;
    a batch on an axis; the wrong calls; and large cases (a lam search on 20000 points, a P-spline of 10^5)."""
    sd, np = L.sd, L.np
    a = L.arr
    SM = sd.smooth
    for name, x, y, w in _scatters(L):
        ww = None if w is None else a(w)
        xe = np.linspace(x.min() - 0.5, x.max() + 0.5, 7) if x.size else np.zeros(0)
        for lam in (None, 0.0, 1e-3, 1.0, INF):

            def fit(x=x, y=y, ww=ww, lam=lam):
                return SM.smoothing_spline(a(x), a(y), lam=lam, weights=ww)

            yield f"smoothing_spline {name} lam={lam}", lambda fit=fit: _sspl(fit())
        yield (
            f"smoothing_spline {name} cv",
            lambda x=x, y=y, ww=ww: _sspl(SM.smoothing_spline(a(x), a(y), criterion="cv", weights=ww)),
        )
        yield (
            f"smoothing_spline {name} predict_interval",
            lambda x=x, y=y, ww=ww, xe=xe: SM.smoothing_spline(a(x), a(y), weights=ww).predict_interval(a(xe)),
        )
        yield f"pspline {name}", lambda x=x, y=y, ww=ww: _sspl(SM.pspline(a(x), a(y), weights=ww))
        yield f"make_smoothing_spline {name}", lambda x=x, y=y, ww=ww: _bsp(SM.make_smoothing_spline(a(x), a(y), w=ww))
        if name in ("random200", "uniform40", "outliers40", "weights30", "shifted30", "line20"):
            xs = np.sort(x)
            ys = y[np.argsort(x, kind="stable")]
            for s in (0.0, 0.05 * x.size, 0.5 * x.size):
                yield f"make_splrep {name} s={s:g}", lambda xs=xs, ys=ys, s=s: _bsp(SM.make_splrep(a(xs), a(ys), s=s))
                yield f"generate_knots {name} s={s:g}", lambda xs=xs, ys=ys, s=s: SM.generate_knots(a(xs), a(ys), s=s)
            t = np.r_[[xs[0]] * 4, xs[3:-3:5], [xs[-1]] * 4]
            yield f"make_lsq_spline {name}", lambda x=x, y=y, t=t, w=ww: _bsp(SM.make_lsq_spline(a(x), a(y), a(t), w=w))
            yield (
                f"make_splrep {name} t",
                lambda xs=xs, ys=ys, t=t: _bsp(SM.make_splrep(a(xs), a(ys), s=0.3 * xs.size, t=a(t))),
            )
    r = L.rng(20261127)
    x = np.sort(10.0 * r.random(150))
    y = np.sin(x) + 0.3 * r.standard_normal(150)
    for d in (6.0, 20.0):
        yield f"smoothing_spline df={d}", lambda d=d: _sspl(SM.smoothing_spline(a(x), a(y), df=d))
    for nseg, k, d in ((10, 3, 2), (25, 2, 1), (40, 3, 3), (8, 1, 2)):
        for lam in (None, 0.0, 10.0, INF):
            yield (
                f"pspline nseg={nseg} k={k} d={d} lam={lam}",
                lambda nseg=nseg, k=k, d=d, lam=lam: _sspl(
                    SM.pspline(a(x), a(y), n_segments=nseg, k=k, diff_order=d, lam=lam)
                ),
            )
    yield "pspline df=7 xb xe", lambda: _sspl(SM.pspline(a(x), a(y), df=7.0, xb=-1.0, xe=11.0))
    Y = np.stack([y, np.cos(x), 2.0 * y], axis=0)
    yield "smoothing_spline batch axis=1", lambda: _sspl(SM.smoothing_spline(a(x), a(Y), axis=1))
    yield "pspline batch axis=-1 cv", lambda: _sspl(SM.pspline(a(x), a(Y), axis=-1, criterion="cv"))
    yn = y.copy()
    yn[[4, 77]] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield f"smoothing_spline nan_policy={pol}", lambda p=pol: _sspl(SM.smoothing_spline(a(x), a(yn), nan_policy=p))
        yield f"pspline nan_policy={pol}", lambda pol=pol: _sspl(SM.pspline(a(x), a(yn), nan_policy=pol))
    for k in (1, 2, 3, 5):
        yield f"make_splrep k={k} s=10", lambda k=k: _bsp(SM.make_splrep(a(x), a(y), k=k, s=10.0))
    xp = np.r_[0.0, np.sort(r.random(98)), 1.0]
    yp = np.sin(2 * np.pi * xp) + 0.1 * r.standard_normal(100)
    yp[-1] = yp[0]
    for k in (1, 3, 4):
        yield f"make_splrep periodic k={k}", lambda k=k: _bsp(SM.make_splrep(a(xp), a(yp), k=k, s=0.5, bc_type=PER))
        yield f"generate_knots periodic k={k}", lambda k=k: SM.generate_knots(a(xp), a(yp), k=k, s=0.5, bc_type=PER)
    th = np.sort(r.random(120)) * 2 * np.pi
    X = np.stack([np.cos(th), np.sin(2 * th)]) + 0.02 * r.standard_normal((2, 120))
    for s in (0.0, 0.05, 0.5):
        yield f"make_splprep s={s}", lambda s=s: _prep(SM.make_splprep(a(X), s=s))
    u = np.linspace(0, 1, 120)
    yield "make_splprep list u", lambda: _prep(SM.make_splprep([X[0].tolist(), X[1].tolist()], u=u, s=0.1))
    yield "make_splrep nest", lambda: _bsp(SM.make_splrep(a(x), a(y), s=5.0, nest=40))
    t_sw = np.r_[[x[0]] * 4, np.linspace(x[1], x[2], 6), [x[9]] * 4]
    for call, f in (
        ("lam and df", lambda: SM.smoothing_spline(a(x), a(y), lam=1.0, df=5.0)),
        ("criterion", lambda: SM.smoothing_spline(a(x), a(y), criterion="aic")),
        ("lam negative", lambda: SM.smoothing_spline(a(x), a(y), lam=-1.0)),
        ("df range", lambda: SM.smoothing_spline(a(x), a(y), df=2.0)),
        ("four x", lambda: SM.smoothing_spline(a([0.0, 1.0, 2.0, 3.0]), a([1.0, 2.0, 0.0, 1.0]))),
        ("complex", lambda: SM.smoothing_spline(a(x), a(y + 1j))),
        ("diff_order", lambda: SM.pspline(a(x), a(y), k=1, diff_order=3)),
        ("xb", lambda: SM.pspline(a(x), a(y), xb=1.0)),
        ("lsq knots", lambda: SM.make_lsq_spline(a(x), a(y), a(x[::-1]))),
        ("lsq schoenberg", lambda: SM.make_lsq_spline(a(x[:10]), a(y[:10]), a(t_sw))),
        ("splrep unreachable", lambda: SM.make_splrep(a(x), a(y), s=1.0, nest=12)),
        ("splrep unsorted", lambda: SM.make_splrep(a(x[::-1]), a(y), s=1.0)),
        ("splrep k", lambda: SM.make_splrep(a(x), a(y), s=1.0, k=6)),
        ("splrep periodic ends", lambda: SM.make_splrep(a(x), a(y), s=1.0, bc_type="periodic")),
        ("splrep w with s=0", lambda: SM.make_splrep(a(x), a(y), w=a(np.ones(150)))),
        ("splprep 1-D", lambda: SM.make_splprep(a(x), s=1.0)),
    ):  # fmt: skip
        yield f"spline error {call}", f
    # large enough to split the lam grid into tasks at 3 and 10 threads
    rb = L.rng(12)
    xb = np.sort(10.0 * rb.random(20000))
    yb = np.sin(xb) + 0.3 * rb.standard_normal(20000)
    yield "smoothing_spline large gcv", lambda: _sspl(SM.smoothing_spline(a(xb), a(yb)))
    qb = np.linspace(0, 10, 3000)
    yield "smoothing_spline large interval", lambda: SM.smoothing_spline(a(xb), a(yb), lam=0.5).predict_interval(a(qb))
    xh = np.sort(10.0 * rb.random(100000))
    yh = np.sin(xh) + 0.3 * rb.standard_normal(100000)
    yield "pspline large", lambda: _sspl(SM.pspline(a(xh), a(yh), n_segments=300))
    yield "make_splrep large", lambda: _bsp(SM.make_splrep(a(xb), a(yb), s=0.09 * 20000))


def _kr(r):
    """A kernel regression result's numbers."""
    return (r.fittedvalues, r.resid, r.leverage, r.df, r.scale, r.nobs, r.bw, r.degree, r.criterion,
            r.criterion_value, r.status, r.converged, repr(r))  # fmt: skip


def _kreg_cases(L):
    """kernel_regression and bw_dpill (SM4 milestone): lowess's sets times degree and bw (numbers, cv_ls, aicc,
    dpill), predict, predict_interval and gradient (with far points), weights, the multivariate sets (bandwidths
    given and chosen by Nelder-Mead), every nan_policy, the inputs' kinds, the wrong calls and large cases."""
    sd, np = L.sd, L.np
    a = L.arr
    KR = sd.smooth.kernel_regression
    for name, x, y, w in _scatters(L):
        ww = None if w is None else a(w)
        span = float(np.ptp(x)) if x.size else 0.0
        xe = np.concatenate([np.linspace(x.min() - span - 1.0, x.max() + span + 1.0, 7), x[:2], [x.min() - 1e6]])
        for deg in (0, 1):
            for bw in (max(span, 1.0) / 20, max(span, 1.0) / 3, "cv_ls", "aicc"):

                def fit(x=x, y=y, ww=ww, deg=deg, bw=bw):
                    return KR(a(x), a(y), degree=deg, bw=bw, weights=ww)

                tag = f"kernel_regression {name} degree={deg} bw={bw if isinstance(bw, str) else f'{bw:.3g}'}"
                yield tag, lambda fit=fit: _kr(fit())
                if not isinstance(bw, str):
                    yield f"{tag} predict", lambda fit=fit, xe=xe: fit().predict(a(xe))
                    yield f"{tag} predict_interval", lambda fit=fit, xe=xe: fit().predict_interval(a(xe), alpha=0.1)
                    if deg == 1:
                        yield f"{tag} gradient", lambda fit=fit, xe=xe: fit().gradient(a(xe))
        yield f"bw_dpill {name}", lambda x=x, y=y: sd.smooth.bw_dpill(a(x), a(y))
    r = L.rng(17)
    for p, n in ((2, 60), (3, 50)):
        X = 2.0 * r.random((n, p))
        Y = np.sin(2 * X[:, 0]) + X[:, -1] ** 2 + 0.2 * r.standard_normal(n)
        Xe = np.concatenate([2.0 * r.random((5, p)), [[-50.0] * p]])
        for deg in (0, 1):
            yield (
                f"kernel_regression multi{p} degree={deg} cv_ls",
                lambda X=X, Y=Y, deg=deg: _kr(KR(a(X), a(Y), degree=deg)),
            )
            yield (
                f"kernel_regression multi{p} degree={deg} bw given",
                lambda X=X, Y=Y, Xe=Xe, deg=deg: (
                    _kr(KR(a(X), a(Y), degree=deg, bw=0.4)),
                    KR(a(X), a(Y), degree=deg, bw=a([0.3] * p)).predict(a(Xe)),
                    KR(a(X), a(Y), degree=deg, bw=a([0.3] * p)).predict_interval(a(Xe)),
                ),
            )
        yield f"kernel_regression multi{p} aicc", lambda X=X, Y=Y: _kr(KR(a(X), a(Y), bw="aicc"))
        yield f"kernel_regression multi{p} gradient", lambda X=X, Y=Y, Xe=Xe: KR(a(X), a(Y), bw=0.5).gradient(a(Xe))
    u = 2.0 * r.random(40)
    yield "kernel_regression lowrank", lambda: _kr(KR(a(np.stack([u, 0.5 * u + 0.25], axis=1)), a(np.sin(u)), bw=0.5))
    xn = np.sort(10.0 * r.random(50))
    yn = np.sin(xn) + 0.3 * r.standard_normal(50)
    yn[[3, 20]] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield f"kernel_regression nan_policy={pol}", lambda pol=pol: _kr(KR(a(xn), a(yn), nan_policy=pol))
    yield "kernel_regression nan predict", lambda: KR(a(xn), a(yn), bw=0.5, nan_policy="omit").predict(a([NAN, 1.0]))
    yield "kernel_regression scalar predict", lambda: KR(a(xn[:30]), a(np.cos(xn[:30])), bw=0.5).predict(2.5)
    yield "kernel_regression list input", lambda: _kr(KR(xn[:30].tolist(), np.cos(xn[:30]).tolist(), bw=0.5))
    yield "kernel_regression int input", lambda: _kr(KR(np.arange(12), np.arange(12) % 4, bw=2))
    yield "kernel_regression summary", lambda: KR(a(xn[:30]), a(np.cos(xn[:30]))).summary()
    for kw in (
        {"degree": 2}, {"degree": 1.0}, {"bw": -1.0}, {"bw": "scott"}, {"bw": a([1.0, 2.0])}, {"bw": INF},
        {"kernel": "epanechnikov"}, {"nan_policy": "drop"}, {"weights": a(np.zeros(12))}, {"weights": a(-np.ones(12))},
        {"bandwidth": 1.0},
    ):  # fmt: skip
        yield (
            f"kernel_regression error {sorted(kw)[0]}={kw[sorted(kw)[0]]!r}",
            lambda kw=kw: KR(a(np.arange(12.0)), a(np.sin(np.arange(12.0))), **kw),
        )
    yield "kernel_regression error lengths", lambda: KR(a(np.arange(5.0)), a(np.arange(4.0)))
    yield "kernel_regression error one x", lambda: KR(a(np.ones(6)), a(np.arange(6.0)))
    yield "kernel_regression error dpill 2-D", lambda: KR(a(np.ones((8, 2))), a(np.arange(8.0)), bw="dpill")
    yield "kernel_regression error gradient", lambda: KR(a(xn[:9]), a(xn[:9]), degree=0, bw=1.0).gradient(1.0)
    yield "bw_dpill error few", lambda: sd.smooth.bw_dpill(a([0.0, 1.0, 2.0]), a([1.0, 0.0, 1.0]))
    yield "bw_dpill error zero", lambda: sd.smooth.bw_dpill(a(np.arange(40.0)), a(np.zeros(40)))
    # large enough to split into many tasks at 3 and 10 threads
    rb = L.rng(19)
    xb = np.sort(10.0 * rb.random(3000))
    yb = np.sin(xb) + 0.3 * rb.standard_normal(3000)
    wb = 0.5 + rb.random(3000)
    wb[::11] = 0.0
    yield "kernel_regression large cv_ls", lambda: _kr(KR(a(xb), a(yb)))
    yield "kernel_regression large weights", lambda: _kr(KR(a(xb), a(yb), degree=0, bw="aicc", weights=a(wb)))
    yield (
        "kernel_regression large predict_interval",
        lambda: KR(a(xb), a(yb), bw=0.3).predict_interval(a(rb.random(5000) * 12 - 1)),
    )
    yield "bw_dpill large", lambda: sd.smooth.bw_dpill(a(xb), a(yb))


def _kd(r):
    """A KDEResult's fields."""
    return (r.bw, r.covariance, r.factor, r.neff, r.nobs, r.ndim, r.kernel, r.bounds, r.weights, repr(r))


def _kde_cases(L):
    """kde and the bw_* rules (SM5 milestone): the density sets rebuilt from L.rng (normal, two modes, a heavy
    tail, a bounded sample, integers, weights, one value; correlated 2-D and 3-D, points on a line) times bw
    (every rule, numbers, per-variable numbers, a covariance), adjust, the kernels and bounds; pdf, logpdf and cdf
    with far points, the binned and exact grids, sample and marginal, every nan_policy, the inputs' kinds, the
    wrong calls and large cases that split into tasks."""
    sd, np = L.sd, L.np
    a = L.arr
    K = sd.smooth.kde
    r = L.rng(20261124)
    sets = [
        ("normal60", r.standard_normal(60), None),
        ("bimodal80", np.concatenate([r.standard_normal(50) - 2.0, 0.5 * r.standard_normal(30) + 2.0]), None),
        ("heavy60", r.standard_normal(60) / np.sqrt(r.chisquare(3, 60) / 3.0), None),
        ("expo60", r.exponential(1.0, 60), None),
        ("ints60", r.poisson(3.0, 60).astype(float), None),
        ("weighted40", r.standard_normal(40), r.random(40) + 0.1),
    ]
    for name, x, w in sets:
        ww = None if w is None else a(w)
        span = float(np.ptp(x))
        xe = np.concatenate([np.linspace(x.min() - span, x.max() + span, 9), x[:2], [x.min() - 30 * span]])
        rules = ("scott", "silverman") if w is not None else ("scott", "silverman", "nrd0", "nrd", "sj", "isj", "cv_ls",
                                                              "cv_ml")  # fmt: skip
        for bw in rules + (0.3,):
            tag = f"kde {name} bw={bw}"

            def fit(x=x, ww=ww, bw=bw):
                return K(a(x), bw=bw, weights=ww)

            yield tag, lambda fit=fit, xe=xe: (_kd(fit()), fit().pdf(a(xe)), fit().logpdf(a(xe)), fit().cdf(a(xe)))
        yield (
            f"kde {name} density",
            lambda x=x, ww=ww: (K(a(x), weights=ww).density(), K(a(x), weights=ww).density(n=1000)),
        )
        yield f"kde {name} density exact", lambda x=x, ww=ww: K(a(x), weights=ww).density(n=64, binned=False)
        for kern in ("epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
            yield (
                f"kde {name} kernel={kern}",
                lambda x=x, ww=ww, xe=xe, kern=kern: (
                    K(a(x), bw=0.4, kernel=kern, weights=ww).pdf(a(xe)),
                    K(a(x), bw=0.4, kernel=kern, weights=ww).cdf(a(xe)),
                    K(a(x), bw=0.4, kernel=kern, weights=ww).density(n=300),
                ),
            )
        yield f"kde {name} adjust", lambda x=x, ww=ww, xe=xe: K(a(x), adjust=0.5, weights=ww).pdf(a(xe))
        yield f"kde {name} sample", lambda x=x, ww=ww: K(a(x), weights=ww).sample((4, 5), rng=sd.random.default_rng(3))
        if w is None:
            for meth in (sd.smooth.bw_scott, sd.smooth.bw_silverman, sd.smooth.bw_nrd0, sd.smooth.bw_nrd,
                         sd.smooth.bw_sj, sd.smooth.bw_isj, sd.smooth.bw_cv_ls, sd.smooth.bw_cv_ml):  # fmt: skip
                yield f"{meth.__name__} {name}", lambda meth=meth, x=x: meth(a(x))
            yield f"bw_sj {name} dpi", lambda x=x: sd.smooth.bw_sj(a(x), method="dpi")
            yield f"bw_isj {name} n_grid=256", lambda x=x: sd.smooth.bw_isj(a(x), n_grid=256)
            yield f"bw_cv_ls {name} nb=1000", lambda x=x: sd.smooth.bw_cv_ls(a(x), nb=1000)
        else:
            yield f"bw_scott {name} weights", lambda x=x, ww=ww: sd.smooth.bw_scott(a(x), weights=ww)
            yield f"bw_silverman {name} weights", lambda x=x, ww=ww: sd.smooth.bw_silverman(a(x), weights=ww)
    xb = sets[3][1]
    for bounds in ((0.0, INF), (0.0, 8.0), (-INF, 9.0)):
        yield (
            f"kde bounds={bounds}",
            lambda bounds=bounds: (
                K(a(xb), bw=0.3, bounds=bounds).pdf(a(np.linspace(-1, 10, 23))),
                K(a(xb), bw=0.3, bounds=bounds).logpdf(a(np.linspace(-1, 10, 23))),
                K(a(xb), bw=0.3, bounds=bounds).cdf(a(np.linspace(-1, 10, 23))),
                K(a(xb), bw=0.3, bounds=bounds).density(),
                K(a(xb), bw=0.3, bounds=bounds).sample(7, rng=sd.random.default_rng(1)),
            ),
        )
    for p, n in ((2, 60), (3, 50)):
        A_ = np.tril(0.6 * r.standard_normal((p, p))) + np.eye(p)
        X = r.standard_normal((n, p)) @ A_.T
        Xe = np.concatenate([r.standard_normal((5, p)), X[:2], [[40.0] * p]])
        for bw in ("scott", "silverman", "cv_ml", 0.5, tuple([0.3] * p), "cov"):
            arg = a(np.cov(X.T) * 0.3) if bw == "cov" else (a(list(bw)) if isinstance(bw, tuple) else bw)
            yield (
                f"kde multi{p} bw={bw}",
                lambda X=X, Xe=Xe, arg=arg: (
                    _kd(K(a(X), bw=arg)),
                    K(a(X), bw=arg).pdf(a(Xe)),
                    K(a(X), bw=arg).logpdf(a(Xe)),
                ),
            )
        yield f"kde multi{p} density", lambda X=X: (K(a(X)).density(n=20), K(a(X)).density(n=12, binned=False))
        yield (
            f"kde multi{p} marginal",
            lambda X=X, Xe=Xe: (_kd(K(a(X)).marginal(0)), K(a(X)).marginal([p - 1, 0]).pdf(a(Xe[:, [p - 1, 0]]))),
        )
        yield f"kde multi{p} sample", lambda X=X: K(a(X)).sample(6, rng=sd.random.default_rng(2))
        yield f"bw_cv_ls multi{p}", lambda X=X: sd.smooth.bw_cv_ls(a(X))
        yield f"bw_scott multi{p}", lambda X=X: sd.smooth.bw_scott(a(X))
    u = r.standard_normal(20)
    yield "kde degenerate line", lambda: K(a(np.stack([u, 2 * u - 1], axis=1)))
    yield "kde degenerate allsame", lambda: K(a(np.full(10, 1.5)))
    xn = r.standard_normal(30)
    xn[[4, 9]] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield (
            f"kde nan_policy={pol}",
            lambda pol=pol: (lambda k: (_kd(k), k.pdf(a([0.0, 1.0]))))(K(a(xn), nan_policy=pol)),
        )
    yield "kde nan query", lambda: K(a(xn[:3]), bw=0.5).pdf(a([NAN, 0.0]))
    yield "kde scalar query", lambda: (K(a(xn[:3]), bw=0.5).pdf(0.25), K(a(xn[:3]), bw=0.5).logpdf(-3.0))
    yield "kde list input", lambda: _kd(K([0.0, 1.0, 3.0, 4.5]))
    yield "kde int input", lambda: _kd(K(np.arange(12)))
    yield (
        "kde tiny bandwidth",
        lambda: (K(a([0.0, 1.0]), bw=2.0**-1060).logpdf(a([0.0, 1e-300])), K(a([0.0]), bw=1e300).pdf(0.0)),
    )
    for kw in (
        {"bw": -1.0}, {"bw": "normal_reference"}, {"bw": INF}, {"kernel": "gau"}, {"adjust": 0.0},
        {"nan_policy": "drop"}, {"weights": a(np.zeros(12))}, {"weights": a(-np.ones(12))}, {"bounds": (3.0, 1.0)},
        {"bounds": (0.5, 20.0)}, {"bw": "sj", "weights": a(np.ones(12))}, {"bandwidth": 1.0},
    ):  # fmt: skip
        yield f"kde error {sorted(kw)[0]}={kw[sorted(kw)[0]]!r}", lambda kw=kw: K(a(np.arange(12.0)), **kw)
    yield "kde error 2-D kernel", lambda: K(a(np.ones((8, 2)) + np.arange(8)[:, None]), kernel="biweight")
    yield "kde error cdf 2-D", lambda: K(a(r.standard_normal((9, 2)))).cdf(a([[0.0, 0.0]]))
    yield "bw_sj error scale", lambda: sd.smooth.bw_sj(a(np.full(10, 2.0)))
    yield "bw_isj error spread", lambda: sd.smooth.bw_isj(a(np.full(10, 2.0)))
    yield "bw_nrd0 error few", lambda: sd.smooth.bw_nrd0(a([1.0]))
    # large enough to split into many tasks at 3 and 10 threads
    rb = L.rng(23)
    xb2 = rb.standard_normal(30000)
    Xb = rb.standard_normal((4000, 2))
    yield "kde large pdf", lambda: K(a(xb2)).pdf(a(np.linspace(-5, 5, 2001)))
    yield "kde large cdf", lambda: K(a(xb2)).cdf(a(np.linspace(-5, 5, 2001)))
    yield "kde large density", lambda: K(a(xb2)).density(n=5000)
    yield "kde large 2-D", lambda: K(a(Xb)).pdf(a(Xb[:1500] * 1.2))
    yield "bw_cv_ml large", lambda: sd.smooth.bw_cv_ml(a(xb2[:4000]))


def _loess_cases(L):
    """loess (SM2 milestone): lowess's sets times degree, span (with span > 1) and family; several variables (two
    and three uniform, scales 1e6 apart, points on a line, a lattice) with normalize on and off; predict,
    predict_interval and two_delta; weights; every nan_policy; the inputs' kinds; the wrong calls; cases large enough
    to split into tasks at 3 and 10 threads (one variable, two variables on the KD tree's path and on the scan)."""
    sd, np = L.sd, L.np
    a = L.arr
    loess = sd.smooth.loess
    for name, x, y, w in _scatters(L):
        if x.size < 8:
            continue
        span = float(np.ptp(x))
        xe = np.concatenate([np.linspace(x.min() - 0.1 * span - 0.5, x.max() + 0.1 * span + 0.5, 7), x[:3]])
        ww = None if w is None else a(w)
        for degree in (0, 1, 2):
            for sp in (0.5, 1.5):
                for fam in ("gaussian", "symmetric"):

                    def fit(x=x, y=y, ww=ww, degree=degree, sp=sp, fam=fam):
                        return loess(a(x), a(y), span=sp, degree=degree, family=fam, weights=ww)

                    yield f"loess {name} degree={degree} span={sp} {fam}", fit
                    if fam == "gaussian":
                        yield (
                            f"loess {name} degree={degree} span={sp} predict",
                            lambda fit=fit, xe=xe: fit().predict(a(xe)),
                        )
        yield (
            f"loess {name} predict_interval",
            lambda x=x, y=y, ww=ww, xe=xe: loess(a(x), a(y), weights=ww).predict_interval(a(xe), alpha=0.1),
        )
    r = L.rng(20261125)
    multis = []
    for p, n in ((2, 60), (3, 60)):
        X = 2.0 * r.random((n, p))
        multis.append((f"uniform{p}_{n}", X, np.sin(X[:, 0]) + 0.5 * X[:, -1] ** 2 + 0.2 * r.standard_normal(n)))
    Xs = 2.0 * r.random((50, 2)) * np.array([1.0, 1e6])
    multis.append(("scales2_50", Xs, np.sin(Xs[:, 0]) + 0.2 * r.standard_normal(50)))
    u = 2.0 * r.random(40)
    multis.append(("lowrank2_40", np.stack([u, 0.5 * u + 0.25], axis=1), np.sin(u) + 0.2 * r.standard_normal(40)))
    g = np.arange(7.0) / 4.0
    Xl = np.stack(np.meshgrid(g, g, indexing="ij"), axis=-1).reshape(-1, 2)
    multis.append(("lattice2_49", Xl, np.cos(Xl[:, 0]) * Xl[:, 1] + 0.1 * r.standard_normal(49)))
    for name, X, y in multis:
        Xe = np.concatenate([X.min(axis=0) + np.ptp(X, axis=0) * r.random((5, X.shape[1])), X[:2]])
        for degree, sp in ((0, 0.5), (1, 0.75), (2, 0.75)):
            for norm in (True, False):

                def fit(X=X, y=y, degree=degree, sp=sp, norm=norm):
                    return loess(a(X), a(y), span=sp, degree=degree, normalize=norm)

                yield f"loess {name} degree={degree} normalize={norm}", fit
                yield (
                    f"loess {name} degree={degree} normalize={norm} predict",
                    lambda fit=fit, Xe=Xe: fit().predict(a(Xe)),
                )
        yield (
            f"loess {name} symmetric predict_interval",
            lambda X=X, y=y, Xe=Xe: loess(a(X), a(y), family="symmetric").predict_interval(a(Xe)),
        )
    xn = np.sort(10.0 * r.random(40))
    yn = np.sin(xn) + 0.3 * r.standard_normal(40)
    xn[5] = NAN
    for pol in ("propagate", "omit", "raise"):
        yield f"loess nan_policy={pol}", lambda pol=pol: loess(a(xn), a(yn), nan_policy=pol)
    yield "loess nan_policy=omit predict", lambda: loess(a(xn), a(yn), nan_policy="omit").predict(a([NAN, 1.0, 20.0]))
    yield "loess scalar predict", lambda: loess(a(xn[6:]), a(yn[6:])).predict(2.5)
    yield "loess list input", lambda: loess(xn[6:].tolist(), yn[6:].tolist(), span=0.5)
    yield "loess int input", lambda: loess(np.arange(20), np.arange(20) % 5, degree=1)
    yield "loess (n, 1) input", lambda: loess(a(xn[6:, None]), a(yn[6:])).predict(a([[1.0], [2.0]]))
    for kw in (
        {"span": 0.0},
        {"span": 0.1},
        {"degree": 3},
        {"degree": 1.5},
        {"family": "robust"},
        {"iterations": 0},
        {"normalize": "yes"},
        {"nan_policy": "drop"},
    ):
        key = sorted(kw)[0]
        yield f"loess error {key}={kw[key]!r}", lambda kw=kw: loess(a(np.arange(12.0)), a(np.arange(12.0) % 3), **kw)
    yield "loess error lengths", lambda: loess(a(np.arange(5.0)), a(np.arange(4.0)))
    yield "loess error inf", lambda: loess(a([0.0, INF, 2.0, 3.0]), a([1.0, 2.0, 3.0, 4.0]))
    yield "loess error overflow", lambda: loess(a([-1.5e308, 1.5e308, 0.0, 1.0]), a([1.0, 2.0, 3.0, 4.0]), degree=0)
    yield (
        "loess error flat column",
        lambda: loess(a(np.c_[np.arange(20.0), np.r_[0.0, np.ones(18), 2.0]]), a(np.arange(20.0))),
    )
    yield "loess error predict columns", lambda: loess(a(multis[0][1]), a(multis[0][2])).predict(1.0)
    # large enough to split into many tasks at 3 and 10 threads
    rb = L.rng(23)
    xb = np.sort(10.0 * rb.random(5000))
    yb = np.sin(xb) + 0.3 * rb.standard_normal(5000)
    wb = 0.5 + rb.random(5000)
    wb[::13] = 0.0
    Xb = 10.0 * rb.random((3000, 2))
    Yb = np.sin(Xb[:, 0]) + np.cos(Xb[:, 1]) + 0.3 * rb.standard_normal(3000)
    yield "loess large", lambda: loess(a(xb), a(yb), span=0.3)
    yield "loess large weights symmetric", lambda: loess(a(xb), a(yb), weights=a(wb), family="symmetric", degree=1)
    yield "loess large 2-D tree", lambda: loess(a(Xb), a(Yb), span=0.03, degree=1)
    yield "loess large 2-D scan", lambda: loess(a(Xb), a(Yb), span=0.5)
    yield (
        "loess large 2-D predict_interval",
        lambda: loess(a(Xb[:1500]), a(Yb[:1500])).predict_interval(a(10.0 * rb.random((2000, 2)))),
    )


def cases(L):
    yield from _lowess_cases(L)
    yield from _loess_cases(L)
    yield from _savgol_cases(L)
    yield from _spline_cases(L)
    yield from _kreg_cases(L)
    yield from _kde_cases(L)
