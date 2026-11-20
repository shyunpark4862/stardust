"""sd.regression against statsmodels, scikit-learn, scipy and R. From
bench/regression/g1/speed.py (at commit 4151d2a) (ols), g1b/speed.py (HAC, cluster, diagnostics, summary),
g2/speed.py (glm), g3/speed.py (penalized), g4/speed.py (rlm M and MM, Theil-Sen,
Siegel), g5/speed.py (quantreg) and g6/speed.py (nls), with more functions, options,
shapes and input conditions since. Only the times are kept: the iteration counts,
coefficient differences and objective excesses of the old tables are accuracy, not speed.

The data of the old sizes are the old scripts' (same generators and seeds; where a script
drew several sizes from one generator, a setup starts the generator at the state after the
earlier draws, so the data are the same without drawing them). statsmodels and
scikit-learn follow sd's thread count (their BLAS and OpenMP pools); scipy runs at its
defaults; R (one session, its reference BLAS) once per round. statsmodels gets its
constant column beforehand.

The iterative comparators are asked for sd's accuracy: scikit-learn's solvers get tol=1e-10
(saga 1e-8), and their labels say so; the note of the case where it was measured gives the
coefficient error at the default tol. statsmodels' IRLS fits converge to sd's answer at
their defaults; statsmodels' QuantReg is an approximate IRLS whose default fit also
computes a kernel sandwich, so it is paired with sd's cov_type='ker'.

Left out: statsmodels' external studentized residuals (n refits), scipy's Theil-Sen and
Siegel at n = 10^5 (an n x n array, 80 GB), R's rq.fit "br" at n = 10^6 (minutes), at k = 50
(12 s) and over nine quantiles (25 s). Two ablations run the public function with one kernel
option changed for the call (the glm inner iterations by QR, the MM small-n search on all
rows; _kernel_option). Some calls take many seconds (statsmodels RLM at (10^6, 50),
Theil-Sen and Siegel at 10^5, nls at 10^6 with p = 10): their rows say so.
"""

from __future__ import annotations

import math
import warnings

from ..registry import Case, Ref, RRef


def _expected_warnings():
    from sklearn.exceptions import ConvergenceWarning as SkConvergence
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as SmConvergence

    return (RuntimeWarning, SkConvergence, SmConvergence)


def _quiet(f, *more):
    """f with the expected warnings ignored (convergence, overflow in the comparators'
    iterations, and `more`); deprecations and the rest still show."""
    cats = _expected_warnings() + more

    def g():
        with warnings.catch_warnings():
            for c in cats:
                warnings.simplefilter("ignore", c)
            return f()

    return g


def _sm_warning(name):
    import statsmodels.tools.sm_exceptions as e

    return getattr(e, name)


def _kernel_option(module, kernel, call, **options):
    """call() with stardust.regression.<module>._k.<kernel> given `options` for the call:
    the module's kernel binding is swapped for a proxy whose `kernel` is the real one with
    the options bound. A missing kernel or option, or a call that no longer reaches the
    kernel, raises, so the case fails instead of timing the default path."""
    import functools
    import importlib

    mod = importlib.import_module(f"stardust.regression.{module}")
    real = mod._k
    bound = functools.partial(getattr(real, kernel), **options)
    used = [0]

    def through(*args, **kwargs):
        used[0] += 1
        return bound(*args, **kwargs)

    class Proxy:
        def __getattr__(self, name):
            return getattr(real, name)

    proxy = Proxy()
    setattr(proxy, kernel, through)

    def f():
        used[0] = 0
        mod._k = proxy
        try:
            out = call()
        finally:
            mod._k = real
        if not used[0]:
            raise RuntimeError(f"the call no longer reaches stardust.regression.{module}._k.{kernel}")
        return out

    return f


def _size(n, k):
    return f"({n:,}, {k})"


def _sm():
    import statsmodels.api as sm

    return sm


def _R():
    import stardust as sd

    return sd.regression


def _sd_rng(seed):
    import stardust as sd

    return sd.random.default_rng(seed)


def _np_rng(state):
    """numpy's default_rng(0) moved to the PCG64 state `state` (see _SKIP)."""
    import numpy as np

    rng = np.random.default_rng(0)
    st = rng.bit_generator.state
    st["state"]["state"] = state
    rng.bit_generator.state = st
    return rng


# The PCG64 state of numpy's default_rng(0) after the draws of the earlier sizes of one old
# script, so a setup draws only its own data: (g1, 3) after G1_SIZES[:3], (g1, batch) after
# all of G1_SIZES, (g2, batch) after all of G2_SIZES. Recompute with
#   rng = np.random.default_rng(0); <the earlier draws>; rng.bit_generator.state["state"]["state"]
_SKIP = {
    ("g1", 3): 79281709638914798765890678983760658124,
    ("g1", "batch"): 106785980383449394103910849043684612640,
    ("g2", "batch"): 13199395297045448037151362524021540636,
}


def _sk_fn(name):
    import sklearn.linear_model as lm

    return getattr(lm, name)


def _sk_tol(cls, **kw):
    return _sk_fn(cls)(**kw)


# ---------------------------------------------------------------- g1: ols, wls, gls, linregress

G1_SIZES = ((1_000, 10), (100_000, 10), (1_000_000, 50), (10_000, 1_000))
G1_BATCH = (10_000, 50, 3)


def _ols_draw(rng, n, k):
    X = rng.standard_normal((n, k))
    y = X @ rng.standard_normal(k) + rng.standard_normal(n)
    return X, y


def _lin_pack(X, y):
    import stardust as sd

    return {"X": X, "y": y, "Xc": _sm().add_constant(X), "Xs": sd.asarray(X), "ys": sd.asarray(y)}


def _g1_data(index, weights=False):
    """index into G1_SIZES, or len(G1_SIZES) for the batch (one generator over all, in order).
    weights: also positive weights w (their own generator) for wls."""

    def setup():
        import numpy as np

        import stardust as sd

        if index == len(G1_SIZES):
            rng = _np_rng(_SKIP[("g1", "batch")])
            B, n, k = G1_BATCH
            X = rng.standard_normal((B, n, k))
            y = np.einsum("bnk,k->bn", X, [1.0, 2.0, 3.0]) + rng.standard_normal((B, n))
            sm = _sm()
            return {
                "X": X,
                "y": y,
                "Xc": [sm.add_constant(X[i]) for i in range(B)],
                "Xs": sd.asarray(X),
                "ys": sd.asarray(y),
                "Xl": [sd.asarray(X[i]) for i in range(B)],
                "yl": [sd.asarray(y[i]) for i in range(B)],
            }
        if ("g1", index) in _SKIP:
            rng = _np_rng(_SKIP[("g1", index)])
        else:
            rng = np.random.default_rng(0)
            for n, k in G1_SIZES[:index]:
                _ols_draw(rng, n, k)
        d = _lin_pack(*_ols_draw(rng, *G1_SIZES[index]))
        if weights:
            w = np.random.default_rng(1).uniform(0.5, 2.0, len(d["y"]))
            d.update(w=w, ws=sd.asarray(w))
        return d

    return setup


def _lin_data(n, k, seed=1):
    def setup():
        import numpy as np

        return _lin_pack(*_ols_draw(np.random.default_rng(seed), n, k))

    return setup


def _ols_inf(r):
    return r.params, r.bse, r.pvalues, r.rsquared


def _linreg():
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _ols_case(size, setup, quick=False):
    n, k = size
    return Case(
        area="regression",
        group="ols",
        name="ols(y, X)",
        size=_size(n, k),
        setup=setup,
        sd=lambda d: lambda R=_R(), y=d["ys"], X=d["Xs"]: R.ols(y, X),
        refs=(
            Ref("statsmodels", lambda d: lambda: _sm().OLS(d["y"], d["Xc"]).fit(), threads="follow"),
            Ref("scikit-learn", lambda d: lambda: _linreg().fit(d["X"], d["y"]), threads="follow", lib="sklearn"),
        ),
        quick=quick,
        note="10^4행: 2^15·2^16행 문턱 아래의 중간 크기" if size == (10_000, 10) else "",
    )


def _g1_cases():
    out = []
    for i, (n, k) in enumerate(G1_SIZES):
        setup = _g1_data(i)
        out.append(_ols_case((n, k), setup, quick=(n, k) == (100_000, 10)))
        if (n, k) in ((1_000, 10), (10_000, 1_000)):  # at the other sizes "bse, pvalues" below does this work
            out.append(
                Case(
                    area="regression",
                    group="ols",
                    name="ols(y, X) + params·bse·pvalues·rsquared",
                    size=_size(n, k),
                    setup=setup,
                    sd=lambda d: lambda R=_R(), y=d["ys"], X=d["Xs"]: _ols_inf(R.ols(y, X)),
                    refs=(
                        Ref(
                            "statsmodels",
                            lambda d: lambda: _ols_inf(_sm().OLS(d["y"], d["Xc"]).fit()),
                            threads="follow",
                        ),
                    ),
                )
            )
    out.append(_ols_case((10_000, 10), _lin_data(10_000, 10)))
    setup = _g1_data(len(G1_SIZES))
    B, n, k = G1_BATCH

    def sm_loop(d):
        sm = _sm()

        def f():
            for i in range(B):
                _ols_inf(sm.OLS(d["y"][i], d["Xc"][i]).fit())

        return f

    def sd_loop(d):
        R = _R()

        def f():
            for i in range(B):
                _ols_inf(R.ols(d["yl"][i], d["Xl"][i]))

        return f

    size = f"{B:,} × ({n}, {k})"
    out.append(
        Case(
            area="regression",
            group="ols-batched",
            name="ols(Y, X) + inference, one call",
            size=size,
            setup=setup,
            sd=lambda d: lambda R=_R(), y=d["ys"], X=d["Xs"]: _ols_inf(R.ols(y, X)),
            refs=(Ref("statsmodels loop", sm_loop, threads="follow", lib="statsmodels"),),
            quick=True,
            note="statsmodels는 Python 반복문으로 하나씩 적합한다",
        )
    )
    out.append(
        Case(
            area="regression",
            group="ols-batched",
            name="ols(y_i, X_i) + inference, Python loop",
            size=size,
            setup=setup,
            sd=sd_loop,
        )
    )
    return out


def _gls_data(n, k):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        idx = np.arange(n)
        S = 0.5 ** np.abs(idx[:, None] - idx[None, :])
        e = np.linalg.cholesky(S) @ rng.standard_normal(n)
        y = X @ rng.standard_normal(k) + e
        d = _lin_pack(X, y)
        d.update(S=S, Ss=sd.asarray(S))
        return d

    return setup


def _wls_gls_cases():
    out = []
    for i in (1, 2):
        n, k = G1_SIZES[i]
        out.append(
            Case(
                area="regression",
                group="wls-gls",
                name="wls(y, X, weights=w)",
                size=_size(n, k),
                setup=_g1_data(i, weights=True),
                sd=lambda d: lambda R=_R(): R.wls(d["ys"], d["Xs"], weights=d["ws"]),
                refs=(
                    Ref(
                        "statsmodels",
                        lambda d: lambda: _sm().WLS(d["y"], d["Xc"], weights=d["w"]).fit(),
                        threads="follow",
                    ),
                    Ref(
                        "scikit-learn",
                        lambda d: lambda: _linreg().fit(d["X"], d["y"], sample_weight=d["w"]),
                        threads="follow",
                        lib="sklearn",
                    ),
                ),
                note="가중치는 U(0.5, 2); scikit-learn은 sample_weight",
            )
        )
    n, k = 2_000, 10
    out.append(
        Case(
            area="regression",
            group="wls-gls",
            name="gls(y, X, sigma=S)",
            size=_size(n, k),
            setup=_gls_data(n, k),
            sd=lambda d: lambda R=_R(): R.gls(d["ys"], d["Xs"], sigma=d["Ss"]),
            refs=(
                Ref("statsmodels", lambda d: lambda: _sm().GLS(d["y"], d["Xc"], sigma=d["S"]).fit(), threads="follow"),
            ),
            note="S는 AR(1) 상관 0.5^|i−j|의 n × n 행렬",
        )
    )
    return out


def _slr_data(n, batch=None, seed=0):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(seed)
        shape = (n,) if batch is None else (batch, n)
        x = rng.standard_normal(shape)
        y = 2.0 * x + 1.0 + rng.standard_normal(shape)
        return {"x": x, "y": y, "xs": sd.asarray(x), "ys": sd.asarray(y)}

    return setup


def _linregress_cases():
    out = []
    for n in (200, 1_000, 1_000_000):
        out.append(
            Case(
                area="regression",
                group="simple",
                name="linregress(x, y)",
                size=f"{n:,}",
                n=n,
                setup=_slr_data(n),
                sd=lambda d: lambda f=_R().linregress: f(d["xs"], d["ys"]),
                refs=(Ref("scipy", lambda d: lambda f=_scipy_stats("linregress"): f(d["x"], d["y"])),),
            )
        )
    B, n = 1_000, 1_000
    out.append(
        Case(
            area="regression",
            group="simple",
            name="linregress(x, y), batched",
            size=f"{B:,} × {n:,}",
            n=B * n,
            setup=_slr_data(n, batch=B),
            sd=lambda d: lambda f=_R().linregress: f(d["xs"], d["ys"]),
            refs=(Ref("scipy", lambda d: lambda f=_scipy_stats("linregress"): f(d["x"], d["y"], axis=-1)),),
            note="scipy는 axis=-1",
        )
    )
    return out


# ---------------------------------------------------------------- g1b: HC, HAC, cluster, diagnostics

G1B_SIZES = ((100_000, 10), (1_000_000, 50))


def _g1b_data(index):
    def setup():
        import numpy as np
        from scipy.signal import lfilter

        import stardust as sd

        sm = _sm()
        rng = np.random.default_rng(0)
        for i, (n, k) in enumerate(G1B_SIZES):
            X = rng.standard_normal((n, k))
            e = rng.standard_normal(n)
            beta = rng.standard_normal(k)
            g = rng.integers(0, 1000, n)
            if i == index:
                e = lfilter([1.0], [1.0, -0.5], e)  # e[t] += 0.5 e[t-1], in order
                y = X @ beta + e
                return {
                    "X": X,
                    "y": y,
                    "g": g,
                    "Xc": sm.add_constant(X),
                    "Xs": sd.asarray(X),
                    "ys": sd.asarray(y),
                    "gs": sd.asarray(g),
                    "L": int(math.floor(4 * (n / 100) ** (2 / 9))),
                }
            del X, e, g
        raise IndexError(index)

    return setup


def _bp(r):
    return r.bse, r.pvalues


def _g1b_cases():
    def sm_diag(d):
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.outliers_influence import OLSInfluence
        from statsmodels.stats.stattools import durbin_watson, jarque_bera

        sm = _sm()

        def f():
            r = sm.OLS(d["y"], d["Xc"]).fit()
            inf = OLSInfluence(r)
            return (
                inf.hat_matrix_diag,
                inf.resid_studentized_internal,
                inf.cooks_distance,
                durbin_watson(r.wresid),
                jarque_bera(r.wresid),
                het_breuschpagan(r.resid, d["Xc"]),
            )

        return f

    def sd_diag(d):
        reg = _R()

        def f():
            r = reg.ols(d["ys"], d["Xs"])
            return (
                r.leverage,
                r.resid_studentized_internal,
                r.cooks_distance,
                r.durbin_watson,
                r.jarque_bera(),
                r.breusch_pagan(),
            )

        return f

    def hc_row(hc):
        return (
            f"ols(y, X, cov_type='{hc}'): bse, pvalues",
            "",
            lambda d: lambda r=_R(): _bp(r.ols(d["ys"], d["Xs"], cov_type=hc)),
            lambda d: lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit(cov_type=hc, use_t=True)),
        )

    out = []
    for i, (n, k) in enumerate(G1B_SIZES):
        setup = _g1b_data(i)
        L = int(math.floor(4 * (n / 100) ** (2 / 9)))
        rows = [
            (
                "ols(y, X): bse, pvalues",
                "",
                lambda d: lambda r=_R(): _bp(r.ols(d["ys"], d["Xs"])),
                lambda d: lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit()),
            ),
            *(hc_row(hc) for hc in (("HC0", "HC1", "HC2", "HC3") if n == 100_000 else ())),
            (
                "ols(y, X, cov_type='HAC'): bse, pvalues",
                f"지연 {L}개",
                lambda d: lambda r=_R(): _bp(r.ols(d["ys"], d["Xs"], cov_type="HAC")),
                lambda d: (
                    lambda: _bp(
                        _sm().OLS(d["y"], d["Xc"]).fit(cov_type="HAC", cov_kwds={"maxlags": d["L"]}, use_t=True)
                    )
                ),
            ),
            (
                "ols(y, X, cov_type='cluster', groups=g): bse, pvalues",
                "군집 1,000개",
                lambda d: lambda r=_R(): _bp(r.ols(d["ys"], d["Xs"], cov_type="cluster", groups=d["gs"])),
                lambda d: (
                    lambda: _bp(
                        _sm().OLS(d["y"], d["Xc"]).fit(cov_type="cluster", cov_kwds={"groups": d["g"]}, use_t=True)
                    )
                ),
            ),
            ("ols(y, X) + diagnostics", "지렛대·내적 스튜던트화 잔차·Cook 거리·DW·JB·BP", sd_diag, sm_diag),
            (
                "ols(y, X).summary()",
                "",
                lambda d: lambda r=_R(): r.ols(d["ys"], d["Xs"]).summary(),
                lambda d: lambda: str(_sm().OLS(d["y"], d["Xc"]).fit().summary()),
            ),
        ]
        for name, note, sdf, smf in rows:
            out.append(
                Case(
                    area="regression",
                    group="ols-cov-diagnostics",
                    name=name,
                    size=_size(n, k),
                    setup=setup,
                    sd=sdf,
                    refs=(Ref("statsmodels", smf, threads="follow"),),
                    note=note,
                    quick=(name.startswith("ols(y, X, cov_type='HAC')") and n == 100_000),
                )
            )
        out.append(
            Case(
                area="regression",
                group="ols-cov-diagnostics",
                name="ols(y, X).resid_studentized_external",
                size=_size(n, k),
                setup=setup,
                sd=lambda d: lambda r=_R(): r.ols(d["ys"], d["Xs"]).resid_studentized_external,
                note="statsmodels는 행마다 다시 적합해(n번) 재지 않았다",
            )
        )
    return out


# ---------------------------------------------------------------- g2: glm

G2_SIZES = ((100_000, 10), (1_000_000, 50))
G2_BATCH = (10_000, 50, 3)
GLM_SHAPES = ((200, 5), (10_000, 10), (10_000, 500))


def _g2_draw(rng, n, k):
    import numpy as np

    X = rng.standard_normal((n, k))
    beta = rng.standard_normal(k) * (0.6 / np.sqrt(k))
    eta = 0.2 + X @ beta
    yl = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    yp = rng.poisson(np.exp(0.3 + 0.5 * (X @ beta))).astype(float)
    return X, yl, yp


def _glm_pack(X, y):
    import stardust as sd

    return {"X": X, "y": y, "Xc": _sm().add_constant(X), "Xs": sd.asarray(X), "ys": sd.asarray(y)}


def _g2_data(index, fam=None):
    def setup():
        import numpy as np

        import stardust as sd

        if index == len(G2_SIZES):
            rng = _np_rng(_SKIP[("g2", "batch")])
            B, n, k = G2_BATCH
            Xb = rng.standard_normal((B, n, k))
            Yb = (rng.uniform(size=(B, n)) < 1 / (1 + np.exp(-(0.2 + Xb @ np.array([0.5, -0.3, 0.2]))))).astype(float)
            sm = _sm()
            return {
                "Xb": Xb,
                "Yb": Yb,
                "Xbs": sd.asarray(Xb),
                "Ybs": sd.asarray(Yb),
                "Xbc": [sm.add_constant(Xb[i]) for i in range(B)],
            }
        rng = np.random.default_rng(0)
        for n, k in G2_SIZES[:index]:
            _g2_draw(rng, n, k)
        X, yl, yp = _g2_draw(rng, *G2_SIZES[index])
        return _glm_pack(X, yl if fam == "binomial" else yp)

    return setup


def _glm_shape_data(n, k):
    """Binomial data of the g2 generator at another shape (seed 1)."""

    def setup():
        import numpy as np

        X, yl, _ = _g2_draw(np.random.default_rng(1), n, k)
        return _glm_pack(X, yl)

    return setup


def _sd_glm(d, **kw):
    reg = _R()

    def f():
        r = reg.glm(d["ys"], d["Xs"], **kw)
        return r.params, r.bse, r.pvalues, r.n_iter

    return _quiet(f)


def _sm_glm(d, family, link=None, fit=None, model=None):
    """statsmodels GLM of `family` (and `link`), with fit(**fit) and model keywords taken
    from the data (model: keyword -> data key); the response is d["y2"] if there is one."""
    sm = _sm()
    fams = sm.families
    fam = getattr(fams, family)(link=getattr(fams.links, link)()) if link else getattr(fams, family)()
    kw = {k: d[v] for k, v in (model or {}).items()}

    def f():
        r = sm.GLM(d.get("y2", d["y"]), d["Xc"], family=fam, **kw).fit(**(fit or {}))
        return r.params, r.bse, r.pvalues

    return _quiet(f)


def _sk_glm(d, fam, solver):
    import numpy as np

    if fam == "binomial":
        kw = dict(cls="LogisticRegression", C=np.inf)
    else:
        kw = dict(cls="PoissonRegressor", alpha=0.0)
    kw.update(dict(solver=solver) if solver != "lbfgs" else dict(max_iter=10000))
    return _quiet(lambda: _sk_tol(tol=1e-10, **kw).fit(d["X"], d["y"]))


GLM_SK_DEFAULT = {
    "binomial": "scikit-learn 기본 tol이면 계수의 상대 오차 lbfgs 6.5e-4(4번 반복), newton-cholesky 8e-8",
    "poisson": "scikit-learn 기본 tol이면 계수의 상대 오차 lbfgs 2.2e-5, newton-cholesky 3.3e-7",
}


def _glm_case(n, k, fam, setup, quick=False, extra=""):
    note = "sd·statsmodels는 params·bse·pvalues까지, scikit-learn은 적합만"
    if (n, k) == (100_000, 10):
        note += "; " + GLM_SK_DEFAULT[fam]
    if extra:
        note += "; " + extra
    return Case(
        area="regression",
        group="glm",
        name=f"glm(y, X, family='{fam}')",
        size=_size(n, k),
        setup=setup,
        sd=lambda d: _sd_glm(d, family=fam),
        refs=(
            Ref("statsmodels", lambda d: _sm_glm(d, fam.capitalize()), threads="follow"),
            Ref("scikit-learn lbfgs tol=1e-10", lambda d: _sk_glm(d, fam, "lbfgs"), threads="follow", lib="sklearn"),
            Ref(
                "scikit-learn newton-cholesky tol=1e-10",
                lambda d: _sk_glm(d, fam, "newton-cholesky"),
                threads="follow",
                lib="sklearn",
            ),
        ),
        quick=quick,
        note=note,
    )


def _g2_cases():
    out = []
    for i, (n, k) in enumerate(G2_SIZES):
        for fam in ("binomial", "poisson"):
            setup = _g2_data(i, fam)
            out.append(_glm_case(n, k, fam, setup, quick=(fam == "binomial" and n == 100_000)))
            if n != 100_000:
                continue
            out.append(
                Case(
                    area="regression",
                    group="glm",
                    name=f"glm(y, X, family='{fam}'), inner iterations by QR",
                    size=_size(n, k),
                    setup=setup,
                    sd=lambda d, fam=fam: _kernel_option("_glm", "glm_fit", _sd_glm(d, family=fam), inner_chol=False),
                    note="공개 함수에 커널 선택지 inner_chol=False를 붙여 부른다(벤치마크 전용)",
                )
            )
            if fam == "binomial":
                out.append(
                    Case(
                        area="regression",
                        group="glm",
                        name="glm(y, X, family='binomial', cov_type='HC0')",
                        size=_size(n, k),
                        setup=setup,
                        sd=lambda d: _sd_glm(d, family="binomial", cov_type="HC0"),
                        refs=(
                            Ref(
                                "statsmodels",
                                lambda d: _sm_glm(d, "Binomial", fit={"cov_type": "HC0"}),
                                threads="follow",
                            ),
                        ),
                        note="params·bse·pvalues까지",
                    )
                )
    for n, k in GLM_SHAPES:
        out.append(
            _glm_case(
                n, k, "binomial", _glm_shape_data(n, k), extra="2^15·2^16행 문턱 아래" if (n, k) == (10_000, 10) else ""
            )
        )
    setup = _g2_data(len(G2_SIZES))
    B, n, k = G2_BATCH

    def sd_batch(d):
        reg = _R()

        def f():
            r = reg.glm(d["Ybs"], d["Xbs"], family="binomial")
            return r.params, r.bse, r.pvalues, r.separation

        return _quiet(f)

    def sd_loop(d):
        reg = _R()

        def f():
            for i in range(B):
                r = reg.glm(d["Yb"][i], d["Xb"][i], family="binomial")
                r.bse, r.pvalues

        return _quiet(f)

    def sm_loop(d):
        sm = _sm()

        def f():
            for i in range(B):
                r = sm.GLM(d["Yb"][i], d["Xbc"][i], family=sm.families.Binomial()).fit()
                r.bse, r.pvalues

        return _quiet(f)

    size = f"{B:,} × ({n}, {k})"
    out.append(
        Case(
            area="regression",
            group="glm-batched",
            name="glm(Y, X, family='binomial') + inference, one call",
            size=size,
            setup=setup,
            sd=sd_batch,
            refs=(Ref("statsmodels loop", sm_loop, threads="follow", lib="statsmodels"),),
            note="statsmodels는 Python 반복문으로 하나씩 적합한다",
        )
    )
    out.append(
        Case(
            area="regression",
            group="glm-batched",
            name="glm(y_i, X_i, family='binomial') + inference, Python loop",
            size=size,
            setup=setup,
            sd=sd_loop,
            note="numpy 배열을 그대로 넘긴다",
        )
    )
    return out


def _fam_data(n, k, kind):
    """Data of one family and link (seed 2): 'probit', 'gamma' (log link), 'exposure'
    (Poisson with an exposure), 'trials' (binomial proportions out of 1-20 trials)."""

    def setup():
        import numpy as np
        from scipy.special import expit, ndtr

        import stardust as sd

        rng = np.random.default_rng(2)
        X = rng.standard_normal((n, k))
        eta = 0.2 + X @ (rng.standard_normal(k) * (0.6 / np.sqrt(k)))
        d = {}
        if kind == "probit":
            y = (rng.uniform(size=n) < ndtr(eta)).astype(float)
        elif kind == "gamma":
            y = rng.gamma(2.0, np.exp(eta) / 2.0)
        elif kind == "exposure":
            e = rng.uniform(0.5, 2.0, n)
            y = rng.poisson(e * np.exp(eta)).astype(float)
            d.update(e=e, es=sd.asarray(e))
        else:
            t = rng.integers(1, 21, n).astype(float)
            s = rng.binomial(t.astype(np.int64), expit(eta)).astype(float)
            y = s / t
            d.update(t=t, ts=sd.asarray(t), y2=np.column_stack([s, t - s]))
        d.update(_glm_pack(X, y))
        return d

    return setup


def _fam_cases():
    rows = [
        ("glm(y, X, family='binomial', link='probit')", "probit", {"family": "binomial", "link": "probit"},
         ("Binomial", "Probit", {}), ((100_000, 10), (1_000_000, 50))),
        ("glm(y, X, family='gamma', link='log')", "gamma", {"family": "gamma", "link": "log"},
         ("Gamma", "Log", {}), ((100_000, 10),)),
        ("glm(y, X, family='poisson', exposure=e)", "exposure", {"family": "poisson"},
         ("Poisson", None, {"exposure": "e"}), ((100_000, 10),)),
        ("glm(y, X, family='binomial', trials=t)", "trials", {"family": "binomial"},
         ("Binomial", None, {}), ((100_000, 10),)),
    ]  # fmt: skip
    notes = {
        "exposure": "노출 U(0.5, 2)",
        "trials": "시행 1–20번의 비율; statsmodels는 (성공, 실패) 두 열 반응",
    }
    out = []
    for name, kind, sdkw, (family, link, model), sizes in rows:
        for n, k in sizes:

            def sdf(d, kind=kind, sdkw=sdkw):
                extra = {"exposure": d["es"]} if kind == "exposure" else {"trials": d["ts"]} if kind == "trials" else {}
                return _sd_glm(d, **sdkw, **extra)

            out.append(
                Case(
                    area="regression",
                    group="glm-families",
                    name=name,
                    size=_size(n, k),
                    setup=_fam_data(n, k, kind),
                    sd=sdf,
                    refs=(
                        Ref(
                            "statsmodels",
                            lambda d, family=family, link=link, model=model: _sm_glm(d, family, link, model=model),
                            threads="follow",
                        ),
                    ),
                    quick=(kind == "probit" and n == 100_000),
                    note="; ".join(x for x in ("params·bse·pvalues까지", notes.get(kind, "")) if x),
                )
            )
    return out


# ---------------------------------------------------------------- input conditions

COND_N, COND_K = 100_000, 10


def _cond_data(kind):
    """(10^5, 10) designs that are hard in one way (seed 3): 'cond4' / 'cond7' (2-norm
    condition number 10^4 / 10^7), 'scale' (column scales from 1 to 10^8), 'collinear' (the
    last column is exactly twice the first), 'separation' (y = [x_1 > 0], logistic only)."""

    def setup():
        import numpy as np

        import stardust as sd

        n, k = COND_N, COND_K
        rng = np.random.default_rng(3)
        X = rng.standard_normal((n, k))
        b = rng.standard_normal(k)
        if kind in ("cond4", "cond7"):
            Q, _ = np.linalg.qr(X)
            V, _ = np.linalg.qr(rng.standard_normal((k, k)))
            s = np.geomspace(1.0, 10.0 ** -int(kind[-1]), k)
            X = (Q * (s * math.sqrt(n))) @ V.T
            b = V @ (b / s) / math.sqrt(k)  # every direction carries signal
        elif kind == "scale":
            sc = np.geomspace(1.0, 1e8, k)
            X = X * sc
            b = b / sc
        elif kind == "collinear":
            X[:, -1] = 2.0 * X[:, 0]
        eta = X @ b
        eta = 0.6 * eta / eta.std()
        e = rng.standard_normal(n)
        u = rng.uniform(size=n)
        y = eta + e
        yl = (X[:, 0] > 0).astype(float) if kind == "separation" else (u < 1 / (1 + np.exp(-(0.2 + eta)))).astype(float)
        return {
            "X": X,
            "y": y,
            "yl": yl,
            "Xc": _sm().add_constant(X),
            "Xs": sd.asarray(X),
            "ys": sd.asarray(y),
            "yls": sd.asarray(yl),
        }

    return setup


COND_LABEL = {
    "cond4": "condition number 10^4",
    "cond7": "condition number 10^7",
    "scale": "column scales 1–10^8",
    "collinear": "one duplicated column",
    "separation": "complete separation",
}


def _cond_cases():
    def sd_glm(d):
        reg = _R()

        def f():
            r = reg.glm(d["yls"], d["Xs"], family="binomial")
            return r.params, r.bse, r.pvalues

        return _quiet(f)

    def sm_glm(d):
        sm = _sm()

        def f():
            r = sm.GLM(d["yl"], d["Xc"], family=sm.families.Binomial()).fit()
            return r.params, r.bse, r.pvalues

        return _quiet(f, _sm_warning("PerfectSeparationWarning"), _sm_warning("SingularMatrixWarning"))

    def sd_qr(d):
        return _quiet(lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=0.5, cov_type="ker"))

    def sm_qr(d):
        return _quiet(
            lambda: _sm().QuantReg(d["y"], d["Xc"]).fit(q=0.5),
            _sm_warning("IterationLimitWarning"),
            _sm_warning("SingularMatrixWarning"),
        )

    def sm_ols(d):
        return _quiet(lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit()), _sm_warning("SingularMatrixWarning"))

    rows = [
        ("ols(y, X)", ("cond4", "cond7", "scale", "collinear"),
         lambda d: lambda r=_R(): _bp(r.ols(d["ys"], d["Xs"])),
         sm_ols),
        ("glm(y, X, family='binomial')", ("cond4", "cond7", "scale", "collinear", "separation"), sd_glm, sm_glm),
        ("quantreg(y, X, q=0.5, cov_type='ker')", ("cond7", "scale", "collinear"), sd_qr, sm_qr),
    ]  # fmt: skip
    notes = {
        "ols(y, X)": "bse·pvalues까지",
        "glm(y, X, family='binomial')": "params·bse·pvalues까지",
        "quantreg(y, X, q=0.5, cov_type='ker')": "statsmodels QuantReg는 근사 IRLS",
    }
    out = []
    for name, kinds, sdf, smf in rows:
        for kind in kinds:
            note = notes[name]
            if kind == "separation":
                note += "; sd glm은 아직 완전 분리를 알아내지 못한다(보고한 라이브러리 버그), 고쳐지면 시간이 바뀐다"
            out.append(
                Case(
                    area="regression",
                    group="conditioning",
                    name=f"{name}, {COND_LABEL[kind]}",
                    size=_size(COND_N, COND_K),
                    setup=_cond_data(kind),
                    sd=sdf,
                    refs=(Ref("statsmodels", smf, threads="follow"),),
                    note=note,
                )
            )
    return out


# ---------------------------------------------------------------- g3: penalized

G3_SIZES = ((100_000, 10), (100_000, 100), (10_000, 1000))
G3_EXTRA = ((200, 5), (1_000, 5_000))  # small n, and k > n


def _g3_data(n, k):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        beta = np.zeros(k)
        m = min(10, k)
        beta[:m] = rng.standard_normal(m)
        y = X @ beta + rng.standard_normal(n)
        yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta / 2))).astype(float)
        yp = rng.poisson(np.exp(0.2 + X @ beta / 4)).astype(float)
        Xs, ys = sd.asarray(X), sd.asarray(y)
        amax = float(np.max(np.abs((X - X.mean(0)).T @ (y - y.mean()))) / n)  # the lasso's smallest all-zero alpha
        grid = amax * np.geomspace(1, 1e-3, 100)
        fold = np.asarray(
            sd.take(sd.asarray([i % 5 for i in range(n)], dtype=sd.int64), sd.random.default_rng(1).permutation(n))
        )
        splits = [(np.nonzero(fold != q)[0], np.nonzero(fold == q)[0]) for q in range(5)]
        rgrid = np.geomspace(10.0, 1e-4, 100)
        return {
            "X": X,
            "y": y,
            "yb": yb,
            "yp": yp,
            "Xs": Xs,
            "ys": ys,
            "ybs": sd.asarray(yb),
            "yps": sd.asarray(yp),
            "grid": grid,
            "gs": sd.asarray(grid),
            "a20": amax / 20,
            "splits": splits,
            "rgrid": rgrid,
            "rgs": sd.asarray(rgrid),
            "n": n,
            "C": 1 / (n * 1e-3),
        }

    return setup


def _g3_cases():
    def sdq(f):
        return lambda d: _quiet(lambda: f(_R(), d))

    def skq(f):
        return lambda d: _quiet(lambda: f(d))

    T = "scikit-learn tol=1e-10"
    ridge = (
        "ridge(y, X, alpha=0.1)",
        sdq(lambda r, d: r.ridge(d["ys"], d["Xs"], alpha=0.1)),
        (
            ("scikit-learn Ridge", skq(lambda d: _sk_tol("Ridge", alpha=0.1 * d["n"]).fit(d["X"], d["y"]))),
            (
                "scikit-learn Ridge svd",
                skq(lambda d: _sk_tol("Ridge", alpha=0.1 * d["n"], solver="svd").fit(d["X"], d["y"])),
            ),
        ),
        "scikit-learn은 alpha = 0.1 n(같은 문제)",
    )
    lasso = (
        "lasso(y, X, alpha=alpha_max/20)",
        sdq(lambda r, d: r.lasso(d["ys"], d["Xs"], alpha=d["a20"])),
        ((T, skq(lambda d: _sk_tol("Lasso", alpha=d["a20"], tol=1e-10).fit(d["X"], d["y"]))),),
        "",
    )
    fits = [
        ridge,
        lasso,
        (
            "elastic_net(y, X, alpha=alpha_max/20)",
            sdq(lambda r, d: r.elastic_net(d["ys"], d["Xs"], alpha=d["a20"])),
            ((T, skq(lambda d: _sk_tol("ElasticNet", alpha=d["a20"], tol=1e-10).fit(d["X"], d["y"]))),),
            "",
        ),
        (
            "lasso(y, X, alpha=100-point grid)",
            sdq(lambda r, d: r.lasso(d["ys"], d["Xs"], alpha=d["gs"])),
            (
                (
                    T,
                    skq(
                        lambda d: _sk_fn("lasso_path")(
                            d["X"] - d["X"].mean(0), d["y"] - d["y"].mean(), alphas=d["grid"], tol=1e-10
                        )
                    ),
                ),
            ),
            "scikit-learn은 lasso_path(중심화 포함)",
        ),
    ]
    cvs = [
        (
            "lasso_cv(y, X, alphas=100-point grid)",
            sdq(lambda r, d: r.lasso_cv(d["ys"], d["Xs"], rng=_sd_rng(1), alphas=d["gs"])),
            ((T, skq(lambda d: _sk_tol("LassoCV", alphas=d["grid"], cv=d["splits"], tol=1e-10).fit(d["X"], d["y"]))),),
            "5겹(같은 접기)",
        ),
        (
            "elastic_net_cv(y, X, alphas=100-point grid)",
            sdq(lambda r, d: r.elastic_net_cv(d["ys"], d["Xs"], rng=_sd_rng(1), alphas=d["gs"])),
            (
                (
                    T,
                    skq(
                        lambda d: _sk_tol("ElasticNetCV", alphas=d["grid"], cv=d["splits"], tol=1e-10).fit(
                            d["X"], d["y"]
                        )
                    ),
                ),
            ),
            "l1_ratio = 0.5, 5겹(같은 접기)",
        ),
        (
            "ridge_cv(y, X, cv='gcv', alphas=100)",
            sdq(lambda r, d: r.ridge_cv(d["ys"], d["Xs"], cv="gcv", alphas=d["rgs"])),
            (("scikit-learn", skq(lambda d: _sk_tol("RidgeCV", alphas=d["rgrid"] * d["n"]).fit(d["X"], d["y"]))),),
            "",
        ),
    ]
    pglm = [
        (
            "penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=1)",
            sdq(lambda r, d: r.penalized_glm(d["ybs"], d["Xs"], family="binomial", alpha=1e-3, l1_ratio=1.0)),
            (
                (
                    "scikit-learn saga tol=1e-8",
                    skq(
                        lambda d: _sk_tol(
                            "LogisticRegression", C=d["C"], l1_ratio=1.0, solver="saga", tol=1e-8, max_iter=100_000
                        ).fit(d["X"], d["yb"])
                    ),
                ),
                (
                    "scikit-learn liblinear",
                    skq(
                        lambda d: _sk_tol("LogisticRegression", C=d["C"], l1_ratio=1.0, solver="liblinear").fit(
                            d["X"], d["yb"]
                        )
                    ),
                ),
            ),
            "liblinear는 절편에도 벌점을 준다(다른 문제라 기본 tol)",
        ),
        (
            "penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=0)",
            sdq(lambda r, d: r.penalized_glm(d["ybs"], d["Xs"], family="binomial", alpha=1e-3, l1_ratio=0.0)),
            (
                (
                    "scikit-learn lbfgs tol=1e-10",
                    skq(lambda d: _sk_tol("LogisticRegression", C=d["C"], tol=1e-10).fit(d["X"], d["yb"])),
                ),
                (
                    "scikit-learn newton-cholesky tol=1e-10",
                    skq(
                        lambda d: _sk_tol("LogisticRegression", C=d["C"], solver="newton-cholesky", tol=1e-10).fit(
                            d["X"], d["yb"]
                        )
                    ),
                ),
            ),
            "",
        ),
        (
            "penalized_glm(y, X, family='poisson', alpha=1e-3, l1_ratio=0)",
            sdq(lambda r, d: r.penalized_glm(d["yps"], d["Xs"], family="poisson", alpha=1e-3, l1_ratio=0.0)),
            (
                (
                    "scikit-learn lbfgs tol=1e-10",
                    skq(
                        lambda d: _sk_tol("PoissonRegressor", alpha=1e-3, tol=1e-10, max_iter=10000).fit(
                            d["X"], d["yp"]
                        )
                    ),
                ),
                (
                    "scikit-learn newton-cholesky tol=1e-10",
                    skq(
                        lambda d: _sk_tol("PoissonRegressor", alpha=1e-3, solver="newton-cholesky", tol=1e-10).fit(
                            d["X"], d["yp"]
                        )
                    ),
                ),
            ),
            "scikit-learn PoissonRegressor(alpha가 같다)",
        ),
    ]
    # the coefficient error of scikit-learn's default tol, where it was measured
    default_err = {
        ("lasso(y, X, alpha=alpha_max/20)", (100_000, 10)): "기본 tol이면 상대 오차 6–9e-8",
        ("elastic_net(y, X, alpha=alpha_max/20)", (100_000, 10)): "기본 tol이면 상대 오차 5–8e-8",
        ("lasso(y, X, alpha=100-point grid)", (100_000, 10)): "기본 tol이면 상대 오차 1.5e-6(10^4 × 1000에서 3e-5)",
        ("penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=1)", (100_000, 10)): (
            "saga 기본 tol이면 상대 오차 3–6e-5"
        ),
        ("penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=0)", (10_000, 1000)): (
            "기본 tol이면 상대 오차 lbfgs 4e-4, newton-cholesky 3e-4"
        ),
        ("penalized_glm(y, X, family='poisson', alpha=1e-3, l1_ratio=0)", (100_000, 10)): (
            "기본 tol이면 상대 오차 lbfgs 1e-4, newton-cholesky 2e-7"
        ),
    }
    quick = {("lasso_cv(y, X, alphas=100-point grid)", (100_000, 10)), ("ridge(y, X, alpha=0.1)", (100_000, 100))}
    out = []

    def add(group, row, n, k, setup):
        name, sdf, refs, note = row
        extra = default_err.get((name, (n, k)))
        out.append(
            Case(
                area="regression",
                group=group,
                name=name,
                size=_size(n, k),
                setup=setup,
                sd=sdf,
                refs=tuple(Ref(label, f, threads="follow", lib="sklearn") for label, f in refs),
                note="; ".join(x for x in (note, extra) if x),
                quick=(name, (n, k)) in quick,
            )
        )

    for n, k in G3_SIZES:
        setup = _g3_data(n, k)
        for row in fits:
            add("penalized", row, n, k, setup)
        for row in cvs:
            add("penalized-cv", row, n, k, setup)
        for row in pglm:
            add("penalized-glm", row, n, k, setup)
    for n, k in G3_EXTRA:
        setup = _g3_data(n, k)
        for row in (ridge, lasso):
            add("penalized", row, n, k, setup)
    return out


# ---------------------------------------------------------------- g4: robust


def _g4_data(n, k, leverage=False):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        y = X @ np.linspace(1.0, 2.0, k) + 0.5 + rng.standard_normal(n)
        y[: n // 20] += 10.0
        if leverage:
            m = n // 5
            X[:m] = rng.normal(5.0, 0.5, (m, k))
            y[:m] = rng.normal(-10.0, 1.0, m)
        return {"X": X, "y": y, "Xs": sd.asarray(X), "ys": sd.asarray(y), "Xc": _sm().add_constant(X)}

    return setup


def _batch_data(B, n, k, noise):
    """B regressions of (n, k) (seed 4): the g4 model (5% outliers) or the g5 model (t_3 noise)."""

    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(4)
        X = rng.standard_normal((B, n, k))
        y = X @ np.linspace(1.0, 2.0, k) + 0.5
        if noise == "outliers":
            y += rng.standard_normal((B, n))
            y[:, : n // 20] += 10.0
        else:
            y += rng.standard_t(3, (B, n))
        sm = _sm()
        return {
            "X": X,
            "y": y,
            "Xs": sd.asarray(X),
            "ys": sd.asarray(y),
            "Xc": [sm.add_constant(X[i]) for i in range(B)],
        }

    return setup


def _slopes_data(n):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n)
        return {"x": x, "y": y, "xs": sd.asarray(x), "ys": sd.asarray(y)}

    return setup


def _sm_rlm(M, collect=False):
    def make(d):
        import gc

        from statsmodels.robust import norms

        sm = _sm()

        def f():
            sm.RLM(d["y"], d["Xc"], M=getattr(norms, M)()).fit()
            if collect:
                gc.collect()  # the fit leaves ~3.6 GB in reference cycles; two of them pass 16 GB

        return _quiet(f)

    return make


def _g4_cases():
    out = []
    for n, k in ((100_000, 10), (1_000_000, 50)):
        setup = _g4_data(n, k)
        for norm, M in (("huber", "HuberT"), ("bisquare", "TukeyBiweight")):
            out.append(
                Case(
                    area="regression",
                    group="robust",
                    name=f"rlm(y, X, norm='{norm}')",
                    size=_size(n, k),
                    setup=setup,
                    sd=lambda d, norm=norm: lambda r=_R(): r.rlm(d["ys"], d["Xs"], norm=norm),
                    refs=(Ref("statsmodels", _sm_rlm(M, collect=n == 1_000_000), threads="follow"),),
                    quick=(norm == "huber" and n == 100_000),
                    note=(
                        "statsmodels 한 번에 20초 안팎, 10–12 GB; 호출마다 gc.collect()를 더한다"
                        "(순환 참조로 남는 메모리, 0.05초)"
                        if n == 1_000_000
                        else ""
                    ),
                )
            )
    for n, k in ((200, 5), (10_000, 10)):
        out.append(
            Case(
                area="regression",
                group="robust",
                name="rlm(y, X, norm='huber')",
                size=_size(n, k),
                setup=_g4_data(n, k),
                sd=lambda d: lambda r=_R(): r.rlm(d["ys"], d["Xs"], norm="huber"),
                refs=(Ref("statsmodels", _sm_rlm("HuberT"), threads="follow"),),
                note="2^15·2^16행 문턱 아래" if n == 10_000 else "",
            )
        )
    for n, k in ((200, 5), (10_000, 5), (100_000, 10), (100_000, 30), (1_000_000, 10)):
        setup = _g4_data(n, k, leverage=True)

        def mm(d):
            return lambda r=_R(): r.rlm(d["ys"], d["Xs"], method="MM", rng=_sd_rng(1))

        out.append(
            Case(
                area="regression",
                group="robust",
                name="rlm(y, X, method='MM')",
                size=_size(n, k),
                setup=setup,
                sd=mm,
                refs=(RRef("R lmrob", "lmrob(y ~ X)", data={"y": "y", "X": "X"}, packages=("robustbase",)),),
                quick=(n, k) == (10_000, 5),
                note="지렛대 무리 20%" + ("; lmrob은 n > 2000에서 묶음 표본 방식" if n > 2000 else ""),
            )
        )
        if (n, k) in ((10_000, 5), (100_000, 10)):
            out.append(
                Case(
                    area="regression",
                    group="robust",
                    name="rlm(y, X, method='MM'), small-n method",
                    size=_size(n, k),
                    setup=setup,
                    sd=lambda d, mm=mm: _kernel_option("_robust", "rlm_fit", mm(d), large_n=False),
                    note="공개 함수에 커널 선택지 large_n=False(모든 행의 후보 500개)를 붙여 부른다(벤치마크 전용)",
                )
            )
    B, n, k = 200, 500, 5

    def sm_rlm_loop(d):
        from statsmodels.robust import norms

        sm = _sm()
        return _quiet(lambda: [sm.RLM(d["y"][i], d["Xc"][i], M=norms.HuberT()).fit() for i in range(B)])

    out.append(
        Case(
            area="regression",
            group="robust-batched",
            name="rlm(Y, X, norm='huber'), one call",
            size=f"{B:,} × ({n}, {k})",
            setup=_batch_data(B, n, k, "outliers"),
            sd=lambda d: lambda r=_R(): r.rlm(d["ys"], d["Xs"], norm="huber"),
            refs=(Ref("statsmodels loop", sm_rlm_loop, threads="follow", lib="statsmodels"),),
            note="statsmodels는 Python 반복문으로 하나씩 적합한다",
        )
    )
    for n in (1_000, 10_000, 100_000):
        setup = _slopes_data(n)
        for name in ("theilslopes", "siegelslopes"):
            refs = ()
            if n <= 10_000:
                refs = (Ref("scipy", lambda d, name=name: lambda f=_scipy_stats(name): f(d["y"], d["x"])),)
            note = ""
            if n == 100_000:
                note = "10스레드만(1스레드는 한 번에 수십 초); scipy는 n × n 배열(80 GB)이라 재지 않았다"
            out.append(
                Case(
                    area="regression",
                    group="robust-slopes",
                    name=f"{name}(y, x)",
                    size=f"{n:,}",
                    n=n,
                    setup=setup,
                    sd=lambda d, name=name: lambda f=getattr(_R(), name): f(d["ys"], d["xs"]),
                    refs=refs,
                    threads=(10,) if n == 100_000 else (1, 10),
                    quick=(name == "theilslopes" and n == 10_000),
                    note=note,
                )
            )
    return out


def _scipy_stats(name):
    import scipy.stats as st

    return getattr(st, name)


# ---------------------------------------------------------------- g5: quantreg

G5_SIZES = ((1_000, 5), (100_000, 10), (1_000_000, 10))
G5_EXTRA = ((200, 5), (10_000, 10), (100_000, 50))  # small n, between the 2^15 / 2^16 thresholds, wide k
QS9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
R_TAUS = "taus <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)"


def _g5_data(n, k):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        y = X @ np.linspace(1.0, 2.0, k) + 0.5 + rng.standard_t(3, n)
        return {"X": X, "y": y, "Xs": sd.asarray(X), "ys": sd.asarray(y), "Xc": _sm().add_constant(X)}

    return setup


def _sm_qr(d, qs=(0.5,)):
    sm = _sm()
    w = _sm_warning("IterationLimitWarning")
    if len(qs) == 1:
        return _quiet(lambda: sm.QuantReg(d["y"], d["Xc"]).fit(q=qs[0]), w)
    return _quiet(lambda: [sm.QuantReg(d["y"], d["Xc"]).fit(q=q) for q in qs], w)


SM_QR_NOTE = "statsmodels QuantReg는 근사 IRLS이고 기본 fit이 커널 샌드위치를 계산해 sd의 'ker'와 짝짓는다"


def _g5_cases():
    rdata = {"y": "y", "X": "X"}
    rsetup = "x <- cbind(1, X)"
    out = []
    for n, k in sorted(G5_SIZES + G5_EXTRA):
        setup = _g5_data(n, k)
        refs = [
            RRef("R fn", 'rq.fit(x, y, tau = 0.5, method = "fn")', data=rdata, setup=rsetup, packages=("quantreg",))
        ]
        br = n <= 100_000 and k <= 10
        if br:
            refs.append(
                RRef("R br", 'rq.fit(x, y, tau = 0.5, method = "br")', data=rdata, setup=rsetup, packages=("quantreg",))
            )
        note = (
            "" if br else "R br은 한 번에 몇 분이라 재지 않았다" if n > 100_000 else "R br은 한 번에 12초 안팎이라 뺐다"
        )
        if n == 10_000:
            note = "2^15·2^16행 문턱 아래"
        out.append(
            Case(
                area="regression",
                group="quantreg",
                name="quantreg(y, X, q=0.5, cov_type='iid')",
                size=_size(n, k),
                setup=setup,
                sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=0.5, cov_type="iid"),
                refs=tuple(refs),
                quick=(n, k) == (100_000, 10),
                note=note,
            )
        )
        out.append(
            Case(
                area="regression",
                group="quantreg",
                name="quantreg(y, X, q=0.5, cov_type='ker')",
                size=_size(n, k),
                setup=setup,
                sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=0.5, cov_type="ker"),
                refs=(Ref("statsmodels", _sm_qr, threads="follow"),),
                note=SM_QR_NOTE if (n, k) == (100_000, 10) else "",
            )
        )
        if (n, k) in G5_SIZES:
            out.append(
                Case(
                    area="regression",
                    group="quantreg",
                    name="quantreg(y, X, q=0.5)",
                    size=_size(n, k),
                    setup=setup,
                    sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=0.5),
                    refs=(
                        RRef(
                            "R fn + nid",
                            'g <- rq(y ~ X, tau = 0.5, method = "fn"); suppressWarnings(summary(g, se = "nid"))',
                            data=rdata,
                            packages=("quantreg",),
                        ),
                    ),
                    quick=n == 1_000,
                    note="기본 cov_type='nid'; R은 fn 적합과 summary(se = \"nid\")",
                )
            )
    n, k = 100_000, 10
    setup = _g5_data(n, k)
    out.append(
        Case(
            area="regression",
            group="quantreg-9q",
            name="quantreg(y, X, q=[0.1, ..., 0.9], cov_type='iid')",
            size=_size(n, k),
            setup=setup,
            sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=QS9, cov_type="iid"),
            refs=(
                RRef(
                    "R fn",
                    'for (t in taus) rq.fit(x, y, tau = t, method = "fn")',
                    data=rdata,
                    setup=f"{rsetup}; {R_TAUS}",
                    packages=("quantreg",),
                ),
            ),
            note="R은 분위마다 한 번씩; R br(한 번에 25초 안팎)은 뺐다",
        )
    )
    out.append(
        Case(
            area="regression",
            group="quantreg-9q",
            name="quantreg(y, X, q=[0.1, ..., 0.9], cov_type='ker')",
            size=_size(n, k),
            setup=setup,
            sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=QS9, cov_type="ker"),
            refs=(Ref("statsmodels", lambda d: _sm_qr(d, QS9), threads="follow"),),
            note="statsmodels는 분위마다 한 번씩",
        )
    )
    out.append(
        Case(
            area="regression",
            group="quantreg-9q",
            name="quantreg(y, X, q=q, cov_type='iid'), one fit per quantile",
            size=_size(n, k),
            setup=setup,
            sd=lambda d: lambda r=_R(): [r.quantreg(d["ys"], d["Xs"], q=q, cov_type="iid") for q in QS9],
        )
    )
    B, n, k = 200, 500, 5

    def sm_loop(d):
        sm = _sm()
        return _quiet(
            lambda: [sm.QuantReg(d["y"][i], d["Xc"][i]).fit(q=0.5) for i in range(B)],
            _sm_warning("IterationLimitWarning"),
        )

    out.append(
        Case(
            area="regression",
            group="quantreg-batched",
            name="quantreg(Y, X, q=0.5, cov_type='ker'), one call",
            size=f"{B:,} × ({n}, {k})",
            setup=_batch_data(B, n, k, "t3"),
            sd=lambda d: lambda r=_R(): r.quantreg(d["ys"], d["Xs"], q=0.5, cov_type="ker"),
            refs=(Ref("statsmodels loop", sm_loop, threads="follow", lib="statsmodels"),),
            note="statsmodels는 Python 반복문으로 하나씩 적합한다",
        )
    )
    return out


# ---------------------------------------------------------------- g6: nls

G6_NS = (100, 10_000, 1_000_000)
CENTRES = (0.7, 1.6, 2.5, 3.4, 4.3)


def _models(xp):
    exp = xp.exp
    m = {}
    m[2] = (
        lambda x, a, b: a * exp(-b * x),
        lambda x, a, b: xp.stack([exp(-b * x), -a * x * exp(-b * x)], axis=1),
        [2.0, 0.8],
    )

    def f4(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c) ** b)

    def j4(x, a, b, c, d):
        u = (x / c) ** b
        den = 1 + u
        lg = xp.log(x / c)
        return xp.stack([1 / den, -(a - d) * u * lg / den**2, (a - d) * u * b / c / den**2, 1 - 1 / den], axis=1)

    m[4] = (f4, j4, [0.2, 2.5, 1.8, 3.0])

    def f10(x, *t):
        return sum(t[2 * k] * exp(-(((x - t[2 * k + 1]) / 0.4) ** 2)) for k in range(5))

    def j10(x, *t):
        cols = []
        for k in range(5):
            a, c = t[2 * k], t[2 * k + 1]
            g = exp(-(((x - c) / 0.4) ** 2))
            cols += [g, a * g * 2 * (x - c) / 0.16]
        return xp.stack(cols, axis=1)

    m[10] = (f10, j10, [1.0, CENTRES[0], 1.5, CENTRES[1], 0.8, CENTRES[2], 1.2, CENTRES[3], 0.6, CENTRES[4]])
    return m


R_FORMULA = {
    2: ("y ~ a*exp(-b*x)", ["a", "b"]),
    4: ("y ~ d + (a - d)/(1 + (x/c)^b)", ["a", "b", "c", "d"]),
    10: (
        "y ~ " + " + ".join(f"a{k}*exp(-((x - c{k})/0.4)^2)" for k in range(5)),
        [v for k in range(5) for v in (f"a{k}", f"c{k}")],
    ),
}
G6_TRUTH = {
    2: [2.0, 0.8],
    4: [0.2, 2.5, 1.8, 3.0],
    10: [1.0, CENTRES[0], 1.5, CENTRES[1], 0.8, CENTRES[2], 1.2, CENTRES[3], 0.6, CENTRES[4]],
}
G6_MODEL = {2: "a exp(-b x)", 4: "4모수 로지스틱", 10: "가우스 5개"}
G6_NFEV = {2: (57, 10), 4: (112, 21), 10: (301, 45)}  # f calls of sd and of curve_fit at n = 10^4
BOUNDS = (0.0, 10.0)  # every parameter; no bound is active at the solution


def _p0(p):
    truth = G6_TRUTH[p]
    if p == 10:
        return [v * 1.1 if j % 2 == 0 else v + 0.05 for j, v in enumerate(truth)]
    return [v * 1.1 for v in truth]


def _g6_data(n, p):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        x = rng.uniform(0.01, 5.0, n)
        f, _, truth = _models(np)[p]
        s = f(x, *truth)
        y = s + 0.02 * (s.max() - s.min()) * rng.standard_normal(n)
        w = np.random.default_rng(1).uniform(0.5, 2.0, n)
        return {
            "x": x,
            "y": y,
            "w": w,
            "xs": sd.asarray(x),
            "ys": sd.asarray(y),
            "ws": sd.asarray(w),
            "msd": _models(sd)[p],
            "mnp": _models(np)[p],
            "p0": _p0(p),
        }

    return setup


def _g6_cases():
    def curve_fit(d, jac=False, **kw):
        from scipy.optimize import curve_fit as cf

        fn, Jn, _ = d["mnp"]
        if jac:
            kw["jac"] = Jn
        if kw.pop("weighted", False):
            kw["sigma"] = 1 / d["w"] ** 0.5
        return _quiet(lambda: cf(fn, d["x"], d["y"], p0=d["p0"], full_output=True, **kw))

    def rnls(p, extra="", weighted=False):
        form, names = R_FORMULA[p]
        start = ", ".join(f"{a} = {v!r}" for a, v in zip(names, _p0(p)))
        if weighted:
            extra += ", weights = w"
        return RRef(
            "R nls",
            f"nls({form}, data = dat, start = list({start}){extra})",
            data={"x": "x", "y": "y", **({"w": "w"} if weighted else {})},
            setup="dat <- data.frame(y = y, x = x" + (", w = w)" if weighted else ")"),
        )

    out = []
    for n in G6_NS:
        for p in (2, 4, 10):
            setup = _g6_data(n, p)
            slow = n == 1_000_000 and p == 10
            sd_calls, sp_calls = G6_NFEV[p]
            out.append(
                Case(
                    area="regression",
                    group="nls",
                    name="nls(f, x, y, p0)",
                    size=f"n={n:,}, p={p}",
                    setup=setup,
                    sd=lambda d: lambda r=_R(), f=d["msd"][0]: r.nls(f, d["xs"], d["ys"], p0=d["p0"]),
                    refs=(Ref("scipy curve_fit", curve_fit), rnls(p)),
                    quick=(n == 10_000 and p == 4),
                    note=f"{G6_MODEL[p]}; sd는 5점 차분, scipy는 2점 차분"
                    + (
                        f"; f 호출 sd {sd_calls}번, scipy {sp_calls}번(해의 정확한 야코비안을 위한 설계의 맞바꿈)"
                        if n == 10_000
                        else ""
                    )
                    + ("; sd 1스레드 한 번에 10초 넘는다" if slow else ""),
                )
            )
            out.append(
                Case(
                    area="regression",
                    group="nls",
                    name="nls(f, x, y, p0, jac=J)",
                    size=f"n={n:,}, p={p}",
                    setup=setup,
                    sd=lambda d: (
                        lambda r=_R(), f=d["msd"][0], J=d["msd"][1]: r.nls(f, d["xs"], d["ys"], p0=d["p0"], jac=J)
                    ),
                    refs=(Ref("scipy curve_fit", lambda d: curve_fit(d, jac=True)),),
                    note=f"{G6_MODEL[p]}; 해석적 야코비안",
                )
            )
            if n != 10_000:
                continue
            out.append(
                Case(
                    area="regression",
                    group="nls",
                    name="nls(f, x, y, p0, bounds=(0, 10))",
                    size=f"n={n:,}, p={p}",
                    setup=setup,
                    sd=lambda d: lambda r=_R(), f=d["msd"][0]: r.nls(f, d["xs"], d["ys"], p0=d["p0"], bounds=BOUNDS),
                    refs=(
                        Ref("scipy curve_fit", lambda d: curve_fit(d, bounds=BOUNDS)),
                        rnls(p, ', algorithm = "port", lower = 0, upper = 10'),
                    ),
                    note=f"{G6_MODEL[p]}; 모두 'trf'(R은 port), 해에서 걸리는 경계는 없다",
                )
            )
            out.append(
                Case(
                    area="regression",
                    group="nls",
                    name="nls(f, x, y, p0, weights=w)",
                    size=f"n={n:,}, p={p}",
                    setup=setup,
                    sd=lambda d: lambda r=_R(), f=d["msd"][0]: r.nls(f, d["xs"], d["ys"], p0=d["p0"], weights=d["ws"]),
                    refs=(
                        Ref("scipy curve_fit", lambda d: curve_fit(d, weighted=True)),
                        rnls(p, weighted=True),
                    ),
                    note=f"{G6_MODEL[p]}; 가중치 U(0.5, 2), scipy는 sigma = 1/√w",
                )
            )
    return out


# ---------------------------------------------------------------- all


# The section title of each group in bench/regression/speed.md (the group is the English key in
# the case ids).
TITLES = {
    "ols": "최소제곱",
    "ols-batched": "최소제곱 일괄",
    "wls-gls": "가중·일반화 최소제곱",
    "ols-cov-diagnostics": "최소제곱 공분산·진단",
    "simple": "단순 회귀",
    "glm": "일반화 선형 모형",
    "glm-batched": "일반화 선형 모형 일괄",
    "glm-families": "일반화 선형 모형 분포족·링크",
    "conditioning": "입력 조건",
    "penalized": "벌점 회귀",
    "penalized-cv": "벌점 회귀 교차 검증",
    "penalized-glm": "정규화 GLM",
    "robust": "강건 회귀",
    "robust-batched": "강건 회귀 일괄",
    "robust-slopes": "강건 기울기",
    "quantreg": "분위 회귀",
    "quantreg-9q": "분위 회귀(분위 9개)",
    "quantreg-batched": "분위 회귀 일괄",
    "nls": "비선형 최소제곱",
}


def cases() -> list[Case]:
    return [
        *_g1_cases(),
        *_wls_gls_cases(),
        *_g1b_cases(),
        *_linregress_cases(),
        *_g2_cases(),
        *_fam_cases(),
        *_cond_cases(),
        *_g3_cases(),
        *_g4_cases(),
        *_g5_cases(),
        *_g6_cases(),
    ]
