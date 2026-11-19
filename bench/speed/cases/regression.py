"""sd.regression against statsmodels, scikit-learn, scipy and R. From
bench/regression/g1/speed.py (ols), g1b/speed.py (HAC, cluster, diagnostics, summary),
g2/speed.py (glm), g3/speed.py (penalized), g4/speed.py (rlm M and MM, Theil-Sen,
Siegel), g5/speed.py (quantreg) and g6/speed.py (nls). Only the times are ported: the
iteration counts, coefficient differences and objective excesses of the old tables are
accuracy, not speed.

The data are the old scripts' (same generators and seeds; where a script drew several
sizes from one generator, the earlier draws are repeated so the stream matches).
statsmodels and scikit-learn are timed with their BLAS limited to sd's thread count
(threadpoolctl, as the old scripts did); scipy runs at its defaults; R (one session, its
reference BLAS) once per round. statsmodels gets its constant column beforehand.

Left out as the old scripts did: statsmodels' external studentized residuals (n refits),
scipy's Theil-Sen and Siegel at n = 10^5 (an n x n array, 80 GB), R's rq.fit "br" at
n = 10^6 (minutes). Two cases call a kernel directly with a benchmark-only option (the
glm inner iterations by QR, the MM small-n search on all rows), as the old scripts did.
Some calls take many seconds (statsmodels RLM at (10^6, 50), Theil-Sen and Siegel at
10^5, nls at 10^6 with p = 10, R's br over nine quantiles): their rows say so.
"""

from __future__ import annotations

import math
import warnings

from ..registry import Case, Ref, RRef


def _quiet(f):
    """f with warnings ignored (the old scripts ran these under simplefilter('ignore'))."""

    def g():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return f()

    return g


def _size(n, k):
    return f"({n:,}, {k})"


def _sm():
    import statsmodels.api as sm

    return sm


# ---------------------------------------------------------------- g1: ols

G1_SIZES = ((1_000, 10), (100_000, 10), (1_000_000, 50), (10_000, 1_000))
G1_BATCH = (10_000, 50, 3)


def _g1_data(index):
    """index into G1_SIZES, or len(G1_SIZES) for the batch (one generator over all, in order)."""

    def setup():
        import numpy as np

        import stardust as sd

        sm = _sm()
        rng = np.random.default_rng(0)
        for i, (n, k) in enumerate(G1_SIZES):
            X = rng.standard_normal((n, k))
            y = X @ rng.standard_normal(k) + rng.standard_normal(n)
            if i == index:
                return {"X": X, "y": y, "Xc": sm.add_constant(X), "Xs": sd.asarray(X), "ys": sd.asarray(y)}
            del X, y
        B, n, k = G1_BATCH
        X = rng.standard_normal((B, n, k))
        y = np.einsum("bnk,k->bn", X, [1.0, 2.0, 3.0]) + rng.standard_normal((B, n))
        return {"X": X, "y": y, "Xc": [sm.add_constant(X[i]) for i in range(B)], "Xs": sd.asarray(X),
                "ys": sd.asarray(y), "Xl": [sd.asarray(X[i]) for i in range(B)],
                "yl": [sd.asarray(y[i]) for i in range(B)]}

    return setup


def _ols_inf(r):
    return r.params, r.bse, r.pvalues, r.rsquared


def _g1_cases():
    import_sd = lambda: __import__("stardust")  # noqa: E731
    out = []
    for i, (n, k) in enumerate(G1_SIZES):
        setup = _g1_data(i)
        out.append(Case(
            area="regression", group="최소제곱", name="ols(y, X)", size=_size(n, k), setup=setup,
            sd=lambda d: (lambda R=import_sd().regression, y=d["ys"], X=d["Xs"]: R.ols(y, X)),
            refs=(Ref("statsmodels", lambda d: (lambda: _sm().OLS(d["y"], d["Xc"]).fit()), threads="follow"),
                  Ref("scikit-learn", lambda d: (lambda: _linreg().fit(d["X"], d["y"])), threads="follow",
                      lib="sklearn")),
            quick=(n, k) == (100_000, 10),
        ))
        out.append(Case(
            area="regression", group="최소제곱", name="ols(y, X) + params·bse·pvalues·rsquared", size=_size(n, k),
            setup=setup,
            sd=lambda d: (lambda R=import_sd().regression, y=d["ys"], X=d["Xs"]: _ols_inf(R.ols(y, X))),
            refs=(Ref("statsmodels", lambda d: (lambda: _ols_inf(_sm().OLS(d["y"], d["Xc"]).fit())),
                      threads="follow"),),
        ))
    setup = _g1_data(len(G1_SIZES))
    B, n, k = G1_BATCH

    def sm_loop(d):
        sm = _sm()

        def f():
            for i in range(B):
                _ols_inf(sm.OLS(d["y"][i], d["Xc"][i]).fit())

        return f

    def sd_loop(d):
        R = import_sd().regression

        def f():
            for i in range(B):
                _ols_inf(R.ols(d["yl"][i], d["Xl"][i]))

        return f

    size = f"{B:,} × ({n}, {k})"
    out.append(Case(
        area="regression", group="최소제곱 일괄", name="ols(Y, X) + 추론, 한 번 호출", size=size, setup=setup,
        sd=lambda d: (lambda R=import_sd().regression, y=d["ys"], X=d["Xs"]: _ols_inf(R.ols(y, X))),
        refs=(Ref("statsmodels 반복문", sm_loop, threads="follow", lib="statsmodels"),), quick=True,
        note="statsmodels는 Python 반복문으로 하나씩 적합한다",
    ))
    out.append(Case(
        area="regression", group="최소제곱 일괄", name="ols(y_i, X_i) + 추론, Python 반복문", size=size, setup=setup,
        sd=sd_loop,
    ))
    return out


def _linreg():
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


# ---------------------------------------------------------------- g1b: HAC, cluster, diagnostics

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
                return {"X": X, "y": y, "g": g, "Xc": sm.add_constant(X), "Xs": sd.asarray(X), "ys": sd.asarray(y),
                        "gs": sd.asarray(g), "L": int(math.floor(4 * (n / 100) ** (2 / 9)))}
            del X, e, g
        raise IndexError(index)

    return setup


def _g1b_cases():
    import stardust  # noqa: F401  (the factories below import it at call time)

    def R():
        import stardust as sd

        return sd.regression

    def sm_diag(d):
        from statsmodels.stats.diagnostic import het_breuschpagan
        from statsmodels.stats.outliers_influence import OLSInfluence
        from statsmodels.stats.stattools import durbin_watson, jarque_bera

        sm = _sm()

        def f():
            r = sm.OLS(d["y"], d["Xc"]).fit()
            inf = OLSInfluence(r)
            return (inf.hat_matrix_diag, inf.resid_studentized_internal, inf.cooks_distance,
                    durbin_watson(r.wresid), jarque_bera(r.wresid), het_breuschpagan(r.resid, d["Xc"]))

        return f

    def sd_diag(d):
        reg = R()

        def f():
            r = reg.ols(d["ys"], d["Xs"])
            return (r.leverage, r.resid_studentized_internal, r.cooks_distance, r.durbin_watson,
                    r.jarque_bera(), r.breusch_pagan())

        return f

    out = []
    for i, (n, k) in enumerate(G1B_SIZES):
        setup = _g1b_data(i)
        L = int(math.floor(4 * (n / 100) ** (2 / 9)))
        rows = [
            ("ols(y, X): bse, pvalues", "",
             lambda d: (lambda r=R(): _bp(r.ols(d["ys"], d["Xs"]))),
             lambda d: (lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit()))),
            ("ols(y, X, cov_type='HAC'): bse, pvalues", f"지연 {L}개",
             lambda d: (lambda r=R(): _bp(r.ols(d["ys"], d["Xs"], cov_type="HAC"))),
             lambda d: (lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit(cov_type="HAC", cov_kwds={"maxlags": d["L"]},
                                                                    use_t=True)))),
            ("ols(y, X, cov_type='cluster', groups=g): bse, pvalues", "군집 1,000개",
             lambda d: (lambda r=R(): _bp(r.ols(d["ys"], d["Xs"], cov_type="cluster", groups=d["gs"]))),
             lambda d: (lambda: _bp(_sm().OLS(d["y"], d["Xc"]).fit(cov_type="cluster", cov_kwds={"groups": d["g"]},
                                                                    use_t=True)))),
            ("ols(y, X) + 진단", "지렛대·내적 스튜던트화 잔차·Cook 거리·DW·JB·BP", sd_diag, sm_diag),
            ("ols(y, X).summary()", "",
             lambda d: (lambda r=R(): r.ols(d["ys"], d["Xs"]).summary()),
             lambda d: (lambda: str(_sm().OLS(d["y"], d["Xc"]).fit().summary()))),
        ]
        for name, note, sdf, smf in rows:
            out.append(Case(
                area="regression", group="최소제곱 공분산·진단", name=name, size=_size(n, k), setup=setup, sd=sdf,
                refs=(Ref("statsmodels", smf, threads="follow"),), note=note,
                quick=(name.startswith("ols(y, X, cov_type='HAC')") and n == 100_000),
            ))
        out.append(Case(
            area="regression", group="최소제곱 공분산·진단", name="ols(y, X).resid_studentized_external",
            size=_size(n, k), setup=setup,
            sd=lambda d: (lambda r=R(): r.ols(d["ys"], d["Xs"]).resid_studentized_external),
            note="statsmodels는 행마다 다시 적합해(n번) 재지 않았다",
        ))
    return out


def _bp(r):
    return r.bse, r.pvalues


# ---------------------------------------------------------------- g2: glm

G2_SIZES = ((100_000, 10), (1_000_000, 50))
G2_BATCH = (10_000, 50, 3)


def _g2_draw(rng, n, k):
    import numpy as np

    X = rng.standard_normal((n, k))
    beta = rng.standard_normal(k) * (0.6 / np.sqrt(k))
    eta = 0.2 + X @ beta
    yl = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)
    yp = rng.poisson(np.exp(0.3 + 0.5 * (X @ beta))).astype(float)
    return X, yl, yp


def _g2_data(index, fam=None):
    def setup():
        import numpy as np

        import stardust as sd

        sm = _sm()
        rng = np.random.default_rng(0)
        for i, (n, k) in enumerate(G2_SIZES):
            X, yl, yp = _g2_draw(rng, n, k)
            if i == index:
                y = yl if fam == "binomial" else yp
                return {"X": X, "y": y, "Xc": sm.add_constant(X), "Xs": sd.asarray(X), "ys": sd.asarray(y),
                        "aux": sd.asarray(np.r_[np.ones((3, n)), np.zeros((1, n))])}
            del X, yl, yp
        B, n, k = G2_BATCH
        Xb = rng.standard_normal((B, n, k))
        Yb = (rng.uniform(size=(B, n)) < 1 / (1 + np.exp(-(0.2 + Xb @ np.array([0.5, -0.3, 0.2]))))).astype(float)
        return {"Xb": Xb, "Yb": Yb, "Xbs": sd.asarray(Xb), "Ybs": sd.asarray(Yb),
                "Xbc": [sm.add_constant(Xb[i]) for i in range(B)]}

    return setup


def _g2_cases():
    def R():
        import stardust as sd

        return sd.regression

    out = []
    for i, (n, k) in enumerate(G2_SIZES):
        for fam in ("binomial", "poisson"):
            setup = _g2_data(i, fam)

            def sd_fit(d, fam=fam):
                reg = R()

                def f():
                    r = reg.glm(d["ys"], d["Xs"], family=fam)
                    return r.params, r.bse, r.pvalues, r.n_iter

                return _quiet(f)

            def sd_qr(d, fam=fam):
                import stardust as sd

                K = sd._core.regression
                fcode, lcode = (1, 2) if fam == "binomial" else (2, 1)
                return _quiet(lambda: K.glm_fit(
                    "glm", d["ys"], d["Xs"], d["aux"], None, 0, fcode, lcode, False, False, False, False, True, -1.0,
                    0, -1, 100, 1e-8, 0, 0.0, inner_steps=0, final_steps=3, inner_chol=False)[0])

            def sm_fit(d, fam=fam):
                sm = _sm()
                smf = sm.families.Binomial() if fam == "binomial" else sm.families.Poisson()

                def f():
                    r = sm.GLM(d["y"], d["Xc"], family=smf).fit()
                    return r.params, r.bse, r.pvalues

                return _quiet(f)

            def sk(d, fam=fam, solver=None):
                from sklearn.linear_model import LogisticRegression, PoissonRegressor

                cls = LogisticRegression if fam == "binomial" else PoissonRegressor
                kw = dict(penalty=None) if fam == "binomial" else dict(alpha=0.0)
                kw.update(dict(solver=solver) if solver else dict(max_iter=10000))
                return _quiet(lambda: cls(**kw).fit(d["X"], d["y"]))

            out.append(Case(
                area="regression", group="일반화 선형 모형", name=f"glm(y, X, family='{fam}')", size=_size(n, k),
                setup=setup, sd=sd_fit,
                refs=(Ref("statsmodels", sm_fit, threads="follow"),
                      Ref("scikit-learn lbfgs", sk, threads="follow", lib="sklearn"),
                      Ref("scikit-learn newton-cholesky", lambda d, sk=sk: sk(d, solver="newton-cholesky"),
                          threads="follow", lib="sklearn")),
                quick=(fam == "binomial" and n == 100_000),
                note="sd·statsmodels는 params·bse·pvalues까지, scikit-learn은 적합만",
            ))
            out.append(Case(
                area="regression", group="일반화 선형 모형", name=f"glm(y, X, family='{fam}'), 중간 반복 QR",
                size=_size(n, k), setup=setup, sd=sd_qr,
                note="커널 직접 호출(inner_chol=False, 벤치마크 전용)",
            ))
    setup = _g2_data(len(G2_SIZES))
    B, n, k = G2_BATCH

    def sd_batch(d):
        reg = R()

        def f():
            r = reg.glm(d["Ybs"], d["Xbs"], family="binomial")
            return r.params, r.bse, r.pvalues, r.separation

        return _quiet(f)

    def sd_loop(d):
        reg = R()

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
    out.append(Case(
        area="regression", group="일반화 선형 모형 일괄", name="glm(Y, X, family='binomial') + 추론, 한 번 호출",
        size=size, setup=setup, sd=sd_batch,
        refs=(Ref("statsmodels 반복문", sm_loop, threads="follow", lib="statsmodels"),),
        note="statsmodels는 Python 반복문으로 하나씩 적합한다",
    ))
    out.append(Case(
        area="regression", group="일반화 선형 모형 일괄", name="glm(y_i, X_i, family='binomial') + 추론, Python 반복문",
        size=size, setup=setup, sd=sd_loop, note="numpy 배열을 그대로 넘긴다",
    ))
    return out


# ---------------------------------------------------------------- g3: penalized

G3_SIZES = ((100_000, 10), (100_000, 100), (10_000, 1000))


def _g3_data(n, k):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(0)
        X = rng.standard_normal((n, k))
        beta = np.zeros(k)
        beta[:10] = rng.standard_normal(10)
        y = X @ beta + rng.standard_normal(n)
        yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta / 2))).astype(float)
        Xs, ys, ybs = sd.asarray(X), sd.asarray(y), sd.asarray(yb)
        amax = float(sd._core.regression.alpha_max("x", ys, Xs, None, 1.0, 0, True, False))
        grid = amax * np.geomspace(1, 1e-3, 100)
        fold = np.asarray(sd.take(sd.asarray([i % 5 for i in range(n)], dtype=sd.int64),
                                  sd.random.default_rng(1).permutation(n)))
        splits = [(np.nonzero(fold != q)[0], np.nonzero(fold == q)[0]) for q in range(5)]
        rgrid = np.geomspace(10.0, 1e-4, 100)
        return {"X": X, "y": y, "yb": yb, "Xs": Xs, "ys": ys, "ybs": ybs, "grid": grid, "gs": sd.asarray(grid),
                "a20": amax / 20, "splits": splits, "rgrid": rgrid, "rgs": sd.asarray(rgrid), "n": n,
                "C": 1 / (n * 1e-3)}

    return setup


def _g3_cases():
    def R():
        import stardust as sd

        return sd.regression

    def sk(name):
        import sklearn.linear_model as lm

        return getattr(lm, name)

    def sdq(f):
        return lambda d: _quiet(lambda: f(R(), d))

    def skq(f):
        return lambda d: _quiet(lambda: f(d))

    rows = [
        ("ridge(y, X, alpha=0.1)",
         sdq(lambda r, d: r.ridge(d["ys"], d["Xs"], alpha=0.1)),
         (("scikit-learn Ridge", skq(lambda d: sk("Ridge")(alpha=0.1 * d["n"]).fit(d["X"], d["y"]))),
          ("scikit-learn Ridge svd", skq(lambda d: sk("Ridge")(alpha=0.1 * d["n"], solver="svd").fit(d["X"], d["y"])))),
         "scikit-learn은 alpha = 0.1 n(같은 문제)"),
        ("lasso(y, X, alpha=alpha_max/20)",
         sdq(lambda r, d: r.lasso(d["ys"], d["Xs"], alpha=d["a20"])),
         (("scikit-learn", skq(lambda d: sk("Lasso")(alpha=d["a20"]).fit(d["X"], d["y"]))),), ""),
        ("elastic_net(y, X, alpha=alpha_max/20)",
         sdq(lambda r, d: r.elastic_net(d["ys"], d["Xs"], alpha=d["a20"])),
         (("scikit-learn", skq(lambda d: sk("ElasticNet")(alpha=d["a20"]).fit(d["X"], d["y"]))),), ""),
        ("lasso(y, X, alpha=100개 격자)",
         sdq(lambda r, d: r.lasso(d["ys"], d["Xs"], alpha=d["gs"])),
         (("scikit-learn", skq(lambda d: sk("lasso_path")(d["X"] - d["X"].mean(0), d["y"] - d["y"].mean(),
                                                         alphas=d["grid"]))),),
         "scikit-learn은 lasso_path(중심화 포함)"),
        ("lasso_cv(y, X, alphas=100개 격자)",
         sdq(lambda r, d: r.lasso_cv(d["ys"], d["Xs"], rng=_sd_rng(1), alphas=d["gs"])),
         (("scikit-learn", skq(lambda d: sk("LassoCV")(alphas=d["grid"], cv=d["splits"]).fit(d["X"], d["y"]))),),
         "5겹(같은 접기)"),
        ("ridge_cv(y, X, cv='gcv', alphas=100개)",
         sdq(lambda r, d: r.ridge_cv(d["ys"], d["Xs"], cv="gcv", alphas=d["rgs"])),
         (("scikit-learn", skq(lambda d: sk("RidgeCV")(alphas=d["rgrid"] * d["n"]).fit(d["X"], d["y"]))),), ""),
        ("penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=1)",
         sdq(lambda r, d: r.penalized_glm(d["ybs"], d["Xs"], family="binomial", alpha=1e-3, l1_ratio=1.0)),
         (("scikit-learn saga", skq(lambda d: sk("LogisticRegression")(C=d["C"], l1_ratio=1.0, solver="saga")
                                    .fit(d["X"], d["yb"]))),
          ("scikit-learn liblinear", skq(lambda d: sk("LogisticRegression")(C=d["C"], l1_ratio=1.0, solver="liblinear")
                                         .fit(d["X"], d["yb"])))),
         "liblinear는 절편에도 벌점을 준다"),
        ("penalized_glm(y, X, family='binomial', alpha=1e-3, l1_ratio=0)",
         sdq(lambda r, d: r.penalized_glm(d["ybs"], d["Xs"], family="binomial", alpha=1e-3, l1_ratio=0.0)),
         (("scikit-learn lbfgs", skq(lambda d: sk("LogisticRegression")(C=d["C"]).fit(d["X"], d["yb"]))),
          ("scikit-learn newton-cholesky", skq(lambda d: sk("LogisticRegression")(C=d["C"], solver="newton-cholesky")
                                               .fit(d["X"], d["yb"])))),
         ""),
    ]
    quick = {("lasso_cv(y, X, alphas=100개 격자)", (100_000, 10)), ("ridge(y, X, alpha=0.1)", (100_000, 100))}
    out = []
    for n, k in G3_SIZES:
        setup = _g3_data(n, k)
        for name, sdf, refs, note in rows:
            out.append(Case(
                area="regression", group="벌점 회귀", name=name, size=_size(n, k), setup=setup, sd=sdf,
                refs=tuple(Ref(label, f, threads="follow", lib="sklearn") for label, f in refs), note=note,
                quick=(name, (n, k)) in quick,
            ))
    return out


def _sd_rng(seed):
    import stardust as sd

    return sd.random.default_rng(seed)


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


def _slopes_data(n):
    def setup():
        import numpy as np

        import stardust as sd

        rng = np.random.default_rng(n)
        x = rng.standard_normal(n)
        y = 2.0 * x + rng.standard_normal(n)
        return {"x": x, "y": y, "xs": sd.asarray(x), "ys": sd.asarray(y)}

    return setup


def _g4_cases():
    def R():
        import stardust as sd

        return sd.regression

    out = []
    for n, k in ((100_000, 10), (1_000_000, 50)):
        setup = _g4_data(n, k)
        for norm, M in (("huber", "HuberT"), ("bisquare", "TukeyBiweight")):
            def sm_rlm(d, M=M, collect=n == 1_000_000):
                import gc

                from statsmodels.robust import norms

                sm = _sm()

                def f():
                    sm.RLM(d["y"], d["Xc"], M=getattr(norms, M)()).fit()
                    if collect:
                        gc.collect()  # the fit leaves ~3.6 GB in reference cycles; two of them pass 16 GB

                return _quiet(f)

            out.append(Case(
                area="regression", group="강건 회귀", name=f"rlm(y, X, norm='{norm}')", size=_size(n, k), setup=setup,
                sd=lambda d, norm=norm: (lambda r=R(): r.rlm(d["ys"], d["Xs"], norm=norm)),
                refs=(Ref("statsmodels", sm_rlm, threads="follow"),),
                quick=(norm == "huber" and n == 100_000),
                note=("statsmodels 한 번에 20초 안팎; 호출마다 gc.collect()를 더한다(순환 참조로 남는 메모리, 0.05초)"
                      if n == 1_000_000 else ""),
            ))
    for n, k in ((10_000, 5), (100_000, 10), (1_000_000, 10)):
        setup = _g4_data(n, k, leverage=True)
        out.append(Case(
            area="regression", group="강건 회귀", name="rlm(y, X, method='MM')", size=_size(n, k), setup=setup,
            sd=lambda d: (lambda r=R(): r.rlm(d["ys"], d["Xs"], method="MM", rng=_sd_rng(1))),
            refs=(RRef("R lmrob", "lmrob(y ~ X)", data={"y": "y", "X": "X"}, packages=("robustbase",)),),
            quick=n == 10_000,
            note="지렛대 무리 20%; lmrob은 n > 2000에서 묶음 표본 방식",
        ))
        if n <= 100_000:
            def small_n(d):
                import stardust as sd

                K = sd._core.regression

                def f():
                    keys = sd.random.default_rng(1).integers(-2**63, 2**63 - 1, endpoint=True, shape=(2,))
                    return K.rlm_fit("rlm", d["ys"], d["Xs"], keys, True, -1.0, 1, 1, 0.0, 0.0, 4.685061, 2, 50, 1e-8,
                                     500, large_n=False)

                return f

            out.append(Case(
                area="regression", group="강건 회귀", name="rlm(y, X, method='MM'), 작은 n 방식", size=_size(n, k),
                setup=setup, sd=small_n, note="커널 직접 호출(large_n=False, 모든 행의 후보 500개; 벤치마크 전용)",
            ))
    for n in (1_000, 10_000, 100_000):
        setup = _slopes_data(n)
        for name in ("theilslopes", "siegelslopes"):
            refs = ()
            if n <= 10_000:
                refs = (Ref("scipy", lambda d, name=name: (lambda f=_scipy_stats(name): f(d["y"], d["x"]))),)
            note = ""
            if n == 100_000:
                note = "1스레드 한 번에 수십 초; scipy는 n × n 배열(80 GB)이라 재지 않았다"
            out.append(Case(
                area="regression", group="강건 기울기", name=f"{name}(y, x)", size=f"{n:,}", n=n, setup=setup,
                sd=lambda d, name=name: (lambda f=getattr(R(), name): f(d["ys"], d["xs"])), refs=refs,
                quick=(name == "theilslopes" and n == 10_000), note=note,
            ))
    return out


def _scipy_stats(name):
    import scipy.stats as st

    return getattr(st, name)


# ---------------------------------------------------------------- g5: quantreg

G5_SIZES = ((1_000, 5), (100_000, 10), (1_000_000, 10))
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


def _g5_cases():
    def R():
        import stardust as sd

        return sd.regression

    def sm_qr(d, qs=(0.5,)):
        sm = _sm()
        if len(qs) == 1:
            return _quiet(lambda: sm.QuantReg(d["y"], d["Xc"]).fit(q=qs[0]))
        return _quiet(lambda: [sm.QuantReg(d["y"], d["Xc"]).fit(q=q) for q in qs])

    rdata = {"y": "y", "X": "X"}
    rsetup = "x <- cbind(1, X)"
    out = []
    for n, k in G5_SIZES:
        setup = _g5_data(n, k)
        refs = [RRef("R fn", 'rq.fit(x, y, tau = 0.5, method = "fn")', data=rdata, setup=rsetup,
                     packages=("quantreg",))]
        if n <= 100_000:
            refs.append(RRef("R br", 'rq.fit(x, y, tau = 0.5, method = "br")', data=rdata, setup=rsetup,
                             packages=("quantreg",)))
        refs.append(Ref("statsmodels", sm_qr, threads="follow"))
        out.append(Case(
            area="regression", group="분위 회귀", name="quantreg(y, X, q=0.5, cov_type='iid')", size=_size(n, k),
            setup=setup, sd=lambda d: (lambda r=R(): r.quantreg(d["ys"], d["Xs"], q=0.5, cov_type="iid")),
            refs=tuple(refs), quick=n == 100_000,
            note="R br은 n = 10^6에서 몇 분 걸려 재지 않았다" if n > 100_000 else "",
        ))
        out.append(Case(
            area="regression", group="분위 회귀", name="quantreg(y, X, q=0.5)", size=_size(n, k), setup=setup,
            sd=lambda d: (lambda r=R(): r.quantreg(d["ys"], d["Xs"], q=0.5)),
            refs=(RRef("R fn + nid",
                       'g <- rq(y ~ X, tau = 0.5, method = "fn"); suppressWarnings(summary(g, se = "nid"))',
                       data=rdata, packages=("quantreg",)),),
            quick=n == 1_000, note="기본 cov_type='nid'; R은 fn 적합과 summary(se = \"nid\")",
        ))
    n, k = 100_000, 10
    setup = _g5_data(n, k)
    out.append(Case(
        area="regression", group="분위 회귀(분위 9개)", name="quantreg(y, X, q=[0.1, ..., 0.9], cov_type='iid')",
        size=_size(n, k), setup=setup,
        sd=lambda d: (lambda r=R(): r.quantreg(d["ys"], d["Xs"], q=QS9, cov_type="iid")),
        refs=(RRef("R fn", 'for (t in taus) rq.fit(x, y, tau = t, method = "fn")', data=rdata,
                   setup=f"{rsetup}; {R_TAUS}", packages=("quantreg",)),
              RRef("R br", 'for (t in taus) rq.fit(x, y, tau = t, method = "br")', data=rdata,
                   setup=f"{rsetup}; {R_TAUS}", packages=("quantreg",)),
              Ref("statsmodels", lambda d: sm_qr(d, QS9), threads="follow")),
        note="R·statsmodels는 분위마다 한 번씩; R br은 한 번에 25초 안팎",
    ))
    out.append(Case(
        area="regression", group="분위 회귀(분위 9개)", name="quantreg(y, X, q=q, cov_type='iid'), 분위마다 따로",
        size=_size(n, k), setup=setup,
        sd=lambda d: (lambda r=R(): [r.quantreg(d["ys"], d["Xs"], q=q, cov_type="iid") for q in QS9]),
    ))
    return out


# ---------------------------------------------------------------- g6: nls

G6_NS = (100, 10_000, 1_000_000)
CENTRES = (0.7, 1.6, 2.5, 3.4, 4.3)


def _models(xp):
    exp = xp.exp
    m = {}
    m[2] = (lambda x, a, b: a * exp(-b * x),
            lambda x, a, b: xp.stack([exp(-b * x), -a * x * exp(-b * x)], axis=1), [2.0, 0.8])

    def f4(x, a, b, c, d):
        return d + (a - d) / (1 + (x / c) ** b)

    def j4(x, a, b, c, d):
        u = (x / c) ** b
        den = 1 + u
        lg = xp.log(x / c)
        return xp.stack([1 / den, -(a - d) * u * lg / den ** 2, (a - d) * u * b / c / den ** 2, 1 - 1 / den], axis=1)

    m[4] = (f4, j4, [0.2, 2.5, 1.8, 3.0])

    def f10(x, *t):
        return sum(t[2 * k] * exp(-((x - t[2 * k + 1]) / 0.4) ** 2) for k in range(5))

    def j10(x, *t):
        cols = []
        for k in range(5):
            a, c = t[2 * k], t[2 * k + 1]
            g = exp(-((x - c) / 0.4) ** 2)
            cols += [g, a * g * 2 * (x - c) / 0.16]
        return xp.stack(cols, axis=1)

    m[10] = (f10, j10, [1.0, CENTRES[0], 1.5, CENTRES[1], 0.8, CENTRES[2], 1.2, CENTRES[3], 0.6, CENTRES[4]])
    return m


R_FORMULA = {
    2: ("y ~ a*exp(-b*x)", ["a", "b"]),
    4: ("y ~ d + (a - d)/(1 + (x/c)^b)", ["a", "b", "c", "d"]),
    10: ("y ~ " + " + ".join(f"a{k}*exp(-((x - c{k})/0.4)^2)" for k in range(5)),
         [v for k in range(5) for v in (f"a{k}", f"c{k}")]),
}
G6_TRUTH = {2: [2.0, 0.8], 4: [0.2, 2.5, 1.8, 3.0],
            10: [1.0, CENTRES[0], 1.5, CENTRES[1], 0.8, CENTRES[2], 1.2, CENTRES[3], 0.6, CENTRES[4]]}
G6_MODEL = {2: "a exp(-b x)", 4: "4모수 로지스틱", 10: "가우스 5개"}


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
        return {"x": x, "y": y, "xs": sd.asarray(x), "ys": sd.asarray(y), "msd": _models(sd)[p],
                "mnp": _models(np)[p], "p0": _p0(p)}

    return setup


def _g6_cases():
    def R():
        import stardust as sd

        return sd.regression

    def curve_fit(d, jac=False):
        from scipy.optimize import curve_fit as cf

        fn, Jn, _ = d["mnp"]
        kw = dict(jac=Jn) if jac else {}
        return _quiet(lambda: cf(fn, d["x"], d["y"], p0=d["p0"], full_output=True, **kw))

    out = []
    for n in G6_NS:
        for p in (2, 4, 10):
            setup = _g6_data(n, p)
            form, names = R_FORMULA[p]
            start = ", ".join(f"{a} = {v!r}" for a, v in zip(names, _p0(p)))
            rref = RRef("R nls", f"nls({form}, data = dat, start = list({start}))", data={"x": "x", "y": "y"},
                        setup="dat <- data.frame(y = y, x = x)")
            slow = n == 1_000_000 and p == 10
            out.append(Case(
                area="regression", group="비선형 최소제곱", name="nls(f, x, y, p0)", size=f"n={n:,}, p={p}",
                setup=setup,
                sd=lambda d: (lambda r=R(), f=d["msd"][0]: r.nls(f, d["xs"], d["ys"], p0=d["p0"])),
                refs=(Ref("scipy curve_fit", curve_fit), rref),
                quick=(n == 10_000 and p == 4),
                note=f"{G6_MODEL[p]}; sd는 5점 차분, scipy는 2점 차분" + ("; sd 1스레드 한 번에 10초 넘는다" if slow else ""),
            ))
            out.append(Case(
                area="regression", group="비선형 최소제곱", name="nls(f, x, y, p0, jac=J)", size=f"n={n:,}, p={p}",
                setup=setup,
                sd=lambda d: (lambda r=R(), f=d["msd"][0], J=d["msd"][1]: r.nls(f, d["xs"], d["ys"], p0=d["p0"], jac=J)),
                refs=(Ref("scipy curve_fit", lambda d: curve_fit(d, jac=True)),),
                note=f"{G6_MODEL[p]}; 해석적 야코비안",
            ))
    return out


# ---------------------------------------------------------------- all


def cases() -> list[Case]:
    return [*_g1_cases(), *_g1b_cases(), *_g2_cases(), *_g3_cases(), *_g4_cases(), *_g5_cases(), *_g6_cases()]
