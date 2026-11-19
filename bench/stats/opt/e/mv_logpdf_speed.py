"""Group E: MultivariateNormal / MultivariateT logpdf, 10^5 points, one thread: the class (fast
path), the kernel without it (the double-double path) and scipy; the largest difference
between the two paths on these points (bench/stats/t9/speed.py's data).

    .venv/bin/python bench/stats/opt/e/mv_logpdf_speed.py
"""

import stardust as sd, numpy as np, timeit, scipy.stats as st

S = sd.stats
_k = sd._core.stats
rng = np.random.default_rng(0)
sd.set_num_threads(1)
for d in (3, 10, 50):
    a = rng.normal(size=(d, d))
    cov = a @ a.T + d * np.eye(d)
    mu = rng.normal(size=d)
    x = rng.normal(size=(100000, d))
    for nm, D, Sp in (
        ("N", S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)), st.multivariate_normal(mu, cov)),
        ("T", S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=5.0), st.multivariate_t(mu, cov, df=5.0)),
    ):
        X = sd.asarray(x)
        D.logpdf(X)
        rows, idx, out = D._points(X, "logpdf")
        f = lambda fast, pdf=False: _k.mv_logpdf(
            rows, idx, D._mu, D._L, D._Ll, D._ldh, D._ldl, D._ok, D._df, D._ih, D._il, fast, pdf
        )
        tk = min(timeit.repeat(lambda: D.logpdf(X), number=1, repeat=7))
        tk0 = min(timeit.repeat(lambda: f(None), number=1, repeat=3))
        ts = min(timeit.repeat(lambda: Sp.logpdf(x), number=1, repeat=5))
        v1 = np.asarray(f(D._fast))
        v0 = np.asarray(f(None))
        p1 = np.asarray(f(D._fast, True))
        p0 = np.asarray(f(None, True))
        print(
            d,
            nm,
            "fast %.2f precise %.2f scipy %.2f ms; ratio %.2f; maxdiff %.2e frac equal %.3f; pdf maxrel %.2e"
            % (
                tk * 1e3,
                tk0 * 1e3,
                ts * 1e3,
                ts / tk,
                np.max(np.abs(v1 - v0) / np.maximum(1, np.abs(v0))),
                np.mean(v1 == v0),
                np.max(np.abs(p1 - p0) / p0),
            ),
        )
