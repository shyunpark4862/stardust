"""Group E: MatrixNormal logpdf, 10^4 matrices, one thread: the class (fast path), the kernel
without it and scipy (bench/stats/t10/speed.py's parameters, and 30 x 20).

    .venv/bin/python bench/stats/opt/e/matnorm_logpdf_speed.py
"""

import stardust as sd, numpy as np, timeit, scipy.stats as st

S = sd.stats
_k = sd._core.stats
rng = np.random.default_rng(0)
sd.set_num_threads(1)
N = 10000
for n, p in ((3, 2), (10, 5), (30, 20)):
    M = rng.normal(size=(n, p))
    U, V = np.eye(n) + 0.3, np.eye(p) + 0.2
    Sp = st.matrix_normal(M, U, V)
    x = Sp.rvs(N, random_state=1)
    D = S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V))
    X = sd.asarray(x)
    D.logpdf(X)
    rows, idx, out = D._points(X, "logpdf")
    f = lambda fu, fv, pdf=False: _k.matnorm_logpdf(
        rows, idx, D._mean, D._U, D._Ul, D._V, D._Vl, D._lu[0], D._lu[1], D._lv[0], D._lv[1], D._ok, fu, fv, pdf
    )
    tk = min(timeit.repeat(lambda: D.logpdf(X), number=1, repeat=7))
    tk0 = min(timeit.repeat(lambda: f(None, None), number=1, repeat=3))
    ts = min(timeit.repeat(lambda: Sp.logpdf(x), number=1, repeat=5))
    v1 = np.asarray(f(*D._fast))
    v0 = np.asarray(f(None, None))
    p1 = np.asarray(f(*D._fast, True))
    p0 = np.asarray(f(None, None, True))
    print(
        n,
        p,
        "fast %.2f precise %.2f scipy %.2f ms; ratio %.2f; maxdiff %.2e frac equal %.3f; pdf maxrel %.2e"
        % (
            tk * 1e3,
            tk0 * 1e3,
            ts * 1e3,
            ts / tk,
            np.max(np.abs(v1 - v0) / np.maximum(1, np.abs(v0))),
            np.mean(v1 == v0),
            np.max(np.abs(p1 - p0) / np.where(p0 > 0, p0, 1)),
        ),
    )
