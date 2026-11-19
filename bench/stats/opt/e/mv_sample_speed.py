"""Group E: the multivariate samples, 10^5 rows, one thread, against scipy; with the first
coordinate's mean and a digest over 1, 3 and 10 threads.

    .venv/bin/python bench/stats/opt/e/mv_sample_speed.py
"""

import stardust as sd, numpy as np, timeit, scipy.stats as st, hashlib

S = sd.stats
rng = np.random.default_rng(0)
N = 100000
rows = []
for d in (3, 10, 50):
    a = rng.normal(size=(d, d))
    cov = a @ a.T + d * np.eye(d)
    mu = rng.normal(size=d)
    rows.append(
        (f"MVN d={d}", S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)), st.multivariate_normal(mu, cov))
    )
    rows.append(
        (
            f"MVT d={d}",
            S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=5.0),
            st.multivariate_t(mu, cov, df=5.0),
        )
    )
p = np.array([0.1, 0.2, 0.3, 0.4])
rows.append(("Multinomial", S.Multinomial(n=20.0, p=sd.asarray(p)), st.multinomial(20, p)))
al = np.array([0.5, 2.0, 3.0, 1.5])
rows.append(("Dirichlet", S.Dirichlet(alpha=sd.asarray(al)), st.dirichlet(al)))
rows.append(("Dirichlet big", S.Dirichlet(alpha=sd.asarray(al + 1)), st.dirichlet(al + 1)))
for name, D, Sp in rows:
    h = hashlib.sha1()
    for th in (1, 3, 10):
        sd.set_num_threads(th)
        r = sd.random.default_rng(3)
        for n in (1, 5, 1000, 7):
            h.update(np.asarray(D.sample(n, rng=r)).tobytes())
    sd.set_num_threads(1)
    r = sd.random.default_rng(1)
    t1 = min(timeit.repeat(lambda: D.sample(N, rng=r), number=1, repeat=5))
    ts = min(timeit.repeat(lambda: Sp.rvs(size=N, random_state=1), number=1, repeat=3))
    x = np.asarray(D.sample(N, rng=r))
    print(
        f"{name:14s} sd {t1 * 1e3:7.2f} scipy {ts * 1e3:7.2f} ratio {ts / t1:5.2f}  mean0 {x[:, 0].mean():.4f} hash3threads {h.hexdigest()[:10]}"
    )
