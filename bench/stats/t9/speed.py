"""T9: speed of the multivariate distributions' logpdf (logpmf) and samples against scipy.stats,
10^5 points (vectors), 1 and 10 threads. The cdfs: bench/stats/t9/cdf_speed.py.

    .venv/bin/python bench/stats/t9/speed.py > bench/stats/t9/speed.md
"""

import timeit
import warnings

import numpy as np
import scipy.stats as st

import stardust as sd

S = sd.stats
N = 10**5


def best(f, repeat=3):
    return min(timeit.repeat(f, number=1, repeat=repeat))


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    rows = []
    for d in (3, 10, 50):
        a = rng.normal(size=(d, d))
        cov = a @ a.T + d * np.eye(d)
        mu = rng.normal(size=d)
        x = rng.normal(size=(N, d))
        rows.append(
            (
                f"MultivariateNormal d={d}",
                S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)),
                st.multivariate_normal(mu, cov),
                x,
                "logpdf",
            )
        )
        rows.append(
            (
                f"MultivariateT d={d}",
                S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=5.0),
                st.multivariate_t(mu, cov, df=5.0),
                x,
                "logpdf",
            )
        )
    p = np.array([0.1, 0.2, 0.3, 0.4])
    xm = rng.multinomial(20, p, size=N).astype(float)
    rows.append(("Multinomial K=4 n=20", S.Multinomial(n=20.0, p=sd.asarray(p)), st.multinomial(20, p), xm, "logpmf"))
    al = np.array([0.5, 2.0, 3.0, 1.5])
    xd = rng.dirichlet(al, size=N)
    rows.append(("Dirichlet K=4", S.Dirichlet(alpha=sd.asarray(al)), st.dirichlet(al), xd, "logpdf"))
    before = sd.get_num_threads()
    print(f"{N:,} points (vectors); ms.\n")
    print("| distribution | method | sd 1T | sd 10T | scipy | scipy / sd 1T | scipy / sd 10T |")
    print("|---|---|---:|---:|---:|---:|---:|")
    try:
        for name, D, Ssp, x, meth in rows:
            X = sd.asarray(x)
            f = getattr(D, meth)
            g = (lambda: Ssp.logpdf(x.T)) if name.startswith("Dirichlet") else (lambda: getattr(Ssp, meth)(x))
            sd.set_num_threads(1)
            t1 = best(lambda: f(X))
            sd.set_num_threads(10)
            t10 = best(lambda: f(X))
            ts = best(g)
            print(
                f"| {name} | {meth} | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {ts * 1e3:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                flush=True,
            )
            r = sd.random.default_rng(1)
            sd.set_num_threads(1)
            t1 = best(lambda: D.sample(N, rng=r))
            sd.set_num_threads(10)
            t10 = best(lambda: D.sample(N, rng=r))
            ts = best(lambda: Ssp.rvs(size=N, random_state=1))
            print(
                f"| {name} | sample | {t1 * 1e3:.2f} | {t10 * 1e3:.2f} | {ts * 1e3:.2f} | {ts / t1:.2f} | {ts / t10:.1f} |",
                flush=True,
            )
    finally:
        sd.set_num_threads(before)


if __name__ == "__main__":
    main()
