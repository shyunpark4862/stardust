"""T9: accuracy of the multivariate normal and t log densities and entropies against
50-digit references (mpmath LU of the matrices as given), over the condition number of the
covariance and the dimension (T29: 1e-14 mixed for kappa <= 1e6, d <= 100), and scipy's error
at the same points. Errors: |error| / max(1, |value|).

    .venv/bin/python bench/stats/t9/accuracy.py > bench/stats/t9/accuracy.md
"""

import math

import mpmath as mp
import numpy as np
import scipy.stats as st

import stardust as sd

mp.mp.dps = 50


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    ev = np.exp(np.linspace(0.0, math.log(kappa), d))
    a = (q * ev) @ q.T
    return (a + a.T) / 2.0


def reference(mu, cov, xs, df=None):
    d = len(mu)
    A = mp.matrix(cov.tolist())
    lu_det = mp.log(mp.det(A))
    out = []
    for x in xs:
        r = mp.matrix([mp.mpf(float(x[i])) - mp.mpf(float(mu[i])) for i in range(d)])
        q = (r.T * mp.lu_solve(A, r))[0]
        if df is None:
            out.append(-(d * mp.log(2 * mp.pi) + lu_det + q) / 2)
        else:
            nu = mp.mpf(df)
            out.append(mp.loggamma((nu + d) / 2) - mp.loggamma(nu / 2) - d / 2 * mp.log(nu * mp.pi) - lu_det / 2
                       - (nu + d) / 2 * mp.log1p(q / nu))
    h = None
    if df is None:
        h = d / mp.mpf(2) * (1 + mp.log(2 * mp.pi)) + lu_det / 2
    else:
        nu = mp.mpf(df)
        a, hh = nu / 2, mp.mpf(d) / 2
        h = (lu_det / 2 + mp.loggamma(a) - mp.loggamma(a + hh) + hh * mp.log(nu * mp.pi)
             + (a + hh) * (mp.digamma(a + hh) - mp.digamma(a)))
    return [float(v) for v in out], float(h)


def err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))


def main():
    rng = np.random.default_rng(0)
    print("| distribution | d | kappa | sd logpdf | scipy logpdf | sd entropy | scipy entropy |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for df in (None, 4.5, 1e9):
        for d in (2, 5, 20, 100):
            for kappa in (1.0, 1e3, 1e6, 1e9):
                cov = spd(rng, d, kappa)
                mu = rng.normal(size=d) * 3.0
                xs = mu + rng.normal(size=(8 if d < 100 else 3, d)) @ np.linalg.cholesky(cov).T * 2.0
                ref, href = reference(mu, cov, xs, df)
                if df is None:
                    D = sd.stats.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov))
                    S = st.multivariate_normal(mu, cov)
                    name = "normal"
                else:
                    D = sd.stats.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=df)
                    S = st.multivariate_t(mu, cov, df=df)
                    name = f"t df={df:g}"
                e1 = err(D.logpdf(sd.asarray(xs)), ref)
                try:
                    e2 = err(S.logpdf(xs), ref)
                except Exception:
                    e2 = float("nan")
                e3 = err(D.entropy(), href)
                try:
                    e4 = err(S.entropy(), href)
                except Exception:
                    e4 = float("nan")
                print(f"| {name} | {d} | {kappa:.0e} | {e1:.1e} | {e2:.1e} | {e3:.1e} | {e4:.1e} |", flush=True)


if __name__ == "__main__":
    main()
