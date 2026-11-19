"""T9: accuracy of the multivariate normal and t log densities and entropies against
Arb references (ball arithmetic on the matrices as given: Arb's LU for the determinant and the
solves, at a working precision that doubles until every ball's radius is below 2^-80 of
max(1, |value|)), over the condition number of the covariance and the dimension (T29: 1e-14
mixed for kappa <= 1e9, d <= 100), and scipy's error at the same points. Errors:
|error| / max(1, |value|).

    .venv/bin/python bench/stats/t9/accuracy.py > bench/stats/t9/accuracy.md
"""

import math

import numpy as np
import scipy.stats as st
from flint import arb, arb_mat, ctx

import stardust as sd

BITS = 80  # the references' radius, relative to max(1, |value|)


def settled(fn):
    """fn() (a tuple of arb balls) at doubling working precision until every radius is below
    2^-BITS of max(1, |value|)."""
    old, prec = ctx.prec, 128
    try:
        while True:
            ctx.prec = prec
            vs = fn()
            if all(v.is_finite() and v.rad() <= arb(2) ** -BITS * max(arb(1), abs(v.mid())) for v in vs):
                return vs
            if prec >= 1 << 13:
                raise ArithmeticError(f"reference did not settle by {prec} bits")
            prec *= 2
    finally:
        ctx.prec = old


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    ev = np.exp(np.linspace(0.0, math.log(kappa), d))
    a = (q * ev) @ q.T
    return (a + a.T) / 2.0


def reference(mu, cov, xs, df=None):
    d = len(mu)

    def fn():
        A = arb_mat(cov.tolist())
        ld = A.det().log()
        R = arb_mat([[float(x[i]) for x in xs] for i in range(d)]) - arb_mat(
            [[float(mu[i])] * len(xs) for i in range(d)]
        )
        Y = A.solve(R)
        pi = arb.pi()
        out = []
        for j in range(len(xs)):
            q = sum((R[i, j] * Y[i, j] for i in range(d)), arb(0))
            if df is None:
                out.append(-(d * (2 * pi).log() + ld + q) / 2)
            else:
                nu = arb(df)
                out.append(
                    ((nu + d) / 2).lgamma()
                    - (nu / 2).lgamma()
                    - arb(d) / 2 * (nu * pi).log()
                    - ld / 2
                    - (nu + d) / 2 * (q / nu).log1p()
                )
        if df is None:
            h = arb(d) / 2 * (1 + (2 * pi).log()) + ld / 2
        else:
            nu = arb(df)
            a, hh = nu / 2, arb(d) / 2
            h = (
                ld / 2
                + a.lgamma()
                - (a + hh).lgamma()
                + hh * (nu * pi).log()
                + (a + hh) * ((a + hh).digamma() - a.digamma())
            )
        return (*out, h)

    *out, h = settled(fn)
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
