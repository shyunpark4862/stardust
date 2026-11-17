"""Group E: the fast path of the multivariate normal / t and matrix normal densities against the
double-double path (the same kernel with fast=None), over d, kappa and kinds of points.

Prints, per case, the share of points whose bits differ between the two (a lower bound of the
share the fast path accepted, its rigorous bound inside ddla::kQuadBudget = 40 units of 2^-53:
an accepted point often rounds to the same double), the largest |fast - precise| for logpdf in
the contract's mixed units max(1, |value|) and for pdf relative, and at the end the overall
maxima. Points the fast path does not accept give the same bits as before by construction.

    .venv/bin/python bench/stats/opt/e/mv_sweep.py [--quick]
"""

import sys

import numpy as np

import stardust as sd

S = sd.stats
_k = sd._core.stats


def cov_with_kappa(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    ev = kappa ** (-np.arange(d) / max(d - 1, 1))
    c = (q * ev) @ q.T
    return (c + c.T) / 2


def points(rng, kind, n, d, mu, L):
    z = rng.normal(size=(n, d))
    if kind == "self":  # drawn from the distribution
        return mu + z @ L.T
    if kind == "iso":  # round the mean, the factor ignored
        return mu + z
    if kind == "far":  # 100 sigma
        return mu + 100.0 * z @ L.T
    if kind == "near":  # 1e-8 sigma (x - mu cancels)
        return mu + 1e-8 * z @ L.T
    if kind == "axis":  # along single coordinates
        e = np.zeros((n, d))
        e[np.arange(n), rng.integers(0, d, n)] = rng.normal(size=n) * 10.0 ** rng.uniform(-3, 3, n)
        return mu + e
    raise ValueError(kind)


def mixed(a, b):
    fin = np.isfinite(b)
    assert np.array_equal(a[~fin], b[~fin], equal_nan=True)
    return float(np.max(np.abs(a[fin] - b[fin]) / np.maximum(1.0, np.abs(b[fin])), initial=0.0))


def rel(a, b):
    ok = np.isfinite(b) & (b > 1e-300)
    return float(np.max(np.abs(a[ok] - b[ok]) / b[ok], initial=0.0))


def main():
    quick = "--quick" in sys.argv
    rng = np.random.default_rng(20261116)
    n = 2000 if quick else 20000
    worst_l = worst_p = 0.0
    total = 0
    print("| case | d | kappa | points | bits changed | logpdf max diff (mixed) | pdf max diff (rel) |")
    print("|---|---:|---:|---|---:|---:|---:|")
    for d in (1, 2, 3, 5, 8, 9, 16, 17, 30, 50, 100):
        for kappa in (1.0, 1e2, 1e4, 1e6, 1e9):
            cov = cov_with_kappa(rng, d, kappa) * 10.0 ** rng.uniform(-2, 2)
            mu = rng.normal(size=d) * 10.0 ** rng.uniform(-1, 3)
            L = np.linalg.cholesky(cov)
            for df in (float("inf"), 3.0, 1e6):
                D = (S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)) if np.isinf(df)
                     else S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=df))
                for kind in ("self", "iso", "far", "near", "axis"):
                    x = points(rng, kind, n if d <= 30 else n // 4, d, mu, L)
                    rows, idx, _ = D._points(sd.asarray(x), "logpdf")
                    D.logpdf(sd.asarray(x[:1]))
                    args = (rows, idx, D._mu, D._L, D._Ll, D._ldh, D._ldl, D._ok, D._df, D._ih, D._il)
                    lf, lp = np.asarray(_k.mv_logpdf(*args, D._fast, False)), np.asarray(_k.mv_logpdf(*args, None, False))
                    pf, pp = np.asarray(_k.mv_logpdf(*args, D._fast, True)), np.asarray(_k.mv_logpdf(*args, None, True))
                    dl, dp = mixed(lf, lp), rel(pf, pp)
                    worst_l, worst_p, total = max(worst_l, dl), max(worst_p, dp), total + len(lf)
                    name = "normal" if np.isinf(df) else f"t df={df:g}"
                    print(f"| {name} | {d} | {kappa:.0e} | {kind} | {np.mean(lf != lp):.3f} | {dl:.2e} | {dp:.2e} |")
    for (r, c) in ((1, 1), (3, 2), (2, 7), (10, 5), (9, 17), (30, 20)):
        for kappa in (1.0, 1e3, 1e6, 1e9):
            U, V = cov_with_kappa(rng, r, kappa), cov_with_kappa(rng, c, np.sqrt(kappa)) * 10.0 ** rng.uniform(-2, 2)
            M = rng.normal(size=(r, c))
            D = S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V))
            LU, LV = np.linalg.cholesky(U), np.linalg.cholesky(V)
            for scale in (1.0, 100.0, 1e-8):
                m = n if r * c <= 60 else n // 8
                x = M + scale * (LU @ rng.normal(size=(m, r, c)) @ LV.T)
                rows, idx, _ = D._points(sd.asarray(x), "logpdf")
                D.logpdf(sd.asarray(x[:1]))
                args = (rows, idx, D._mean, D._U, D._Ul, D._V, D._Vl, D._lu[0], D._lu[1], D._lv[0], D._lv[1], D._ok)
                lf, lp = np.asarray(_k.matnorm_logpdf(*args, *D._fast, False)), np.asarray(_k.matnorm_logpdf(*args, None, None, False))
                pf, pp = np.asarray(_k.matnorm_logpdf(*args, *D._fast, True)), np.asarray(_k.matnorm_logpdf(*args, None, None, True))
                dl, dp = mixed(lf, lp), rel(pf, pp)
                worst_l, worst_p, total = max(worst_l, dl), max(worst_p, dp), total + len(lf)
                print(f"| matrix normal x{scale:g} | {r}x{c} | {kappa:.0e} | self | {np.mean(lf != lp):.3f} | {dl:.2e} | {dp:.2e} |")
    print(f"\n{total:,} points. Largest fast - precise: logpdf {worst_l:.3e} (mixed), pdf {worst_p:.3e} (relative).")


if __name__ == "__main__":
    main()
