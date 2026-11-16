"""T10: accuracy of the Wishart, inverse Wishart and matrix normal log densities and entropies
against 50-digit references (mpmath on the matrices as given), over the condition number of
the scale and the dimension (T30), and scipy's error at the same points. Errors are
|error| / max(1, |value|).

    .venv/bin/python bench/stats/t10/accuracy.py > bench/stats/t10/accuracy.md
"""

import math
import warnings

import mpmath as mp
import numpy as np
import scipy.stats as st

import stardust as sd

mp.mp.dps = 50


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (q * np.exp(np.linspace(0.0, math.log(kappa), d))) @ q.T
    return (a + a.T) / 2.0


def mvlgamma(p, a):
    return p * (p - 1) / mp.mpf(4) * mp.log(mp.pi) + mp.fsum(mp.loggamma(a + mp.mpf(1 - j) / 2) for j in range(1, p + 1))


def mvdigamma(p, a):
    return mp.fsum(mp.digamma(a + mp.mpf(1 - j) / 2) for j in range(1, p + 1))


def ref_wishart(df, V, xs, inverse):
    p = V.shape[0]
    n = mp.mpf(df)
    Vm = mp.matrix(V.tolist())
    ldv = mp.log(mp.det(Vm))
    out = []
    for x in xs:
        X = mp.matrix(x.tolist())
        ldx = mp.log(mp.det(X))
        if not inverse:
            tr = sum((mp.inverse(Vm) * X)[i, i] for i in range(p))
            out.append((n - p - 1) / 2 * ldx - tr / 2 - n * p / 2 * mp.log(2) - n / 2 * ldv - mvlgamma(p, n / 2))
        else:
            tr = sum((Vm * mp.inverse(X))[i, i] for i in range(p))
            out.append(n / 2 * ldv - n * p / 2 * mp.log(2) - mvlgamma(p, n / 2) - (n + p + 1) / 2 * ldx - tr / 2)
    if not inverse:
        h = (p + 1) / mp.mpf(2) * ldv + p * (p + 1) / mp.mpf(2) * mp.log(2) + mvlgamma(p, n / 2) \
            - (n - p - 1) / 2 * mvdigamma(p, n / 2) + n * p / 2
    else:
        h = (p + 1) / mp.mpf(2) * ldv - p * (p + 1) / mp.mpf(2) * mp.log(2) + mvlgamma(p, n / 2) \
            - (n + p + 1) / 2 * mvdigamma(p, n / 2) + n * p / 2
    return [float(v) for v in out], float(h)


def ref_matnorm(M, U, V, xs):
    n, p = M.shape
    Um, Vm = mp.matrix(U.tolist()), mp.matrix(V.tolist())
    lu, lv = mp.log(mp.det(Um)), mp.log(mp.det(Vm))
    Ui, Vi = mp.inverse(Um), mp.inverse(Vm)
    out = []
    for x in xs:
        D = mp.matrix((x - 0.0).tolist()) - mp.matrix(M.tolist())
        T = Vi * D.T * Ui * D
        tr = sum(T[i, i] for i in range(p))
        out.append(-n * p / mp.mpf(2) * mp.log(2 * mp.pi) - p / mp.mpf(2) * lu - n / mp.mpf(2) * lv - tr / 2)
    h = n * p / mp.mpf(2) * (1 + mp.log(2 * mp.pi)) + p / mp.mpf(2) * lu + n / mp.mpf(2) * lv
    return [float(v) for v in out], float(h)


def err(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


def main():
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(0)
    print("| distribution | p | df | kappa | sd logpdf | scipy logpdf | sd entropy | scipy entropy |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for inverse in (False, True):
        for p in (2, 5, 12):
            for df_extra in (0.5, 10.0, 1e8):
                df = p - 1 + df_extra if df_extra < 1 else p + df_extra
                for kappa in (1.0, 1e4, 1e8):
                    V = spd(rng, p, kappa)
                    S = (st.invwishart if inverse else st.wishart)(min(df, 1e6), V)
                    xs = S.rvs(3, random_state=1)
                    xs = xs.reshape((-1, p, p))
                    # exactly symmetric, from the lower triangle (the one we read; scipy's samples
                    # are symmetric only to rounding, which moves X^-1 of a near-singular X)
                    lo = np.tril(xs)
                    xs = lo + np.transpose(np.tril(xs, -1), (0, 2, 1))
                    ref, href = ref_wishart(df, V, xs, inverse)
                    D = (sd.stats.InverseWishart if inverse else sd.stats.Wishart)(df=df, scale=sd.asarray(V))
                    Sd = (st.invwishart if inverse else st.wishart)(df, V)
                    e1 = err(D.logpdf(sd.asarray(xs)), ref)
                    try:
                        e2 = err(Sd.logpdf(xs.transpose(1, 2, 0)), ref)
                    except Exception:
                        e2 = float("nan")
                    e3 = err(D.entropy(), href)
                    e4 = err(Sd.entropy(), href)
                    name = "InverseWishart" if inverse else "Wishart"
                    print(f"| {name} | {p} | {df:g} | {kappa:.0e} | {e1:.1e} | {e2:.1e} | {e3:.1e} | {e4:.1e} |", flush=True)
    for (n, p) in ((2, 3), (4, 4), (10, 6)):
        for kappa in (1.0, 1e4, 1e8):
            M = rng.normal(size=(n, p))
            U, V = spd(rng, n, kappa), spd(rng, p, kappa)
            S = st.matrix_normal(M, U, V)
            xs = S.rvs(3, random_state=2).reshape((-1, n, p))
            ref, href = ref_matnorm(M, U, V, xs)
            D = sd.stats.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V))
            print(f"| MatrixNormal | {n}x{p} | - | {kappa:.0e} | {err(D.logpdf(sd.asarray(xs)), ref):.1e} | "
                  f"{err(S.logpdf(xs), ref):.1e} | {err(D.entropy(), href):.1e} | {err(S.entropy(), href):.1e} |",
                  flush=True)


if __name__ == "__main__":
    main()
