"""Reference values for the multivariate and matrix distributions (T9, T10) with Arb.

    .venv/bin/python tools/mv_ref.py                 # every family, on 10 processes
    .venv/bin/python tools/mv_ref.py wishart --jobs=4

writes tests/stats/ref_mv/<family>.npz, read by bench/accuracy (python -m bench.accuracy).
The inputs come from seeded generators and are stored, so the file is the sweep: a rerun
writes the same file.

A family is a list of groups; group i is the keys "g<i>_<name>" (the distribution's
parameters, the points x, and per method hi + lo as double-doubles). The values are the
textbook definitions in ball arithmetic on the inputs as given (the doubles are exact balls),
at doubling precision until every ball is settled (tools/arbref.py): log densities and
entropies to 2^-110 max(1, |value|), densities to 2^-110 relative.

    mvnormal     logpdf, pdf, entropy;  d = 1, 2, 5, 20, 100; kappa(cov) = 1, 1e3, 1e6, 1e9
    mvt          the same for df = 0.7, 4.5, 30, 1e9
    wishart      logpdf, entropy;  p = 1, 2, 5, 12, 30; df = p - 1/2, p + 10, p + 1e8, p + 1e30;
    invwishart   kappa(scale) = 1, 1e4, 1e9 (the guaranteed region: kappa <= 1e9, df <= 1e30)
    matnormal    logpdf, entropy;  (n, p) up to 20 x 15; kappa = 1, 1e4, 1e9
    multinomial  logpmf, entropy;  n up to 1e5 (the entropy as the exact sum over the marginals)
    dirichlet    logpdf, entropy
    tvn_cond     the condition numbers of the tvn cdf references (tests/stats/ref_mvcdf/tvn.npz)

Determinants and solves are Arb's LU on the matrices as given (arb_mat), so nothing of a
floating-point factorization enters a reference.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import pathlib
import sys

import numpy as np
from flint import arb, arb_mat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from arbref import DOUBLE_TINY, NotSettled, precision, settled, to_dd  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "stats" / "ref_mv"
START, MAX_PREC = 128, 1 << 13


def all_settled(fn):
    """fn() -> {method: [arb, ...]} at doubling precision until every ball is settled (log
    densities and entropies against max(1, |value|), densities relative); {method: (hi, lo)}."""
    p = START
    while p <= MAX_PREC:
        with precision(p):
            out = fn()
            ok = all(
                v.is_finite() and settled(v, floor=DOUBLE_TINY if m == "pdf" else 1)
                for m, vs in out.items()
                for v in vs
            )
        if ok:
            res = {}
            for m, vs in out.items():
                dd = [to_dd(v) for v in vs]
                res[m] = (np.array([h for h, _ in dd]), np.array([lo for _, lo in dd]))
            return res
        p *= 2
    raise NotSettled(fn)


def spd(rng, d, kappa):
    """A symmetric positive definite matrix with eigenvalues 1..kappa (log-spaced)."""
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (q * np.exp(np.linspace(0.0, math.log(kappa), d))) @ q.T
    return (a + a.T) / 2.0


def lower_sym(xs):
    """Exactly symmetric, from the lower triangle (the one the library reads)."""
    lo = np.tril(xs)
    return lo + np.swapaxes(np.tril(xs, -1), -1, -2)


def amat(a):
    return arb_mat(np.asarray(a, dtype=float).tolist())


def mvlgamma(p, a):
    return p * (p - 1) / arb(4) * arb.pi().log() + sum(((a + arb(1 - j) / 2).lgamma() for j in range(1, p + 1)), arb(0))


def mvdigamma(p, a):
    return sum(((a + arb(1 - j) / 2).digamma() for j in range(1, p + 1)), arb(0))


def trace_solve(A, B):
    """tr(A^-1 B) by Arb's LU solve."""
    Y = A.solve(B)
    return sum((Y[i, i] for i in range(A.nrows())), arb(0))


# ------------------------------------------------------------------------------ families
# Each family: groups(rng) -> list of input dicts (numpy arrays and floats), and
# ref(group) -> {method: [arb, ...]} at the current precision.


def mv_groups(rng, dfs):
    out = []
    for df in dfs:
        for d in (1, 2, 5, 20, 100):
            for kappa in (1.0, 1e3, 1e6, 1e9) if d > 1 else (1.0,):
                cov = spd(rng, d, kappa)
                mu = rng.normal(size=d) * 3.0
                L = np.linalg.cholesky(cov)
                z = rng.normal(size=(8 if d < 100 else 3, d)) * 2.0
                far = mu + 30.0 * np.sqrt(np.diag(cov))
                x = np.vstack([mu, mu + z @ L.T, far])
                out.append({"mu": mu, "cov": cov, "x": x, "df": float(df), "kappa": kappa})
    return out


def mv_ref(g):
    mu, cov, xs, df = g["mu"], g["cov"], g["x"], g["df"]
    d = len(mu)
    A = amat(cov)
    ld = A.det().log()
    R = arb_mat([[float(x[i]) for x in xs] for i in range(d)]) - arb_mat([[float(mu[i])] * len(xs) for i in range(d)])
    Y = A.solve(R)
    pi = arb.pi()
    lp = []
    for j in range(len(xs)):
        q = sum((R[i, j] * Y[i, j] for i in range(d)), arb(0))
        if math.isinf(df):
            lp.append(-(d * (2 * pi).log() + ld + q) / 2)
        else:
            nu = arb(df)
            lp.append(
                ((nu + d) / 2).lgamma()
                - (nu / 2).lgamma()
                - arb(d) / 2 * (nu * pi).log()
                - ld / 2
                - (nu + d) / 2 * (q / nu).log1p()
            )
    if math.isinf(df):
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
    pdf = [v.exp() for v in lp]  # measured where |logpdf| < 700 (bench/accuracy)
    return {"logpdf": lp, "pdf": pdf, "entropy": [h]}


def wishart_groups(rng, inverse):
    import stardust as sd

    out = []
    for p in (1, 2, 5, 12, 30):
        for extra in (-0.5, 10.0, 1e8, 1e30):
            df = p + extra
            for kappa in (1.0, 1e4, 1e9) if p > 1 else (1.0,):
                V = spd(rng, p, kappa)
                D = (sd.stats.InverseWishart if inverse else sd.stats.Wishart)(df=df, scale=sd.asarray(V))
                seed = int(rng.integers(2**31))
                xs = [np.asarray(D.sample(3, rng=sd.random.default_rng(seed)))]
                for m in ("mean", "mode"):
                    v = np.asarray(getattr(D, m)())
                    if np.all(np.isfinite(v)):
                        xs.append(v[None])
                out.append({"scale": V, "x": lower_sym(np.concatenate(xs)), "df": df, "kappa": kappa})
    return out


def wishart_ref(g, inverse):
    V, xs, df = g["scale"], g["x"], g["df"]
    p = V.shape[0]
    n = arb(df)
    Vm = amat(V)
    ldv = Vm.det().log()
    lp = []
    for x in xs:
        X = amat(x)
        ldx = X.det().log()
        if not inverse:
            tr = trace_solve(Vm, X)
            lp.append((n - p - 1) / 2 * ldx - tr / 2 - n * p / 2 * arb(2).log() - n / 2 * ldv - mvlgamma(p, n / 2))
        else:
            tr = trace_solve(X, Vm)  # tr(V X^-1) = tr(X^-1 V)
            lp.append(n / 2 * ldv - n * p / 2 * arb(2).log() - mvlgamma(p, n / 2) - (n + p + 1) / 2 * ldx - tr / 2)
    s = -1 if inverse else 1
    c = (n + p + 1) if inverse else (n - p - 1)
    h = (
        (p + 1) / arb(2) * ldv
        + s * p * (p + 1) / arb(2) * arb(2).log()
        + mvlgamma(p, n / 2)
        - c / 2 * mvdigamma(p, n / 2)
        + n * p / 2
    )
    return {"logpdf": lp, "entropy": [h]}


def matnormal_groups(rng):
    import scipy.stats as st

    out = []
    for n, p in ((1, 1), (2, 3), (4, 4), (10, 6), (20, 15)):
        for kappa in (1.0, 1e4, 1e9) if n * p > 1 else (1.0,):
            M = rng.normal(size=(n, p))
            U, V = spd(rng, n, kappa), spd(rng, p, kappa)
            xs = st.matrix_normal(M, U, V).rvs(3, random_state=int(rng.integers(2**31))).reshape((-1, n, p))
            out.append({"mu": M, "rowcov": U, "colcov": V, "x": np.concatenate([M[None], xs]), "kappa": kappa})
    return out


def matnormal_ref(g):
    M, U, V, xs = g["mu"], g["rowcov"], g["colcov"], g["x"]
    n, p = M.shape
    Um, Vm, Mm = amat(U), amat(V), amat(M)
    lu, lv = Um.det().log(), Vm.det().log()
    lp = []
    for x in xs:
        Dm = amat(x) - Mm
        tr = trace_solve(Vm, Dm.transpose() * Um.solve(Dm))  # tr(V^-1 D' U^-1 D)
        lp.append(-n * p / arb(2) * (2 * arb.pi()).log() - p / arb(2) * lu - n / arb(2) * lv - tr / 2)
    h = n * p / arb(2) * (1 + (2 * arb.pi()).log()) + p / arb(2) * lu + n / arb(2) * lv
    return {"logpdf": lp, "entropy": [h]}


def multinomial_groups(rng):
    out = []
    for n, K in ((1, 2), (10, 3), (50, 10), (1000, 5), (100000, 3)):
        p = rng.dirichlet(np.ones(K))
        if K == 5:
            p[0] = 1e-10  # a tiny probability: its log is large, its count almost surely 0
            p /= p.sum()  # (the library takes p summing to 1 within sqrt(eps))
        x = [np.floor(n * p / p.sum())]
        x[0][-1] = n - x[0][:-1].sum()  # >= 0: the others are rounded down
        x += list(rng.multinomial(n, p / p.sum(), size=3).astype(float))
        e = np.zeros(K)
        e[int(np.argmax(p))] = n
        x.append(e)
        out.append({"n": float(n), "p": p, "x": np.array(x, dtype=float)})
    return out


def multinomial_ref(g):
    n, pf, xs = int(g["n"]), g["p"], g["x"]
    P = [arb(float(v)) for v in pf]
    S = sum(P, arb(0))
    lnf = arb(n + 1).lgamma()
    lp = []
    for x in xs:
        lp.append(
            lnf
            - sum((arb(float(v) + 1).lgamma() for v in x), arb(0))
            + sum((float(v) * q.log() for v, q in zip(x, P) if v > 0), arb(0))
            - n * S.log()
        )
    # H = -log n! - n sum q log q + sum_i E[log X_i!], X_i ~ Bin(n, q_i): exact sums over k
    h = -lnf
    for q in P:
        q = q / S
        h -= n * q * q.log()
        r = q / (1 - q)
        pmf, lfact, e = (1 - q) ** n, arb(0), arb(0)
        for k in range(1, n + 1):
            pmf *= r * (n - k + 1) / k
            lfact += arb(k).log()
            e += pmf * lfact
        h += e
    return {"logpmf": lp, "entropy": [h]}


def dirichlet_groups(rng):
    out = []
    for alpha in ([0.5, 0.5], [1.0, 2.0, 3.0], [1e6, 2e6, 3e6], [0.1] * 10, [50.0, 0.3, 7.0]):
        a = np.array(alpha)
        xs = []
        for y in rng.dirichlet(a, size=4):
            y = np.maximum(np.round(y * 2.0**30), 1.0) / 2.0**30  # multiples of 2^-30: the last
            y[-1] = 0.0  # component makes the sum exactly 1
            y[-1] = 1.0 - y.sum()
            if y[-1] > 0:
                xs.append(y)
        out.append({"alpha": a, "x": np.array(xs)})
    return out


def dirichlet_ref(g):
    A = [arb(float(v)) for v in g["alpha"]]
    a0 = sum(A, arb(0))
    K = len(A)
    lnB = sum((a.lgamma() for a in A), arb(0)) - a0.lgamma()
    lp = [-lnB + sum(((a - 1) * arb(float(v)).log() for a, v in zip(A, x)), arb(0)) for x in g["x"]]
    h = lnB + (a0 - K) * a0.digamma() - sum(((a - 1) * a.digamma() for a in A), arb(0))
    return {"logpdf": lp, "entropy": [h]}


FAMILIES = {
    "mvnormal": (lambda rng: mv_groups(rng, (math.inf,)), mv_ref),
    "mvt": (lambda rng: mv_groups(rng, (0.7, 4.5, 30.0, 1e9)), mv_ref),
    "wishart": (lambda rng: wishart_groups(rng, False), lambda g: wishart_ref(g, False)),
    "invwishart": (lambda rng: wishart_groups(rng, True), lambda g: wishart_ref(g, True)),
    "matnormal": (matnormal_groups, matnormal_ref),
    "multinomial": (multinomial_groups, multinomial_ref),
    "dirichlet": (dirichlet_groups, dirichlet_ref),
}
SEEDS = {name: i + 1 for i, name in enumerate(FAMILIES)}


def _work(args):
    family, g = args
    return all_settled(lambda: FAMILIES[family][1](g))


def write(family, jobs):
    gen, _ = FAMILIES[family]
    groups = gen(np.random.default_rng(SEEDS[family]))
    order = sorted(range(len(groups)), key=lambda i: -np.asarray(groups[i]["x"]).size)  # big ones first
    with mp.get_context("spawn").Pool(jobs, maxtasksperchild=8) as pool:
        refs = dict(zip(order, pool.map(_work, [(family, groups[i]) for i in order], chunksize=1)))
    arrays = {}
    for i, g in enumerate(groups):
        for k, v in g.items():
            arrays[f"g{i}_{k}"] = np.asarray(v)
        for m, (hi, lo) in refs[i].items():
            arrays[f"g{i}_{m}_hi"], arrays[f"g{i}_{m}_lo"] = hi, lo
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{family}.npz", **arrays)
    pts = sum(len(r[m][0]) for r in refs.values() for m in r)
    print(f"{family}: {len(groups)} groups, {pts} values -> {(OUT / f'{family}.npz').relative_to(ROOT)}", flush=True)


def write_tvn_cond():
    """The condition numbers of tests/stats/ref_mvcdf/tvn.npz's points: sum over the six inputs of
    |d log P / d log x| by central differences of sd's own values (tests/stats/test_mvcdf.py's
    _cond3; a condition number needs a few digits), inf where P < 1e-300 or a perturbed value
    underflows. Written to ref_mv/tvn_cond.npz, read by bench/accuracy."""
    sys.path.insert(0, str(ROOT))
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / "tvn.npz")
    h, r, hi = z["h"], z["r"], z["hi"]
    cond = np.full(len(h), np.inf)
    norm = hi >= 1e-300
    cond[norm] = T._cond3(h[norm], r[norm])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "tvn_cond.npz", cond=cond)
    print(f"tvn_cond: {len(cond)} points -> {(OUT / 'tvn_cond.npz').relative_to(ROOT)}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("families", nargs="*", help=", ".join((*FAMILIES, "tvn_cond")))
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()
    bad = set(a.families) - {*FAMILIES, "tvn_cond"}
    if bad:
        ap.error(f"unknown families {sorted(bad)}; choose from {', '.join(FAMILIES)}")
    for f in a.families or (*FAMILIES, "tvn_cond"):
        if f == "tvn_cond":
            write_tvn_cond()
        else:
            write(f, min(a.jobs, 10))


if __name__ == "__main__":
    main()
