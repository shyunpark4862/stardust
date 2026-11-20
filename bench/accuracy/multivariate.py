"""The multivariate area: the multivariate and matrix distributions (T9, T10) and the
multivariate normal and t cdfs.

References: tests/stats/ref_mv/<family>.npz (tools/mv_ref.py) for the log densities, densities
and entropies; tests/stats/ref_mvcdf/<set>.npz (tools/mvcdf_ref.py) for the cdfs. Every point
of ref_mv is inside the guaranteed region (kappa <= 1e9, d and p <= 100, Wishart df <= 1e30),
whose contract is |error| <= 1e-14 max(1, |value|) (docs/api/stats.md, T29, T30) -- the measure
of tests/stats/test_multivariate.py and test_matrix.py; the density's is that error carried
through exp, |relative error| <= 1e-14 max(1, |logpdf|). The cdf regions and bounds are
tests/stats/test_mvcdf.py's, through its own helpers.
"""

from __future__ import annotations

import pathlib

import numpy as np

from .common import Custom, compare_libs, region

ROOT = pathlib.Path(__file__).resolve().parents[2]
REF_MV = ROOT / "tests" / "stats" / "ref_mv"
GUARANTEED = 1e-14

# family -> (class name, methods, constructor from a group's arrays)
FAMILIES = {
    "mvnormal": ("MultivariateNormal", ("logpdf", "pdf", "entropy")),
    "mvt": ("MultivariateT", ("logpdf", "pdf", "entropy")),
    "wishart": ("Wishart", ("logpdf", "entropy")),
    "invwishart": ("InverseWishart", ("logpdf", "entropy")),
    "matnormal": ("MatrixNormal", ("logpdf", "entropy")),
    "multinomial": ("Multinomial", ("logpmf", "entropy")),
    "dirichlet": ("Dirichlet", ("logpdf", "entropy")),
}


def _groups(family):
    z = np.load(REF_MV / f"{family}.npz")
    out = {}
    for k in z.files:
        g, key = k[1:].split("_", 1)
        out.setdefault(int(g), {})[key] = z[k]
    return [out[i] for i in sorted(out)]


def _dist(family, g):
    import stardust as sd

    S, a = sd.stats, sd.asarray
    if family == "mvnormal":
        return S.MultivariateNormal(mu=a(g["mu"]), cov=a(g["cov"]))
    if family == "mvt":
        return S.MultivariateT(mu=a(g["mu"]), shape=a(g["cov"]), df=float(g["df"]))
    if family == "wishart":
        return S.Wishart(df=float(g["df"]), scale=a(g["scale"]))
    if family == "invwishart":
        return S.InverseWishart(df=float(g["df"]), scale=a(g["scale"]))
    if family == "matnormal":
        return S.MatrixNormal(mu=a(g["mu"]), rowcov=a(g["rowcov"]), colcov=a(g["colcov"]))
    if family == "multinomial":
        return S.Multinomial(n=float(g["n"]), p=a(g["p"]))
    return S.Dirichlet(alpha=a(g["alpha"]))


def _label(family, g):
    if family in ("mvnormal", "mvt"):
        s = f"d={len(g['mu'])} kappa={float(g['kappa']):g}"
        return s + (f" df={float(g['df']):g}" if family == "mvt" else "")
    if family in ("wishart", "invwishart"):
        return f"p={len(g['scale'])} df={float(g['df']):g} kappa={float(g['kappa']):g}"
    if family == "matnormal":
        return f"{g['mu'].shape[0]}x{g['mu'].shape[1]} kappa={float(g['kappa']):g}"
    if family == "multinomial":
        return f"n={float(g['n']):g} K={len(g['p'])}"
    return f"alpha={np.array2string(g['alpha'], precision=3, separator=',')}"


def _values(family, method, g, lib):
    """The method's values at the group's points from sd, scipy or R (None: no such function)."""
    import stardust as sd

    if lib == "sd":
        D = _dist(family, g)
        if method == "entropy":
            return np.atleast_1d(np.asarray(D.entropy(), dtype=float))
        return np.atleast_1d(np.asarray(getattr(D, method)(sd.asarray(g["x"])), dtype=float))
    if lib == "R":
        from . import rsession

        if method == "entropy" or family not in ("multinomial", "mvnormal", "mvt"):
            return None
        if family == "multinomial":  # base R
            x, p = g["x"], g["p"]
            return np.array([rsession.evaluate("dmultinom(a, prob = b, log = TRUE)", xi, p)[0] for xi in x])
        x, mu, cov = g["x"], g["mu"], g["cov"]  # mvtnorm's dmvnorm and dmvt
        d = len(mu)
        log = "TRUE" if method == "logpdf" else "FALSE"
        dens = (f"dmvnorm(X, mean = M, sigma = S, log = {log})" if family == "mvnormal"
                else f"dmvt(X, delta = M, sigma = S, df = {float(g['df'])!r}, log = {log})")  # fmt: skip
        expr = (f"suppressMessages(library(mvtnorm)); X <- matrix(X, ncol = {d}, byrow = TRUE); "
                f"S <- matrix(S, nrow = {d}, byrow = TRUE); as.numeric({dens})")  # fmt: skip
        return np.atleast_1d(rsession.evaluate_named(expr, {"X": x, "M": mu, "S": cov}))
    import scipy.stats as st

    x = g.get("x")
    if family == "mvnormal":
        D = st.multivariate_normal(g["mu"], g["cov"])
    elif family == "mvt":
        D = st.multivariate_t(g["mu"], g["cov"], df=float(g["df"]))
    elif family in ("wishart", "invwishart"):
        D = (st.invwishart if family == "invwishart" else st.wishart)(float(g["df"]), g["scale"])
        x = np.moveaxis(x, 0, -1)  # scipy takes (p, p, n)
    elif family == "matnormal":
        D = st.matrix_normal(g["mu"], g["rowcov"], g["colcov"])
    elif family == "multinomial":
        D = st.multinomial(float(g["n"]), g["p"])
    else:
        D = st.dirichlet(g["alpha"])
        x = x.T  # scipy takes (K, n)
    if method == "entropy":
        return np.atleast_1d(np.asarray(D.entropy(), dtype=float))
    return np.atleast_1d(np.asarray(getattr(D, method)(x), dtype=float))


def _measure_method(family, method, lib="sd"):
    from tests.special import refdata as R

    gots, his, los, lps, where = [], [], [], [], []
    for g in _groups(family):
        got = _values(family, method, g, lib)
        if got is None:
            return None
        hi, lo = g[f"{method}_hi"], g[f"{method}_lo"]
        gots.append(got)
        his.append(hi)
        los.append(lo)
        lps.append(g["logpdf_hi"] if method == "pdf" else hi)
        where += [f"{_label(family, g)} point {i}" for i in range(len(hi))]
    got, hi, lo, lp = (np.concatenate(v) for v in (gots, his, los, lps))
    where = np.array(where)
    excluded = None
    if lib != "sd":  # a comparator's NaN or inf where the reference is finite: left out and counted
        keep = np.isfinite(got) | ~np.isfinite(hi)
        excluded = int((~keep).sum())
        got, hi, lo, lp, where = (v[keep] for v in (got, hi, lo, lp, where))
    with np.errstate(all="ignore"):
        ulp, rel = R.ulp_error(got, hi, lo), R.rel_error(got, hi, lo)
        if method == "pdf":  # the logpdf's error through exp, where the density is a normal double
            keep = np.abs(lp) < 700
            got, hi, lo, lp, ulp, rel, where = (v[keep] for v in (got, hi, lo, lp, ulp, rel, where))
            c = rel / np.maximum(1.0, np.abs(lp))
            return {"guaranteed": region(where, ulp=ulp, rel=rel, contract=c, bound=GUARANTEED, excluded=excluded)}
        m = R.mixed_error(got, hi, lo)
    return {"guaranteed": region(where, ulp=ulp, rel=rel, mixed=m / 2.0**-52, contract=m, bound=GUARANTEED,
                                 excluded=excluded)}  # fmt: skip


# ------------------------------------------------------------------------------ cdfs


def _mv(S, X, df=np.inf, atol=1e-6, lib="sd"):
    """(values, error estimates) of P(X <= x) per row of X, covariance S (d, d) or one per row;
    scipy has no estimate (zeros), R has no multivariate cdf in its base packages (None)."""
    from tests.stats import test_mvcdf as T

    if lib == "sd":
        return T.mv(S, X, df=df, atol=atol)
    if lib == "R":  # mvtnorm's pmvnorm and pmvt (Genz-Bretz at abseps = atol), a fixed seed
        from . import rsession

        X = np.atleast_2d(X)
        d = X.shape[1]
        S = np.broadcast_to(S, (len(X), d, d))
        dfs = np.broadcast_to(np.asarray(df, dtype=float), (len(X),))
        expr = (
            f"suppressMessages(library(mvtnorm)); X <- matrix(X, ncol = {d}, byrow = TRUE); "
            f"S <- array(S, dim = c({d}, {d}, nrow(X))); "
            f"sapply(seq_len(nrow(X)), function(i) tryCatch({{ set.seed(0); "
            f"alg <- GenzBretz(abseps = {atol!r}, releps = 0); "
            f"if (is.infinite(D[i])) pmvnorm(upper = X[i, ], sigma = S[, , i], algorithm = alg)[1] "
            f"else pmvt(upper = X[i, ], sigma = S[, , i], df = D[i], algorithm = alg)[1] }}, "
            f"error = function(e) NaN))"
        )  # S[, , i] read column-major: each matrix is symmetric
        v = rsession.evaluate_named(expr, {"X": X, "S": np.transpose(S, (0, 2, 1)), "D": dfs})
        return v, np.zeros(len(X))
    if lib != "scipy":
        return None
    import scipy.stats as st

    X = np.atleast_2d(X)
    S = np.broadcast_to(S, (len(X), *np.shape(S)[-2:]))
    dfs = np.broadcast_to(np.asarray(df, dtype=float), (len(X),))
    v = np.empty(len(X))
    for i in range(len(X)):
        z = np.zeros(X.shape[1])
        try:
            if np.isinf(dfs[i]):
                v[i] = st.multivariate_normal(z, S[i], abseps=atol, releps=0.0, seed=0).cdf(X[i])
            else:  # scipy's t cdf takes no tolerance: its default
                v[i] = st.multivariate_t(z, S[i], df=dfs[i]).cdf(X[i], random_state=0)
        except ValueError, np.linalg.LinAlgError:  # rejected (a near-singular matrix): left out, counted
            v[i] = np.nan
    return v, np.zeros(len(X))


def _bvn(lib="sd"):
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / "bvn.npz")
    h, k, r, hi, lo = z["h"], z["k"], z["r"], z["hi"], z["lo"]
    S = np.tile(np.eye(2), (len(h), 1, 1))
    S[:, 0, 1] = S[:, 1, 0] = r
    r_ = _mv(S, np.c_[h, k], lib=lib)
    if r_ is None:
        return None
    v, e = r_
    where = np.array([f"h={a!r} k={b!r} r={c!r}" for a, b, c in zip(h.tolist(), k.tolist(), r.tolist())])
    return _exact_regions(
        v, e, hi, lo, where, loose=lambda rel, m: rel / (10 * T.EPS * np.maximum(1.0, -np.log(hi[m]))), lib=lib
    )


def _exact_regions(v, e, hi, lo, where, loose, guaranteed=None, cond=None, lib="sd"):
    """guaranteed (relative 1e-14), loose (the test's bound for the rest of the normal range)
    and tiny (P < 1e-300: absolute 1e-300)."""
    from tests.special import refdata as R
    from tests.stats import test_mvcdf as T

    with np.errstate(all="ignore"):
        rel = T.rel_err(v, hi, lo)
        ulp = R.ulp_error(v, hi, lo)
    norm = hi >= 1e-300
    g = norm & (hi >= 1e-15) if guaranteed is None else norm & guaranteed
    keep = np.ones(len(v), bool) if lib == "sd" else np.isfinite(v) | ~np.isfinite(hi)
    exc = (lambda mask: int((mask & ~keep).sum())) if lib != "sd" else (lambda mask: None)  # noqa: E731
    out = {}
    ex_g, ex_m, ex_t = exc(g), exc(norm & ~g), exc(~norm)
    norm, g = norm & keep, g & keep
    tiny = ~(hi >= 1e-300) & keep
    est = np.where(e == 0, 0.0, np.inf)  # the exact paths report an error of 0
    if g.any():
        out["guaranteed"] = region(
            where[g], ulp=ulp[g], rel=rel[g], contract=np.maximum(rel[g], est[g]), bound=GUARANTEED, excluded=ex_g
        )
    m = norm & ~g
    if m.any():
        extra = {}
        if cond is not None:  # where cond is finite (a perturbed value that underflows has no log)
            f = np.isfinite(cond[m])
            extra["rel_cond"] = rel[m][f] / (2.0**-53 * np.maximum(cond[m][f], 1.0))
        out["loose"] = region(
            where[m],
            ulp=ulp[m],
            rel=rel[m],
            contract=np.maximum(loose(rel[m], m), est[m]),
            bound=1.0,
            excluded=ex_m,
            **extra,
        )
    t = tiny
    if t.any():
        a = np.abs(v[t] - hi[t])
        out["tiny"] = region(where[t], abs_err=a, contract=a, bound=1e-300, excluded=ex_t)
    return out


def _tvn(lib="sd"):
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / "tvn.npz")
    h, R3, hi, lo = z["h"], z["r"], z["hi"], z["lo"]
    r_ = _mv(T.corr(R3), h, lib=lib)
    if r_ is None:
        return None
    v, e = r_
    cond = np.load(REF_MV / "tvn_cond.npz")["cond"]  # tools/mv_ref.py tvn_cond
    negs = (R3 < 0).sum(axis=1)
    g = ((hi >= 1e-15) & (negs <= 1)) | ((hi >= 1e-10) & (cond <= 100))
    where = np.array([f"h={a} r={b}" for a, b in zip(h.tolist(), R3.tolist())])
    # the rest: within 8 u cond of the inputs' rounding where cond is finite (else unbounded)
    loose = lambda rel, m: np.where(np.isfinite(cond[m]), rel / (8 * 2.0**-53 * np.maximum(cond[m], 1.0)), 0.0)  # noqa: E731
    return _exact_regions(v, e, hi, lo, where, loose, guaranteed=g, cond=cond, lib=lib)


def _t_exact(name, lib="sd"):
    from tests.special import refdata as R
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / f"{name}.npz")
    h, R3, nu, hi, lo = z["h"], z["r"], z["nu"], z["hi"], z["lo"]
    d = h.shape[1]
    S = np.tile(np.eye(d), (len(h), 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))[: 1 if d == 2 else 3]):
        S[:, a, b] = S[:, b, a] = R3[:, q]
    r_ = _mv(S, h, df=nu, lib=lib)
    if r_ is None:
        return None
    v, e = r_
    keep = np.isfinite(v) | ~np.isfinite(hi)
    excluded = None if lib == "sd" else int((~keep).sum())
    v, e, h, R3, nu, hi, lo = (a[keep] for a in (v, e, h, R3, nu, hi, lo))
    with np.errstate(all="ignore"):
        rel = T.rel_err(v, hi, lo)
        ulp = R.ulp_error(v, hi, lo)
    where = np.array([f"h={a} r={b} nu={c!r}" for a, b, c in zip(h.tolist(), R3.tolist(), nu.tolist())])
    c = np.maximum(rel, np.where(e == 0, 0.0, np.inf))
    return {"guaranteed": region(where, ulp=ulp, rel=rel, contract=c, bound=T.T_EXACT_REL[name], excluded=excluded)}


def _factor(kind, lib="sd"):
    """QMC against the one-factor references: per atol, the actual error and its ratio to the
    reported estimate; the contract is the test's (estimate <= atol, error <= 1.5 atol,
    error <= 2 estimate + slack)."""
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / f"factor_{kind}.npz")
    h, lam, d, nu, ref = z["h"], z["lam"], z["d"], z["nu"], z["hi"]
    out = {}
    if kind == "n":
        runs = [(f"atol={a:g}", a, np.arange(len(d)), 1e-12) for a in (1e-6, 1e-4)]
    else:
        exact = np.flatnonzero(d <= 3)
        runs = [("exact d<=3", None, exact, 0.0), ("atol=1e-05", 1e-5, np.flatnonzero(d > 3), 1e-10)]
    for label, atol, idx, slack in runs:
        err, est, where = [], [], []
        for i in idx:
            n = d[i]
            S = np.outer(lam[i, :n], lam[i, :n])
            np.fill_diagonal(S, 1.0)
            r_ = _mv(S, h[i, :n], df=nu[i], atol=atol or 1e-6, lib=lib)
            if r_ is None:
                return None
            v, e = r_
            err.append(abs(v[0] - ref[i]))
            est.append(e[0])
            where.append(f"d={n} h={h[i, :n].tolist()} lam={lam[i, :n].tolist()} nu={float(nu[i])!r}")
        err, est, where = np.array(err), np.array(est), np.array(where)
        excluded = None
        if lib != "sd":  # NaN: the library rejects the point (mvtnorm's pmvt takes integer df only)
            keep = np.isfinite(err)
            excluded = int((~keep).sum())
            err, est, where, idx = err[keep], est[keep], where[keep], idx[keep]
        if atol is None:  # the exact mixture against the ~1e-11 quadrature: relative 1e-13
            rel = err / ref[idx]
            c = np.maximum(rel, np.where(est == 0, 0.0, np.inf))
            out[label] = region(where, rel=rel, contract=c, bound=1e-13, excluded=excluded)
            continue
        if lib != "sd":  # no error estimate: the actual error against 1.5 atol only
            out[label] = region(where, abs_err=err, contract=err / (1.5 * atol), bound=1.0, excluded=excluded)
            continue
        c = np.maximum.reduce([est / atol, err / (1.5 * atol), err / (2 * est + slack)])
        out[label] = region(where, abs_err=err, est_ratio=err / est, contract=c, bound=1.0)
    return out


def sweeps():
    out = []
    for family, (cls, methods) in FAMILIES.items():
        for m in methods:
            fn = lambda lib="sd", f=family, m=m: _measure_method(f, m, lib)  # noqa: E731
            out.append(Custom("multivariate", f"{cls}.{m}", fn, compare_libs(fn, ("scipy", "R"))))
    for name, fn in (
        ("cdf.bvn", _bvn),
        ("cdf.tvn", _tvn),
        ("cdf.bvt", lambda lib="sd": _t_exact("bvt", lib)),
        ("cdf.tvt", lambda lib="sd": _t_exact("tvt", lib)),
        ("cdf.factor_normal", lambda lib="sd": _factor("n", lib)),
        ("cdf.factor_t", lambda lib="sd": _factor("t", lib)),
    ):
        out.append(Custom("multivariate", name, fn, compare_libs(fn, ("scipy", "R"), seconds=300)))
    return out
