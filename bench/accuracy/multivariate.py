"""The multivariate and matrix distributions of sd.stats (the group mv of the stats area:
sweep ids stats/mv/...): their densities, entropies and pmfs, and the multivariate normal and
t cdfs.

References: tests/stats/ref/mv/<family>.npz for the log densities, densities and entropies and
tests/stats/ref/mv/<set>.npz for the cdfs (tools/stats_ref.py), read through
tests/stats/refdata.py, whose contract functions (mv_density_contract, bvn_contract, ...) judge
here as in the tests. Every density point is inside the guaranteed region (kappa <= 1e9, d and
p <= 100, Wishart df <= 1e30): |error| <= 1e-14 max(1, |value|) (docs/api/stats.md §3), for
the density that error carried through exp, relative 1e-14 max(1, |logpdf|) per point. The
cdf regions and bounds are tests/stats/test_mvcdf.py's.
"""

from __future__ import annotations

import numpy as np

from .common import Custom, compare_libs, region, ulps

GUARANTEED = 1e-14


def _families():
    from tests.stats import refdata

    return refdata.MV_FAMILIES


def _groups(family):
    from tests.stats import refdata

    return refdata.mv_groups(family)


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
    if lib == "sd":
        from tests.stats import refdata

        return refdata.mv_values(family, method, g)
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
    from tests.stats import refdata as S

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
    mask, err, bound = S.mv_density_contract(method, got, hi, lo, lp)["guaranteed"]
    got, hi, lo, err, where = got[mask], hi[mask], lo[mask], err[mask], where[mask]
    with np.errstate(all="ignore"):
        ulp, rel = ulps(got, hi), R.rel_error(got, hi, lo)
        if method == "pdf":
            return {"guaranteed": region(where, ulp=ulp, rel=rel, contract=err, bound=bound, excluded=excluded)}
    e = region(where, ulp=ulp, mixed=err / 2.0**-52, contract=err, bound=bound, excluded=excluded)
    e["nonfinite"] = int((np.isfinite(hi) & ~np.isfinite(got)).sum())  # the mixed measure's companion
    return {"guaranteed": e}


# ------------------------------------------------------------------------------ cdfs


def _mv(S, X, df=np.inf, atol=1e-6, lib="sd"):
    """(values, error estimates) of P(X <= x) per row of X, covariance S (d, d) or one per row;
    scipy has no estimate (zeros), R has no multivariate cdf in its base packages (None)."""
    from tests.stats import refdata

    if lib == "sd":
        return refdata.mv_cdf_eval(S, X, df=df, atol=atol)
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


def _cdf_set(name):
    from tests.stats import refdata

    return refdata.mv_cdf(name)


def _cdf_regions(v, e, hi, lo, where, contract, lib, cond=None):
    """The region entries of a cdf set from its contract ({region: (mask, error, bound)}); a
    comparator's NaN or inf where the reference is finite is left out of its regions and counted."""
    from tests.stats import refdata

    keep = np.ones(len(v), bool) if lib == "sd" else np.isfinite(v) | ~np.isfinite(hi)
    with np.errstate(all="ignore"):
        rel = refdata.cdf_rel(v, hi, lo)
        ulp = ulps(v, hi)
    out = {}
    for label, (mask, err, bound) in contract.items():
        m = mask & keep
        if not mask.any():
            continue
        exc = None if lib == "sd" else int((mask & ~keep).sum())
        if label == "tiny":
            out[label] = region(where[m], abs_err=err[m], contract=err[m], bound=bound, excluded=exc)
            continue
        extra = {}
        if label == "loose" and cond is not None:  # where cond is finite (a perturbed value that underflows has no log)
            f = np.isfinite(cond[m])
            extra["rel_cond"] = rel[m][f] / (2.0**-53 * np.maximum(cond[m][f], 1.0))
        out[label] = region(where[m], ulp=ulp[m], rel=rel[m], contract=err[m], bound=bound, excluded=exc, **extra)
    return out


def _bvn(lib="sd"):
    from tests.stats import refdata

    z = _cdf_set("bvn")
    h, k, r, hi, lo = z["h"], z["k"], z["r"], z["hi"], z["lo"]
    S = np.tile(np.eye(2), (len(h), 1, 1))
    S[:, 0, 1] = S[:, 1, 0] = r
    r_ = _mv(S, np.c_[h, k], lib=lib)
    if r_ is None:
        return None
    v, e = r_
    where = np.array([f"h={a!r} k={b!r} r={c!r}" for a, b, c in zip(h.tolist(), k.tolist(), r.tolist())])
    return _cdf_regions(v, e, hi, lo, where, refdata.bvn_contract(v, e, hi, lo), lib)


def _tvn(lib="sd"):
    from tests.stats import refdata

    z = _cdf_set("tvn")
    h, R3, hi, lo = z["h"], z["r"], z["hi"], z["lo"]
    r_ = _mv(refdata.corr(R3), h, lib=lib)
    if r_ is None:
        return None
    v, e = r_
    cond = refdata.tvn_cond(h, R3)  # the test's condition number, from sd's own values
    where = np.array([f"h={a} r={b}" for a, b in zip(h.tolist(), R3.tolist())])
    return _cdf_regions(v, e, hi, lo, where, refdata.tvn_contract(v, e, hi, lo, R3, cond), lib, cond)


def _t_exact(name, lib="sd"):
    from tests.stats import refdata

    z = _cdf_set(name)
    h, R3, nu, hi, lo = z["h"], z["r"], z["nu"], z["hi"], z["lo"]
    d = h.shape[1]
    S = np.tile(np.eye(d), (len(h), 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))[: 1 if d == 2 else 3]):
        S[:, a, b] = S[:, b, a] = R3[:, q]
    r_ = _mv(S, h, df=nu, lib=lib)
    if r_ is None:
        return None
    v, e = r_
    where = np.array([f"h={a} r={b} nu={c!r}" for a, b, c in zip(h.tolist(), R3.tolist(), nu.tolist())])
    return _cdf_regions(v, e, hi, lo, where, refdata.t_exact_contract(d, v, e, hi, lo), lib)


def _factor(kind, lib="sd"):
    """QMC against the one-factor references: per atol, the actual error and its ratio to the
    reported estimate, held to tests/stats/refdata.py's qmc_contract; the t at d <= 3 is exact
    (the scale mixture) and held to T_EXACT_REL against the Arb references."""
    from tests.stats import refdata

    z = _cdf_set(f"factor_{kind}")
    h, lam, d, nu, ref, ref_lo = z["h"], z["lam"], z["d"], z["nu"], z["hi"], z["lo"]
    out = {}
    if kind == "n":
        runs = [(f"atol={a:g}", a, np.arange(len(d)), 1e-12) for a in (1e-6, 1e-4)]
    else:
        runs = [("exact d<=3", None, np.flatnonzero(d <= 3), 0.0), ("atol=1e-05", 1e-5, np.flatnonzero(d > 3), 1e-10)]
    for label, atol, idx, slack in runs:
        vals, est, where = [], [], []
        for i in idx:
            n = d[i]
            S = np.outer(lam[i, :n], lam[i, :n])
            np.fill_diagonal(S, 1.0)
            r_ = _mv(S, h[i, :n], df=nu[i], atol=atol or 1e-6, lib=lib)
            if r_ is None:
                return None
            v, e = r_
            vals.append(v[0])
            est.append(e[0])
            where.append(f"d={n} h={h[i, :n].tolist()} lam={lam[i, :n].tolist()} nu={float(nu[i])!r}")
        vals, est, where = np.array(vals), np.array(est), np.array(where)
        err = np.abs(vals - ref[idx])
        excluded = None
        if lib != "sd":  # NaN: the library rejects the point (mvtnorm's pmvt takes integer df only)
            keep = np.isfinite(err)
            excluded = int((~keep).sum())
            vals, err, est, where, idx = vals[keep], err[keep], est[keep], where[keep], idx[keep]
        if atol is None:
            parts = [refdata.t_exact_contract(int(d[i]), vals[j : j + 1], est[j : j + 1], ref[i : i + 1],
                                              ref_lo[i : i + 1])["guaranteed"] for j, i in enumerate(idx)]  # fmt: skip
            c = np.array([p[1][0] / p[2] for p in parts])
            rel = refdata.cdf_rel(vals, ref[idx], ref_lo[idx])
            out[label] = region(where, ulp=ulps(vals, ref[idx]), rel=rel, contract=c, bound=1.0, excluded=excluded)
            continue
        if lib != "sd":  # no error estimate: the actual error against 1.5 atol only
            out[label] = region(where, abs_err=err, contract=err / (1.5 * atol), bound=1.0, excluded=excluded)
            continue
        out[label] = region(where, abs_err=err, est_ratio=err / est, contract=refdata.qmc_contract(err, est, atol, slack),
                            bound=1.0)  # fmt: skip
    return out


def sweeps():
    out = []
    for family, (cls, methods) in _families().items():
        for m in methods:
            fn = lambda lib="sd", f=family, m=m: _measure_method(f, m, lib)  # noqa: E731
            out.append(Custom("stats", f"mv/{cls}.{m}", fn, compare_libs(fn, ("scipy", "R"))))
    for name, fn in (
        ("cdf.bvn", _bvn),
        ("cdf.tvn", _tvn),
        ("cdf.bvt", lambda lib="sd": _t_exact("bvt", lib)),
        ("cdf.tvt", lambda lib="sd": _t_exact("tvt", lib)),
        ("cdf.factor_normal", lambda lib="sd": _factor("n", lib)),
        ("cdf.factor_t", lambda lib="sd": _factor("t", lib)),
    ):
        out.append(Custom("stats", f"mv/{name}", fn, compare_libs(fn, ("scipy", "R"), seconds=300)))
    return out
