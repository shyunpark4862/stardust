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

from .common import Custom, region

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


def _measure_method(family, method):
    import stardust as sd
    from tests.special import refdata as R

    gots, his, los, lps, where = [], [], [], [], []
    for g in _groups(family):
        D = _dist(family, g)
        if method == "entropy":
            got = np.atleast_1d(np.asarray(D.entropy(), dtype=float))
        else:
            got = np.atleast_1d(np.asarray(getattr(D, method)(sd.asarray(g["x"])), dtype=float))
        hi, lo = g[f"{method}_hi"], g[f"{method}_lo"]
        gots.append(got)
        his.append(hi)
        los.append(lo)
        lps.append(g["logpdf_hi"] if method == "pdf" else hi)
        where += [f"{_label(family, g)} point {i}" for i in range(len(hi))]
    got, hi, lo, lp = (np.concatenate(v) for v in (gots, his, los, lps))
    where = np.array(where)
    with np.errstate(all="ignore"):
        ulp, rel = R.ulp_error(got, hi, lo), R.rel_error(got, hi, lo)
        if method == "pdf":  # the logpdf's error through exp, where the density is a normal double
            keep = np.abs(lp) < 700
            got, hi, lo, lp, ulp, rel, where = (v[keep] for v in (got, hi, lo, lp, ulp, rel, where))
            c = rel / np.maximum(1.0, np.abs(lp))
            return {"guaranteed": region(where, ulp=ulp, rel=rel, contract=c, bound=GUARANTEED)}
        m = R.mixed_error(got, hi, lo)
    return {"guaranteed": region(where, ulp=ulp, rel=rel, mixed=m / 2.0**-52, contract=m, bound=GUARANTEED)}


# ------------------------------------------------------------------------------ cdfs


def _bvn():
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / "bvn.npz")
    h, k, r, hi, lo = z["h"], z["k"], z["r"], z["hi"], z["lo"]
    S = np.tile(np.eye(2), (len(h), 1, 1))
    S[:, 0, 1] = S[:, 1, 0] = r
    v, e = T.mv(S, np.c_[h, k])
    where = np.array([f"h={a!r} k={b!r} r={c!r}" for a, b, c in zip(h, k, r)])
    return _exact_regions(
        v, e, hi, lo, where, loose=lambda rel, m: rel / (10 * T.EPS * np.maximum(1.0, -np.log(hi[m])))
    )


def _exact_regions(v, e, hi, lo, where, loose, guaranteed=None, cond=None):
    """guaranteed (relative 1e-14), loose (the test's bound for the rest of the normal range)
    and tiny (P < 1e-300: absolute 1e-300)."""
    from tests.special import refdata as R
    from tests.stats import test_mvcdf as T

    with np.errstate(all="ignore"):
        rel = T.rel_err(v, hi, lo)
        ulp = R.ulp_error(v, hi, lo)
    norm = hi >= 1e-300
    g = norm & (hi >= 1e-15) if guaranteed is None else norm & guaranteed
    out = {}
    est = np.where(e == 0, 0.0, np.inf)  # the exact paths report an error of 0
    if g.any():
        out["guaranteed"] = region(
            where[g], ulp=ulp[g], rel=rel[g], contract=np.maximum(rel[g], est[g]), bound=GUARANTEED
        )
    m = norm & ~g
    if m.any():
        extra = {}
        if cond is not None:  # where cond is finite (a perturbed value that underflows has no log)
            f = np.isfinite(cond[m])
            extra["rel_cond"] = rel[m][f] / (2.0**-53 * np.maximum(cond[m][f], 1.0))
        out["loose"] = region(
            where[m], ulp=ulp[m], rel=rel[m], contract=np.maximum(loose(rel[m], m), est[m]), bound=1.0, **extra
        )
    t = ~norm
    if t.any():
        a = np.abs(v[t] - hi[t])
        out["tiny"] = region(where[t], abs_err=a, contract=a, bound=1e-300)
    return out


def _tvn():
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / "tvn.npz")
    h, R3, hi, lo = z["h"], z["r"], z["hi"], z["lo"]
    v, e = T.mv(T.corr(R3), h)
    cond = np.full(len(h), np.inf)
    norm = hi >= 1e-300
    cond[norm] = T._cond3(h[norm], R3[norm])
    negs = (R3 < 0).sum(axis=1)
    g = ((hi >= 1e-15) & (negs <= 1)) | ((hi >= 1e-10) & (cond <= 100))
    where = np.array([f"h={list(a)} r={list(b)}" for a, b in zip(h, R3)])
    # the rest: within 8 u cond of the inputs' rounding where cond is finite (else unbounded)
    loose = lambda rel, m: np.where(np.isfinite(cond[m]), rel / (8 * 2.0**-53 * np.maximum(cond[m], 1.0)), 0.0)  # noqa: E731
    return _exact_regions(v, e, hi, lo, where, loose, guaranteed=g, cond=cond)


def _t_exact(name):
    from tests.special import refdata as R
    from tests.stats import test_mvcdf as T

    z = np.load(T.REF / f"{name}.npz")
    h, R3, nu, hi, lo = z["h"], z["r"], z["nu"], z["hi"], z["lo"]
    d = h.shape[1]
    S = np.tile(np.eye(d), (len(h), 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))[: 1 if d == 2 else 3]):
        S[:, a, b] = S[:, b, a] = R3[:, q]
    v, e = T.mv(S, h, df=nu)
    with np.errstate(all="ignore"):
        rel = T.rel_err(v, hi, lo)
        ulp = R.ulp_error(v, hi, lo)
    where = np.array([f"h={list(a)} r={list(b)} nu={c!r}" for a, b, c in zip(h, R3, nu)])
    c = np.maximum(rel, np.where(e == 0, 0.0, np.inf))
    return {"guaranteed": region(where, ulp=ulp, rel=rel, contract=c, bound=T.T_EXACT_REL[name])}


def _factor(kind):
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
            v, e = T.mv(S, h[i, :n], df=nu[i], atol=atol or 1e-6)
            err.append(abs(v[0] - ref[i]))
            est.append(e[0])
            where.append(f"d={n} h={list(h[i, :n])} lam={list(lam[i, :n])} nu={nu[i]!r}")
        err, est, where = np.array(err), np.array(est), np.array(where)
        if atol is None:  # the exact mixture against the ~1e-11 quadrature: relative 1e-13
            rel = err / ref[idx]
            c = np.maximum(rel, np.where(est == 0, 0.0, np.inf))
            out[label] = region(where, rel=rel, contract=c, bound=1e-13)
            continue
        c = np.maximum.reduce([est / atol, err / (1.5 * atol), err / (2 * est + slack)])
        out[label] = region(where, abs_err=err, est_ratio=err / est, contract=c, bound=1.0)
    return out


def sweeps():
    out = []
    for family, (cls, methods) in FAMILIES.items():
        for m in methods:
            out.append(Custom("multivariate", f"{cls}.{m}", lambda f=family, m=m: _measure_method(f, m)))
    out += [
        Custom("multivariate", "cdf.bvn", _bvn),
        Custom("multivariate", "cdf.tvn", _tvn),
        Custom("multivariate", "cdf.bvt", lambda: _t_exact("bvt")),
        Custom("multivariate", "cdf.tvt", lambda: _t_exact("tvt")),
        Custom("multivariate", "cdf.factor_normal", lambda: _factor("n")),
        Custom("multivariate", "cdf.factor_t", lambda: _factor("t")),
    ]
    return out
