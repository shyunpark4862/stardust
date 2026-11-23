"""sd.smooth's accuracy sweeps: one per reference family of tests/smooth/ref (tools/smooth_ref.py) once the
functions it measures are public.

SM0 made the references, the comparators' records (scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1 and
R 4.6.2: tests/smooth/data, from tools/smooth_probe.py) and the float64 prototypes' measurements
(tests/smooth/data/prototype.json); each milestone adds the sweep of its functions. The plan
(docs/design/smooth.md §7.5), each judged through tests/smooth/refdata.py (local: lowess since SM1):

  local       SM1: lowess fitted values, predict and leverage on local.npz in units of C_LOCAL kappa u scale;
              comparators statsmodels' lowess and R's lowess, live on the same points
  loess       SM2: loess fitted values, predict and leverage on local.npz and local_nd.npz, the same units
              (windows of kappa > 2^36 apart); comparator R's loess(surface = 'direct'), live
  spline      SM3: the coefficients (spline.npz, u kappa max|c|), fitted values, leverages and tr L
              (spline_fit.npz, u kappa max(C_LEVERAGE, kappa)) and the GCV and CV values; comparators scipy's
              make_smoothing_spline and make_lsq_spline, R's smooth.spline(all.knots = TRUE) at the converted
              lambda
  fitpack     SM3: make_splrep's spline against the penalized problem of its own knots and the s it reached
  kreg        SM4: Nadaraya-Watson and local linear on kreg.npz (the far points where every weight underflows
              included); comparator statsmodels' KernelReg
  kde         SM5: pdf (relative, (8 + 2p + log2 n + D0) u), logpdf and cdf on kde.npz, the compact kernels and
              the reflection; the binned density against its method bound; comparators scipy's gaussian_kde,
              scikit-learn's KernelDensity (exact sums), R's density and statsmodels' KDEUnivariate (binned)
  bandwidth   SM5: the rules and the cross-validation criteria on bandwidth.npz; comparators R's bw.* and
              statsmodels' bandwidths
  savgol      SM6: savgol_coeffs on savgol.npz (u max|h|) and savgol_filter on savgol_filter.npz (gamma_{N+8}
              sum |h_j x_j|); comparator scipy (whose high orders the sweep shows wrong)
"""

from __future__ import annotations

import math

import numpy as np

from .common import Custom, compare_libs, region

# ---- local (SM1: lowess) -------------------------------------------------------------------------------------
# lowess at it = 0 on every set of local.npz with frac 0.3, 2/3 and 1: the fitted values (kind 0), predict at the
# evaluation points (kind 1) and the leverages, in units of the contract C_LOCAL kappa_1 u sum |l_j||y_j| (the
# leverage: C_LOCAL kappa_1 u, the same form with sum |l_j| >= 1 for the hat row); regions by method and by the
# data's kind. Points with fewer than two positive weights have no fit (y and NaN by the rule) and are left out. statsmodels (it = 0; predict through xvals) and
# R's lowess (fitted values; sets without weights) are the comparator columns.

_LOWESS = (("lowess_f0.3", 0.3), ("lowess_f0.67", 2 / 3), ("lowess_f1", 1.0))
_KINDS = (
    ("spread", {"clustered", "scale", "subnormal", "magnitude"}),
    ("ties", {"ties", "degenerate", "tiny", "halfzero"}),
    ("weights", {"weights"}),
)


def _kind(tags):
    for name, want in _KINDS:
        if tags & want:
            return name
    return "plain"


def _lowess_values(lib, s, frac):
    """(fitted, predict at s.xe or None, leverage or None) of a library at it = 0, or None without it."""
    if lib == "sd":
        import stardust as sd

        w = None if s.w is None else sd.asarray(s.w)
        r = sd.smooth.lowess(sd.asarray(s.x), sd.asarray(s.y), frac=frac, it=0, weights=w)
        return (
            np.asarray(r.fittedvalues.tolist()),
            np.asarray(r.predict(sd.asarray(s.xe)).tolist()),
            np.asarray(r.leverage.tolist()),
        )
    if s.w is not None or s.n < 2:
        return None
    if lib == "statsmodels":
        from statsmodels.nonparametric.smoothers_lowess import lowess

        fitted = lowess(s.y, s.x, frac=frac, it=0, return_sorted=False)
        pred = lowess(s.y, s.x, frac=frac, it=0, xvals=s.xe)
        return fitted, pred, None
    if lib == "R":
        from . import rsession

        expr = f"o <- order(x); r <- numeric(length(x)); r[o] <- lowess(x, y, f = {frac!r}, iter = 0, delta = 0)$y; r"
        return rsession.evaluate_named(expr, {"x": s.x, "y": s.y}), None, None
    return None


def _local_regions(lib):
    from tests.smooth import refdata
    from tests.smooth import samples as S

    r = refdata.ref("local")
    pts = {}
    for sid in sorted(set(r.inputs["set"].tolist())):
        s = S.scatter(sid)
        kind = _kind(s.tags)
        for method, frac in _LOWESS:
            got = _lowess_values(lib, s, frac)
            if got is None:
                continue
            m = r.methods[method]
            for k, values in ((0, got[0]), (1, got[1])):
                if values is None:
                    continue
                pos, idx = refdata.at(r, method, sid, k)
                for p, i in zip(pos, idx):
                    f = {name: v[p] for name, v in m.fields.items()}
                    if not f["npos"] >= 2 or math.isnan(m.hi[p]):
                        continue
                    key = f"{'fit' if k == 0 else 'predict'}/{kind}"
                    e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
                    bound = refdata.local_bound(f["scale"], f["kappa"])
                    e["where"].append(f"{sid} {method} #{i}")
                    e["contract"].append(refdata.units(float(values[i]), m.hi[p], m.lo[p], bound))
                    e["rel"].append(abs(values[i] - m.hi[p]) / f["scale"] if f["scale"] else 0.0)
                    if k == 0 and got[2] is not None:
                        lev, ref = float(got[2][i]), f["leverage"]
                        lb = refdata.C_LOCAL * f["kappa"] * refdata.U  # the hat row has sum |l_j| >= 1
                        e = pts.setdefault("leverage", {"where": [], "contract": [], "rel": []})
                        e["where"].append(f"{sid} {method} #{i}")
                        e["contract"].append(0.0 if lev == ref else abs(lev - ref) / lb if lb else math.inf)
                        e["rel"].append(abs(lev - ref) / abs(ref) if ref else 0.0)
    return {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    } or None


# ---- loess (SM2) --------------------------------------------------------------------------------------------------
# loess on every set of local.npz (one variable: degree 0, 1, 2, span 0.5, 0.75, 1.5, the sets' weights) and
# local_nd.npz (several, normalize off): the fitted values (kind 0), predict at the evaluation points (kind 1) and
# the leverages in units of the contract C_LOCAL kappa u sum |l_j||y_j| (the leverage C_LOCAL kappa u), regions by
# method and by the data's kind; windows of kappa > 2^36 (decision SM107: the coordinates' rounding times kappa^2)
# are their own region "beyond", in units of kappa^2 u sum |l_j||y_j|. R's loess(surface='direct',
# statistics='exact') is the comparator column (sets without weights).

_LOESS_KAPPA_MAX = 2.0**36


def _loess_values(lib, X, y, w, Xe, degree, span, one_d):
    if lib == "sd":
        import stardust as sd

        r = sd.smooth.loess(
            sd.asarray(X), sd.asarray(y), span=span, degree=degree, normalize=False,
            weights=None if w is None else sd.asarray(w),
        )  # fmt: skip
        return (
            np.asarray(r.fittedvalues.tolist()),
            np.asarray(r.predict(sd.asarray(Xe)).tolist()),
            np.asarray(r.leverage.tolist()),
        )
    if lib == "R":
        if w is not None or len(y) < 4:
            return None
        from . import rsession

        Xm = np.asarray(X, float).reshape(len(y), -1)
        p = Xm.shape[1]
        cols = "; ".join(f"X{a + 1} <- X[seq_len(n) + {a} * n]" for a in range(p))
        ecols = "; ".join(f"E{a + 1} <- E[seq_len(m) + {a} * m]" for a in range(p))
        form = " + ".join(f"X{a + 1}" for a in range(p))
        new = ", ".join(f"X{a + 1} = E{a + 1}" for a in range(p))
        expr = (
            f"n <- length(y); m <- length(E) / {p}; {cols}; {ecols}; d <- data.frame({form.replace(' + ', ', ')}, y = y); "
            f"f <- tryCatch(loess(y ~ {form}, d, span = {span!r}, degree = {degree}, normalize = FALSE, "
            "control = loess.control(surface = 'direct', statistics = 'exact')), error = function(e) NULL); "
            f"if (is.null(f)) rep(NaN, n + m) else c(fitted(f), predict(f, data.frame({new})))"
        )
        Em = np.asarray(Xe, float).reshape(-1, p)
        out = rsession.evaluate_named(expr, {"X": Xm.T.ravel(), "y": y, "E": Em.T.ravel()})
        return out[: len(y)], out[len(y) :], None
    return None


def _loess_regions(lib):
    from tests.smooth import refdata
    from tests.smooth import samples as S

    pts = {}
    for fam in ("local", "local_nd"):
        r = refdata.ref(fam)
        for sid in sorted(set(r.inputs["set"].tolist())):
            one_d = fam == "local"
            s = S.scatter(sid) if one_d else S.multi(sid)
            X, Xe = (s.x, s.xe) if one_d else (s.X, s.Xe)
            kind = _kind(s.tags) if one_d else "nd"
            for method in [m for m in r.methods if m.startswith("loess")]:
                degree, span = int(method[7]), float(method.split("_s")[1])
                if min(s.n, math.floor(s.n * span + 1e-5)) < (1 if degree == 0 else 2 if one_d or degree == 1 else 6):
                    continue
                try:
                    got = _loess_values(lib, X, s.y, getattr(s, "w", None), Xe, degree, span, one_d)
                except ValueError:
                    continue
                if got is None:
                    continue
                m = r.methods[method]
                for k, values in ((0, got[0]), (1, got[1])):
                    pos, idx = refdata.at(r, method, sid, k)
                    for p_, i in zip(pos, idx):
                        f = {name: v[p_] for name, v in m.fields.items()}
                        if math.isnan(m.hi[p_]) or math.isnan(values[i]):
                            continue
                        beyond = f["kappa"] > _LOESS_KAPPA_MAX
                        key = "beyond" if beyond else f"{'fit' if k == 0 else 'predict'}/{fam}/{kind}"
                        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
                        bound = (
                            f["kappa"] ** 2 * refdata.U * f["scale"]
                            if beyond
                            else refdata.local_bound(f["scale"], f["kappa"])
                        )
                        e["where"].append(f"{sid} {method} #{i}")
                        e["contract"].append(refdata.units(float(values[i]), m.hi[p_], m.lo[p_], bound))
                        e["rel"].append(abs(values[i] - m.hi[p_]) / f["scale"] if f["scale"] else 0.0)
                        if k == 0 and got[2] is not None and not beyond:
                            lev, ref = float(got[2][i]), f["leverage"]
                            lb = refdata.C_LOCAL * f["kappa"] * refdata.U
                            e = pts.setdefault(f"leverage/{fam}", {"where": [], "contract": [], "rel": []})
                            e["where"].append(f"{sid} {method} #{i}")
                            e["contract"].append(0.0 if lev == ref else abs(lev - ref) / lb if lb else math.inf)
                            e["rel"].append(abs(lev - ref) / abs(ref) if ref else 0.0)
    return {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    } or None


# ---- savgol (SM6: savgol_coeffs, savgol_filter) -----------------------------------------------------------------
# coeffs: every coefficient of savgol.npz (the exact rationals of tests/smooth/samples.SAVGOL) in units of the
# contract C_SAVGOL u max|h*| (refdata.savgol_bound); coeffs_high: orders up to the window, where the degree
# recurrence at the window's points and scipy's Vandermonde lose every digit (scipy's ValueError there counts as
# NaN), against the orthonormal Gram recurrence in Arb balls at 6 N + 300 bits (radius below 2^-80 max|h|), the same
# units; filter_<mode>: every output sample of savgol_filter.npz in units of gamma_{N+8} sum |h_j*| |x_j|
# (refdata.savgol_filter_bound). Comparator:
# scipy's savgol_coeffs(use='dot') and savgol_filter in the same units.

_SAVGOL_HIGH = ((64, 63, 0, None), (101, 100, 0, None), (101, 60, 0, 0), (101, 90, 1, 3), (201, 150, 0, 0),
                (201, 200, 1, 100), (301, 250, 2, 5), (301, 120, 0, None))  # fmt: skip


def _arb_savgol(n, d, deriv, pos):
    """The coefficients (use='dot', delta = 1) in Arb at 6 n + 300 bits: (float64 of the midpoints, max radius)."""
    from flint import arb, ctx

    old = ctx.prec
    ctx.prec = 6 * n + 300
    try:
        c = arb(n - 1) / 2
        s = [arb(i) - c for i in range(n)]
        sp = (arb(2 * (n // 2) - 1) / 2 if n % 2 == 0 else arb(n // 2)) if pos is None else arb(pos)
        sp = sp - c
        n2 = arb(n) * n
        sqb = [arb(0)] + [((arb(k) * k * (n2 - k * k)) / (4 * (4 * arb(k) * k - 1))).sqrt() for k in range(1, d + 1)]
        q0 = 1 / arb(n).sqrt()
        prev, cur = [arb(0)] * n, [q0] * n
        dprev, dcur = [arb(0)] * (deriv + 1), [q0] + [arb(0)] * deriv
        out = [arb(0)] * n
        for k in range(d + 1):
            if k >= deriv:
                out = [o + dcur[deriv] * q for o, q in zip(out, cur)]
            if k == d:
                break
            nxt = [(si * q - sqb[k] * pq) / sqb[k + 1] for si, q, pq in zip(s, cur, prev)]
            dcur, dprev = (
                [
                    (sp * dcur[v] + (v * dcur[v - 1] if v else 0) - sqb[k] * dprev[v]) / sqb[k + 1]
                    for v in range(deriv + 1)
                ],
                dcur,
            )
            prev, cur = cur, nxt
        return np.array([float(o.mid()) for o in out]), max(float(o.rad()) for o in out)
    finally:
        ctx.prec = old


def _savgol_lib(lib):
    if lib == "sd":
        import stardust as sd

        return (
            lambda n, d, deriv, pos: np.asarray(sd.smooth.savgol_coeffs(n, d, deriv=deriv, pos=pos, use="dot")),
            lambda x, n, d, deriv, mode, cval: np.asarray(
                sd.smooth.savgol_filter(x, n, d, deriv=deriv, mode=mode, cval=cval)
            ),
        )
    if lib == "scipy":
        import warnings

        from scipy.signal import savgol_coeffs, savgol_filter

        def coeffs(n, d, deriv, pos):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    return savgol_coeffs(n, d, deriv=deriv, pos=pos, use="dot")
                except ValueError:  # its Vandermonde overflows at high orders: NaN, as a failed point
                    return np.full(n, np.nan)

        return coeffs, lambda x, n, d, deriv, mode, cval: savgol_filter(x, n, d, deriv=deriv, mode=mode, cval=cval)
    return None


def _savgol_regions(lib):
    from tests.smooth import refdata
    from tests.smooth import samples as S

    fns = _savgol_lib(lib)
    if fns is None:
        return None
    coeffs, filt = fns
    out = {}
    r = refdata.ref("savgol")
    m = r.methods["coeffs"]
    where, con, rel = [], [], []
    for case in S.SAVGOL:
        sid = S.savgol_id(case)
        pos, idx = refdata.at(r, "coeffs", sid)
        norm = float(r.group_inputs["norm_inf"][refdata.group_of(r, sid)])
        bound = refdata.savgol_bound(norm) if norm > 0 else refdata.U
        got = coeffs(*case)
        for p, i in zip(pos, idx):
            hi, lo = float(m.hi[p]), float(m.lo[p])
            g = float(got[i])
            where.append(f"{sid}#{int(i)}")
            con.append(refdata.units(g, hi, lo, bound))
            rel.append(abs((g - hi) - lo) / norm if norm > 0 else abs(g))
    out["coeffs"] = region(where, contract=np.array(con), rel=np.array(rel), bound=1.0)
    where, con, rel = [], [], []
    for case in _SAVGOL_HIGH:
        ref, _ = _arb_savgol(*case)
        norm = float(np.abs(ref).max())
        got = coeffs(*case)
        for i in range(case[0]):
            where.append(f"{case}#{i}")
            err = abs(float(got[i]) - ref[i])
            con.append(err / refdata.savgol_bound(norm))
            rel.append(err / norm)
    out["coeffs_high"] = region(where, contract=np.array(con), rel=np.array(rel), bound=1.0)
    rf = refdata.ref("savgol_filter")
    for mode in S.SAVGOL_MODES:
        mm = rf.methods[mode]
        where, con = [], []
        for sid in rf.group_inputs["set"].tolist():
            sig, f = sid.split("|")
            n, d, deriv = map(int, f.split(","))
            got = filt(S.signal(sig).x, n, d, deriv, mode, S.SAVGOL_CVAL)
            pos, idx = refdata.at(rf, mode, sid)
            for p, i in zip(pos, idx):
                b = refdata.savgol_filter_bound(n, float(mm.fields["scale"][p]))
                where.append(f"{sid}#{int(i)}")
                con.append(refdata.units(float(got[i]), float(mm.hi[p]), float(mm.lo[p]), b))
        out[f"filter_{mode}"] = region(where, contract=np.array(con), bound=1.0)
    return out


# ---- spline (SM3: smoothing_spline, pspline, make_lsq_spline) ----------------------------------------------------
# The coefficients of spline.npz in units of C_SPLINE u kappa max|c| (kappa: the stacked matrix's 2-norm condition,
# the reference's), and on spline_fit.npz the leverages (u kappa max(C_LEVERAGE, kappa)) and tr L (n times that);
# regions by method family (ss: the smoothing spline at lam = c range^3, ps: the P-spline, lsq) and by the data's kind.
# A fit the guard refuses (decision SM58: an estimated condition number above 2^43) is excluded and counted.
# Comparator: scipy's make_smoothing_spline (sets of distinct x without weights) and make_lsq_spline, coefficients in
# the same units.


def _spline_fit(lib, s, method, gin):
    """The coefficients and (sd) the result, or None when the library has no such fit or refuses it."""
    if lib == "sd":
        import stardust as sd

        w = None if s.w is None else sd.asarray(s.w)
        x, y = sd.asarray(s.x), sd.asarray(s.y)
        try:
            if method.startswith("ss_"):
                r = sd.smooth.smoothing_spline(x, y, lam=float(gin[f"lam_{method}"]), weights=w)
            elif method.startswith("ps_"):
                r = sd.smooth.pspline(x, y, lam=float(method[3:]), n_segments=int(gin["n_segments"]), weights=w)
            else:
                b = sd.smooth.make_lsq_spline(x, y, sd.asarray(np.asarray(gin["t_lsq"])), w=w)
                return np.asarray(b.c.tolist()), None
        except sd.linalg.LinAlgError:
            return "refused"
        return np.asarray(r.spline.c.tolist()), r
    if lib == "scipy":
        import scipy.interpolate as si

        try:
            if method.startswith("ss_"):
                if s.w is not None or np.unique(s.x).size != s.n or np.any(np.diff(s.x) <= 0):
                    return None
                return si.make_smoothing_spline(s.x, s.y, lam=float(gin[f"lam_{method}"])).c, None
            if method == "lsq":
                w = None if s.w is None else s.w
                if np.any(np.diff(s.x) < 0):
                    return None
                return si.make_lsq_spline(s.x, s.y, np.asarray(gin["t_lsq"]), w=w).c, None
        except Exception:  # noqa: BLE001 -- scipy's refusal is a missing value, not a failure of the sweep
            return None
    return None


def _spline_regions(lib):
    from tests.smooth import refdata
    from tests.smooth import samples as S

    rc, rf = refdata.ref("spline"), refdata.ref("spline_fit")
    pts, refused = {}, {}

    def put(key, where, units, rel):
        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
        e["where"].append(where)
        e["contract"].append(units)
        e["rel"].append(rel)

    for g, sid in enumerate(rc.group_inputs["set"].tolist()):
        s = S.scatter(sid)
        gin = {k: v[g] for k, v in rc.group_inputs.items()}
        gf = refdata.group_of(rf, sid)
        kind = _kind(s.tags)
        for method in rc.methods:
            pos, idx = refdata.at(rc, method, sid)
            if pos.size == 0:
                continue
            tm = rf.methods[f"trl_{method}"]
            j = np.flatnonzero(tm.index == gf)
            if j.size == 0:
                continue
            kappa = float(tm.fields["kappa"][j[0]])
            got = _spline_fit(lib, s, method, gin)
            fam = method.split("_")[0]
            if got is None:
                continue
            if isinstance(got, str):
                refused[fam] = refused.get(fam, 0) + 1
                continue
            coef, res = got
            m = rc.methods[method]
            cmax = float(np.abs(m.hi[pos]).max())
            bound = refdata.spline_bound(kappa, cmax)
            for p, i in zip(pos, idx):
                put(f"coef/{fam}/{kind}", f"{sid} {method} #{i}", refdata.units(float(coef[i]), m.hi[p], m.lo[p], bound),
                    abs(coef[i] - m.hi[p]) / cmax if cmax else 0.0)  # fmt: skip
            if res is None:
                continue
            fpos, fidx = refdata.at(rf, method, sid)
            lev = np.asarray(res.leverage.tolist())
            lb = refdata.leverage_bound(kappa)
            for p, i in zip(fpos, fidx):
                ref = float(rf.methods[method].fields["leverage"][p])
                put(f"leverage/{fam}", f"{sid} {method} #{i}", refdata.units(float(lev[i]), ref, 0.0, lb),
                    abs(lev[i] - ref))  # fmt: skip
            put(f"trace/{fam}", f"{sid} {method}",
                refdata.units(res.df, tm.hi[j[0]], tm.lo[j[0]], refdata.trace_bound(s.n, kappa)),
                abs(res.df - tm.hi[j[0]]) / max(tm.hi[j[0]], 1.0))  # fmt: skip
    out = {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    }
    for fam, n in sorted(refused.items()):
        out[f"refused/{fam}"] = region([f"{n} fits"], excluded=n)
    return out or None


# ---- fitpack (SM3: make_splrep) ----------------------------------------------------------------------------------
# make_splrep on the sorted samples and on random sets at s = 0.09 n (and 0.9 n): the smoothing condition
# |sum (w r)^2 - s| / (0.001 s) (the contract: at most 1 unless the least-squares polynomial is within s), with the
# number of knots; comparator scipy's make_splrep, the same measure.


def _fitpack_regions(lib):
    import warnings

    from tests.smooth import samples as S

    sets = []
    for sid in S.scatter_ids(tag="sorted", exclude=("subnormal", "degenerate", "tiny", "magnitude", "ties")):
        s0 = S.scatter(sid)
        if s0.w is None:
            sets.append((sid, s0.x, s0.y))
    g = np.random.default_rng(20261123)
    for n in (100, 1000, 10000):
        x = np.sort(10.0 * g.random(n))
        sets.append((f"random/{n}", x, np.sin(x) + 0.3 * g.standard_normal(n)))
    if lib == "sd":
        import stardust as sd

        def fit(x, y, s):
            b = sd.smooth.make_splrep(sd.asarray(x), sd.asarray(y), s=s)
            return np.asarray(b(sd.asarray(x)).tolist()), int(b.t.size)
    elif lib == "scipy":
        import scipy.interpolate as si

        def fit(x, y, s):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = si.make_splrep(x, y, s=s)
            return b(x), b.t.size
    else:
        return None
    where, con, rel = [], [], []
    for sid, x, y in sets:
        for f in (0.09, 0.9):
            s = f * x.size
            v, nt = fit(x, y, s)
            fp = float(np.sum((v - y) ** 2))
            where.append(f"{sid} s={s:g} ({nt} knots)")
            con.append(0.0 if (nt == 8 and fp <= s) else abs(fp - s) / (0.001 * s))
            rel.append(abs(fp - s) / s)
    return {"condition": region(where, contract=np.array(con), rel=np.array(rel), bound=1.0)}


# ---- kreg (SM4: kernel_regression) ---------------------------------------------------------------------------------
# predict at every point of kreg.npz (the evaluation points, kind 1, and the far point, kind 2, where every double
# weight underflows) for nw_a, nw_b (degree 0) and ll_a, ll_b (degree 1) in units of the contract
# (C_KREG kappa + 1) u sum |l_j||y_j| (refdata.kreg_bound; kappa 1 for the local constant, the intercept's variance
# inflation a [M^-1]_00 at the point for the local line); regions by degree, 1-D or n-D and far. Comparator:
# statsmodels' KernelReg(reg_type='lc' or 'll').fit at the same points and bandwidths (sets without weights; its far
# points are NaN, 0/0).


def _kreg_kappa(X, w, x0, bw):
    z = (X - x0) / bw
    D = np.sum(z * z, axis=1)
    pos = np.ones(len(X), bool) if w is None else w > 0
    v = np.where(pos, (1.0 if w is None else w) * np.exp(-(D - D[pos].min()) / 2), 0.0)
    a = v.sum()
    zb = (v[:, None] * z).sum(axis=0) / a
    o = np.argsort(-v, kind="stable")  # C = R^T R from the weighted centred rows, heaviest first (graded rows)
    R = np.linalg.qr((np.sqrt(v)[:, None] * (z - zb))[o], mode="r")
    d = np.abs(np.diag(R))
    if not np.all(d > 0):
        return math.inf
    g = np.linalg.solve(R.T, zb)
    return 1.0 + a * float(g @ g)


def _kreg_regions(lib):
    import warnings

    from tests.smooth import refdata
    from tests.smooth import samples as S

    r = refdata.ref("kreg")
    pts = {}
    for g, sid in enumerate(r.group_inputs["set"].tolist()):
        if sid in S.scatter_ids():
            s = S.scatter(sid)
            X, y, w, Xe = s.x[:, None], s.y, s.w, s.xe[:, None]
        else:
            s = S.multi(sid)
            X, y, w, Xe = s.X, s.y, s.w, s.Xe
        p = X.shape[1]
        if lib != "sd" and w is not None:
            continue
        far = np.asarray(r.group_inputs["far"][g], dtype=float)
        Q = np.concatenate([Xe, far[None, :]])
        for tag in ("a", "b"):
            bw = np.asarray(r.group_inputs[f"bw_{tag}"][g], dtype=float)
            for name, deg in (("nw", 0), ("ll", 1)):
                if lib == "sd":
                    import stardust as sd

                    res = sd.smooth.kernel_regression(
                        sd.asarray(X), sd.asarray(y), degree=deg, bw=sd.asarray(bw),
                        weights=None if w is None else sd.asarray(w),
                    )  # fmt: skip
                    got = np.asarray(res.predict(sd.asarray(Q)).tolist())
                elif lib == "statsmodels":
                    import statsmodels.api as sm

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        k = sm.nonparametric.KernelReg(y, X, "c" * p, reg_type="lc" if deg == 0 else "ll", bw=list(bw))
                        got = np.asarray(k.fit(Q)[0], dtype=float)
                else:
                    return None
                m = r.methods[f"{name}_{tag}"]
                for kind in (1, 2):
                    pos, idx = refdata.at(r, f"{name}_{tag}", sid, kind)
                    for q, i in zip(pos, idx):
                        row = int(i) if kind == 1 else Xe.shape[0]
                        kap = 1.0 if deg == 0 else _kreg_kappa(X, w, Q[row], bw)
                        sc = float(m.fields["scale"][q])
                        key = f"{'nw' if deg == 0 else 'll'}/{'1d' if p == 1 else 'nd'}" + ("/far" if kind == 2 else "")
                        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
                        e["where"].append(f"{sid} {name}_{tag} #{int(i)}{' far' if kind == 2 else ''}")
                        e["contract"].append(
                            refdata.units(float(got[row]), m.hi[q], m.lo[q], refdata.kreg_bound(sc, kap))
                        )
                        e["rel"].append(abs(float(got[row]) - m.hi[q]) / sc if sc else 0.0)
    return {
        key: region(p["where"], rel=np.array(p["rel"]), contract=np.array(p["contract"]), bound=1.0)
        for key, p in sorted(pts.items())
    } or None


# ---- kde and bandwidth (SM5: kde, the bw_* rules) -------------------------------------------------------------------
# kde: pdf (relative, kde_pdf_bound: (8 + 2p + log2 n + D0) kappa_A u with kappa_A = sqrt(cond H) for a full
# covariance, decision SM117), logpdf (kde_logpdf_bound), cdf (kde_cdf_bound) on every evaluation point of kde.npz
# (the far tails included); the compact kernels' pdf (kde_kernel_bound) and the reflection; the binned density at its
# grid points against the exact density in units of its method bound (binned_bound; 1-D at the references' bandwidth
# and a quarter of it, n-D at 48 points per axis). Comparators: scipy's gaussian_kde (the same covariance) and
# scikit-learn's KernelDensity (1-D, the same bandwidth, exact sums) on the exact regions; statsmodels'
# KDEUnivariate(fft=True) and R's density (the recorded values) on the binned region, in the same units.
# bandwidth: the closed forms (relative, units of C_BW u) and the roots and minimizers (units of BW_ROOT_REL) on
# bandwidth.npz; comparators R's bw.* (recorded) and statsmodels' cv_ls / cv_ml (live, the minimizers).


def _kde_regions(lib):
    import warnings

    from tests.smooth import refdata
    from tests.smooth import samples as S

    r = refdata.ref("kde")
    pts = {}

    def put(key, where, units, rel):
        e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
        e["where"].append(where)
        e["contract"].append(units)
        e["rel"].append(rel)

    for g, sid in enumerate(r.group_inputs["set"].tolist()):
        s = S.density(sid)
        n, p = s.n, s.p
        H = np.asarray(r.group_inputs["H"][g], dtype=float)
        if p == 1:
            h = float(r.group_inputs["bw"][g])
            Hm = np.array([[h * h]])
        else:
            Hm = H
        kap = 1.0 if p == 1 else math.sqrt(np.linalg.cond(H))
        Hinv = np.linalg.inv(Hm)
        if lib == "sd":
            import stardust as sd

            w = None if s.w is None else sd.asarray(s.w)
            res = sd.smooth.kde(sd.asarray(s.x), bw=h if p == 1 else sd.asarray(H), weights=w)
            pdf = np.asarray(res.pdf(sd.asarray(s.xe)).tolist())
            lp = np.asarray(res.logpdf(sd.asarray(s.xe)).tolist())
            cdf = np.asarray(res.cdf(sd.asarray(s.xe)).tolist()) if p == 1 else None
        elif lib == "scipy":
            from scipy.stats import gaussian_kde

            data = s.x if p == 1 else s.x.T
            k = gaussian_kde(data, bw_method=1.0, weights=s.w)
            k.covariance = Hm  # the references' covariance
            k.cho_cov = np.linalg.cholesky(Hm)
            k.log_det = 2 * np.log(np.diag(k.cho_cov * np.sqrt(2 * np.pi))).sum()
            q = s.xe if p == 1 else s.xe.T
            pdf, lp = k(q), k.logpdf(q)
            cdf = np.array([k.integrate_box_1d(-np.inf, t) for t in s.xe]) if p == 1 else None
        elif lib == "sklearn":
            if p != 1:
                continue
            from sklearn.neighbors import KernelDensity

            kd = KernelDensity(bandwidth=h, rtol=0, atol=0).fit(s.x[:, None], sample_weight=s.w)
            lp = kd.score_samples(s.xe[:, None])
            pdf, cdf = np.exp(lp), None
        elif lib in ("statsmodels", "R"):  # the binned region only (below)
            continue
        else:
            return None
        sub = "1d" if p == 1 else "nd"
        for meth, got in (("pdf", pdf), ("logpdf", lp), ("cdf", cdf)):
            if got is None:
                continue
            m = r.methods[meth]
            pos, idx = refdata.at(r, meth, sid, 1)
            for q, i in zip(pos, idx):
                ref = float(m.hi[q])
                d = (s.xe[i] - s.x) if p > 1 else (s.xe[i] - s.x)[:, None]
                d0 = float(np.min(np.einsum("ij,jk,ik->i", d, Hinv, d)))
                if meth == "pdf":
                    b = refdata.kde_pdf_bound(n, p, d0, ref, kap)
                elif meth == "logpdf":
                    b = refdata.kde_logpdf_bound(n, p, ref, kap)
                else:
                    b = refdata.kde_cdf_bound(n)
                tail = "/tail" if d0 > 100 else ""
                rel = abs(float(got[i]) - ref) / abs(ref) if ref else 0.0
                put(
                    f"{meth}/{sub}{tail}", f"{sid} #{int(i)}", refdata.units(float(got[i]), ref, float(m.lo[q]), b), rel
                )
        if lib != "sd" or p != 1:
            continue
        wn = np.full(n, 1.0 / n) if s.w is None else s.w / s.w.sum()
        for kern in ("epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
            got = np.asarray(
                sd.smooth.kde(sd.asarray(s.x), bw=h, kernel=kern, weights=w).pdf(sd.asarray(s.xe)).tolist()
            )
            m = r.methods[f"pdf_{kern}"]
            a2 = {"epanechnikov": 5.0, "rectangular": 3.0, "triangular": 6.0, "biweight": 7.0}.get(kern)
            a = math.sqrt(a2) if a2 else 1 / math.sqrt(1 / 3 - 2 / math.pi**2)
            for q, i in zip(*refdata.at(r, f"pdf_{kern}", sid, 1)):
                z = (s.xe[i] - s.x) / h
                rr = np.abs(z) / a
                dk = {
                    "epanechnikov": 1.5 * rr / a**2,
                    "rectangular": np.where(np.abs(rr - 1) < 1e-12, 1e12, 0.0),
                    "triangular": np.full(n, 1 / a**2),
                    "biweight": 3.75 * rr * np.abs(1 - rr * rr) / a**2,
                    "cosine": np.pi * np.abs(np.sin(np.pi * np.minimum(rr, 1))) / (2 * a * a),
                }[kern]
                slope = float(np.sum(np.where(rr < 1 + 1e-12, wn * np.abs(z) * dk, 0.0))) / h
                ref = float(m.hi[q])
                b = refdata.kde_kernel_bound(n, ref, slope)
                put("kernels", f"{sid} {kern} #{int(i)}", refdata.units(float(got[i]), ref, float(m.lo[q]), b),
                    abs(float(got[i]) - ref) / ref if ref else 0.0)  # fmt: skip
        if s.bounds is not None:
            got = np.asarray(
                sd.smooth.kde(sd.asarray(s.x), bw=h, bounds=s.bounds, weights=w).pdf(sd.asarray(s.xe)).tolist()
            )
            m = r.methods["pdf_reflect"]
            for q, i in zip(*refdata.at(r, "pdf_reflect", sid, 1)):
                t = float(s.xe[i])
                pts_ = [t] + [2 * b_ - t for b_ in s.bounds if math.isfinite(b_)]
                parts = np.asarray(res.pdf(sd.asarray(pts_)).tolist())
                b = sum(refdata.kde_rel(n, 1, float(np.min(((v - s.x) / h) ** 2))) * f for v, f in zip(pts_, parts))
                ref = float(m.hi[q])
                put("reflect", f"{sid} #{int(i)}", refdata.units(float(got[i]), ref, float(m.lo[q]), b + 2.0**-53 * ref),
                    abs(float(got[i]) - ref) / ref if ref else 0.0)  # fmt: skip
    # the binned density
    for g, sid in enumerate(r.group_inputs["set"].tolist()):
        s = S.density(sid)
        if s.p != 1:
            if lib != "sd":
                continue
            import stardust as sd

            res = sd.smooth.kde(sd.asarray(s.x))
            H = np.asarray(res.covariance.tolist())
            d = res.density(n=48)
            e = res.density(n=48, binned=False)
            steps = [(float(a[-1]) - float(a[0])) / 47 for a in d.x]
            k0 = (2 * math.pi) ** (-s.p / 2) / math.sqrt(np.linalg.det(H))
            b = refdata.binned_bound_nd(steps, np.diag(np.linalg.inv(H)), k0)
            err = float(np.max(np.abs(np.asarray(d.density.tolist()) - np.asarray(e.density.tolist()))))
            put("binned/nd", f"{sid} n=48", err / b, err)
            continue
        h0 = float(r.group_inputs["bw"][g])
        for scale in (1.0, 0.25):
            hh = h0 * scale
            import stardust as sd

            exact_res = sd.smooth.kde(sd.asarray(s.x), bw=hh, weights=None if s.w is None else sd.asarray(s.w))
            if lib == "sd":
                d = exact_res.density(n=512)
                x, y = np.asarray(d.x.tolist()), np.asarray(d.density.tolist())
            elif lib == "statsmodels":
                if s.w is not None:
                    continue
                import statsmodels.api as sm

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    k = sm.nonparametric.KDEUnivariate(s.x)
                    k.fit(bw=hh, fft=True, gridsize=512, cut=3)
                x, y = np.asarray(k.support), np.asarray(k.density)
            elif lib == "R":
                if scale != 1.0:
                    continue
                rec = refdata.data("r").get(f"density/{sid}/gaussian")
                if rec is None:
                    continue
                v = rec["values"]
                x, y = np.linspace(v[0], v[1], 512), np.asarray(v[2:])
            else:
                continue
            exact = np.asarray(exact_res.pdf(sd.asarray(x)).tolist())
            step = (x[-1] - x[0]) / (x.size - 1)
            err = float(np.max(np.abs(y - exact)))
            put("binned/1d", f"{sid} h x {scale:g}", err / refdata.binned_bound(step, hh), err)
    return {
        key: region(p_["where"], rel=np.array(p_["rel"]), contract=np.array(p_["contract"]), bound=1.0)
        for key, p_ in sorted(pts.items())
    } or None


def _bandwidth_regions(lib):
    import warnings

    from tests.smooth import refdata
    from tests.smooth import samples as S

    r = refdata.ref("bandwidth")
    pts = {}
    closed = ("scott", "silverman", "nrd0", "nrd")
    r_index = {"nrd0": 0, "nrd": 1, "sj_ste": 4, "sj_dpi": 5}
    for meth in sorted(r.methods):
        if meth.endswith("_value"):
            continue
        m = r.methods[meth]
        for q, at in enumerate(m.index):
            sid = str(r.inputs["set"][at])
            s = S.density(sid)
            ref = float(m.hi[q])
            if ref == 0:
                continue
            if lib == "sd":
                import stardust as sd

                x = sd.asarray(s.x)
                w = None if s.w is None else sd.asarray(s.w)
                got = {
                    "scott": lambda: sd.smooth.bw_scott(x, weights=w),
                    "silverman": lambda: sd.smooth.bw_silverman(x, weights=w),
                    "nrd0": lambda: sd.smooth.bw_nrd0(x),
                    "nrd": lambda: sd.smooth.bw_nrd(x),
                    "sj_ste": lambda: sd.smooth.bw_sj(x),
                    "sj_dpi": lambda: sd.smooth.bw_sj(x, method="dpi"),
                    "isj": lambda: sd.smooth.bw_isj(x, n_grid=int(r.group_inputs["isj_n"][0])),
                    "cvls_min": lambda: sd.smooth.bw_cv_ls(x),
                    "cvml_min": lambda: sd.smooth.bw_cv_ml(x),
                }[meth]()
            elif lib == "R":
                if meth not in r_index or s.w is not None:
                    continue
                got = refdata.data("r")[f"bw/{sid}"]["values"][r_index[meth]]
            elif lib == "statsmodels":
                if meth not in ("cvls_min", "cvml_min"):
                    continue
                import statsmodels.api as sm

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    got = float(
                        sm.nonparametric.KDEMultivariate(s.x, "c", bw="cv_ls" if meth == "cvls_min" else "cv_ml").bw[0]
                    )
            else:
                return None
            rel = abs((got - ref) - float(m.lo[q])) / abs(ref)
            bound = refdata.C_BW * 2.0**-53 if meth in closed else refdata.BW_ROOT_REL
            key = (
                "closed" if meth in closed else ("sj" if meth.startswith("sj") else ("isj" if meth == "isj" else "cv"))
            )
            e = pts.setdefault(key, {"where": [], "contract": [], "rel": []})
            e["where"].append(f"{sid} {meth}")
            e["contract"].append(rel / bound)
            e["rel"].append(rel)
    return {
        key: region(p_["where"], rel=np.array(p_["rel"]), contract=np.array(p_["contract"]), bound=1.0)
        for key, p_ in sorted(pts.items())
    } or None


def sweeps():
    return [
        Custom("smooth", "local", lambda: _local_regions("sd"), compare_libs(_local_regions, ("statsmodels", "R"))),
        Custom("smooth", "loess", lambda: _loess_regions("sd"), compare_libs(_loess_regions, ("R",))),
        Custom("smooth", "savgol", lambda: _savgol_regions("sd"), compare_libs(_savgol_regions, ("scipy",))),
        Custom("smooth", "spline", lambda: _spline_regions("sd"), compare_libs(_spline_regions, ("scipy",))),
        Custom("smooth", "fitpack", lambda: _fitpack_regions("sd"), compare_libs(_fitpack_regions, ("scipy",))),
        Custom("smooth", "kreg", lambda: _kreg_regions("sd"), compare_libs(_kreg_regions, ("statsmodels",))),
        Custom(
            "smooth",
            "kde",
            lambda: _kde_regions("sd"),
            compare_libs(_kde_regions, ("scipy", "sklearn", "statsmodels", "R")),
        ),
        Custom(
            "smooth",
            "bandwidth",
            lambda: _bandwidth_regions("sd"),
            compare_libs(_bandwidth_regions, ("R", "statsmodels")),
        ),
    ]
