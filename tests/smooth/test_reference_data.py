"""The reference and comparison data of sd.smooth (docs/design/smooth.md §4, §7): every file belongs to the test
data as they are (digests), the references pass checks of their own (exact polynomials reproduced, leverages in
[0, 1], traces and criteria consistent with the fitted values, kernels and Savitzky-Golay filters that reproduce
what they must), the recorded scipy 1.18.1, statsmodels 0.15.0, scikit-learn 1.9.1 and R 4.6.2 results agree with
the references where they compute the same estimator and show the behaviours the design departs from (§2.9), and
milestone SM0's prototype measurements stay within the contract constants of refdata.py. The recorded files are
compared, not the installed libraries, so the tests do not change with them."""

import math
from fractions import Fraction

import numpy as np
import pytest

from . import refdata
from . import samples as S

REFS = {f: refdata.ref(f) for f in refdata.FAMILIES}
SCIPY = refdata.data("scipy")
SM = refdata.data("statsmodels")
SK = refdata.data("sklearn")
R = refdata.data("r")
PROTO = refdata.data("prototype")


def value(rec):
    v = rec["value"]
    if isinstance(v, dict) and "array" in v:
        return np.array(v["array"], dtype=float)
    return v


def per_set(family, method, sid, kind=None):
    """(hi, lo, fields) of a set's points with that method, ordered by the points' index (NaN where dropped)."""
    r = REFS[family]
    pos, idx = refdata.at(r, method, sid, kind)
    m = r.methods[method]
    order = np.argsort(idx, kind="stable")
    pos, idx = pos[order], idx[order]
    return idx, m.hi[pos], m.lo[pos], {k: v[pos] for k, v in m.fields.items()}


def obj(family, sid):
    if family in ("local", "spline", "spline_fit"):
        return S.scatter(sid)
    if family == "local_nd":
        return S.multi(sid)
    if family == "kreg":
        return S.scatter(sid) if sid in S.scatter_ids() else S.multi(sid)
    if family in ("kde", "bandwidth"):
        return S.density(sid)
    if family == "savgol_filter":
        return S.signal(sid.split("|")[0])
    raise KeyError(family)


# ---- the files belong to the data -----------------------------------------------------------


def test_every_family_is_there_with_its_provenance():
    assert refdata.names() == sorted(refdata.FAMILIES)
    for f, r in REFS.items():
        assert r.meta["arb"].startswith("python-flint 0.9.0"), f
        assert r.meta["tool"] == "tools/smooth_ref.py"
        assert r.meta["target_bits"] >= 106
        assert r.meta["n_requested"] == r.points, f
    for name, rec in (("scipy", SCIPY), ("statsmodels", SM), ("sklearn", SK), ("r", R), ("prototype", PROTO)):
        assert rec["_provenance"]["tool"] == "tools/smooth_probe.py", name
    assert SCIPY["_provenance"]["scipy"] == "1.18.1"
    assert SM["_provenance"]["statsmodels"] == "0.15.0"
    assert SK["_provenance"]["scikit-learn"] == "1.9.1"
    assert R["_provenance"]["R"].startswith("R version 4.6.2")


def test_points_are_the_generators_points():
    for f, r in REFS.items():
        for sid, dig in set(zip(r.inputs["set"].tolist(), r.inputs["digest"].tolist())):
            if f == "savgol":
                case = next(c for c in S.SAVGOL if S.savgol_id(c) == sid)
                n, d, deriv, pos = case
                assert S.digest(np.array([n, d, deriv, -1 if pos is None else pos])) == dig
                continue
            assert obj(f, sid).digest == dig, (f, sid)
    for name, rec in (("scipy", SCIPY), ("statsmodels", SM), ("sklearn", SK), ("r", R)):
        for key, v in rec.items():
            if not isinstance(v, dict) or not v.get("digest"):
                continue
            sid = key.split("/", 1)[1]
            sid = next(
                (
                    i
                    for i in (*S.scatter_ids(), *S.multi_ids(), *S.density_ids(), *S.signal_ids())
                    if sid.startswith(i + "/") or sid.startswith(i + "|") or sid == i
                ),
                None,
            )
            if sid is None:
                continue
            o = (
                S.scatter(sid)
                if sid in S.scatter_ids()
                else S.multi(sid)
                if sid in S.multi_ids()
                else S.density(sid)
                if sid in S.density_ids()
                else S.signal(sid)
            )
            assert o.digest == v["digest"], (name, key)


def test_every_data_set_is_covered():
    covered = {f: set(REFS[f].inputs["set"].tolist()) for f in REFS}
    assert covered["local"] == set(S.scatter_ids())
    assert covered["local_nd"] == set(S.multi_ids())
    assert covered["kde"] == set(S.density_ids(exclude=("degenerate",)))
    assert covered["bandwidth"] == set(S.density_ids(tag="1d"))
    assert covered["savgol"] == {S.savgol_id(c) for c in S.SAVGOL}
    assert covered["spline"] == covered["spline_fit"]
    assert {"random/50", "clustered/40", "weights/30", "ties/30", "line/20"} <= covered["spline"]


def test_points_are_dropped_only_where_no_fit_exists():
    """A local fit has no value where no point of the window has a positive weight (one x and an evaluation point
    elsewhere, a single point, zero weights) or, in several variables, where the window's design loses rank (points
    on a line); nothing else is dropped."""
    allowed = {"local": {"allsame/8", "tiny/1", "tiny/2", "tiny/3"}, "local_nd": {"lowrank2/40"}}
    for f in ("local", "local_nd"):
        r = REFS[f]
        for name, m in r.methods.items():
            sets = set(r.inputs["set"][m.dropped].tolist())
            assert sets <= allowed[f], (f, name, sets - allowed[f])
    # bandwidth's minimizers (SM5 milestone): none where the criterion's smallest grid point is the interval's end
    # (the integers of discrete/60 make cv_ls fall toward h = 0)
    ends = {"cvls_min": {"discrete/60"}, "cvml_min": set()}
    for name, m in REFS["bandwidth"].methods.items():
        assert set(REFS["bandwidth"].inputs["set"][m.dropped].tolist()) <= ends.get(name, set()), name
    for f in REFS:
        if f not in ("local", "local_nd", "bandwidth"):
            for name, m in REFS[f].methods.items():
                assert m.dropped.size == 0, (f, name)


# ---- the references check themselves --------------------------------------------------------


@pytest.mark.parametrize(
    ("sid", "poly", "degrees"),
    [
        ("line/20", lambda x: 2 * x + 1, (1, 2)),
        ("quadratic/20", lambda x: x * x - 3 * x + 2, (2,)),
        ("constant/20", lambda x: 0 * x + Fraction(3, 4), (0, 1, 2)),
    ],
)
def test_local_fits_reproduce_their_polynomials(sid, poly, degrees):
    r = REFS["local"]
    s = S.scatter(sid)
    pts = {0: s.x, 1: s.xe}
    names = [
        n
        for n in r.methods
        if (n.startswith("lowess") and 1 in degrees) or (n.startswith("loess_d") and int(n[7]) in degrees)
    ]
    assert names
    for name in names:
        for kind in (0, 1):
            idx, hi, lo, _ = per_set("local", name, sid, kind)
            for i, h, l in zip(idx, hi, lo):
                exact = poly(Fraction(float(pts[kind][i])))
                assert Fraction(h) + Fraction(l) == exact or abs(Fraction(h) + Fraction(l) - exact) <= abs(
                    exact
                ) * Fraction(1, 2**104), (name, sid, kind, i)


def test_local_leverages_are_in_the_unit_interval():
    for f in ("local", "local_nd"):
        for name, m in REFS[f].methods.items():
            kind = REFS[f].inputs["kind"][m.index]
            lev = m.fields["leverage"][kind == 0]
            assert np.all((lev >= 0) & (lev <= 1 + 1e-15)), (f, name)
            assert np.all(np.isnan(m.fields["leverage"][kind != 0])), (f, name)
            assert np.all(m.fields["kappa"] >= 1.0 - 1e-12), (f, name)


def test_spline_traces_and_criteria_agree_with_the_fitted_values():
    rf = REFS["spline_fit"]
    for g, sid in enumerate(rf.group_inputs["set"].tolist()):
        s = S.scatter(sid)
        for name in ("ss_1e-8", "ss_1e-4", "ss_1e-1", "ss_1e2", "ps_1e-2", "ps_1", "ps_1e3", "lsq"):
            if name not in rf.methods or f"trl_{name}" not in rf.methods:
                continue
            idx, fit, _, fields = per_set("spline_fit", name, sid)
            if idx.size == 0:
                continue
            tm = rf.methods[f"trl_{name}"]
            gi = np.flatnonzero(tm.index == g)
            assert gi.size == 1
            trl = tm.hi[gi[0]]
            lev = fields["leverage"]
            assert np.all((lev >= -1e-15) & (lev <= 1 + 1e-15)), (sid, name)
            assert abs(lev.sum() - trl) <= 1e-13 * max(trl, 1.0), (sid, name)
            gm = rf.methods.get(f"gcv_{name}")
            if gm is not None and g in gm.index and "magnitude" not in s.tags:  # RSS of y near 1e300 overflows
                gv = gm.hi[np.flatnonzero(gm.index == g)[0]]
                rss = float(((s.y[idx] - fit) ** 2).sum())
                assert math.isclose(gv, s.n * rss / (s.n - trl) ** 2, rel_tol=1e-9, abs_tol=1e-25), (sid, name)


@pytest.mark.parametrize("name", ["ss_1e-8", "ss_1e2", "ps_1", "ps_1e3", "lsq"])
def test_splines_reproduce_a_line(name):
    """The smoothing spline's penalty vanishes on a line and the least-squares spline contains it: both return the
    line exactly. The P-spline's second differences vanish on a line only on exactly uniform knots; its knots are
    doubles xb + j h (h rounded), so it returns the line to rounding."""
    s = S.scatter("line/20")
    idx, hi, lo, _ = per_set("spline_fit", name, "line/20")
    assert idx.size == s.n
    for i, h, l in zip(idx, hi, lo):
        exact = 2 * Fraction(float(s.x[i])) + 1
        if name.startswith("ps"):
            assert abs(Fraction(h) + Fraction(l) - exact) <= Fraction(1, 2**50), (name, i)
        else:
            assert Fraction(h) + Fraction(l) == exact, (name, i)


def test_kernel_regression_reproduces_a_line_and_survives_underflow():
    s = S.scatter("line/20")
    for name in ("ll_a", "ll_b"):
        idx, hi, lo, _ = per_set("kreg", name, "line/20", kind=1)
        for i, h in zip(idx, hi):
            assert math.isclose(h, 2 * float(s.xe[i]) + 1, rel_tol=2e-16, abs_tol=1e-300), (name, i)
    r = REFS["kreg"]
    for name in ("nw_a", "nw_b", "ll_a", "ll_b"):
        m = r.methods[name]
        far = r.inputs["kind"][m.index] == 2
        assert far.sum() == len(r.group_inputs["set"]) and np.all(np.isfinite(m.hi[far])), name


def test_density_references_are_consistent():
    r = REFS["kde"]
    for did in r.group_inputs["set"].tolist():
        s = S.density(did)
        idx, pdf, _, _ = per_set("kde", "pdf", did)
        _, logpdf, _, _ = per_set("kde", "logpdf", did)
        ok = pdf > 1e-300
        assert np.allclose(np.log(pdf[ok]), logpdf[ok], rtol=1e-14, atol=0), did
        assert np.all(np.isfinite(logpdf)), did  # the far tail: the density underflows, its logarithm does not
        if s.p == 1:
            _, cdf, _, _ = per_set("kde", "cdf", did)
            order = np.argsort(s.xe[idx], kind="stable")
            assert np.all(np.diff(cdf[order]) >= 0) and cdf.min() >= 0 and cdf.max() <= 1, did
            if s.bounds is not None:
                _, refl, _, _ = per_set("kde", "pdf_reflect", did)
                lo, hi = s.bounds
                outside = (s.xe[idx] < lo) | (s.xe[idx] > hi)
                assert np.all(refl[outside] == 0) and np.all(refl[~outside] >= pdf[~outside]), did


def test_savgol_references_are_filters():
    r = REFS["savgol"]
    for case in S.SAVGOL:
        n, d, deriv, pos = case
        idx, hi, lo, _ = per_set("savgol", "coeffs", S.savgol_id(case))
        assert idx.size == n
        h = [Fraction(a) + Fraction(b) for a, b in zip(hi, lo)]
        p = Fraction(2 * (n // 2) - 1, 2) if (pos is None and n % 2 == 0) else Fraction(n // 2 if pos is None else pos)
        t = [Fraction(i) - p for i in range(n)]
        if deriv > d:
            assert all(c == 0 for c in h), case
            continue
        # the filter differentiates every polynomial of degree <= d exactly: sum h_i t_i^j = j! [j == deriv]
        for j in range(min(d, 6) + 1):
            want = math.factorial(deriv) if j == deriv else 0
            got = sum(c * ti**j for c, ti in zip(h, t))
            assert abs(got - want) <= Fraction(1, 2**95) * sum(abs(c) * abs(ti) ** j for c, ti in zip(h, t)), (case, j)
    assert r.points == sum(c[0] for c in S.SAVGOL)


def test_savgol_filter_references_reproduce_a_cubic():
    x = S.signal("cubic/40").x
    k = np.arange(40.0)
    for sid in REFS["savgol_filter"].group_inputs["set"].tolist():
        if not sid.startswith("cubic/40|"):
            continue
        n, d, deriv = map(int, sid.split("|")[1].split(","))
        if d < 3 or n % 2 == 0:  # an even window's output is the fit half a step off its sample
            continue
        want = {0: x, 1: 0.375 * k * k - 4 * k + 3, 2: 0.75 * k - 4}[deriv]
        half = n // 2
        for mode in S.SAVGOL_MODES:
            idx, hi, _, _ = per_set("savgol_filter", mode, sid)
            inner = (idx >= half) & (idx < 40 - half) if mode != "interp" else np.ones(idx.size, bool)
            assert np.array_equal(hi[inner], want[idx[inner]]), (sid, mode)


# ---- the recorded libraries against the references ------------------------------------------


def test_statsmodels_lowess_is_the_reference_where_x_has_a_unit_scale():
    """statsmodels' lowess (it = 0, delta = 0) computes the same estimator; it departs only through its absolute
    1e-12 guard on the window's weighted spread, which the narrow and subnormal sets reach (rule below)."""
    seen = 0
    for sid in S.scatter_ids(tag="plain"):
        for tag, frac in (("0.3", "lowess_f0.3"), ("0.67", "lowess_f0.67"), ("1", "lowess_f1")):
            rec = SM[f"lowess/{sid}/f{tag}/it0/delta0"]
            got = value(rec)
            idx, hi, _, fields = per_set("local", frac, sid, kind=0)
            err = np.abs(got[idx] - hi) / (fields["kappa"] * fields["scale"] * refdata.U)
            assert err.max() <= 256, (sid, tag, err.max())
            seen += 1
    assert seen >= 12


def test_r_lowess_differs_from_the_reference_by_its_weight_cut():
    """R cuts the tricube weights at 0.001 h and 0.999 h (docs/design/smooth.md §2.2): relative differences of
    order 1e-9, never more than 1e-7."""
    worst = 0.0
    for sid in S.scatter_ids(tag="plain"):
        for tag in ("0.3", "0.67", "1"):
            got = np.array(R[f"lowess/{sid}/f{tag}/it0/delta0"]["values"])
            idx, hi, _, fields = per_set("local", f"lowess_f{tag}", sid, kind=0)
            worst = max(worst, float((np.abs(got[idx] - hi) / fields["scale"]).max()))
    assert 1e-12 < worst < 1e-7


def test_r_loess_direct_is_the_reference():
    worst = 0.0
    for sid in S.scatter_ids(tag="plain"):
        for name in ("d0_s0.75", "d1_s0.75", "d2_s0.75", "d2_s0.5", "d1_s1.5"):
            rec = R.get(f"loess/{sid}/{name}/gaussian")
            if rec is None:
                continue
            vals = np.array(rec["values"])
            s = S.scatter(sid)
            idx, hi, _, fields = per_set("local", f"loess_{name}", sid, kind=0)
            fitted = vals[5 : 5 + s.n]
            worst = max(
                worst, float((np.abs(fitted[idx] - hi) / (fields["kappa"] * fields["scale"] * refdata.U)).max())
            )
    assert worst <= 64  # R's exact direct fit solves the same local least squares (its own rounding)


def test_scipy_smoothing_spline_coefficients_are_the_reference():
    rc = REFS["spline"]
    seen = 0
    for g, sid in enumerate(rc.group_inputs["set"].tolist()):
        for c in ("1e-8", "1e-4"):
            rec = SCIPY.get(f"ss/{sid}/{c}")
            if rec is None or "error" in rec:
                continue
            idx, hi, _, _ = per_set("spline", f"ss_{c}", sid)
            got = value(rec)
            if got.size != idx.size or S.scatter(sid).w is not None:
                continue
            tm = REFS["spline_fit"].methods[f"trl_ss_{c}"]
            kappa = float(tm.fields["kappa"][np.flatnonzero(tm.index == g)[0]])
            if kappa * refdata.U > 1e-6 or "scale" in S.scatter(sid).tags:  # scipy's own rounding grows there
                continue
            assert np.abs(got - hi[idx]).max() <= 1e6 * refdata.U * kappa * np.abs(hi).max(), (sid, c)
            seen += 1
    assert seen >= 10


def test_sklearn_and_scipy_gaussian_densities_are_the_reference():
    r = REFS["kde"]
    for did in S.density_ids(tag="1d", exclude=("degenerate",)):
        idx, logpdf, _, _ = per_set("kde", "logpdf", did)
        got = value(SK[f"kde/{did}/gaussian"])
        assert np.allclose(got[idx], logpdf, rtol=1e-12, atol=1e-12), did
    for did in S.density_ids(tag="nd"):
        if did not in set(r.inputs["set"].tolist()):
            continue
        rec = value(SCIPY[f"kde/{did}/scott"])
        idx, pdf, _, _ = per_set("kde", "pdf", did)
        got = np.array(rec["pdf"]["array"])
        ok = pdf > 1e-250
        assert np.allclose(got[idx][ok], pdf[ok], rtol=1e-10, atol=0), did


def test_sklearn_kernels_at_the_half_width_are_the_references():
    """sklearn's bandwidth of the compact kernels is the support's half-width (rule below): at h times the half-width
    of the unit-variance kernel they are the references' kernels of standard deviation h (R's scale)."""
    for did in S.density_ids(tag="1d", exclude=("degenerate",)):
        for sk, ref in (
            ("epanechnikov", "pdf_epanechnikov"),
            ("tophat", "pdf_rectangular"),
            ("linear", "pdf_triangular"),
        ):
            idx, pdf, _, _ = per_set("kde", ref, did)
            got = np.exp(value(SK[f"kde/{did}/{sk}/sd"]))
            near = np.abs(got[idx] - pdf) <= 1e-12 * np.maximum(pdf, 1e-3)
            # a point within rounding of the support's edge can fall either side
            assert near.mean() >= 0.95, (did, sk)


def test_bandwidth_rules_are_the_reference():
    for did in S.density_ids(tag="1d", exclude=("degenerate",)):
        nrd0 = per_set("bandwidth", "nrd0", did)[1][0]
        nrd = per_set("bandwidth", "nrd", did)[1][0]
        rb = R[f"bw/{did}"]["values"]
        assert math.isclose(rb[0], nrd0, rel_tol=1e-15) and math.isclose(rb[1], nrd, rel_tol=1e-15), did
        if S.density(did).w is None:
            sc = value(SCIPY[f"kde/{did}/scott"])["factor"]
            scott = per_set("bandwidth", "scott", did)[1][0]
            assert math.isclose(sc * float(np.std(S.density(did).x, ddof=1)), scott, rel_tol=1e-14), did


def test_scipy_savgol_filter_is_the_reference_at_low_orders():
    """scipy's filters at low orders (its coefficients within a few hundred u there) are the references' in every
    mode, even windows included: the references read the samples as scipy.ndimage does."""
    for sid in REFS["savgol_filter"].group_inputs["set"].tolist():
        for mode in S.SAVGOL_MODES:
            got = value(SCIPY[f"savgol_filter/{sid}|{mode}"])
            idx, hi, _, fields = per_set("savgol_filter", mode, sid)
            assert np.all(np.abs(got[idx] - hi) <= 1e4 * refdata.U * fields["scale"]), (sid, mode)


# ---- the departures the design documents (§2.9) ------------------------------------------------


def test_recorded_departures():
    # scipy: GCV's choice depends on the unit of x and fails on 10^4 random x
    assert min(SCIPY["rule/gcv_depends_on_the_unit_of_x"].values()) > 0.03
    assert SCIPY["rule/gcv_fails_on_10^4_random_x"]["error"] == "ValueError"
    assert "error" in SCIPY["rule/smoothing_spline_repeated_x"]
    assert "error" in SCIPY["rule/smoothing_spline_lam_inf"]
    assert "error" in SCIPY["rule/smoothing_spline_4_points"]
    w = SCIPY["rule/smoothing_spline_weights_are_w"]
    assert w["w2_vs_half_lam"] < 1e-12 < w["w2_vs_quarter_lam"]
    lsq = SCIPY["rule/lsq_spline_weights_are_w_squared"]
    assert lsq["vs_sum_w2_r2"] < 1e-10 < lsq["vs_sum_w_r2"]
    sp = SCIPY["rule/splrep_weights_are_w_squared"]
    assert abs(sp["s_reached_sum_w2_r2"] - sp["s"]) <= 0.001 * sp["s"] + 1e-12
    assert any(
        v["knots_splrep"] != v["knots_make_splrep"] for v in SCIPY["rule/splrep_and_make_splrep_differ"].values()
    )
    assert SCIPY["rule/make_splrep_s0_is_interpolation"] < 1e-12
    k = SCIPY["rule/gaussian_kde_scalar_is_a_factor"]
    assert math.isclose(k["kernel_sd"], 0.5 * k["sample_sd"], rel_tol=1e-14)
    assert "column" in SCIPY["rule/gaussian_kde_points_are_columns"]["msg"]  # (60, 2) read as 2 points in 60 dimensions
    assert "error" in SCIPY["rule/gaussian_kde_degenerate"]
    assert value(SCIPY["rule/savgol_complex_input"]) == "float64"
    assert any("ComplexWarning" in w for w in SCIPY["rule/savgol_complex_input"].get("warnings", []))
    assert "error" in SCIPY["rule/savgol_nan/interp"]
    assert "value" in SCIPY["rule/savgol_nan/mirror"]
    # statsmodels: a sorted table, NaN dropped, the absolute guard that makes lowess depend on the unit of x
    assert value(SM["rule/lowess_returns_a_sorted_table"]) == [200, 2]
    assert value(SM["rule/lowess_drops_nan"]) == [199, 2]
    assert SM["rule/lowess_depends_on_the_unit_of_x"]["1e-08"] > 0.1
    # the compact kernels' bandwidth is the support's half-width in sklearn and statsmodels, the standard deviation
    # in R and sd: the density at the centre of one point at bandwidth 1 is 3/4, not 3 / (4 sqrt 5)
    assert SK["rule/kernel_bandwidth_is_the_half_width"]["epanechnikov"]["value"] == pytest.approx(0.75, rel=1e-15)
    assert SM["rule/kernel_bandwidth_is_the_half_width"]["epa"]["value"] == pytest.approx(0.75, rel=1e-15)
    assert SK["rule/default_bandwidth"]["value"] == 1.0


def test_scipy_savgol_coefficients_go_wrong_at_high_orders():
    """scipy's Vandermonde least squares returns, without an error or a warning, coefficients far from the exact
    ones at high orders (docs/design/smooth.md §2.6): measured against the references."""
    for sid, bad in (("101,10,0,c", True), ("301,30,0,c", True), ("51,45,0,c", True), ("101,3,0,c", False)):
        idx, hi, _, _ = per_set("savgol", "coeffs", sid)
        got = value(SCIPY[f"savgol/{sid}"])
        err = np.abs(got[idx] - hi).max() / np.abs(hi).max()
        assert (err > 1e-3) if bad else (err < 1e-12), sid
        assert "warnings" not in SCIPY[f"savgol/{sid}"]


# ---- milestone SM0's prototypes within the contract constants ----------------------------------


def test_prototypes_hold_the_contract_constants():
    """The float64 prototypes of the planned algorithms (tools/smooth_probe.py proto) against the references: the
    constants of refdata.py are those measurements with a margin (docs/design/smooth.md §4, decisions SM27-SM33)."""
    assert PROTO["lowess/worst"]["closed"] <= refdata.C_LOCAL / 2
    assert PROTO["lowess/worst"]["recentred"] <= refdata.C_LOCAL / 2
    assert PROTO["lowess/worst"]["prefix"] > 1e6  # the prefix-sum path is rejected (§3.1)
    assert PROTO["loess/worst"]["qr"] <= refdata.C_LOCAL / 8
    assert math.isinf(PROTO["loess/worst"]["cholesky"])  # the plain Cholesky fails on the clustered windows
    assert PROTO["loess_conditioning/largest_good_ratio"] >= 2.0**32  # the refined solve's threshold (SM28)
    assert PROTO["loess_conditioning/smallest_bad_ratio"] > 2.0**32
    good = PROTO["spline/worst_well_conditioned"]
    assert PROTO["spline/worst"]["qr_coef"] <= refdata.C_SPLINE / 4
    assert good["leverage"] <= refdata.C_LEVERAGE
    assert PROTO["spline/worst"]["leverage_kappa2"] <= 1.0
    assert PROTO["kde_cut/worst_over_contract"] <= 1.0
    assert PROTO["kde_binned/worst_ratio"] <= 1.0
    for case in S.SAVGOL:
        rec = PROTO[f"savgol/{S.savgol_id(case)}"]
        assert rec["bits106"] <= refdata.C_SAVGOL / 8, case
