"""kde, KDEResult and DensityResult (SM5 milestone; docs/design/smooth.md §2.5, §3.5, §4.5; decisions SM18-SM22,
SM26, SM31, SM115-SM126): the reference contract on kde.npz (pdf and logpdf relative, the far tails included, the
full covariance with its condition, the diagonal one, cdf, the compact kernels, the reflection), the binned density
against its method bound, scipy's gaussian_kde (the same covariance, rules and values), scikit-learn's KernelDensity
and statsmodels' half widths, R's density and kde2d, the tree path's bits, the thread counts, sample, marginal and
the inputs and errors."""

import functools
import math

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata
from . import samples as S
from .conftest import LIVE

A = sd.asarray
KDE = sd.smooth.kde
REF = refdata.ref("kde")
SC = refdata.data("scipy")
SK = refdata.data("sklearn")
SM = refdata.data("statsmodels")
R = refdata.data("r")
U = 2.0**-53
HALF = {"epanechnikov": 5.0, "rectangular": 3.0, "triangular": 6.0, "biweight": 7.0}


def _np(a):
    return np.asarray(a.tolist() if hasattr(a, "tolist") else a, dtype=float)


def _d0(xe, x, Hinv):
    d = np.atleast_2d(xe - x) if x.ndim == 2 else (xe - x)[:, None]
    return float(np.min(np.einsum("ij,jk,ik->i", d, np.atleast_2d(Hinv), d)))


# ---- the reference contract (§4.5) ---------------------------------------------------------------------------


@functools.cache
def _units():
    """{(method, set, index): units of its contract}."""
    out = {}
    for g, sid in enumerate(REF.group_inputs["set"].tolist()):
        s = S.density(sid)
        n, p = s.n, s.p
        H = np.asarray(REF.group_inputs["H"][g], dtype=float)
        w = None if s.w is None else A(s.w)
        if p == 1:
            h = float(REF.group_inputs["bw"][g])
            fits = [("", KDE(A(s.x), bw=h, weights=w), np.array([[h * h]]), 1.0)]
        else:
            bwd = np.asarray(REF.group_inputs["bw_diag"][g], dtype=float)
            fits = [
                ("", KDE(A(s.x), bw=A(H), weights=w), H, math.sqrt(np.linalg.cond(H))),
                ("_diag", KDE(A(s.x), bw=A(bwd), weights=w), np.diag(bwd**2), 1.0),
            ]
        for tag, res, Hm, kap in fits:
            Hinv = np.linalg.inv(Hm)
            pdf, lp = _np(res.pdf(A(s.xe))), _np(res.logpdf(A(s.xe)))
            for meth, got in (("pdf", pdf), ("logpdf", lp)):
                m = REF.methods[meth + tag]
                pos, idx = refdata.at(REF, meth + tag, sid, 1)
                for q, i in zip(pos, idx):
                    ref = float(m.hi[q])
                    if meth == "pdf":
                        b = refdata.kde_pdf_bound(n, p, _d0(s.xe[i], s.x, Hinv), ref, kap)
                    else:
                        b = refdata.kde_logpdf_bound(n, p, ref, kap)
                    out[(meth + tag, sid, int(i))] = refdata.units(float(got[i]), ref, float(m.lo[q]), b)
        if p != 1:
            continue
        res = fits[0][1]
        cdf = _np(res.cdf(A(s.xe)))
        m = REF.methods["cdf"]
        for q, i in zip(*refdata.at(REF, "cdf", sid, 1)):
            out[("cdf", sid, int(i))] = refdata.units(float(cdf[i]), m.hi[q], m.lo[q], refdata.kde_cdf_bound(n))
        wn = np.full(n, 1.0 / n) if s.w is None else s.w / s.w.sum()
        for kern in ("epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
            got = _np(KDE(A(s.x), bw=h, kernel=kern, weights=w).pdf(A(s.xe)))
            m = REF.methods[f"pdf_{kern}"]
            a = math.sqrt(HALF[kern]) if kern != "cosine" else 1 / math.sqrt(1 / 3 - 2 / math.pi**2)
            for q, i in zip(*refdata.at(REF, f"pdf_{kern}", sid, 1)):
                z = (s.xe[i] - s.x) / h
                r = np.abs(z) / a
                inside = r < 1 + 1e-12
                dk = {  # |K'(z)| of the kernel of standard deviation 1
                    "epanechnikov": 1.5 * r / a**2,
                    "rectangular": np.where(np.abs(r - 1) < 1e-12, 1e12, 0.0),
                    "triangular": np.full(n, 1 / a**2),
                    "biweight": 3.75 * r * np.abs(1 - r * r) / a**2,
                    "cosine": np.pi * np.abs(np.sin(np.pi * np.minimum(r, 1))) / (2 * a * a),
                }[kern]
                slope = float(np.sum(np.where(inside, wn * np.abs(z) * dk, 0.0))) / h
                b = refdata.kde_kernel_bound(n, float(m.hi[q]), slope)
                out[(f"pdf_{kern}", sid, int(i))] = refdata.units(float(got[i]), m.hi[q], m.lo[q], b)
        if s.bounds is not None:
            got = _np(KDE(A(s.x), bw=h, bounds=s.bounds, weights=w).pdf(A(s.xe)))
            m = REF.methods["pdf_reflect"]
            for q, i in zip(*refdata.at(REF, "pdf_reflect", sid, 1)):
                t = float(s.xe[i])
                pts = [t] + [2 * b - t for b in s.bounds if math.isfinite(b)]
                parts = _np(res.pdf(A(pts)))
                b = sum(refdata.kde_rel(n, 1, _d0(np.array([v]), s.x, 1 / h**2)) * f for v, f in zip(pts, parts))
                out[("pdf_reflect", sid, int(i))] = refdata.units(float(got[i]), m.hi[q], m.lo[q], b + U * abs(m.hi[q]))
    return out


@pytest.mark.parametrize(
    "method",
    ["pdf", "logpdf", "pdf_diag", "logpdf_diag", "cdf", "pdf_epanechnikov", "pdf_rectangular", "pdf_triangular",
     "pdf_biweight", "pdf_cosine", "pdf_reflect"],
)  # fmt: skip
def test_reference_contract(method):
    got = {k: v for k, v in _units().items() if k[0] == method}
    assert got and len(got) == REF.methods[method].hi.size
    worst = max(got.items(), key=lambda kv: kv[1])
    assert worst[1] <= 1.0, worst


def test_far_tails_are_relative():
    # the far points of every set (30 ranges out): the pdf underflows to 0 or a subnormal, logpdf stays relative
    for g, sid in enumerate(REF.group_inputs["set"].tolist()):
        s = S.density(sid)
        if s.p != 1:
            continue
        res = KDE(A(s.x), bw=float(REF.group_inputs["bw"][g]), weights=None if s.w is None else A(s.w))
        lp = _np(res.logpdf(A(s.xe[[0, -1]])))
        assert np.all(lp < -1000) and np.all(np.isfinite(lp))
        assert np.all(_np(res.pdf(A(s.xe[[0, -1]]))) == 0.0)


# ---- the binned density (§4.5) --------------------------------------------------------------------------------


@pytest.mark.parametrize("sid", S.density_ids(tag="1d", exclude=("degenerate",)))
def test_binned_bound(sid):
    s = S.density(sid)
    g = refdata.group_of(REF, sid)
    h0 = float(REF.group_inputs["bw"][g])
    for scale in (1.0, 0.25):
        h = h0 * scale
        res = KDE(A(s.x), bw=h, weights=None if s.w is None else A(s.w))
        for n in (512, 100, 2000):
            d = res.density(n=n)
            x = _np(d.x)
            assert x.size == n
            exact = _np(res.pdf(d.x))
            step = (x[-1] - x[0]) / (max(n, 512) - 1)
            bound = refdata.binned_bound(step, h)
            if n < 512:  # interpolated from 512 points: the interpolation's delta^2 / 8 max|f''| too
                bound += step * step / (8 * math.sqrt(2 * math.pi) * h**3)
            assert np.max(np.abs(_np(d.density) - exact)) <= bound, (sid, scale, n)
            assert np.all(_np(d.density) >= 0)


@pytest.mark.parametrize("sid", ["normal2/60", "normal3/50"])
def test_binned_bound_nd(sid):
    s = S.density(sid)
    res = KDE(A(s.x))
    H = _np(res.covariance)
    k0 = (2 * math.pi) ** (-s.p / 2) / math.sqrt(np.linalg.det(H))
    for n in (24, 48):
        d = res.density(n=n)
        exact = res.density(n=n, binned=False)
        steps = [(float(a[-1]) - float(a[0])) / (n - 1) for a in d.x]
        bound = refdata.binned_bound_nd(steps, np.diag(np.linalg.inv(H)), k0)
        assert d.density.shape == (n,) * s.p
        assert np.max(np.abs(_np(d.density) - _np(exact.density))) <= bound


def test_binned_kernels_and_reflection():
    s = S.density("normal/200")
    for kern in ("epanechnikov", "triangular", "biweight", "cosine"):
        res = KDE(A(s.x), bw=0.33, kernel=kern)
        d = res.density()
        assert np.max(np.abs(_np(d.density) - _np(res.pdf(d.x)))) < 1e-3
    e = S.density("exponential/60")
    res = KDE(A(e.x), bw=0.3, bounds=(0.0, math.inf))
    d = res.density()
    assert float(d.x[0]) == 0.0  # the grid is clipped to the bounds
    assert np.max(np.abs(_np(d.density) - _np(res.pdf(d.x)))) <= refdata.binned_bound(float(d.x[1] - d.x[0]), 0.3)


# ---- scipy's gaussian_kde -------------------------------------------------------------------------------------


@pytest.mark.parametrize("rule", ["scott", "silverman"])
@pytest.mark.parametrize("did", [d for d in S.density_ids(exclude=("degenerate",))])
def test_scipy_rules_and_values(did, rule):
    s = S.density(did)
    rec = SC[f"kde/{did}/{rule}"]
    assert rec["digest"] == s.digest
    res = KDE(A(s.x), bw=rule, weights=None if s.w is None else A(s.w))
    v = rec["value"]
    assert math.isclose(res.factor, v["factor"], rel_tol=4 * U)
    assert math.isclose(res.neff, v["neff"], rel_tol=4 * U)
    assert np.allclose(_np(res.covariance), np.asarray(v["covariance"]["array"]), rtol=16 * U, atol=0)
    got = _np(res.pdf(A(s.xe)))
    want = np.asarray(v["pdf"]["array"])
    big = want > 1e-200
    assert np.allclose(got[big], want[big], rtol=1e-11, atol=0)
    # logpdf in the far tail: scipy's own logsumexp agrees to its rounding
    assert np.allclose(_np(res.logpdf(A(s.xe))), np.asarray(v["logpdf"]["array"]), rtol=1e-10, atol=1e-12)


def test_scipy_differences():
    d = S.density("normal/200")
    # [scipy와 다름] a number is the kernel's standard deviation, scipy's is a factor of the sample's
    rec = SC["rule/gaussian_kde_scalar_is_a_factor"]
    assert math.isclose(KDE(A(d.x), bw=0.5 * rec["sample_sd"]).bw, rec["kernel_sd"], rel_tol=1e-15)
    # [scipy와 다름] the points are rows (scipy reads (60, 2) as 60 variables of 2 points)
    assert SC["rule/gaussian_kde_points_are_columns"]["error"] == "ValueError"
    assert KDE(A(S.density("normal2/60").x)).ndim == 2
    # the same degenerate data: LinAlgError
    with pytest.raises(sd.linalg.LinAlgError):
        KDE(A(S.density("allsame/10").x))
    with pytest.raises(sd.linalg.LinAlgError):
        KDE(A(S.density("line2/20").x))
    # the far tail: scipy's pdf 0 and logpdf finite; so are sd's
    tail = SC["rule/gaussian_kde_far_tail"]["value"]
    res = KDE(A(d.x))
    assert res.pdf(200.0) == 0.0
    assert math.isclose(res.logpdf(200.0), tail["logpdf"]["array"][0], rel_tol=1e-13)


@LIVE
def test_scipy_live_full_covariance():
    from scipy.stats import gaussian_kde

    g = np.random.default_rng(5)
    X = g.standard_normal((300, 3)) @ np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.2]])
    q = g.standard_normal((40, 3))
    k = gaussian_kde(X.T)
    res = KDE(A(X))
    assert np.allclose(_np(res.covariance), k.covariance, rtol=1e-14, atol=1e-15)
    assert np.allclose(np.log(_np(res.pdf(A(q)))), np.log(k(q.T)), rtol=1e-12, atol=1e-11)  # scipy: D0 u in the tail
    one = KDE(A(X[:, 0]))
    assert math.isclose(one.cdf(0.3) - one.cdf(-0.4), gaussian_kde(X[:, 0]).integrate_box_1d(-0.4, 0.3), rel_tol=1e-12)


# ---- scikit-learn and statsmodels: the kernels' scale (SM34) -----------------------------------------------------


@pytest.mark.parametrize("did", S.density_ids(tag="1d", exclude=("degenerate",)))
def test_sklearn_kernels(did):
    s = S.density(did)
    w = None if s.w is None else A(s.w)
    for name, kern in (("gaussian", "gaussian"), ("epanechnikov/sd", "epanechnikov"), ("tophat/sd", "rectangular"),
                       ("linear/sd", "triangular")):  # fmt: skip
        rec = SK[f"kde/{did}/{name}"]
        assert rec["digest"] == s.digest
        b = rec["bandwidth"]
        half = {"gaussian": 1.0, "epanechnikov": 5.0, "rectangular": 3.0, "triangular": 6.0}[kern]
        res = KDE(A(s.x), bw=b / math.sqrt(half) if kern != "gaussian" else b, kernel=kern, weights=w)
        got = _np(res.logpdf(A(s.xe)))
        want = np.asarray(rec["value"]["array"], dtype=float)
        fin = np.isfinite(want) & np.isfinite(got)
        # points on a support's edge by rounding may differ; elsewhere sklearn's exact sums agree
        close = np.isclose(got[fin], want[fin], rtol=1e-12, atol=1e-12)
        assert close.mean() >= 0.9, (did, name)
        assert np.array_equal(np.isinf(got), np.isinf(want)) or kern != "gaussian"


def test_half_width_rule():
    for lib, rec in (("sklearn", SK["rule/kernel_bandwidth_is_the_half_width"]),
                     ("statsmodels", SM["rule/kernel_bandwidth_is_the_half_width"])):  # fmt: skip
        epa = rec.get("epanechnikov", rec.get("epa"))["value"]
        assert math.isclose(epa, 0.75)  # their bandwidth 1: the kernel's half width is 1
        # sd: bw is the standard deviation, the peak is 3 / (4 sqrt 5)
        res = KDE(A([0.0]), bw=1.0, kernel="epanechnikov")
        assert math.isclose(res.pdf(0.0), 3 / (4 * math.sqrt(5)), rel_tol=1e-15), lib
    assert SK["rule/default_bandwidth"]["value"] == 1.0


def test_statsmodels_kernels_binned():
    s = S.density("normal/200")
    for code, kern, half in (
        ("epa/sd", "epanechnikov", 5.0),
        ("tri/sd", "triangular", 6.0),
        ("biw/sd", "biweight", 7.0),
    ):
        rec = SM[f"kde_kernel/normal/200/{code}"]["value"]
        x = np.asarray(rec["support"]["array"])
        res = KDE(A(s.x), bw=rec["bw"] / math.sqrt(half), kernel=kern)
        got = _np(res.pdf(A(x)))
        want = np.asarray(rec["density"]["array"]) if "density" in rec else None
        if want is None:
            continue
        assert np.max(np.abs(got - want)) < 0.05 * got.max()  # statsmodels' grid evaluation, not exact


# ---- R ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("did", S.density_ids(tag="1d", exclude=("degenerate",)))
def test_r_density(did):
    s = S.density(did)
    g = refdata.group_of(REF, did)
    h = float(REF.group_inputs["bw"][g])
    for kern in ("gaussian", "epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
        rec = R.get(f"density/{did}/{kern}")
        if rec is None:  # R's density has no record of the weighted sample
            continue
        assert rec["digest"] == s.digest
        v = rec["values"]
        lo, hi, y = v[0], v[1], np.asarray(v[2:])
        res = KDE(A(s.x), bw=h, kernel=kern, weights=None if s.w is None else A(s.w))
        d = res.density(n=512, lower=lo, upper=hi)
        exact = _np(res.pdf(d.x))
        step = (hi - lo) / 511
        if kern == "gaussian":
            # R's binned values and sd's are both within the method's bound of the exact density
            b = refdata.binned_bound(step, h)
            assert np.max(np.abs(y - exact)) <= 2 * b
            assert np.max(np.abs(_np(d.density) - exact)) <= b
        else:  # the compact kernels (no bound): sd's binning is no farther from the exact density than R's
            assert np.max(np.abs(_np(d.density) - exact)) <= 1.5 * np.max(np.abs(y - exact)) + 1e-12


def test_r_kde2d():
    # MASS::kde2d(x, y, h = c(bandwidth.nrd(x), bandwidth.nrd(y)), n): a product kernel of standard
    # deviations h / 4 evaluated exactly at the grid
    s = S.density("normal2/60")
    v = R["kde2d/normal2/60"]["values"]
    hx, hy = v[0], v[1]
    n = int(round(math.sqrt(len(v) - 1))) - 1  # 2 + 2 n + n^2 values
    gx, gy = np.asarray(v[2 : 2 + n]), np.asarray(v[2 + n : 2 + 2 * n])
    z = np.asarray(v[2 + 2 * n :]).reshape(n, n, order="F")
    res = KDE(A(s.x), bw=A([hx / 4, hy / 4]))
    d = res.density(n=n, lower=[gx[0], gy[0]], upper=[gx[-1], gy[-1]], binned=False)
    assert np.allclose(_np(d.x[0]), gx, rtol=1e-14) and np.allclose(_np(d.x[1]), gy, rtol=1e-14)
    assert np.allclose(_np(d.density), z, rtol=1e-12, atol=1e-300)


# ---- the tree path and the threads (§1.3, decisions SM115, SM116) -----------------------------------------------


@pytest.fixture
def tree_mode():
    old = sd._core.smooth.kde_tree_mode(0)
    yield
    sd._core.smooth.kde_tree_mode(old)


@pytest.mark.usefixtures("tree_mode")
def test_tree_and_scan_have_the_same_bits():
    g = np.random.default_rng(3)
    X = g.standard_normal((3000, 2)) @ np.array([[1.0, 0.8], [0.0, 0.3]])
    X3 = g.standard_normal((2000, 3))
    q = np.concatenate([g.standard_normal((200, 2)) * 3, X[:20], [[40.0, -40.0]]])
    q3 = g.standard_normal((150, 3)) * 2
    w = g.random(3000) + 0.1
    cases = [
        (KDE(A(X)), q),
        (KDE(A(X), bw=0.05), q),
        (KDE(A(X), bw=A([0.02, 0.4]), weights=A(w)), q),
        (KDE(A(X3), bw=0.1), q3),
        (KDE(A(X3 + 1e8), bw=0.1), q3 + 1e8),
    ]
    for res, qq in cases:
        out = []
        for mode in (1, 2):
            sd._core.smooth.kde_tree_mode(mode)
            out.append((res.pdf(A(qq)), res.logpdf(A(qq))))
        assert bits_equal(out[0], out[1])
    # kernel_regression's local constant and local line over several variables (the tree that SM4 left)
    y = np.sin(X[:800, 0]) + X[:800, 1]
    for degree in (0, 1):
        got = []
        for mode in (1, 2):
            sd._core.smooth.kde_tree_mode(mode)
            r = sd.smooth.kernel_regression(A(X[:800]), A(y), degree=degree, bw=A([0.1, 0.05]))
            got.append((r.fittedvalues, r.leverage, r.predict(A(q))))
        assert bits_equal(got[0], got[1]), degree


@pytest.mark.usefixtures("restore_threads")
def test_thread_counts():
    g = np.random.default_rng(9)
    x = g.standard_normal(20000)
    X = g.standard_normal((5000, 2))
    q = np.linspace(-5, 5, 3001)

    def results():
        a = KDE(A(x))
        b = KDE(A(X))
        c = KDE(A(np.abs(x)), bw="nrd0", bounds=(0.0, math.inf))
        return [a.pdf(A(q)), a.logpdf(A(q)), a.cdf(A(q)), a.density(n=4096), b.pdf(A(X[:3000] * 1.3)),
                b.density(n=64), c.pdf(A(q)), c.density(), sd.smooth.bw_cv_ls(A(x[:3000])),
                sd.smooth.bw_sj(A(x)), sd.smooth.bw_isj(A(x))]  # fmt: skip

    sd.set_num_threads(1)
    base = results()
    for t in (2, 3, 7, 10):
        sd.set_num_threads(t)
        for k, (a, b) in enumerate(zip(base, results())):
            assert bits_equal(a, b), (t, k)


def test_parallel_sort_of_many_points():
    """From 2^16 points the radix sort runs its passes on chunks in parallel (decision SM139): the order is the
    stable one of (coordinates, weight, index) whatever the thread count, with -0 equal to +0."""
    g = np.random.default_rng(11)
    n = 1 << 17
    X = np.round(g.standard_normal((n, 2)), 2)  # many equal keys
    X[:50, 0] = -0.0
    X[50:100, 0] = 0.0
    w = np.round(g.random(n), 1)
    order = np.lexsort((np.arange(n), w, X[:, 1], X[:, 0] + 0.0))
    order = order[w[order] > 0]
    outs = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        outs.append(sd._core.smooth.kde_prepare(A(X), A(w)))
    sd.set_num_threads(10)
    for o in outs[1:]:
        assert bits_equal(o[0], outs[0][0]) and bits_equal(o[1], outs[0][1]) and o[2] == outs[0][2]
    assert bits_equal(outs[0][0], A(X[order])) and bits_equal(outs[0][1], A(w[order]))


def test_input_order():
    s = S.density("weighted/40")
    perm = np.random.default_rng(1).permutation(s.n)
    a = KDE(A(s.x), weights=A(s.w))
    b = KDE(A(s.x[perm]), weights=A(s.w[perm]))
    assert bits_equal(a.pdf(A(s.xe)), b.pdf(A(s.xe))) and a.bw == b.bw
    t = S.density("normal3/50")
    p3 = np.random.default_rng(2).permutation(t.n)
    assert bits_equal(KDE(A(t.x)).pdf(A(t.xe)), KDE(A(t.x[p3])).pdf(A(t.xe)))


def test_scaling_by_powers_of_two():
    s = S.density("bimodal/80")
    a, b = KDE(A(s.x), bw=0.3), KDE(A(s.x * 2.0**-600), bw=0.3 * 2.0**-600)
    la, lb = _np(a.logpdf(A(s.xe))), _np(b.logpdf(A(s.xe * 2.0**-600)))
    assert np.allclose(la + 600 * math.log(2), lb, rtol=1e-14, atol=1e-12)
    pa, pb = _np(a.pdf(A(s.xe))), _np(b.pdf(A(s.xe * 2.0**-600)))
    normal = pa > 2.0**-1000
    assert np.array_equal(pa[normal] * 2.0**600, pb[normal])  # the same bits: only the power of two moves
    # a bandwidth near the bottom of the double range still gives the normalised value (no overflow of 1 / h)
    tiny = KDE(A([0.0]), bw=2.0**-1060)
    assert math.isclose(tiny.logpdf(0.0), 1060 * math.log(2) - 0.5 * math.log(2 * math.pi), rel_tol=1e-15)
    assert tiny.pdf(0.0) == math.inf


# ---- the distribution's methods ----------------------------------------------------------------------------------


def test_cdf_integral_and_bounds():
    s = S.density("exponential/60")
    for kern in ("gaussian", "epanechnikov", "cosine", "biweight", "triangular", "rectangular"):
        res = KDE(A(s.x), bw=0.4, kernel=kern)
        assert res.cdf(-math.inf) == 0.0 and math.isclose(res.cdf(math.inf), 1.0, rel_tol=4 * U)
        q = np.linspace(-1, 6, 2001)
        c = _np(res.cdf(A(q)))
        assert np.all(np.diff(c) >= -1e-15)
        dens = np.trapezoid(_np(res.pdf(A(q))), q)
        assert math.isclose(dens, c[-1] - c[0], rel_tol=2e-5 if kern != "rectangular" else 1e-3), kern
        b = KDE(A(s.x), bw=0.4, kernel=kern, bounds=(0.0, math.inf))
        assert b.cdf(-0.1) == 0.0 and b.pdf(-0.1) == 0.0
        assert math.isclose(b.cdf(1e9), 1.0, rel_tol=1e-14)
    with pytest.raises(ValueError, match="1-D"):
        KDE(A(S.density("normal2/60").x)).cdf(A([[0.0, 0.0]]))


def test_sample():
    s = S.density("bimodal/80")
    res = KDE(A(s.x))
    rng = sd.random.default_rng(7)
    v = res.sample(20000, rng=rng)
    assert v.shape == (20000,)
    assert abs(float(sd.mean(v)) - float(np.mean(s.x))) < 0.05
    assert abs(float(sd.var(v)) - (np.var(s.x) + res.bw**2)) < 0.15
    assert bits_equal(
        res.sample((3, 4), rng=sd.random.default_rng(1)), res.sample((3, 4), rng=sd.random.default_rng(1))
    )
    assert isinstance(res.sample(rng=sd.random.default_rng(1)), float)
    m = KDE(A(S.density("normal2/60").x))
    assert m.sample((5,), rng=rng).shape == (5, 2) and m.sample(rng=rng).shape == (2,)
    for kern in ("epanechnikov", "rectangular", "triangular", "biweight", "cosine"):
        k = KDE(A([0.0]), bw=1.0, kernel=kern)
        z = _np(k.sample(40000, rng=sd.random.default_rng(3)))
        assert abs(z.mean()) < 0.03 and abs(z.var() - 1.0) < 0.05, kern
        a = math.sqrt(HALF[kern]) if kern in HALF else 1 / math.sqrt(1 / 3 - 2 / math.pi**2)
        assert np.all(np.abs(z) <= a + 1e-12)
    b = KDE(A(np.abs(s.x)), bw=0.5, bounds=(0.0, 4.0))
    z = _np(b.sample(5000, rng=rng))
    assert z.min() >= 0.0 and z.max() <= 4.0
    with pytest.raises(TypeError, match="rng"):
        res.sample(3, rng=None)


def test_marginal():
    s = S.density("normal3/50")
    res = KDE(A(s.x))
    m = res.marginal([2, 0])
    H = _np(res.covariance)
    assert np.array_equal(_np(m.covariance), H[np.ix_([2, 0], [2, 0])])
    full = KDE(A(s.x[:, [2, 0]]), bw=A(H[np.ix_([2, 0], [2, 0])]))
    assert bits_equal(m.pdf(A(s.xe[:, [2, 0]])), full.pdf(A(s.xe[:, [2, 0]])))
    one = res.marginal(1)
    assert one.ndim == 1 and math.isclose(one.bw, math.sqrt(H[1, 1]), rel_tol=1e-15)
    # the integral over the other variables, numerically
    q = np.linspace(-30, 30, 20001)
    x0 = 0.3
    mesh = np.stack([np.full_like(q, x0), q], axis=1)
    two = res.marginal([1, 2])
    assert math.isclose(np.trapezoid(_np(two.pdf(A(mesh))), q), one.pdf(x0), rel_tol=1e-6)
    with pytest.raises(ValueError, match="repeat"):
        res.marginal([0, 0])
    with pytest.raises(ValueError, match="between"):
        res.marginal(3)


# ---- fields, inputs and errors -------------------------------------------------------------------------------


def test_fields_and_repr():
    s = S.density("weighted/40")
    res = KDE(A(s.x), weights=A(s.w), adjust=1.5)
    assert res.ndim == 1 and res.nobs == 40 and res.kernel == "gaussian" and res.bounds is None
    assert math.isclose(res.bw, 1.5 * sd.smooth.bw_scott(A(s.x), weights=A(s.w)), rel_tol=4 * U)
    assert math.isclose(float(sd.sum(res.weights)), 1.0, rel_tol=1e-15)
    assert res.covariance.shape == (1, 1) and math.isclose(float(res.covariance[0, 0]), res.bw**2, rel_tol=1e-15)
    assert repr(res).startswith("KDEResult(nobs=40, ndim=1, bw=")
    with pytest.raises(AttributeError):
        res.bw = 1.0
    with pytest.raises(ValueError):
        res.covariance[0, 0] = 1.0
    n = KDE(A(S.density("normal/50").x), bw="nrd0")
    assert math.isnan(n.factor) and n.bw == sd.smooth.bw_nrd0(A(S.density("normal/50").x))
    assert KDE(A([[0.0, 1.0], [1.0, 0.5], [2.0, 2.0]]), bw=A([0.5, 0.2])).bw.shape == (2,)


def test_nan_policy():
    x = np.array([0.0, 1.0, np.nan, 2.0])
    p = KDE(A(x), bw=0.5)
    assert math.isnan(p.pdf(0.0)) and math.isnan(p.bw)
    o = KDE(A(x), bw=0.5, nan_policy="omit")
    assert o.nobs == 3 and bits_equal(o.pdf(A([0.5])), KDE(A(x[[0, 1, 3]]), bw=0.5).pdf(A([0.5])))
    with pytest.raises(ValueError, match="NaN"):
        KDE(A(x), nan_policy="raise")
    assert math.isnan(KDE(A([0.0, 1.0]), bw=0.5).pdf(math.nan))


def test_errors():
    x = A([0.0, 1.0, 3.0])
    with pytest.raises(ValueError, match="'kernel' must be"):
        KDE(x, kernel="gau")
    with pytest.raises(ValueError, match="'bw' must be"):
        KDE(x, bw="normal_reference")
    with pytest.raises(ValueError, match="> 0"):
        KDE(x, bw=-1.0)
    with pytest.raises(ValueError, match="'adjust'"):
        KDE(x, adjust=0.0)
    with pytest.raises(ValueError, match="1-D"):
        KDE(A([[0.0, 1.0], [1.0, 2.0]]), kernel="epanechnikov")
    with pytest.raises(ValueError, match="1-D"):
        KDE(A([[0.0, 1.0], [1.0, 2.0]]), bw="nrd0")
    with pytest.raises(ValueError, match="within"):
        KDE(x, bounds=(0.5, 2.0))
    with pytest.raises(ValueError, match="takes no weights"):
        KDE(x, bw="sj", weights=A([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="symmetric"):
        KDE(A([[0.0, 1.0], [1.0, 2.0], [3.0, 1.0]]), bw=A([[1.0, 0.1], [0.2, 1.0]]))
    with pytest.raises(sd.linalg.LinAlgError):
        KDE(A([[0.0, 1.0], [1.0, 2.0], [3.0, 1.0]]), bw=A([[1.0, 2.0], [2.0, 1.0]]))
    with pytest.raises(ValueError, match="infinity"):
        KDE(A([0.0, math.inf]))
    with pytest.raises(ValueError, match="'weights' must be >= 0"):
        KDE(x, weights=A([1.0, -1.0, 1.0]))
    with pytest.raises(TypeError):
        KDE(x, 0.5)
    with pytest.raises(ValueError, match="at most 3"):
        KDE(A(np.random.default_rng(0).standard_normal((10, 4)))).density()


@pytest.mark.parametrize("binned", [True, False])
@pytest.mark.parametrize("p", [1, 2])
def test_density_arrays_are_read_only(binned, p):
    """DensityResult's arrays are read-only like every result's (API.md §2.2; decision SM145), NaN data too."""
    data = A(np.linspace(0, 1, 30)) if p == 1 else A(np.random.default_rng(4).random((30, 2)))
    for k in (KDE(data), KDE(A([math.nan, 1.0, 2.0]))) if p == 1 else (KDE(data),):
        d = k.density(binned=binned, n=16)
        for a in (d.x if p > 1 else (d.x,)) + (d.density,):
            with pytest.raises(ValueError, match="read-only"):
                a[(0,) * a.ndim] = 1.0
