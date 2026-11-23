"""The bandwidth rules and selectors (SM5 milestone; docs/design/smooth.md §3.6, §4.5; decisions SM19, SM127-SM132):
the references on bandwidth.npz (the closed forms to C_BW u, the roots and minimizers to BW_ROOT_REL), R's bw.nrd0,
bw.nrd, bw.SJ and bw.ucv (the same equations; R's uniroot and optimize stop at 0.1 lower), statsmodels' cv_ls and
cv_ml (the same criteria; its fmin stops earlier), scipy's factors, the scale equivariance, the threads and the
errors."""

import math

import numpy as np
import pytest

import stardust as sd

from . import refdata
from . import samples as S

A = sd.asarray
B = sd.smooth
REF = refdata.ref("bandwidth")
R = refdata.data("r")
SM = refdata.data("statsmodels")
SK = refdata.data("sklearn")
U = 2.0**-53


def _call(meth, s, h=None):
    x = A(s.x)
    w = None if s.w is None else A(s.w)
    return {
        "scott": lambda: B.bw_scott(x, weights=w),
        "silverman": lambda: B.bw_silverman(x, weights=w),
        "nrd0": lambda: B.bw_nrd0(x),
        "nrd": lambda: B.bw_nrd(x),
        "sj_ste": lambda: B.bw_sj(x),
        "sj_dpi": lambda: B.bw_sj(x, method="dpi"),
        "isj": lambda: B.bw_isj(x, n_grid=int(REF.group_inputs["isj_n"][0])),
        "cvls_min": lambda: B.bw_cv_ls(x),
        "cvml_min": lambda: B.bw_cv_ml(x),
        "cvls_value": lambda: B._bandwidth._cv_value(x, h, kind="ls"),
        "cvml_value": lambda: B._bandwidth._cv_value(x, h, kind="ml"),
    }[meth]()


@pytest.mark.parametrize("meth", sorted(REF.methods))
def test_references(meth):
    m = REF.methods[meth]
    closed = meth in ("scott", "silverman", "nrd0", "nrd")
    values = meth.endswith("_value")
    assert m.hi.size >= 5
    for q, at in enumerate(m.index):
        sid = str(REF.inputs["set"][at])
        s = S.density(sid)
        h = float(REF.group_inputs["h"][REF.group[at]])
        got = _call(meth, s, h)
        ref = float(m.hi[q])
        if ref == 0:
            assert got == 0, (meth, sid)
            continue
        rel = abs((got - ref) - float(m.lo[q])) / abs(ref)
        bound = refdata.C_BW * U if closed else (64 * U if values else refdata.BW_ROOT_REL)
        assert rel <= bound, (meth, sid, rel)


@pytest.mark.parametrize("did", S.density_ids(tag="1d", exclude=("weights",)))
def test_r_rules(did):
    s = S.density(did)
    rec = R[f"bw/{did}"]
    assert rec["digest"] == s.digest
    nrd0, nrd, ucv, _bcv, sj_ste, sj_dpi = rec["values"][:6]
    x = A(s.x)
    assert math.isclose(B.bw_nrd0(x), nrd0, rel_tol=4 * U)
    assert math.isclose(B.bw_nrd(x), nrd, rel_tol=4 * U, abs_tol=0)
    if "degenerate" in s.tags:
        with pytest.raises(ValueError, match="scale"):
            B.bw_sj(x)
        return
    n = s.n
    sdv = float(np.std(s.x, ddof=1))
    q = np.percentile(s.x, [25, 75])
    scale = min(sdv, (q[1] - q[0]) / 1.349)
    if math.isfinite(sj_dpi):
        assert math.isclose(B.bw_sj(x, method="dpi"), sj_dpi, rel_tol=1e-12)  # the same closed form and counts
    if math.isfinite(sj_ste):
        # R's uniroot stops at tol = 0.1 lower = 0.01 hmax; sd solves the same equation to 1e-13
        assert abs(B.bw_sj(x) - sj_ste) <= 0.01 * 1.144 * scale * n**-0.2
    if math.isfinite(ucv):
        # R's optimize stops at tol = 0.01 hmax (hmax = 1.144 sd n^(-1/5)); bw_cv_ls(nb=1000) minimizes R's ucv
        tol = 0.01 * 1.144 * sdv * n**-0.2
        got = B.bw_cv_ls(x, nb=1000)
        assert abs(got - ucv) <= 2 * tol or B._bandwidth._cv_value(x, got, nb=1000) <= B._bandwidth._cv_value(
            x, ucv, nb=1000
        )


def test_statsmodels_cross_validation():
    s = S.density("normal/200")
    x = A(s.x)
    for kind, name, fn in (("ls", "cv_ls", B.bw_cv_ls), ("ml", "cv_ml", B.bw_cv_ml)):
        theirs = SM[f"kde_multivariate/normal/200/{name}"]["value"]["array"][0]
        ours = fn(x)
        # the same criterion; statsmodels' fmin stops earlier: close, and sd's criterion no worse
        assert abs(ours - theirs) <= 1e-3 * theirs
        a, b = (B._bandwidth._cv_value(x, v, kind=kind) for v in (ours, theirs))
        assert (a <= b) if kind == "ls" else (a >= b)


def test_scipy_and_sklearn_factors():
    for did in ("normal/50", "normal/200", "bimodal/80"):  # scikit-learn's factor ignores weights
        s = S.density(did)
        w = None if s.w is None else A(s.w)
        sd_ = float(np.sqrt(np.cov(s.x, aweights=s.w)))
        for rule, fn in (("scott", B.bw_scott), ("silverman", B.bw_silverman)):
            f = SK[f"bandwidth/{did}/{rule}"]["value"]  # scikit-learn's 'scott' is scipy's factor alone
            assert math.isclose(fn(A(s.x), weights=w), f * sd_, rel_tol=1e-14)
    s = S.density("normal3/50")
    h = B.bw_scott(A(s.x))
    assert h.shape == (3,)
    f = SK["bandwidth/normal3/50/scott"]["value"]
    assert np.allclose(np.asarray(h.tolist()), f * np.sqrt(np.diag(np.cov(s.x.T))), rtol=1e-14)


def test_statsmodels_rules_differ():
    # [statsmodels와 다름] statsmodels' bw_scott and bw_silverman are 1.059 / 0.9 min(sd, IQR / 1.349) n^(-1/5)
    s = S.density("normal/200")
    rec = SM["bw/normal/200"]
    assert not math.isclose(B.bw_scott(A(s.x)), rec["bw_scott"]["value"], rel_tol=1e-3)
    q = np.percentile(s.x, [25, 75])
    their = 1.059 * min(np.std(s.x, ddof=1), (q[1] - q[0]) / 1.349) * s.n**-0.2
    assert math.isclose(their, rec["bw_scott"]["value"], rel_tol=1e-12)


def test_isj_against_the_default_grid():
    # the default 2^14 points and the reference's 256 give close bandwidths on smooth data
    s = S.density("normal/200")
    a, b = B.bw_isj(A(s.x)), B.bw_isj(A(s.x), n_grid=256)
    assert abs(a - b) <= 0.05 * a
    g = np.random.default_rng(4)
    big = np.concatenate([g.standard_normal(20000) - 3, 0.3 * g.standard_normal(20000) + 2])
    h = B.bw_isj(A(big))
    assert 0.02 < h < 0.2  # resolves the narrow mode, unlike the normal reference
    assert h < B.bw_nrd0(A(big))


def test_multivariate_cross_validation():
    s = S.density("normal2/60")
    h = B.bw_cv_ml(A(s.x))
    assert h.shape == (2,) and bool(sd.all(h > 0))
    hl = B.bw_cv_ls(A(s.x))
    assert hl.shape == (2,)
    # the search's point is no worse than its neighbours
    v0 = B._bandwidth._cv_value(A(s.x), h, kind="ml")
    for f in (0.97, 1.03):
        assert v0 >= B._bandwidth._cv_value(A(s.x), h * f, kind="ml") - 1e-9 * abs(v0)
    res = sd.smooth.kde(A(s.x), bw="cv_ml")
    assert np.array_equal(np.asarray(res.bw.tolist()), np.asarray(h.tolist()))


def test_scale_equivariance():
    s = S.density("bimodal/80")
    # the same bits times the power of two (the searches run in log(h / h_ref)); bw_sj's h^(5/7) rounds anew
    for fn in (B.bw_scott, B.bw_nrd0, B.bw_nrd, B.bw_isj, B.bw_cv_ls, B.bw_cv_ml):
        a, b = fn(A(s.x)), fn(A(s.x * 2.0**40))
        assert b == a * 2.0**40, fn.__name__
    assert B.bw_sj(A(s.x * 2.0**40)) == pytest.approx(B.bw_sj(A(s.x)) * 2.0**40, rel=1e-12)
    # R's bins are (int)(x / d), counted from 0, so a shift moves them (R's rule, kept): close, not the same
    assert B.bw_sj(A(s.x + 1e6)) == pytest.approx(B.bw_sj(A(s.x)), rel=1e-2)


def test_order_free():
    s = S.density("heavytail/60")
    p = np.random.default_rng(0).permutation(s.n)
    for fn in (B.bw_scott, B.bw_nrd0, B.bw_sj, B.bw_isj, B.bw_cv_ls, B.bw_cv_ml):
        assert fn(A(s.x)) == fn(A(s.x[p])), fn.__name__


def test_errors():
    with pytest.raises(ValueError, match="at least 2"):
        B.bw_nrd0(A([1.0]))
    with pytest.raises(ValueError, match="finite"):
        B.bw_sj(A([1.0, math.nan, 2.0]))
    with pytest.raises(ValueError, match="'method' must be"):
        B.bw_sj(A([1.0, 2.0, 4.0]), method="ucv")
    with pytest.raises(ValueError, match="'nb' must be >= 1"):
        B.bw_sj(A([1.0, 2.0, 4.0]), nb=0)
    with pytest.raises(ValueError, match="spread"):
        B.bw_isj(A([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="scale"):
        B.bw_sj(A(np.full(10, 1.5)))
    with pytest.raises(ValueError, match="1-D"):
        B.bw_nrd(A([[1.0, 2.0], [3.0, 4.0]]))
    with pytest.raises(ValueError, match="'nb'"):
        B.bw_cv_ls(A([[1.0, 2.0], [3.0, 4.0], [5.0, 7.0]]), nb=100)
    with pytest.raises(ValueError, match="spread"):
        B.bw_cv_ml(A(np.full(10, 2.0)))
    with pytest.raises(TypeError):
        B.bw_scott("abc")
    with pytest.raises(ValueError, match="'weights' must be >= 0"):
        B.bw_scott(A([1.0, 2.0]), weights=A([1.0, -1.0]))
