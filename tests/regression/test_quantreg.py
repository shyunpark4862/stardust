"""G5: quantreg (docs/api/regression.md §6, docs/design/regression.md §3.6).

Exactness against rationals (tools/regref.py's quantreg: the vertex of the returned basis
solved exactly, its residuals, objective and optimality certificate), R's quantreg 6.1
(tests/regression/ref/quantreg_r.json, made by tools/gen_quantreg_r.py; R is not needed
here): rq's "br" and "fn" fits and summary.rq's 'nid', 'iid' and 'ker' standard errors,
statsmodels' QuantReg, the vertex rule of non-unique solutions (against scipy's linear
programs), warm starts, batches, threads, NaN policies, edge cases and errors."""

import json
import math
import os
import pathlib
import sys
import warnings

import numpy as np
import pytest
from scipy.optimize import linprog

import stardust as sd
from stardust.regression._quantile import _bandwidth

R = sd.regression
REF = json.loads((pathlib.Path(__file__).parent / "ref" / "quantreg_r.json").read_text())
PROBS = {k: v for k, v in REF.items() if not k.startswith("_")}
CASES = [(name, j) for name, p in PROBS.items() for j in range(len(p["q"]))]

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
regref = pytest.importorskip("regref")

U = 2.0 ** -53


def a(x):
    return np.asarray(x.tolist() if isinstance(x, sd.ndarray) else x, dtype=float)


def design(p):
    y = np.asarray(p["y"], dtype=float)
    return y, np.asarray(p["X"], dtype=float).reshape(len(y), -1)


def r_(v):
    return np.array([np.nan if x is None else x for x in v], dtype=float)


def rel(got, ref):
    got, ref = a(got), r_(ref)
    return float(np.max(np.abs(got - ref) / np.abs(ref)))


def norm_rel(got, ref):
    got, ref = a(got), r_(ref)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def rho(r, q):
    r = np.asarray(r, dtype=float)
    return float(np.sum(r * (q - (r < 0))))


def basis(r):
    return [int(i) for i in a(r._raw[7]).astype(int) if i >= 0]


def same(x, y):
    return np.array_equal(a(x), a(y), equal_nan=True)


def exact(p, q, r):
    y, X = design(p)
    return regref.quantreg(y.tolist(), X.tolist(), basis(r), q, intercept=p["intercept"])


def lexmin(y, X, q, fstar, intercept=True):
    """The lexicographic minimum of the minimizers by scipy's linear programs: min params[0]
    over {objective <= f*}, then params[1] with params[0] fixed, ... (to the LP's tolerance)."""
    n = len(y)
    Z = np.column_stack([np.ones(n), X]) if intercept else X
    k = Z.shape[1]
    A_eq = np.hstack([Z, np.eye(n), -np.eye(n)])
    obj = np.r_[np.zeros(k), q * np.ones(n), (1 - q) * np.ones(n)]
    A_ub, b_ub = [obj], [fstar * (1 + 1e-12) + 1e-12]
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    out = []
    for j in range(k):
        c = np.zeros(k + 2 * n)
        c[j] = 1.0
        res = linprog(c, A_ub=np.array(A_ub), b_ub=b_ub, A_eq=A_eq, b_eq=y, bounds=bounds, method="highs")
        out.append(res.fun)
        row = np.zeros(k + 2 * n)
        row[j] = 1.0
        A_ub.append(row)
        b_ub.append(res.fun + 1e-9 * max(1.0, abs(res.fun)))
    return np.array(out)


# ---- exactness: the vertex of the basis, in rationals

@pytest.mark.parametrize("name,j", CASES)
def test_exact_vertex_and_certificate(name, j):
    p = PROBS[name]
    q = p["q"][j]
    r = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"])
    assert r.converged
    ref = exact(p, q, r)
    assert ref["optimal"]
    hi = np.array([v[0] for v in ref["params"]])
    lo = np.array([v[1] for v in ref["params"]])
    assert np.linalg.norm((a(r.params) - hi) - lo) <= 2e-16 * np.linalg.norm(hi)
    obj = sum(ref["objective"])
    assert abs(r.objective - obj) <= 4e-16 * obj
    # unique as certified, except a tie within rounding of the decimal q (q n an integer
    # up to the double's error: median10 at q = 0.3 is unique by 1.1e-16 exactly)
    if ref["unique"] and ref["margin"] > 1e-14:
        assert not r.nonunique
    else:
        assert r.nonunique == (not ref["unique"] or ref["margin"] <= 1e-14)
    # the basis rows' residuals vanish up to the refined solve
    y, X = design(p)
    Z = np.column_stack([np.ones(len(y)), X]) if p["intercept"] else X
    b = basis(r)
    scale = np.abs(y[b]) + np.abs(Z[b]) @ np.abs(a(r.params))
    assert np.all(np.abs(a(r.resid)[b]) <= 4 * U * scale)


def certify(y, X, q, intercept=True, **kw):
    r = R.quantreg(y, X, q=q, intercept=intercept, **kw)
    ref = regref.quantreg(np.asarray(y, float).tolist(), np.asarray(X, float).reshape(len(y), -1).tolist(), basis(r),
                          q, intercept=intercept)
    return r, ref


def test_exact_ill_conditioned_and_wide():
    rng = np.random.default_rng(11)
    n = 400
    t = rng.uniform(0, 1, n)
    X = np.column_stack([t, t ** 2, t ** 3, t ** 4, t ** 5])  # kappa ~ 1e5 with the intercept
    y = np.sin(3 * t) + 0.1 * rng.standard_normal(n)
    for q in (0.1, 0.5, 0.9):
        r, ref = certify(y, X, q)
        assert r.converged and ref["optimal"]
        hi = np.array([v[0] for v in ref["params"]])
        assert np.linalg.norm(a(r.params) - hi) <= 1e-15 * np.linalg.norm(hi)
    X = rng.standard_normal((300, 25))
    y = X @ rng.standard_normal(25) + rng.standard_t(3, 300)
    r, ref = certify(y, X, 0.3)
    assert r.converged and ref["optimal"] and ref["unique"] and not r.nonunique


def test_exact_duplicates_and_ties():
    rng = np.random.default_rng(12)
    g = rng.integers(0, 4, 500).astype(float)
    D = np.column_stack([(g == v).astype(float) for v in (1, 2, 3)])
    y = rng.poisson(2 + g).astype(float)
    for q in (0.2, 0.5, 0.8):
        r, ref = certify(y, D, q)
        assert r.converged and ref["optimal"]
        assert r.nonunique == (not ref["unique"])
        assert abs(r.objective - sum(ref["objective"])) <= 1e-15 * sum(ref["objective"])
    # an exact fit (every point on the plane, in exact arithmetic): residuals 0, the vertex
    # of the first independent rows
    X = rng.integers(-20, 20, (50, 2)).astype(float)
    y = 3.0 + X @ np.array([1.0, 2.0])
    r, ref = certify(y, X, 0.4)
    assert r.converged and ref["optimal"] and ref["unique"] and not r.nonunique
    assert r.objective == 0.0 and a(r.params).tolist() == [3.0, 1.0, 2.0]
    # a plane through rounded values: residuals at rounding level count as zero (the
    # snapping tolerance), so the vertex is optimal to that level
    X = rng.standard_normal((50, 2))
    y = 3.0 + X @ np.array([1.0, 2.0])
    r = R.quantreg(y, X, q=0.4)
    assert r.converged and r.objective <= 50 * 8 * U * np.max(np.abs(y))


# ---- R

@pytest.mark.parametrize("name,j", CASES)
def test_against_r_fits(name, j):
    p = PROBS[name]
    q = p["q"][j]
    t = f"t{j}_"
    Rv = p["R"]
    r = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"])
    # the objective: br's exact vertex to rounding; fn's interior point never below
    assert abs(r.objective - Rv[t + "obj_br"][0]) <= 4e-14 * r.objective
    assert Rv[t + "obj_fn"][0] >= r.objective * (1 - 4e-15)
    ref = exact(p, q, r)
    if ref["unique"] and ref["margin"] > 1e-14:
        # R's br solves its basis less accurately than sd (engel at q = 0.75: 1.0e-14; sd is
        # within 6e-17 of the exact vertex)
        assert norm_rel(r.params, Rv[t + "br"]) <= 4e-14
        assert norm_rel(r.params, Rv[t + "fn"]) <= 1e-6
    # R's "Solution may be nonunique" is a heuristic: it warns at the unique optima of
    # dummies60 and grid40 at q = 0.5 (certified unique above) and never misses one here
    if r.nonunique:
        assert Rv[t + "brnu"][0] == 1.0
    assert r.nonunique == (Rv[t + "brnu"][0] == 1.0) or (name, q) in (("dummies60", 0.5), ("grid40", 0.5))


def inner_unique(p, q, r, bw):
    """Whether iid's median regression on the residuals' order statistics is unique (R's br
    and sd pick different vertices otherwise)."""
    y, _ = design(p)
    n, kk = len(y), r.rank
    res = a(r.resid)
    eps = 2.0 ** -26 * (np.max(np.abs(y)) or 1.0)  # summary.rq's eps relative to y (G67)
    pz = int(np.sum(np.abs(res) < eps))
    h = max(kk + 1, math.ceil(n * _bandwidth(q, n, bw == "hall-sheather")))
    if h + pz + 1 > n:
        return False
    o = np.argsort(np.abs(res), kind="stable")
    w = np.sort(res[o][pz:h + pz + 1])
    return not R.quantreg(w, np.arange(pz + 1, h + pz + 2) / (n - kk), q=0.5).nonunique


@pytest.mark.parametrize("name,j", CASES)
def test_against_r_standard_errors(name, j):
    p = PROBS[name]
    q = p["q"][j]
    t = f"t{j}_"
    Rv = p["R"]
    for bw, suf in (("hall-sheather", ""), ("bofinger", "_bof")):
        r = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"], cov_type="nid", bandwidth=bw)
        ref = r_(Rv[t + "se_nid" + suf])
        if np.all(np.isfinite(ref)):
            assert rel(r.bse, ref) <= 2e-13
        h = _bandwidth(q, len(p["y"]), bw == "hall-sheather")
        while q - h < 0 or q + h > 1:  # summary.rq's halving
            h /= 2
        assert r.bandwidth == h
    # 'ker' and 'iid' at sd's own vertex (R's br may pick another where it is not unique)
    rk = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"], cov_type="ker")
    assert rel(rk.bse, Rv[t + "sdv_se_ker"]) <= 1e-13
    ri = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"], cov_type="iid")
    if inner_unique(p, q, ri, "hall-sheather"):
        assert rel(ri.bse, Rv[t + "sdv_se_iid"]) <= 1e-13
    if not ri.nonunique:
        ri_b = R.quantreg(p["y"], p["X"], q=q, intercept=p["intercept"], cov_type="iid", bandwidth="bofinger")
        if inner_unique(p, q, ri_b, "bofinger") and np.all(np.isfinite(r_(Rv[t + "se_iid_bof"]))):
            assert rel(ri_b.bse, Rv[t + "se_iid_bof"]) <= 1e-13


def test_iid_differs_from_r_only_by_the_inner_vertex():
    # stack loss at q = 0.75: R's iid sparsity comes from a median regression on 7 order
    # statistics that has more than one solution; R's br and sd's rule take different ones
    p = PROBS["stackloss"]
    r = R.quantreg(p["y"], p["X"], q=0.75, cov_type="iid")
    assert not inner_unique(p, 0.75, r, "hall-sheather")
    assert rel(r.bse, p["R"]["t2_se_iid"]) > 1e-3


# ---- the vertex rule of non-unique solutions (G63)

@pytest.mark.parametrize("name,j", [c for c in CASES if c[0] in ("median10", "dummies60", "grid40", "stackloss")])
def test_vertex_rule_lexicographic_minimum(name, j):
    p = PROBS[name]
    q = p["q"][j]
    y, X = design(p)
    r = R.quantreg(y, X, q=q, intercept=p["intercept"])
    lm = lexmin(y, X, q, r.objective, p["intercept"])
    assert np.max(np.abs(a(r.params) - lm)) <= 1e-7 * max(1.0, np.max(np.abs(lm)))


def test_median_of_a_sample():
    # an even n: every value between the middle two minimizes; the rule gives the lower one
    # (the type-1 quantile), R's br too; fn gives the midpoint
    y = [4.0, 1.0, 3.0, 2.0, 6.0, 5.0]
    r = R.quantreg(y, np.zeros((6, 0)))
    assert float(r.params[0]) == 3.0 and r.nonunique
    r = R.quantreg(y[:5], np.zeros((5, 0)))
    assert float(r.params[0]) == 3.0 and not r.nonunique
    # ties on the median: unique although degenerate
    r = R.quantreg([1.0, 2.0, 2.0, 3.0], np.zeros((4, 0)))
    assert float(r.params[0]) == 2.0 and not r.nonunique
    rng = np.random.default_rng(3)
    v = rng.standard_normal(200)
    for q in (0.05, 0.37, 0.5, 0.99):
        r = R.quantreg(v, np.zeros((200, 0)), q=q)
        assert float(r.params[0]) == np.sort(v)[math.ceil(200 * q) - 1]


# ---- statsmodels (its QuantReg is an IRLS approximation)

def test_against_statsmodels():
    import statsmodels.api as sm

    for name in ("engel", "hetero1000", "t5000"):
        p = PROBS[name]
        y, X = design(p)
        for q in p["q"]:
            r = R.quantreg(y, X, q=q)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = sm.QuantReg(y, sm.add_constant(X)).fit(q=q)
            assert rho(y - sm.add_constant(X) @ f.params, q) >= r.objective * (1 - 4e-16)
            assert np.max(np.abs(f.params - a(r.params))) <= 1e-3 * np.max(np.abs(a(r.params)))
            assert r.prsquared == pytest.approx(1 - r.objective / rho(y - np.sort(y)[math.ceil(len(y) * q) - 1], q),
                                                rel=1e-14)


# ---- quantile arrays (G62), batches, threads

def problems_for_paths():
    rng = np.random.default_rng(5)
    n = 2000
    X = rng.standard_normal((n, 4))
    yield "t", X @ [1.0, 2.0, 3.0, 4.0] + rng.standard_t(2, n), X
    g = np.repeat([0.0, 1.0, 2.0], 200)
    yield "dummies", rng.poisson(3 + g).astype(float), np.column_stack([(g == 1) * 1.0, (g == 2) * 1.0])
    x = np.tile(np.arange(1.0, 21.0), 30)
    yield "grid", np.round(0.3 * x + rng.normal(0, 2, x.size)), x[:, None]


@pytest.mark.parametrize("cov_type", ["nid", "iid", "ker"])
def test_quantile_array_equals_separate_fits(cov_type):
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    for _, y, X in problems_for_paths():
        ra = R.quantreg(y, X, q=qs, cov_type=cov_type)
        assert ra.params.shape == (5, X.shape[1] + 1) and ra.fittedvalues.shape == (5, len(y))
        assert a(ra.n_iter)[1:].tolist() == [0, 0, 0, 0]  # warm: from the previous quantile's basis
        for j, q in enumerate(qs):
            r1 = R.quantreg(y, X, q=q, cov_type=cov_type)
            for f in ("params", "bse", "resid", "fittedvalues"):
                assert same(getattr(ra, f)[j], getattr(r1, f)), (f, q)
            assert float(ra.objective[j]) == r1.objective and bool(ra.nonunique[j]) == r1.nonunique
        assert same(ra.predict(X[:7])[2], R.quantreg(y, X, q=0.5, cov_type=cov_type).predict(X[:7]))


def test_batch_equals_loop():
    rng = np.random.default_rng(6)
    X = rng.standard_normal((3, 500, 2))
    Y = X @ np.array([1.0, -1.0]) + rng.standard_t(3, (3, 500))
    for qs in (0.3, [0.3, 0.8]):
        rb = R.quantreg(Y, X, q=qs)
        for i in range(3):
            r1 = R.quantreg(Y[i], X[i], q=qs)
            for f in ("params", "bse", "resid", "pvalues"):
                assert same(getattr(rb, f)[i], getattr(r1, f))
            assert same(rb.objective[i], r1.objective)
    # one X, several y
    rb = R.quantreg(Y, X[0], q=0.6)
    assert same(rb.params[2], R.quantreg(Y[2], X[0], q=0.6).params)


def test_threads_bit_identical():
    rng = np.random.default_rng(7)
    n = 40000  # past the row passes' parallel threshold (2^15)
    X = rng.standard_normal((n, 3))
    y = X @ [1.0, 2.0, 3.0] + rng.standard_t(2, n)
    old = sd.get_num_threads()
    try:
        out = []
        for th in (1, 10):
            sd.set_num_threads(th)
            r = R.quantreg(y, X, q=[0.2, 0.5], cov_type="nid")
            out.append((a(r.params), a(r.bse), a(r.resid), a(r.n_iter), a(r.n_pivots)))
        for u, v in zip(*out):
            assert np.array_equal(u, v)
    finally:
        sd.set_num_threads(old)


def test_duplicate_rows_merge_to_the_same_bits():
    # tripled rows: the same minimizers; the merged problem's canonical vertex is the same
    rng = np.random.default_rng(8)
    X = rng.standard_normal((300, 2))
    y = X @ [1.0, 2.0] + rng.standard_normal(300)
    r1 = R.quantreg(y, X, q=0.3)
    r3 = R.quantreg(np.tile(y, 3), np.tile(X, (3, 1)), q=0.3)
    assert same(r1.params, r3.params)
    assert r3.objective == pytest.approx(3 * r1.objective, rel=1e-15)


# ---- options, edge cases

def test_extreme_quantiles():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((300, 2))
    y = X @ [1.0, 2.0] + rng.standard_normal(300)
    for q in (1e-6, 0.002, 0.998, 1 - 1e-6):
        r, ref = certify(y, X, q, cov_type="iid")
        assert r.converged and ref["optimal"]
    # q -> 0: the fit lies below every point
    r = R.quantreg(y, X, q=1e-9)
    assert np.all(a(r.resid) >= -1e-12)


def test_aliased_columns():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((200, 2))
    y = X @ [1.0, 2.0] + rng.standard_normal(200)
    Xc = np.column_stack([X, 2 * X[:, 0] - X[:, 1]])
    r = R.quantreg(y, Xc, q=0.4)
    r0 = R.quantreg(y, X, q=0.4)
    assert r.rank == 3 and math.isnan(float(r.params[3])) and math.isnan(float(r.bse[3]))
    assert same(r.params[:3], r0.params) and same(r.bse[:3], r0.bse) and r.df_resid == r0.df_resid
    assert same(r.predict(Xc[:5]), r0.predict(X[:5]))
    rc = R.quantreg(y, np.column_stack([np.ones(200), X]), q=0.4)  # a constant column: the intercept stays
    assert math.isnan(float(rc.params[1])) and same(rc.params[[0, 2, 3]], r0.params)


def test_small_and_degenerate_shapes():
    rng = np.random.default_rng(13)
    X = rng.standard_normal((5, 2))
    y = rng.standard_normal(5)
    r = R.quantreg(y[:3], X[:3])  # n == k: the exact fit
    assert r.converged and np.max(np.abs(a(r.resid))) <= 1e-15
    assert np.all(np.isnan(a(r.bse))) and r.df_resid == 0
    r = R.quantreg(y[:2], X[:2])  # n < k: the first columns
    assert r.rank == 2 and math.isnan(float(r.params[2]))
    r = R.quantreg(y, X[:, 0], q=0.5, intercept=False)
    assert r.params.shape == (1,) and r.converged
    r = R.quantreg(y, np.zeros((5, 1)), intercept=False)  # nothing to fit
    assert r.rank == 0 and math.isnan(float(r.params[0])) and r.objective == pytest.approx(rho(y, 0.5))


def test_nan_policies_and_inf():
    rng = np.random.default_rng(14)
    X = rng.standard_normal((100, 2))
    y = X @ [1.0, 2.0] + rng.standard_normal(100)
    yn = y.copy()
    yn[3] = np.nan
    r = R.quantreg(yn, X)
    assert np.all(np.isnan(a(r.params))) and math.isnan(r.nobs) and r.rank == 0 and not r.converged
    keep = np.arange(100) != 3
    assert same(R.quantreg(yn, X, nan_policy="omit").params, R.quantreg(y[keep], X[keep]).params)
    with pytest.raises(ValueError, match=r"quantreg\(\): the input contains NaN \(nan_policy='raise'\)"):
        R.quantreg(yn, X, nan_policy="raise")
    Y = np.stack([y, yn])
    rb = R.quantreg(Y, X)
    assert same(rb.params[0], R.quantreg(y, X).params) and np.all(np.isnan(a(rb.params[1])))
    with pytest.raises(ValueError, match="nan_policy='omit' takes one regression"):
        R.quantreg(Y, X, nan_policy="omit")
    yi = y.copy()
    yi[5] = np.inf
    with pytest.raises(ValueError, match=r"quantreg\(\): 'y' contains infinity"):
        R.quantreg(yi, X, nan_policy="omit")
    Xi = X.copy()
    Xi[2, 1] = -np.inf
    with pytest.raises(ValueError, match=r"quantreg\(\): 'X' contains infinity"):
        R.quantreg(y, Xi)


def test_errors():
    y, X = np.arange(10.0), np.ones((10, 1))
    for q in (0.0, 1.0, -0.1, math.nan, [0.2, 1.0], [[0.5]], []):
        with pytest.raises(ValueError, match="'q' must be a number or a 1-dimensional array"):
            R.quantreg(y, X, q=q)
    with pytest.raises(TypeError, match="'q' must be"):
        R.quantreg(y, X, q="0.5")
    with pytest.raises(ValueError, match=r"'cov_type' must be 'iid', 'nid' or 'ker', not 'boot'"):
        R.quantreg(y, X, cov_type="boot")
    with pytest.raises(ValueError, match=r"'bandwidth' must be 'hall-sheather' or 'bofinger', not 'x'"):
        R.quantreg(y, X, bandwidth="x")
    with pytest.raises(ValueError, match="'max_iter'"):
        R.quantreg(y, X, max_iter=0)
    with pytest.raises(ValueError, match="'tol'"):
        R.quantreg(y, X, tol=0.0)
    with pytest.raises(ValueError, match="the model has no parameters"):
        R.quantreg(y, np.zeros((10, 0)), intercept=False)
    with pytest.raises(ValueError, match="'X' must have 10 rows like 'y', not 9"):
        R.quantreg(y, X[:9])
    with pytest.raises(TypeError, match="complex"):
        R.quantreg(y.astype(complex), X)


def test_result_fields_and_summary():
    p = PROBS["engel"]
    import stardust.frame as fr

    y, X = design(p)
    r = R.quantreg(fr.Series(y, name="foodexp"), fr.DataFrame({"income": X[:, 0]}), q=0.25)
    assert r.param_names == ["const", "income"]
    assert r.q == 0.25 and r.cov_type == "nid" and r.nobs == 235.0 and r.df_model == 1.0 and r.df_resid == 233.0
    assert isinstance(r.n_iter, int) and isinstance(r.n_pivots, int) and isinstance(r.converged, bool)
    t = a(r.tvalues)
    assert same(r.pvalues, 2 * sd.stats.StudentT(df=233.0).ccdf(sd.asarray(np.abs(t))))
    ci = a(r.conf_int(0.1))
    assert ci.shape == (2, 2) and np.all(ci[:, 0] < a(r.params)) and np.all(a(r.params) < ci[:, 1])
    assert tuple(r.cov_params().shape) == (2, 2) and same(sd.sqrt(sd.linalg.diagonal(r.cov_params())), r.bse)
    assert np.max(np.abs(a(r.fittedvalues) + a(r.resid) - y)) <= 1e-13 * np.max(np.abs(y))
    assert same(r.predict(X), r.fittedvalues)
    s = r.summary()
    assert "QuantReg Regression Results" in s and "income" in s and "Pseudo R-squared" in s
    with pytest.raises(ValueError, match=r"^summary\(\): takes one regression at one quantile, not 2 quantiles$"):
        R.quantreg(y, X, q=[0.2, 0.4]).summary()
    rq = R.quantreg(y, X, q=sd.asarray([0.2, 0.4]))
    assert rq.objective.shape == (2,) and rq.predict(X[:3]).shape == (2, 3) and rq.conf_int().shape == (2, 2, 2)
    assert "QuantRegResult(q=0.25" in repr(r) and repr(rq) == "QuantRegResult(2 quantiles, 2 parameters)"
    rb = R.quantreg(np.vstack([y, y]), X, q=[0.2, 0.4])
    assert repr(rb) == "QuantRegResult(batch shape (2,) and 2 quantiles, 2 parameters)"
    assert "not batch shape (2,) and 2 quantiles" in pytest.raises(ValueError, rb.summary).value.args[0]
    assert repr(R.quantreg(np.vstack([y, y]), X)) == "QuantRegResult(batch shape (2,), 2 parameters)"


@pytest.mark.parametrize("cov_type", ["nid", "iid", "ker"])
def test_scale_equivariance(cov_type):
    # summary.rq's eps relative to max |y| (G67): y scaled by c scales params and bse by c
    # (with R's absolute eps, f is clipped to 0 or every residual counts as zero in small units)
    rng = np.random.default_rng(15)
    X = rng.standard_normal((400, 2))
    yc = X @ [1.0, 2.0] + rng.standard_t(3, 400)
    yp = X @ [1.0, 2.0] + 0.5
    yp[120:] += rng.standard_t(3, 280)  # 30% on the plane: zero residuals, pz matters for 'iid'
    cases = [(yc, X)]
    if cov_type != "nid":  # nid: the fits at q +- h are the same plane there, f = 0 (as in R)
        cases.append((yp, X))
    # (integer outcomes are left out: their residuals tie in |r|, 'iid' orders ties by row, an
    # order the rounding of c y changes; 'nid' there is degenerate, f ~ 0 on groups whose
    # quantile does not move between q - h and q + h)
    for y, Z in cases:
        for q in (0.3, 0.5):
            r = R.quantreg(y, Z, q=q, cov_type=cov_type)
            assert np.all(np.isfinite(a(r.bse)))
            if cov_type == "iid" and y is yp and q == 0.5:
                assert np.sum(np.abs(a(r.resid)) < 2.0 ** -26 * np.max(np.abs(y))) >= 100
            for c in (1e-9, 1e-3, 1e3, 1e9):
                rc = R.quantreg(c * y, Z, q=q, cov_type=cov_type)
                assert np.max(np.abs(a(rc.params) - c * a(r.params))) <= 1e-13 * c * np.max(np.abs(a(r.params)))
                assert np.max(np.abs(a(rc.bse) / (c * a(r.bse)) - 1)) <= 1e-13, (c, q)
