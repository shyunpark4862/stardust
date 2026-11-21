"""sd.htest's correlation tests (milestone H5; docs/api/htest.md §6, docs/design/htest.md §3.6,
§4): the accuracy contract against the Arb and exact-rational references (tests/htest/ref),
batches lane by lane, the thread count, nan_policy, the degenerate inputs (H11, H20), the H5
decisions (H62-H66: spearmanr and spearmanrho, no Spearman exact distribution, weightedtau's NaN
p-value, no `correlation` alias, pearsonr's r against sd.corrcoef), the exact Kendall
distribution, scipy's definitions, R's cor.test, the results and the messages."""

import math
import time

import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

from .._bits import bits_equal
from . import problems as P
from . import refdata as R

FAMILIES = R.H5
REFS = {fam: R.load(fam) for fam in FAMILIES}
CASES = [(fam, pid) for fam in FAMILIES for pid in REFS[fam]]
call = R.call
h = sd.htest
PAIRED = (h.pearsonr, h.spearmanrho, h.pointbiserialr, h.kendalltau, h.weightedtau, h.chatterjeexi)


def _quiet(f, *a, **k):
    import warnings

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


# ---- the accuracy contract (docs/design/htest.md §4) --------------------------------------


@pytest.mark.parametrize(("fam", "pid"), CASES, ids=[pid for _, pid in CASES])
def test_contract_against_the_references(fam, pid):
    """r, rho, tau, xi and D to 1e-14 relative (exact rationals, or the square roots of exact
    ones, rounded once); the p-values to 1e-14 max(1, kappa_p) (kappa_p in 1 - r^2 for Pearson and
    Spearman: sd forms it without cancellation), Kendall's exact ones to 1e-14 (the inversions'
    distribution); pearsonr's Fisher interval within 1e-14 of its ends' scale."""
    rec = REFS[fam][pid]
    r = call(P.PROBLEMS[pid])
    for field, (e, c) in R.errors(fam, pid, r).items():
        assert c <= 1.0, (field, e)
    if "method" in rec:
        assert r.method == rec["method"]
    if "table" in rec:
        assert r.table.tolist() == rec["table"]


def test_every_h5_function_has_references():
    assert {P.PROBLEMS[pid].func for _, pid in CASES} == {
        "pearsonr",
        "spearmanrho",
        "spearmanr",
        "pointbiserialr",
        "kendalltau",
        "weightedtau",
        "somersd",
        "chatterjeexi",
    }


@pytest.mark.parametrize("pid", [p.id for p in P.PROBLEMS.values() if p.milestone == "H5" and "batch" in p.tags])
def test_batch_is_lane_by_lane(pid):
    """A batch (axis=-1) gives each lane's bits of the lane alone (axis=None), its methods too (H48)."""
    p = P.PROBLEMS[pid]
    r = call(p)
    kw = {k: v for k, v in p.kwargs.items() if k != "axis"}
    lanes = p.args[0].shape[0]
    assert r.statistic.shape == (lanes,)
    for i in range(lanes):
        one = call(P.Problem(p.id, p.func, tuple(a[i] for a in p.args), kw))
        for f in type(one)._FIELDS:
            got, want = getattr(r, f), getattr(one, f)
            if isinstance(want, str):
                assert got[i] == want
            else:
                assert bits_equal(np.asarray(got)[i], want), (i, f)
        if hasattr(one, "conf_int"):
            assert bits_equal(np.asarray(r.conf_int())[i], np.asarray(one.conf_int()))


def test_threads_do_not_change_the_bits():
    rng = np.random.default_rng(17)
    x = rng.normal(size=200_000)
    y = 0.01 * x + rng.normal(size=200_000)
    bx, by = rng.normal(size=(3000, 30)), rng.normal(size=(3000, 30))
    calls = [
        lambda: h.pearsonr(x, y),
        lambda: h.spearmanrho(x, y),
        lambda: h.spearmanrho(np.round(x, 1), np.round(y, 1)),
        lambda: h.kendalltau(x[:50_000], y[:50_000]),
        lambda: h.weightedtau(x[:50_000], y[:50_000]),
        lambda: h.chatterjeexi(x[:50_000], y[:50_000]),
        lambda: h.pearsonr(bx, by, axis=-1),
        lambda: h.spearmanrho(bx, by, axis=-1),
        lambda: h.kendalltau(bx, by, axis=-1),
        lambda: h.chatterjeexi(bx, by, axis=-1),
        lambda: h.spearmanr(bx[:40].T, axis=0),
    ]
    before = sd.get_num_threads()
    try:
        want = None
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            got = [np.asarray([np.asarray(r.statistic), np.asarray(r.pvalue)]) for r in (c() for c in calls)]
            if want is None:
                want = got
            else:
                for g, w in zip(got, want):
                    assert bits_equal(g, w)
    finally:
        sd.set_num_threads(before)


# ---- scipy's definitions ------------------------------------------------------------------


def _close(a, b, rtol):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.allclose(a, b, rtol=rtol, atol=0.0, equal_nan=True)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_tests_are_scipys(seed):
    """Away from scipy's losses (|r| near 1, docs/design/htest.md §7) the values are scipy 1.18.1's
    to rounding: every alternative, Kendall's variants and methods, the weighted tau's options,
    Somers' D from samples and tables, Chatterjee's xi (no ties in x: scipy breaks those by its
    sort)."""
    rng = np.random.default_rng(seed)
    for n in (5, 14, 40, 300):
        x = rng.normal(size=n)
        y = 0.3 * x + rng.normal(size=n)
        xt, yt = np.round(x * 2), np.round(y * 2)
        for alt in ("two-sided", "less", "greater"):
            a, b = h.pearsonr(x, y, alternative=alt), st.pearsonr(x, y, alternative=alt)
            assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (n, alt)
            ci = b.confidence_interval()
            assert _close(a.conf_int(), [ci.low, ci.high], 1e-12)
            ci = b.confidence_interval(confidence_level=0.9)
            assert _close(a.conf_int(alpha=0.1), [ci.low, ci.high], 1e-12)
            for u, v in ((x, y), (xt, yt)):
                a, b = h.spearmanrho(u, v, alternative=alt), st.spearmanrho(u, v, alternative=alt)
                assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (n, alt)
                for var in ("b", "c"):
                    for m in ("auto", "asymptotic"):
                        a = h.kendalltau(u, v, alternative=alt, variant=var, method=m)
                        b = st.kendalltau(u, v, alternative=alt, variant=var, method=m)
                        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (n, alt, var, m)
                if n <= 40 or u is xt:  # (scipy's pair sums are O((r c)^2) on the table)
                    a, b = h.somersd(u, v, alternative=alt), st.somersd(u, v, alternative=alt)
                    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12)
                    assert a.table.tolist() == b.table.tolist()
        for u, v in ((x, y), (xt, yt)):
            for add in (True, False):
                for rank in (True, None, False, rng.integers(0, n, n)):
                    a, b = h.weightedtau(u, v, rank=rank, additive=add), st.weightedtau(u, v, rank=rank, additive=add)
                    assert abs(a.statistic - b.statistic) < 1e-14 and math.isnan(a.pvalue), (n, add)
            a = h.weightedtau(u, v, weigher=lambda r: 1.0 / (r + 1) ** 2)
            b = st.weightedtau(u, v, weigher=lambda r: 1.0 / (r + 1) ** 2)
            assert abs(a.statistic - b.statistic) < 1e-14
        a, b = h.pointbiserialr(x > 0, y), st.pointbiserialr(x > 0, y)
        assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12)
        xs = rng.permutation(n).astype(float)
        for v in (y, yt):
            for cont in (True, False):
                a, b = h.chatterjeexi(xs, v, y_continuous=cont), st.chatterjeexi(xs, v, y_continuous=cont)
                assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12), (n, cont)
    t = rng.integers(0, 6, size=(4, 5)).astype(float)
    a, b = h.somersd(t), st.somersd(t)
    assert _close([a.statistic, a.pvalue], [b.statistic, b.pvalue], 1e-12) and a.table.tolist() == t.tolist()


def test_kendalltau_methods():
    """'auto' as scipy's (H46): exact without ties for n <= 33, or with at most one pair
    discordant (or concordant) at any n; the method field says which, lane by lane (H48)."""
    rng = np.random.default_rng(5)
    assert h.kendalltau(rng.normal(size=33), rng.normal(size=33)).method == "exact"
    assert h.kendalltau(rng.normal(size=34), rng.normal(size=34)).method == "asymptotic"
    x = np.arange(500.0)
    assert h.kendalltau(x, x).method == "exact"
    y = x.copy()
    y[[3, 4]] = y[[4, 3]]
    r = h.kendalltau(x, y)
    assert r.method == "exact" and r.pvalue == 0.0  # 2 (1 + 499) / 500!: below the doubles
    assert h.kendalltau([1.0, 2, 2, 3], [1.0, 3, 2, 4]).method == "asymptotic"  # ties
    a = rng.normal(size=(2, 40))
    a[0, 20:] = np.nan
    r = h.kendalltau(a, rng.normal(size=(2, 40)), axis=-1, nan_policy="omit")
    assert r.method == ["exact", "asymptotic"]
    # under 'propagate' a NaN lane's method is decided on its other values (H53)
    r = h.kendalltau(a, rng.normal(size=(2, 40)), axis=-1)
    assert r.method == ["exact", "asymptotic"] and math.isnan(r.pvalue[0])


# ---- Kendall's exact distribution ---------------------------------------------------------


def test_kendall_exact_distribution():
    """The integer counts (n <= 33) and the double-double probabilities (beyond) are the same tails;
    the deep tails are 2/n! (no discordant pair) and 2 n/n! (one), below scipy's range at n = 200;
    the work cap of method='exact' is a ValueError, and so are ties (scipy's rule)."""
    k = sd._core.htest
    for n in (5, 12, 20, 33):
        for c in (0, 1, n, n * (n - 1) // 4):
            a, b = k.kendall_tail(n, c, False), k.kendall_tail(n, c, True)
            assert math.isclose(a[0], b[0], rel_tol=2e-15) and math.isclose(a[1], b[1], rel_tol=2e-15), (n, c)
    x = np.arange(60.0)
    assert math.isclose(h.kendalltau(x, x).pvalue, 2.0 / math.factorial(60), rel_tol=1e-15)
    x = np.arange(150.0)
    p = h.kendalltau(x, x[::-1].copy(), alternative="less").pvalue
    assert math.isclose(p, 1.0 / math.factorial(150), rel_tol=1e-14)
    p = h.kendalltau(np.arange(200.0), np.arange(200.0), alternative="greater").pvalue
    assert p == 0.0 or p < 1e-300  # 1/200! = 1.3e-375: below the doubles
    rng = np.random.default_rng(3)
    u, v = rng.normal(size=120), rng.normal(size=120)
    t0 = time.perf_counter()
    e, a = h.kendalltau(u, v, method="exact"), h.kendalltau(u, v)
    assert time.perf_counter() - t0 < 5.0 and abs(e.pvalue - a.pvalue) < 0.01 and e.statistic == a.statistic
    with pytest.raises(ValueError, match=r"kendalltau\(\): method='exact' needs about .* use method='asymptotic'"):
        h.kendalltau(rng.normal(size=3000), rng.normal(size=3000), method="exact")
    with pytest.raises(ValueError, match=r"kendalltau\(\): method='exact' needs samples without ties"):
        h.kendalltau([1.0, 2, 2, 3], [1.0, 3, 2, 4], method="exact")


# ---- the H5 decisions ---------------------------------------------------------------------


def test_spearmanr_matrix_is_spearmanrho():
    """spearmanr (scipy's older form, H62): two variables give numbers, more the (k, k) matrices
    whose entries are spearmanrho's bits; the diagonal 1 and 0; axis 0 (columns), 1 (rows) or None
    (flattened); b adds its variables."""
    rng = np.random.default_rng(11)
    a = rng.normal(size=(30, 4))
    a[:, 1] = np.round(a[:, 1])
    r = h.spearmanr(a, axis=0)
    assert r.statistic.shape == (4, 4) and repr(r) == "SpearmanrResult(batch shape (4, 4))"
    for i in range(4):
        assert r.statistic[i, i] == 1.0 and r.pvalue[i, i] == 0.0
        for j in range(4):
            if i != j:
                o = h.spearmanrho(a[:, i], a[:, j])
                assert r.statistic[i, j] == o.statistic and r.pvalue[i, j] == o.pvalue
    b = st.spearmanr(a)
    assert _close(r.statistic, b.statistic, 1e-12) and _close(r.pvalue, b.pvalue, 1e-10)
    t = h.spearmanr(a.T, axis=1)
    assert bits_equal(t.statistic, r.statistic)
    two = h.spearmanr(a[:, 0], a[:, 2])
    stat, p = two
    assert isinstance(stat, float) and stat == h.spearmanrho(a[:, 0], a[:, 2]).statistic
    m = h.spearmanr(a[:, :2], a[:, 2:], axis=0)
    assert bits_equal(m.statistic, r.statistic)
    # nan_policy: a variable with a NaN has NaN entries ('propagate'), pairwise deletion ('omit')
    an = a.copy()
    an[3, 2] = np.nan
    r = h.spearmanr(an, axis=0)
    S = np.asarray(r.statistic)
    assert np.isnan(S[2]).all() and np.isnan(S[:, 2]).all() and r.statistic[0, 1] == r.statistic[1, 0]
    r = h.spearmanr(an, axis=0, nan_policy="omit")
    keep = ~np.isnan(an[:, 2])
    assert r.statistic[0, 2] == h.spearmanrho(a[keep, 0], a[keep, 2]).statistic
    assert r.statistic[0, 1] == h.spearmanrho(a[:, 0], a[:, 1]).statistic
    b = st.spearmanr(an, nan_policy="omit")
    assert _close(r.statistic, b.statistic, 1e-12)
    with pytest.raises(ValueError, match=r"spearmanr\(\): the input contains NaN \(nan_policy='raise'\)"):
        h.spearmanr(an, axis=0, nan_policy="raise")
    # axis=None flattens: one variable each
    assert h.spearmanr(a[:, :2], a[:, 2:]).statistic == h.spearmanrho(a[:, :2], a[:, 2:]).statistic


def test_no_spearman_exact_distribution():
    """H63: spearmanrho's p-value is the t approximation (scipy's) at every n; R's cor.test default
    is AS 89's exact distribution (recorded beside its exact = FALSE value in r.json) and differs
    on small samples, up to 1.5 times on spearmanrho/small."""
    data = R.data("r")
    seen = 0
    for pid in ("spearmanrho/small", "spearmanr/small", "spearmanrho/large", "spearmanrho/less"):
        got = data[pid]["result"]
        r = call(P.PROBLEMS[pid])
        assert math.isclose(got["p.value"][0], r.pvalue, rel_tol=1e-12) and math.isclose(
            got["estimate"][0], r.statistic, rel_tol=1e-13
        ), pid
        e = abs(got["p.value.default"][0] - r.pvalue) / r.pvalue
        assert e > 1e-6, pid
        seen += 1
    assert seen == 4


def test_weightedtau_pvalue_and_no_alias():
    """H64: weightedtau's pvalue is always NaN (scipy defines none); H65: no result has scipy's
    `correlation` alias."""
    rng = np.random.default_rng(13)
    x, y = rng.normal(size=20), rng.normal(size=20)
    r = h.weightedtau(x, y)
    assert math.isnan(r.pvalue) and len(r) == 2
    b = h.weightedtau(rng.normal(size=(3, 9)), rng.normal(size=(3, 9)), axis=-1)
    assert np.isnan(np.asarray(b.pvalue)).all()
    for f in PAIRED:
        assert not hasattr(f(x, y), "correlation"), f.__name__
    assert not hasattr(h.somersd(np.round(x), np.round(y)), "correlation")
    assert not hasattr(h.spearmanr(x, y), "correlation")


def test_pearsonr_against_corrcoef():
    """H66: pearsonr's r is the htest kernel's (double-double sums), sd.corrcoef's the array area's
    (numpy's): both within their contracts, the last digits may differ."""
    rng = np.random.default_rng(17)
    diff = 0
    for _ in range(200):
        n = int(rng.integers(3, 60))
        x = rng.normal(size=n)
        y = rng.normal() * x + rng.normal(size=n)
        r = h.pearsonr(x, y)
        c = float(sd.corrcoef(sd.asarray(np.stack([x, y])))[0, 1])
        assert math.isclose(r.statistic, c, rel_tol=1e-13)
        diff += r.statistic != c
    assert diff > 0  # (the two are not made bit-identical)


# ---- degenerate inputs, nan_policy --------------------------------------------------------


def test_degenerate_inputs():
    nan = math.isnan
    for f in PAIRED:
        r = f([], [])
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
        r = f([1.0], [2.0])
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
    # a constant sample (H20's rule for Pearson: m2 <= (eps mean)^2)
    for f in (h.pearsonr, h.spearmanrho, h.kendalltau, h.weightedtau):
        r = f(np.full(6, 2.0), np.arange(6.0))
        assert nan(r.statistic) and nan(r.pvalue), f.__name__
    r = h.pearsonr(1e8 + np.array([0.0, 1e-9, 0.0, 1e-9]), np.arange(4.0))
    assert nan(r.statistic)  # (scipy warns that it is nearly constant and gives a number)
    assert nan(h.chatterjeexi(np.arange(6.0), np.full(6, 1.0)).statistic)
    # n = 2: r = +-1 and p = 1 (scipy's); an exact line: |r| = 1, p = 0
    r = h.pearsonr([1.0, 2.0], [5.0, 3.0])
    assert r.statistic == -1.0 and r.pvalue == 1.0 and r.conf_int().tolist() == [-1.0, 1.0]
    r = h.pearsonr(np.arange(10.0), 2.0 * np.arange(10.0))
    assert r.statistic == 1.0 and r.pvalue == 0.0
    # an infinity: Pearson's r is undefined; the ranks keep it in its place
    assert nan(h.pearsonr([1.0, np.inf, 3.0, 4.0], [1.0, 2.0, 3.0, 5.0]).statistic)
    assert h.spearmanrho([1.0, np.inf, 3.0, 4.0], [1.0, 9.0, 3.0, 5.0]).statistic == 1.0
    # extreme scales: the unit of 2^-e
    rng = np.random.default_rng(19)
    x, y = rng.normal(size=30), rng.normal(size=30)
    a, b = h.pearsonr(x, y), h.pearsonr(x * 1e300, y * 1e-300)
    assert math.isclose(a.statistic, b.statistic, rel_tol=1e-15) and math.isclose(a.pvalue, b.pvalue, rel_tol=1e-14)


def test_nan_policy():
    rng = np.random.default_rng(31)
    x, y = rng.normal(size=30), rng.normal(0.3, 1.0, size=30)
    xn = x.copy()
    xn[4] = np.nan
    keep = np.arange(30) != 4
    for f in PAIRED:
        r = f(xn, y)
        assert math.isnan(r.statistic) and math.isnan(r.pvalue), f.__name__
        a, b = f(xn, y, nan_policy="omit"), f(x[keep], y[keep])
        assert bits_equal([a.statistic, a.pvalue], [b.statistic, b.pvalue]), f.__name__
        with pytest.raises(ValueError, match=r"the input contains NaN \(nan_policy='raise'\)"):
            f(xn, y, nan_policy="raise")
    # weightedtau with NaN under 'propagate': NaN (scipy ranks a NaN above every number)
    assert not math.isnan(_quiet(st.weightedtau, xn, y).statistic)


def test_axis_and_keepdims():
    rng = np.random.default_rng(37)
    x, y = rng.normal(size=(3, 40, 2)), rng.normal(size=(1, 40, 2))
    r = h.pearsonr(x, y, axis=1)
    assert r.statistic.shape == (3, 2) and r.conf_int().shape == (3, 2, 2)
    one = h.pearsonr(x[2, :, 1], y[0, :, 1])
    assert r.pvalue[2, 1] == one.pvalue and bits_equal(np.asarray(r.conf_int())[2, 1], np.asarray(one.conf_int()))
    r = h.kendalltau(x, y, axis=1)
    assert r.method == [["asymptotic", "asymptotic"]] * 3  # (n = 40)
    assert h.kendalltau(x[:, :30], y[:, :30], axis=1).method == [["exact", "exact"]] * 3
    r = h.chatterjeexi(x, y, axis=1, keepdims=True)
    assert r.statistic.shape == (3, 1, 2)
    r = h.spearmanrho(x, y, axis=(1, 2))
    assert r.statistic.shape == (3,)
    r = h.weightedtau(x, y, axis=1, rank=rng.integers(0, 9, size=(3, 40, 2)))
    assert r.statistic.shape == (3, 2)


# ---- R ------------------------------------------------------------------------------------

R_ESTIMATE_KNOWN = {"pearsonr/offset", "pearsonr/offset-both"}  # R's centring rounds the offset


def test_r_cor_test():
    """R's cor.test (pearson, kendall with exact as sd's method, spearman exact = FALSE; tests/htest/
    data/r.json): the same estimate and Fisher interval to 1e-12 but where R rounds the offset
    data's centring; the p-values against the references are test_reference_data.py's."""
    data = R.data("r")
    seen = 0
    for pid, p in P.PROBLEMS.items():
        if p.func not in ("pearsonr", "pointbiserialr", "spearmanrho", "spearmanr", "kendalltau"):
            continue
        got = (data.get(pid) or {}).get("result")
        if not got or got.get("estimate", [None])[0] is None:
            continue
        r = call(p)
        e = abs(got["estimate"][0] - r.statistic) / max(abs(r.statistic), 1e-300)
        if pid in R_ESTIMATE_KNOWN:
            assert 1e-12 < e < 1e-8, pid
        else:
            assert e <= 1e-12, (pid, e)
            if "conf.int" in got and p.func == "pearsonr":
                assert _close(got["conf.int"], r.conf_int(), 1e-12), pid
        seen += 1
    assert seen >= 40


# ---- results and messages -----------------------------------------------------------------


def test_results():
    rng = np.random.default_rng(41)
    x, y = rng.normal(size=12), rng.normal(size=12)
    r = h.pearsonr(x, y)
    stat, p = r
    assert (stat, p) == (r[0], r[1]) and len(r) == 2 and isinstance(r.statistic, float)
    with pytest.raises(AttributeError):
        r.statistic = 1.0
    ci = r.conf_int()
    ci[0] = 5.0  # a new array at each call
    assert r.conf_int()[0] != 5.0
    lo, hi = h.pearsonr(x, y, alternative="greater").conf_int().tolist()
    assert hi == 1.0 and lo < r.statistic
    assert h.pearsonr(x, y, alternative="less").conf_int().tolist()[0] == -1.0
    with pytest.raises(ValueError, match=r"conf_int\(\): 'alpha' must be between 0 and 1 \(exclusive\)"):
        r.conf_int(alpha=1.5)
    r = h.kendalltau(x, y)
    assert repr(r).startswith("KendalltauResult(statistic=") and "method='exact'" in repr(r)
    r = h.somersd(np.round(x), np.round(y))
    with pytest.raises(TypeError):
        iter(r)
    assert r.table.dtype == sd.int64
    with pytest.raises(ValueError):
        r.table[0, 0] = 1  # a read-only array (H21)
    b = h.spearmanrho(rng.normal(size=(2, 3, 10)), rng.normal(size=(2, 3, 10)), axis=-1)
    assert repr(b) == "SpearmanrhoResult(batch shape (2, 3))"
    assert not hasattr(h.pearsonr(x, y), "confidence_interval")


def test_messages():
    x, y = [0.1, 0.4, -0.2, 1.3, 0.7], [0.3, 1.1, 0.0, -0.5, 2.0]
    with pytest.raises(TypeError, match=r"pearsonr\(\): 'method' must be None, not str"):
        h.pearsonr(x, y, method="permutation")
    with pytest.raises(TypeError, match=r"spearmanrho\(\): 'method' must be None, not str"):
        h.spearmanrho(x, y, method="exact")
    with pytest.raises(ValueError, match=r"pearsonr\(\): the paired samples must agree in size, not 5 and 4"):
        h.pearsonr(x, y[:4])
    with pytest.raises(ValueError, match=r"kendalltau\(\): 'variant' must be"):
        h.kendalltau(x, y, variant="a")
    with pytest.raises(ValueError, match=r"kendalltau\(\): 'method' must be"):
        h.kendalltau(x, y, method="permutation")
    with pytest.raises(ValueError, match=r"chatterjeexi\(\): 'method' must be"):
        h.chatterjeexi(x, y, method="auto")
    with pytest.raises(TypeError, match=r"chatterjeexi\(\): 'y_continuous' must be"):
        h.chatterjeexi(x, y, y_continuous=1)
    with pytest.raises(TypeError, match=r"weightedtau\(\): 'weigher' must be None or a callable, not float"):
        h.weightedtau(x, y, weigher=1.0)
    with pytest.raises(ValueError, match=r"weightedtau\(\): 'rank' must hold finite numbers"):
        h.weightedtau(x, y, rank=[1.0, np.nan, 2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match=r"somersd\(\): 'x' must hold non-negative integer counts"):
        h.somersd([[1.0, 2.5], [3.0, 1.0]])
    with pytest.raises(ValueError, match=r"somersd\(\): the table must hold at least two nonzero counts"):
        h.somersd([[0.0, 0.0], [3.0, 0.0]])
    with pytest.raises(ValueError, match=r"somersd\(\): 'y' is required when 'x' is a 1-D sample"):
        h.somersd(x)
    with pytest.raises(ValueError, match=r"somersd\(\): 'y' must be None when 'x' is a table"):
        h.somersd([[1.0, 2.0], [3.0, 1.0]], x)
    with pytest.raises(ValueError, match=r"somersd\(\): 'x' and 'y' must not contain NaN"):
        h.somersd([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(ValueError, match=r"somersd\(\): the table of 20000 x 20000 distinct values"):
        h.somersd(np.arange(20000.0), np.arange(20000.0))
    with pytest.raises(ValueError, match=r"spearmanr\(\): needs at least 2 variables to compare, not 1"):
        h.spearmanr(x)
    with pytest.raises(ValueError, match=r"spearmanr\(\): 'axis' must be None, 0 or 1, not 2"):
        h.spearmanr(np.ones((3, 3, 3)), axis=2)
    with pytest.raises(ValueError, match=r"spearmanr\(\): 'a' must have at most 2 dimensions, not 3"):
        h.spearmanr(np.ones((3, 3, 3)), axis=0)
    with pytest.raises(ValueError, match=r"spearmanr\(\): the variables must have the same number of observations"):
        h.spearmanr(x, y[:3])
    with pytest.raises(TypeError, match=r"pearsonr\(\): unexpected keyword argument 'nan'"):
        h.pearsonr(x, y, nan="omit")
    with pytest.raises(TypeError, match=r"pearsonr\(\): 'x' must be"):
        h.pearsonr({"a": 1}, y)
