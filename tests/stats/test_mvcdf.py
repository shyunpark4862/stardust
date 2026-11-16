"""The multivariate normal and t cdfs (T9, csrc/stats/mvcdf.hpp), through the native kernel
stardust._core.stats.mv_cdf(cov, x, batch, df, atol, max_points) -> (values, errors).

References (tests/stats/ref_mvcdf/, written by tools/mvcdf_ref.py):
  bvn, tvn   Phi2 and Phi3 as rigorous Arb balls (double-double hi + lo)
  factor_n   one-factor normals d = 4..10, Arb
  factor_t   one-factor t's d = 2..8, two-level adaptive quadrature (~1e-11)
The accuracy bounds below are the measured ones (bench/stats/t9/cdf_accuracy.md), with a
small margin.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.special as sp
import scipy.stats as st

import stardust as sd
from stardust._core import stats as C

REF = Path(__file__).parent / "ref_mvcdf"
EPS = 2.0**-52
INF = math.inf


def mv(cov, x, df=INF, atol=1e-6, max_points=0, batch=None):
    """values, errors of P(X <= x); cov (d, d) or (B, d, d), x (d,) or (M, d)."""
    cov = np.asarray(cov, dtype=float)
    x = np.atleast_2d(np.asarray(x, dtype=float))
    if cov.ndim == 2:
        cov = cov[None]
    if batch is None:
        batch = np.zeros(len(x), np.int64) if len(cov) == 1 else np.arange(len(x), dtype=np.int64)
    df = np.broadcast_to(np.asarray(df, dtype=float), (len(cov),)).copy()
    v, e = C.mv_cdf(sd.asarray(cov), sd.asarray(x), sd.asarray(np.asarray(batch, np.int64)), sd.asarray(df),
                    atol, max_points)
    return np.asarray(v), np.asarray(e)


def corr(R3):
    """(M, 3) correlations (r12, r13, r23) as (M, 3, 3) matrices."""
    M = len(R3)
    S = np.tile(np.eye(3), (M, 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
        S[:, a, b] = S[:, b, a] = R3[:, q]
    return S


def rel_err(v, hi, lo):
    return np.abs((v - hi) - lo) / hi


# ---------------------------------------------------------------- d = 1..3, exact


def test_d1_is_ndtr_and_stdtr():
    x = np.linspace(-30, 8, 200)
    for var in (1.0, 4.0, 0.01):
        v, e = mv([[var]], x[:, None])
        np.testing.assert_array_equal(v, np.asarray(sd.special.ndtr(sd.asarray(x / math.sqrt(var)))))
        assert np.all(e == 0)
        for df in (0.5, 3.0, 40.0):
            v, e = mv([[var]], x[:, None], df=df)
            np.testing.assert_array_equal(v, np.asarray(sd.special.stdtr(sd.asarray(np.full_like(x, df)),
                                                                         sd.asarray(x / math.sqrt(var)))))
            assert np.all(e == 0)


def test_bvn_reference():
    z = np.load(REF / "bvn.npz")
    h, k, r, hi, lo = z["h"], z["k"], z["r"], z["hi"], z["lo"]
    M = len(h)
    S = np.tile(np.eye(2), (M, 1, 1))
    S[:, 0, 1] = S[:, 1, 0] = r
    v, e = mv(S, np.c_[h, k])
    assert np.all(e == 0)
    norm = hi >= 1e-300
    rel = rel_err(v[norm], hi[norm], lo[norm])
    # 1e-14 relative down to P = 1e-15, for every r in (-1, 1) (measured 6.9e-15)
    assert rel[hi[norm] >= 1e-15].max() <= 1e-14
    # below, the rounding of the exponent q ~ -ln P inside exp: ~6 eps |ln P| measured
    assert np.all(rel <= 10 * EPS * np.maximum(1.0, -np.log(hi[norm])))
    assert np.all(np.abs(v[~norm] - hi[~norm]) <= 1e-300)


def _cond3(h, R):
    """sum over the six inputs of |d log P / d log x| (central differences, our own values)."""
    st_ = 1e-6
    c = np.zeros(len(h))
    for i in range(6):
        Hp, Hm, Rp, Rm = h.copy(), h.copy(), R.copy(), R.copy()
        if i < 3:
            Hp[:, i] *= 1 + st_
            Hm[:, i] *= 1 - st_
        else:
            Rp[:, i - 3] *= 1 + st_
            Rm[:, i - 3] *= 1 - st_
        with np.errstate(divide="ignore", invalid="ignore"):
            c += np.abs((np.log(mv(corr(Rp), Hp)[0]) - np.log(mv(corr(Rm), Hm)[0])) / (2 * st_))
    return c


def test_tvn_reference():
    z = np.load(REF / "tvn.npz")
    h, R, hi, lo = z["h"], z["r"], z["hi"], z["lo"]
    v, e = mv(corr(R), h)
    assert np.all(e == 0)
    norm = hi >= 1e-300
    h, R, v, hi, lo = h[norm], R[norm], v[norm], hi[norm], lo[norm]
    rel = rel_err(v, hi, lo)
    negs = (R < 0).sum(axis=1)
    well = _cond3(h, R)
    # the measured regions of 1e-14: at most one negative correlation and P >= 1e-15
    # (6.6e-15); any correlations with P >= 1e-10 and a condition number <= 100 (7.7e-15)
    assert rel[(hi >= 1e-15) & (negs <= 1)].max() <= 1e-14
    assert rel[(hi >= 1e-10) & (well <= 100)].max() <= 1e-14
    # everywhere: within a few rounding errors of the inputs, u cond (measured max 3.4)
    fin = np.isfinite(well)  # (a perturbed value that underflows has no log)
    assert fin.mean() > 0.95
    assert np.all(rel[fin] <= 8 * 2.0**-53 * np.maximum(well[fin], 1.0))


@pytest.mark.parametrize("name", ["bvt", "tvt"])
def test_t_exact_reference(name):
    z = np.load(REF / f"{name}.npz")
    h, R, nu, hi, lo = z["h"], z["r"], z["nu"], z["hi"], z["lo"]
    d = h.shape[1]
    S = np.tile(np.eye(d), (len(h), 1, 1))
    for q, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))[: 1 if d == 2 else 3]):
        S[:, a, b] = S[:, b, a] = R[:, q]
    v, e = mv(S, h, df=nu)
    assert np.all(e == 0)
    rel = rel_err(v, hi, lo)
    assert rel.max() <= T_EXACT_REL[name]


T_EXACT_REL = {"bvt": 5e-15, "tvt": 1e-14}  # measured 9.2e-16 and 1.9e-15


def test_symmetry_and_scale():
    rng = np.random.default_rng(3)
    for d in (2, 3):
        A = rng.normal(size=(d, d))
        S = A @ A.T + 0.3 * np.eye(d)
        x = rng.normal(size=(20, d))
        D = np.array([0.5, 3.0, 7.0][:d])
        v1, _ = mv(S, x)
        v2, _ = mv(S * D[:, None] * D[None, :], x * D)
        np.testing.assert_allclose(v2, v1, rtol=4e-15, atol=0)
        p = rng.permutation(d)
        v3, _ = mv(S[np.ix_(p, p)], x[:, p])
        np.testing.assert_allclose(v3, v1, rtol=1e-14, atol=0)


def test_bvn_limits():
    # r -> +-1: Phi(min(h, k)) and max(0, Phi(h) + Phi(k) - 1)
    for h, k in ((0.3, -0.2), (-3.0, -3.0), (2.0, -1.0), (-5.0, 4.0)):
        for r, ref in ((1 - 1e-12, sp.ndtr(min(h, k))), (-1 + 1e-12, max(0.0, sp.ndtr(h) + sp.ndtr(k) - 1))):
            v, _ = mv([[1, r], [r, 1]], [h, k])
            assert abs(v[0] - ref) <= 1e-6 * max(ref, 1e-300) + 1e-6


# ---------------------------------------------------------------- QMC


def test_factor_normal():
    z = np.load(REF / "factor_n.npz")
    h, lam, d, ref = z["h"], z["lam"], z["d"], z["hi"]
    for atol in (1e-6, 1e-4):
        for i in range(len(d)):
            n = d[i]
            S = np.outer(lam[i, :n], lam[i, :n])
            np.fill_diagonal(S, 1.0)
            v, e = mv(S, h[i, :n], atol=atol)
            # the estimate is statistical (3 standard errors over 12 shifts): measured, the
            # actual error exceeds it at up to ~7% of the points, by at most 1.9x
            assert e[0] <= atol
            assert abs(v[0] - ref[i]) <= 1.5 * atol
            assert abs(v[0] - ref[i]) <= 2 * e[0] + 1e-12


def test_factor_t():
    z = np.load(REF / "factor_t.npz")
    h, lam, d, nu, ref = z["h"], z["lam"], z["d"], z["nu"], z["hi"]
    for i in range(len(d)):
        n = d[i]
        S = np.outer(lam[i, :n], lam[i, :n])
        np.fill_diagonal(S, 1.0)
        if n <= 3:  # the exact mixture: as good as the quadrature reference (~1e-13)
            v, e = mv(S, h[i, :n], df=nu[i])
            assert e[0] == 0 and abs(v[0] - ref[i]) <= 1e-13 * ref[i]
        elif i % 3 == 0:  # a third of the QMC points (the slow ones)
            v, e = mv(S, h[i, :n], df=nu[i], atol=1e-5)
            assert e[0] <= 1e-5
            assert abs(v[0] - ref[i]) <= 1.5e-5
            assert abs(v[0] - ref[i]) <= 2 * e[0] + 1e-10


def test_t_against_scipy():
    rng = np.random.default_rng(11)
    for d in (2, 4):
        A = rng.normal(size=(d, d))
        S = A @ A.T + 0.5 * np.eye(d)
        x = rng.normal(size=(2, d))
        v, e = mv(S, x, df=5.0, atol=1e-5)
        for i in range(2):
            ref = st.multivariate_t(np.zeros(d), S, df=5.0).cdf(x[i], maxpts=200_000, random_state=1)
            assert abs(v[i] - ref) <= 3e-4  # scipy's own error (one lattice, no error control)


def test_normal_against_scipy_d5():
    rng = np.random.default_rng(12)
    A = rng.normal(size=(5, 5))
    S = A @ A.T + 0.5 * np.eye(5)
    x = rng.normal(size=(3, 5))
    v, e = mv(S, x)
    ref = st.multivariate_normal.cdf(x, np.zeros(5), S, abseps=1e-8, releps=0, maxpts=5_000_000, rng=1)
    np.testing.assert_allclose(v, ref, atol=2e-6, rtol=0)


def test_df_inf_is_normal():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(5, 5))
    S = A @ A.T + np.eye(5)
    x = rng.normal(size=(4, 5))
    np.testing.assert_array_equal(mv(S, x, df=INF)[0], mv(S, x)[0])


def test_max_points():
    rng = np.random.default_rng(5)
    A = rng.normal(size=(6, 6))
    S = A @ A.T + np.eye(6)
    x = rng.normal(size=6) + 1
    v, e = mv(S, x, atol=1e-12, max_points=20_000)
    assert e[0] > 1e-12  # stopped by the cap: the error says so
    v2, e2 = mv(S, x, atol=1e-6)
    assert abs(v[0] - v2[0]) <= e[0] + e2[0]


# ---------------------------------------------------------------- limits and invalid input


def test_infinite_limits():
    rng = np.random.default_rng(6)
    A = rng.normal(size=(5, 5))
    S = A @ A.T + np.eye(5)
    x = rng.normal(size=5)
    # +inf drops the variable: the marginal, here a 3-d exact one
    y = x.copy()
    y[[1, 3]] = INF
    keep = [0, 2, 4]
    v, e = mv(S, y)
    v3, e3 = mv(S[np.ix_(keep, keep)], x[keep])
    assert v[0] == v3[0] and e[0] == 0 and e3[0] == 0
    for df in (INF, 3.0):
        z = np.full(5, INF)
        z[2] = x[2]
        v, e = mv(S, z, df=df)
        ref = sp.ndtr(x[2] / math.sqrt(S[2, 2])) if df == INF else sp.stdtr(df, x[2] / math.sqrt(S[2, 2]))
        assert abs(v[0] - ref) <= 1e-15 and e[0] == 0
        assert tuple(mv(S, np.full(5, INF), df=df)[0]) == (1.0,)
        w = x.copy()
        w[4] = -INF
        assert tuple(mv(S, w, df=df)) == ((0.0,), (0.0,))
        w[0] = INF
        assert mv(S, w, df=df)[0][0] == 0.0


def test_nan_and_invalid():
    S = np.array([[1.0, 0.5, 0.2], [0.5, 1.0, 0.1], [0.2, 0.1, 1.0]])
    x = np.array([0.1, 0.2, 0.3])
    w = x.copy()
    w[1] = np.nan
    assert np.isnan(mv(S, w)[0][0]) and np.isnan(mv(S, w)[1][0])
    w[0] = -INF  # NaN wins over -inf
    assert np.isnan(mv(S, w)[0][0])
    bad = [S.copy() for _ in range(4)]
    bad[0][0, 1] = bad[0][1, 0] = 1.0  # singular (rho = 1)
    bad[1][2, 2] = -1.0
    bad[2][1, 0] = np.nan
    bad[3] = np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1]])  # indefinite
    for B in bad:
        for df in (INF, 4.0):
            for xx in (x, np.r_[x[:2], INF], np.r_[x[:2], -INF]):
                v, e = mv(B, xx, df=df)
                assert np.isnan(v[0]) and np.isnan(e[0])
    for df in (0.0, -1.0, np.nan):
        assert np.isnan(mv(S, x, df=df)[0][0])
    # the upper triangle is not read (LAPACK potrf 'L')
    U = S.copy()
    U[0, 2] = 99.0
    assert mv(U, x)[0][0] == mv(S, x)[0][0]


def test_argument_errors():
    S = sd.asarray(np.eye(2)[None])
    x = sd.asarray(np.zeros((1, 2)))
    b = sd.asarray(np.zeros(1, np.int64))
    df = sd.asarray(np.array([INF]))
    with pytest.raises(ValueError):
        C.mv_cdf(S, x, b, df, 0.0, 0)
    with pytest.raises(ValueError):
        C.mv_cdf(S, x, b, df, 1e-6, -1)
    with pytest.raises(IndexError):
        C.mv_cdf(S, x, sd.asarray(np.array([1], np.int64)), df, 1e-6, 0)
    with pytest.raises(TypeError):
        C.mv_cdf(S, x, sd.asarray(np.zeros(1)), df, 1e-6, 0)
    with pytest.raises(ValueError):
        C.mv_cdf(S, sd.asarray(np.zeros((1, 3))), b, df, 1e-6, 0)


# ---------------------------------------------------------------- determinism


def _mixed_batch():
    rng = np.random.default_rng(9)
    covs, xs, bt, dfs = [], [], [], []
    for i, (d, df) in enumerate([(2, INF), (3, INF), (5, INF), (7, INF), (3, 4.0), (4, 2.5)]):
        A = rng.normal(size=(d, d))
        S = np.zeros((7, 7))
        S[:d, :d] = A @ A.T + np.eye(d)
        S[d:, d:] = np.eye(7 - d)
        covs.append(S)
        dfs.append(df)
        for _ in range(4):
            x = np.full(7, INF)  # the padding drops out
            x[:d] = rng.normal(size=d)
            xs.append(x)
            bt.append(i)
    return np.array(covs), np.array(xs), np.array(bt), np.array(dfs)


def test_thread_count_and_layout_do_not_change_bits():
    S, X, bt, df = _mixed_batch()
    old = sd.get_num_threads()
    try:
        outs = []
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            outs.append(mv(S, X, df=df, atol=1e-5, batch=bt))
            # one point per call: its lattice runs on the pool instead of the points
            one = [mv(S[bt[i]], X[i], df=df[bt[i]], atol=1e-5) for i in range(len(X))]
            outs.append((np.concatenate([o[0] for o in one]), np.concatenate([o[1] for o in one])))
        for v, e in outs[1:]:
            np.testing.assert_array_equal(v, outs[0][0])
            np.testing.assert_array_equal(e, outs[0][1])
    finally:
        sd.set_num_threads(old)


def test_repeatable():
    S, X, bt, df = _mixed_batch()
    a = mv(S, X, df=df, atol=1e-5, batch=bt)
    b = mv(S, X, df=df, atol=1e-5, batch=bt)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
