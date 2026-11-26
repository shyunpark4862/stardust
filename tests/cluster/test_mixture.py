"""Gaussian mixtures (src/stardust/cluster/_mixture.py, csrc/cluster/mixture.cpp; docs/api/cluster.md §5;
docs/design/cluster.md §2.5, §3.7, §4.5; decisions CL32-CL35, CL41, CL43 and CL75-CL81): the log densities and
responsibilities at given factors, the factors of given covariances and the M step at given responsibilities against
the exact references (tools/cluster_ref.py mixture) and their contract, scikit-learn's fixed point from the same start,
the stop, the starts and rng's two words, the thread counts and the input order, sample, nan_policy, the scale and
the errors."""

import math
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from .._bits import bits_equal
from . import refdata as D
from . import samples as S
from .conftest import LIVE

CL = sd.cluster
CORE = sd._core.cluster
REF = D.ref("mixture")
A = sd.asarray
TYPES = ("full", "tied", "diag", "spherical")
CODE = {t: i for i, t in enumerate(TYPES)}


def _np(a):
    return np.asarray(a)


def _rng(seed=0):
    return sd.random.default_rng(seed)


def _per(name, mid, kind):
    pos, idx = D.at(REF, name, mid, kind)
    o = np.argsort(idx, kind="stable")
    m = REF.methods[name]
    return m.hi[pos[o]], m.lo[pos[o]]


def _shape(ctype, k, p):
    return {"full": (k, p, p), "tied": (p, p), "diag": (k, p), "spherical": (k,)}[ctype]


def _factors(mid):
    mx = S.mixture(mid)
    g = D.group_of(REF, mid)
    return np.asarray(REF.group_inputs["prec_chol"][g], float).reshape(_shape(mx.covariance_type, mx.k, mx.p))


def _logdet(ctype, U, k, p):
    if ctype in ("full", "tied"):
        blocks = U if ctype == "full" else np.repeat(U[None], k, axis=0)
        return np.array([np.sum(np.log(np.diag(B))) for B in blocks])
    if ctype == "diag":
        return np.sum(np.log(U), axis=1)
    return p * np.log(U)


def _blobs(n=600, p=3, k=3, seed=1, scale=1.0):
    g = np.random.default_rng(seed)
    centers = 5.0 * g.standard_normal((k, p))
    lab = g.integers(0, k, n)
    return centers[lab] + scale * g.standard_normal((n, p))


def _precisions(res):
    U = _np(res.precisions_cholesky)
    t = res.covariance_type
    if t == "full":
        return np.einsum("kij,klj->kil", U, U)
    if t == "tied":
        return U @ U.T
    return U**2


# ---- the contract at given parameters (§4.5) ------------------------------------------------------------------


def test_log_densities_and_responsibilities_at_given_factors():
    worst = {"logdens": 0.0, "score": 0.0, "resp": 0.0}
    for mid in S.mixture_ids():
        mx = S.mixture(mid)
        k, p, t = mx.k, mx.p, mx.covariance_type
        U = _factors(mid)
        ld = _logdet(t, U, k, p)
        lab, resp, lpn = CORE.gmm_estimate(A(mx.X), A(mx.means), A(np.ascontiguousarray(U)), A(ld),
                                           A(np.log(mx.weights)), CODE[t], 0)  # fmt: skip
        wlp = np.column_stack([sum(_per(f"logdens_u_{j}", mid, 0)) + math.log(mx.weights[j]) for j in range(k)])
        bE, bs, br = D.mixture_bounds(t, mx.X, mx.weights, mx.means, U, wlp)
        sh, sl = _per("score_u", mid, 0)
        err = np.abs(_np(lpn) - sh - sl)
        assert np.all(err <= bs), (mid, np.max(err / bs))
        worst["score"] = max(worst["score"], float(np.max(err / bs)))
        rr = np.column_stack([sum(_per(f"resp_u_{j}", mid, 0)) for j in range(k)])
        err = np.abs(_np(resp) - rr)
        assert np.all(err <= br), (mid, np.max(err / br))
        worst["resp"] = max(worst["resp"], float(np.max(err / br)))
        assert np.array_equal(_np(lab), np.argmax(wlp, axis=1)), mid  # no near tie in these sets
        # a component's log density: score_samples of a one-component mixture is that density
        for j in range(k):
            Uj = U[j : j + 1] if t != "tied" else U
            _, _, l1 = CORE.gmm_estimate(A(mx.X), A(mx.means[j : j + 1]), A(np.ascontiguousarray(Uj)),
                                         A(ld[j : j + 1]), A(np.zeros(1)), CODE[t], 0)  # fmt: skip
            ref = sum(_per(f"logdens_u_{j}", mid, 0))
            err = np.abs(_np(l1) - ref)
            assert np.all(err <= bE[:, j]), (mid, j, np.max(err / bE[:, j]))
            worst["logdens"] = max(worst["logdens"], float(np.max(err / bE[:, j])))
    assert all(0 < v < 1 for v in worst.values()), worst


def test_factors_of_given_covariances():
    for mid in S.mixture_ids():
        mx = S.mixture(mid)
        k, p, t = mx.k, mx.p, mx.covariance_type
        Ug, ld = CORE.gmm_precisions(A(np.ascontiguousarray(mx.covariances)), CODE[t], k, p)
        h, lo = _per("prec_chol", mid, 2)
        ref = (h + lo).reshape(_shape(t, k, p))
        b = D.precision_bound(t, mx.covariances, ref, k, p)
        err = np.abs(_np(Ug) - ref)
        assert np.all(err <= b), (mid, np.max(err / np.maximum(b, 1e-300)))
        assert np.allclose(_np(ld), _logdet(t, ref, k, p), rtol=1e-12, atol=1e-14)
    # a covariance that is not positive definite has no factor
    assert CORE.gmm_precisions(A(np.array([[[1.0, 2.0], [2.0, 1.0]]])), 0, 1, 2) is None
    assert CORE.gmm_precisions(A(np.array([[1.0, 0.0]])), 2, 1, 2) is None


def test_m_step_at_given_responsibilities():
    for mid in S.mixture_ids():
        mx = S.mixture(mid)
        k, t = mx.k, mx.covariance_type
        R = np.column_stack([_per(f"resp_{j}", mid, 0)[0] for j in range(k)])  # the rounded ones the reference took
        w, m, c = CORE.gmm_mstep(A(mx.X), A(R), CODE[t], 1e-6)
        got = np.concatenate([_np(w).ravel(), _np(m).ravel(), _np(c).ravel()])
        h, lo = _per("mstep", mid, 1)
        ref = h + lo
        b = D.mstep_bound(t, mx.X, R, ref)
        err = np.abs(got - ref)
        assert np.all(err <= b), (mid, int(np.argmax(err / b)), np.max(err / b))
        assert abs(float(np.sum(_np(w))) - 1.0) <= 4 * D.U


# ---- scikit-learn's fixed point from the same start ------------------------------------------------------------


@LIVE
@pytest.mark.parametrize("ctype", TYPES)
@pytest.mark.parametrize("init", ["kmeans", "random"])
def test_scikit_learns_fixed_point_from_the_same_start(ctype, init):
    from sklearn.mixture import GaussianMixture

    X = _blobs(900, 3, 3, seed=4)
    start = CL.gaussian_mixture(X, 3, rng=_rng(5), covariance_type=ctype, init=init, max_iter=0)
    w0, m0, P0 = _np(start.weights), _np(start.means), _precisions(start)
    kw = dict(covariance_type=ctype, weights_init=w0, means_init=m0, precisions_init=P0, tol=1e-9, max_iter=500)
    a = CL.gaussian_mixture(X, 3, rng=_rng(), **kw)
    b = GaussianMixture(3, **kw).fit(X)
    perm = _np(a._fit_perm)  # label -> the fit's component, which is scikit-learn's numbering here
    assert a.converged and b.converged_ and a.n_iter == b.n_iter_
    assert a.lower_bound == pytest.approx(b.lower_bound_, rel=1e-13)
    assert np.allclose(_np(a.weights), b.weights_[perm], rtol=1e-10, atol=0)
    assert np.allclose(_np(a.means), b.means_[perm], rtol=1e-10, atol=1e-12)
    cov_b = b.covariances_ if ctype == "tied" else b.covariances_[perm]
    assert np.allclose(_np(a.covariances), cov_b, rtol=1e-9, atol=1e-12)
    assert np.array_equal(perm[_np(a.labels)], b.predict(X))
    n = X.shape[0]
    assert a.loglik == pytest.approx(b.score(X) * n, rel=1e-12)
    assert a.bic == pytest.approx(b.bic(X), rel=1e-12) and a.aic == pytest.approx(b.aic(X), rel=1e-12)
    assert a.n_parameters == b._n_parameters()
    Xe = X[::37] + 0.5
    assert np.allclose(_np(a.score_samples(Xe)), b.score_samples(Xe), rtol=1e-11, atol=1e-11)
    assert np.allclose(_np(a.predict_proba(Xe)), b.predict_proba(Xe)[:, perm], rtol=0, atol=1e-11)
    assert np.array_equal(perm[_np(a.predict(Xe))], b.predict(Xe))


@LIVE
def test_scikit_learns_defaults_and_kmeans_start():
    from sklearn.mixture import GaussianMixture

    X = _blobs(2000, 4, 4, seed=7)
    a = CL.gaussian_mixture(X, 4, rng=_rng(1))
    b = GaussianMixture(4, random_state=0).fit(X)  # another k-means start: the same well separated fixed point
    assert a.converged and a.n_iter <= 100
    assert a.lower_bound == pytest.approx(b.lower_bound_, rel=1e-9)
    assert a.loglik == pytest.approx(b.score(X) * X.shape[0], rel=1e-9)
    # init='kmeans' is the labels of one kmeans with the start's two words (CL41, CL76)
    k0, k1 = _rng(1).integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,)).tolist()
    s0, s1 = CORE.subkey(k0, k1, 0)
    o = CORE.lex_order(A(X))
    km = CORE.kmeans(A(X), o, None, 4, 0, None, 1, 300, 1e-4, s0, s1, 0)
    r0 = CL.gaussian_mixture(X, 4, rng=_rng(1), max_iter=0)
    R = np.zeros((X.shape[0], 4))
    R[np.arange(X.shape[0]), _np(km[0][0])] = 1.0
    w, m, _ = CORE.gmm_mstep(A(X[_np(o)]), A(R), 0, 1e-6)
    perm = _np(r0._fit_perm)
    assert np.allclose(_np(r0.means), _np(m)[perm], rtol=1e-14, atol=1e-14)
    nk = R.sum(axis=0) + 10 * 2.0**-52
    assert np.allclose(_np(r0.weights), (nk / X.shape[0])[perm], rtol=1e-15)  # the start's weights: nk / n


# ---- the iteration ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ctype", TYPES)
def test_the_stop_the_labels_and_the_log_likelihood(ctype):
    X = _blobs(800, 2, 3, seed=2, scale=1.5)
    r = CL.gaussian_mixture(X, 3, rng=_rng(3), covariance_type=ctype, tol=1e-6)
    assert r.converged and 1 <= r.n_iter < 100
    prev = CL.gaussian_mixture(X, 3, rng=_rng(3), covariance_type=ctype, tol=1e-6, max_iter=r.n_iter - 1)
    if r.n_iter > 1:
        assert not prev.converged and prev.n_iter == r.n_iter - 1
        assert abs(r.lower_bound - prev.lower_bound) < 1e-6  # the stop (CL33)
    assert np.array_equal(_np(r.labels), _np(r.predict(X)))  # labels at the final parameters, as predict
    lp = _np(r.score_samples(X))
    assert r.loglik == pytest.approx(float(np.sum(lp)), rel=1e-13) and r.objective == r.loglik
    P = _np(r.predict_proba(X))
    assert np.allclose(P.sum(axis=1), 1.0, atol=1e-14) and np.array_equal(np.argmax(P, axis=1), _np(r.labels))
    assert np.sum(_np(r.weights)) == pytest.approx(1.0, abs=1e-15)
    k, p = 3, 2
    npar = (
        {"full": k * p * (p + 1) // 2, "tied": p * (p + 1) // 2, "diag": k * p, "spherical": k}[ctype] + k * p + k - 1
    )
    assert r.n_parameters == npar
    assert r.bic == -2 * r.loglik + npar * math.log(800) and r.aic == -2 * r.loglik + 2 * npar
    assert tuple(r.covariances.shape) == tuple(r.precisions_cholesky.shape) == _shape(ctype, k, p)
    s = r.summary()
    assert "Gaussian Mixture Results" in s and "Weight" in s and "Lower bound:" in s
    assert repr(r).startswith("GaussianMixtureResult(nobs=800, n_clusters=3, loglik=")


def test_max_iter_zero_keeps_the_start():
    X = _blobs(300, 2, 2, seed=3)
    r = CL.gaussian_mixture(X, 2, rng=_rng(), max_iter=0)
    assert r.n_iter == 0 and not r.converged and r.lower_bound == -math.inf
    w0, m0 = np.array([0.25, 0.75]), np.array([[0.0, 0.0], [1.0, 1.0]])
    P0 = np.array([np.eye(2) * 2.0, np.eye(2) * 0.5])
    r = CL.gaussian_mixture(X, 2, rng=_rng(), weights_init=w0, means_init=m0, precisions_init=P0, max_iter=0)
    perm = _np(r._fit_perm)
    assert np.array_equal(_np(r.weights), w0[perm]) and np.array_equal(_np(r.means), m0[perm])
    assert np.allclose(_precisions(r), P0[perm], rtol=1e-15)
    assert np.allclose(_np(r.covariances), np.linalg.inv(P0)[perm], rtol=1e-15)


def test_starts_rng_and_n_init():
    X = _blobs(500, 2, 4, seed=5, scale=2.0)
    g = _rng(8)
    st = g.state["position"]
    a = CL.gaussian_mixture(X, 4, rng=g, n_init=3, init="random")
    assert g.state["position"] == st + 2  # two words a call (CL41)
    b = CL.gaussian_mixture(X, 4, rng=_rng(8), n_init=3, init="random")
    assert bits_equal(a.means, b.means) and a.best_init == b.best_init and a.n_init == 3
    # the best start is the largest lower bound (the first on a tie)
    lbs = [CL.gaussian_mixture(X, 4, rng=_rng(8), n_init=s + 1, init="random").lower_bound for s in range(3)]
    assert a.lower_bound == max(lbs)
    for init in ("kmeans", "k-means++", "random", "random_from_data"):
        r = CL.gaussian_mixture(X, 4, rng=_rng(9), init=init)
        assert r.n_clusters == 4 and np.isfinite(r.loglik), init
    # k-means++'s start: one-hot responsibilities at the rows kmeans_plusplus picks
    k0, k1 = _rng(9).integers(-(2**63), 2**63 - 1, endpoint=True, shape=(2,)).tolist()
    o = CORE.lex_order(A(X))
    rows = _np(CORE.kmeans_plusplus(A(X), o, 4, k0, k1, 0))
    r0 = CL.gaussian_mixture(X, 4, rng=_rng(9), init="k-means++", max_iter=0)
    Xk = X[_np(o)]
    got = _np(r0.means)[np.argsort(_np(r0._fit_perm))]
    assert np.allclose(got, Xk[rows] / (1 + 10 * 2.0**-52), rtol=1e-15)
    # all three given: nothing drawn (the same fit for any rng)
    w0, m0, P0 = _np(a.weights), _np(a.means), _precisions(a)
    c = CL.gaussian_mixture(X, 4, rng=_rng(1), weights_init=w0, means_init=m0, precisions_init=P0)
    d = CL.gaussian_mixture(X, 4, rng=_rng(2), weights_init=w0, means_init=m0, precisions_init=P0, n_init=2)
    assert bits_equal(c.means, d.means) and bits_equal(c.covariances, d.covariances)


@pytest.mark.parametrize("ctype", TYPES)
def test_thread_counts_and_input_order(ctype):
    X = _blobs(12000, 4, 5, seed=11, scale=1.5)

    def results(Y, init="kmeans"):
        r = CL.gaussian_mixture(Y, 5, rng=_rng(4), covariance_type=ctype, init=init, n_init=2)
        return r, [r.weights, r.means, r.covariances, r.precisions_cholesky, r.loglik, r.lower_bound, r.n_iter,
                   r.score_samples(Y[:50]), r.predict_proba(Y[:50])]  # fmt: skip

    sd.set_num_threads(1)
    base, bv = results(X)
    try:
        for t in (3, 10):
            sd.set_num_threads(t)
            _, v = results(X)
            for q, (a, b) in enumerate(zip(bv, v)):
                assert bits_equal(a, b), (t, q)
    finally:
        sd.set_num_threads(10)
    perm = np.random.default_rng(0).permutation(len(X))
    for init in ("kmeans", "random", "random_from_data", "k-means++"):
        a, av = results(X, init)
        b, bw = results(X[perm], init)
        assert np.array_equal(S.first_seen(_np(a.labels)[perm]), _np(b.labels)), init
        # the same points in another order: the same fit (CL7), components in first-seen order
        q = S.first_seen(_np(a.labels)[perm])
        order_a = [int(_np(a.labels)[perm][np.nonzero(q == j)[0][0]]) for j in range(5)]
        assert np.array_equal(_np(a.means)[order_a], _np(b.means)) and a.loglik == b.loglik, init


def test_sample():
    X = _blobs(3000, 2, 3, seed=6)
    for ctype in TYPES:
        r = CL.gaussian_mixture(X, 3, rng=_rng(1), covariance_type=ctype)
        g = _rng(2)
        st = g.state["position"]
        s = r.sample((40000,), rng=g)
        assert g.state["position"] == st + 2
        assert tuple(s.X.shape) == (40000, 2) and tuple(s.labels.shape) == (40000,)
        lab, Y = _np(s.labels), _np(s.X)
        frac = np.bincount(lab, minlength=3) / 40000
        assert np.allclose(frac, _np(r.weights), atol=0.01), ctype
        for j in range(3):
            Yj = Y[lab == j]
            assert np.allclose(Yj.mean(axis=0), _np(r.means)[j], atol=0.08), (ctype, j)
            cov = {"full": lambda: _np(r.covariances)[j], "tied": lambda: _np(r.covariances),
                   "diag": lambda: np.diag(_np(r.covariances)[j]),
                   "spherical": lambda: np.eye(2) * _np(r.covariances)[j]}[ctype]()  # fmt: skip
            assert np.allclose(np.cov(Yj.T), cov, atol=0.1 * np.max(np.abs(cov))), (ctype, j)
    one = r.sample(rng=_rng(3))
    assert tuple(one.X.shape) == (2,) and isinstance(one.labels, int)
    grid = r.sample((2, 3), rng=_rng(3))
    assert tuple(grid.X.shape) == (2, 3, 2) and tuple(grid.labels.shape) == (2, 3)
    flat = r.sample((6,), rng=_rng(3))
    assert bits_equal(_np(grid.X).reshape(6, 2), flat.X)  # one stream element per draw
    sd.set_num_threads(1)
    try:
        a = r.sample((5000,), rng=_rng(4))
    finally:
        sd.set_num_threads(10)
    b = r.sample((5000,), rng=_rng(4))
    assert bits_equal(a.X, b.X) and bits_equal(a.labels, b.labels)


def test_nan_policy_omit():
    X = _blobs(400, 2, 2, seed=8)
    Xn = X.copy()
    Xn[[5, 77], 1] = np.nan
    with pytest.raises(ValueError, match=r"contains NaN"):
        CL.gaussian_mixture(Xn, 2, rng=_rng())
    r = CL.gaussian_mixture(Xn, 2, rng=_rng(), nan_policy="omit")
    keep = ~np.isnan(Xn).any(axis=1)
    b = CL.gaussian_mixture(X[keep], 2, rng=_rng())
    lab = _np(r.labels)
    assert lab[5] == -1 and lab[77] == -1 and r.nobs == 398 and int(np.sum(_np(r.omitted))) == 2
    assert np.array_equal(lab[keep], _np(b.labels)) and bits_equal(r.means, b.means)
    assert "Omitted rows (NaN):" in r.summary()


def test_scale_and_far_points():
    X = _blobs(600, 2, 3, seed=9)
    a = CL.gaussian_mixture(X, 3, rng=_rng(), reg_covar=0.0)
    for e in (-900, 600):  # CL43 (CL77): data out of [2^-250, 2^250] on a power-of-two grid
        b = CL.gaussian_mixture(X * 2.0**e, 3, rng=_rng(), reg_covar=0.0)
        assert np.array_equal(_np(a.labels), _np(b.labels)) and a.n_iter == b.n_iter, e
        assert np.allclose(_np(b.means), _np(a.means) * 2.0**e, rtol=1e-12, atol=0), e
        assert np.allclose(_np(b.precisions_cholesky), _np(a.precisions_cholesky) * 2.0**-e, rtol=1e-12), e
        assert b.loglik == pytest.approx(a.loglik - 600 * 2 * e * math.log(2), rel=1e-12), e
        assert np.array_equal(_np(b.predict(X[:20] * 2.0**e)), _np(a.labels)[:20])
    # far from the origin the difference form keeps the digits scikit-learn's expansion loses (diag, spherical)
    Xf = X + 1e8
    for ctype in ("diag", "spherical"):
        r = CL.gaussian_mixture(Xf, 3, rng=_rng(), covariance_type=ctype)
        x = [Fraction(v) for v in Xf[0]]
        out = []
        for j in range(3):
            mu = [Fraction(v) for v in _np(r._fit_means)[j]]
            U = _np(r._fit_U)[j]
            us = [Fraction(float(U[d] if ctype == "diag" else U)) for d in range(2)]
            q = sum(((x[d] - mu[d]) * us[d]) ** 2 for d in range(2))
            out.append(float(q))
        got = []
        for j in range(3):
            Uj = _np(r._fit_U)[j : j + 1]
            _, _, l1 = CORE.gmm_estimate(A(Xf[:1]), A(_np(r._fit_means)[j : j + 1]), A(np.ascontiguousarray(Uj)),
                                         A(np.zeros(1)), A(np.zeros(1)), CODE[ctype], 0)  # fmt: skip
            got.append(-2 * float(_np(l1)[0]) - 2 * math.log(2 * math.pi))
        assert np.allclose(got, out, rtol=1e-12, atol=1e-9), ctype
        # scikit-learn's expanded form at the same parameters (sum mu^2 lambda - 2 x mu lambda + x^2 lambda)
        from sklearn.mixture._gaussian_mixture import _estimate_log_gaussian_prob

        sk = _estimate_log_gaussian_prob(Xf[:1], _np(r._fit_means), _np(r._fit_U), ctype)[0]
        ld = _logdet(ctype, _np(r._fit_U), 3, 2)
        sk_q = -2 * (sk - ld) - 2 * math.log(2 * math.pi)
        assert np.max(np.abs(sk_q - np.array(out))) > 1e-3, ctype  # [sklearn과 다름]: the expansion cancels


def test_ill_defined_covariances_and_errors():
    from stardust.linalg import LinAlgError

    X = np.repeat(np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.5]]), 5, axis=0)
    with pytest.raises(LinAlgError, match=r"gaussian_mixture\(\): component \d has an ill-defined covariance; "
                       r"increase 'reg_covar' or check for duplicate points"):  # fmt: skip
        CL.gaussian_mixture(X, 3, rng=_rng(), reg_covar=0.0)
    with pytest.raises(LinAlgError, match=r"the tied covariance is ill-defined"):
        CL.gaussian_mixture(np.zeros((6, 2)) + [[1.0, 2.0]], 1, rng=_rng(), reg_covar=0.0, covariance_type="tied",
                            init="random")  # fmt: skip
    r = CL.gaussian_mixture(X, 3, rng=_rng())  # reg_covar keeps the duplicates well defined
    assert r.converged and np.all(_np(r.covariances)[:, 0, 0] >= 1e-6)


@pytest.mark.parametrize(
    "call, err, msg",
    [
        (lambda: CL.gaussian_mixture(np.eye(3), 2), TypeError, r"missing 1 required keyword-only argument: 'rng'"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), n_components=2), TypeError,
         r"unexpected keyword argument 'n_components'"),
        (lambda: CL.gaussian_mixture(np.eye(3), 4, rng=_rng()), ValueError,
         r"'k' \(4\) is larger than the number of points \(3\)"),
        (lambda: CL.gaussian_mixture(np.ones((1, 2)), 1, rng=_rng()), ValueError, r"at least 2 points"),
        (lambda: CL.gaussian_mixture(np.zeros((5, 2)), 2, rng=_rng()), ValueError,
         r"'k' \(2\) is larger than the number of distinct points \(1\)"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), covariance_type="general"), ValueError,
         r"'covariance_type' must be 'full', 'tied', 'diag' or 'spherical'"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), init="k-means"), ValueError,
         r"'init' must be 'kmeans', 'k-means\+\+', 'random' or 'random_from_data'"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), tol=-1.0), ValueError, r"'tol' must be finite"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), reg_covar=math.nan), ValueError, r"'reg_covar'"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), max_iter=-1), ValueError, r"'max_iter' must be >= 0"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), n_init=0), ValueError, r"'n_init' must be >= 1"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), weights_init=[0.5, 0.6]), ValueError,
         r"'weights_init' must sum to 1"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), weights_init=[1.5, -0.5]), ValueError,
         r"'weights_init' must be in \[0, 1\]"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), means_init=np.zeros((3, 3))), ValueError,
         r"'means_init' must have shape \(2, 3\)"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), covariance_type="diag",
                                     precisions_init=np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])), ValueError,
         r"'precisions_init' must be positive"),
        (lambda: CL.gaussian_mixture(np.eye(3), 1, rng=_rng(), covariance_type="tied",
                                     precisions_init=np.array([[1.0, 2.0, 0], [0, 1.0, 0], [0, 0, 1.0]])),
         ValueError, r"'precisions_init' must be symmetric"),
        (lambda: CL.gaussian_mixture(np.eye(3), 1, rng=_rng(), covariance_type="tied",
                                     precisions_init=np.array([[1.0, 2.0, 0], [2.0, 1.0, 0], [0, 0, 1.0]])),
         ValueError, r"'precisions_init' must be positive definite \(component 0 is not\)"),
        (lambda: CL.gaussian_mixture(np.eye(3), 2, rng=_rng(), nan_policy="propagate"), ValueError,
         r"'nan_policy' must be 'raise' or 'omit'"),
    ],
)  # fmt: skip
def test_errors(call, err, msg):
    with pytest.raises(err, match=msg):
        call()


def test_result_methods_errors():
    X = _blobs(200, 2, 2, seed=10)
    r = CL.gaussian_mixture(X, 2, rng=_rng())
    with pytest.raises(ValueError, match=r"predict\(\): 'X_new' must have 2 columns"):
        r.predict(np.zeros((3, 3)))
    with pytest.raises(ValueError, match=r"score_samples\(\): 'X_new' must be finite"):
        r.score_samples(np.array([[np.nan, 0.0]]))
    with pytest.raises(TypeError, match=r"sample\(\): missing 1 required keyword-only argument: 'rng'"):
        r.sample(3)
    with pytest.raises(TypeError, match=r"predict_proba\(\): missing 1 required positional argument"):
        r.predict_proba()
    with pytest.raises(AttributeError):
        r.weights = None
