"""Multivariate distributions (T9, docs/api/stats.md §3; decisions T29): densities and entropies
against 50-digit references and scipy, batching, the NaN and support rules, and samples (their
moments, determinism). The cdfs are tests/stats/test_mvcdf.py."""

import math

import mpmath as mp
import numpy as np
import pytest
import scipy.stats as st

import stardust as sd

S = sd.stats


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (q * np.exp(np.linspace(0.0, math.log(kappa), d))) @ q.T
    return (a + a.T) / 2.0


def _mixed(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))


def _ref_logpdf(mu, cov, xs, df=None):
    with mp.workdps(50):
        d = len(mu)
        A = mp.matrix(cov.tolist())
        ld = mp.log(mp.det(A))
        out = []
        for x in xs:
            r = mp.matrix([mp.mpf(float(x[i])) - mp.mpf(float(mu[i])) for i in range(d)])
            q = (r.T * mp.lu_solve(A, r))[0]
            if df is None:
                out.append(-(d * mp.log(2 * mp.pi) + ld + q) / 2)
            else:
                nu = mp.mpf(df)
                out.append(mp.loggamma((nu + d) / 2) - mp.loggamma(nu / 2) - d / 2 * mp.log(nu * mp.pi) - ld / 2
                           - (nu + d) / 2 * mp.log1p(q / nu))
        if df is None:
            h = d / mp.mpf(2) * (1 + mp.log(2 * mp.pi)) + ld / 2
        else:
            nu = mp.mpf(df)
            a, hh = nu / 2, mp.mpf(d) / 2
            h = (ld / 2 + mp.loggamma(a) - mp.loggamma(a + hh) + hh * mp.log(nu * mp.pi)
                 + (a + hh) * (mp.digamma(a + hh) - mp.digamma(a)))
        return np.array([float(v) for v in out]), float(h)


@pytest.mark.parametrize("df", [None, 0.7, 4.5, 1e9])
@pytest.mark.parametrize(("d", "kappa"), [(1, 1.0), (2, 1e3), (5, 1e6), (12, 1e9), (40, 1e9)])
def test_logpdf_and_entropy_against_50_digits(df, d, kappa):
    rng = np.random.default_rng(d * 7 + int(math.log10(kappa)))
    cov = spd(rng, d, kappa)
    mu = rng.normal(size=d) * 3.0
    xs = mu + rng.normal(size=(5, d)) @ np.linalg.cholesky(cov).T * 2.0
    ref, href = _ref_logpdf(mu, cov, xs, df)
    if df is None:
        D = S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov))
    else:
        D = S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=df)
    assert _mixed(D.logpdf(sd.asarray(xs)), ref) <= 1e-14
    assert np.max(np.abs(np.asarray(D.pdf(sd.asarray(xs))) / np.exp(ref) - 1.0)) <= 1e-14 * max(1.0, np.abs(ref).max())
    assert _mixed(D.entropy(), href) <= 1e-14


def test_matches_scipy_on_easy_cases():
    rng = np.random.default_rng(1)
    cov = spd(rng, 3, 10.0)
    mu = rng.normal(size=3)
    xs = rng.normal(size=(7, 3))
    _close = lambda a, b: np.testing.assert_allclose(np.asarray(a, float), b, rtol=1e-13, atol=1e-14)
    _close(S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)).logpdf(sd.asarray(xs)),
           st.multivariate_normal(mu, cov).logpdf(xs))
    _close(S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=3.0).logpdf(sd.asarray(xs)),
           st.multivariate_t(mu, cov, df=3.0).logpdf(xs))
    p = np.array([0.2, 0.3, 0.5])
    x = np.array([[2, 3, 5], [0, 0, 10], [10, 0, 0], [1, 1, 8]], float)
    _close(S.Multinomial(n=10.0, p=sd.asarray(p)).logpmf(sd.asarray(x)), st.multinomial(10, p).logpmf(x))
    _close(S.Multinomial(n=10.0, p=sd.asarray(p)).entropy(), st.multinomial(10, p).entropy())
    a = np.array([0.5, 2.0, 3.0])
    xd = rng.dirichlet(a, size=5)
    _close(S.Dirichlet(alpha=sd.asarray(a)).logpdf(sd.asarray(xd)), st.dirichlet(a).logpdf(xd.T))
    _close(S.Dirichlet(alpha=sd.asarray(a)).entropy(), st.dirichlet(a).entropy())


def test_multinomial_and_dirichlet_against_50_digits():
    # large n: log n! and the expectations of log X_i! cancel from 1e6 to 12.6
    with mp.workdps(50):
        n, pf = 100000, [0.2, 0.3, 0.5]
        h = -mp.loggamma(n + 1) - n * sum(mp.mpf(q) * mp.log(mp.mpf(q)) for q in pf)
        for q in pf:
            q = mp.mpf(q)
            mu, s = n * float(q), math.sqrt(n * float(q) * (1 - float(q)))
            lb = lambda k: mp.loggamma(n + 1) - mp.loggamma(k + 1) - mp.loggamma(n - k + 1) + k * mp.log(q) + \
                (n - k) * mp.log(1 - q)
            h += mp.fsum(mp.exp(lb(k)) * mp.loggamma(k + 1) for k in range(int(mu - 40 * s), int(mu + 40 * s)))
        assert abs(S.Multinomial(n=float(n), p=sd.asarray(pf)).entropy() - float(h)) <= 1e-14 * abs(float(h))
        a = [mp.mpf(1e6), mp.mpf(2e6), mp.mpf(3e6)]
        a0 = sum(a)
        hd = sum(mp.loggamma(x) for x in a) - mp.loggamma(a0) + (a0 - 3) * mp.digamma(a0) - \
            sum((x - 1) * mp.digamma(x) for x in a)
        assert abs(S.Dirichlet(alpha=sd.asarray([1e6, 2e6, 3e6])).entropy() - float(hd)) <= 1e-14 * abs(float(hd))
        # a large-count log pmf: n p_i summing to a value far from 1 in log terms
        x = [20000.0, 30000.0, 50000.0]
        lp = mp.loggamma(n + 1) - sum(mp.loggamma(v + 1) for v in x) + sum(v * mp.log(mp.mpf(q)) for v, q in zip(x, pf))
        got = S.Multinomial(n=float(n), p=sd.asarray(pf)).logpmf(sd.asarray(x))
        assert abs(got - float(lp)) <= 1e-14 * max(1.0, abs(float(lp)))


def test_batching():
    mu = sd.asarray(np.zeros((3, 2)))
    cov = sd.asarray(np.eye(2) * np.array([1.0, 2.0, 3.0])[:, None, None])
    D = S.MultivariateNormal(mu=mu, cov=cov)
    x = sd.asarray(np.ones((5, 1, 2)))
    v = np.asarray(D.logpdf(x))
    assert v.shape == (5, 3)
    for j, s2 in enumerate((1.0, 2.0, 3.0)):
        assert np.allclose(v[:, j], st.multivariate_normal(np.zeros(2), np.eye(2) * s2).logpdf(np.ones(2)), rtol=1e-14)
    assert np.asarray(D.entropy()).shape == (3,)
    assert np.asarray(D.mean()).shape == (3, 2) and np.asarray(D.covariance()).shape == (3, 2, 2)
    assert np.asarray(D.sample(4, rng=sd.random.default_rng(1))).shape == (4, 3, 2)
    assert isinstance(S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray(np.eye(2))).logpdf(
        sd.asarray([1.0, 1.0])), float)
    T = S.MultivariateT(mu=sd.asarray([0.0, 0.0]), shape=sd.asarray(np.eye(2)), df=sd.asarray([1.0, 5.0, np.inf]))
    assert np.asarray(T.logpdf(sd.asarray([0.5, 0.5]))).shape == (3,)
    M = S.Multinomial(n=sd.asarray([5.0, 10.0]), p=sd.asarray([0.5, 0.5]))
    assert np.allclose(np.asarray(M.pmf(sd.asarray([2.0, 3.0]))), [st.binom(5, 0.5).pmf(2), 0.0])
    with pytest.raises(ValueError, match="broadcast"):
        S.MultivariateNormal(mu=sd.asarray(np.zeros((3, 2))), cov=sd.asarray(np.tile(np.eye(2), (4, 1, 1))))
    with pytest.raises(ValueError, match="components"):
        D.logpdf(sd.asarray([1.0, 2.0, 3.0]))


def test_invalid_parameters_give_nan():
    bad = S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray([[1.0, 2.0], [2.0, 1.0]]))
    assert math.isnan(bad.logpdf(sd.asarray([0.0, 0.0]))) and math.isnan(bad.entropy())
    assert np.isnan(np.asarray(bad.mean())).all() and np.isnan(np.asarray(bad.sample(2, rng=sd.random.default_rng(1)))).all()
    assert math.isnan(S.MultivariateNormal(mu=sd.asarray([np.nan, 0.0]), cov=sd.asarray(np.eye(2))).logpdf(
        sd.asarray([0.0, 0.0])))
    assert math.isnan(S.MultivariateT(mu=sd.asarray([0.0]), shape=sd.asarray([[1.0]]), df=-1.0).entropy())
    assert math.isnan(S.Multinomial(n=2.5, p=sd.asarray([0.5, 0.5])).pmf(sd.asarray([1.0, 1.5])))
    assert math.isnan(S.Multinomial(n=2.0, p=sd.asarray([0.5, 0.6])).entropy())  # p sums to 1.1
    assert not math.isnan(S.Multinomial(n=2.0, p=sd.asarray([0.5, 0.5 + 1e-12])).entropy())  # within sqrt(eps)
    assert math.isnan(S.Dirichlet(alpha=sd.asarray([1.0, 0.0])).entropy())


def test_support_rules():
    D = S.Dirichlet(alpha=sd.asarray([2.0, 3.0, 4.0]))
    assert D.pdf(sd.asarray([0.5, 0.6, -0.1])) == 0.0
    assert D.pdf(sd.asarray([0.5, 0.3, 0.3])) == 0.0  # sum 1.1
    assert D.pdf(sd.asarray([0.2, 0.3, 0.5])) > 0.0
    assert D.logpdf(sd.asarray([0.0, 0.5, 0.5])) == -math.inf  # alpha_0 > 1 at x_0 = 0
    assert S.Dirichlet(alpha=sd.asarray([0.5, 3.0])).pdf(sd.asarray([0.0, 1.0])) == math.inf
    M = S.Multinomial(n=4.0, p=sd.asarray([0.25, 0.75]))
    assert M.pmf(sd.asarray([1.0, 2.0])) == 0.0  # sum 3
    assert M.pmf(sd.asarray([1.5, 2.5])) == 0.0
    assert S.Multinomial(n=4.0, p=sd.asarray([0.0, 1.0])).pmf(sd.asarray([0.0, 4.0])) == 1.0
    N = S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray(np.eye(2)))
    assert N.pdf(sd.asarray([math.inf, 0.0])) == 0.0


def test_moments():
    T = S.MultivariateT(mu=sd.asarray([1.0, 2.0]), shape=sd.asarray([[2.0, 0.5], [0.5, 1.0]]),
                        df=sd.asarray([0.5, 1.5, 4.0, np.inf]))
    m = np.asarray(T.mean())
    assert np.isnan(m[0]).all() and np.allclose(m[1:], [1.0, 2.0])
    c = np.asarray(T.covariance())
    assert np.isnan(c[0]).all() and np.isinf(c[1]).all() and np.allclose(c[2], np.array([[2, .5], [.5, 1]]) * 2.0)
    assert np.allclose(c[3], [[2, 0.5], [0.5, 1]])
    p = np.array([0.2, 0.3, 0.5])
    M = S.Multinomial(n=10.0, p=sd.asarray(p))
    assert np.allclose(np.asarray(M.mean()), 10 * p)
    assert np.allclose(np.asarray(M.covariance()), st.multinomial(10, p).cov())
    a = np.array([0.5, 2.0, 3.0])
    Dr = S.Dirichlet(alpha=sd.asarray(a))
    assert np.allclose(np.asarray(Dr.mean()), st.dirichlet(a).mean())
    assert np.allclose(np.asarray(Dr.covariance()), st.dirichlet(a).cov())


def test_samples():
    rng = np.random.default_rng(3)
    cov = spd(rng, 3, 20.0)
    mu = np.array([1.0, -2.0, 0.5])
    N = 200000
    for D, m_true, c_true in [
        (S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)), mu, cov),
        (S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=8.0), mu, cov * 8 / 6),
    ]:
        x = np.asarray(D.sample(N, rng=sd.random.default_rng(5)))
        assert x.shape == (N, 3)
        assert np.all(np.abs(x.mean(0) - m_true) < 5 * np.sqrt(np.diag(c_true) / N))
        assert np.allclose(np.cov(x.T), c_true, rtol=0.03, atol=0.03)
        # a 1-D projection follows its law exactly: a' X is normal (t) with a' cov a
        a = np.array([0.3, -0.5, 1.0])
        proj = (x - m_true) @ a / math.sqrt(a @ cov @ a)
        law = st.norm() if isinstance(D, S.MultivariateNormal) else st.t(8.0)
        assert st.kstest(proj, law.cdf).pvalue > 1e-4
    p = np.array([0.2, 0.3, 0.5])
    xm = np.asarray(S.Multinomial(n=30.0, p=sd.asarray(p)).sample(N, rng=sd.random.default_rng(6)))
    assert np.all(xm.sum(1) == 30) and np.all(xm == np.floor(xm))
    assert np.all(np.abs(xm.mean(0) - 30 * p) < 5 * np.sqrt(30 * p * (1 - p) / N))
    # each count is binomial: chi square on the first
    counts = np.bincount(xm[:, 0].astype(int), minlength=31)
    expected = st.binom(30, 0.2).pmf(np.arange(31)) * N
    keep = expected > 5
    assert st.chisquare(counts[keep], expected[keep] * counts[keep].sum() / expected[keep].sum()).pvalue > 1e-4
    a = np.array([0.5, 2.0, 3.0])
    xd = np.asarray(S.Dirichlet(alpha=sd.asarray(a)).sample(N, rng=sd.random.default_rng(7)))
    assert np.allclose(xd.sum(1), 1.0, atol=1e-14)
    assert st.kstest(xd[:, 0], st.beta(a[0], a.sum() - a[0]).cdf).pvalue > 1e-4  # marginals are beta
    # tiny concentrations: no NaN (gammas in logs)
    xt = np.asarray(S.Dirichlet(alpha=sd.asarray([1e-3, 1e-3])).sample(1000, rng=sd.random.default_rng(8)))
    assert np.all(np.isfinite(xt)) and np.allclose(xt.sum(1), 1.0)


def test_samples_are_deterministic():
    D = S.MultivariateT(mu=sd.asarray([0.0, 0.0]), shape=sd.asarray([[1.0, 0.3], [0.3, 2.0]]), df=3.0)
    before = sd.get_num_threads()
    try:
        outs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            outs.append(np.asarray(D.sample(20000, rng=sd.random.default_rng(1))))
    finally:
        sd.set_num_threads(before)
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])
    rng = sd.random.default_rng(2)
    a = np.asarray(D.sample(100, rng=rng))
    b = np.asarray(D.sample(100, rng=rng))
    whole = np.asarray(D.sample(200, rng=sd.random.default_rng(2)))
    assert np.array_equal(np.concatenate([a, b]), whole)
    assert rng.state["position"] == 200


def test_argument_rules():
    with pytest.raises(TypeError):
        S.MultivariateNormal(sd.asarray([0.0]), sd.asarray([[1.0]]))  # keyword-only
    with pytest.raises(ValueError, match="last two axes"):
        S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray(np.eye(3)))
    with pytest.raises(TypeError, match="Generator"):
        S.Dirichlet(alpha=sd.asarray([1.0, 2.0])).sample(3, rng=1)
    D = S.MultivariateNormal(mu=sd.asarray([0.0, 1.0]), cov=sd.asarray(np.eye(2)))
    assert repr(D).startswith("MultivariateNormal(mu=")
    assert D.mu is not None and D.cov is not None


def test_cdf_through_the_classes():
    # the kernels' accuracy is tests/stats/test_mvcdf.py; here the wrapper: shapes, mu, the
    # error output and the options
    N = S.MultivariateNormal(mu=sd.asarray([1.0, -1.0]), cov=sd.asarray([[2.0, 0.6], [0.6, 1.0]]))
    x = np.array([[1.5, -0.2], [0.0, 0.0]])
    ref = st.multivariate_normal([1.0, -1.0], [[2.0, 0.6], [0.6, 1.0]]).cdf(x)
    np.testing.assert_allclose(np.asarray(N.cdf(sd.asarray(x))), ref, rtol=1e-6)
    v, e = N.cdf(sd.asarray(x), return_error=True)
    assert np.all(np.asarray(e) == 0.0)  # d = 2: the exact formula
    assert isinstance(N.cdf(sd.asarray([1.0, 1.0])), float)
    B = S.MultivariateNormal(mu=sd.asarray(np.zeros((3, 2))), cov=sd.asarray(np.eye(2) * np.array([1.0, 2.0, 3.0])[:, None, None]))
    assert B.cdf(sd.asarray([[0.0, 0.0]])).shape == (3,)
    assert np.allclose(np.asarray(B.cdf(sd.asarray([0.0, 0.0]))), 0.25)
    T = S.MultivariateT(mu=sd.asarray(np.zeros(5)), shape=sd.asarray(np.eye(5) + 0.3), df=6.0)
    v, e = T.cdf(sd.asarray(np.ones(5)), atol=1e-4, return_error=True)
    assert 0.0 < e <= 1e-4 and 0.0 < v < 1.0
    with pytest.raises(ValueError, match="atol"):
        N.cdf(sd.asarray([0.0, 0.0]), atol=0.0)
    with pytest.raises(ValueError, match="max_points"):
        N.cdf(sd.asarray([0.0, 0.0]), max_points=0)
    bad = S.MultivariateNormal(mu=sd.asarray([0.0, 0.0]), cov=sd.asarray([[1.0, 2.0], [2.0, 1.0]]))
    assert math.isnan(bad.cdf(sd.asarray([0.0, 0.0])))
