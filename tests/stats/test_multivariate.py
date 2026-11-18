"""Multivariate distributions (T9, docs/api/stats.md §3; decisions T29): densities and entropies
against Arb references and scipy, batching, the NaN and support rules, and samples (their
moments, determinism). The cdfs are tests/stats/test_mvcdf.py."""

import math

import numpy as np
import pytest
import scipy.stats as st
from flint import arb, arb_mat, ctx

import stardust as sd

S = sd.stats


def _settled(fn, bits=80):
    """fn() (a tuple of arb balls) at doubling working precision until every ball's radius
    is below 2^-bits of max(1, |value|): references certified far below the 1e-14 measured."""
    old, prec = ctx.prec, 128
    try:
        while True:
            ctx.prec = prec
            vs = fn()
            if all(v.is_finite() and v.rad() <= arb(2) ** -bits * max(arb(1), abs(v.mid())) for v in vs):
                return vs
            if prec >= 1 << 13:
                raise ArithmeticError(f"reference did not settle by {prec} bits")
            prec *= 2
    finally:
        ctx.prec = old


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (q * np.exp(np.linspace(0.0, math.log(kappa), d))) @ q.T
    return (a + a.T) / 2.0


def _mixed(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b)))


def _ref_logpdf(mu, cov, xs, df=None):
    """Log densities at xs and the entropy, in ball arithmetic on the matrix as given: its
    determinant and solves by Arb's LU, the inputs exact."""
    d = len(mu)

    def fn():
        A = arb_mat(cov.tolist())
        ld = A.det().log()
        R = arb_mat([[float(x[i]) for x in xs] for i in range(d)]) - arb_mat([[float(mu[i])] * len(xs) for i in range(d)])
        Y = A.solve(R)
        pi = arb.pi()
        out = []
        for j in range(len(xs)):
            q = sum((R[i, j] * Y[i, j] for i in range(d)), arb(0))
            if df is None:
                out.append(-(d * (2 * pi).log() + ld + q) / 2)
            else:
                nu = arb(df)
                out.append(((nu + d) / 2).lgamma() - (nu / 2).lgamma() - arb(d) / 2 * (nu * pi).log() - ld / 2
                           - (nu + d) / 2 * (q / nu).log1p())
        if df is None:
            h = arb(d) / 2 * (1 + (2 * pi).log()) + ld / 2
        else:
            nu = arb(df)
            a, hh = nu / 2, arb(d) / 2
            h = (ld / 2 + a.lgamma() - (a + hh).lgamma() + hh * (nu * pi).log()
                 + (a + hh) * ((a + hh).digamma() - a.digamma()))
        return (*out, h)

    *out, h = _settled(fn)
    return np.array([float(v) for v in out]), float(h)


@pytest.mark.parametrize("df", [None, 0.7, 4.5, 1e9])
@pytest.mark.parametrize(("d", "kappa"), [(1, 1.0), (2, 1e3), (5, 1e6), (12, 1e9), (40, 1e9)])
def test_logpdf_and_entropy_against_arb(df, d, kappa):
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


def test_multinomial_and_dirichlet_against_arb():
    # large n: log n! and the expectations of log X_i! cancel from 1e6 to 12.6
    n, pf = 100000, [0.2, 0.3, 0.5]

    def multinomial_entropy():
        h = -arb(n + 1).lgamma() - n * sum((arb(q) * arb(q).log() for q in pf), arb(0))
        tail = arb(0)
        for qf in pf:
            q = arb(qf)
            mu, s = n * qf, math.sqrt(n * qf * (1 - qf))
            k0, k1 = int(mu - 40 * s), int(mu + 40 * s)
            # E[log X!] over k in [k0, k1), X ~ Binomial(n, q): the pmf and log k! by their
            # recurrences from k0
            lf = arb(k0 + 1).lgamma()
            lb = arb(n + 1).lgamma() - lf - arb(n - k0 + 1).lgamma() + k0 * q.log() + (n - k0) * (1 - q).log()
            w, r = lb.exp(), q / (1 - q)
            acc = arb(0)
            for k in range(k0, k1):
                acc += w * lf
                w = w * r * (n - k) / (k + 1)
                lf += arb(k + 1).log()
            h += acc
            # the rest: P(|X - nq| >= 40 s) <= 2 exp(-2 (40 s)^2 / n) (Hoeffding), times log n!
            t = min(mu - k0, k1 - mu)
            tail += 2 * (arb(-2 * t * t) / n).exp() * arb(n + 1).lgamma()
        return (h + arb(0, tail.upper()),)

    (h,) = _settled(multinomial_entropy)
    assert abs(S.Multinomial(n=float(n), p=sd.asarray(pf)).entropy() - float(h)) <= 1e-14 * abs(float(h))

    def dirichlet_entropy():
        a = [arb(1e6), arb(2e6), arb(3e6)]
        a0 = sum(a, arb(0))
        return (sum((x.lgamma() for x in a), arb(0)) - a0.lgamma() + (a0 - 3) * a0.digamma()
                - sum(((x - 1) * x.digamma() for x in a), arb(0)),)

    (hd,) = _settled(dirichlet_entropy)
    assert abs(S.Dirichlet(alpha=sd.asarray([1e6, 2e6, 3e6])).entropy() - float(hd)) <= 1e-14 * abs(float(hd))
    # a large-count log pmf: n p_i summing to a value far from 1 in log terms
    x = [20000.0, 30000.0, 50000.0]
    (lp,) = _settled(lambda: (arb(n + 1).lgamma() - sum((arb(v + 1).lgamma() for v in x), arb(0))
                              + sum((v * arb(q).log() for v, q in zip(x, pf)), arb(0)),))
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


def _mv_kernel(D, x, fast, pdf=False):
    k = sd._core.stats
    rows, idx, _ = D._points(sd.asarray(x), "logpdf")
    return np.asarray(k.mv_logpdf(rows, idx, D._mu, D._L, D._Ll, D._ldh, D._ldl, D._ok, D._df, D._ih, D._il,
                                  D._fast if fast else None, pdf))


@pytest.mark.parametrize("d", [1, 2, 3, 8, 9, 20, 50])
@pytest.mark.parametrize("kappa", [1.0, 1e3, 1e9])
def test_fast_path_agrees_with_the_double_double_path(d, kappa):
    """The plain-double quadratic form is taken only where its bound allows: within 4.4e-15
    (mixed for logpdf, relative for pdf) of the double-double path, the same bits for a point
    alone, in a batch, strided, and on any number of threads."""
    rng = np.random.default_rng(d * 7 + int(math.log10(kappa)))
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    cov = (q * kappa ** (-np.arange(d) / max(d - 1, 1))) @ q.T
    cov = (cov + cov.T) / 2
    mu = rng.normal(size=d)
    L = np.linalg.cholesky(cov)
    x = np.concatenate([mu + rng.normal(size=(600, d)) @ L.T, mu + rng.normal(size=(600, d)),
                        mu + 50.0 * rng.normal(size=(300, d)) @ L.T])
    for D in (S.MultivariateNormal(mu=sd.asarray(mu), cov=sd.asarray(cov)),
              S.MultivariateT(mu=sd.asarray(mu), shape=sd.asarray(cov), df=4.5)):
        got = np.asarray(D.logpdf(sd.asarray(x)))
        fast, precise = _mv_kernel(D, x, True), _mv_kernel(D, x, False)
        assert np.array_equal(got, fast)
        assert np.max(np.abs(fast - precise) / np.maximum(1.0, np.abs(precise))) <= 4.5e-15
        pf, pp = _mv_kernel(D, x, True, True), _mv_kernel(D, x, False, True)
        ok = (pp > 1e-300) & np.isfinite(pp)
        assert np.max(np.abs(pf[ok] - pp[ok]) / pp[ok], initial=0.0) <= 4.5e-15
        assert np.array_equal(pf[~ok], pp[~ok])
        for i in (0, 599, 600, 1499):
            assert D.logpdf(sd.asarray(x[i])) == got[i]
        wide = np.zeros((len(x), 2 * d))
        wide[:, ::2] = x
        assert np.array_equal(np.asarray(D.logpdf(sd.asarray(wide)[:, ::2])), got)
        before = sd.get_num_threads()
        try:
            for th in (1, 3, 10):
                sd.set_num_threads(th)
                assert np.array_equal(np.asarray(D.logpdf(sd.asarray(x))), got)
        finally:
            sd.set_num_threads(before)


def test_fast_path_special_values_and_batches():
    """NaN, infinities and invalid batch elements go around the fast path as before; a batch of
    factors keeps each element's own inverse."""
    rng = np.random.default_rng(5)
    covs = np.stack([np.eye(3), np.diag([1.0, 4.0, 9.0]), -np.eye(3), np.array([[2.0, 1, 0], [1, 2, 1], [0, 1, 2]])])
    D = S.MultivariateNormal(mu=sd.asarray(np.zeros(3)), cov=sd.asarray(covs))
    x = rng.normal(size=(5, 4, 3))
    got = np.asarray(D.logpdf(sd.asarray(x)))
    assert np.isnan(got[:, 2]).all()
    for b in (0, 1, 3):
        want = st.multivariate_normal(np.zeros(3), covs[b]).logpdf(x[:, b])
        np.testing.assert_allclose(got[:, b], want, rtol=1e-14)
    one = S.MultivariateNormal(mu=sd.asarray(np.zeros(3)), cov=sd.asarray(covs[3]))
    assert math.isnan(one.logpdf(sd.asarray([0.0, float("nan"), float("inf")])))
    assert one.logpdf(sd.asarray([0.0, float("inf"), 1.0])) == -math.inf
    assert one.pdf(sd.asarray([0.0, -float("inf"), 1.0])) == 0.0
    huge = np.array([[1e308, -1e308, 1e308], [1e200, 1e-200, -1e155]])  # the quadratic form overflows: as before
    assert np.array_equal(_mv_kernel(one, huge, True), _mv_kernel(one, huge, False), equal_nan=True)


def test_dirichlet_fast_path_against_arb():
    """The plain-double sum of the Dirichlet's log density (where its bound allows) against
    Arb, bulk and corners."""
    rng = np.random.default_rng(8)
    for al in ([0.5, 2.0, 3.0, 1.5], [0.02, 0.7], [30.0, 1.0, 250.0], [1e-3, 1e3, 2.5, 1.0, 7.0]):
        al = np.array(al)
        x = rng.dirichlet(np.maximum(al, 0.3), size=40)
        D = S.Dirichlet(alpha=sd.asarray(al))
        got = np.asarray(D.logpdf(sd.asarray(x)))
        gotp = np.asarray(D.pdf(sd.asarray(x)))

        def wants():
            am = [arb(float(a)) for a in al]
            c = sum(am, arb(0)).lgamma() - sum((a.lgamma() for a in am), arb(0))
            return tuple(c + sum(((a - 1) * arb(float(v)).log() for a, v in zip(am, row)), arb(0)) for row in x)

        old = ctx.prec
        try:
            ws = _settled(wants)
            ctx.prec = 128  # the comparisons below, in ball arithmetic on the settled references
            for i, want in enumerate(ws):
                assert abs(arb(got[i]) - want) <= 1e-14 * max(arb(1), abs(want)), (al, i)
                assert abs(arb(gotp[i]) - want.exp()) <= 1e-14 * want.exp(), (al, i)
                assert D.logpdf(sd.asarray(x[i])) == got[i]
        finally:
            ctx.prec = old


def test_multivariate_samples_split_and_threads():
    """Rows draw their words four rows at a time: the same bits however a call is split and on
    any number of threads."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(9, 9))
    cov = a @ a.T + np.eye(9)
    dists = [S.MultivariateNormal(mu=sd.asarray(np.arange(9.0)), cov=sd.asarray(cov)),
             S.MultivariateT(mu=sd.asarray(np.arange(9.0)), shape=sd.asarray(cov), df=0.7),
             S.Multinomial(n=500.0, p=sd.asarray([0.1, 0.2, 0.3, 0.4])),
             S.Dirichlet(alpha=sd.asarray([0.5, 2.0, 3.0, 1.5])), S.Dirichlet(alpha=sd.asarray([1e-3, 0.01, 2.0]))]
    before = sd.get_num_threads()
    try:
        for D in dists:
            sd.set_num_threads(1)
            whole = np.asarray(D.sample(3001, rng=sd.random.default_rng(4)))
            for th in (1, 3, 10):
                sd.set_num_threads(th)
                r = sd.random.default_rng(4)
                parts = [np.asarray(D.sample(n, rng=r)) for n in (1, 2, 5, 993, 2000)]
                assert np.array_equal(np.concatenate(parts), whole)
    finally:
        sd.set_num_threads(before)
    x = np.asarray(S.Dirichlet(alpha=sd.asarray([0.5, 2.0, 3.0, 1.5])).sample(200000, rng=sd.random.default_rng(1)))
    np.testing.assert_allclose(x.mean(axis=0), np.array([0.5, 2.0, 3.0, 1.5]) / 7.0, atol=2e-3)
    np.testing.assert_allclose(x.sum(axis=1), 1.0, rtol=1e-14)


def test_dirichlet_density_keeps_alpha_minus_one_exactly():
    # alpha - 1 rounds for alpha < 1/2; times log(1e-300) the rounding reached the density
    # (3.8e-14 relative). Truth: Arb at 128 bits, the alphas and x exact.
    D = S.Dirichlet(alpha=sd.asarray([0.3, 0.6]))
    x = sd.asarray([[1e-300, 1.0]])
    assert float(np.asarray(D.pdf(x))[0]) == pytest.approx(2.398706130865886e+209, rel=2e-15)
    assert float(np.asarray(D.logpdf(x))[0]) == pytest.approx(482.11521391559705, rel=2e-15)


# ---- extreme arguments (docs/design/stats.md §3.5)

def test_logpdf_where_the_quadratic_form_overflows():
    # the normal: -Q / 2 below every double is -inf (NaN before: inf - inf in the substitution)
    for d, kappa in ((1, 1.0), (3, 1e6), (50, 1e9)):
        cov = spd(np.random.default_rng(d), d, kappa)
        N = S.MultivariateNormal(mu=sd.asarray(np.zeros(d)), cov=sd.asarray(cov))
        for x in (np.full(d, 1e308), np.full(d, -1e308), 1e308 * (-1.0) ** np.arange(d)):
            assert N.logpdf(sd.asarray(x)) == -math.inf
            assert N.pdf(sd.asarray(x)) == 0.0
    # the t's log density falls like log |x|: finite at every finite x, against Arb
    ctx.prec = 300
    for df in (0.5, 3.0, 1e9, 1e300):
        for d, x0, scale in ((1, 1e308, 1.0), (2, 1e308, 1.0), (3, -1e308, 1.0), (2, 1e200, 1e-300), (2, 1e-100, 1e-300),
                             (3, 1e308, 1e300)):
            T = S.MultivariateT(mu=sd.asarray(np.zeros(d)), shape=sd.asarray(np.eye(d) * scale), df=df)
            nu, q = arb(df), d * arb(x0) ** 2 / arb(scale)
            ref = (((nu + d) / 2).lgamma() - (nu / 2).lgamma() - arb(d) / 2 * (nu * arb.pi()).log()
                   - arb(d) / 2 * arb(scale).log() - (nu + d) / 2 * (q / nu).log1p())
            got = T.logpdf(sd.asarray(np.full(d, x0)))
            ref = float(ref.mid())
            assert got == ref or abs(got - ref) <= 1e-14 * max(1.0, abs(ref)), (df, d, x0, scale, got, ref)
    # a mean at the other end of the doubles: x - mu itself overflows
    N = S.MultivariateNormal(mu=sd.asarray([-1e308, -1e308]), cov=sd.asarray(np.eye(2)))
    assert N.logpdf(sd.asarray([1e308, 1e308])) == -math.inf
    T = S.MultivariateT(mu=sd.asarray([-1e308]), shape=sd.asarray([[1.0]]), df=2.0)
    ref = float((arb(1.5).lgamma() - arb(1).lgamma() - (2 * arb.pi()).log() / 2 - arb(1.5) * ((arb(2e308)) ** 2 / 2).log1p()).mid())
    assert abs(T.logpdf(sd.asarray([1e308])) - ref) <= 1e-14 * abs(ref)


def test_multinomial_entropy_is_bounded():
    import time

    t0 = time.perf_counter()
    # too many terms (the sum of the categories' standard deviations above 2^17), or n beyond
    # 2^53: NaN (the loop over k = k + 1 never ended at n = 1e300)
    for n in (1e12, 1e15, 2.0**53 * 4, 1e100, 1e300):
        assert math.isnan(S.Multinomial(n=n, p=sd.asarray([0.5, 0.5])).entropy())
    assert time.perf_counter() - t0 < 5.0
    # a point mass has entropy 0 at any n; small standard deviations are summed at any n <= 2^53
    assert S.Multinomial(n=1e300, p=sd.asarray([1.0, 0.0])).entropy() == 0.0
    h = S.Multinomial(n=1e9, p=sd.asarray([1e-9, 1.0 - 1e-9])).entropy()
    assert abs(h - 1.3048422422) < 1e-6  # Poisson(1)'s entropy, 1.30484..., to O(1e-9)


def test_degenerate_degrees_of_freedom():
    # subnormal df: every method NaN (df / 2 is 0)
    T = S.MultivariateT(mu=sd.asarray([0.0, 0.0]), shape=sd.asarray(np.eye(2)), df=5e-324)
    assert math.isnan(T.logpdf(sd.asarray([0.0, 1.0]))) and math.isnan(T.entropy())
    assert np.isnan(np.asarray(T.sample(3, rng=sd.random.default_rng(1)))).all()
    # StudentT's boundary (2026-11-19): below 2^-1021 df / 2 is not exact, NaN; 2^-1021 is in
    T = S.MultivariateT(mu=sd.asarray([0.0, 0.0]), shape=sd.asarray(np.eye(2)), df=2.0**-1022)
    assert math.isnan(T.logpdf(sd.asarray([0.0, 1.0])))
    assert math.isnan(S.StudentT(df=2.0**-1022).logpdf(1.0))
    T = S.MultivariateT(mu=sd.asarray([0.0, 0.0]), shape=sd.asarray(np.eye(2)), df=2.0**-1021)
    assert math.isfinite(T.logpdf(sd.asarray([0.0, 1.0])))


def test_multinomial_logpmf_beyond_exact_counts():
    # n + 1.0 is n from 2^53 on: log n! was log (n-1)! (a pmf of 6e7 at n = 1e17). Where the
    # double-double sum no longer holds the value (n beyond ~1e13 near the mode): NaN, never a pmf > 1
    ctx.prec = 400
    for e in (9, 12, 15):
        n = arb(10) ** e
        ref = float(((n + 1).lgamma() - 2 * (n / 2 + 1).lgamma() - n * arb(2).log()).mid())
        got = S.Multinomial(n=10.0**e, p=sd.asarray([0.5, 0.5])).logpmf(sd.asarray([10.0**e / 2, 10.0**e / 2]))
        assert (e > 12 and math.isnan(got)) or abs(got - ref) <= 1e-13 * abs(ref), (e, got, ref)
    for n in (1e17, 1e100, 1e300):
        v = S.Multinomial(n=n, p=sd.asarray([0.5, 0.5])).logpmf(sd.asarray([n / 2, n / 2]))
        assert math.isnan(v) or v <= 0.0
        v = S.Multinomial(n=n, p=sd.asarray([0.5, 0.5])).pmf(sd.asarray([n / 2, n / 2]))
        assert math.isnan(v) or 0.0 <= v <= 1.0
