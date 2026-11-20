"""Matrix distributions (T10, docs/api/stats.md §3.1; decisions T30): log densities and entropies
against Arb references, scipy's moments, the support, NaN and divergence rules, batching,
and samples (their marginal laws, determinism)."""

import math

import numpy as np
import pytest
import scipy.stats as st
from flint import arb, arb_mat, ctx

import stardust as sd

S = sd.stats


def spd(rng, d, kappa):
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    a = (q * np.exp(np.linspace(0.0, math.log(kappa), d))) @ q.T
    return (a + a.T) / 2.0


def sym(x):
    """Exactly symmetric from the lower triangle (the one read)."""
    return np.tril(x) + np.swapaxes(np.tril(x, -1), -1, -2)


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


def _mvlgamma(p, a):
    return p * (p - 1) / arb(4) * arb.pi().log() + sum(((a + arb(1 - j) / 2).lgamma() for j in range(1, p + 1)), arb(0))


def _mvdigamma(p, a):
    return sum(((a + arb(1 - j) / 2).digamma() for j in range(1, p + 1)), arb(0))


def _trace_solve(A, B):
    """tr(A^-1 B) by Arb's LU solve (A, B arb_mat)."""
    Y = A.solve(B)
    return sum((Y[i, i] for i in range(A.nrows())), arb(0))


def _ref_wishart(df, V, xs, inverse):
    """Log densities at xs and the entropy, in ball arithmetic on the matrices as given."""
    p = V.shape[0]

    def fn():
        n = arb(df)
        Vm = arb_mat(V.tolist())
        ldv = Vm.det().log()
        out = []
        for x in xs:
            X = arb_mat(x.tolist())
            ldx = X.det().log()
            if inverse:
                tr = _trace_solve(X, Vm)  # tr(V X^-1) = tr(X^-1 V)
                out.append(
                    n / 2 * ldv - n * p / 2 * arb(2).log() - _mvlgamma(p, n / 2) - (n + p + 1) / 2 * ldx - tr / 2
                )
            else:
                tr = _trace_solve(Vm, X)
                out.append(
                    (n - p - 1) / 2 * ldx - tr / 2 - n * p / 2 * arb(2).log() - n / 2 * ldv - _mvlgamma(p, n / 2)
                )
        s = -1 if inverse else 1
        k = (n + p + 1) if inverse else (n - p - 1)
        h = (
            (p + 1) / arb(2) * ldv
            + s * p * (p + 1) / arb(2) * arb(2).log()
            + _mvlgamma(p, n / 2)
            - k / 2 * _mvdigamma(p, n / 2)
            + n * p / 2
        )
        return (*out, h)

    *out, h = _settled(fn)
    return np.array([float(v) for v in out]), float(h)


def _mixed(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.max(np.abs(a - b) / np.maximum(1.0, np.abs(b))))


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize(
    ("p", "extra", "kappa"), [(1, 0.5, 1.0), (2, 0.5, 1e4), (3, 7.0, 1e8), (5, 1e8, 1e4), (8, 3.0, 1e6)]
)
def test_wishart_against_arb(inverse, p, extra, kappa):
    rng = np.random.default_rng(p * 13 + int(math.log10(kappa)))
    V = spd(rng, p, kappa)
    df = p - 1 + extra
    law = (st.invwishart if inverse else st.wishart)(min(df, 1e6), V)
    xs = sym(np.asarray(law.rvs(3, random_state=4)).reshape((-1, p, p)))
    ref, href = _ref_wishart(df, V, xs, inverse)
    D = (S.InverseWishart if inverse else S.Wishart)(df=df, scale=sd.asarray(V))
    assert _mixed(D.logpdf(sd.asarray(xs)), ref) <= 1e-14
    assert _mixed(D.entropy(), href) <= 1e-14
    pdf = np.asarray(D.pdf(sd.asarray(xs)))
    ok = np.abs(ref) < 700
    assert np.allclose(pdf[ok], np.exp(ref[ok]), rtol=1e-14 * max(1.0, np.abs(ref).max()), atol=0)


def test_inverse_wishart_entropy_against_monte_carlo():
    # scipy 1.18's invwishart.entropy() is wrong (2.97 here); -E[log pdf] says 0.19 +- 0.006
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 3))
    V = a @ a.T + np.eye(3)
    D = S.InverseWishart(df=6.5, scale=sd.asarray(V))
    x = np.asarray(D.sample(200000, rng=sd.random.default_rng(1)))
    lp = np.asarray(D.logpdf(sd.asarray(x)))
    assert abs(-lp.mean() - D.entropy()) < 5 * lp.std() / math.sqrt(lp.size)


def test_matrix_normal_against_arb():
    rng = np.random.default_rng(5)
    for n, p, kappa in ((1, 1, 1.0), (2, 3, 1e4), (5, 4, 1e8)):
        M = rng.normal(size=(n, p))
        U, V = spd(rng, n, kappa), spd(rng, p, kappa)
        xs = st.matrix_normal(M, U, V).rvs(3, random_state=1).reshape((-1, n, p))

        def fn():
            Um, Vm = arb_mat(U.tolist()), arb_mat(V.tolist())
            lu, lv = Um.det().log(), Vm.det().log()
            out = []
            for x in xs:
                Dm = arb_mat(x.tolist()) - arb_mat(M.tolist())
                tr = _trace_solve(Vm, Dm.transpose() * Um.solve(Dm))  # tr(V^-1 D' U^-1 D)
                out.append(-n * p / arb(2) * (2 * arb.pi()).log() - p / arb(2) * lu - n / arb(2) * lv - tr / 2)
            h = n * p / arb(2) * (1 + (2 * arb.pi()).log()) + p / arb(2) * lu + n / arb(2) * lv
            return (*out, h)

        *vals, hb = _settled(fn)
        ref, h = [float(v) for v in vals], float(hb)
        D = S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V))
        assert _mixed(D.logpdf(sd.asarray(xs)), ref) <= 1e-14
        assert _mixed(D.entropy(), h) <= 1e-14


def test_moments_match_scipy():
    rng = np.random.default_rng(2)
    V = spd(rng, 3, 10.0)
    for df in (2.5, 4.0, 7.5):
        W, Ws = S.Wishart(df=df, scale=sd.asarray(V)), st.wishart(df, V)
        np.testing.assert_allclose(np.asarray(W.mean()), Ws.mean(), rtol=1e-14)
        np.testing.assert_allclose(np.asarray(W.variance()), Ws.var(), rtol=1e-14)
        if df >= 4.0:
            np.testing.assert_allclose(np.asarray(W.mode()), Ws.mode(), rtol=1e-14)
        else:
            assert np.isnan(np.asarray(W.mode())).all()
    I, Is = S.InverseWishart(df=8.0, scale=sd.asarray(V)), st.invwishart(8.0, V)
    np.testing.assert_allclose(np.asarray(I.mean()), Is.mean(), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(I.mode()), Is.mode(), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(I.variance()), Is.var(), rtol=1e-13)
    # divergence, as InverseGamma: the mean diverges for df <= p + 1, the variance is inf for
    # p + 1 < df <= p + 3 and NaN below
    Vd = np.array([[2.0, 0.0], [0.0, 1.0]])
    m = np.asarray(S.InverseWishart(df=2.5, scale=sd.asarray(Vd)).mean())
    assert m[0, 0] == math.inf and m[0, 1] == 0.0
    assert np.isinf(np.asarray(S.InverseWishart(df=4.5, scale=sd.asarray(Vd)).variance())).all()
    assert np.isnan(np.asarray(S.InverseWishart(df=2.5, scale=sd.asarray(Vd)).variance())).all()
    # p = 1: the inverse Wishart is InverseGamma(df/2, scale/2)
    for df in (1.5, 3.0, 5.5, 9.0):
        I1 = S.InverseWishart(df=df, scale=sd.asarray([[3.0]]))
        G = S.InverseGamma(a=df / 2, scale=1.5)
        assert np.asarray(I1.mean())[0, 0] == pytest.approx(G.mean(), rel=1e-15) or math.isinf(G.mean())
        assert I1.logpdf(sd.asarray([[0.7]])) == pytest.approx(G.logpdf(0.7), rel=1e-14)
        assert I1.entropy() == pytest.approx(G.entropy(), rel=1e-14, abs=1e-14)
    M = rng.normal(size=(2, 3))
    MN = S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(np.eye(2)), colcov=sd.asarray(np.eye(3)))
    np.testing.assert_array_equal(np.asarray(MN.mean()), M)


def test_support_and_nan_rules():
    W = S.Wishart(df=4.0, scale=sd.asarray(np.eye(2)))
    assert W.pdf(sd.asarray([[1.0, 2.0], [2.0, 1.0]])) == 0.0  # not positive definite
    assert W.logpdf(sd.asarray([[1.0, 0.0], [0.0, -1.0]])) == -math.inf
    # only the lower triangle is read
    a = W.logpdf(sd.asarray([[2.0, 99.0], [0.5, 1.0]]))
    b = W.logpdf(sd.asarray([[2.0, 0.5], [0.5, 1.0]]))
    assert a == b
    assert math.isnan(S.Wishart(df=0.5, scale=sd.asarray(np.eye(2))).entropy())  # df <= p - 1
    assert math.isnan(S.Wishart(df=3.0, scale=sd.asarray([[1.0, 2.0], [2.0, 1.0]])).logpdf(sd.asarray(np.eye(2))))
    assert np.isnan(np.asarray(S.InverseWishart(df=0.9, scale=sd.asarray(np.eye(2))).mean())).all()
    assert math.isnan(
        S.MatrixNormal(mu=sd.asarray([[np.nan]]), rowcov=sd.asarray([[1.0]]), colcov=sd.asarray([[1.0]])).entropy()
    )


def test_batching():
    V = sd.asarray(np.eye(2) * np.array([1.0, 2.0, 3.0])[:, None, None])
    W = S.Wishart(df=sd.asarray([3.0, 4.0, 5.0]), scale=V)
    assert np.asarray(W.logpdf(sd.asarray(np.eye(2)))).shape == (3,)
    assert np.asarray(W.logpdf(sd.asarray(np.tile(np.eye(2), (4, 1, 1, 1))))).shape == (4, 3)
    for j, (df, s) in enumerate(((3.0, 1.0), (4.0, 2.0), (5.0, 3.0))):
        assert np.asarray(W.logpdf(sd.asarray(np.eye(2))))[j] == pytest.approx(
            st.wishart(df, np.eye(2) * s).logpdf(np.eye(2)), rel=1e-13
        )
    assert np.asarray(W.entropy()).shape == (3,)
    assert np.asarray(W.mean()).shape == (3, 2, 2)
    assert np.asarray(W.sample(5, rng=sd.random.default_rng(1))).shape == (5, 3, 2, 2)
    assert isinstance(S.Wishart(df=3.0, scale=sd.asarray(np.eye(2))).entropy(), float)
    MN = S.MatrixNormal(mu=sd.asarray(np.zeros((4, 2, 3))), rowcov=sd.asarray(np.eye(2)), colcov=sd.asarray(np.eye(3)))
    assert np.asarray(MN.logpdf(sd.asarray(np.zeros((2, 3))))).shape == (4,)
    assert np.asarray(MN.sample(6, rng=sd.random.default_rng(1))).shape == (6, 4, 2, 3)


def test_samples():
    rng = np.random.default_rng(3)
    V = spd(rng, 3, 20.0)
    N = 100000
    for df in (2.5, 7.0):
        x = np.asarray(S.Wishart(df=df, scale=sd.asarray(V)).sample(N, rng=sd.random.default_rng(2)))
        assert np.allclose(x, np.swapaxes(x, -1, -2))
        assert st.kstest(x[:, 0, 0] / V[0, 0], st.chi2(df).cdf).pvalue > 1e-4  # W_11 / V_11 ~ chi^2(df)
        # positive definite (df < p gives eigenvalues near 0, to the rounding of the product)
        ev = np.linalg.eigvalsh(x)
        assert np.all(ev > -1e-13 * ev[:, -1:])
        # a' W a / a' V a ~ chi^2(df) for any a
        a = np.array([0.3, -1.0, 0.5])
        assert st.kstest(np.einsum("i,nij,j->n", a, x, a) / (a @ V @ a), st.chi2(df).cdf).pvalue > 1e-4
    x = np.asarray(S.InverseWishart(df=6.5, scale=sd.asarray(V)).sample(N, rng=sd.random.default_rng(3)))
    # X_11 ~ InverseGamma((df - p + 1) / 2, V_11 / 2)
    assert st.kstest(x[:, 0, 0], st.invgamma((6.5 - 3 + 1) / 2, scale=V[0, 0] / 2).cdf).pvalue > 1e-4
    M = rng.normal(size=(2, 3))
    U, Vc = spd(rng, 2, 3.0), spd(rng, 3, 3.0)
    xm = np.asarray(
        S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(Vc)).sample(
            N, rng=sd.random.default_rng(4)
        )
    )
    assert st.kstest((xm[:, 1, 2] - M[1, 2]) / math.sqrt(U[1, 1] * Vc[2, 2]), st.norm.cdf).pvalue > 1e-4
    c = np.cov(xm.reshape(N, -1).T)
    np.testing.assert_allclose(c, np.kron(U, Vc), atol=0.05)


def test_samples_are_deterministic():
    W = S.InverseWishart(df=5.0, scale=sd.asarray([[1.0, 0.3], [0.3, 2.0]]))
    before = sd.get_num_threads()
    try:
        outs = []
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            outs.append(np.asarray(W.sample(20000, rng=sd.random.default_rng(1))))
    finally:
        sd.set_num_threads(before)
    assert np.array_equal(outs[0], outs[1]) and np.array_equal(outs[0], outs[2])
    rng = sd.random.default_rng(2)
    parts = np.concatenate([np.asarray(W.sample(50, rng=rng)), np.asarray(W.sample(50, rng=rng))])
    assert np.array_equal(parts, np.asarray(W.sample(100, rng=sd.random.default_rng(2))))


def test_argument_rules():
    with pytest.raises(TypeError):
        S.Wishart(3.0, sd.asarray(np.eye(2)))
    with pytest.raises(ValueError, match="square"):
        S.Wishart(df=3.0, scale=sd.asarray(np.ones((2, 3))))
    with pytest.raises(ValueError, match="rowcov"):
        S.MatrixNormal(mu=sd.asarray(np.zeros((2, 3))), rowcov=sd.asarray(np.eye(3)), colcov=sd.asarray(np.eye(3)))
    with pytest.raises(ValueError, match="last two axes"):
        S.Wishart(df=3.0, scale=sd.asarray(np.eye(2))).logpdf(sd.asarray(np.eye(3)))
    with pytest.raises(TypeError, match="Generator"):
        S.Wishart(df=3.0, scale=sd.asarray(np.eye(2))).sample(rng=0)
    MN = S.MatrixNormal(mu=sd.asarray(np.zeros((2, 2))), rowcov=sd.asarray(np.eye(2)), colcov=sd.asarray(np.eye(2)))
    assert repr(MN).startswith("MatrixNormal(mu=") and MN.mu is not None


@pytest.mark.parametrize("shape", [(1, 1), (3, 2), (2, 9), (10, 5), (17, 12)])
@pytest.mark.parametrize("kappa", [1.0, 1e2, 1e8])
def test_matrix_normal_fast_path_agrees_with_the_double_double_path(shape, kappa):
    """The plain-double |L_U^-1 (X - M) L_V^-T|^2 is taken only where its bound allows: within
    4.4e-15 of the double-double path, the same bits for a point alone and on any number of
    threads."""
    n, p = shape
    rng = np.random.default_rng(n * 31 + p + int(math.log10(kappa)))

    def cov(d, k):
        q, _ = np.linalg.qr(rng.normal(size=(d, d)))
        c = (q * k ** (-np.arange(d) / max(d - 1, 1))) @ q.T
        return (c + c.T) / 2

    U, V, M = cov(n, kappa), cov(p, math.sqrt(kappa)), rng.normal(size=(n, p))
    D = S.MatrixNormal(mu=sd.asarray(M), rowcov=sd.asarray(U), colcov=sd.asarray(V))
    LU, LV = np.linalg.cholesky(U), np.linalg.cholesky(V)
    x = np.concatenate([M + LU @ rng.normal(size=(300, n, p)) @ LV.T, M + rng.normal(size=(300, n, p))])
    got = np.asarray(D.logpdf(sd.asarray(x)))
    k = sd._core.stats
    rows, idx, _ = D._points(sd.asarray(x), "logpdf")
    args = (rows, idx, D._mean, D._U, D._Ul, D._V, D._Vl, D._lu[0], D._lu[1], D._lv[0], D._lv[1], D._ok)
    precise = np.asarray(k.matnorm_logpdf(*args, None, None, False))
    assert np.max(np.abs(got - precise) / np.maximum(1.0, np.abs(precise))) <= 4.5e-15
    pf, pp = np.asarray(D.pdf(sd.asarray(x))), np.asarray(k.matnorm_logpdf(*args, None, None, True))
    ok = (pp > 1e-300) & np.isfinite(pp)
    assert np.max(np.abs(pf[ok] - pp[ok]) / pp[ok], initial=0.0) <= 4.5e-15
    assert np.array_equal(pf[~ok], pp[~ok])
    for i in (0, 299, 300, 599):
        assert D.logpdf(sd.asarray(x[i])) == got[i]
    before = sd.get_num_threads()
    try:
        for th in (1, 3, 10):
            sd.set_num_threads(th)
            assert np.array_equal(np.asarray(D.logpdf(sd.asarray(x))), got)
    finally:
        sd.set_num_threads(before)


def test_logpdf_at_huge_df_is_right_or_nan():
    # the terms are of size df p log df: past df ~ 1e17 the double-double sum does not hold the
    # value (-4690 where the truth is -35.8 at df = 1e30): NaN there, right below
    ctx.prec = 3000
    for p in (1, 3):
        for df in (1e9, 1e12, 1e15, 1e18, 1e30, 1e100, 1e300):
            n, X = arb(df), arb(df)
            lg = sum((((n - j) / 2).lgamma() for j in range(p)), arb(0)) + arb(p * (p - 1)) / 4 * arb.pi().log()
            ref = float(((n - p - 1) / 2 * p * X.log() - p * X / 2 - n * p / 2 * arb(2).log() - lg).mid())
            got = sd.stats.Wishart(df=df, scale=sd.asarray(np.eye(p))).logpdf(sd.asarray(np.eye(p) * df))
            assert math.isnan(got) or abs(got - ref) <= 1e-13 * max(1.0, abs(ref)), (p, df, got, ref)
            if df <= 1e15:
                assert not math.isnan(got)
            ref = float((n / 2 * p * n.log() - n * p / 2 * arb(2).log() - lg - p * n / 2).mid())
            got = sd.stats.InverseWishart(df=df, scale=sd.asarray(np.eye(p) * df)).logpdf(sd.asarray(np.eye(p)))
            assert math.isnan(got) or abs(got - ref) <= 1e-13 * max(1.0, abs(ref)), (p, df, got, ref)
            if df <= 1e15:
                assert not math.isnan(got)


def test_log_density_far_out_is_a_value_or_minus_inf():
    # X (or X - M) beyond 2^400: scaled before the factor and the solves; it was NaN (inf - inf)
    mn = sd.stats.MatrixNormal(mu=sd.zeros((1, 1)), rowcov=sd.eye(1), colcov=sd.eye(1))
    c = -0.5 * math.log(2 * math.pi)
    assert float(mn.logpdf(sd.asarray([[1.5e154]]))) == pytest.approx(-(0.5 * 1.5e154) * 1.5e154 + c, rel=1e-15)
    for v in (1.9e154, 1e200, 1.7e308, float("inf")):
        assert float(mn.logpdf(sd.asarray([[v]]))) == -math.inf
        assert float(mn.pdf(sd.asarray([[v]]))) == 0.0
    w = sd.stats.Wishart(df=5.0, scale=sd.eye(2))
    assert float(w.logpdf(1.7e308 * sd.eye(2))) == pytest.approx(-1.7e308, rel=1e-15)
    assert float(w.logpdf(sd.asarray([[math.inf, 0.0], [0.0, 1.0]]))) == -math.inf
    iw = sd.stats.InverseWishart(df=5.0, scale=sd.eye(2))
    assert math.isfinite(float(iw.logpdf(1.7e308 * sd.eye(2))))
