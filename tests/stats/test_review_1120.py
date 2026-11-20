"""stats (2026-11-20 review, batch T): invalid parameters never reach the scalar fallbacks
(ST-17: the Pareto that did not return), the median of the t on the log side (ST-1), signed
zeros (ST-14), numpy scalars (ST-3, CX-3), infinite terms in the compensated sums (ST-4), the
matrix normal's entropy (ST-6), the Dirichlet and multinomial covariances (ST-7), iqr's q
(ST-8, ST-9), messages (ST-10, CX-4, CX-10, CX-11, CX-14), point masses (ST-11), scalar inputs
(ST-15), the beta's deep logcdf next to 1 (R120), result types (CX-16), and the decisions
D-4, D-5, D-13, D-25, D-30, D-32."""

import math
import subprocess
import sys

import numpy as np
import pytest
from flint import arb, ctx

import stardust as sd

S = sd.stats
A = sd.asarray

# ---- ST-17: invalid parameters ------------------------------------------------------------

PARETO_HANGS = [
    (-1.0, 0.0, 0.0, "pdf"),
    (-1.0, 0.0, -0.0, "pdf"),
    (1.0, 0.0, 0.0, "pdf"),
    (0.0, 0.0, 0.0, "pdf"),
    (1.0, 0.0, 0.0, "logpdf"),
    (1.0, -1.0, -0.5, "pdf"),
    (1.0, -1.0, -0.5, "cdf"),
    (1.0, -1.0, -0.5, "logpdf"),
]

_CHILD = """
import math, sys, stardust as sd
S = sd.stats
for b, s, x, m in {cases!r}:
    v = getattr(S.Pareto(b=b, scale=s), m)(x)
    assert math.isnan(v), (b, s, x, m, v)
    # an array of them beside a valid element: the valid one keeps its value
    d = S.Pareto(b=sd.asarray([b, 2.0, b]), scale=sd.asarray([s, 1.0, s]))
    r = getattr(d, m)(sd.asarray([x, 1.5, x])).tolist()
    assert math.isnan(r[0]) and math.isnan(r[2]) and r[1] == getattr(S.Pareto(b=2.0, scale=1.0), m)(1.5), r
# the log ratio of a Weibull-like scale <= 0 (lifetime and weibull detail) through the engine
for m in ("pdf", "logpdf", "cdf", "ccdf", "logcdf", "logccdf"):
    for s in (-math.inf, -1.0, -0.0, 0.0):
        for x in (-1e300, -1.0, -0.5, -0.0, 0.0, 5e-324, 1e-310):
            assert math.isnan(getattr(S.Pareto(b=1.0, scale=s), m)(x))
            assert math.isnan(getattr(S.Weibull(c=1.0, scale=s), m)(x))
print("ok")
"""


def test_invalid_pareto_ends_with_nan():
    # each call used to loop forever holding the GIL: a child process with a time limit
    r = subprocess.run(
        [sys.executable, "-c", _CHILD.format(cases=PARETO_HANGS)], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-2000:]


def test_chunk_kernels_see_only_good_elements():
    # invalid, NaN-input and limit elements mixed with valid ones in one chunk: the valid ones
    # keep the bits of their scalar calls, the others are NaN or the limit's value
    df = A([3.0, -1.0, math.inf, 3.0, math.nan, 7.0])
    x = A([0.5, 0.5, 0.5, math.nan, 0.5, -2.0])
    got = S.StudentT(df=df).cdf(x).tolist()
    want = [
        S.StudentT(df=3.0).cdf(0.5),
        math.nan,
        S.Normal().cdf(0.5),
        math.nan,
        math.nan,
        S.StudentT(df=7.0).cdf(-2.0),
    ]
    assert all(g == w or (math.isnan(g) and math.isnan(w)) for g, w in zip(got, want)), (got, want)


# ---- ST-1, ST-14 -----------------------------------------------------------------------------


@pytest.mark.parametrize("df", [1e-4, 1e-3, 1e6, 1.1e6, 1e9, math.inf])
def test_student_t_median_on_the_log_side(df):
    X = S.StudentT(df=df)
    for v in (X.ilogcdf(math.log(0.5)), X.ilogccdf(math.log(0.5)), X.icdf(0.5), X.iccdf(0.5)):
        assert v == 0.0 and math.copysign(1.0, v) == 1.0, (df, v)
    assert S.NoncentralT(df=2e6, nc=0.0).ilogcdf(math.log(0.5)) == 0.0


def test_signed_zeros():
    def pos0(v):
        return v == 0.0 and math.copysign(1.0, v) == 1.0

    assert pos0(S.StudentT(df=5.0).iccdf(0.5)) and pos0(S.NoncentralT(df=5.0, nc=0.0).iccdf(0.5))
    assert pos0(S.StudentT(df=5.0).ilogccdf(math.log(0.5)))
    P = S.Poisson(mu=-0.0)
    for v in (P.variance(), P.standard_deviation(), P.mean(), P.median(), S.Binomial(n=3.0, p=-0.0).variance()):
        assert pos0(v), v
    assert pos0(S.Bernoulli(p=-0.0).mean())
    # a support on the whole line keeps the sign of the location
    assert math.copysign(1.0, S.Normal(mu=-0.0).mean()) == -1.0


# ---- ST-3, CX-3: numpy scalars -----------------------------------------------------------


def test_numpy_scalar_parameters():
    assert repr(S.Normal(mu=np.int64(1))) == "Normal(mu=1.0, sigma=1.0)"
    assert repr(S.Normal(mu=np.float64(2.0))) == "Normal(mu=2.0, sigma=1.0)"
    assert S.Normal(mu=np.float32(1.5)).mean() == 1.5
    assert S.Binomial(n=np.int64(10), p=0.3).mean() == S.Binomial(n=10, p=0.3).mean()
    assert S.Normal(mu=True).mu is True and S.Normal(mu=np.bool_(True)).mu is True
    assert S.Wishart(df=np.int64(3), scale=sd.eye(2)).mean().tolist() == [[3.0, 0.0], [0.0, 3.0]]
    assert S.Multinomial(n=np.int64(3), p=A([0.5, 0.5])).mean().tolist() == [1.5, 1.5]
    assert math.isnan(S.skew(np.int64(3))) and math.isnan(S.skew(np.array(2.0)))
    with pytest.raises(TypeError, match=r"^Normal\(\): 'mu' must be a real number or an sd.ndarray, not complex$"):
        S.Normal(mu=1j)


# ---- ST-4: infinite terms --------------------------------------------------------------------


def test_compensated_sums_with_infinities():
    inf = math.inf
    assert S.describe([1.0, inf, 2.0]).mean == inf
    assert S.describe([1.0, -inf, 2.0]).mean == -inf
    assert math.isnan(S.describe([1.0, inf, -inf]).mean)
    assert S.moment([1e160, 2e160, -1e160], order=2, center=0.0) == inf
    assert S.moment([1e70, 2e70, -1e70], order=6) == inf
    assert S.moment([1.0, inf], order=2, center=0.0) == inf
    col = S.describe(sd.asarray(np.array([[1.0, inf], [2.0, 3.0], [4.0, 5.0]])), axis=0).mean.tolist()
    assert col[0] == 7.0 / 3.0 and col[1] == inf


# ---- ST-6, ST-7, ST-11 ---------------------------------------------------------------------


def test_matrix_normal_entropy_against_arb():
    ctx.prec = 200
    for n, p, su, sv in ((100, 100, 1e-300, 1.1e300), (100, 100, 1e-300, 1e300), (3, 2, 2.0, 0.5)):
        M = S.MatrixNormal(mu=sd.zeros((n, p)), rowcov=sd.eye(n) * su, colcov=sd.eye(p) * sv)
        ex = (
            arb(n * p) / 2 * (1 + (2 * arb.pi()).log())
            + arb(p) / 2 * n * arb(su).log()
            + arb(n) / 2 * p * arb(sv).log()
        )
        want = float(ex.mid())
        assert abs(M.entropy() - want) <= 1e-14 * max(1.0, abs(want)), (n, p, su, sv)


def test_covariance_diagonals_do_not_cancel():
    c = S.Dirichlet(alpha=A([1e8, 1e-8])).covariance()
    assert abs(float(c[0, 0]) - 9.9999999e-25) <= 1e-15 * 1e-24 and float(c[0, 1]) == -float(c[0, 0])
    m = S.Multinomial(n=1.0, p=A([1 - 1e-10, 1e-10])).covariance()
    assert abs(float(m[0, 0]) - 9.999999999e-11) <= 1e-15 * 1e-10
    # a0^2 (a0 + 1) beyond the doubles
    big = S.Dirichlet(alpha=A([1e200, 3e200])).covariance().tolist()
    assert all(math.isfinite(v) for row in big for v in row) and big[0][0] > 0.0
    # off the diagonal of a zero probability: +0.0 (0.0 - 0)
    z = S.Multinomial(n=2.0, p=A([0.0, 1.0])).covariance()
    assert math.copysign(1.0, float(z[0, 1])) == 1.0


def test_point_masses_have_zero_entropy():
    for a, b in ((1.0, 1.0), (0.5, 0.5), (2.0, 3.0)):
        assert S.BetaBinomial(n=0.0, a=a, b=b).entropy() == 0.0
    assert S.Dirichlet(alpha=A([2.0])).entropy() == 0.0
    assert S.Dirichlet(alpha=A([[2.0], [0.5]])).entropy().tolist() == [0.0, 0.0]
    assert math.isnan(S.Dirichlet(alpha=A([-1.0])).entropy())


# ---- ST-8, ST-9, CX-5: iqr ------------------------------------------------------------------


def test_iqr_q():
    x = [1.0, 2.0, 5.0, 9.0, 10.0]
    assert S.iqr(x, q=(0.75, 0.25)) == 7.0 == S.iqr(x)
    assert S.iqr(x, q=[np.float64(0.25), np.int64(1)]) == 8.0
    for q, err in (((True, 0.5), TypeError), (("0.25", "0.75"), TypeError), (3, TypeError)):
        with pytest.raises(err, match=r"^iqr\(\): 'q' must be a pair of numbers in \[0, 1\], not "):
            S.iqr(x, q=q)
    for q in ((0.25, 0.5, 0.75), (0.5,), (-0.1, 0.5), (0.2, math.nan)):
        with pytest.raises(ValueError, match=r"^iqr\(\): 'q' must be a pair of numbers in \[0, 1\]$"):
            S.iqr(x, q=q)
    a = sd.asarray(np.random.default_rng(3).random((7, 9)))
    one = S.iqr(a, axis=1, keepdims=True)
    q = sd.quantile(a, A([0.25, 0.75]), axis=1, keepdims=True)
    assert one.tolist() == (q[1] - q[0]).tolist()


# ---- ST-10, CX-4, CX-10, CX-11, CX-14: messages -------------------------------------------


def test_messages():
    with pytest.raises(
        ValueError, match=r"^MultivariateNormal\(\): shapes could not be broadcast together: \(2,\) \(3,\)$"
    ):
        S.MultivariateNormal(mu=sd.zeros((2, 2)), cov=sd.zeros((3, 2, 2)))
    with pytest.raises(ValueError, match=r"^logpdf\(\): shapes could not be broadcast together: \(3,\) \(4,\)$"):
        S.MultivariateNormal(mu=sd.zeros((4, 2)), cov=sd.eye(2)).logpdf(sd.zeros((3, 2)))
    with pytest.raises(OverflowError, match=r"^skew\(\): 'axis' is out of range$"):
        S.skew(sd.zeros((2, 2)), axis=2**70)
    with pytest.raises(OverflowError, match=r"^rankdata\(\): 'axis' is out of range$"):
        S.rankdata(sd.zeros((2, 2)), axis=-(2**70))
    with pytest.raises(OverflowError, match=r"^differential_entropy\(\): 'window_length' is out of range$"):
        S.differential_entropy(sd.arange(10.0), window_length=2**70)
    with pytest.raises(ValueError, match=r"^differential_entropy\(\): 'window_length' must be >= 1$"):
        S.differential_entropy(sd.arange(10.0), window_length=0)
    M = S.MultivariateNormal(mu=sd.zeros(2), cov=sd.eye(2))
    with pytest.raises(ValueError, match=r"^cdf\(\): 'max_points' must be >= 1$"):
        M.cdf(sd.zeros(2), max_points=0)
    with pytest.raises(OverflowError, match=r"^cdf\(\): 'max_points' is out of range$"):
        M.cdf(sd.zeros(2), max_points=-(2**70))
    with pytest.raises(ValueError, match=r"^skew\(\): inhomogeneous nested sequence"):
        S.skew([[1.0, 2.0], [3.0]])
    with pytest.raises(TypeError, match=r"^skew\(\): 'x' must be an sd.ndarray, a sequence of numbers or a frame"):
        S.skew({1: 2})
    with pytest.raises(
        TypeError, match=r"^skew\(\): 'x' must be an sd.ndarray, a sequence of numbers .*, not complex$"
    ):
        S.skew(1j)


# ---- ST-15, CX-17 ------------------------------------------------------------------------


def test_scalar_inputs_give_floats():
    for f in (S.zscore, S.rankdata):
        for v in (3.0, 3, np.float64(3.0), np.array(3.0)):
            assert type(f(v)) is float, (f, v)
    assert S.rankdata(3.0) == 1.0 and math.isnan(S.zscore(3.0))
    assert S.zscore([3.0]).shape == (1,)


# ---- R120 -----------------------------------------------------------------------------------


def test_beta_deep_logcdf_next_to_one():
    ctx.prec = 2500
    for a, b, x in ((1.0, 1e-300, 0.95), (1.0, 1e-300, 1 - 1e-10), (2.0, 1e-306, 0.9000001), (0.5, 1e-295, 0.99)):
        got = S.Beta(a=a, b=b).logcdf(x)
        want = float(arb(x).beta_lower(arb(a), arb(b), 1).log().mid())
        assert abs(got - want) <= 1e-14 * abs(want), (a, b, x, got, want)
        # the upper side by symmetry
        assert S.Beta(a=b, b=a).logccdf(1.0 - x) == pytest.approx(got, rel=1e-13)
    assert S.Beta(a=1.0, b=1e-300).logcdf(0.95) == math.log(S.Beta(a=1.0, b=1e-300).cdf(0.95))
    # where the tail is subnormal the log is still NaN
    assert math.isnan(S.Beta(a=1e3, b=2.0**-1021).logcdf(0.95))
    # an inverse there: the forward check now has the log tail (Arb: the closest double)
    X = S.Beta(a=1e5, b=1e-10)
    r = X.icdf(1e-300)
    assert r == 0.9934092416074045
    assert abs(float(arb(r).beta_lower(arb(1e5), arb(1e-10), 1).mid()) / 1e-300 - 1.0) < 1e-11


# ---- CX-16, D-25 -------------------------------------------------------------------------------


def test_result_types():
    from stardust import _results

    assert S.DescribeResult is _results.DescribeResult and S.ModeResult is _results.ModeResult
    r = S.MultivariateNormal(mu=sd.zeros(2), cov=sd.eye(2)).cdf(sd.zeros(2), return_error=True)
    assert type(r).__name__ == "CdfResult" and r._fields == ("value", "error")
    v, e = r
    assert v == r.value == 0.25 and e == r.error == 0.0
    assert not hasattr(S, "CdfResult")


def test_wishart_variance():
    W = S.Wishart(df=3.0, scale=sd.eye(2))
    assert W.variance().tolist() == [[6.0, 3.0], [3.0, 6.0]]
    assert not hasattr(W, "var") and not hasattr(S.InverseWishart(df=5.0, scale=sd.eye(2)), "var")


# ---- D-4, D-5 ------------------------------------------------------------------------------


def test_parameters_are_read_only_copies():
    mu, cov = sd.zeros(2), sd.eye(2)
    X = S.MultivariateNormal(mu=mu, cov=cov)
    mu[0], cov[0, 0] = 3.0, 100.0
    assert X.mean().tolist() == [0.0, 0.0] and X.cov.tolist() == [[1.0, 0.0], [0.0, 1.0]]
    assert X.covariance().tolist() == X.cov.tolist() and X.logpdf(sd.zeros(2)) == pytest.approx(
        -math.log(2 * math.pi), rel=1e-15
    )
    a = A([2.0])
    G = S.Gamma(a=a)
    a[0] = -1.0
    assert G.mean().tolist() == [2.0] and G.a.tolist() == [2.0]
    for setter in (lambda: G.a.__setitem__(0, 1.0), lambda: X.mu.__setitem__(0, 1.0)):
        with pytest.raises(ValueError, match="read-only"):
            setter()
    # dtype as given (an int64 parameter stays int64)
    assert S.Binomial(n=A([10, 20]), p=0.5).n.dtype == sd.int64


def test_summaries_are_read_only():
    G = S.Gamma(a=A([2.0, 3.0]))
    lo, hi = G.support()
    summaries = [G.mean(), G.variance(), G.standard_deviation(), G.skewness(), G.kurtosis(), G.median(), G.mode()]
    summaries += [G.entropy(), lo, hi]
    M = S.MultivariateT(mu=sd.zeros((2, 3)), shape=sd.eye(3), df=5.0)
    summaries += [M.mean(), M.covariance(), M.entropy(), S.Dirichlet(alpha=A([1.0, 2.0])).covariance()]
    W = S.Wishart(df=A([3.0, 4.0]), scale=sd.eye(2))
    summaries += [W.mean(), W.mode(), W.variance(), W.entropy()]
    for s in summaries:
        with pytest.raises(ValueError, match="read-only"):
            s[(0,) * s.ndim] = 1.0
        assert s.copy().flags["WRITEABLE"] if hasattr(s, "flags") else True
    # the elementwise methods' results stay writable (as out=)
    c = G.cdf(A([1.0, 2.0]))
    c[0] = 0.5
    assert type(S.Gamma(a=2.0).mean()) is float


# ---- D-13: discrete quantiles round-trip --------------------------------------------------


def test_discrete_icdf_of_cdf():
    X = S.Poisson(mu=2.2919601461531838)
    assert X.icdf(X.cdf(2.0)) == 2.0
    rng = np.random.default_rng(7)
    for _ in range(20):
        X = S.Binomial(n=float(rng.integers(1, 100)), p=float(rng.random()))
        k = sd.arange(0.0, 30.0)
        c = np.asarray(X.cdf(k))
        back = np.asarray(X.icdf(sd.asarray(c)))
        for i in range(30):
            if 0.0 < c[i] <= 0.9:  # where the small side is not near the cdf's rounding
                first = i
                while first > 0 and c[first - 1] == c[i]:
                    first -= 1
                assert back[i] == first, (X, i, c[i], back[i])
    # near 1 the contract on the small side 1 - p keeps the exact quantile
    P = S.Poisson(mu=0.0025)
    assert P.icdf(0.9999999999999992) == 5.0


# ---- D-30, D-32 ------------------------------------------------------------------------------


def test_descriptive_stats_take_frame_columns():
    s = sd.frame.Series([1.0, 2.0, 4.0, None])
    assert math.isnan(S.skew(s)) and S.skew(s, nan_policy="omit") == S.skew([1.0, 2.0, 4.0])
    df = sd.frame.DataFrame({"a": [1.0, 2.0, 4.0], "b": [3.0, 5.0, 6.0]})
    assert S.describe(df, axis=0).mean.tolist() == [7.0 / 3.0, 14.0 / 3.0]
    with pytest.raises(TypeError, match=r"^skew\(\): 'x' must have numeric columns$"):
        S.skew(sd.frame.Series(["a", "b"]))


def test_describe_empty():
    d = S.describe(sd.zeros(0))
    assert d.nobs == 0 and all(math.isnan(v) for v in (*d.minmax, d.mean, d.variance, d.skewness, d.kurtosis))
    d = S.describe(sd.zeros((3, 0)), axis=0)
    assert d.nobs == 3 and d.mean.shape == (0,) and d.minmax[0].shape == (0,)
    d = S.describe(sd.zeros((3, 0)), axis=1)
    assert d.nobs == 0 and d.mean.shape == (3,) and np.isnan(np.asarray(d.mean)).all()
    assert S.skew(sd.zeros((3, 0)), axis=0).shape == (0,)


def test_result_objects_are_read_only():
    x = sd.asarray(np.arange(12.0).reshape(3, 4))
    d = S.describe(x, axis=0)
    m = S.mode(x, axis=0)
    r = S.MultivariateNormal(mu=sd.zeros((2, 2)), cov=sd.eye(2)).cdf(sd.zeros(2), return_error=True)
    for a in (*d.minmax, d.mean, d.variance, d.skewness, d.kurtosis, m.mode, m.count, r.value, r.error):
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 1.0
    nobs = S.describe(x, axis=0, nan_policy="omit").nobs
    with pytest.raises(ValueError, match="read-only"):
        nobs[0] = 1
    # without axes: Python numbers
    assert type(S.describe(x).mean) is float and type(S.mode(x).count) is int
