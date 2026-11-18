"""The fast paths of the regression kernels give the bits of the plain loops.

The row passes in double-double take NEON lanes over rows (x beta on a row-major X, four
rows at a time, the rest of a chunk row by row) or over columns (the column sums, four
columns at a time); the column-major packing of the penalized fits writes tiles of rows;
the column norms of the QR and the 'nid' fits at q +- h of a quantile array run on the
pool. Each does the scalar operations in the scalar order, so (docs/design/regression.md
§3.1, §3.5, §3.6):
- a row that repeats an earlier one gets the same fitted value whether it falls in a
  vector group or in a chunk's scalar tail (n = 4096 + 3: the last three rows are a chunk's
  tail, copies of the first three);
- a row-major X and the same X column-major (the column-outer scalar loop) give the same
  bits;
- 1 thread and 10 threads give the same bits;
- a quantile array's 'nid' standard errors are those of the quantiles fitted one by one.
Sizes 2^12 and 2^13 rows (a power-of-two chunk stride for the packing), with and without
weights.
"""

import numpy as np
import pytest

import stardust as sd

R = sd.regression


def a(v):
    return np.asarray(v)


def same(x, y):
    x, y = a(x), a(y)
    return x.shape == y.shape and x.tobytes() == y.tobytes()


def fields(r, names):
    return [a(getattr(r, nm)) for nm in names]


def at_threads(th, f):
    old = sd.get_num_threads()
    sd.set_num_threads(th)
    try:
        return f()
    finally:
        sd.set_num_threads(old)


def design(n, k, seed, tail_copies=False):
    g = np.random.default_rng(seed)
    X = g.standard_normal((n, k)) * g.uniform(0.5, 3.0, k)
    if tail_copies:
        X[-3:] = X[:3]
    return g, X


SIZES = [4096, 8192, 4099]


def check_layouts_threads(fit, X, names):
    """fit(X) at 1 and 10 threads, X row-major and column-major: the same bits."""
    ref = at_threads(1, lambda: fields(fit(np.ascontiguousarray(X)), names))
    for th in (1, 10):
        for XX in (np.ascontiguousarray(X), np.asfortranarray(X)):
            got = at_threads(th, lambda: fields(fit(XX), names))
            for nm, u, v in zip(names, ref, got):
                assert same(u, v), (nm, th, XX.flags["C_CONTIGUOUS"])
    return ref


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("weighted", [False, True])
def test_linear_row_passes(n, weighted):
    g, X = design(n, 7, 1, tail_copies=True)
    y = X @ g.standard_normal(7) + 0.3 + g.standard_normal(n)
    y[-3:] = y[:3]
    w = g.uniform(0.2, 3.0, n)
    w[5:200:7] = 0.0
    w[-3:] = w[:3]

    def fit(XX):
        if weighted:
            return R.wls(sd.asarray(y), sd.asarray(XX), weights=sd.asarray(w), cov_type="HC1")
        return R.ols(sd.asarray(y), sd.asarray(XX), cov_type="HC1")

    params, bse, fitted, resid = check_layouts_threads(fit, X, ["params", "bse", "fittedvalues", "resid"])
    assert same(fitted[-3:], fitted[:3]) and same(resid[-3:], resid[:3])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("family,link", [("binomial", "logit"), ("binomial", "probit"), ("poisson", "log")])
def test_glm_row_passes(n, family, link):
    g, X = design(n, 6, 2, tail_copies=True)
    X /= np.abs(X).max(0)
    eta = X @ g.uniform(-0.5, 0.5, 6)
    if family == "binomial":
        y = (g.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    else:
        y = g.poisson(np.exp(eta)).astype(float)
    y[-3:] = y[:3]
    vw = g.uniform(0.5, 2.0, n)
    vw[-3:] = vw[:3]

    def fit(XX):
        return R.glm(sd.asarray(y), sd.asarray(XX), family=family, link=link, var_weights=sd.asarray(vw))

    out = check_layouts_threads(fit, X, ["params", "bse", "linear_predictor", "deviance", "null_deviance"])
    eta_fit = out[2]
    assert same(eta_fit[-3:], eta_fit[:3])


@pytest.mark.parametrize("n", SIZES)
@pytest.mark.parametrize("method", ["M", "MM"])
def test_rlm_row_passes(n, method):
    g, X = design(n, 4, 3, tail_copies=True)
    y = X @ g.standard_normal(4) + 0.5 + g.standard_normal(n)
    y[g.random(n) < 0.05] += 10.0
    y[-3:] = y[:3]

    def fit(XX):
        rng = sd.random.default_rng(5) if method == "MM" else None
        return R.rlm(sd.asarray(y), sd.asarray(XX), method=method, rng=rng)

    out = check_layouts_threads(fit, X, ["params", "bse", "fittedvalues", "weights", "scale"])
    assert same(out[2][-3:], out[2][:3])


@pytest.mark.parametrize("n", [4096, 8192])
def test_penalized_packing(n):
    # ridge's QR and the naive updates (more than 1,024 columns) pack column-major tiles
    g, X = design(n, 40, 4)
    y = X[:, :5].sum(1) + g.standard_normal(n)
    w = g.uniform(0.5, 2.0, n)
    check_layouts_threads(lambda XX: R.ridge(sd.asarray(y), sd.asarray(XX), alpha=sd.asarray([1e-6, 0.1, 10.0]),
                                             weights=sd.asarray(w), standardize=True), X, ["params", "ssr", "df_model"])
    g, X = design(n, 1030, 5)
    b = np.zeros(1030)
    b[:5] = 1.0
    y = X @ b + g.standard_normal(n)
    check_layouts_threads(lambda XX: R.lasso(sd.asarray(y), sd.asarray(XX), alpha=0.05, weights=sd.asarray(w[:n])),
                          X, ["params", "n_iter", "dual_gap"])
    yb = (g.random(n) < 1 / (1 + np.exp(-(X @ b) / 2))).astype(float)
    check_layouts_threads(lambda XX: R.penalized_glm(sd.asarray(yb), sd.asarray(XX), family="binomial", alpha=0.02,
                                                     l1_ratio=1.0), X, ["params", "deviance", "n_iter"])


@pytest.mark.parametrize("n", [4096, 8192])
@pytest.mark.parametrize("family", ["binomial", "poisson"])
def test_penalized_glm_eval(n, family):
    # eta (NEON rows from b0), the logistic mu and the deviance's softplus
    g, X = design(n, 12, 6, tail_copies=True)
    b = np.zeros(12)
    b[:4] = 0.4
    eta = X @ b
    y = (g.random(n) < 1 / (1 + np.exp(-eta))).astype(float) if family == "binomial" else g.poisson(np.exp(eta / 2)).astype(float)
    w = g.uniform(0.5, 2.0, n)
    for l1 in (1.0, 0.5, 0.0):
        check_layouts_threads(lambda XX: R.penalized_glm(sd.asarray(y), sd.asarray(XX), family=family,
                                                         alpha=sd.asarray([0.05, 0.005]), l1_ratio=l1,
                                                         weights=sd.asarray(w)), X, ["params", "deviance", "dual_gap"])


@pytest.mark.parametrize("n", [4096, 8192])
def test_quantreg_nid_array_tasks(n):
    g, X = design(n, 4, 7)
    y = X @ g.standard_normal(4) + 0.5 + g.standard_t(3, n)
    qs = [0.1, 0.25, 0.5, 0.75, 0.9]
    ys, Xs = sd.asarray(y), sd.asarray(X)
    arr = {th: at_threads(th, lambda: R.quantreg(ys, Xs, q=sd.asarray(qs))) for th in (1, 10)}
    for nm in ("params", "bse", "bandwidth"):
        assert same(getattr(arr[1], nm), getattr(arr[10], nm))
    for i, q in enumerate(qs):
        one = R.quantreg(ys, Xs, q=q)
        assert same(a(arr[10].params)[i], one.params)
        assert same(a(arr[10].bse)[i], one.bse)
