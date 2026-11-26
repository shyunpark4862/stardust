"""Gaussian mixtures (docs/api/cluster.md §5; docs/design/cluster.md §2.5, §3.7, §4.5; decisions CL32-CL35 and those of
milestone CL4 in §1.1, CL75-CL81).

gaussian_mixture is scikit-learn's GaussianMixture: the EM iteration with its four covariance types ('full', 'tied',
'diag', 'spherical'), reg_covar added to the diagonals, its starts ('kmeans': the labels of one kmeans; 'k-means++',
'random', 'random_from_data'), stop (the change of the mean log-likelihood at the E step below tol) and default
values. The kernels (csrc/cluster/mixture.cpp) take the rows in key order (CL7) and compute the quadratic forms in the
difference form (scikit-learn expands them), the second moments about the means and every sum over the points by
CL48's chunk sums, so a fit has the same bits on any number of threads. The result keeps both of scikit-learn's
values: lower_bound (the last E step's mean, scikit-learn's lower_bound_) and loglik (the sum at the final
parameters), and bic and aic as fields.
"""

import collections
import math
import numbers

from .. import _calls
from .. import _core as _sd
from .. import random as _random
from .._calls import function
from ._common import (
    _array,
    check_alloc,
    check_rng,
    choice,
    contiguous,
    integer,
    key,
    labels_back,
    points,
    positive_int,
    real,
    type_name,
)
from ._common import rows as _rows
from ._kmeans import _distinct_check
from ._results import ClusterResult

_core = _sd.cluster
TYPES = ("full", "tied", "diag", "spherical")
INITS = ("kmeans", "k-means++", "random", "random_from_data")
_TYPE_CODE = {t: i for i, t in enumerate(TYPES)}
_INIT_CODE = {"random": 2, "random_from_data": 3}
# the start's kmeans (CL76): scikit-learn's KMeans(n_clusters=k, n_init=1) defaults
_KMEANS_MAX_ITER, _KMEANS_TOL = 300, 1e-4

# The draws of GaussianMixtureResult.sample: the points (*shape, p) and their components (*shape) by label
MixtureSample = collections.namedtuple("MixtureSample", ["X", "labels"])


# ---- arguments ---------------------------------------------------------------------------------------------------


def _nonneg(v, func, arg):
    t = real(v, func, arg)
    if not (t >= 0 and math.isfinite(t)):
        raise ValueError(f"{func}(): '{arg}' must be finite and >= 0, not {t!r}")
    return t


def _float_array(v, func, arg):
    if v is None or isinstance(v, (bool, str, bytes, numbers.Number)):
        raise TypeError(f"{func}(): '{arg}' must be an array, not {type_name(v)}")
    a = _array(v, func, arg)
    if not bool(_sd.all(_sd.isfinite(a))):
        raise ValueError(f"{func}(): '{arg}' must be finite")
    return a


def _shape_check(a, want, func, arg):
    if tuple(a.shape) != want:
        raise ValueError(f"{func}(): '{arg}' must have shape {want}, not {tuple(a.shape)}")


def _weights_init(v, func, k):
    """scikit-learn's _check_weights: (k,), in [0, 1], summing to 1 within 1e-8."""
    if v is None:
        return None
    a = _float_array(v, func, "weights_init")
    _shape_check(a, (k,), func, "weights_init")
    if bool(_sd.any(a < 0)) or bool(_sd.any(a > 1)):
        raise ValueError(f"{func}(): 'weights_init' must be in [0, 1]")
    s = float(_sd.sum(a))
    if not abs(1.0 - s) <= 1e-8:
        raise ValueError(f"{func}(): 'weights_init' must sum to 1 (within 1e-8), not {s!r}")
    return contiguous(a)


def _means_init(v, func, k, p):
    if v is None:
        return None
    a = _float_array(v, func, "means_init")
    if a.ndim == 1 and p == 1:
        a = _sd.reshape(a, (a.shape[0], 1))
    _shape_check(a, (k, p), func, "means_init")
    return contiguous(a)


def _precisions_init(v, func, ctype, k, p):
    """scikit-learn's _check_precisions: the shape of the type; full and tied symmetric (numpy's isclose: within
    1e-8 + 1e-5 |P^T|) and positive definite (the factor's pivots, checked by the kernel), diag and spherical > 0."""
    if v is None:
        return None
    a = _float_array(v, func, "precisions_init")
    want = {"full": (k, p, p), "tied": (p, p), "diag": (k, p), "spherical": (k,)}[ctype]
    _shape_check(a, want, func, "precisions_init")
    if ctype in ("full", "tied"):
        sq = _sd.reshape(a, (-1, p, p))
        t = _sd.permute_dims(sq, (0, 2, 1))
        if not bool(_sd.all(_sd.abs(sq - t) <= 1e-8 + 1e-5 * _sd.abs(t))):
            raise ValueError(f"{func}(): 'precisions_init' must be symmetric")
    elif bool(_sd.any(a <= 0)):
        raise ValueError(f"{func}(): 'precisions_init' must be positive")
    return contiguous(a)


def n_parameters(ctype, k, p):
    """scikit-learn's _n_parameters: the free parameters of the model."""
    cov = {"full": k * p * (p + 1) // 2, "diag": k * p, "tied": p * (p + 1) // 2, "spherical": k}[ctype]
    return cov + p * k + k - 1


# ---- the result --------------------------------------------------------------------------------------------------


class GaussianMixtureResult(ClusterResult):
    """The result of gaussian_mixture (docs/api/cluster.md §5).

    Fields: labels ((n,) int64: each point's component of largest weighted density at the final parameters; first-seen
    numbering, -1 for the rows nan_policy='omit' left out), n_clusters (k), nobs, omitted, weights ((k,)), means
    ((k, p)), covariances and precisions_cholesky (scikit-learn's shapes: full (k, p, p), tied (p, p), diag (k, p),
    spherical (k,); component j is label j), loglik (= objective: the log-likelihood of the points at the final
    parameters), lower_bound (scikit-learn's lower_bound_: the mean log-likelihood at the last E step), bic, aic,
    n_parameters, n_iter, converged, covariance_type, n_init and best_init.
    """

    _TITLE = "Gaussian Mixture Results"
    _METHOD = "Gaussian mixture (EM)"
    _OBJECTIVE = ("Log-likelihood:", "loglik")
    _TUNING = (
        ("Covariance:", "covariance_type"),
        ("No. Starts:", "n_init"),
        ("Best start:", "best_init"),
        ("Lower bound:", "lower_bound"),
        ("BIC:", "bic"),
        ("AIC:", "aic"),
    )
    _COLUMNS = (("Weight", "weights"),)
    _REPR = ("nobs", "n_clusters", "loglik", "n_iter")

    def _estimate(self, X_new, func):
        Xn = points(X_new, func, "X_new")
        p = self.means.shape[1]
        if Xn.shape[1] != p:
            raise ValueError(f"{func}(): 'X_new' must have {p} columns (the fit's), not {Xn.shape[1]}")
        if not bool(_sd.all(_sd.isfinite(Xn))):
            raise ValueError(f"{func}(): 'X_new' must be finite (no NaN or infinity)")
        out = _core.gmm_estimate(
            Xn, self._fit_means, self._fit_U, self._fit_logdet, self._fit_logw, self._type, self._fit_scale
        )
        if out is None:
            raise ValueError(f"{func}(): 'X_new' overflows on the fit's scale (its points are too far from the data)")
        return out

    def predict(self, X_new, /):
        """The component of largest weighted density of each new point ((m,) int64; on a tie the component the fit
        numbered first, so the fit's own points get the fit's labels)."""
        lab, _, _ = self._estimate(X_new, "predict")
        return _sd.take(self._fit_numbers, lab)

    def predict_proba(self, X_new, /):
        """The responsibilities of the components for new points ((m, k), rows summing to 1; column j is label j)."""
        _, resp, _ = self._estimate(X_new, "predict_proba")
        return contiguous(resp[:, self._fit_perm])

    def score_samples(self, X_new, /):
        """The log density of the mixture at each new point ((m,); scikit-learn's score_samples)."""
        _, _, lpn = self._estimate(X_new, "score_samples")
        return lpn

    def sample(self, shape=None, /, *, rng):
        """Points drawn from the mixture: MixtureSample(X, labels) with X of shape (*shape, p) and labels (*shape)
        (shape None: one point (p,) and an int). Each draw picks its component by the weights, then the component's
        normal (sd.stats' normal sampler); rng gives two 64-bit words a call."""
        func = "sample"
        shp = _random._sample_shape(shape)
        rng = check_rng(rng, func)
        N = math.prod(shp) if shp else 1
        p = int(self.means.shape[1])
        check_alloc(8 * N * (p + 1), func)
        k0, k1 = key(rng)
        X, lab = _core.gmm_sample(
            N, self.weights, self._fit_perm, self._fit_means, self._fit_U, self._type, self._fit_scale, k0, k1
        )
        if not shp:
            return MixtureSample(_sd.reshape(X, (p,)), int(lab.tolist()[0]))
        return MixtureSample(_sd.reshape(X, shp + (p,)), _sd.reshape(lab, shp))


_calls.methods(GaussianMixtureResult)


# ---- gaussian_mixture ----------------------------------------------------------------------------------------------


def _ill(func, ctype, comp):
    if comp < 0:
        return ValueError(f"{func}(): 'precisions_init' must be positive definite (component {-1 - comp} is not)")
    from ..linalg import LinAlgError

    what = (
        "the tied covariance is ill-defined" if ctype == "tied" else f"component {comp} has an ill-defined covariance"
    )
    return LinAlgError(f"{func}(): {what}; increase 'reg_covar' or check for duplicate points")


@function
def gaussian_mixture(
    X,
    k,
    /,
    *,
    rng,
    covariance_type="full",
    tol=0.001,
    reg_covar=1e-06,
    max_iter=100,
    n_init=1,
    init="kmeans",
    weights_init=None,
    means_init=None,
    precisions_init=None,
    nan_policy="raise",
):
    """A Gaussian mixture of k components fitted by EM (scikit-learn's GaussianMixture).

    X holds the points ((n, p), a 1-D X is (n, 1); at least 2); k the components (1 .. n). rng (an
    sd.random.Generator, required) draws the starts: two 64-bit words a call. covariance_type 'full' (one matrix a
    component), 'tied' (one shared), 'diag' (a diagonal a component) or 'spherical' (one variance a component);
    reg_covar is added to the diagonals. Each of n_init starts takes its responsibilities from init ('kmeans': one
    kmeans' labels; 'k-means++': k-means++'s k points; 'random': random responsibilities; 'random_from_data': k
    random points), replaced by weights_init, means_init and precisions_init where given (all three given: no
    draw), then iterates E and M steps until the mean log-likelihood changes by less than tol (at most max_iter
    times; 0 keeps the start). The start of largest lower bound is kept. nan_policy 'raise' (the default) or 'omit'.

    The quadratic forms are computed in the difference form and the covariances about the means (scikit-learn
    expands both); the same points in another order give the same fit, and any thread count the same bits. A
    covariance that is not positive definite after reg_covar is a LinAlgError.
    """
    func = "gaussian_mixture"
    X = points(X, func)
    n_input = X.shape[0]
    X, _, omitted = _rows(X, func, nan_policy)
    n, p = X.shape
    k = positive_int(k, func, "k")
    if n < 2:
        raise ValueError(f"{func}(): 'X' must have at least 2 points, not {n}")
    if k > n:
        raise ValueError(f"{func}(): 'k' ({k}) is larger than the number of points ({n})")
    ctype = choice(covariance_type, func, "covariance_type", TYPES)
    tol = _nonneg(tol, func, "tol")
    reg = _nonneg(reg_covar, func, "reg_covar")
    max_iter = integer(max_iter, func, "max_iter")
    if max_iter < 0:
        raise ValueError(f"{func}(): 'max_iter' must be >= 0, not {max_iter}")
    n_init = positive_int(n_init, func, "n_init")
    init = choice(init, func, "init", INITS)
    w0 = _weights_init(weights_init, func, k)
    m0 = _means_init(means_init, func, k, p)
    P0 = _precisions_init(precisions_init, func, ctype, k, p)
    rng = check_rng(rng, func)
    k0, k1 = key(rng)
    order = _core.lex_order(X)
    data = None
    if w0 is not None and m0 is not None and P0 is not None:
        code = 4  # nothing to draw (scikit-learn computes no responsibilities either)
    elif init == "kmeans":
        _distinct_check(X, order, None, k, func)
        code = 0
        data = _sd.empty((n_init, n), dtype=_sd.int64)
        for s in range(n_init):
            s0, s1 = _core.subkey(k0, k1, s)  # the start's kmeans: its two words at start s (CL41)
            out = _core.kmeans(X, order, None, k, 0, None, 1, _KMEANS_MAX_ITER, _KMEANS_TOL, s0, s1, 0)
            data[s] = out[0][0]
    elif init == "k-means++":
        code = 1
        data = _sd.empty((n_init, k), dtype=_sd.int64)
        for s in range(n_init):
            data[s] = _core.kmeans_plusplus(X, order, k, k0, k1, s)
    else:
        code = _INIT_CODE[init]
    out = _core.gmm_fit(
        X, order, k, _TYPE_CODE[ctype], code, data, w0, m0, P0, n_init, max_iter, tol, reg, k0, k1
    )  # fmt: skip
    if out[0] is None:
        raise _ill(func, ctype, int(out[1]))
    lab_key, w, mu, cov, U, lb, ll, n_iter, converged, best_init, e, mu_s, U_s, logdet_s, logw = out
    labels, number = _core.first_seen(lab_key, order, k)
    perm = _sd.empty((k,), dtype=_sd.int64)
    perm[number] = _sd.arange(k, dtype=_sd.int64)  # perm[label] = the fit's component
    per = ctype != "tied"
    npar = n_parameters(ctype, k, p)
    om = omitted if omitted is not None else _sd.zeros((n_input,), dtype=_sd.bool)
    return GaussianMixtureResult(
        labels=labels_back(labels, omitted),
        n_clusters=k,
        nobs=n,
        omitted=om,
        weights=contiguous(w[perm]),
        means=contiguous(mu[perm]),
        covariances=contiguous(cov[perm]) if per else cov,
        precisions_cholesky=contiguous(U[perm]) if per else U,
        loglik=float(ll),
        objective=float(ll),
        lower_bound=float(lb),
        bic=-2.0 * float(ll) + npar * math.log(n),
        aic=-2.0 * float(ll) + 2.0 * npar,
        n_parameters=npar,
        n_iter=int(n_iter),
        converged=bool(converged),
        covariance_type=ctype,
        n_init=n_init,
        best_init=int(best_init),
        _type=_TYPE_CODE[ctype],
        _fit_means=mu_s,
        _fit_U=U_s,
        _fit_logdet=logdet_s,
        _fit_logw=logw,
        _fit_scale=int(e),
        _fit_numbers=number,
        _fit_perm=perm,
    )
