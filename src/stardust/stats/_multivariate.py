"""Multivariate distributions (T9, docs/api/stats.md §3, docs/design/stats.md §6; decisions T29).

MultivariateNormal(mu=, cov=), MultivariateT(mu=, shape=, df=), Multinomial(n=, p=) and
Dirichlet(alpha=), keyword-only as the univariate ones. Parameters are batched: the leading
axes broadcast (mu (..., d) with cov (..., d, d)), and an input x (..., d) broadcasts its
leading axes against the batch. Parameters outside the domain (a covariance that is not
positive definite, p not summing to 1, ...) give NaN in every method of that batch element
(T8). Results are float64; a result with no axes is a Python float.

The kernels are stardust._core.stats.mv_*, multinomial_*, dirichlet_* (double-double
Cholesky, quadratic forms and log densities) and mv_cdf (Genz; bind/stats_mvcdf.cpp).
"""

import math
import operator

from .. import _core as _sd_core

_k = _sd_core.stats
_SQRT_EPS = math.sqrt(2.0**-52)
_MIN_DF = 2.0**-1022


def _type_name(v):
    return type(v).__name__


def _array(v, cls, arg, min_ndim):
    sd = _sd_core
    if isinstance(v, sd.ndarray):
        a = v
    elif isinstance(v, (bool, int, float)) and min_ndim == 0:
        return float(v)
    else:
        try:
            a = sd.asarray(v)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{cls}: '{arg}' must be an sd.ndarray or a sequence of numbers, not {_type_name(v)}") \
                from e
    if a.dtype == sd.complex128:
        raise TypeError(f"{cls}: '{arg}' must be real, not a complex128 array")
    if a.ndim < min_ndim:
        raise ValueError(f"{cls}: '{arg}' must have at least {min_ndim} dimension{'s' if min_ndim > 1 else ''}")
    return a if a.dtype == sd.float64 else a.astype(sd.float64)


def _shape_of(v):
    return tuple(v.shape) if isinstance(v, _sd_core.ndarray) else ()


def _broadcast(*shapes, what="the parameters"):
    """numpy's broadcast of shapes, or ValueError."""
    nd = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(nd):
        dims = {s[len(s) - nd + i] for s in shapes if len(s) - nd + i >= 0} - {1}
        if len(dims) > 1:
            raise ValueError(f"{what} could not be broadcast together: {', '.join(map(str, shapes))}")
        out.append(dims.pop() if dims else 1)
    return tuple(out)


def _flat(v, shape):
    """v (array or float) broadcast to shape and flattened to (prod(shape[:k]), ...) rows as a
    float64 array; `shape` here is the full target shape and the caller reshapes."""
    sd = _sd_core
    if not isinstance(v, sd.ndarray):
        return sd.full(shape if shape else (1,), float(v))
    return sd.broadcast_to(v, shape) if shape else sd.reshape(v, (1,))


def _rows(v, batch, tail):
    """v broadcast to batch + tail, as (B,) + tail."""
    sd = _sd_core
    B = math.prod(batch)
    full = batch + tail
    if not isinstance(v, sd.ndarray):
        return sd.full((B,) + tail, float(v))
    return sd.reshape(sd.broadcast_to(v, full), (B,) + tail)


_FAST_MAX_DOUBLES = 1 << 23  # 64 MB of inverse factors per distribution


def _fast_factors(L, Ll, ok):
    """The inverse factors of the densities' fast path (_core.stats.mv_fast_prep): (B, size)
    doubles, size = padded(d) (d + 3) with d padded to a multiple of 8, or False where a huge
    batch of them would take more memory than the path is worth (every point then takes the
    double-double path)."""
    B, d = L.shape[0], L.shape[1]
    if B * ((d + 7) // 8 * 8) * (d + 3) > _FAST_MAX_DOUBLES:
        return False
    return _k.mv_fast_prep(L, Ll, ok)


class _Batched:
    """What the multivariate distributions share: the batch shape, the per-point layout and
    the result shapes."""

    __slots__ = ("_p", "_batch", "_B", "_d", "_ok")
    _names = ()

    def __repr__(self):
        args = ", ".join(f"{n}={v!r}" for n, v in zip(self._names, self._p))
        return f"{type(self).__name__}({args})"

    def _points(self, x, name):
        """x (..., d) against the batch: (x rows (M, d), batch index (M,), the result's shape)."""
        sd = _sd_core
        cls = type(self).__name__
        a = _array(x, f"{name}()", "x", 1)
        if a.shape[-1] != self._d:
            raise ValueError(f"{name}(): 'x' must have {self._d} components on its last axis, not {a.shape[-1]}")
        out = _broadcast(tuple(a.shape[:-1]), self._batch, what=f"{name}(): 'x' and the parameters")
        M = math.prod(out)
        rows = sd.reshape(sd.broadcast_to(a, out + (self._d,)), (max(M, 0), self._d))
        if self._batch:
            idx = sd.reshape(sd.broadcast_to(sd.reshape(sd.arange(self._B), self._batch), out), (M,))
        else:
            idx = sd.zeros((M,), dtype=sd.int64)
        return rows, idx, out

    def _result(self, v, shape):
        if not shape:
            return float(v[0])
        return _sd_core.reshape(v, shape)

    def _per_batch(self, v, tail=()):
        if not self._batch and not tail:
            return float(v[0])
        return _sd_core.reshape(v, self._batch + tail)

    def _sample_rows(self, shape, rng):
        from .. import random as _random

        if not isinstance(rng, _random.Generator):
            raise TypeError(f"sample(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), "
                            f"not {_type_name(rng)}")
        shp = _random._shape(shape, "sample")
        N = math.prod(shp) * self._B
        return shp, N, rng._reserve(N)

    def _nan_where_bad(self, v, tail):
        """v (B,) + tail with NaN where the batch element's parameters are invalid."""
        sd = _sd_core
        ok = sd.reshape(self._ok, (self._B,) + (1,) * len(tail))
        return sd.where(ok != 0.0, v, float("nan"))


class _Gaussian(_Batched):
    """MultivariateNormal and MultivariateT: mu, the scale matrix and df (inf: the normal)."""

    __slots__ = ("_mu", "_cov", "_L", "_Ll", "_ldh", "_ldl", "_df", "_ih", "_il", "_fast")

    def _setup(self, cls, mu, cov, df):
        sd = _sd_core
        mu = _array(mu, cls, "mu", 1)
        cov = _array(cov, cls, self._names[1], 2)
        d = mu.shape[-1]
        if cov.shape[-1] != cov.shape[-2] or cov.shape[-1] != d:
            raise ValueError(f"{cls}: '{self._names[1]}' must be ({d}, {d}) on its last two axes (mu has {d} "
                             f"components), not {tuple(cov.shape[-2:])}")
        batch = _broadcast(tuple(mu.shape[:-1]), tuple(cov.shape[:-2]), _shape_of(df), what=f"{cls}: the parameters")
        self._batch, self._B, self._d = batch, math.prod(batch), d
        self._mu = _rows(mu, batch, (d,))
        self._cov = _rows(cov, batch, (d, d))
        self._df = _rows(df, batch, ())
        L, Ll, ldh, ldl, ok, self._ih, self._il = _k.mv_chol(self._cov)
        # df below the smallest normal double is outside the domain too (df / 2 rounds to 0)
        good = sd.all(sd.isfinite(self._mu), axis=1) & ~sd.isnan(self._df) & (self._df >= _MIN_DF)
        self._ok = sd.where(good, ok, 0.0)
        self._L, self._Ll, self._ldh, self._ldl = L, Ll, ldh, ldl
        self._fast = None

    def _log(self, x, name, as_pdf):
        rows, idx, out = self._points(x, name)
        if self._fast is None:  # the inverse factors of the densities' fast path, on first use
            self._fast = _fast_factors(self._L, self._Ll, self._ok)
        v = _k.mv_logpdf(rows, idx, self._mu, self._L, self._Ll, self._ldh, self._ldl, self._ok, self._df, self._ih,
                         self._il, None if self._fast is False else self._fast, as_pdf)
        return self._result(v, out)

    def pdf(self, x, /):
        """The density at x (..., d)."""
        return self._log(x, "pdf", True)

    def logpdf(self, x, /):
        """log pdf(x), in double-double inside (accurate where the density underflows)."""
        return self._log(x, "logpdf", False)

    def cdf(self, x, /, *, atol=1e-6, return_error=False, max_points=None):
        """P(X <= x), every coordinate. For the normal with d <= 3 an exact formula (to ~1e-14);
        else Genz's quasi-Monte Carlo with a fixed seed (deterministic) to `atol`. With
        return_error=True: (value, error estimate)."""
        sd = _sd_core
        cls = type(self).__name__
        if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not atol > 0:
            raise ValueError(f"{cls}.cdf(): 'atol' must be a positive number")
        if return_error is not True and return_error is not False:
            raise TypeError(f"{cls}.cdf(): 'return_error' must be True or False, not {_type_name(return_error)}")
        mp = 0
        if max_points is not None:
            if isinstance(max_points, bool):
                raise TypeError(f"{cls}.cdf(): 'max_points' must be an int or None, not bool")
            mp = operator.index(max_points)
            if mp < 1:
                raise ValueError(f"{cls}.cdf(): 'max_points' must be positive")
        rows, idx, out = self._points(x, "cdf")
        shifted = rows - sd.take(self._mu, idx, axis=0)
        cov = sd.where(sd.reshape(self._ok, (self._B, 1, 1)) != 0.0, self._cov, float("nan"))
        v, err = _k.mv_cdf(cov, shifted, idx, self._df, float(atol), mp)
        if return_error:
            return self._result(v, out), self._result(err, out)
        return self._result(v, out)

    def entropy(self):
        """The differential entropy, in nats."""
        return self._per_batch(_k.mv_entropy(self._ldh, self._ldl, self._ok, self._df, self._d))

    def sample(self, shape=(), /, *, rng):
        """Samples of shape `shape` + the batch shape + (d,): mu + L z (and sqrt(df / V) for the
        t), one stream element per vector (the same values on any thread count)."""
        shp, N, (key, pos) = self._sample_rows(shape, rng)
        v = _k.mv_sample(key[0], key[1], pos, N, self._mu, self._L, self._ok, self._df)
        return _sd_core.reshape(v, shp + self._batch + (self._d,))


class MultivariateNormal(_Gaussian):
    """The multivariate normal N(mu, cov): mu (..., d), cov (..., d, d) symmetric positive
    definite (its lower triangle is read, as sd.linalg.cholesky)."""

    __slots__ = ()
    _names = ("mu", "cov")

    def __init__(self, *, mu, cov):
        self._p = (mu, cov)
        self._setup("MultivariateNormal", mu, cov, float("inf"))

    @property
    def mu(self):
        """The mean, as given (read-only)."""
        return self._p[0]

    @property
    def cov(self):
        """The covariance matrix, as given (read-only)."""
        return self._p[1]

    def mean(self):
        """mu, broadcast to the batch: (..., d)."""
        return _sd_core.reshape(self._nan_where_bad(self._mu, (self._d,)), self._batch + (self._d,))

    def covariance(self):
        """cov, broadcast to the batch: (..., d, d)."""
        return _sd_core.reshape(self._nan_where_bad(self._cov, (self._d, self._d)), self._batch + (self._d, self._d))


class MultivariateT(_Gaussian):
    """The multivariate t with location mu (..., d), scale matrix `shape` (..., d, d) and
    degrees of freedom df > 0 (...; df = inf is MultivariateNormal(mu, shape), T29)."""

    __slots__ = ()
    _names = ("mu", "shape", "df")

    def __init__(self, *, mu, shape, df):
        df_v = _array(df, "MultivariateT", "df", 0)
        self._p = (mu, shape, df)
        self._setup("MultivariateT", mu, shape, df_v)

    @property
    def mu(self):
        """The location, as given (read-only)."""
        return self._p[0]

    @property
    def shape(self):
        """The scale matrix, as given (read-only)."""
        return self._p[1]

    @property
    def df(self):
        """The degrees of freedom, as given (read-only)."""
        return self._p[2]

    def mean(self):
        """mu where df > 1, else NaN: (..., d)."""
        sd = _sd_core
        m = sd.where(sd.reshape(self._df, (self._B, 1)) > 1.0, self._mu, float("nan"))
        return sd.reshape(self._nan_where_bad(m, (self._d,)), self._batch + (self._d,))

    def covariance(self):
        """shape df / (df - 2) where df > 2; for 1 < df <= 2 the entries diverge (inf with the
        sign of shape's, 0 where shape's is 0); NaN for df <= 1 (as StudentT's variance)."""
        sd = _sd_core
        df = sd.reshape(self._df, (self._B, 1, 1))
        finite = self._cov * sd.where(sd.isinf(df), 1.0, df / (df - 2.0))
        diverge = sd.where(self._cov == 0.0, 0.0, sd.where(self._cov > 0.0, float("inf"), -float("inf")))
        c = sd.where(df > 2.0, finite, sd.where(df > 1.0, diverge, float("nan")))
        return sd.reshape(self._nan_where_bad(c, (self._d, self._d)), self._batch + (self._d, self._d))


class Multinomial(_Batched):
    """The multinomial distribution of n trials over K categories with probabilities p
    (..., K): n a non-negative integer (...); p non-negative, summing to 1 within sqrt(eps)
    (then divided by its sum, as sd.random.choice's p; T29)."""

    __slots__ = ("_n", "_pp")
    _names = ("n", "p")

    def __init__(self, *, n, p):
        sd = _sd_core
        n_v = _array(n, "Multinomial", "n", 0)
        p_v = _array(p, "Multinomial", "p", 1)
        self._p = (n, p)
        K = p_v.shape[-1]
        batch = _broadcast(_shape_of(n_v), tuple(p_v.shape[:-1]), what="Multinomial: the parameters")
        self._batch, self._B, self._d = batch, math.prod(batch), K
        self._n = _rows(n_v, batch, ())
        self._pp = _rows(p_v, batch, (K,))
        s = sd.sum(self._pp, axis=1)
        good = (sd.all(self._pp >= 0.0, axis=1) & (sd.abs(s - 1.0) <= _SQRT_EPS)
                & (self._n >= 0.0) & (sd.floor(self._n) == self._n) & sd.isfinite(self._n))
        self._ok = sd.where(good, 1.0, 0.0)

    @property
    def n(self):
        """The number of trials, as given (read-only)."""
        return self._p[0]

    @property
    def p(self):
        """The category probabilities, as given (read-only)."""
        return self._p[1]

    def _log(self, x, name, as_pmf):
        rows, idx, out = self._points(x, name)
        return self._result(_k.multinomial_logpmf(rows, idx, self._n, self._pp, self._ok, as_pmf), out)

    def pmf(self, x, /):
        """P(X = x) for counts x (..., K): 0 unless they are non-negative integers summing to n."""
        return self._log(x, "pmf", True)

    def logpmf(self, x, /):
        """log pmf(x), in double-double inside."""
        return self._log(x, "logpmf", False)

    def _probs(self):
        sd = _sd_core
        return self._pp / sd.reshape(sd.sum(self._pp, axis=1), (self._B, 1))

    def mean(self):
        """n p: (..., K)."""
        sd = _sd_core
        m = sd.reshape(self._n, (self._B, 1)) * self._probs()
        return sd.reshape(self._nan_where_bad(m, (self._d,)), self._batch + (self._d,))

    def covariance(self):
        """n (diag(p) - p p^T): (..., K, K)."""
        sd = _sd_core
        p = self._probs()
        K = self._d
        outer = sd.reshape(p, (self._B, K, 1)) * sd.reshape(p, (self._B, 1, K))
        diag = sd.reshape(p, (self._B, K, 1)) * sd.reshape(sd.eye(K), (1, K, K))
        c = sd.reshape(self._n, (self._B, 1, 1)) * (diag - outer)
        return sd.reshape(self._nan_where_bad(c, (K, K)), self._batch + (K, K))

    def entropy(self):
        """The entropy, in nats."""
        return self._per_batch(_k.multinomial_entropy(self._n, self._pp, self._ok))

    def sample(self, shape=(), /, *, rng):
        """Count vectors of shape `shape` + the batch shape + (K,), by conditional binomials."""
        shp, N, (key, pos) = self._sample_rows(shape, rng)
        v = _k.multinomial_sample(key[0], key[1], pos, N, self._n, self._pp, self._ok)
        return _sd_core.reshape(v, shp + self._batch + (self._d,))


class Dirichlet(_Batched):
    """The Dirichlet distribution with concentrations alpha (..., K), all positive and finite,
    on the simplex (x_i >= 0, sum x = 1): x off it (by more than sqrt(eps) K in the sum) has
    density 0 (T29)."""

    __slots__ = ("_alpha",)
    _names = ("alpha",)

    def __init__(self, *, alpha):
        sd = _sd_core
        a = _array(alpha, "Dirichlet", "alpha", 1)
        self._p = (alpha,)
        K = a.shape[-1]
        batch = tuple(a.shape[:-1])
        self._batch, self._B, self._d = batch, math.prod(batch), K
        self._alpha = _rows(a, batch, (K,))
        good = sd.all((self._alpha > 0.0) & sd.isfinite(self._alpha), axis=1)
        self._ok = sd.where(good, 1.0, 0.0)

    @property
    def alpha(self):
        """The concentrations, as given (read-only)."""
        return self._p[0]

    def _log(self, x, name, as_pdf):
        rows, idx, out = self._points(x, name)
        return self._result(_k.dirichlet_logpdf(rows, idx, self._alpha, self._ok, as_pdf), out)

    def pdf(self, x, /):
        """The density at x (..., K) on the simplex, 0 off it."""
        return self._log(x, "pdf", True)

    def logpdf(self, x, /):
        """log pdf(x), in double-double inside."""
        return self._log(x, "logpdf", False)

    def mean(self):
        """alpha / sum(alpha): (..., K)."""
        sd = _sd_core
        m = self._alpha / sd.reshape(sd.sum(self._alpha, axis=1), (self._B, 1))
        return sd.reshape(self._nan_where_bad(m, (self._d,)), self._batch + (self._d,))

    def covariance(self):
        """(a_i a0 delta_ij - a_i a_j) / (a0^2 (a0 + 1)): (..., K, K)."""
        sd = _sd_core
        K = self._d
        a = self._alpha
        a0 = sd.reshape(sd.sum(a, axis=1), (self._B, 1, 1))
        ai = sd.reshape(a, (self._B, K, 1))
        aj = sd.reshape(a, (self._B, 1, K))
        c = (ai * a0 * sd.reshape(sd.eye(K), (1, K, K)) - ai * aj) / (a0 * a0 * (a0 + 1.0))
        return sd.reshape(self._nan_where_bad(c, (K, K)), self._batch + (K, K))

    def entropy(self):
        """The differential entropy, in nats."""
        return self._per_batch(_k.dirichlet_entropy(self._alpha, self._ok))

    def sample(self, shape=(), /, *, rng):
        """Points of the simplex, shape `shape` + the batch shape + (K,): gammas normalized."""
        shp, N, (key, pos) = self._sample_rows(shape, rng)
        v = _k.dirichlet_sample(key[0], key[1], pos, N, self._alpha, self._ok)
        return _sd_core.reshape(v, shp + self._batch + (self._d,))
