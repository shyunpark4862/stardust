"""Matrix distributions (T10, docs/api/stats.md §3.1, docs/design/stats.md §6.3; decisions T30).

Wishart(df=, scale=), InverseWishart(df=, scale=) and MatrixNormal(mu=, rowcov=, colcov=),
with scipy's names (mu for the matrix normal's mean: T18, and mean() is a method) and methods (pdf, logpdf, sample, mean, mode, var, entropy; the matrix
normal without mode and var). Batched as the multivariate ones (T29): the leading axes of the
parameters broadcast, an input x (..., p, p) (or (..., n, p)) broadcasts its leading axes
against them. A matrix argument's lower triangle is read; x not positive definite is off the
support (pdf 0). Invalid parameters (df <= p - 1, a scale that is not positive definite) give
NaN in every method of that batch element (T8).
"""

import math

from .. import _core as _sd_core
from ._multivariate import _array, _broadcast, _fast_factors, _rows, _shape_of, _type_name

_k = _sd_core.stats


class _MatrixBase:
    __slots__ = ("_p", "_batch", "_B", "_shape", "_ok")
    _names = ()

    def __repr__(self):
        args = ", ".join(f"{n}={v!r}" for n, v in zip(self._names, self._p))
        return f"{type(self).__name__}({args})"

    def _points(self, x, name):
        sd = _sd_core
        cls = type(self).__name__
        a = _array(x, f"{name}()", "x", 2)
        if tuple(a.shape[-2:]) != self._shape:
            raise ValueError(f"{name}(): 'x' must be {self._shape} on its last two axes, not {tuple(a.shape[-2:])}")
        out = _broadcast(tuple(a.shape[:-2]), self._batch, what=f"{name}(): 'x' and the parameters")
        M = math.prod(out)
        rows = sd.reshape(sd.broadcast_to(a, out + self._shape), (M,) + self._shape)
        if self._batch:
            idx = sd.reshape(sd.broadcast_to(sd.reshape(sd.arange(self._B), self._batch), out), (M,))
        else:
            idx = sd.zeros((M,), dtype=sd.int64)
        return rows, idx, out

    def _result(self, v, shape):
        return float(v[0]) if not shape else _sd_core.reshape(v, shape)

    def _per_batch(self, v):
        return float(v[0]) if not self._batch else _sd_core.reshape(v, self._batch)

    def _matrices(self, v):
        """(B,) + shape rows with NaN where invalid, in the batch shape."""
        sd = _sd_core
        ok = sd.reshape(self._ok, (self._B, 1, 1))
        return sd.reshape(sd.where(ok != 0.0, v, float("nan")), self._batch + tuple(v.shape[1:]))

    def _sample_rows(self, shape, rng):
        from .. import random as _random

        if not isinstance(rng, _random.Generator):
            raise TypeError(f"sample(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), "
                            f"not {_type_name(rng)}")
        shp = _random._shape(shape, "sample")
        N = math.prod(shp) * self._B
        return shp, N, rng._reserve(N)


def _diverge(m):
    """Entries that diverge: +-inf by the sign, 0 where the entry is 0."""
    sd = _sd_core
    return sd.where(m == 0.0, 0.0, sd.where(m > 0.0, float("inf"), -float("inf")))


class _WishartBase(_MatrixBase):
    __slots__ = ("_df", "_scale", "_L", "_Ll", "_ldh", "_ldl")
    _names = ("df", "scale")
    _inverse = False

    def __init__(self, *, df, scale):
        sd = _sd_core
        cls = type(self).__name__
        df_v = _array(df, cls, "df", 0)
        sc = _array(scale, cls, "scale", 2)
        p = sc.shape[-1]
        if sc.shape[-2] != p:
            raise ValueError(f"{cls}: 'scale' must be square on its last two axes, not {tuple(sc.shape[-2:])}")
        self._p = (df, scale)
        batch = _broadcast(_shape_of(df_v), tuple(sc.shape[:-2]), what=f"{cls}: the parameters")
        self._batch, self._B, self._shape = batch, math.prod(batch), (p, p)
        self._df = _rows(df_v, batch, ())
        self._scale = _rows(sc, batch, (p, p))
        L, Ll, ldh, ldl, ok, _, _ = _k.mv_chol(self._scale)
        good = (self._df > p - 1.0) & sd.isfinite(self._df)
        self._ok = sd.where(good, ok, 0.0)
        self._L, self._Ll, self._ldh, self._ldl = L, Ll, ldh, ldl

    @property
    def df(self):
        """The degrees of freedom, as given (read-only)."""
        return self._p[0]

    @property
    def scale(self):
        """The scale matrix, as given (read-only)."""
        return self._p[1]

    def _log(self, x, name, as_pdf):
        rows, idx, out = self._points(x, name)
        v = _k.wishart_logpdf(rows, idx, self._df, self._L, self._Ll, self._ldh, self._ldl, self._ok, self._inverse,
                              as_pdf)
        return self._result(v, out)

    def pdf(self, x, /):
        """The density at the symmetric matrix x (..., p, p), its lower triangle read; 0 where x is
        not positive definite."""
        return self._log(x, "pdf", True)

    def logpdf(self, x, /):
        """log pdf(x), in double-double inside."""
        return self._log(x, "logpdf", False)

    def entropy(self):
        """The differential entropy, in nats."""
        p = self._shape[0]
        return self._per_batch(_k.wishart_entropy(self._df, self._ldh, self._ldl, self._ok, p, self._inverse))

    def sample(self, shape=(), /, *, rng):
        """Symmetric positive definite matrices of shape `shape` + the batch shape + (p, p)
        (Bartlett's decomposition), one stream element per matrix."""
        shp, N, (key, pos) = self._sample_rows(shape, rng)
        v = _k.wishart_sample(key[0], key[1], pos, N, self._df, self._L, self._ok, self._inverse)
        return _sd_core.reshape(v, shp + self._batch + self._shape)

    def _dfm(self):
        return _sd_core.reshape(self._df, (self._B, 1, 1))


class Wishart(_WishartBase):
    """The Wishart distribution W_p(df, scale): df > p - 1 real, scale (..., p, p) symmetric
    positive definite (its lower triangle read)."""

    __slots__ = ()

    def mean(self):
        """df scale: (..., p, p)."""
        return self._matrices(self._dfm() * self._scale)

    def mode(self):
        """(df - p - 1) scale for df >= p + 1, else NaN (the density's peak is singular)."""
        sd = _sd_core
        p = self._shape[0]
        df = self._dfm()
        return self._matrices(sd.where(df >= p + 1.0, (df - p - 1.0) * self._scale, float("nan")))

    def var(self):
        """The variance of each entry: df (scale_ij^2 + scale_ii scale_jj)."""
        sd = _sd_core
        p = self._shape[0]
        s = self._scale
        d = sd.linalg.diagonal(s)
        dd = sd.reshape(d, (self._B, p, 1)) * sd.reshape(d, (self._B, 1, p))
        return self._matrices(self._dfm() * (s * s + dd))


class InverseWishart(_WishartBase):
    """The inverse Wishart distribution IW_p(df, scale): X^-1 ~ Wishart(df, scale^-1); df > p - 1
    real, scale (..., p, p) symmetric positive definite (its lower triangle read)."""

    __slots__ = ()
    _inverse = True

    def mean(self):
        """scale / (df - p - 1) for df > p + 1; else it diverges (+-inf by the sign of scale's
        entry, 0 where it is 0), as InverseGamma's mean."""
        sd = _sd_core
        p = self._shape[0]
        df = self._dfm()
        return self._matrices(sd.where(df > p + 1.0, self._scale / (df - p - 1.0), _diverge(self._scale)))

    def mode(self):
        """scale / (df + p + 1)."""
        p = self._shape[0]
        return self._matrices(self._scale / (self._dfm() + p + 1.0))

    def var(self):
        """The variance of each entry, ((df-p+1) s_ij^2 + (df-p-1) s_ii s_jj) / ((df-p)(df-p-1)^2 (df-p-3))
        for df > p + 3; inf for p + 1 < df <= p + 3 (the mean finite), NaN for df <= p + 1 (as
        InverseGamma's variance)."""
        sd = _sd_core
        p = self._shape[0]
        s = self._scale
        d = sd.linalg.diagonal(s)
        dd = sd.reshape(d, (self._B, p, 1)) * sd.reshape(d, (self._B, 1, p))
        df = self._dfm()
        v = ((df - p + 1.0) * s * s + (df - p - 1.0) * dd) / ((df - p) * (df - p - 1.0) ** 2 * (df - p - 3.0))
        return self._matrices(sd.where(df > p + 3.0, v, sd.where(df > p + 1.0, float("inf"), float("nan"))))


class MatrixNormal(_MatrixBase):
    """The matrix normal MN(mu, rowcov, colcov) of n x p matrices: vec(X) ~ N(vec(mu),
    colcov (x) rowcov) (scipy's matrix_normal(mean, rowcov, colcov)); rowcov (..., n, n) and colcov (..., p, p) symmetric positive definite
    (their lower triangles read)."""

    __slots__ = ("_mean", "_U", "_Ul", "_V", "_Vl", "_lu", "_lv", "_fast")
    _names = ("mu", "rowcov", "colcov")

    def __init__(self, *, mu, rowcov, colcov):
        sd = _sd_core
        cls = "MatrixNormal"
        m = _array(mu, cls, "mu", 2)
        u = _array(rowcov, cls, "rowcov", 2)
        v = _array(colcov, cls, "colcov", 2)
        n, p = m.shape[-2], m.shape[-1]
        if tuple(u.shape[-2:]) != (n, n):
            raise ValueError(f"{cls}: 'rowcov' must be ({n}, {n}) on its last two axes (mu is {n} x {p}), "
                             f"not {tuple(u.shape[-2:])}")
        if tuple(v.shape[-2:]) != (p, p):
            raise ValueError(f"{cls}: 'colcov' must be ({p}, {p}) on its last two axes (mu is {n} x {p}), "
                             f"not {tuple(v.shape[-2:])}")
        self._p = (mu, rowcov, colcov)
        batch = _broadcast(tuple(m.shape[:-2]), tuple(u.shape[:-2]), tuple(v.shape[:-2]), what=f"{cls}: the parameters")
        self._batch, self._B, self._shape = batch, math.prod(batch), (n, p)
        self._mean = _rows(m, batch, (n, p))
        U, Ul, luh, lul, oku, _, _ = _k.mv_chol(_rows(u, batch, (n, n)))
        V, Vl, lvh, lvl, okv, _, _ = _k.mv_chol(_rows(v, batch, (p, p)))
        finite = sd.all(sd.reshape(sd.isfinite(self._mean), (self._B, n * p)), axis=1)
        self._ok = sd.where(finite & (oku != 0.0) & (okv != 0.0), 1.0, 0.0)
        self._U, self._Ul, self._V, self._Vl = U, Ul, V, Vl
        self._lu, self._lv = (luh, lul), (lvh, lvl)
        self._fast = None

    @property
    def mu(self):
        """The mean matrix, as given (read-only)."""
        return self._p[0]

    @property
    def rowcov(self):
        """The among-row covariance, as given (read-only)."""
        return self._p[1]

    @property
    def colcov(self):
        """The among-column covariance, as given (read-only)."""
        return self._p[2]

    def _log(self, x, name, as_pdf):
        rows, idx, out = self._points(x, name)
        if self._fast is None:  # the inverse factors of the densities' fast path, on first use
            self._fast = (_fast_factors(self._U, self._Ul, self._ok), _fast_factors(self._V, self._Vl, self._ok))
        v = _k.matnorm_logpdf(rows, idx, self._mean, self._U, self._Ul, self._V, self._Vl, self._lu[0], self._lu[1],
                              self._lv[0], self._lv[1], self._ok, *(None if f is False else f for f in self._fast), as_pdf)
        return self._result(v, out)

    def pdf(self, x, /):
        """The density at x (..., n, p)."""
        return self._log(x, "pdf", True)

    def logpdf(self, x, /):
        """log pdf(x), in double-double inside."""
        return self._log(x, "logpdf", False)

    def mean(self):
        """The mean matrix, broadcast to the batch: (..., n, p)."""
        return self._matrices(self._mean)

    def entropy(self):
        """np/2 (1 + log 2 pi) + p/2 log|rowcov| + n/2 log|colcov|."""
        sd = _sd_core
        n, p = self._shape
        lu = self._lu[0] + self._lu[1]
        lv = self._lv[0] + self._lv[1]
        h = 0.5 * n * p * (1.0 + math.log(2.0 * math.pi)) + 0.5 * p * lu + 0.5 * n * lv
        return self._per_batch(sd.where(self._ok != 0.0, h, float("nan")))

    def sample(self, shape=(), /, *, rng):
        """Matrices of shape `shape` + the batch shape + (n, p): mean + L_U Z L_V^T."""
        shp, N, (key, pos) = self._sample_rows(shape, rng)
        v = _k.matnorm_sample(key[0], key[1], pos, N, self._mean, self._U, self._V, self._ok)
        return _sd_core.reshape(v, shp + self._batch + self._shape)
