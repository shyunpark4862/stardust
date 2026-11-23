"""Kernel density estimation: kde, KDEResult and DensityResult (docs/api/smooth.md §5, docs/design/smooth.md §2.5,
§3.5; decisions SM18-SM22, SM26, SM31, SM115-SM126).

kde(data) fixes the estimator f(x) = sum_j w_j K_H(x - x_j) (weights summing to 1): the points (n,) or (n, p), the
kernel's covariance H from a rule or given, the kernel and the reflection bounds. The result evaluates it with the
names of a distribution (pdf, logpdf, cdf, sample) and on a grid (density). pdf and logpdf are the definition's sum
with a proven cut (relative accuracy, the tails included); density(binned=True) is the linear-binning FFT
approximation with an absolute error bound. The C++ core is csrc/smooth/kde.cpp.
"""

import collections
import math
import numbers

from .. import _calls
from .. import _core as _sd
from .._calls import function
from ._bandwidth import _factor, _moments, _nrd, _nrd0
from ._common import contiguous, policy, real, type_name
from ._common import points as _points

_core = _sd.smooth


def _linalg_diag(a):
    from .. import linalg

    return contiguous(linalg.diagonal(a))


_KERNELS = {"gaussian": 0, "epanechnikov": 1, "rectangular": 2, "triangular": 3, "biweight": 4, "cosine": 5}
_RULES = ("scott", "silverman", "nrd0", "nrd", "sj", "isj", "cv_ls", "cv_ml")
_ONE_D_RULES = ("nrd0", "nrd", "sj", "isj")
_HALF = {
    "epanechnikov": math.sqrt(5.0),
    "rectangular": math.sqrt(3.0),
    "triangular": math.sqrt(6.0),
    "biweight": math.sqrt(7.0),
    "cosine": 1.0 / math.sqrt(1.0 / 3.0 - 2.0 / math.pi**2),
}
_GRID_N = {1: 512, 2: 128, 3: 64}  # density()'s default points per axis (decision SM121)
_TAU = 8.3

DensityResult = collections.namedtuple("DensityResult", ["x", "density"])
DensityResult.__doc__ = """The density on a grid: x (1-D: the grid; n-D: the tuple of the axes' grids) and density
(the values; n-D: (n_1, ..., n_p), density[i, j] at (x[0][i], x[1][j]))."""


def _density_result(x, density):
    """DensityResult(x, density) with its arrays read-only, as every result's arrays (API.md §2.2)."""
    for a in (x if isinstance(x, tuple) else (x,)) + (density,):
        a._set_readonly()
    return DensityResult(x, density)


def _data(func, data, weights, nan_policy):
    """(X (n, p), w or None, keep or None, has_nan, one_d) checked as the area's data (docs/api/smooth.md §1.2)."""
    from ._common import vector
    from ._kreg import _matrix

    nan_policy = policy(nan_policy, func)
    if isinstance(data, numbers.Number) or data is None or isinstance(data, (str, bytes)):
        if isinstance(data, numbers.Number):
            raise ValueError(f"{func}(): 'data' must be 1-D or 2-D, not 0-D")
        raise TypeError(
            f"{func}(): 'data' must be an sd.ndarray, a sequence of numbers, a numpy array or a frame "
            f"Series or DataFrame, not {type_name(data)}"
        )
    X = _matrix(data, func, "data")
    if X.ndim not in (1, 2):
        raise ValueError(f"{func}(): 'data' must be 1-D or 2-D, not {X.ndim}-D")
    one_d = X.ndim == 1
    n = X.shape[0]
    X = contiguous(_sd.reshape(X, (n, 1)) if one_d else X)
    if X.shape[1] == 0:
        raise ValueError(f"{func}(): 'data' must have at least 1 column")
    w = None
    if weights is not None:
        w = vector(weights, func, "weights")
        if w.shape[0] != n:
            raise ValueError(f"{func}(): 'weights' must have one value per point ({n}), not {w.shape[0]}")
    if n == 0:
        raise ValueError(f"{func}(): 'data' must have at least 1 point")
    for arg, a in [("data", X)] + ([("weights", w)] if w is not None else []):
        if bool(_sd.any(_sd.isinf(a))):
            raise ValueError(f"{func}(): '{arg}' contains infinity")
    if w is not None and bool(_sd.any(w < 0)):
        raise ValueError(f"{func}(): 'weights' must be >= 0")
    bad = _sd.any(_sd.isnan(X), axis=1)
    if w is not None:
        bad = _sd.logical_or(bad, _sd.isnan(w))
    keep = None
    if bool(_sd.any(bad)):
        if nan_policy == "raise":
            raise ValueError(f"{func}(): the input contains NaN (nan_policy='raise')")
        if nan_policy == "propagate":
            return X, w, None, True, one_d
        keep = _sd.logical_not(bad)
        X = contiguous(X[keep])
        w = contiguous(w[keep]) if w is not None else None
        if X.shape[0] == 0:
            raise ValueError(f"{func}(): no point is left without NaN (nan_policy='omit')")
    if w is not None and not bool(_sd.any(w > 0)):
        raise ValueError(f"{func}(): at least one weight must be positive")
    return X, w, keep, False, one_d


def _linalg_error(msg):
    from ..linalg import LinAlgError

    return LinAlgError(msg)


class KDEResult:
    """The result of kde(): the fields bw, covariance, factor, neff, nobs, ndim, kernel, bounds, weights and the
    methods pdf, logpdf, cdf, sample, density and marginal. Not a public constructor (docs/api/smooth.md §7)."""

    _FIELDS = ("bw", "covariance", "factor", "neff", "nobs", "ndim", "kernel", "bounds", "weights")

    def __init__(self, state, **fields):
        for v in fields.values():
            if isinstance(v, _sd.ndarray):
                v._set_readonly()
        object.__setattr__(self, "_f", fields)
        object.__setattr__(self, "_s", state)  # None after a NaN (propagate)

    def __getattr__(self, name):
        f = object.__getattribute__(self, "_f")
        if name in f:
            return f[name]
        raise AttributeError(f"KDEResult has no field {name!r}")

    def __setattr__(self, name, value):
        raise AttributeError("KDEResult: the fields are read-only")

    def __delattr__(self, name):
        raise AttributeError("KDEResult: the fields are read-only")

    def __dir__(self):
        return sorted(set(super().__dir__()) | set(self._FIELDS))

    def __repr__(self):
        from .. import _summary

        return _summary.one_line(
            "KDEResult", [("nobs", self.nobs), ("ndim", self.ndim), ("bw", self.bw), ("kernel", self.kernel)]
        )

    # ---- evaluation

    def _queries(self, x, func):
        """(an (m, p) C-contiguous float64 array, the result's shape)."""
        p = self.ndim
        if p == 1:
            pts, shape = _points(x, func, "x")
            return contiguous(_sd.reshape(pts, (pts.shape[0], 1))), shape
        from ._kreg import _matrix

        if isinstance(x, numbers.Number):
            raise ValueError(f"{func}(): 'x' must have {p} columns, not a number")
        a = _matrix(x, func, "x")
        if a.ndim == 1:
            if a.shape[0] != p:
                raise ValueError(f"{func}(): 'x' must have {p} values (one point) or be (m, {p}), not {a.shape[0]}")
            return contiguous(_sd.reshape(a, (1, p))), ()
        if a.ndim != 2 or a.shape[1] != p:
            raise ValueError(f"{func}(): 'x' must be (m, {p}), not {tuple(a.shape)}")
        return contiguous(a), (a.shape[0],)

    def _eval(self, x, func, what):
        if what == 2 and self.ndim != 1:
            raise ValueError("cdf(): the distribution function is for 1-D data")
        q, shape = self._queries(x, func)
        s = object.__getattribute__(self, "_s")
        if s is None:
            v = _sd.full((q.shape[0],), math.nan)
        else:
            v = _core.kde_eval(
                s["Xs"], s["ws"], s["sw"], s["wr"], s["kernel"], s["bwv"], s["H"], q, what, s["lo"], s["hi"]
            )
        if shape == ():
            return float(v[0])
        return _sd.reshape(v, shape)

    def pdf(self, x, /):
        """The density at x: 1-D data take any shape (a number gives a float), n-D data (m, p) or one point (p,). The
        definition's sum with the proven cut: relative error at most (8 + 2p + log2 n + D0) u, D0 the squared
        distance in bandwidths to the nearest point (docs/design/smooth.md §4.5)."""
        return self._eval(x, "pdf", 0)

    def logpdf(self, x, /):
        """log pdf(x), accurate where the density underflows (the largest term is taken out of the exponent)."""
        return self._eval(x, "logpdf", 1)

    def cdf(self, x, /):
        """The distribution function at x (1-D): sum w_j G((x - x_j) / h), G the kernel's distribution function;
        with bounds, that of the reflected estimator."""
        return self._eval(x, "cdf", 2)

    # ---- sampling

    def sample(self, shape=None, /, *, rng):
        """Random points of the estimator: a data point drawn by its weight plus the kernel's noise (with bounds,
        folded back inside them). rng is an sd.random.Generator. shape=None gives one point (a float, or (p,) for
        n-D data); else an array of shape `shape` (+ (p,) for n-D data)."""
        from .. import random as _random
        from ..stats import Beta, Normal

        if not isinstance(rng, _random.Generator):
            raise TypeError(
                f"sample(): 'rng' must be an sd.random.Generator (sd.random.default_rng(seed)), not {type_name(rng)}"
            )
        shp = () if shape is None else _random._sample_shape(shape)
        p = self.ndim
        s = object.__getattribute__(self, "_s")
        count = 1
        for d in shp:
            count *= d
        if s is None:
            out = _sd.full((count, p), math.nan)
        else:
            Xs = s["Xs"]
            m = Xs.shape[0]
            wn = (s["ws"] / s["sw"]) if s["ws"] is not None else _sd.full((m,), 1.0 / m)
            idx = rng.choice(m, shape=(count,), p=contiguous(wn)) if count else _sd.zeros((0,), dtype=_sd.int64)
            base = Xs[idx] if count else _sd.zeros((0, p))
            kernel = self.kernel
            if kernel == "gaussian":
                z = Normal().sample((count, p), rng=rng) if count else _sd.zeros((0, p))
                if s["H"] is None:
                    noise = z * s["bwv"]
                else:
                    A = _sd.linalg.cholesky(s["H"])
                    noise = z @ A.T
            else:
                a = _HALF[kernel] * self.bw
                noise = _sd.reshape(_compact_noise(kernel, count, rng, Beta), (count, 1)) * a
            out = base + noise
            if self.bounds is not None:
                out = _fold(out, *self.bounds)
        if shape is None:
            return float(out[0, 0]) if p == 1 else _sd.reshape(out, (p,))
        return _sd.reshape(out, shp if p == 1 else shp + (p,))

    # ---- the grid

    def _grid_axes(self, n, cut, lower, upper, func):
        p = self.ndim
        s = object.__getattribute__(self, "_s")
        if n is None:
            n = _GRID_N.get(p, 32)
        from ._common import integer

        n = integer(n, func, "n")
        if n < 2:
            raise ValueError(f"{func}(): 'n' must be >= 2")
        cut = real(cut, func, "cut")
        if not (cut >= 0 and math.isfinite(cut)):
            raise ValueError(f"{func}(): 'cut' must be a finite number >= 0")

        def ends(v, name):
            if v is None:
                return [None] * p
            if p == 1 and isinstance(v, numbers.Real) and not isinstance(v, bool):
                return [real(v, func, name)]
            try:
                vals = [real(t, func, name) for t in v]
            except TypeError:
                raise TypeError(
                    f"{func}(): '{name}' must be a number (1-D) or {p} numbers, not {type_name(v)}"
                ) from None
            if len(vals) != p:
                raise ValueError(f"{func}(): '{name}' must have {p} values, not {len(vals)}")
            return vals

        lo_in, hi_in = ends(lower, "lower"), ends(upper, "upper")
        bwv = self.bw if p == 1 else None
        los, his = [], []
        for l in range(p):
            sd_l = float(bwv) if p == 1 else float(self.bw[l])
            if s is None:
                lo, hi = math.nan, math.nan
            else:
                col = s["Xs"][:, l]
                lo = float(_sd.min(col)) - cut * sd_l if lo_in[l] is None else lo_in[l]
                hi = float(_sd.max(col)) + cut * sd_l if hi_in[l] is None else hi_in[l]
                if p == 1 and self.bounds is not None:
                    lo = max(lo, self.bounds[0]) if lo_in[0] is None else lo
                    hi = min(hi, self.bounds[1]) if hi_in[0] is None else hi
                if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    raise ValueError(f"{func}(): the grid needs finite 'lower' < 'upper', not {lo!r} and {hi!r}")
            los.append(lo)
            his.append(hi)
        return n, los, his

    def density(self, *, n=None, cut=3.0, lower=None, upper=None, binned=True):
        """The estimator on an even grid: DensityResult(x, density). The grid spans [min - cut h, max + cut h] per
        axis (lower, upper replace the ends; bounds clip them) with n points per axis (None: 512 in 1-D, 128 in 2-D,
        64 in 3-D). binned=True (R's and statsmodels' way) bins the points linearly and convolves by FFT: absolute
        error at most delta^2 / (8 sqrt(2 pi) h^3) + phi(8.3) / h at a grid point (1-D; delta the step; fewer than
        512 points are computed on 512 and interpolated). binned=False evaluates pdf at every grid point."""
        if not isinstance(binned, bool):
            raise TypeError(f"density(): 'binned' must be a bool, not {type_name(binned)}")
        p = self.ndim
        if p > 3:
            raise ValueError("density(): the grid is for at most 3 variables")
        n, los, his = self._grid_axes(n, cut, lower, upper, "density")
        s = object.__getattribute__(self, "_s")
        axes = []
        for l in range(p):
            delta = (his[l] - los[l]) / (n - 1)
            axes.append(_sd.arange(n, dtype=_sd.float64) * delta + los[l])
        if s is None:
            dens = _sd.full((n,) * p, math.nan)
            return _density_result(axes[0] if p == 1 else tuple(axes), dens)
        nbytes = 8 * n**p
        if nbytes > _sd._physical_memory():
            raise MemoryError(f"density(): cannot allocate {nbytes} bytes: more than this machine's physical memory")
        if not binned:
            if p == 1:
                return _density_result(axes[0], self.pdf(axes[0]))
            mesh = _sd.meshgrid(*axes, indexing="ij")
            pts = contiguous(_sd.stack([_sd.reshape(g, (n**p,)) for g in mesh], axis=1))
            return _density_result(tuple(axes), _sd.reshape(self.pdf(pts), (n,) * p))
        calc = max(n, 512) if p == 1 else n
        if calc != n:
            delta = (his[0] - los[0]) / (calc - 1)
            grid_c = _sd.arange(calc, dtype=_sd.float64) * delta + los[0]
        out = _core.kde_grid(
            s["Xs"], s["ws"], s["sw"], s["kernel"], s["bwv"], s["H"], _sd.asarray(los), _sd.asarray(his),
            _sd.asarray([float(calc)] * p), s["lo"], s["hi"], float(_sd._physical_memory()) / 4,
        )  # fmt: skip
        if out is None:
            raise MemoryError(
                "density(): the binned grid and its FFT arrays need more than a quarter of this "
                "machine's physical memory; give fewer points or binned=False"
            )
        if p == 1:
            if calc != n:
                out = _sd.interp(axes[0], grid_c, out)
            return _density_result(axes[0], out)
        return _density_result(tuple(axes), _sd.reshape(out, (n,) * p))

    # ---- margins

    def marginal(self, dims, /):
        """The marginal density of the variables dims (an int or a sequence of ints) as a KDEResult: the Gaussian
        kernel's covariance restricted to them (the same weights, no rule factor beyond the given one)."""
        p = self.ndim
        if isinstance(dims, numbers.Integral) and not isinstance(dims, bool):
            ds = [int(dims)]
        else:
            try:
                ds = [int(d) if isinstance(d, numbers.Integral) and not isinstance(d, bool) else None for d in dims]
            except TypeError:
                raise TypeError(
                    f"marginal(): 'dims' must be an int or a sequence of ints, not {type_name(dims)}"
                ) from None
            if None in ds:
                raise TypeError("marginal(): 'dims' must be an int or a sequence of ints")
        if not ds:
            raise ValueError("marginal(): 'dims' must name at least one variable")
        ds = [d + p if d < 0 else d for d in ds]
        if any(not 0 <= d < p for d in ds):
            raise ValueError(f"marginal(): 'dims' must be between -{p} and {p - 1}")
        if len(set(ds)) != len(ds):
            raise ValueError("marginal(): 'dims' must not repeat a variable")
        s = object.__getattribute__(self, "_s")
        q = len(ds)
        sel = _sd.asarray(ds)
        cov = self.covariance[sel][:, sel]
        bw = float(self.bw) if p == 1 else self.bw[sel]
        if s is None:
            state = None
        else:
            Xs, ws, sw, wr = _core.kde_prepare(contiguous(s["Xs"][:, sel]), s["ws"])
            if s["H"] is None:
                bwv, H = contiguous(s["bwv"][sel]), None
            else:
                bwv, H = (contiguous(_sd.sqrt(_linalg_diag(cov))), None) if q == 1 else (None, contiguous(cov.copy()))
            state = dict(s, Xs=Xs, ws=ws, sw=sw, wr=wr, bwv=bwv, H=H)
        return KDEResult(
            state,
            bw=(float(bw) if p == 1 else float(bw[0])) if q == 1 else contiguous(_sd.asarray(bw).copy()),
            covariance=contiguous(cov.copy()),
            factor=self.factor,
            neff=self.neff,
            nobs=self.nobs,
            ndim=q,
            kernel=self.kernel,
            bounds=self.bounds if q == 1 and p == 1 else None,
            weights=self.weights,
        )


def _compact_noise(kernel, count, rng, Beta):
    """count draws of the compact kernel of half width 1 (decision SM124)."""
    if count == 0:
        return _sd.zeros((0,))
    if kernel == "rectangular":
        return 2.0 * rng.random((count,)) - 1.0
    if kernel == "triangular":
        return rng.random((count,)) + rng.random((count,)) - 1.0
    if kernel == "biweight":
        return 2.0 * Beta(a=3.0, b=3.0).sample((count,), rng=rng) - 1.0
    if kernel == "epanechnikov":  # Devroye: the median of three uniforms on (-1, 1), unless the third is the largest
        u1, u2, u3 = (2.0 * rng.random((count,)) - 1.0 for _ in range(3))
        a1, a2, a3 = _sd.abs(u1), _sd.abs(u2), _sd.abs(u3)
        return _sd.where(_sd.logical_and(a3 >= a2, a3 >= a1), u2, u3)
    # cosine: the inverse of G(r) = (1 + r + sin(pi r) / pi) / 2 by bisection (60 halvings)
    u = rng.random((count,))
    lo, hi = _sd.full((count,), -1.0), _sd.full((count,), 1.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        g = 0.5 * (1.0 + mid + _sd.sin(math.pi * mid) / math.pi)
        below = g < u
        lo, hi = _sd.where(below, mid, lo), _sd.where(below, hi, mid)
    return 0.5 * (lo + hi)


def _fold(x, lo, hi):
    """x folded into [lo, hi] by reflections (one bound infinite: one reflection)."""
    if math.isinf(lo) and math.isinf(hi):
        return x
    if math.isinf(hi):
        return _sd.where(x < lo, 2.0 * lo - x, x)
    if math.isinf(lo):
        return _sd.where(x > hi, 2.0 * hi - x, x)
    width = hi - lo
    t = _sd.remainder(x - lo, 2.0 * width)
    t = _sd.where(t > width, 2.0 * width - t, t)
    return t + lo


def _bandwidth(func, bw, X, w, p, one_d):
    """(rule or None, bw vector or None, H or None, factor) before adjust."""
    if isinstance(bw, str):
        if bw not in _RULES:
            listed = ", ".join(f"'{r}'" for r in _RULES[:-1]) + f" or '{_RULES[-1]}'"
            raise ValueError(f"{func}(): 'bw' must be a number > 0, {listed}, not {bw!r}")
        return bw, None, None, math.nan
    if isinstance(bw, bool) or bw is None:
        raise TypeError(f"{func}(): 'bw' must be a number > 0, an array or a rule name, not {type_name(bw)}")
    if isinstance(bw, numbers.Real) or (type(bw).__module__ == "numpy" and getattr(bw, "ndim", None) == 0):
        h = real(bw, func, "bw")
        if not (h > 0 and math.isfinite(h)):
            raise ValueError(f"{func}(): 'bw' must be a finite number > 0")
        return None, _sd.full((p,), h), None, math.nan
    from ._common import array

    a = array(bw, func, "bw")
    if a.ndim == 1:
        if a.shape[0] != p:
            raise ValueError(f"{func}(): 'bw' must have {p} values (one per variable), not {a.shape[0]}")
        if not bool(_sd.all(_sd.logical_and(a > 0, _sd.isfinite(a)))):
            raise ValueError(f"{func}(): 'bw' must be finite numbers > 0")
        return None, contiguous(a.copy()), None, math.nan
    if a.ndim == 2 and tuple(a.shape) == (p, p):
        if not bool(_sd.all(_sd.isfinite(a))):
            raise ValueError(f"{func}(): the covariance 'bw' must be finite")
        if not bool(_sd.all(a == a.T)):
            raise ValueError(f"{func}(): the covariance 'bw' must be symmetric")
        return None, None, contiguous(a.copy()), math.nan
    raise ValueError(
        f"{func}(): 'bw' must be a number, {p} numbers or a ({p}, {p}) covariance, not shape {tuple(a.shape)}"
    )


@function
def kde(data, /, *, bw="scott", adjust=1.0, kernel="gaussian", weights=None, bounds=None, nan_policy="propagate"):
    """The kernel density estimate of data ((n,) or (n, p); points are rows) with weights (normalised to sum 1).

    bw: a rule ('scott' (the default) and 'silverman' as scipy's gaussian_kde, with the full weighted covariance;
    the 1-D rules 'nrd0', 'nrd', 'sj', 'isj'; 'cv_ls', 'cv_ml' per variable), a number (the kernel's standard
    deviation for every variable), p numbers (per variable) or a (p, p) covariance. adjust multiplies the
    bandwidth. kernel: 'gaussian', or (1-D) 'epanechnikov', 'rectangular', 'triangular', 'biweight', 'cosine' with
    bw their standard deviation (R's scale). bounds=(lo, hi) (1-D; one may be infinite) reflects the estimator at the
    ends. Returns a KDEResult."""
    func = "kde"
    if not isinstance(kernel, str):
        raise TypeError(f"{func}(): 'kernel' must be a string, not {type_name(kernel)}")
    if kernel not in _KERNELS:
        listed = ", ".join(f"'{k}'" for k in list(_KERNELS)[:-1]) + f" or '{list(_KERNELS)[-1]}'"
        raise ValueError(f"{func}(): 'kernel' must be {listed}, not {kernel!r}")
    adj = real(adjust, func, "adjust")
    if not (adj > 0 and math.isfinite(adj)):
        raise ValueError(f"{func}(): 'adjust' must be a finite number > 0")
    X, w, keep, has_nan, one_d = _data(func, data, weights, nan_policy)
    n, p = X.shape
    if kernel != "gaussian" and p != 1:
        raise ValueError(f"{func}(): kernel={kernel!r} is for 1-D data; several variables take 'gaussian'")
    lo_b, hi_b = -math.inf, math.inf
    if bounds is not None:
        if p != 1:
            raise ValueError(f"{func}(): 'bounds' is for 1-D data")
        try:
            lo_b, hi_b = (real(v, func, "bounds") for v in bounds)
        except TypeError, ValueError:
            raise TypeError(f"{func}(): 'bounds' must be (lo, hi), two numbers") from None
        if not (lo_b < hi_b) or lo_b == math.inf or hi_b == -math.inf:
            raise ValueError(f"{func}(): 'bounds' must have lo < hi")
        if not has_nan and bool(_sd.any(_sd.logical_or(X < lo_b, X > hi_b))):
            raise ValueError(f"{func}(): every point must lie within 'bounds'")
    rule, bwv, H, factor = _bandwidth(func, bw, X, w, p, one_d)
    if rule in _ONE_D_RULES and p != 1:
        raise ValueError(f"{func}(): bw='{rule}' is for 1-D data")
    if rule is not None and rule not in ("scott", "silverman") and weights is not None:
        raise ValueError(f"{func}(): bw='{rule}' takes no weights; give 'bw' as a number or use 'scott' or 'silverman'")
    n_in = keep.shape[0] if keep is not None else n
    wfield = None
    if w is not None:
        wn = w / float(_sd.sum(w)) if not has_nan else w
        wfield = contiguous(_sd.full((n_in,), math.nan)) if keep is not None else None
        if keep is not None:
            wfield[keep] = wn
        else:
            wfield = contiguous(wn.copy())
    if has_nan:
        nan_bw = math.nan if p == 1 else _sd.full((p,), math.nan)
        return KDEResult(
            None,
            bw=nan_bw,
            covariance=_sd.full((p, p), math.nan),
            factor=math.nan,
            neff=math.nan,
            nobs=n,
            ndim=p,
            kernel=kernel,
            bounds=None if bounds is None else (lo_b, hi_b),
            weights=None if w is None else contiguous(w.copy()),
        )
    Xs, ws, sw, wr, mean, cov, neff = _moments(X, w)
    del mean
    if rule in ("scott", "silverman"):
        factor = _factor(rule, neff, p)
        if p == 1:
            v = float(cov[0, 0])
            bwv = _sd.full((1,), factor * math.sqrt(v) if v >= 0 else math.nan)
        else:
            H = contiguous(cov * (factor * factor))
    elif rule is not None:
        from . import _bandwidth as B

        x1 = contiguous(_sd.reshape(X, (n,))) if p == 1 else X
        if rule == "nrd0":
            h = _nrd0(x1) if n >= 2 else math.nan
        elif rule == "nrd":
            h = _nrd(x1) if n >= 2 else math.nan
        elif rule == "sj":
            h = B.bw_sj(x1)
        elif rule == "isj":
            h = B.bw_isj(x1)
        elif rule == "cv_ls":
            h = B.bw_cv_ls(x1 if p == 1 else X)
        else:
            h = B.bw_cv_ml(x1 if p == 1 else X)
        bwv = _sd.full((1,), float(h)) if p == 1 else contiguous(_sd.asarray(h).copy())
    if bwv is not None:
        bwv = contiguous(bwv * adj)
        if not bool(_sd.all(_sd.logical_and(bwv > 0, _sd.isfinite(bwv)))):
            raise _linalg_error(
                f"{func}(): the bandwidth is not positive (the data have no spread); give 'bw' as a number"
            )
        H_field = contiguous(_sd.diag(bwv * bwv))
        bw_field = float(bwv[0]) if p == 1 else contiguous(bwv.copy())
    else:
        H = contiguous(H * (adj * adj))
        try:
            _core.kde_eval(Xs, ws, sw, wr, 0, None, H, _sd.zeros((0, p)), 0, -math.inf, math.inf)
        except ValueError:
            raise _linalg_error(
                f"{func}(): the kernel's covariance is not positive definite (the points lie in a lower dimension "
                f"or the rule gives 0); give 'bw' as numbers"
            ) from None
        if p == 1:  # a (1, 1) covariance is a bandwidth
            bwv, H = _sd.full((1,), math.sqrt(float(H[0, 0]))), None
            H_field = contiguous(_sd.diag(bwv * bwv))
            bw_field = float(bwv[0])
        else:
            H_field = H
            bw_field = contiguous(_sd.sqrt(_linalg_diag(H)))
    state = {
        "Xs": Xs,
        "ws": ws,
        "sw": sw,
        "wr": wr,
        "kernel": _KERNELS[kernel],
        "bwv": bwv,
        "H": H,
        "lo": lo_b,
        "hi": hi_b,
    }
    return KDEResult(
        state,
        bw=bw_field,
        covariance=contiguous(H_field.copy()),
        factor=factor,
        neff=float(neff),
        nobs=n,
        ndim=p,
        kernel=kernel,
        bounds=None if bounds is None else (lo_b, hi_b),
        weights=wfield,
    )


_calls.methods(KDEResult)
