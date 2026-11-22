"""The barycentric rationals of sd.interpolate (docs/api/interpolate.md §5; docs/design/interpolate.md §2.6,
§3.6; milestone IP4): AAA and FloaterHormannInterpolator, with their common evaluation, poles, residues and zeros.

A rational is r(z) = sum_j w_j f_j / (z - z_j) / sum_j w_j / (z - z_j) on its support points z_j with values f_j and
weights w_j. Real support points, weights and values at real points are evaluated by the barycentric evaluator
that BarycentricInterpolator uses (csrc/interpolate/polyint.cpp), anything complex by its complex twin; at a
support point the value is f_j, at an infinite point the limit sum w f / sum w (scipy). The poles and zeros are the
finite generalized eigenvalues of scipy's arrowhead pair (LAPACK ggev; decision IP20), the residues scipy's
formula (csrc/interpolate/aaa.cpp).

AAA (scipy's algorithm, step by step): the samples whose value is not finite are dropped and the points sorted
(numpy's unique); it stops at max |f - r| <= rtol max |f| or after min(max_terms, samples) steps, and says which
in `converged` instead of warning (IP20). clean_up() removes the Froissart doublets and refits, returning their
number without a warning; its argument is clean_up_tol, as the constructor's (IP32).

FloaterHormannInterpolator: the samples whose value is not finite are dropped (scipy) and the points sorted
(scipy's unique), and they must be distinct (decision IP63); the weights are scipy's closed form.
"""

import math
import numbers

from .. import _core as _sd
from .._calls import method
from . import _common as C
from ._polyint import _finish, _readonly

_core = _sd.interpolate
_EPS34 = (2.0**-52) ** 0.75


def _cpoints(z, name, arg="z"):
    """Evaluation points that may be complex: (a C-contiguous 1-D float64 or complex128 array, their shape)."""
    sd = _sd
    if isinstance(z, numbers.Number) or C._is_numpy_scalar(z):
        if isinstance(z, bool) or type(z).__name__ == "bool_":
            return sd.asarray([float(z)]), ()
        try:
            v = complex(z)
        except TypeError:
            raise TypeError(f"{name}(): '{arg}' must be a number, not {C.type_name(z)}") from None
        if isinstance(z, numbers.Real):
            return sd.asarray([v.real]), ()
        return sd.asarray([v]), ()
    a = C.array(z, name, arg, complex_ok=True)
    shape = tuple(a.shape)
    return C.contiguous(sd.reshape(a, (a.size,))), shape


def _nonneg(v, name, arg):
    t = C.real(v, name, arg)
    if not (math.isfinite(t) and t >= 0):
        raise ValueError(f"{name}(): '{arg}' must be finite and 0 or more, not {t!r}")
    return t


class _Rational:
    """A barycentric rational: support points _zj (m,), values _fj (m, K), weights _wj (m,)."""

    __slots__ = ("_zj", "_fj", "_wj", "_rest", "_axis", "_poles", "_residues", "_roots")

    def _set(self, zj, fj, wj):
        self._zj, self._fj, self._wj = _readonly(zj), _readonly(fj), _readonly(wj)
        self._poles = self._residues = self._roots = None

    def _real(self):
        f64 = _sd.float64
        return self._zj.dtype == f64 and self._wj.dtype == f64 and self._fj.dtype == f64

    def __call__(self, z, /):
        """The rational's value at z (real or complex)."""
        name = type(self).__name__
        sd = _sd
        flat, shape = _cpoints(z, name)
        if self._zj.shape[0] == 0:
            raise ValueError(f"{name}(): the rational has no support points")
        if flat.dtype == sd.float64 and self._zj.dtype == sd.float64 and self._wj.dtype == sd.float64:
            out = _core.bary_eval(self._zj, self._wj, self._fj, flat, True)
        else:
            c = sd.complex128
            out = _core.rat_eval(C.contiguous(self._zj.astype(c)), C.contiguous(self._wj.astype(c)),
                                 C.contiguous(self._fj.astype(c)), C.contiguous(flat.astype(c)))  # fmt: skip
        return _finish(out, shape, self._rest, self._axis)

    def _one_column(self, name):
        if self._rest != ():
            raise ValueError(f"{name}(): the values must be 1-D, not a batch of shape {self._rest}")
        return _sd.reshape(self._fj, (self._fj.shape[0],))

    def _common(self, *arrays):
        """The arrays in one dtype for the eigenvalue and residue routines."""
        sd = _sd
        if all(a.dtype == sd.float64 for a in arrays):
            return [C.contiguous(a) for a in arrays]
        return [C.contiguous(a.astype(sd.complex128)) for a in arrays]

    def poles(self):
        """The finite poles (complex128): the generalized eigenvalues of scipy's arrowhead pair."""
        if self._poles is None:
            z, w = self._common(self._zj, self._wj)
            self._poles = _readonly(_core.bary_eig(z, w, w))
        return self._poles

    def residues(self):
        """The residues at poles() (complex128); the values must be 1-D."""
        f = self._one_column("residues")
        if self._residues is None:
            z, ff, w = self._common(self._zj, f, self._wj)
            self._residues = _readonly(_core.bary_residues(z, ff, w, C.contiguous(self.poles())))
        return self._residues

    def roots(self):
        """The finite zeros (complex128); the values must be 1-D."""
        f = self._one_column("roots")
        if self._roots is None:
            z, ff, w = self._common(self._zj, f, self._wj)
            self._roots = _readonly(_core.bary_eig(z, w, C.contiguous(w * ff)))
        return self._roots

    @property
    def weights(self):
        """The barycentric weights (m,), read-only."""
        return self._wj


class AAA(_Rational):
    """The AAA rational approximation (Nakatsukasa, Sete, Trefethen) of the samples (x, y), real or complex."""

    __slots__ = ("_zs", "_fs", "_errors", "_converged")

    @method("AAA")
    def __init__(self, x, y, /, *, rtol=None, max_terms=100, clean_up=True, clean_up_tol=1e-13):
        name = "AAA"
        sd = _sd
        tol = _EPS34 if rtol is None else _nonneg(rtol, name, "rtol")
        terms = C.integer(max_terms, name, "max_terms")
        if terms < 1:
            raise ValueError(f"{name}(): 'max_terms' must be 1 or more, not {terms}")
        if not (clean_up is True or clean_up is False):
            raise TypeError(f"{name}(): 'clean_up' must be True or False, not {C.type_name(clean_up)}")
        ctol = _nonneg(clean_up_tol, name, "clean_up_tol")
        z = C.array(x, name, "x", complex_ok=True)
        f = C.array(y, name, "y", complex_ok=True)
        if z.ndim != 1:
            raise ValueError(f"{name}(): 'x' must be 1-D, not {z.ndim}-D")
        if f.ndim != 1:
            raise ValueError(f"{name}(): 'y' must be 1-D, not {f.ndim}-D")
        if f.shape[0] != z.shape[0]:
            raise ValueError(f"{name}(): 'y' must have as many points as 'x' ({z.shape[0]}), not {f.shape[0]}")
        if not _core.all_finite(C.contiguous(z)):
            raise ValueError(f"{name}(): 'x' must be finite")
        if z.dtype != f.dtype:
            z, f = z.astype(sd.complex128), f.astype(sd.complex128)
        zs, fs, zj, fj, wj, errors, converged = _core.aaa_fit(C.contiguous(z), C.contiguous(f), tol, terms)
        if zs.shape[0] == 0:
            raise ValueError(f"{name}(): no sample has a finite value")
        self._zs, self._fs = _readonly(zs), _readonly(fs)
        self._errors = _readonly(errors)
        self._converged = bool(converged)
        self._rest, self._axis = (), 0
        self._set(zj, sd.reshape(fj, (fj.shape[0], 1)), wj)
        if clean_up:
            self._clean(ctol)

    def _clean(self, tol):
        sd = _sd
        f = sd.reshape(self._fj, (self._fj.shape[0],))
        ni, zj, fj, wj = _core.aaa_clean_up(self._zs, self._fs, self._zj, C.contiguous(f), self._wj, tol)
        if ni:
            self._set(zj, sd.reshape(fj, (fj.shape[0], 1)), wj)
        return int(ni)

    def clean_up(self, *, clean_up_tol=1e-13):
        """Removes the Froissart doublets (poles whose |residue| / distance to the samples is below clean_up_tol
        times the geometric mean of |y|) and their nearest support points, refits the weights, and returns how many
        doublets it found."""
        return self._clean(_nonneg(clean_up_tol, "clean_up", "clean_up_tol"))

    @property
    def support_points(self):
        """The support points in the order AAA chose them, read-only."""
        return self._zj

    @property
    def support_values(self):
        """The samples' values at the support points, read-only."""
        return _readonly(C.contiguous(_sd.reshape(self._fj, (self._fj.shape[0],))))

    @property
    def errors(self):
        """max |f - r| over the samples after each step, read-only."""
        return self._errors

    @property
    def converged(self):
        """True when the run stopped at max |f - r| <= rtol max |f| (False: it reached its step limit)."""
        return self._converged

    def __repr__(self):
        return f"AAA(support_points={self._zj.shape[0]}, converged={self._converged})"


class FloaterHormannInterpolator(_Rational):
    """The Floater-Hormann rational interpolant of blending degree d through (points, values)."""

    __slots__ = ()

    @method("FloaterHormannInterpolator")
    def __init__(self, points, values, /, *, d=3, axis=0):
        name = "FloaterHormannInterpolator"
        sd = _sd
        x = C.array(points, name, "points")
        if x.ndim != 1:
            raise ValueError(f"{name}(): 'points' must be 1-D, not {x.ndim}-D")
        dd = C.integer(d, name, "d")
        y = C.array(values, name, "values", complex_ok=True)
        ax = C.axis_index(axis, y.ndim, name)
        n = x.shape[0]
        if y.shape[ax] != n:
            raise ValueError(f"{name}(): 'values' must have as many points as 'points' ({n}) along axis {ax}, not "
                             f"{y.shape[ax]}")  # fmt: skip
        if not _core.all_finite(C.contiguous(x)):
            raise ValueError(f"{name}(): 'points' must be finite")
        if ax:
            y = sd.moveaxis(y, ax, 0)
        rest = tuple(y.shape[1:])
        y2 = sd.reshape(y, (n, math.prod(rest)))
        keep = sd.all(sd.isfinite(y2), axis=1) if y2.shape[1] else sd.ones((n,), dtype=sd.bool)
        x, y2 = x[keep], y2[keep]
        order = sd.argsort(x, stable=True)
        x, y2 = C.contiguous(x[order]), C.contiguous(y2[order])
        m = x.shape[0]
        if m > 1 and bool(sd.any(x[1:] == x[:-1])):
            raise ValueError(f"{name}(): 'points' must be distinct")
        if not 0 <= dd < m:
            raise ValueError(f"{name}(): 'd' must satisfy 0 <= d < n, the number of finite samples ({m}), not {dd}")
        self._rest, self._axis = rest, ax
        self._set(x, y2, _core.fh_weights(x, dd))

    def __repr__(self):
        return f"FloaterHormannInterpolator(points={self._zj.shape[0]})"


def _own_methods(cls):
    """Copies of _Rational's public methods named after cls, so that a wrong call names the public class
    (`AAA.roots() got an unexpected keyword argument ...`, not the private base; API.md §2.1)."""
    import types

    for name in ("__call__", "poles", "residues", "roots"):
        f = getattr(_Rational, name)
        g = types.FunctionType(f.__code__, f.__globals__, f.__name__, f.__defaults__, f.__closure__)
        g.__kwdefaults__, g.__doc__, g.__qualname__ = f.__kwdefaults__, f.__doc__, f"{cls.__name__}.{name}"
        setattr(cls, name, g)


_own_methods(AAA)
_own_methods(FloaterHormannInterpolator)
