"""The result objects of the iterative solvers and the eigensolvers of sd.sparse (docs/api/sparse.md
§6, §7; decisions SP72, SP79 and SP87 in docs/design/sparse.md §1.1).

One class per solver, named after it (CgResult, GmresResult, ..., LsmrResult), on the field
results' base (stardust._results._FieldResult, as sd.optimize's OptimizeResult): every field is a
read-only property and an array field a read-only array, the same at each access without a copy
(API.md §2.2). The classes are not exported (as sd.htest's, H6).

A result unpacks into scipy 1.18.1's tuple (SP72): `x, info = cg(A, b)`, and lsqr's ten and
lsmr's eight values in scipy's order. The fields keep this library's names (optimize O4,
regression): status is scipy's info (cg, bicgstab, gmres, minres) or istop (lsqr, lsmr); n_iter is
scipy's itn (and what cg's, bicgstab's and gmres's info counts); lsqr's r1norm, r2norm, anorm,
acond, arnorm, xnorm are normr, normr_damped, norma, conda, normar, normx, lsmr's names.

summary() is the box every result shares (stardust._summary, API.md §2.3): the solver and how it
stopped, the counts and norms, then x (cut at both ends when long; complex x as its real and
imaginary parts). repr is one line of the main fields.

The eigensolvers' results (EigshResult, EigsResult, SvdsResult; SP87) unpack into scipy's
(w, v) and (u, s, vh), whatever return_eigenvectors / return_singular_vectors asked: a part not
computed is None (the result's form does not change with an option, docs/api/linalg.md §1.6).
The fields are sd.linalg's names (eigenvalues, eigenvectors; U, S, Vh) and the state of the
iteration: converged, status (ARPACK's info: 0, 1 the iteration limit, 3 no shifts could be
applied), message, n_iter (restarts), n_converged and n_op (products with OP, or with A and A^H
for svds). When the iteration stops short, the converged pairs are the result (scipy raises
ArpackNoConvergence with them).
"""

from .. import _calls
from .. import _core as _sd
from .. import _summary as _S
from .._results import _FieldResult


def _iter(self):
    return iter(tuple(getattr(self, f) for f in self._UNPACK))


def _len(self):
    return len(self._UNPACK)


def _getitem(self, i):
    return tuple(getattr(self, f) for f in self._UNPACK)[i]


class _SolveResult(_FieldResult):
    """The base of the solvers' results."""

    __slots__ = ()
    _ABSENT = "the solver does not give it"
    _UNPACK = ("x", "status")  # scipy's tuple (SP72)
    _TITLE = ""
    _REPR = ("converged", "status", "n_iter", "normr", "x")
    _HEAD = ()  # (field, label) of the right column of summary()'s header

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.__iter__, cls.__len__, cls.__getitem__ = _iter, _len, _getitem

    def __init__(self, **fields):
        self._set((), fields, True)  # the caller's arrays are copied once

    @_calls.method("summary")
    def summary(self):
        """The result as a text table: the solver, convergence, status and message, the counts
        and norms, then x (cut at both ends when long)."""
        return _summary(self)

    def __repr__(self):
        parts = []
        for k in self._REPR:
            if k not in self._fields:
                continue
            v = self._fields[k]
            if isinstance(v, _sd.ndarray) and v.dtype == _sd.complex128:  # (compact writes real values)
                text = f"<array of shape {_S.shape_text(tuple(v.shape))}>"
            else:
                text = _S.compact(v)
            parts.append(f"{k}={text}")
        return f"{type(self).__name__}(" + ", ".join(parts) + ")"


class CgResult(_SolveResult):
    """cg's result: x, converged, status, message, n_iter, normr; unpacks as (x, status)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "normr")
    _TITLE = "Conjugate Gradient Results"
    _HEAD = (("n_iter", "No. Iterations:"), ("normr", "Residual norm:"))


class BicgstabResult(_SolveResult):
    """bicgstab's result: x, converged, status, message, n_iter, normr; unpacks as (x, status)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "normr")
    _TITLE = "BiCGSTAB Results"
    _HEAD = (("n_iter", "No. Iterations:"), ("normr", "Residual norm:"))


class GmresResult(_SolveResult):
    """gmres's result: x, converged, status, message, n_iter (restart cycles, the unit of
    max_iter), n_inner (Arnoldi steps), normr; unpacks as (x, status)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "n_inner", "normr")
    _TITLE = "GMRES Results"
    _REPR = ("converged", "status", "n_iter", "n_inner", "normr", "x")
    _HEAD = (("n_iter", "No. Restart Cycles:"), ("n_inner", "No. Inner Iterations:"), ("normr", "Residual norm:"))


class MinresResult(_SolveResult):
    """minres's result: x, converged, status, message, n_iter, normr, norma, conda, normar,
    normx (the estimates of ||b - A x||, ||A||, cond(A), ||A r||, ||x||); unpacks as (x, status)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "normr", "norma", "conda", "normar", "normx")
    _TITLE = "MINRES Results"
    _HEAD = (("n_iter", "No. Iterations:"), ("normr", "Residual norm:"), ("normar", "Norm of A r:"),
             ("norma", "Norm of A:"), ("conda", "Cond. of A:"), ("normx", "Norm of x:"))  # fmt: skip


class LsqrResult(_SolveResult):
    """lsqr's result; unpacks as scipy's (x, istop, itn, r1norm, r2norm, anorm, acond, arnorm,
    xnorm, var) = (x, status, n_iter, normr, normr_damped, norma, conda, normar, normx, var)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "normr", "normr_damped", "norma", "conda", "normar",
               "normx", "var")  # fmt: skip
    _UNPACK = ("x", "status", "n_iter", "normr", "normr_damped", "norma", "conda", "normar", "normx", "var")
    _TITLE = "LSQR Results"
    _HEAD = (("n_iter", "No. Iterations:"), ("normr", "Residual norm:"), ("normr_damped", "Damped residual norm:"),
             ("normar", "Norm of A^H r:"), ("norma", "Norm of A:"), ("conda", "Cond. of A:"),
             ("normx", "Norm of x:"))  # fmt: skip


class LsmrResult(_SolveResult):
    """lsmr's result; unpacks as scipy's (x, istop, itn, normr, normar, norma, conda, normx)."""

    __slots__ = ()
    _FIELDS = ("x", "converged", "status", "message", "n_iter", "normr", "normar", "norma", "conda", "normx")
    _UNPACK = ("x", "status", "n_iter", "normr", "normar", "norma", "conda", "normx")
    _TITLE = "LSMR Results"
    _HEAD = (("n_iter", "No. Iterations:"), ("normr", "Residual norm:"), ("normar", "Norm of A^H r:"),
             ("norma", "Norm of A:"), ("conda", "Cond. of A:"), ("normx", "Norm of x:"))  # fmt: skip


def _num(v):
    return str(v) if isinstance(v, int) else f"{float(v):#.6g}"


def _summary(r):
    f = r._fields
    left = [("Converged:", str(f["converged"])), ("Status:", str(f["status"]))]
    right = [(label, _num(f[k])) for k, label in r._HEAD if k in f]
    wide = [("Message:", f["message"])] if "message" in f else []
    x = f["x"]
    n = x.shape[0]
    cplx = x.dtype == _sd.complex128
    cols = ["real", "imag"] if cplx else ["x"]
    vals = x.tolist() if n <= _S.MAX_ROWS else None
    rows = []
    for i in _S.shown(n):
        if i is None:
            rows.append(None)
            continue
        v = vals[i] if vals is not None else x[i]
        cells = [_S.forg(v.real, 4), _S.forg(v.imag, 4)] if cplx else [_S.forg(v, 4)]
        rows.append((f"x[{i}]", cells))
    tables = [_S.Table(cols, rows)] if n else []
    return _S.render(r._TITLE, header=(left, right), wide_rows=wide, tables=tables)


# ---------------------------------------------------------------- the eigensolvers (SP87)


class _EigenResult(_FieldResult):
    """The base of the eigensolvers' results."""

    __slots__ = ()
    _ABSENT = "the eigensolver does not give it"
    _UNPACK = ("eigenvalues", "eigenvectors")
    _VALUES = "eigenvalues"
    _TITLE = ""
    _OPS = "No. OP x:"

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        cls.__iter__, cls.__len__, cls.__getitem__ = _iter, _len, _getitem

    def __init__(self, **fields):
        self._set((), fields, True)

    @_calls.method("summary")
    def summary(self):
        """The result as a text table: convergence, status and message, the counts, then the
        eigenvalues (singular values) found."""
        f = self._fields
        left = [("Converged:", str(f["converged"])), ("Status:", str(f["status"]))]
        right = [("No. Restarts:", str(f["n_iter"])), (self._OPS, str(f["n_op"])),
                 ("No. Converged:", str(f["n_converged"]))]  # fmt: skip
        vals = f[self._VALUES]
        n = vals.shape[0]
        cplx = vals.dtype == _sd.complex128
        cols = ["real", "imag"] if cplx else ["value"]
        listed = vals.tolist() if n <= _S.MAX_ROWS else None
        rows = []
        for i in _S.shown(n):
            if i is None:
                rows.append(None)
                continue
            v = listed[i] if listed is not None else vals[i]
            rows.append((f"[{i}]",
                         [_S.forg(v.real, 6), _S.forg(v.imag, 6)] if cplx else [_S.forg(v, 6)]))  # fmt: skip
        tables = [_S.Table(cols, rows)] if n else []
        return _S.render(self._TITLE, header=(left, right), wide_rows=[("Message:", f["message"])], tables=tables)

    def __repr__(self):
        parts = []
        for k in ("converged", "status", "n_iter", "n_converged", self._VALUES):
            v = self._fields[k]
            if isinstance(v, _sd.ndarray):
                text = f"<array of shape {_S.shape_text(tuple(v.shape))}>"
            else:
                text = _S.compact(v)
            parts.append(f"{k}={text}")
        return f"{type(self).__name__}(" + ", ".join(parts) + ")"


class EigshResult(_EigenResult):
    """eigsh's result: eigenvalues (float64), eigenvectors (columns; None when not asked for),
    converged, status, message, n_iter, n_converged, n_op; unpacks as (eigenvalues, eigenvectors)."""

    __slots__ = ()
    _FIELDS = ("eigenvalues", "eigenvectors", "converged", "status", "message", "n_iter", "n_converged", "n_op")
    _TITLE = "Lanczos Eigenvalue Results"


class EigsResult(_EigenResult):
    """eigs's result: eigenvalues (complex128), eigenvectors (columns; None when not asked for),
    converged, status, message, n_iter, n_converged, n_op; unpacks as (eigenvalues, eigenvectors)."""

    __slots__ = ()
    _FIELDS = ("eigenvalues", "eigenvectors", "converged", "status", "message", "n_iter", "n_converged", "n_op")
    _TITLE = "Arnoldi Eigenvalue Results"


class SvdsResult(_EigenResult):
    """svds's result: U, S (ascending), Vh (U and Vh None when not asked for), converged, status,
    message, n_iter, n_converged, n_op (products with A and A^H); unpacks as (U, S, Vh)."""

    __slots__ = ()
    _FIELDS = ("U", "S", "Vh", "converged", "status", "message", "n_iter", "n_converged", "n_op")
    _UNPACK = ("U", "S", "Vh")
    _VALUES = "S"
    _TITLE = "Singular Value Results"
    _OPS = "No. A x, A^H x:"


__all__ = ["CgResult", "BicgstabResult", "GmresResult", "MinresResult", "LsqrResult", "LsmrResult", "EigshResult",
           "EigsResult", "SvdsResult"]  # fmt: skip
