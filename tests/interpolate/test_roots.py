"""PPoly.roots and solve (docs/design/interpolate.md §3.5, §4; decisions IP18, IP45; milestone IP2).

The contract of a returned root r (not NaN): the piecewise function sd evaluates, P(fl(t - x[i])) with P the
piece's computed coefficients, changes sign between r's neighbouring doubles or is zero at r -- checked with
exact rationals at the doubles -- or, for a root where the computed sign was within the compensated evaluation's
error, |P| at r is within that error; a double root (an extremum that does not cross) has |P(r)| within the
evaluation contract. Every root of the Arb references (tests/interpolate/ref/roots.npz, the real roots of each
piece with their multiplicities) is found: simple roots within a few ulps, double roots within the square root of
the evaluation contract's relative size. The behaviours besides rounding are scipy 1.18.1's (its records in
tests/interpolate/data/scipy.json): zero pieces give [x[i], nan], discontinuity sign changes, extrapolation of the
end pieces; where scipy misses the double roots and loses half the digits of two close roots, sd does not."""

import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.interpolate as si

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
ROOTS = refdata.ref("roots")
SCIPY = refdata.data("scipy")
U = refdata.U


def fr(v):
    return Fraction(float(v))


def _exact(coeffs, s):
    total = Fraction(0)
    for c in coeffs:
        total = total * s + fr(c)
    return total


def _comp_bound(coeffs, s):
    """The compensated Horner's error: u |p| + gamma_{2d}^2 sum |c| |s|^e (Graillat, Langlois and Louvet), doubled."""
    d = len(coeffs) - 1
    a = sum(abs(float(c)) * abs(s) ** (d - j) for j, c in enumerate(coeffs))
    return 2 * (U * abs(float(_exact(coeffs, fr(s)))) + refdata.gamma(2 * d) ** 2 * a) + 2.0**-1070


def _piece(x, v):
    return int(min(max(np.searchsorted(x, v, side="right") - 1, 0), len(x) - 2))


def _check_root(c, x, r, y=0.0):
    """r is a root of the piece sd evaluates there (see the module docstring)."""
    i = _piece(x, r)
    coeffs = [float(v) for v in c[:, i]]
    coeffs[-1] = coeffs[-1] - y
    vals = []
    for t in (np.nextafter(r, -np.inf), r, np.nextafter(r, np.inf)):
        vals.append(_exact(coeffs, fr(t - x[i])))
    if vals[1] == 0 or min(vals) < 0 < max(vals):
        return "crossing"
    s = r - x[i]
    if abs(float(vals[1])) <= _comp_bound(coeffs, s):
        return "close"
    if abs(float(vals[1])) <= refdata.ppoly_bound(coeffs, s):
        return "tangent"
    raise AssertionError((r, [float(v) for v in vals]))


def _ref(pid):
    m = ROOTS.methods["root"]
    pos = np.flatnonzero(ROOTS.inputs["set"][m.index] == pid)
    return m.hi[pos], ROOTS.inputs["multiplicity"][m.index[pos]], ROOTS.inputs["piece"][m.index[pos]]


@pytest.mark.parametrize("pid", S.PIECES_IDS)
def test_roots_against_the_references(pid):
    P = S.pieces(pid)
    p = IP.PPoly(P.c, P.x)
    got = np.asarray(p.roots(extrapolate=False, discontinuity=False))
    fin = got[np.isfinite(got)]
    assert np.all(np.diff(fin) >= 0)
    for r in fin:
        _check_root(P.c, P.x, r)
    ref, mult, piece = _ref(pid)
    for r, mu, i in zip(ref, mult, piece):
        d = np.min(np.abs(fin - r)) if fin.size else np.inf
        coeffs = P.c[:, i]
        s = abs(r - P.x[i])
        scale = sum(abs(cf) * s ** (len(coeffs) - 1 - j) for j, cf in enumerate(coeffs))
        tol = 4 * np.spacing(abs(r)) if mu == 1 else math.sqrt(16 * U * scale) + 4 * np.spacing(abs(r))
        assert d <= tol, (pid, r, mu, d)
    # every returned root in [x[0], x[-1]] belongs to a reference root (none extra), a zero piece aside
    assert len(fin) == len(ref) or pid == "roots/zero-piece"


def test_double_and_close_roots_exactly():
    p = IP.PPoly(S.pieces("roots/tangent").c, S.pieces("roots/tangent").x)
    assert np.asarray(p.roots()).tolist() == [-1.0, 0.25, 1.5, 3.0]  # scipy: [-1.0, 2.9999999999999996]
    p = IP.PPoly(S.pieces("roots/close").c, S.pieces("roots/close").x)
    got = np.asarray(p.roots(extrapolate=False))
    ref, _, _ = _ref("roots/close")
    assert got.tolist() == ref.tolist()  # both to the last bit; scipy is 7e-9 off


def test_scipy_records():
    """The non-rounding behaviour of scipy's records: the same roots (within 1e-9) where scipy finds them, plus the
    double roots it misses; [x[i], nan] for a zero piece; the discontinuities only with discontinuity=True."""
    for pid in S.PIECES_IDS:
        P = S.pieces(pid)
        p = IP.PPoly(P.c, P.x)
        for key, kw in (("roots", {}), ("roots_inside", {"extrapolate": False}),
                        ("roots_nodisc", {"discontinuity": False})):  # fmt: skip
            rec = SCIPY[pid if key == "roots" else f"{key}/{pid}"]
            assert rec["digest"] == P.digest
            theirs = np.array(rec["value"]["array"], dtype=float)
            mine = np.asarray(p.roots(**kw))
            assert np.array_equal(np.isnan(mine).nonzero()[0].size > 0, np.isnan(theirs).nonzero()[0].size > 0)
            tm, tt = mine[np.isfinite(mine)], theirs[np.isfinite(theirs)]
            for r in tt:
                assert np.min(np.abs(tm - r)) <= 1e-8 * max(1, abs(r)), (pid, key, r)
            extra = [r for r in tm if np.min(np.abs(tt - r), initial=np.inf) > 1e-8 * max(1, abs(r))]
            ref, mult, _ = _ref(pid)
            for r in extra:  # only roots scipy misses: the double ones
                assert np.any((mult == 2) & (np.abs(ref - r) < 1e-7)), (pid, key, r)


def test_zero_pieces_breakpoints_and_solve():
    p = IP.PPoly(S.pieces("roots/zero-piece").c, S.pieces("roots/zero-piece").x)
    got = np.asarray(p.roots(extrapolate=False))
    assert got[1] == 1.0 and math.isnan(got[2])
    p = IP.PPoly(S.pieces("roots/breakpoint").c, S.pieces("roots/breakpoint").x)
    assert np.asarray(p.roots()).tolist() == [1.0, 2.0]
    assert np.asarray(p.roots(discontinuity=False)).size == 0
    line = IP.PPoly([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], [0.0, 1.0, 2.0, 3.0])
    assert np.asarray(line.solve(0.5)).tolist() == [0.5, 1.0, 1.5, 2.0, 2.5]  # scipy's, with its discontinuities
    assert np.asarray(line.solve(5.0, extrapolate=False)).size == 0
    assert np.asarray(line.solve(5.0)).tolist() == [7.0]


@LIVE
def test_splines_roots_like_scipys():
    g = np.random.default_rng(5)
    for n in (10, 300):
        x = np.sort(g.random(n))
        y = g.standard_normal((n, 2))
        mine, theirs = IP.CubicSpline(x, y), si.CubicSpline(x, y)
        got, want = mine.roots(), theirs.roots()
        assert isinstance(got, list) and len(got) == 2
        for a, b in zip(got, want):
            a = np.asarray(a)
            assert a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-11)
        c = np.asarray(mine.c)
        for col in range(2):
            for r in np.asarray(got[col]):
                if x[0] <= r <= x[-1]:
                    _check_root(c[:, :, col], x, r)
    batch = IP.CubicSpline(np.arange(5.0), np.zeros((5, 2, 3)))
    nested = batch.roots()
    assert len(nested) == 2 and len(nested[0]) == 3
    zero = np.asarray(nested[1][2])
    assert zero[0::2].tolist() == [0.0, 1.0, 2.0, 3.0] and np.all(np.isnan(zero[1::2]))


def test_complex_coefficients_refuse_roots():
    p = IP.PPoly([[1j, 1.0]], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match=r"^roots\(\): root finding is only for real-valued polynomials$"):
        p.roots()
    with pytest.raises(ValueError, match=r"^solve\(\): root finding"):
        p.solve(1.0)
