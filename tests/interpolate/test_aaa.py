"""AAA (docs/api/interpolate.md §5; docs/design/interpolate.md §3.6, §4; decisions IP20, IP32; milestone IP4).

Its properties: the support values exactly at the support points, a converged object within rtol max|f| at every
sample (sd's evaluation), no warning; its evaluation against the computed rational (support points, values,
weights) evaluated exactly (Fractions on real data) or by Arb (complex data) in units of refdata.rational_bound
(there is no reference file: the weights are sd's own, so the reference is made here; docs/design/interpolate.md
§7.2); and against scipy 1.18.1: the same support points in the same order on every sample set of
tests/interpolate/samples.py, with and without clean_up, the same errors to rounding, the same poles, residues
and zeros to rounding."""

import math
import warnings
from fractions import Fraction

import numpy as np
import pytest

import stardust as sd

from . import refdata
from . import samples as S
from .conftest import LIVE

IP = sd.interpolate
EPS34 = (2.0**-52) ** 0.75


def _finite(s):
    keep = np.isfinite(s.f)
    return s.z[keep], s.f[keep]


@pytest.mark.parametrize("aid", S.AAA_IDS)
def test_properties(aid):
    s = S.aaa(aid)
    z, f = _finite(s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = IP.AAA(s.z, s.f, clean_up=False)
    assert a.converged
    sp, sv = np.asarray(a.support_points), np.asarray(a.support_values)
    assert np.array_equal(np.asarray(a(sp)), sv)  # the support values exactly
    got = np.asarray(a(z))
    assert np.max(np.abs(got - f)) <= EPS34 * np.max(np.abs(f))  # converged: within rtol max|f| everywhere
    assert np.asarray(a.errors)[-1] <= EPS34 * np.max(np.abs(f))
    assert a.weights.dtype == a.support_points.dtype == (sd.complex128 if np.iscomplexobj(s.z) else sd.float64)


def test_not_converged_is_an_attribute_not_a_warning():
    s = S.aaa("aaa/abs")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        a = IP.AAA(s.z, s.f, max_terms=5)
    assert a.converged is False and np.asarray(a.errors).shape == (5,)
    one = IP.AAA([0.0, 1.0, 2.0], [1.0, 1.0, 1.0])  # constant data: one support point, converged at once
    assert one.converged and one(5.0) == 1.0


def _exact_parts(Z, W, F, v):
    lam = [w / (v - z) for z, w in zip(Z, W)]
    den = sum(lam, Fraction(0))
    num = sum((t * f for t, f in zip(lam, F)), Fraction(0))
    return (
        num / den,
        sum((abs(t * f) for t, f in zip(lam, F)), Fraction(0)),
        sum((abs(t) for t in lam), Fraction(0)),
        den,
    )


def _arb_parts(Z, W, F, v):
    from flint import acb, arb, ctx

    old = ctx.prec
    ctx.prec = 300
    try:
        lam = [acb(complex(w)) / (acb(complex(v)) - acb(complex(z))) for z, w in zip(Z, W)]
        den = sum(lam[1:], lam[0])
        num = sum((t * acb(complex(f)) for t, f in zip(lam[1:], F[1:])), lam[0] * acb(complex(F[0])))
        sf = sum((abs(t * acb(complex(f))) for t, f in zip(lam, F)), arb(0))
        s = sum((abs(t) for t in lam), arb(0))
        return num / den, sf, s, den
    finally:
        ctx.prec = old


def _points(z):
    """The samples, the midpoints of neighbours and a few points beyond (complex data: off the curve too)."""
    if np.iscomplexobj(z):
        return np.concatenate([z, 0.9 * z[::7], 1.2 * z[::11], [0.3 + 0.2j, 2.0 - 1.0j]])
    zs = np.sort(z)
    return np.concatenate([z, (zs[1:] + zs[:-1]) / 2, [zs[0] - 0.5, zs[-1] + 0.25]])


@pytest.mark.parametrize("aid", S.AAA_IDS)
def test_evaluation_contract(aid):
    s = S.aaa(aid)
    a = IP.AAA(s.z, s.f)
    zj, fj, wj = (np.asarray(v) for v in (a.support_points, a.support_values, a.weights))
    m = zj.size
    q = _points(_finite(s)[0])
    got = np.asarray(a(q))
    real = not np.iscomplexobj(zj)
    if real:
        Z, W, F = ([Fraction(float(v)) for v in arr] for arr in (zj, wj, fj))
    worst = 0.0
    for k, v in enumerate(q):
        if v in zj:
            assert got[k] == fj[list(zj).index(v)]
            continue
        if real:
            r, sf, s_, den = _exact_parts(Z, W, F, Fraction(float(v)))
            err = float(abs(Fraction(float(got[k])) - r))
            b = refdata.rational_bound(m, float(sf), float(s_), float(den), float(r), m + 3, m + 2, m + 1)
        else:
            from flint import acb

            r, sf, s_, den = _arb_parts(zj, wj, fj, v)
            err = float(abs(acb(complex(got[k])) - r).upper())
            rr = float(abs(r).upper())
            b = refdata.rational_bound(m, float(sf.upper()), float(s_.upper()), float(abs(den).lower()), rr,
                                       3 * m + 16, 2 * m + 16, 2 * m + 8)  # fmt: skip
        u = err / b
        assert u <= 1, (aid, k, u)
        worst = max(worst, u)
    assert worst > 1e-4


@LIVE
@pytest.mark.parametrize("aid", S.AAA_IDS)
def test_same_support_points_as_scipy(aid):
    import scipy.interpolate as si

    s = S.aaa(aid)
    for clean in (False, True):
        a, b = IP.AAA(s.z, s.f, clean_up=clean), si.AAA(s.z, s.f, clean_up=clean)
        assert np.array_equal(np.asarray(a.support_points), b.support_points), (aid, clean)
        assert np.array_equal(np.asarray(a.support_values), b.support_values)
        e, eb = np.asarray(a.errors), b.errors
        assert e.shape == eb.shape and np.allclose(e, eb, rtol=1e-6, atol=1e-12 * eb.max())
        assert np.allclose(np.asarray(a.weights) / b.weights, 1, rtol=1e-6) or np.allclose(
            np.asarray(a.weights) / b.weights, -1, rtol=1e-6
        )  # the singular vector's sign is LAPACK's
        q = _points(_finite(s)[0])
        assert np.allclose(np.asarray(a(q)), b(q), rtol=1e-9, atol=1e-9 * np.max(np.abs(b(q)[np.isfinite(b(q))])))


@LIVE
def test_poles_residues_roots_follow_scipy():
    import scipy.interpolate as si

    for aid in ("aaa/tan", "aaa/circle", "aaa/nonfinite"):
        s = S.aaa(aid)
        a, b = IP.AAA(s.z, s.f), si.AAA(s.z, s.f)
        for name in ("poles", "roots"):
            got, want = np.sort_complex(np.asarray(getattr(a, name)())), np.sort_complex(getattr(b, name)())
            assert got.shape == want.shape and np.allclose(got, want, rtol=1e-6, atol=1e-8), (aid, name)
        pa, pb = np.asarray(a.poles()), b.poles()
        ra, rb = np.asarray(a.residues()), b.residues()
        ia, ib = np.lexsort((pa.imag.round(6), pa.real.round(6))), np.lexsort((pb.imag.round(6), pb.real.round(6)))
        assert np.allclose(ra[ia], rb[ib], rtol=1e-5, atol=1e-8, equal_nan=True), aid
        assert a.poles().dtype == a.residues().dtype == a.roots().dtype == sd.complex128
    c = IP.AAA(S.aaa("aaa/circle").z, S.aaa("aaa/circle").f)
    k = int(np.argmin(np.abs(np.asarray(c.poles()) - 1.5)))  # exp(z) / (z - 1.5): the pole 1.5, residue e^1.5
    assert abs(c.poles()[k] - 1.5) < 1e-9 and abs(c.residues()[k] - math.exp(1.5)) < 1e-8


@LIVE
def test_clean_up_counts_without_warning():
    import scipy.interpolate as si

    s = S.aaa("aaa/circle")
    a = IP.AAA(s.z, s.f, clean_up=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert a.clean_up() == 0  # nothing to remove: no warning, the object as it was
    # scipy's example: rtol = 0 runs into rounding and leaves doublets; clean_up removes them
    z = np.exp(1j * 2 * np.pi * np.linspace(0, 1, num=1000))
    f = np.log(2 + z**4) / (1 - 16 * z**4)
    a = IP.AAA(z, f, rtol=0, max_terms=30, clean_up=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = si.AAA(z, f, rtol=0, max_terms=30, clean_up=False)
    assert np.array_equal(np.asarray(a.support_points), b.support_points)  # above the noise: scipy's steps
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        n = a.clean_up(clean_up_tol=1e-13)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nb = b.clean_up(1e-13)
    assert n == nb
    assert np.asarray(a.support_points).size == b.support_points.size


def test_samples_dropped_and_sorted_as_scipy():
    z = np.array([0.5, 0.1, 0.9, 0.1, 0.3])
    f = np.array([1.0, 2.0, np.nan, 3.0, 4.0])  # a NaN sample is dropped; a repeated point keeps its first sample
    a = IP.AAA(z, f)
    assert set(np.asarray(a.support_points)) <= {0.1, 0.3, 0.5}
    assert a(0.1) == 2.0
    with pytest.raises(ValueError, match=r"^AAA\(\): no sample has a finite value$"):
        IP.AAA([0.0, 1.0], [np.nan, np.inf])


def test_complex_points_on_a_real_rational():
    s = S.aaa("aaa/nonfinite")  # samples of 1 / (z - 0.499): the rational itself
    a = IP.AAA(s.z, s.f)
    v = a(0.25 + 0.1j)
    assert isinstance(v, complex)
    assert abs(v - 1 / (0.25 + 0.1j - 0.5 + 1e-3)) < 1e-9
    assert a(math.inf) == a(-math.inf)  # the limit sum w f / sum w
    assert math.isnan(a(math.nan))
