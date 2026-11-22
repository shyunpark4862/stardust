"""BarycentricInterpolator, KroghInterpolator, their function forms and FloaterHormannInterpolator
(docs/api/interpolate.md §5; docs/design/interpolate.md §3.6, §4; milestone IP4).

Contracts against tests/interpolate/ref/polynomial.npz (the exact interpolant of the float data, its first two
derivatives, the Lebesgue function, sum |l_j y_j| and the Floater-Hormann d = 3 rational) and against exact rationals
of the computed coefficients (refdata: barycentric_bound, rational_nodal_bound, krogh_table_bound, krogh_bound,
rational_bound); the behaviour that is not rounding (shapes, axes, nodes, orders beyond the degree, the changing
methods) against scipy 1.18.1."""

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
POLY = refdata.ref("polynomial")


def _ref(method, sid):
    """(hi, lo) of a set's points in the set's order."""
    m = POLY.methods[method]
    pos = np.flatnonzero(POLY.inputs["set"][m.index] == sid)
    order = np.argsort(POLY.inputs["index"][m.index[pos]], kind="stable")
    return m.hi[pos[order]], m.lo[pos[order]]


def _err(got, hi, lo):
    return abs(Fraction(float(got)) - Fraction(float(hi)) - Fraction(float(lo)))


def _distinct():
    return [pid for pid in S.POLY_IDS if S.poly(pid).distinct]


# ---------------------------------------------------------------- BarycentricInterpolator: contracts


def test_barycentric_contract_end_to_end():
    worst = 0.0
    for pid in _distinct():
        P = S.poly(pid)
        n = P.xi.size
        got = np.asarray(IP.BarycentricInterpolator(P.xi, P.yi)(P.x))
        hi, lo = _ref("value", pid)
        lam, _ = _ref("lebesgue", pid)
        at, _ = _ref("abs_terms", pid)
        for k, v in enumerate(P.x):
            if v in P.xi:  # a node: the data itself
                assert got[k] == P.yi[list(P.xi).index(v)]
                continue
            err = _err(got[k], hi[k], lo[k])
            u = float(err) / refdata.barycentric_bound(n, at[k], lam[k], hi[k])
            assert u <= 1, (pid, k, u)
            worst = max(worst, u)
    assert worst > 0.01  # the contract is not vacuous


def test_barycentric_derivatives_contract_end_to_end():
    worst = 0.0
    for pid in _distinct():
        P = S.poly(pid)
        n = P.xi.size
        B = IP.BarycentricInterpolator(P.xi, P.yi)
        X = [Fraction(float(v)) for v in P.xi]
        D = refdata.differentiation_matrix(X)
        v_hat = [Fraction(float(v)) for v in P.yi]
        bound = [0.0] * n
        nodal = sd.asarray(P.yi.reshape(-1, 1).copy())
        for order, method in ((1, "d1"), (2, "d2")):
            bound = refdata.rational_nodal_bound(D, v_hat, bound)
            nodal = sd._core.interpolate.bary_node_derivative(B.xi, B.wi, nodal)
            v_hat = [Fraction(float(v)) for v in np.asarray(nodal)[:, 0]]
            got = np.asarray(B.derivative(P.x, der=order))
            hi, lo = _ref(method, pid)
            for k, v in enumerate(P.x):
                ell = refdata.lagrange_basis(X, Fraction(float(v)))
                q = sum((e * t for e, t in zip(ell, v_hat)), Fraction(0))
                lam = float(sum((abs(e) for e in ell), Fraction(0)))
                at = float(sum((abs(e * t) for e, t in zip(ell, v_hat)), Fraction(0)))
                b = refdata.barycentric_bound(n, at, lam, float(q)) + sum(abs(float(e)) * t for e, t in zip(ell, bound))
                err = _err(got[k], hi[k], lo[k])
                u = 0.0 if err == 0 else float(err) / (1.01 * b)
                assert u <= 1, (pid, order, k, u)
                worst = max(worst, u)
    assert worst > 1e-4


# ---------------------------------------------------------------- KroghInterpolator: contracts


def test_krogh_table_contract():
    worst = 0.0
    for pid in S.POLY_IDS:
        P = S.poly(pid)
        K = IP.KroghInterpolator(P.xi, P.yi)
        c_hat = np.asarray(K._c)[:, 0]
        X, Y = [Fraction(float(v)) for v in P.xi], [Fraction(float(v)) for v in P.yi]
        exact = refdata.krogh_table_exact(X, Y)
        bound = refdata.krogh_table_bound(X, Y)
        for k in range(len(X)):
            err = abs(Fraction(float(c_hat[k])) - exact[k])
            u = 0.0 if err == 0 else float(err) / bound[k]
            assert u <= 1, (pid, k, u)
            worst = max(worst, u)
    assert worst > 1e-3


def test_krogh_evaluation_contract_and_end_to_end():
    worst, worst_e2e = 0.0, 0.0
    for pid in S.POLY_IDS:
        P = S.poly(pid)
        K = IP.KroghInterpolator(P.xi, P.yi)
        c_hat = np.asarray(K._c)[:, 0]
        X, Y = [Fraction(float(v)) for v in P.xi], [Fraction(float(v)) for v in P.yi]
        C = [Fraction(float(v)) for v in c_hat]
        tb = refdata.krogh_table_bound(X, Y)
        for m, method in ((0, "value"), (1, "d1"), (2, "d2")):
            got = np.asarray(K(P.x) if m == 0 else K.derivative(P.x, der=m))
            assert np.array_equal(got, np.asarray(K.derivatives(P.x, der=3))[m])  # one pass sequence
            hi, lo = _ref(method, pid)
            for k, v in enumerate(P.x):
                V = Fraction(float(v))
                exact = refdata.newton_passes(X, C, V, m) * math.factorial(m)
                err = abs(Fraction(float(got[k])) - exact)
                u = 0.0 if err == 0 else float(err) / refdata.krogh_bound(X, c_hat, V, m)
                assert u <= 1, (pid, m, k, u)
                worst = max(worst, u)
                err = _err(got[k], hi[k], lo[k])
                u = 0.0 if err == 0 else float(err) / refdata.krogh_bound(X, c_hat, V, m, tb)
                assert u <= 1, (pid, m, k, u)
                worst_e2e = max(worst_e2e, u)
    assert worst > 1e-3 and worst_e2e > 1e-3


def test_krogh_high_degree_has_no_warning():
    x = np.linspace(-1, 1, 41)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        IP.KroghInterpolator(x, 1 / (1 + 25 * x**2))(0.3)
        IP.krogh_interpolate(x, np.sin(x), [0.1, 0.2], der=2)


# ---------------------------------------------------------------- FloaterHormannInterpolator: contracts


def _fh_exact_weights(X, d):
    n = len(X)
    w = []
    for k in range(n):
        acc = Fraction(0)
        for i in range(max(k - d, 0), min(k + 1, n - d)):
            prod = Fraction(1)
            for j in range(i, i + d + 1):
                if j != k:
                    prod *= abs(X[k] - X[j])
            acc += 1 / prod
        w.append(acc if (k - d) % 2 == 0 else -acc)
    return w


def _rational_parts(Z, W, F, v):
    """(r, sum|lam f|, sum|lam|, sum lam) exactly at a point v that is no node."""
    lam = [w / (v - z) for z, w in zip(Z, W)]
    num = sum((t * f for t, f in zip(lam, F)), Fraction(0))
    den = sum(lam, Fraction(0))
    return (
        num / den,
        sum((abs(t * f) for t, f in zip(lam, F)), Fraction(0)),
        sum((abs(t) for t in lam), Fraction(0)),
        den,
    )


def test_floater_hormann_contracts():
    d = 3
    worst = [0.0, 0.0, 0.0]
    for pid in _distinct():
        P = S.poly(pid)
        F = IP.FloaterHormannInterpolator(P.xi, P.yi, d=d)
        X = [Fraction(float(v)) for v in np.sort(P.xi)]
        Yd = dict(zip(P.xi, P.yi))
        Y = [Fraction(float(Yd[float(v)])) for v in X]
        w_hat = np.asarray(F.weights)
        exact = _fh_exact_weights(X, d)
        for k, (a, b) in enumerate(zip(w_hat, exact)):
            u = float(abs(Fraction(float(a)) - b) / abs(b)) / refdata.gamma(3 * d)
            assert u <= 1, (pid, k, u)
            worst[0] = max(worst[0], u)
        W = [Fraction(float(v)) for v in w_hat]
        got = np.asarray(F(P.x))
        hi, lo = _ref("fh3", pid)
        n = len(X)
        for k, v in enumerate(P.x):
            V = Fraction(float(v))
            if V in X:
                assert got[k] == Yd[float(v)]
                continue
            r, sf, s_, den = _rational_parts(X, W, Y, V)
            err = abs(Fraction(float(got[k])) - r)
            u = float(err) / refdata.rational_bound(n, float(sf), float(s_), float(den), float(r), n + 3, n + 2, n + 1)
            assert u <= 1, (pid, k, u)
            worst[1] = max(worst[1], u)
            r, sf, s_, den = _rational_parts(X, exact, Y, V)
            err = _err(got[k], hi[k], lo[k])
            e = 3 * d
            u = float(err) / refdata.rational_bound(n, float(sf), float(s_), float(den), float(r), n + 3 + e, n + 2 + e,
                                                    n + 1 + e)  # fmt: skip
            assert u <= 1, (pid, k, u)
            worst[2] = max(worst[2], u)
    assert min(worst) > 1e-3


# ---------------------------------------------------------------- behaviour: scipy's, with the area's rules


@LIVE
def test_barycentric_follows_scipy():
    import scipy.interpolate as si

    g = np.random.default_rng(40)
    x = np.sort(g.random(9))
    y = g.standard_normal((2, 9, 3))
    q = g.random((4, 5)) * 1.4 - 0.2
    for axis in (1, -2):
        b, s = IP.BarycentricInterpolator(x, y, axis=axis), si.BarycentricInterpolator(x, y, axis=axis)
        for got, want in ((b(q), s(q)), (b.derivative(q, der=2), s.derivative(q, der=2)),
                          (b.derivatives(q, der=3), s.derivatives(q, der=3)), (b(0.3), s(0.3))):  # fmt: skip
            got = np.asarray(got)
            assert got.shape == want.shape
            assert np.allclose(got, want, rtol=1e-8, atol=1e-10 * np.max(np.abs(want)))
    b, s = IP.BarycentricInterpolator(x, y[0, :, 0]), si.BarycentricInterpolator(x, y[0, :, 0])
    assert isinstance(b(0.5), float) and np.isclose(b(0.5), s(0.5))
    assert np.array_equal(np.asarray(b(x)), y[0, :, 0])  # the nodes give the data
    assert np.asarray(b.derivatives(0.3)).shape == s.derivatives(0.3).shape == (9,)
    assert np.all(np.asarray(b.derivative(q, der=9)) == 0) and np.all(s.derivative(q, der=9) == 0)
    assert b.derivative(math.nan, der=12) != b.derivative(math.nan, der=12)  # NaN, as below the degree (IP64)
    bc = IP.BarycentricInterpolator(x, y[0, :, 0] + 1j)
    assert np.allclose(np.asarray(bc(q)), si.BarycentricInterpolator(x, y[0, :, 0] + 1j)(q))
    assert np.allclose(np.asarray(b.wi) / s.wi, (np.asarray(b.wi) / s.wi)[0])  # a common factor of scipy's


def test_barycentric_weights_are_scaled_and_do_not_overflow():
    x = np.cos(np.pi * np.arange(3001) / 3000)  # the plain product of 3000 differences under- and overflows
    b = IP.BarycentricInterpolator(x, np.exp(x))
    w = np.abs(np.asarray(b.wi))
    assert np.all(np.isfinite(w)) and np.all(w > 0) and 0.5 <= w.max() < 1
    assert abs(b(0.123) - math.exp(0.123)) < 1e-13
    tiny = np.array([0.0, 1e-310, 1.0])  # a point a subnormal distance from a node: that node's value
    bt = IP.BarycentricInterpolator(tiny, [2.0, 3.0, 4.0])
    assert bt(5e-324) == 2.0 and bt(1e-310) == 3.0


def test_barycentric_weights_follow_the_input_order_bit_for_bit():
    g = np.random.default_rng(41)
    x = g.standard_normal(40)
    w1 = np.asarray(IP.BarycentricInterpolator(x).wi)
    w2 = np.asarray(IP.BarycentricInterpolator(x).wi)
    assert np.array_equal(w1, w2)


def test_add_xi_and_set_yi():
    g = np.random.default_rng(42)
    x, y = g.random(12), g.standard_normal((12, 2))
    b = IP.BarycentricInterpolator(x[:7], y[:7])
    b.add_xi(x[7:], y[7:])
    fresh = IP.BarycentricInterpolator(x, y)
    q = g.random(20)
    assert np.array_equal(np.asarray(b.wi), np.asarray(fresh.wi))  # IP62: the weights of all the nodes
    assert np.array_equal(np.asarray(b(q)), np.asarray(fresh(q)))
    b.set_yi(2 * y)
    assert np.array_equal(np.asarray(b(q)), np.asarray(IP.BarycentricInterpolator(x, 2 * y)(q)))
    e = IP.BarycentricInterpolator(x)
    assert e.yi is None
    e.set_yi(y[:, 0])
    assert np.array_equal(np.asarray(e.yi), y[:, 0])
    ya = IP.BarycentricInterpolator(x[:5], y[:5].T, axis=1)
    ya.add_xi(x[5:7], y[5:7].T)
    assert np.asarray(ya.yi).shape == (2, 7)


@LIVE
def test_krogh_follows_scipy():
    import scipy.interpolate as si

    xi = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 2.0])
    yi = np.stack([[1.0, 2.0, 0.0, -1.0, 3.0, 1.0], [0.0, 1.0, 2.0, 0.0, 0.0, -1.0]], axis=1)
    q = np.linspace(-0.5, 2.5, 13).reshape(13, 1)
    k, s = IP.KroghInterpolator(xi, yi), si.KroghInterpolator(xi, yi)
    pairs = [(k(q), s(q)), (k.derivative(q, der=3), s.derivative(q, der=3)), (k.derivatives(q), s.derivatives(q)),
             (k.derivatives(0.5, der=9), s.derivatives(0.5, der=9))]  # fmt: skip
    for got, want in pairs:
        got = np.asarray(got)
        assert got.shape == want.shape
        assert np.allclose(got, want, rtol=1e-12, atol=1e-12)
    # Hermite data: the values and the derivatives at the repeated nodes
    one = IP.KroghInterpolator(xi, yi[:, 0])
    assert one(0.0) == 1.0 and one.derivative(0.0) == 2.0 and abs(one.derivative(1.0, der=2) - 3.0) < 1e-13
    assert np.asarray(IP.krogh_interpolate(xi, yi[:, 0], [0.5, 1.5], der=[0, 2])).shape == (2, 2)
    assert np.allclose(IP.krogh_interpolate(xi, yi[:, 0], 0.5, der=1), si.krogh_interpolate(xi, yi[:, 0], 0.5, der=1))
    assert np.allclose(IP.barycentric_interpolate([0, 1, 3], [1, 2, 0], [0.5, 2.0], der=2),
                       si.barycentric_interpolate([0, 1, 3], [1, 2, 0], [0.5, 2.0], der=2))  # fmt: skip
    kc = IP.KroghInterpolator(xi, yi[:, 0] * (1 + 2j))
    assert np.allclose(np.asarray(kc(q)), si.KroghInterpolator(xi, yi[:, 0] * (1 + 2j))(q))


@LIVE
def test_floater_hormann_follows_scipy():
    import scipy.interpolate as si

    g = np.random.default_rng(43)
    x = g.permutation(np.linspace(-5, 5, 15))  # any order: sorted as scipy
    y = 1 / (1 + x**2)
    f, s = IP.FloaterHormannInterpolator(x, y), si.FloaterHormannInterpolator(x, y)
    assert np.array_equal(np.asarray(f.weights), s.weights)  # scipy's closed form, the same operations
    z = np.concatenate([np.linspace(-6, 6, 25), x, [np.inf, -np.inf]])
    assert np.allclose(np.asarray(f(z)), s(z), rtol=1e-13, atol=1e-15)
    zc = z[:25] + 0.5j
    assert np.allclose(np.asarray(f(zc)), s(zc), rtol=1e-13, atol=1e-15)
    for name in ("poles", "roots", "residues"):
        got, want = np.asarray(getattr(f, name)()), getattr(s, name)()
        order = np.argsort(np.asarray(f.poles()).round(8)) if name == "residues" else None
        if order is None:
            assert np.allclose(np.sort_complex(got), np.sort_complex(want), rtol=1e-7, atol=1e-9), name
        else:
            assert np.allclose(got[order], want[np.argsort(s.poles().round(8))], rtol=1e-6), name
    yb = np.stack([y, 2 * y], axis=1)
    fb = IP.FloaterHormannInterpolator(x, yb, d=2)
    assert np.allclose(np.asarray(fb(z[:25])), si.FloaterHormannInterpolator(x, yb, d=2)(z[:25]))
    yn = y.copy()
    yn[3] = np.nan  # a sample that is not finite is dropped (scipy)
    assert np.allclose(np.asarray(IP.FloaterHormannInterpolator(x, yn)(z[:25])),
                       si.FloaterHormannInterpolator(x, yn)(z[:25]))  # fmt: skip


def test_floater_hormann_batch_axis_and_one_column_methods():
    g = np.random.default_rng(44)
    x = np.sort(g.random(10))
    y = g.standard_normal((3, 10, 2))
    f = IP.FloaterHormannInterpolator(x, y, axis=1)
    q = g.random((4, 5))
    assert np.asarray(f(q)).shape == (3, 4, 5, 2)  # x's shape at the axis (decision IP65)
    with pytest.raises(ValueError, match=r"^residues\(\): the values must be 1-D"):
        f.residues()
    assert f.poles().dtype == sd.complex128


@pytest.mark.parametrize("pid", S.POLY_IDS)
def test_function_forms_are_the_objects(pid):
    P = S.poly(pid)
    k = IP.KroghInterpolator(P.xi, P.yi)
    assert np.array_equal(np.asarray(IP.krogh_interpolate(P.xi, P.yi, P.x)), np.asarray(k(P.x)))
    assert np.array_equal(
        np.asarray(IP.krogh_interpolate(P.xi, P.yi, P.x, der=2)), np.asarray(k.derivative(P.x, der=2))
    )
    if P.distinct:
        b = IP.BarycentricInterpolator(P.xi, P.yi)
        got = IP.barycentric_interpolate(P.xi, P.yi, P.x, der=[0, 1])
        assert np.array_equal(np.asarray(got[0]), np.asarray(b(P.x)))
        assert np.array_equal(np.asarray(got[1]), np.asarray(b.derivative(P.x)))
