"""fixed_quad, its Gauss-Legendre nodes and weights, and newton_cotes (docs/api/integrate.md §7;
docs/design/integrate.md §3.7, decisions C42, C43, C52-C54): every node and weight correctly
rounded (the table of n <= 100 against Arb live, n > 100 against tests/integrate/ref/gauss.json,
also through the exact path), scipy 1.18.1's call points and values
(tests/integrate/data/scipy_samples.json), the sum's contract, and newton_cotes against scipy's
table and an exact solve of the moment equations."""

import hashlib
import math
from fractions import Fraction

import numpy as np
import pytest
import scipy.integrate as si

import stardust as sd
from stardust.integrate import _gauss

from . import problems as P
from . import refdata

I = sd.integrate  # noqa: E741
EPS = 2.0**-52


def _nearest(ball):
    """The double nearest an Arb ball's value, which must not hold a rounding boundary."""
    from flint import arb, ctx

    old = ctx.prec
    ctx.prec = 2600
    try:
        hi = float(ball.mid())
        best = min(
            (hi, math.nextafter(hi, math.inf), math.nextafter(hi, -math.inf)), key=lambda c: abs(ball.mid() - arb(c))
        )
        for c in (math.nextafter(best, math.inf), math.nextafter(best, -math.inf)):
            assert not ball.contains((arb(best) + arb(c)) / 2)
    finally:
        ctx.prec = old
    return best


def test_table_is_arb_rounded():
    """n = 1 ... 100: every node and weight is the double nearest Arb's value (arb.legendre_p_root
    at 256 bits, radius far below the rounding)."""
    from flint import arb, ctx

    for n in range(1, _gauss.TABLE_MAX + 1):
        x, w = (v.tolist() for v in _gauss.legendre(n))
        old = ctx.prec
        ctx.prec = 256
        try:
            for k in range(n):
                X, W = arb.legendre_p_root(n, n - 1 - k, weight=True)  # Arb's zeros descend
                want_x = 0.0 if X.is_exact() and X == 0 else _nearest(X)
                assert x[k] == want_x, (n, k)
                assert w[k] == _nearest(W), (n, k)
        finally:
            ctx.prec = old


def _ref_half(n):
    e = refdata.gauss()[str(n)]
    return [float.fromhex(v) for v in e["x"]], [float.fromhex(v) for v in e["w"]]


def test_past_the_table():
    ref = refdata.gauss()
    assert set(ref) == {"101", "128", "150", "200", "256", "500", "1000"}
    for key in ref:
        n = int(key)
        x, w = (v.tolist() for v in _gauss.legendre(n))
        hx, hw = _ref_half(n)
        assert x[n // 2 :] == hx and w[n // 2 :] == hw, n
        assert x[: n // 2] == [-v for v in reversed(hx[n % 2 :])] and w[: n // 2] == hw[n % 2 :][::-1], n


def test_exact_path():
    """Every value sent through the exact arithmetic (the escalation of an ambiguous rounding)
    gives the same doubles."""
    saved = _gauss._SCALE
    try:
        for n in (101, 128):
            _gauss._cache.pop(n, None)
            _gauss._SCALE = 2.0**200
            x, w = (v.tolist() for v in _gauss.legendre(n))
            hx, hw = _ref_half(n)
            assert x[n // 2 :] == hx and w[n // 2 :] == hw, n
    finally:
        _gauss._SCALE = saved
        _gauss._cache.clear()


def test_nodes_ascending_and_symmetric():
    for n in (1, 2, 5, 99, 100, 101, 333):
        x, w = (v.tolist() for v in _gauss.legendre(n))
        assert len(x) == len(w) == n
        assert all(a < b for a, b in zip(x, x[1:]))
        assert x == [-v for v in reversed(x)] and w == list(reversed(w))
        assert abs(math.fsum(w) - 2) < 64 * EPS * n


# ---- fixed_quad


class Recorder:
    def __init__(self, f):
        self.f, self.h, self.shapes, self.count = f, hashlib.sha256(), [], 0

    def __call__(self, y):
        a = np.asarray(y.tolist(), dtype=np.float64)
        self.shapes.append(list(a.shape))
        self.count += a.size
        self.h.update(a.tobytes())
        return sd.asarray(self.f(a))


def _fixed_runs():
    for p in P.INTEGRALS:
        if p.numeric and p.weight is None and math.isfinite(p.a) and math.isfinite(p.b):
            for n in (5, 20, 100):
                yield f"{p.id}/n={n}", p, n


def test_scipy_points_and_values():
    rec = refdata.data("scipy_samples.json")["fixed_quad"]
    runs = list(_fixed_runs())
    assert {k for k, _, _ in runs} == set(rec)
    same_points = {5: 0, 20: 0, 100: 0}
    same_value = {5: 0, 20: 0, 100: 0}
    total = {5: 0, 20: 0, 100: 0}
    for key, p, n in runs:
        r = rec[key]
        f = Recorder(p.f_np)
        with np.errstate(all="ignore"):
            v = I.fixed_quad(f, p.a, p.b, n=n)
        assert isinstance(v, float)
        assert r["second"] is None and r["calls"]["calls"] == 1 and f.shapes == [[n]]
        total[n] += 1
        same_points[n] += f.h.hexdigest() == r["calls"]["sha256"]
        sv = r["integral"]
        same_value[n] += v == sv or (math.isnan(v) and math.isnan(sv))
    assert total == {5: 30, 20: 30, 100: 30}
    # n = 5: the correctly rounded nodes are scipy's; past it scipy's roots_legendre differs in
    # the last bits (docs/design/integrate.md §7.9)
    assert same_points == {5: 30, 20: 0, 100: 0}
    assert same_value == {5: 4, 20: 4, 100: 0}


def test_sum_contract():
    """The value is (b - a)/2 sum w_i f_i of the values func returned, within the pairwise sum's
    bound (ceil(log2 n) + 4) eps sum |w_i f_i| of the exact rational sum."""
    worst = 0.0
    for p in P.INTEGRALS[:40]:
        if not (p.numeric and p.weight is None and math.isfinite(p.a) and math.isfinite(p.b)):
            continue
        for n in (5, 20, 100, 150):
            seen = []

            def f(y, p=p, seen=seen):
                v = np.asarray(p.f_np(np.asarray(y.tolist())), dtype=float)
                seen.append(v)
                return sd.asarray(v)

            with np.errstate(all="ignore"):
                got = I.fixed_quad(f, p.a, p.b, n=n)
            fv = seen[0]
            if not np.all(np.isfinite(fv)):
                continue
            w = _gauss.legendre(n)[1].tolist()
            s = sum((Fraction(wi) * Fraction(float(fi)) for wi, fi in zip(w, fv)), Fraction(0))
            half = Fraction((p.b - p.a) / 2.0)
            exact = half * s
            size = abs(half) * sum(abs(Fraction(wi) * Fraction(float(fi))) for wi, fi in zip(w, fv))
            bound = (math.ceil(math.log2(n)) + 4) * EPS * size
            err = abs(Fraction(got) - exact)
            if size:
                worst = max(worst, float(err / bound))
                assert err <= bound, (p.id, n)
    assert worst < 0.2  # 0.118 (docs/design/integrate.md §7.9)


def test_vector_valued_and_forms():
    v = I.fixed_quad(lambda x: sd.stack([sd.exp(x), x * x]), 0.0, 1.0, n=8)
    assert isinstance(v, sd.ndarray) and v.shape == (2,)
    assert v.tolist() == pytest.approx([math.e - 1, 1 / 3], rel=1e-14)
    assert I.fixed_quad(lambda x: [float(t) ** 3 for t in x.tolist()], 0, 2, n=2) == pytest.approx(4.0, rel=4 * EPS)
    assert I.fixed_quad(lambda x: np.asarray(x.tolist()) ** 2, 1.0, -1.0) == pytest.approx(-2 / 3, rel=4 * EPS)
    assert I.fixed_quad(sd.exp, 3.0, 3.0) == 0.0
    big = I.fixed_quad(sd.cos, 0.0, math.pi / 2, n=1000)
    assert big == pytest.approx(1.0, rel=8 * EPS)


def test_fixed_quad_errors():
    def err(kind, text, f):
        with pytest.raises(kind) as e:
            f()
        assert str(e.value) == text

    err(ValueError, "fixed_quad(): 'b' must be finite (Gaussian quadrature needs a finite interval)",
        lambda: I.fixed_quad(sd.exp, 0.0, math.inf))  # fmt: skip
    err(ValueError, "fixed_quad(): 'a' must be finite (Gaussian quadrature needs a finite interval)",
        lambda: I.fixed_quad(sd.exp, math.nan, 1.0))  # fmt: skip
    err(ValueError, "fixed_quad(): 'n' must be >= 1", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=0))
    err(TypeError, "fixed_quad(): 'n' must be an int, not float", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=5.0))
    err(TypeError, "fixed_quad(): 'func' must be callable, not float", lambda: I.fixed_quad(1.0, 0.0, 1.0))
    err(ValueError, "fixed_quad(): 'func' must return shape (5,) or (..., 5) for its argument of shape (5,), "
        "not a scalar",
        lambda: I.fixed_quad(lambda x: 1.0, 0.0, 1.0))  # fmt: skip
    err(ValueError, "fixed_quad(): 'func' must return shape (5,) or (..., 5) for its argument of shape (5,), "
        "not (4,)",
        lambda: I.fixed_quad(lambda x: x[1:], 0.0, 1.0))  # fmt: skip
    err(TypeError, "fixed_quad(): 'func' must return real values, not a bool array",
        lambda: I.fixed_quad(lambda x: x > 0, 0.0, 1.0))  # fmt: skip
    err(TypeError, "fixed_quad(): 'func' must return real values, not a complex128 array",
        lambda: I.fixed_quad(lambda x: x * 1j, 0.0, 1.0))  # fmt: skip
    err(TypeError, "fixed_quad(): unexpected keyword argument 'args'", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, args=()))
    # C109: the nodes past the table cost n^2 the first time; 10^4 is the most
    err(ValueError, "fixed_quad(): 'n' must be at most 10000", lambda: I.fixed_quad(sd.exp, 0.0, 1.0, n=10001))


# ---- newton_cotes


def _exact_solve(pos, n):
    """The weights from the moment equations sum a_i r_i^k = N^(k+1)/(k+1), k = 0 ... N, by
    Gaussian elimination in rationals (another algorithm than newton_cotes's)."""
    m = len(pos)
    A = [[r**k for r in pos] + [Fraction(n) ** (k + 1) / (k + 1)] for k in range(m)]
    for c in range(m):
        piv = next(r for r in range(c, m) if A[r][c] != 0)
        A[c], A[piv] = A[piv], A[c]
        for r in range(m):
            if r != c and A[r][c] != 0:
                f = A[r][c] / A[c][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[i][m] / A[i][i] for i in range(m)]


def test_newton_cotes_scipy_table():
    rec = refdata.data("scipy_samples.json")["newton_cotes"]
    for key, r in rec.items():
        an, B = I.newton_cotes(int(key))
        assert an.tolist() == r["an"] and B == r["B"], key  # scipy's table, exact rationals rounded
        an2, B2 = si.newton_cotes(int(key), 1)
        assert an.tolist() == an2.tolist() and B == B2


def test_newton_cotes_exact():
    for pos in ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [0, 0.5, 2], [0, 1, 1.5, 2.25, 4],
                [0, 2, 1, 3]):  # fmt: skip
        n = len(pos) - 1
        an, B = I.newton_cotes(pos)
        exact = _exact_solve([Fraction(v) for v in pos], n)
        assert an.tolist() == [float(v) for v in exact], pos
        equally = all(b - a == 1 for a, b in zip(pos, pos[1:]))
        p = n + 2 if n % 2 == 0 and equally else n + 1
        b = (Fraction(n) ** (p + 1) / (p + 1) - sum(a * Fraction(r) ** p for a, r in zip(exact, pos))) / math.factorial(
            p
        )
        assert B == float(b)
    an, B = I.newton_cotes(16)
    assert an.tolist() == I.newton_cotes(list(range(17)))[0].tolist()
    assert (
        I.newton_cotes([5.0, 7.0, 1.0], equal=True)[0].tolist() == I.newton_cotes(2)[0].tolist()
    )  # scipy: the length only


def test_newton_cotes_errors():
    def err(kind, text, f):
        with pytest.raises(kind) as e:
            f()
        assert str(e.value) == text

    err(ValueError, "newton_cotes(): 'rn' must be >= 1 (the number of intervals)", lambda: I.newton_cotes(0))
    err(
        TypeError,
        "newton_cotes(): 'rn' must be an int or a sequence of numbers, not float",
        lambda: I.newton_cotes(4.0),
    )
    err(
        TypeError,
        "newton_cotes(): 'rn' must be an int or a sequence of numbers, not bool",
        lambda: I.newton_cotes(True),
    )
    err(ValueError, "newton_cotes(): the sample positions must start at 0 and end at N = 2",
        lambda: I.newton_cotes([0, 1, 3]))  # fmt: skip
    err(ValueError, "newton_cotes(): the sample positions must be distinct", lambda: I.newton_cotes([0, 1, 1, 3]))
    err(ValueError, "newton_cotes(): 'rn' must hold at least 2 positions", lambda: I.newton_cotes([0]))
    err(TypeError, "newton_cotes(): 'equal' must be True or False, not int", lambda: I.newton_cotes(3, equal=1))
    # C110: the exact rationals cost about N^3
    err(ValueError, "newton_cotes(): 'rn' must be at most 256 (the number of intervals)", lambda: I.newton_cotes(257))
    err(ValueError, "newton_cotes(): 'rn' must hold at most 257 positions", lambda: I.newton_cotes(list(range(258))))


def test_newton_cotes_remembered_rules_are_copies():
    # C108: an int N takes the remembered rule; each call gives its own writable array
    an, B = I.newton_cotes(6)
    an[0] = 99.0
    an2, B2 = I.newton_cotes(6)
    assert an2[0] != 99.0 and B2 == B
    assert an2.tolist() == I.newton_cotes([0, 1, 2, 3, 4, 5, 6])[0].tolist()
