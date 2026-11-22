"""PCHIP's monotonicity (docs/design/interpolate.md §4, its three layers; milestone IP2) on monotone data: the samples'
monotone curves, random monotone data from fixed seeds and hypothesis' data, steps of zero included.

  1. every computed slope is 0 or has the sign of the data, and |s| <= 3 (1 + 4u) |D| + 2^-1074 for both
     neighbouring divided differences D (the floor: a subnormal slope's rounding, decision IP51)
  2. the cubic of the computed coefficients is monotone on every interval up to the coefficients' contract (its
     derivative, a quadratic, is at least minus the coefficient bounds' derivative on [0, h]: exact rationals);
     exactly monotone it is not in general, since a node slope of 0 is the derivative there only before rounding
  3. sorted points give values that go backwards, or past an interval's end values, only within the evaluation
     contract
And makima's flat runs (three equal values or more) are exactly flat; Akima's are flat except at a run's end node
in Akima's undefined case m1 = m2 != m3 = m4, where Akima (1970) takes the mean (scipy the same)."""

from fractions import Fraction

import numpy as np
from hypothesis import example, given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples
from . import refdata
from . import samples as S

IP = sd.interpolate
U = refdata.U


def fr(v):
    return Fraction(float(v))


def _monotone_cubic(c3, c2, c1, h, sign, s0, s1, d):
    """Whether sign * (3 c3 s^2 + 2 c2 s + c1) >= -(3 e3 s^2 + 2 e2 s) on [0, h], exactly: the derivative of the
    computed cubic keeps the sign of the data up to the coefficients' contract (e3, e2: refdata.hermite_bound,
    with its floor for subnormal coefficients, IP51). Exact monotonicity cannot hold: where a node slope is 0 the
    derivative there is 0 only before rounding (IP46)."""
    b3, b2 = refdata.hermite_bound(h, s0, s1, d)
    e3, e2 = fr(b3), fr(b2)
    a, b, c, h = 3 * fr(c3) * sign + 3 * e3, 2 * fr(c2) * sign + 2 * e2, fr(c1) * sign, fr(h)
    vals = [c, a * h * h + b * h + c]
    if a != 0:
        v = -b / (2 * a)
        if 0 < v < h:
            vals.append(a * v * v + b * v + c)
    return min(vals) >= 0


def _check(x, y):
    sign = 1.0 if y[-1] >= y[0] else -1.0
    p = IP.PchipInterpolator(x, y)
    c = np.asarray(p.c)
    D = np.diff(y) / np.diff(x)
    s = np.append(c[2], p(x[-1], nu=1))
    s[-1] = np.asarray(sd._core.interpolate.pchip_slopes(sd.asarray(x), sd.asarray(y.reshape(-1, 1).copy())))[-1, 0]
    # 1. the slopes
    for i in range(len(x)):
        for j in (i - 1, i):
            if 0 <= j < len(D):
                assert s[i] == 0 or np.sign(s[i]) == sign, (i, s[i])
                if D[j] != 0:  # the floor: a subnormal slope's rounding (IP51)
                    assert abs(s[i]) <= 3 * (1 + 4 * U) * abs(D[j]) + 2.0**-1074, (i, s[i], D[j])
    # 2. each piece exactly monotone
    for i in range(len(x) - 1):
        assert _monotone_cubic(c[0, i], c[1, i], c[2, i], x[i + 1] - x[i], sign, s[i], s[i + 1], D[i]), i
    # 3. sorted points
    q = np.sort(np.concatenate([np.linspace(x[0], x[-1], 501), x, np.nextafter(x, np.inf)]))
    q = q[(q >= x[0]) & (q <= x[-1])]
    v = np.asarray(p(q))
    idx = np.clip(np.searchsorted(x, q, side="right") - 1, 0, len(x) - 2)
    bounds = np.array([refdata.ppoly_bound(c[:, i], q[k] - x[i]) for k, i in enumerate(idx)])
    back = sign * (v[:-1] - v[1:])
    assert np.all(back <= bounds[:-1] + bounds[1:]), float(np.max(back - bounds[:-1] - bounds[1:]))
    lo, hi = np.minimum(y[idx], y[idx + 1]), np.maximum(y[idx], y[idx + 1])
    assert np.all(v >= lo - bounds) and np.all(v <= hi + bounds)


def test_monotone_samples():
    for cid in S.curve_ids("monotone"):
        c = S.curve(cid)
        _check(c.x, c.y)


def test_random_monotone_data():
    for seed in range(300):
        g = np.random.default_rng([20261123, seed])
        n = int(g.integers(3, 40))
        x = np.cumsum(g.exponential(1.0, n)) * 10.0 ** g.uniform(-3, 3)
        steps = g.exponential(1.0, n - 1) * 10.0 ** g.uniform(-3, 3, n - 1)
        steps[g.random(n - 1) < 0.2] = 0.0
        y = np.concatenate([[0.0], np.cumsum(steps)]) * (1 if seed % 2 else -1)
        if np.all(np.diff(x) > 0):
            _check(x, y)


def test_subnormal_monotone_data():
    """Steps near and below the smallest normal double: coefficients that cannot hold their relative precision,
    within the contract's floor (IP51)."""
    for seed in range(200):
        g = np.random.default_rng([20261124, seed])
        n = int(g.integers(3, 12))
        x = np.cumsum(g.uniform(0.1, 10.0, n))
        steps = g.exponential(1.0, n - 1) * 10.0 ** g.uniform(-322, -300, n - 1)
        steps[g.random(n - 1) < 0.2] = 0.0
        steps[g.random(n - 1) < 0.2] = g.exponential(1.0)  # an ordinary step among the tiny ones
        y = np.concatenate([[0.0], np.cumsum(steps)]) * (1 if seed % 2 else -1)
        _check(x, y)


@settings(max_examples=examples(150), deadline=None)
@given(st.lists(st.tuples(st.floats(1e-3, 1e3), st.floats(0, 1e3)), min_size=2, max_size=30), st.booleans())
@example([(6.0, 2.2250738585072014e-308), (1.0, 1.0)], False)  # subnormal coefficients (IP51)
@example([(1.0, 5e-324), (2.0, 1e-320), (0.5, 0.0), (3.0, 2e-310)], True)
def test_hypothesis_monotone_data(pairs, down):
    x = np.cumsum([1.0] + [p[0] for p in pairs])
    y = np.cumsum([0.0] + [p[1] for p in pairs]) * (-1.0 if down else 1.0)
    _check(x, y)


def test_akima_flat_runs():
    c = S.curve("flat/akima")
    runs = [i for i in range(len(c.x) - 2) if c.y[i] == c.y[i + 1] == c.y[i + 2]]
    assert runs
    for meth in ("akima", "makima"):
        p = IP.Akima1DInterpolator(c.x, c.y, method=meth)
        cc = np.asarray(p.c)
        s = np.append(cc[2], p(c.x[-1], nu=1))
        for i in runs:
            assert s[i + 1] == 0.0, (meth, i)  # a run's inner node: always 0
            if meth == "makima":
                for j in (i, i + 1):  # both intervals of the run
                    assert cc[0, j] == cc[1, j] == cc[2, j] == 0.0 and cc[3, j] == c.y[j], (meth, j)
                assert np.all(np.asarray(p(np.linspace(c.x[i], c.x[i + 2], 33))) == c.y[i]), (meth, i)
    # Akima's undefined case at the end of the first run: D = 0, 0 before node 2 and 13, 13 after it
    p = IP.Akima1DInterpolator(c.x, c.y)
    assert np.asarray(p.c)[2, 2] == 0.5 * (13.0 + 0.0)
