"""The linear kernel's exact properties (docs/design/interpolate.md §3.2, §4; base/lerp.hpp): the ends are exact to
the sign of zero, a == b gives a, the value is monotone in x and stays between a and b inside the interval, and the
non-finite ends follow numpy's rule. Seen through sd.interp on two nodes (inside) and through the interpolate area's
linear path (_core.interpolate.linear, which also extends the end pieces), on the intervals of
tests/interpolate/samples.py, the doubles next to their ends, and hypothesis's intervals."""

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

import stardust as sd

from .._tier import examples
from . import samples as S

INF, NAN = math.inf, math.nan
LINEAR = sd._core.interpolate.linear
EXTEND, NAN_OUTSIDE, PERIODIC = 1, 0, 2


def on_interval(x0, x1, a, b, x):
    """sd.interp of the points x (inside [x0, x1]) on the interval."""
    return np.asarray(sd.interp(sd.asarray(np.asarray(x, dtype=float)), sd.asarray([x0, x1]), sd.asarray([a, b])))


def extended(x0, x1, a, b, x):
    """The linear path with the end pieces extended (any x)."""
    out = LINEAR(sd.asarray(np.asarray(x, dtype=float)), sd.asarray([x0, x1]), sd.asarray([[a], [b]]), EXTEND)
    return np.asarray(out)[:, 0]


def neighbours(v, k):
    out, lo, hi = [v], v, v
    for _ in range(k):
        lo, hi = math.nextafter(lo, -INF), math.nextafter(hi, INF)
        out += [lo, hi]
    return out


def check_interval(x0, x1, a, b, x):
    """The properties of §3.2 on one finite interval and its points (sorted, inside)."""
    x = np.unique(np.asarray(x, dtype=float))
    x = x[(x >= x0) & (x <= x1)]
    v = on_interval(x0, x1, a, b, x)
    assert np.array_equal(v, extended(x0, x1, a, b, x))  # the two paths are one kernel
    ends = on_interval(x0, x1, a, b, [x0, x1])
    assert math.copysign(1, ends[0]) == math.copysign(1, a) and ends[0] == a
    assert math.copysign(1, ends[1]) == math.copysign(1, b) and ends[1] == b
    if a == b:
        assert np.all(v == a)
        if a == 0:  # -0 inside only when both ends are -0
            both = math.copysign(1, a) < 0 and math.copysign(1, b) < 0
            assert np.all(np.signbit(v[(x > x0) & (x < x1)]) == both)
    assert np.all(v >= min(a, b)) and np.all(v <= max(a, b))
    up, down = v[1:] >= v[:-1], v[1:] <= v[:-1]  # (np.diff could overflow)
    assert np.all(up) if b >= a else np.all(down)


def test_sample_intervals():
    for L in S.lerps():
        if not S.lerp_finite(L):
            continue
        pts = list(L.x) + neighbours(L.x0, 40) + neighbours(L.x1, 40)
        check_interval(L.x0, L.x1, L.a, L.b, pts)


def test_signed_zeros():
    # inside, -0 only when both ends are -0 (numpy gives +0 there, docs/design/interpolate.md §3.2)
    for a, b in ((-0.0, -0.0), (0.0, -0.0), (-0.0, 0.0), (0.0, 0.0)):
        v = on_interval(0.0, 1.0, a, b, [0.0, 0.5, 1.0])
        inside = -1.0 if math.copysign(1, a) < 0 and math.copysign(1, b) < 0 else 1.0
        assert [math.copysign(1, t) for t in v] == [math.copysign(1, a), inside, math.copysign(1, b)], (a, b)


def test_non_finite_ends():
    # numpy's rule (tests/interpolate/data/numpy.json nonfinite/*): the end at t = 0, 1; inside the infinite end,
    # NaN for opposite infinities or a NaN end; outside, a + t (b - a) in IEEE arithmetic
    x = [0.0, 0.25, 0.5, 1.0]
    want = {
        (INF, 1.0): [INF, INF, INF, 1.0],
        (1.0, INF): [1.0, INF, INF, INF],
        (INF, INF): [INF] * 4,
        (-INF, INF): [-INF, NAN, NAN, INF],
        (NAN, 1.0): [NAN, NAN, NAN, 1.0],
        (1.0, NAN): [1.0, NAN, NAN, NAN],
        (-INF, 1.0): [-INF, -INF, -INF, 1.0],
    }
    for (a, b), w in want.items():
        got = on_interval(0.0, 1.0, a, b, x)
        assert np.array_equal(got, w, equal_nan=True), (a, b, got)
        assert np.array_equal(extended(0.0, 1.0, a, b, x), got, equal_nan=True)
    with np.errstate(all="ignore"):
        for a, b in want:
            for t in (-1.5, 2.0):
                assert np.array_equal(extended(0.0, 1.0, a, b, [t]), [a + t * (b - a)], equal_nan=True), (a, b, t)


def test_extension_is_monotone_and_meets_the_ends():
    g = np.random.default_rng(3)
    for _ in range(300):
        x0 = g.standard_normal() * 10.0 ** g.integers(-3, 4)
        x1 = x0 + g.exponential() * 10.0 ** g.integers(-3, 4)
        a, b = g.standard_normal(2) * 10.0 ** g.integers(-5, 6, 2)
        if g.random() < 0.3:
            b = a
        w = x1 - x0
        x = np.sort(np.concatenate([g.uniform(x0 - 50 * w, x1 + 50 * w, 200), neighbours(x0, 5), neighbours(x1, 5)]))
        v = extended(x0, x1, a, b, x)
        up, down = v[1:] >= v[:-1], v[1:] <= v[:-1]  # (np.diff could overflow)
        assert np.all(up) if b >= a else np.all(down)
        assert extended(x0, x1, a, b, [x0])[0] == a and extended(x0, x1, a, b, [x1])[0] == b


def test_wide_and_subnormal_widths():
    # a width that overflows is halved (decision IP27), a subnormal width is exact enough to keep the ends
    for x0, x1 in ((-1e308, 1.5e308), (-1.7976931348623157e308, 1.7976931348623157e308), (1e-310, 1e-310 + 3e-320)):
        pts = neighbours(x0, 20) + neighbours(x1, 20) + list(np.linspace(x0 / 2, x1 / 2, 50) * 2)
        check_interval(x0, x1, 2.0, -3.0, pts)
        assert np.all(np.isfinite(on_interval(x0, x1, 2.0, -3.0, [v for v in pts if x0 <= v <= x1])))
    # far outside: x - x0 overflows, still finite and on the line's side
    v = extended(-1e308, -0.9e308, 0.0, 1.0, [1.7e308])
    assert np.isfinite(v[0]) and v[0] > 1.0


@st.composite
def interval(draw):
    fin = st.floats(allow_nan=False, allow_infinity=False)
    x0 = draw(fin)
    x1 = draw(fin.filter(lambda v: v > x0))
    a, b = draw(fin), draw(fin)
    if draw(st.booleans()):
        b = a
    k = draw(st.integers(0, 30))
    inner = draw(st.lists(st.floats(min_value=x0, max_value=x1), max_size=30))
    return x0, x1, a, b, neighbours(x0, k) + neighbours(x1, k) + inner


@settings(deadline=None, max_examples=examples(300))
@given(interval())
def test_hypothesis_properties(case):
    check_interval(*case)


@settings(deadline=None, max_examples=examples(200))
@given(interval(), st.lists(st.floats(allow_nan=False, allow_infinity=False), min_size=1, max_size=20))
def test_hypothesis_extension_monotone(case, far):
    x0, x1, a, b, pts = case
    x = np.sort(np.array(pts + far, dtype=float))
    with np.errstate(all="ignore"):
        v = extended(x0, x1, a, b, x)
    ok = np.isfinite(v)
    if ok.all():
        up, down = v[1:] >= v[:-1], v[1:] <= v[:-1]  # (np.diff could overflow)
        assert np.all(up) if b >= a else np.all(down)
