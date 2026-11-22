"""Every linear path gives the same bits (docs/design/interpolate.md §2.2, §3.2): sd.interp and the interpolate
area's linear path (_core.interpolate.linear, which the k = 1 B-spline and the 1-D linear grid will evaluate through)
agree on [xp[0], xp[-1]]; a batch of columns is each column alone; complex values are their parts. The k = 1 B-spline
(milestone IP3) evaluates through the linear path; the grid joins this test when it is public (milestone IP5)."""

import math

import numpy as np

import stardust as sd

from .._bits import bits_equal
from . import samples as S

A = sd.asarray
LINEAR = sd._core.interpolate.linear
NAN_OUTSIDE, EXTEND, PERIODIC = 0, 1, 2


def _col(y):
    return A(np.ascontiguousarray(np.asarray(y).reshape(len(y), -1)))


def test_interp_and_the_linear_path_on_every_curve():
    for cid in S.curve_ids():
        c = S.curve(cid)
        if c.n < 2 or np.iscomplexobj(c.y) or np.asarray(c.y).ndim != 1:
            continue
        inside = c.xe[(c.xe >= c.x[0]) & (c.xe <= c.x[-1])]
        mine = sd.interp(A(inside), A(c.x), A(c.y))
        for rule in (NAN_OUTSIDE, EXTEND):
            got = np.asarray(LINEAR(A(inside), A(c.x), _col(c.y), rule))[:, 0]
            assert bits_equal(got, mine), (cid, rule)
        # periodic: the value at the wrapped point x[0] + (x - x[0]) % (x[-1] - x[0]) (scipy's rule: x[-1] is x[0],
        # and the wrap itself may round a point inside)
        got = np.asarray(LINEAR(A(c.xe), A(c.x), _col(c.y), PERIODIC))[:, 0]
        wrapped = c.x[0] + np.remainder(c.xe - c.x[0], c.x[-1] - c.x[0])
        assert bits_equal(got, np.asarray(sd.interp(A(wrapped), A(c.x), A(c.y)))), cid


def test_outside_rules():
    xp, fp = np.array([0.0, 1.0, 3.0]), np.array([2.0, -1.0, 5.0])
    x = np.array([-2.0, -0.5, 0.0, 3.0, 3.5, 7.0, math.nan])
    nan = np.asarray(LINEAR(A(x), A(xp), _col(fp), NAN_OUTSIDE))[:, 0]
    assert np.array_equal(nan, [math.nan, math.nan, 2.0, 5.0, math.nan, math.nan, math.nan], equal_nan=True)
    ext = np.asarray(LINEAR(A(x), A(xp), _col(fp), EXTEND))[:, 0]
    assert ext[:6].tolist() == [8.0, 3.5, 2.0, 5.0, 6.5, 17.0] and math.isnan(ext[6])
    # periodic: wrapped into [xp[0], xp[-1]] with numpy's remainder, as scipy's PPoly does
    per = np.asarray(LINEAR(A(x), A(xp), _col(fp), PERIODIC))[:, 0]
    wrapped = xp[0] + np.remainder(x - xp[0], xp[-1] - xp[0])
    assert bits_equal(per[:6], np.asarray(sd.interp(A(wrapped[:6]), A(xp), A(fp))))
    assert math.isnan(per[6])


def test_columns_and_complex_parts():
    g = np.random.default_rng(9)
    xp = np.sort(g.standard_normal(50))
    y = g.standard_normal((50, 4))
    x = g.standard_normal(3000) * 1.5
    for rule in (NAN_OUTSIDE, EXTEND, PERIODIC):
        got = np.asarray(LINEAR(A(x), A(xp), A(y), rule))
        for k in range(4):
            assert bits_equal(got[:, k], np.asarray(LINEAR(A(x), A(xp), _col(y[:, k]), rule))[:, 0]), (rule, k)
        z = y[:, :2] + 1j * y[:, 2:]
        gz = np.asarray(LINEAR(A(x), A(xp), A(z), rule))
        assert bits_equal(gz.real, got[:, :2]) and bits_equal(gz.imag, got[:, 2:]), rule
    # a NaN point: NaN in both parts of a complex value
    gz = np.asarray(LINEAR(A(np.array([math.nan])), A(xp), A(y[:, :1] + 1j), EXTEND))
    assert math.isnan(gz[0, 0].real) and math.isnan(gz[0, 0].imag)


def test_shapes_of_the_private_call():
    xp, y = A([0.0, 1.0]), A([[0.0], [1.0]])
    assert LINEAR(A(np.zeros(0)), xp, y, 1).shape == (0, 1)
    assert LINEAR(A([0.5]), xp, A(np.zeros((2, 0))), 1).shape == (1, 0)


def test_the_k1_bspline_is_the_linear_path():
    """make_interp_spline(xp, fp, k=1) and BSpline(t, c, 1) on every curve, the three outside rules, a batch and
    complex values: the linear path's bits (and so sd.interp's inside)."""
    IP = sd.interpolate
    g = np.random.default_rng(10)
    for cid in S.curve_ids():
        c = S.curve(cid)
        if c.n < 2:
            continue
        y = np.asarray(c.y)
        b = IP.make_interp_spline(c.x, y, k=1)
        cols = np.ascontiguousarray(y.reshape(c.n, -1))
        for ext, rule in (("periodic", PERIODIC), (False, NAN_OUTSIDE), (True, EXTEND)):
            got = np.asarray(b(c.xe, extrapolate=ext)).reshape(c.xe.size, -1)
            assert bits_equal(got, np.asarray(LINEAR(A(c.xe), A(c.x), A(cols), rule))), (cid, ext)
    xp = np.sort(g.standard_normal(40))
    y = g.standard_normal((40, 3)) + 1j * g.standard_normal((40, 3))
    x = g.standard_normal(500)
    got = np.asarray(IP.make_interp_spline(xp, y, k=1)(x))
    inside = (x >= xp[0]) & (x <= xp[-1])
    for k in range(3):
        assert bits_equal(got[inside, k], np.asarray(sd.interp(A(x[inside]), A(xp), A(y[:, k]))))
