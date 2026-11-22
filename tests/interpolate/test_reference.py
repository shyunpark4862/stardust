"""The linear contract against the exact references (docs/design/interpolate.md §4; tests/interpolate/ref/linear.npz,
exact rationals rounded once): |got - y| <= gamma_6 (|a| + |b|) + 3 eta inside the interval and gamma_10 |t| (|a| + |b|)
+ 3 eta outside (refdata.linear_bound), and exactly the data at the nodes.

Inside, sd.interp on the whole curve (and on the two nodes of each kernel interval); outside, the linear path of the
interpolate area with the end pieces extended (what the k = 1 B-spline and the grid's fill_value=None will use);
sd.interp clamps there, which the test checks too."""

from fractions import Fraction

import numpy as np

import stardust as sd

from . import refdata
from . import samples as S

LIN = refdata.ref("linear")
M = LIN.methods["lerp"]
LINEAR = sd._core.interpolate.linear


def _points(sid):
    """(positions in the method's arrays, point numbers) of a set, in the set's point order."""
    pos = np.flatnonzero(LIN.inputs["set"][M.index] == sid)
    idx = M.index[pos]
    order = np.argsort(LIN.inputs["index"][idx], kind="stable")
    return pos[order], idx[order]


def _units(got, pos, idx):
    """Each point's error in units of the contract; inf where a node's value is not the data exactly."""
    out = np.empty(len(idx))
    for q, (p, k) in enumerate(zip(pos, idx)):
        x0, x1, a, b, x = (float(LIN.inputs[n][k]) for n in ("x0", "x1", "a", "b", "x"))
        t = (Fraction(x) - Fraction(x0)) / (Fraction(x1) - Fraction(x0))
        if t in (0, 1):  # a node: exact, to the sign of zero
            want = a if t == 0 else b
            same = got[q] == want and np.signbit(got[q]) == np.signbit(want)
            out[q] = 0.0 if same else np.inf
            continue
        out[q] = refdata.linear_units(float(got[q]), M.hi[p], M.lo[p], a, b, float(t))
    return out


def _curve(cid):
    c = S.curve(cid)
    pos, idx = _points(f"curve/{cid}")
    assert np.array_equal(LIN.inputs["x"][idx], c.xe)
    return c, pos, idx


def test_interp_on_the_curves_inside():
    worst = 0.0
    for cid in S.real_curve_ids():
        c, pos, idx = _curve(cid)
        got = np.asarray(sd.interp(sd.asarray(c.xe), sd.asarray(c.x), sd.asarray(c.y)))
        inside = (c.xe >= c.x[0]) & (c.xe <= c.x[-1])
        u = _units(got[inside], pos[inside], idx[inside])
        assert np.all(u <= 1), (cid, float(u.max()))
        worst = max(worst, float(u.max(initial=0.0)))
        # outside: numpy's clamp to the end values
        assert np.all(got[c.xe < c.x[0]] == c.y[0]) and np.all(got[c.xe > c.x[-1]] == c.y[-1])
    assert worst > 0  # the contract is not vacuous: some points round


def test_linear_path_on_the_curves_everywhere():
    for cid in S.real_curve_ids():
        c, pos, idx = _curve(cid)
        got = np.asarray(LINEAR(sd.asarray(c.xe), sd.asarray(c.x), sd.asarray(c.y.reshape(-1, 1).copy()), 1))[:, 0]
        u = _units(got, pos, idx)
        assert np.all(u <= 1), (cid, float(u.max()))
        inside = (c.xe >= c.x[0]) & (c.xe <= c.x[-1])
        mine = np.asarray(sd.interp(sd.asarray(c.xe[inside]), sd.asarray(c.x), sd.asarray(c.y)))
        assert np.array_equal(got[inside], mine)  # one kernel, the same bits (§2.2)


def test_kernel_intervals():
    for L in S.lerps():
        if not S.lerp_finite(L):
            continue
        pos, idx = _points(f"lerp/{L.id}")
        x = LIN.inputs["x"][idx]
        assert np.array_equal(x, L.x)
        inside = (x >= L.x0) & (x <= L.x1)
        got = np.asarray(sd.interp(sd.asarray(x[inside]), sd.asarray([L.x0, L.x1]), sd.asarray([L.a, L.b])))
        u = _units(got, pos[inside], idx[inside])
        assert np.all(u <= 1), (L.id, float(u.max()))
        ext = np.asarray(LINEAR(sd.asarray(x), sd.asarray([L.x0, L.x1]), sd.asarray([[L.a], [L.b]]), 1))[:, 0]
        u = _units(ext, pos, idx)
        assert np.all(u <= 1), (L.id, float(u.max()))
