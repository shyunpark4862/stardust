"""Every interpolation object pickles and copies (sd.ndarray itself does not pickle, as sd.spatial's KDTree):
the copy has the same class, the same arrays to the bit and computes the same bits; a modifying method on the
copy leaves the original as it was."""

import copy
import pickle

import numpy as np

import stardust as sd

IP = sd.interpolate


def _objects():
    g = np.random.default_rng(3)
    x = np.linspace(0.0, 1.0, 9)
    y = np.sin(3.0 * x)
    yb = np.stack([y, np.cos(x)], axis=1)
    pts, d = g.random((40, 2)), g.standard_normal(40)
    grid = ([0.0, 1.0, 2.0, 3.0], [0.0, 0.5, 1.5, 2.0, 3.0])
    gv = g.standard_normal((4, 5))
    q1 = np.linspace(-0.2, 1.2, 15)
    q2 = g.random((12, 2)) * 1.5
    qg = g.random((12, 2)) * 3.0
    return [
        (IP.PPoly(np.asarray(IP.CubicSpline(x, y).c), x), q1),
        (IP.CubicSpline(x, yb, bc_type="natural"), q1),
        (IP.CubicSpline(x, y + 1j * y), q1),
        (IP.PchipInterpolator(x, y), q1),
        (IP.Akima1DInterpolator(x, y, method="makima"), q1),
        (IP.CubicHermiteSpline(x, y, np.cos(x)), q1),
        (IP.make_interp_spline(x, y, k=5), q1),
        (IP.make_interp_spline(x, y, k=1), q1),
        (IP.BarycentricInterpolator(x, yb), q1),
        (IP.KroghInterpolator(x, y), q1),
        (IP.AAA(x, y), q1),
        (IP.AAA(x + 0j, y + 1j * y), q1 + 0.25j),
        (IP.FloaterHormannInterpolator(x, yb, d=2), q1),
        (IP.RegularGridInterpolator(grid, gv, method="cubic"), qg),
        (IP.RegularGridInterpolator(grid, gv + 1j * gv, bounds_error=False, fill_value=None), qg),
        (IP.RBFInterpolator(pts, d, kernel="gaussian", epsilon=2.0), q2),
        (IP.RBFInterpolator(pts, d, neighbors=10), q2),
        (IP.NearestNDInterpolator(pts, d, rescale=True), q2),
        (IP.LinearNDInterpolator(pts, d, fill_value=-1.0), q2),
        (IP.LinearNDInterpolator(g.random((30, 3)), d[:30] + 1j * d[10:], rescale=True), g.random((12, 3))),
        (IP.CloughTocher2DInterpolator(pts, np.stack([d, d[::-1]], axis=1), rescale=True), q2),
    ]


def _bits(v):
    a = np.asarray(v)
    return a.dtype, a.shape, a.tobytes()


def test_round_trips_compute_the_same_bits():
    for obj, q in _objects():
        want = _bits(obj(q))
        for twin in (pickle.loads(pickle.dumps(obj)), copy.deepcopy(obj), copy.copy(obj)):
            assert type(twin) is type(obj)
            assert repr(twin) == repr(obj)
            assert _bits(twin(q)) == want, type(obj).__name__


def test_arrays_and_their_write_protection_come_back():
    cs = IP.CubicSpline([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 0.0, 1.0])
    twin = pickle.loads(pickle.dumps(cs))
    assert _bits(twin.c) == _bits(cs.c) and _bits(twin.x) == _bits(cs.x)
    assert not twin.c._flags()["writeable"] and not twin.x._flags()["writeable"]
    assert twin.extrapolate == cs.extrapolate and twin.axis == cs.axis
    a = IP.AAA(np.linspace(-1.0, 1.0, 30), np.exp(np.linspace(-1.0, 1.0, 30)))
    poles = a.poles()  # cached on the object
    t = pickle.loads(pickle.dumps(a))
    assert _bits(t.poles()) == _bits(poles) and t.converged == a.converged
    assert _bits(t.errors) == _bits(a.errors)


def test_clough_tocher_keeps_its_gradients():
    g = np.random.default_rng(4)
    c = IP.CloughTocher2DInterpolator(g.random((50, 2)), g.standard_normal(50), maxiter=3)
    twin = pickle.loads(pickle.dumps(c))
    assert _bits(twin.grad) == _bits(c.grad) and twin.niter == c.niter == 3 and twin.converged is c.converged is False
    assert _bits(twin.tri.simplices) == _bits(c.tri.simplices) and _bits(twin.points) == _bits(c.points)
    assert not twin.grad._flags()["writeable"]


def test_a_modified_copy_leaves_the_original():
    b = IP.BarycentricInterpolator([0.0, 1.0, 2.0], [1.0, 2.0, 0.0])
    want = _bits(b(0.5))
    twin = pickle.loads(pickle.dumps(b))
    twin.add_xi([3.0], [5.0])
    assert _bits(b(0.5)) == want and twin.xi.shape == (4,) and b.xi.shape == (3,)
    p = IP.PPoly([[1.0, 2.0]], [0.0, 1.0, 2.0])
    twin = copy.deepcopy(p)
    twin.extend([[3.0]], [3.0])
    assert p.x.shape == (3,) and twin.x.shape == (4,)


def test_empty_arrays():
    p = IP.CubicSpline([0.0, 1.0, 2.0], np.zeros((3, 0)))
    twin = pickle.loads(pickle.dumps(p))
    assert twin.c.shape == p.c.shape and twin(0.5).shape == (0,)
