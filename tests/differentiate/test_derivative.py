"""derivative (docs/api/differentiate.md §2; docs/design/integrate.md §3.8, decisions C26-C37):
scipy 1.18.1's evaluation points, counts and statuses on the whole collection
(tests/differentiate/data/scipy_derivative.json), the Arb contract (a converged derivative is
within atol + rtol |df| of the exact one on the collection), the call form (C26, O38, O48), the
inputs' rules and the results' shapes."""

import hashlib
import math

import numpy as np
import pytest

import stardust as sd

from . import problems as P
from . import refdata

D = sd.differentiate
TINY = 2.2250738585072014e-308
RTOL = 2.0**-26


class Recorder:
    """The points a function is called with, as tools/gen_differentiate_scipy.py records scipy's:
    the shapes, the number of points and the sha256 of their float64 bytes in call order."""

    def __init__(self, fn):
        self.fn, self.shapes, self.count, self.h = fn, [], 0, hashlib.sha256()

    def __call__(self, x, *args):
        a = np.asarray(x, dtype=np.float64)
        self.shapes.append(tuple(a.shape))
        self.count += a.size
        self.h.update(np.ascontiguousarray(a).tobytes())
        return sd.asarray(self.fn(a, *[np.asarray(t) for t in args]))


def _scipy_shapes(rec):
    """scipy's call shapes, one entry per call as sd makes them: sd's first call is 1-D (O48), the
    batch's elements in C order ((): (1,); a 3 x 3 batch: (9,)); the same points in the same order."""
    out = []
    for k, (shape, times) in enumerate(rec["shapes"]):
        first = (int(np.prod(shape)),) if k == 0 else tuple(shape)
        out += [first] + [tuple(shape)] * (times - 1)
    return out


def _runs():
    """(key, f(x, *args) on numpy, x, options, args) of every scipy record."""
    for f in P.ELEMENTARY:
        for x in f.points:
            yield f"{f.name}/{x!r}", f.f_np, x, {}, ()
        yield f"batch/{f.name}", f.f_np, np.array(f.points), {}, ()
    for e in P.EDGES:
        yield f"edge/{e.id}", P.BY_NAME[e.function].f_np, e.x, e.options, ()
    for f in P.ELEMENTARY:
        yield f"nd/{f.name}", f.f_np, np.array(f.points).reshape(P.ND_SHAPE), {}, ()
    for aid, g, x, args in P.ARGS:
        yield aid, g, x, {}, args


RUNS = list(_runs())
# x^3 at 0: the exact weights and the sum over the pairs of nodes make every estimate the same
# tiny value, so the error is 0 and the second iteration converges; scipy's LAPACK-solved weights
# and BLAS sum leave a 1e-14 jitter that its default atol never accepts (status -2 after
# max_iter, docs/api/differentiate.md §1.4)
SD_CONVERGES = {"edge/cube/0/default"}


def test_scipy_points_counts_and_statuses():
    data = refdata.data("scipy_derivative.json")
    assert data.pop("scipy") == "1.18.1"
    assert {k for k, *_ in RUNS} == set(data)
    same_df = 0
    for key, f, x, opts, args in RUNS:
        rec = Recorder(f)
        with np.errstate(all="ignore"):
            r = D.derivative(rec, x, args=tuple(sd.asarray(a) for a in args), **opts)
        want = data[key]
        if key in SD_CONVERGES:
            assert r.status == 0 and r.n_iter == 2 and r.nfev == 11 and want["status"] == -2
            assert r.df == 0.0 or abs(r.df) < 1e-15
            continue
        assert rec.h.hexdigest() == want["calls"]["sha256"], key  # every point, to the bit, in order
        assert rec.shapes == _scipy_shapes(want["calls"]) and rec.count == want["calls"]["count"], key
        for k in ("status", "n_iter", "nfev", "converged"):
            assert np.asarray(getattr(r, k)).tolist() == want[k], (key, k)
        assert np.asarray(r.x).tolist() == want["x"]
        got, ref = np.asarray(r.df, dtype=float), np.asarray(want["df"], dtype=float)
        ok = np.asarray(want["status"]) == 0
        # the weights differ (C28) and scipy sums with BLAS: the last digits, within the tolerance
        # both passed (not always within the error estimates)
        tol = opts.get("atol", TINY) + opts.get("rtol", RTOL) * np.abs(ref)
        assert np.all(np.abs(got - ref)[ok] <= tol[ok]), key
        assert np.array_equal(np.isnan(got), np.isnan(ref)), key
        same_df += np.array_equal(got, ref, equal_nan=True)
    assert same_df == 5  # of 124 records: the weights and the sum differ (docs/design/integrate.md §7.8)


def test_arb_contract():
    """A converged derivative is within atol + rtol |df| of the exact derivative at every point of
    the collection (the stopping test's meaning); the worst relative error is 6.2e-11 (tanh at
    -0.5)."""
    ref = refdata.derivatives()
    worst, converged = 0.0, 0
    points = [(f"{f.name}/{x!r}", f, x, {}) for f in P.ELEMENTARY for x in f.points]
    points += [(f"edge/{e.id}", P.BY_NAME[e.function], e.x, e.options) for e in P.EDGES]
    for key, f, x, opts in points:
        with np.errstate(all="ignore"):
            r = D.derivative(lambda a, f=f: sd.asarray(f.f_np(np.asarray(a))), x, **opts)
        if r.status != 0:
            assert key in {"edge/log/1e-3/default", "edge/log/1e-3/right", "edge/sqrt/1e-6/default",
                           "edge/exp/1/order-2", "edge/exp/1/factor-0.5"}, key  # fmt: skip
            continue
        converged += 1
        hi, lo = ref[f"{f.name}/{float(x)!r}"]["df"]
        err = abs((r.df - hi) - lo)
        assert err <= opts.get("atol", TINY) + opts.get("rtol", RTOL) * abs(r.df), key
        if hi:
            worst = max(worst, err / abs(hi))
    assert converged == 111
    assert worst < 1e-10


def test_batch_is_the_elements_alone():
    # each element's arithmetic is its own: a batch gives every element's bits alone (scipy's
    # BLAS sum does not: docs/design/integrate.md §7.7), for any thread count
    for f in P.ELEMENTARY:
        b = D.derivative(lambda a, f=f: sd.asarray(f.f_np(np.asarray(a))), sd.asarray(f.points))
        for i, x in enumerate(f.points):
            one = D.derivative(lambda a, f=f: sd.asarray(f.f_np(np.asarray(a))), x)
            assert [b.df[i], b.error[i], b.status[i], b.n_iter[i], b.nfev[i]] == [
                one.df,
                one.error,
                one.status,
                one.n_iter,
                one.nfev,
            ] or (math.isnan(one.df) and math.isnan(b.df[i]))


def test_threads_and_large_batches():
    x = sd.linspace(-3.0, 3.0, 40001)  # above the 2^15 elements of the parallel passes
    a = sd.linspace(0.5, 2.0, 40001)
    got = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        try:
            r = D.derivative(lambda z, a: sd.sin(a * z) / a, x, args=(a,))
        finally:
            sd.set_num_threads(10)
        got.append([np.asarray(getattr(r, k)).tobytes() for k in ("df", "error", "status", "n_iter", "nfev")])
    assert got[0] == got[1] == got[2]
    want = np.cos(np.asarray(a) * np.asarray(x))
    assert np.max(np.abs(np.asarray(r.df) - want)) < 1e-9


def test_threads_by_points():
    # C111: 5,000 elements of 8 points run on the threads (the passes split by points); the bits are
    # those of 1 thread and of each element alone
    x = sd.linspace(-3.0, 3.0, 5000)
    got = []
    for t in (1, 3, 10):
        sd.set_num_threads(t)
        try:
            r = D.derivative(sd.sin, x)
        finally:
            sd.set_num_threads(10)
        got.append([np.asarray(getattr(r, k)).tobytes() for k in ("df", "error", "status", "n_iter", "nfev")])
    assert got[0] == got[1] == got[2]
    for i in (0, 2500, 4999):
        one = D.derivative(sd.sin, float(x[i]))
        assert (one.df, one.error, one.n_iter) == (float(r.df[i]), float(r.error[i]), int(r.n_iter[i]))


def test_call_form():
    # C26, O38, O48: the first call (k,) on every element, then (k, p) of the elements in
    # progress with the args as (k, 1); p = order points, then 2
    seen = []

    def f(x, c):
        seen.append((tuple(x.shape), tuple(c.shape), np.asarray(c).ravel().tolist()))
        return sd.exp(c * x) if x.ndim == 1 else sd.exp(c * x)

    r = D.derivative(f, [0.0, 1.0, 2.0], args=([1.0, 50.0, 1.0],), max_iter=3)
    assert seen[0] == ((3,), (3,), [1.0, 50.0, 1.0])
    assert seen[1] == ((3, 8), (3, 1), [1.0, 50.0, 1.0])
    assert seen[2] == ((3, 2), (3, 1), [1.0, 50.0, 1.0])
    assert r.status.tolist()[0] == 0 and r.n_iter.tolist() == [2, 3, 2]
    assert seen[3] == ((1, 2), (1, 1), [50.0])  # only the element in progress
    assert r.nfev.tolist() == [11, 13, 11]
    # order: odd rounds up to the next even (scipy's (order + 1) // 2 terms); one-sided 2n points
    shapes = []
    D.derivative(lambda x: shapes.append(x.shape) or sd.exp(x), 1.0, order=5, step_direction=1, max_iter=2)
    assert shapes == [(1,), (1, 6), (1, 2)]


def test_results_shapes_and_fields():
    r = D.derivative(sd.exp, 1.0)
    assert type(r.df) is float and type(r.converged) is bool and type(r.status) is int
    assert r.converged and r.status == 0 and r.n_iter == 2 and r.nfev == 11 and r.x == 1.0
    assert abs(r.df - math.e) < 1e-13 and 0 < r.error < 1e-10
    for name in ("ddf", "message", "success", "nit"):
        assert not hasattr(r, name)
    b = D.derivative(sd.exp, sd.zeros((2, 3)), initial_step=[0.5, 0.25, 0.125])
    assert b.df.shape == (2, 3) and b.x.shape == (2, 3) and b.status.dtype == sd.int64
    assert b.converged.dtype == sd.bool and bool(sd.all(b.converged))
    with pytest.raises(ValueError, match="read-only"):
        b.df[0, 0] = 1.0
    assert repr(b) == "DifferentiateResult(batch shape (2, 3), 6 of 6 converged)"
    e = D.derivative(sd.exp, [])
    assert e.df.shape == (0,) and repr(e) == "DifferentiateResult(batch shape (0,), 0 of 0 converged)"


def test_elementwise_failures():
    # scipy's per-element outcomes: a step <= 0 is NaN (-3), so is a NaN step_direction (C32);
    # a NaN from f stops its element with -3, the others go on; f(x) not finite is NaN too
    r = D.derivative(sd.exp, [1.0, 1.0, 1.0, 1.0], initial_step=[0.5, 0.0, -1.0, math.nan])
    assert r.status.tolist() == [0, -3, -3, -3] and r.nfev.tolist() == [11, 9, 9, 9]
    assert [math.isnan(v) for v in r.df.tolist()] == [False, True, True, True]
    r = D.derivative(sd.exp, 1.0, step_direction=math.nan)
    assert r.status == -3 and math.isnan(r.df) and r.nfev == 9 and r.n_iter == 1
    r = D.derivative(sd.log, [1e-3, 1.0])
    assert r.status.tolist() == [-3, 0]
    r = D.derivative(lambda x: sd.where(x == 1.0, math.nan, sd.exp(x)), 1.0)
    assert r.status == -3 and math.isnan(r.df)  # f(x) enters with weight 0: 0 * NaN
    r = D.derivative(sd.exp, math.inf)
    assert r.status == -3 and r.nfev == 9
    r = D.derivative(sd.exp, 1.0, max_iter=1)
    assert r.status == -2 and r.n_iter == 1 and r.nfev == 9 and not r.converged
    r = D.derivative(sd.exp, 1.0, step_factor=0.5)  # growing steps: the error grows (-1), as scipy
    assert r.status == -1 and r.n_iter == 3


def test_one_sided_and_options():
    for d in (1, -1, 2.5, -7):
        r = D.derivative(sd.log, 1e-3, step_direction=d, initial_step=1e-4)
        assert r.converged and abs(r.df - 1e3) < 1e-6, d
    r = D.derivative(sd.exp, 1.0, atol=0.0, rtol=1e-4)
    assert r.converged and r.n_iter == 2
    r = D.derivative(sd.exp, 1.0, order=12)
    assert r.converged and abs(r.df - math.e) < 1e-12
    r = D.derivative(sd.exp, 1.0, step_factor=1.5)
    assert r.converged and abs(r.df - math.e) < 1e-10


def test_argument_errors():
    with pytest.raises(TypeError, match=r"derivative\(\): 'f' must be callable, not float"):
        D.derivative(1.0, 1.0)
    with pytest.raises(TypeError, match=r"derivative\(\): unexpected keyword argument 'tolerances'"):
        D.derivative(sd.exp, 1.0, tolerances={})
    with pytest.raises(TypeError, match=r"derivative\(\): unexpected keyword argument 'maxiter'"):
        D.derivative(sd.exp, 1.0, maxiter=3)
    with pytest.raises(TypeError, match=r"derivative\(\): unexpected keyword argument 'preserve_shape'"):
        D.derivative(sd.exp, 1.0, preserve_shape=True)
    with pytest.raises(ValueError, match=r"derivative\(\): 'step_factor' must be a finite number > 0 other than 1"):
        D.derivative(sd.exp, 1.0, step_factor=1.0)
    for bad in (0.0, -2.0, math.inf, math.nan):  # docs/api/differentiate.md §2, C31: one message for all
        with pytest.raises(
            ValueError, match=r"^derivative\(\): 'step_factor' must be a finite number > 0 other than 1$"
        ):
            D.derivative(sd.exp, 1.0, step_factor=bad)
    with pytest.raises(ValueError, match=r"derivative\(\): 'order' must be >= 1"):
        D.derivative(sd.exp, 1.0, order=0)
    with pytest.raises(TypeError, match=r"derivative\(\): 'order' must be an int, not float"):
        D.derivative(sd.exp, 1.0, order=8.0)
    with pytest.raises(ValueError, match=r"derivative\(\): 'max_iter' must be >= 1"):
        D.derivative(sd.exp, 1.0, max_iter=0)
    with pytest.raises(ValueError, match=r"derivative\(\): 'atol' must be a finite number >= 0"):
        D.derivative(sd.exp, 1.0, atol=-1.0)
    with pytest.raises(ValueError, match=r"derivative\(\): 'rtol' must be a finite number >= 0"):
        D.derivative(sd.exp, 1.0, rtol=math.nan)
    with pytest.raises(TypeError, match=r"derivative\(\): 'x' must be a real number or an array, not bool"):
        D.derivative(sd.exp, True)
    with pytest.raises(ValueError, match=r"derivative\(\): the inputs must broadcast together"):
        D.derivative(sd.exp, [1.0, 2.0], initial_step=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"derivative\(\): 'args' must be a tuple, not list"):
        D.derivative(sd.exp, 1.0, args=[1.0])


def test_value_errors():
    with pytest.raises(
        ValueError, match=r"derivative\(\): 'f' must return shape \(1,\) like its argument, not a scalar"
    ):
        D.derivative(lambda x: 1.0, 1.0)
    with pytest.raises(ValueError, match=r"'f' must return shape \(2, 8\) like its argument, not \(2, 1\)"):
        D.derivative(lambda x: x if x.ndim == 1 else x[:, :1], [1.0, 2.0])
    with pytest.raises(TypeError, match=r"derivative\(\): 'f' must return real values, not a bool array"):
        D.derivative(lambda x: x > 0, 1.0)
    with pytest.raises(TypeError, match=r"'f' must return an sd.ndarray or a sequence of numbers, not str"):
        D.derivative(lambda x: "a", 1.0)
    with pytest.raises(TypeError, match=r"derivative\(\): 'f' must return real values, not a complex128 array"):
        D.derivative(lambda x: x * 1j, 1.0)

    def boom(x):
        raise KeyError("mine")

    with pytest.raises(KeyError, match="mine"):  # the user's error as it is
        D.derivative(boom, 1.0)
    # numpy arrays and sequences are accepted and converted
    r = D.derivative(lambda x: np.exp(np.asarray(x)), [1.0, 2.0])
    assert bool(sd.all(r.converged))
    r = D.derivative(lambda x: np.asarray(x, dtype=np.float32) * 0 + 1, 1.0)
    assert r.df == 0.0 and r.converged


def test_summary():
    s = D.derivative(sd.exp, 1.0).summary()
    lines = s.splitlines()
    assert lines[0].strip() == "Differentiation Results" and set(lines[1]) == {"="} and len(lines[1]) == 78
    assert "Derivative:" in s and "2.71828" in s and "Error estimate:" in s and "Function Evals:" in s
    assert "Converged:" in s and "Point:" in s and "No. Iterations:" in s
    b = D.derivative(sd.exp, [1.0, 2.0]).summary()
    assert "Batch shape:" in b and "2 of 2" in b and "[1]" in b
    head = [line for line in b.splitlines() if "df" in line][0].split()
    assert head == ["x", "df", "error", "converged", "status", "n_iter", "nfev"]
    with pytest.raises(TypeError, match=r"summary\(\): takes 0 positional arguments but 1 was given"):
        D.derivative(sd.exp, 1.0).summary(1)


def test_nan_point_and_order_one():
    """docs/api/differentiate.md §1.4, §2: a NaN x ends with status -3 (df NaN) after the first
    iteration; order 1 is the next even order, 2 (the same points and value)."""
    r = D.derivative(sd.exp, [math.nan, 1.0])
    assert r.status.tolist() == [-3, 0] and math.isnan(r.df.tolist()[0])
    one, two = D.derivative(sd.exp, 1.0, order=1), D.derivative(sd.exp, 1.0, order=2)
    assert (one.df, one.error, one.nfev, one.n_iter, one.status) == (
        two.df,
        two.error,
        two.nfev,
        two.n_iter,
        two.status,
    )


def test_order_cap():
    # CAL-2 (C118): every even order up to 18 keeps the contract at the eleven functions' nine
    # points (Arb); from 20 on it does not, and higher orders converge to wrong values (exp'(1) = 0
    # at order 160) and take ever longer. The same cap for jacobian and hessian
    for o in (19, 20, 60, 10**9):
        with pytest.raises(ValueError, match=r"^derivative\(\): 'order' must be at most 18$"):
            D.derivative(sd.exp, 1.0, order=o)
    with pytest.raises(ValueError, match=r"^jacobian\(\): 'order' must be at most 18$"):
        D.jacobian(lambda x: x, [1.0], order=20)
    with pytest.raises(ValueError, match=r"^hessian\(\): 'order' must be at most 18$"):
        D.hessian(lambda x: x[0] ** 2, [1.0], order=20)
    for o in (17, 18):
        r = D.derivative(sd.exp, 1.0, order=o)
        assert r.converged and abs(r.df - math.e) <= 2.0**-26 * math.e


def test_one_sided_weights_only_when_a_side_is_taken():
    # CAL-7: the one-sided weights (in Q(sqrt(c)), most of the first call's cost) are computed only
    # when an element takes a side; the results are the same bits
    from stardust.differentiate import _weights as W

    c = 1.0 + 2.0**-20  # a step factor no other test uses: its weights are not kept yet
    W.one_sided_weights.cache_clear()
    central = D.derivative(sd.exp, sd.asarray([0.5, 1.0]), step_factor=c)
    D.hessian(lambda x: sd.exp(x[0]) * x[1] ** 2, [0.5, 1.0], step_factor=c)
    D.jacobian(lambda x: sd.exp(x), [0.5, 1.0], step_factor=c, step_direction=[0, math.nan])
    assert W.one_sided_weights.cache_info().currsize == 0
    mixed = D.derivative(sd.exp, sd.asarray([0.5, 1.0]), step_factor=c, step_direction=sd.asarray([0.0, 1.0]))
    assert W.one_sided_weights.cache_info().currsize == 1
    assert mixed.df[0] == central.df[0] and mixed.nfev[0] == central.nfev[0]
