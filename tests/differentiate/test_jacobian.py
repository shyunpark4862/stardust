"""jacobian and hessian (docs/api/differentiate.md §3; docs/design/integrate.md §3.8, decisions
C27, C29, C30, C33, C37): scipy 1.18.1's evaluation points and counts on the MGH collection
(tests/differentiate/data/scipy_jacobian.json, scipy_hessian.json), the Arb contract (normwise,
tests/differentiate/ref/mgh.npz), the vectorized call form (C27) and the results."""

import hashlib
import math

import numpy as np
import pytest

import stardust as sd

from . import problems as P
from . import refdata

D = sd.differentiate
RTOL = 2.0**-26

# The cases whose calls differ from scipy's: one entry or more stops at another iteration. Their
# entries are near-zero derivatives or at the stopping test's edge, where the last digits of df
# decide (the weights, C28, and the sums, sd's over the pairs of nodes and scipy's BLAS:
# docs/design/integrate.md §7.8); with scipy's own weights the same four Jacobians differ, so the
# sum order alone decides them.
JACOBIAN_CALLS_DIFFER = {"mgh20-n9-m31-x1", "mgh27-n30-m30-x1", "mgh27-n40-m40-x1", "mgh28-n10-m10-x1"}
HESSIAN_CALLS_DIFFER = {
    "mgh13-n4-m4-x1", "mgh10-n3-m16-x1", "mgh35-n9-m9-x1", "mgh04-n2-m3-x1", "mgh14-n4-m6-x1",
    "mgh18-n6-m13-x1", "mgh24-n10-m20-x1", "mgh30-n10-m10-x1", "mgh31-n10-m10-x1",
}  # fmt: skip


class Recorder:
    def __init__(self, fn):
        self.fn, self.calls, self.count, self.h = fn, 0, 0, hashlib.sha256()

    def __call__(self, x):
        a = np.asarray(x, dtype=np.float64)
        assert a.ndim == 2  # (m, N): C27
        self.calls += 1
        self.count += a.size
        self.h.update(np.ascontiguousarray(a).tobytes())  # scipy's (m, m, ..., p) in C order: the same bytes
        return sd.asarray(self.fn(a))


def _case_runs(kind):
    data = refdata.data(f"scipy_{kind}.json")
    assert data.pop("scipy") == "1.18.1"
    by_id = {c.id: c for c in P.mgh_cases()}
    for cid, want in data.items():
        c = by_id[cid]
        x0 = np.array([float(v) for v in c.x0()])
        fn = P.mgh_F_np(c) if kind == "jacobian" else P.mgh_f_np(c)
        rec = Recorder(fn)
        r = (D.jacobian if kind == "jacobian" else D.hessian)(rec, x0)
        yield cid, r, rec, want


@pytest.mark.parametrize("kind", ["jacobian", "hessian"])
def test_scipy_points_and_counts(kind):
    value = "df" if kind == "jacobian" else "ddf"
    differ = JACOBIAN_CALLS_DIFFER if kind == "jacobian" else HESSIAN_CALLS_DIFFER
    found, entries, same = set(), 0, 0
    for cid, r, rec, want in _case_runs(kind):
        status, nfev = np.asarray(r.status), np.asarray(r.nfev)
        agree = (status == np.asarray(want["status"])) & (nfev == np.asarray(want["nfev"]))
        entries += agree.size
        same += int(agree.sum())
        if rec.h.hexdigest() != want["calls"]["sha256"]:
            found.add(cid)
            continue
        assert rec.calls == want["calls"]["calls"] and rec.count == want["calls"]["count"], cid
        # the same calls, yet an entry may stop at another iteration (the calls go on while any
        # entry does): counted below
        got, ref = np.asarray(getattr(r, value)), np.asarray(want[value], dtype=float)
        both = (status == 0) & (np.asarray(want["status"]) == 0)
        scale = np.abs(ref).max()
        assert np.all(np.abs(got - ref)[both] <= 0.01 * RTOL * scale), cid  # worst 3e-4 rtol
    assert found == differ
    # the entries whose status and count are scipy's (the differing cases' other entries included)
    assert (entries, same) == ((7456, 7289) if kind == "jacobian" else (1748, 1517))


@pytest.mark.parametrize("kind", ["jacobian", "hessian", "gradient"])
def test_arb_contract(kind):
    """Normwise: a converged entry's error is below rtol times the largest entry of the exact
    matrix on the whole collection (the worst is 0.0037 rtol for the Jacobians, mgh11, and 0.11
    for the Hessians, mgh04: Brown's badly scaled function; the gradients of sum F_i^2, jacobian's
    scalar f, 0.0031 at mgh35 n = 8). Entrywise the relative test fails at near-zero entries (an
    absolute error of rtol |J| is noise there), so it is not the contract."""
    ref = refdata.mgh()
    worst, converged = 0.0, 0
    for cid, v in ref.items():
        if v[kind] is None:
            continue
        c = next(c for c in P.mgh_cases() if c.id == cid)
        x0 = np.array([float(t) for t in c.x0()])
        fn = P.mgh_F_np(c) if kind == "jacobian" else P.mgh_f_np(c)
        r = (D.hessian if kind == "hessian" else D.jacobian)(lambda x, fn=fn: sd.asarray(fn(np.asarray(x))), x0)
        got = np.asarray(r.ddf if kind == "hessian" else r.df)
        hi, lo = v[kind]
        ok = np.asarray(r.status) == 0
        converged += int(ok.sum())
        err = np.abs((got - hi) - lo)[ok]
        worst = max(worst, float(err.max(initial=0.0) / (RTOL * max(np.abs(hi).max(), 1e-300))))
    assert worst < (0.2 if kind == "hessian" else 0.01)
    assert converged == {"jacobian": 7389, "hessian": 1612, "gradient": 342}[kind]


def rosen(x):
    return sd.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2, axis=0)


def test_jacobian_call_form_and_shapes():
    seen = []

    def F(x):
        seen.append(tuple(x.shape))
        return sd.stack([x[0] * x[1], sd.sin(x[0]), x[1] ** 3])

    r = D.jacobian(F, [1.0, 2.0])
    assert seen[:3] == [(2, 2), (2, 16), (2, 4)]  # (m, m), then (m, m p): every column, every call
    assert all(s == (2, 4) for s in seen[2:])
    assert r.df.shape == (3, 2) and r.status.shape == (3, 2) and r.n_iter.shape == (3, 2)
    want = [[2.0, 1.0], [math.cos(1.0), 0.0], [0.0, 12.0]]
    assert np.allclose(np.asarray(r.df), want, atol=1e-12)
    assert r.nfev.tolist()[0] == [11, 11] and not hasattr(r, "x")
    assert repr(r).startswith("DifferentiateResult(batch shape (3, 2), ")
    g = D.jacobian(rosen, [1.2, 0.8, 1.1])
    assert g.df.shape == (3,) and np.allclose(np.asarray(g.df), [307.6, -275.6, 92.0], rtol=1e-12)
    # the columns: variable j's points with the others at x
    cols = []
    D.jacobian(lambda x: cols.append(np.asarray(x).copy()) or x[0] + x[1], [1.0, 2.0], max_iter=1)
    X = cols[1]  # (2, 2 * 8)
    assert np.all(X[1, :8] == 2.0) and np.all(X[0, 8:] == 1.0)
    assert sorted(X[0, :8] - 1.0) == sorted(X[1, 8:] - 2.0)


def test_jacobian_options():
    F = lambda x: sd.stack([sd.log(x[0]), sd.exp(x[1])])  # noqa: E731
    r = D.jacobian(F, [1e-3, 1.0], initial_step=[1e-4, 0.5])
    assert bool(sd.all(r.converged[:, 1])) and abs(r.df[0, 0] - 1e3) < 1e-5
    r = D.jacobian(F, [1e-3, 1.0], step_direction=[1, 0], initial_step=[1e-4, 0.5])
    assert abs(r.df[0, 0] - 1e3) < 1e-5
    r = D.jacobian(F, [1e-3, 1.0])  # x - 0.5 < 0 for the first variable: its column is NaN
    assert r.status.tolist()[0][0] == -3
    r = D.jacobian(F, [1.0, 1.0], max_iter=1)
    assert r.status.tolist() == [[-2, -2], [-2, -2]] and r.nfev.tolist() == [[9, 9], [9, 9]]


def test_hessian():
    r = D.hessian(rosen, [1.2, 0.8, 1.1])
    want = [[1410.0, -480.0, 0.0], [-480.0, 530.0, -320.0], [0.0, -320.0, 200.0]]
    assert np.allclose(np.asarray(r.ddf), want, atol=1e-9)
    assert r.ddf.shape == (3, 3) and r.nfev.dtype == sd.int64
    for name in ("df", "n_iter", "x"):
        assert not hasattr(r, name)
    # scipy's counts: an entry sums the inner evaluations of its gradient entry at the outer
    # calls until it stopped (121 = 11 + 8 * 11 + 2 * 11 at two outer iterations)
    assert r.nfev.tolist()[1][1] == 121
    # rtol: 0 or at least 100 eps (C29; scipy raises a smaller one to 100 eps with a warning)
    with pytest.raises(ValueError, match=r"hessian\(\): 'rtol' must be 0 or >= 2.220446049250313e-14"):
        D.hessian(rosen, [1.0, 1.0], rtol=1e-15)
    r = D.hessian(lambda x: x[0] ** 2 * x[1], [1.0, 2.0], rtol=0.0, atol=1e-6)
    assert bool(sd.all(r.converged))
    r = D.hessian(lambda x: x[0] ** 2 * x[1], [1.0, 2.0], rtol=100 * 2.0**-52)
    assert np.allclose(np.asarray(r.ddf), [[4.0, 2.0], [2.0, 0.0]], atol=1e-9)
    # initial_step: one per variable, in both differentiations (C30; scipy's broadcasting fails)
    r = D.hessian(lambda x: sd.log(x[0]) * x[1], [1e-3, 2.0], initial_step=[1e-4, 0.5])
    assert abs(r.ddf[0, 0] + 2e6) < 1e-2 and abs(r.ddf[0, 1] - 1e3) < 1e-6
    with pytest.raises(TypeError, match=r"hessian\(\): unexpected keyword argument 'step_direction'"):
        D.hessian(rosen, [1.0, 1.0], step_direction=1)


def test_errors():
    with pytest.raises(ValueError, match=r"jacobian\(\): 'x' must be 1-dimensional, not a scalar"):
        D.jacobian(rosen, 1.0)
    with pytest.raises(ValueError, match=r"jacobian\(\): 'x' must be 1-dimensional, not shape \(1, 2\)"):
        D.jacobian(rosen, [[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"hessian\(\): 'x' is empty \(no variables\)"):
        D.hessian(rosen, [])
    with pytest.raises(ValueError, match=r"jacobian\(\): 'initial_step' must have 2 values like 'x', not 3"):
        D.jacobian(rosen, [1.0, 2.0], initial_step=[1.0, 2.0, 3.0])
    with pytest.raises(TypeError, match=r"jacobian\(\): unexpected keyword argument 'args'"):
        D.jacobian(rosen, [1.0, 2.0], args=(1,))
    first = r"jacobian\(\): 'f' must return shape \(2,\) or \(k, 2\) for its argument of shape \(2, 2\)"
    with pytest.raises(ValueError, match=first + r", not \(3, 3\)"):
        D.jacobian(lambda x: sd.ones((3, 3)), [1.0, 2.0])
    with pytest.raises(
        ValueError, match=r"'f' must return shape \(2, 16\) for its argument of shape \(2, 16\), not \(1, 16\)"
    ):
        D.jacobian(lambda x: x if x.shape[1] == 2 else x[:1], [1.0, 2.0])
    with pytest.raises(
        ValueError, match=r"hessian\(\): 'f' must return shape \(4,\) for its argument of shape \(2, 4\), not \(2, 4\)"
    ):
        D.hessian(lambda x: x, [1.0, 2.0])
    with pytest.raises(TypeError, match=r"jacobian\(\): 'f' must return real values, not a bool array"):
        D.jacobian(lambda x: x > 0, [1.0, 2.0])
    # docs/api/differentiate.md §1.2: complex values and strings are TypeError, for both functions
    for fn in (D.jacobian, D.hessian):
        name = fn.__name__
        with pytest.raises(TypeError, match=rf"{name}\(\): 'f' must return real values, not a complex128 array"):
            fn(lambda x: x[0] * 1j, [1.0, 2.0])
        with pytest.raises(
            TypeError, match=rf"{name}\(\): 'f' must return an sd.ndarray or a sequence of numbers, not str"
        ):
            fn(lambda x: "a", [1.0, 2.0])
    # hessian's own checks (§3: the options are jacobian's)
    with pytest.raises(ValueError, match=r"hessian\(\): 'x' must be 1-dimensional, not a scalar"):
        D.hessian(rosen, 1.0)
    with pytest.raises(ValueError, match=r"hessian\(\): 'x' must be 1-dimensional, not shape \(1, 2\)"):
        D.hessian(rosen, [[1.0, 2.0]])
    with pytest.raises(ValueError, match=r"hessian\(\): 'initial_step' must have 2 values like 'x', not 3"):
        D.hessian(rosen, [1.0, 2.0], initial_step=[0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match=r"hessian\(\): 'max_iter' must be >= 1"):
        D.hessian(rosen, [1.0, 2.0], max_iter=0)

    def boom(x):
        raise KeyError("mine")

    with pytest.raises(KeyError, match="mine"):
        D.hessian(boom, [1.0, 2.0])


def test_summary():
    s = D.hessian(rosen, [1.2, 0.8, 1.1]).summary()
    assert "Batch shape:" in s and "(3, 3)" in s and "[2, 2]" in s
    head = [line for line in s.splitlines() if "ddf" in line][0].split()
    assert head == ["ddf", "error", "converged", "status", "nfev"]
    j = D.jacobian(rosen, [1.2, 0.8, 1.1]).summary()
    head = [line for line in j.splitlines() if "df" in line.split()][0].split()
    assert head == ["df", "error", "converged", "status", "n_iter", "nfev"]


def test_non_finite_points():
    """docs/api/differentiate.md §3: a non-finite variable is accepted and the entries it enters
    end with status -3 (scipy's); the others are untouched."""

    def f(x):
        return sd.stack([x[0] ** 2, x[0] * x[1]])

    j = D.jacobian(f, [1.0, math.inf])
    assert (
        j.status.tolist() == [[0, -3], [-3, -3]] and j.df.tolist()[0][0] == D.jacobian(f, [1.0, 2.0]).df.tolist()[0][0]
    )
    h = D.hessian(lambda x: x[0] ** 2 * x[1] + sd.sin(x[1]), [1.0, math.nan])
    assert h.status.tolist() == [[-3, -3], [-3, -3]] and all(math.isnan(v) for row in h.ddf.tolist() for v in row)


def test_hessian_options():
    """max_iter, order and step_factor reach the inner jacobian (§3)."""
    f = lambda x: x[0] ** 2 * x[1] + sd.sin(x[1])  # noqa: E731
    assert D.hessian(f, [1.0, 2.0], max_iter=1).status.tolist() == [[-2, -2], [-2, -2]]
    for kw in ({"order": 4}, {"step_factor": 3.0}, {"initial_step": [0.25, 0.5]}):
        r = D.hessian(f, [1.0, 2.0], **kw)
        assert r.converged.tolist() == [[True, True], [True, True]], kw
        assert abs(r.ddf.tolist()[0][1] - 2.0) <= 1e-8 and abs(r.ddf.tolist()[1][1] + math.sin(2.0)) <= 1e-8


def test_bits_do_not_depend_on_the_thread_count():
    """docs/api/differentiate.md §1.4: the same bits at any thread count (jacobian's entries pass
    the elementwise loop; 200 variables x 8 points pass its threshold of 2^15 points per pass
    with the 200 outputs)."""
    rng = np.random.default_rng(3)
    x0 = rng.uniform(0.5, 1.5, 200)

    def f(x):
        return sd.sin(x) * sd.exp(0.1 * x)  # (200, N): the diagonal Jacobian of a vector function

    def run():
        j = D.jacobian(f, x0)
        h = D.hessian(lambda x: sd.sum(sd.cos(x[:6]) * x[:6], axis=0), x0[:6])
        return (np.asarray(j.df.tolist()).view(np.int64).tolist(), j.nfev.tolist(),
                np.asarray(h.ddf.tolist()).view(np.int64).tolist(), h.nfev.tolist())  # fmt: skip

    old = sd.get_num_threads()
    try:
        got = {}
        for t in (1, 3, 10):
            sd.set_num_threads(t)
            got[t] = run()
    finally:
        sd.set_num_threads(old)
    assert got[1] == got[3] == got[10]
