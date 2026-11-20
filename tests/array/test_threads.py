"""Threads. Every result must be bit-for-bit the same for any thread count
(API.md §3), and the pool must survive several Python threads at once."""

import math
import os
import threading
import time

import numpy as np
import pytest

import stardust as sd

from .conftest import assert_same, run_guarded

COUNTS = [1, 2, 3, 7, 10]


pytestmark = pytest.mark.usefixtures("restore_threads")


def bits(r):
    """Comparable bytes of any result (array, scalar or tuple of them)."""
    if isinstance(r, tuple):
        return tuple(bits(x) for x in r)
    if isinstance(r, sd.ndarray):
        a = np.asarray(r)
        return (a.shape, a.dtype.str, np.ascontiguousarray(a).tobytes())
    return repr(r)


def battery():
    """Large enough to cross the parallel threshold, in several layouts."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal(600_000)
    f[::997] = np.nan
    m = rng.standard_normal((700, 900))
    t3 = rng.standard_normal((60, 70, 80)).transpose(2, 0, 1)
    i = rng.integers(-1000, 1000, (500, 700))
    c = rng.standard_normal(300_000) + 1j * rng.standard_normal(300_000)
    b = rng.random((800, 600)) > 0.5
    idx = rng.integers(0, 600_000, 400_000)
    S = sd.asarray
    F, M, T3, I, C, B, IDX = S(f), S(m), S(t3), S(i), S(c), S(b), S(idx)
    clean = S(np.nan_to_num(f))
    P = S(1.0 + np.nan_to_num(f) * 1e-4)
    ops = {
        "f + m-ish": lambda: F * 2.0 + 1.0,
        "exp": lambda: sd.exp(M),
        "transposed copy": lambda: T3.copy(),
        "int + float (buffered cast)": lambda: I + M[:500, :700],
        "where": lambda: sd.where(S(b[:700, :600]), S(m[:700, :600]), 0.0),
        "clip": lambda: sd.clip(M, min=-0.5, max=0.5),
        "sum full": lambda: sd.sum(clean),
        "sum full nan": lambda: sd.sum(F),
        "sum axis 0": lambda: sd.sum(M, axis=0),
        "sum axis 1": lambda: sd.sum(M, axis=1),
        "sum 3d axes": lambda: sd.sum(T3, axis=(0, 2)),
        "sum complex": lambda: sd.sum(S(c)),
        "sum bool": lambda: sd.sum(B),
        "sum int": lambda: sd.sum(I),
        "prod int": lambda: sd.prod(I % 3 + 1),
        "mean/var/std": lambda: (sd.mean(M), sd.var(M), sd.std(M, axis=0)),
        "max/min nan": lambda: (sd.max(F), sd.min(clean), sd.max(M, axis=1)),
        "argmax/argmin": lambda: (sd.argmax(F), sd.argmin(clean), sd.argmax(M, axis=0), sd.argmin(M, axis=1)),
        "any/all": lambda: (sd.any(B), sd.all(B), sd.any(B, axis=0)),
        "cumulative_sum": lambda: sd.cumulative_sum(M, axis=1),
        "sort rows": lambda: (sd.sort(M), sd.argsort(M, axis=0)),
        "unique": lambda: sd.unique_values(I),
        "matmul int": lambda: I[:300, :400] @ I[:400, :350],
        "matmul float": lambda: M[:300] @ M[:300].T,
        "gather": lambda: F[IDX],
        "astype": lambda: sd.astype(M, sd.int64),
        "sum complex big": lambda: sd.sum(C),
        # Blocked over a reduced axis / multi-accumulator / blocked scans
        # (docs/design/array.md §6.9). Not numpy's bits, but still the same for
        # every thread count.
        "reduced-axis blocks": lambda: (
            sd.mean(M, axis=0),
            sd.var(M, axis=0),
            sd.prod(M, axis=0),
            sd.max(M, axis=0),
            sd.sum(B, axis=0),
            sd.sum(I, axis=0),
        ),
        "prod float/complex": lambda: (
            sd.prod(P),
            sd.prod(P[::3]),
            sd.prod(S(c[:200_000] / abs(c[:200_000]))),
            sd.prod(M[:, :400] * 0.001 + 1.0, axis=1),
        ),
        "long scans": lambda: (
            sd.cumulative_sum(clean),
            sd.cumulative_prod(P),
            sd.cumulative_sum(C),
            sd.cumulative_sum(F[::-2]),
            sd.cumulative_sum(I.reshape(-1)[:100_003]),
            sd.cumulative_sum(S(m.reshape(-1)[:90_000].reshape(30_000, 3)), axis=0),
            sd.cumulative_prod(P, include_initial=True),
        ),
        # Take along one axis, masks in chunks (docs/design/array.md §7.3, §7.4).
        "take with blocks": lambda: (M[:, S(idx[:500] % 900)], T3[:, :, S(idx[:40] % 70)], M[S(idx[:300] % 700)]),
        "mask gather": lambda: (F[S(f > 0.3)], F[S(f > 2.5)], M[S(m > 0)], M[:, S(m[0] > 0)]),
        "mask scatter": lambda: masked_assign(F, S(f > 0.3)),
        # Small batched BLAS matmuls on the pool (docs/design/array.md §9.4)
        "batched small matmul": lambda: (
            S(m[:640].reshape(10, 64, 900)[:, :, :64]) @ S(m[:640, :64].reshape(10, 64, 64)),
            S(m.reshape(-1)[:64_000].reshape(1000, 8, 8)) @ M[:8, :8],
            S(c[:32_000].reshape(500, 8, 8)) @ S(c[:32_000].reshape(500, 8, 8)),
        ),
        # One long lane sorted in pieces and merged (docs/design/array.md §8.5)
        "long 1-D sort": lambda: (
            sd.sort(F),
            sd.argsort(F),
            sd.sort(F, stable=False),
            sd.argsort(S(idx), descending=True),
            sd.unique_all(S(idx % 5000)),
            sd.argsort(S(b.reshape(-1)), stable=False),
            sd.argsort(S(i.reshape(-1) % 50.0), stable=False),
        ),
        # Full reductions with a cast, blocked (docs/design/array.md §6.5)
        "full reductions with a cast": lambda: (
            sd.sum(B),
            sd.mean(I),
            sd.mean(B),
            sd.sum(I, dtype=sd.float64),
            sd.var(I),
            sd.any(M),
            sd.all(F),
            sd.prod(B[:, :5]),
            sd.sum(S(c), dtype=sd.complex128),
            sd.mean(S(t3 > 0)),
        ),
    }
    return ops


def masked_assign(x, mask):
    a = x.copy()
    a[mask] = -1.0
    b = x.copy()
    b[mask] = sd.arange(float(sd.sum(mask)))
    return a, b


def test_results_do_not_depend_on_thread_count():
    ops = battery()
    reference = {}
    for n in COUNTS:
        sd.set_num_threads(n)
        for name, fn in ops.items():
            got = bits(fn())
            if n == COUNTS[0]:
                reference[name] = got
            else:
                assert got == reference[name], f"{name}: differs with {n} threads"


def test_parallel_sum_is_the_serial_tree():
    # The parallel pairwise sum walks the serial recursion tree: the same bits on any
    # thread count; numpy's tree has a narrower leaf, so only the value is close.
    x = np.random.default_rng(1).standard_normal(3_000_001)
    c = x[:2_000_000] + 1j * x[1_000_001:]
    got = []
    for n in (1, 10):
        sd.set_num_threads(n)
        got.append((sd.sum(sd.asarray(x)), sd.sum(sd.asarray(c))))
    assert got[0] == got[1]
    assert abs(got[0][0] - np.sum(x)) <= 1e-12 * np.sum(np.abs(x))
    assert abs(got[0][1] - np.sum(c)) <= 1e-12 * np.sum(np.abs(c))


def test_set_and_get_num_threads():
    sd.set_num_threads(3)
    assert sd.get_num_threads() == 3
    sd.set_num_threads(1)
    assert sd.get_num_threads() == 1
    for bad, exc in [(0, ValueError), (-1, ValueError), (True, TypeError), (2.0, TypeError), (2000, ValueError)]:
        with pytest.raises(exc):
            sd.set_num_threads(bad)
    sd.set_num_threads(np.int64(2))  # a numpy integer is an int (DESIGN.md §10.1 #13)
    assert sd.get_num_threads() == 2
    for bad, shown in ((np.True_, "numpy.bool"), (None, "NoneType")):
        with pytest.raises(TypeError, match=rf"^set_num_threads\(\): 'n' must be an int, not {shown}$"):
            sd.set_num_threads(bad)
    sd.set_num_threads(1)
    with pytest.raises(TypeError):
        sd.set_num_threads(n=2)  # positional-only


def test_default_is_the_performance_cores():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c", "import stardust as s; print(s.get_num_threads())"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    perf = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout)
    assert int(out) == perf


def test_several_python_threads_at_once():
    # While one thread's job holds the pool (with the GIL released), another
    # thread's job runs on its own thread: both must still be right.
    sd.set_num_threads(10)
    rng = np.random.default_rng(2)
    data = [rng.standard_normal(400_000) for _ in range(6)]
    sd.set_num_threads(1)  # the reference: computed alone, on one thread
    want = [(sd.sum(sd.asarray(d)), sd.max(sd.asarray(d)), sd.exp(sd.asarray(d)).tolist()[:1000]) for d in data]
    sd.set_num_threads(10)
    errors = []

    def work(k):
        try:
            x = sd.asarray(data[k])
            for _ in range(20):
                got = (sd.sum(x), sd.max(x), sd.exp(x).tolist()[:1000])
                if got[0] != want[k][0] or got[1] != want[k][1] or got[2] != want[k][2]:
                    errors.append(k)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(repr(e))

    threads = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []


def test_resize_while_others_compute():
    stop = False
    errors = []
    x = sd.asarray(np.ones(500_000))

    def work():
        while not stop:
            if sd.sum(x) != 500_000.0:
                errors.append("wrong")

    t = threading.Thread(target=work)
    t.start()
    for n in [1, 4, 10, 2, 10]:
        sd.set_num_threads(n)
    stop = True
    t.join()
    assert errors == []
    assert math.isfinite(sd.sum(x))


def test_full_reductions_with_a_cast_are_exact_or_close():
    # Blocked (docs/design/array.md §6.5): integer results exact, float sums
    # within rounding of numpy (not its bits: docs/api/array.md §7.8).
    rng = np.random.default_rng(2)
    b = rng.random(3_000_001) < 0.3
    i = rng.integers(-(10**6), 10**6, (1500, 2001))
    for n in (1, 10):
        sd.set_num_threads(n)
        assert sd.sum(sd.asarray(b)) == np.sum(b)
        assert sd.sum(sd.asarray(b[::3])) == np.sum(b[::3])
        assert sd.sum(sd.asarray(i), dtype=sd.float64) == pytest.approx(np.sum(i, dtype=np.float64), rel=1e-15)
        assert sd.mean(sd.asarray(i)) == pytest.approx(np.mean(i), rel=1e-12)
        assert sd.mean(sd.asarray(b)) == np.mean(b)  # an integer-valued sum is exact


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_parallel_call_in_a_forked_child():
    # The child has none of the parent's workers; it must not wait for them.
    rc, out = run_guarded(
        """
        import os, signal, warnings
        warnings.simplefilter("ignore", DeprecationWarning)
        import stardust as sd
        sd.set_num_threads(4)
        x = sd.ones(2_000_000)
        print(sd.sum(x), flush=True)
        pid = os.fork()
        if pid == 0:
            signal.alarm(20)
            ok = sd.sum(x * 2.0) == 4_000_000.0 and sd.get_num_threads() == 4
            os._exit(0 if ok else 3)
        _, status = os.waitpid(pid, 0)
        print("child", os.waitstatus_to_exitcode(status))
        """
    )
    assert rc == 0, out
    assert out.split() == ["2000000.0", "child", "0"]


def test_results_do_not_depend_on_the_thread_count_after_pauses():
    rng = np.random.default_rng(0)
    x = sd.asarray(rng.standard_normal(40_000))
    old = sd.get_num_threads()
    try:
        results = []
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            time.sleep(0.003)  # the workers fall asleep; late ones must not touch the next job
            results.append([np.asarray(x + x), sd.sum(x), np.asarray(sd.exp(x))])
        for r in results[1:]:
            assert np.array_equal(r[0], results[0][0]) and r[1] == results[0][1]
            assert np.array_equal(r[2], results[0][2])
    finally:
        sd.set_num_threads(old)


def test_gather_scatter_meshgrid_tril_on_threads():
    rng = np.random.default_rng(2)
    m = rng.standard_normal((300, 400))
    i, j = rng.integers(-300, 300, 200_000), rng.integers(0, 400, 200_000)
    v = rng.standard_normal(50_000)
    old = sd.get_num_threads()
    try:
        for n in (1, 10):
            sd.set_num_threads(n)
            sm = sd.asarray(m)
            assert_same(sm[sd.asarray(i), sd.asarray(j)], m[i, j])
            assert_same(sm[sd.asarray(i[:, None]), sd.asarray(j[:400])], m[i[:, None], j[:400]])  # 2-D index shape
            t = m.copy()
            st = sd.asarray(t.copy())
            st[sd.asarray(i), sd.asarray(j)] = 1.5
            t[i, j] = 1.5
            assert_same(st, t)
            for ix in ("xy", "ij"):
                for got, want in zip(
                    sd.meshgrid(sd.asarray(v[:700]), sd.asarray(v[:600]), indexing=ix),
                    np.meshgrid(v[:700], v[:600], indexing=ix),
                ):
                    assert_same(got, want)
            assert_same(sd.meshgrid(sd.asarray(v))[0], np.meshgrid(v)[0])
            assert_same(sd.tril(sm, k=3), np.tril(m, k=3))
            assert_same(sd.triu(sm, k=-2), np.triu(m, k=-2))
            with pytest.raises(IndexError, match="index 400 is out of bounds for axis 1 with size 400"):
                sm[sd.asarray(i[:3]), sd.asarray([0, 400, 1])]
    finally:
        sd.set_num_threads(old)


def test_sums_and_var_do_not_depend_on_the_thread_count():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(2_000_003)
    c = x[:1_000_000] + 1j * x[1_000_000:2_000_000]
    old = sd.get_num_threads()
    try:
        res = []
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            sx = sd.asarray(x)
            res.append((sd.sum(sx), sd.mean(sx), sd.var(sx), sd.sum(sd.asarray(c)), sd.var(sd.asarray(c))))
        assert all(r == res[0] for r in res)
    finally:
        sd.set_num_threads(old)
    assert abs(res[0][0] - np.sum(x)) <= 1e-12 * np.sum(np.abs(x))
    assert abs(res[0][2] - np.var(x)) <= 1e-13 * np.var(x)
    # a short sum below the leaf width is the plain left-to-right sum
    left_to_right = 0.0
    for v in [0.1] * 15:
        left_to_right += v
    assert sd.sum(sd.asarray([0.1] * 15)) == left_to_right
