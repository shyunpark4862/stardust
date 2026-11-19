"""The GIL rule shared by every area (API.md §3): work at or above an area's
parallel threshold runs without the GIL whatever the thread count, so other
Python threads keep running during a long call even with
``sd.set_num_threads(1)``. (The frame area's own checks are in
tests/frame/test_threads.py.)

How it is observed: a watcher thread loops in Python and records the longest
pause between two of its iterations. A call that holds the GIL freezes the
watcher for the whole call, so the pause is as long as the call; a call that
releases it leaves only short pauses.
"""

import sys
import threading
import time

import numpy as np
import pytest

import stardust as sd

N = 8_000_000  # 64 MB of float64: a millisecond or more per call on one thread


class Watcher:
    """A Python thread that measures how long it was kept from running."""

    def __init__(self):
        self.longest = 0.0
        self.turns = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run)

    def _run(self):
        last = time.perf_counter()
        while not self._stop.is_set():
            now = time.perf_counter()
            if now - last > self.longest:
                self.longest = now - last
            last = now
            self.turns += 1

    def __enter__(self):
        self._thread.start()
        while self.turns == 0:
            time.sleep(0.001)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def longest_pause_during(self, call):
        """(longest pause of the watcher, duration of call()) in seconds."""
        time.sleep(0.002)  # the watcher runs; this thread then starts the call at once
        self.longest = 0.0
        start = time.perf_counter()
        call()
        took = time.perf_counter() - start
        time.sleep(0.002)  # lets the watcher finish (and record) a pause it is still in
        return self.longest, took


class D:
    """Inputs, built by a module fixture and dropped after the module."""


@pytest.fixture(scope="module", autouse=True)
def data():
    rng = np.random.default_rng(7)
    D.x = sd.asarray(rng.standard_normal(N))
    D.u = sd.asarray(rng.random(N))
    D.short = sd.asarray(rng.standard_normal(N // 4))
    D.ints = sd.asarray(rng.integers(0, 1 << 40, N // 4))
    D.flags = sd.asarray(rng.random(N) < 0.5)
    D.stack = sd.asarray(rng.standard_normal((6000, 24, 24)) + 24.0 * np.eye(24))
    D.square = sd.asarray(rng.standard_normal((1200, 1200)) + 1200.0 * np.eye(1200))
    D.vectors = sd.asarray(rng.standard_normal(N))
    D.normal = sd.stats.Normal()
    D.design = sd.asarray(rng.standard_normal((400_000, 8)))
    D.response = sd.asarray(rng.standard_normal(400_000))
    D.designs = sd.asarray(rng.standard_normal((20_000, 50, 3)))
    D.responses = sd.asarray(rng.standard_normal((20_000, 50)))
    D.binary = sd.asarray((rng.standard_normal(400_000) > 0).astype(float))
    D.binaries = sd.asarray((rng.standard_normal((20_000, 50)) > 0).astype(float))
    before, interval = sd.get_num_threads(), sys.getswitchinterval()
    # A thread waiting for the GIL gets it within this time from a thread that
    # runs Python code. Short, so that such waits (the watcher's turn just
    # before a call starts, the caller's wait after it) are small next to a call.
    sys.setswitchinterval(0.0005)
    yield
    sys.setswitchinterval(interval)
    sd.set_num_threads(before)
    for k in [k for k in vars(D) if not k.startswith("_")]:
        delattr(D, k)


CALLS = {
    # array: the iterator's element-wise loops
    "x + x": lambda: D.x + D.x,
    "exp": lambda: sd.exp(D.x),
    "x += x (in place)": lambda: D.u.__iadd__(D.u),
    "overlapping in place": lambda: sd.add(D.u[1:], D.u[:-1], out=D.u[1:]),
    "astype": lambda: sd.astype(D.x, sd.int64),
    "where": lambda: sd.where(D.flags, D.x, 0.0),
    # array: reductions
    "sum": lambda: sd.sum(D.x),
    "sum bool": lambda: sd.sum(D.flags),
    "prod": lambda: sd.prod(D.u),
    "max": lambda: sd.max(D.x),
    "var": lambda: sd.var(D.x),
    "argmax": lambda: sd.argmax(D.x),
    "count_nonzero": lambda: sd.count_nonzero(D.x),
    "cumulative_sum int": lambda: sd.cumulative_sum(D.ints),
    "cumulative_sum float": lambda: sd.cumulative_sum(D.x),
    # array: lanes and the functions that split by hand
    "sort": lambda: sd.sort(D.short),
    "argsort int": lambda: sd.argsort(D.ints),
    "mask gather": lambda: D.x[D.flags],
    "searchsorted": lambda: sd.searchsorted(sd.linspace(-3.0, 3.0, 1000), D.short),
    "bincount": lambda: sd.bincount(D.ints % 1000),
    # array: serial loops that never split
    "bincount weights": lambda: sd.bincount(D.ints % 1000, weights=D.short),
    "unique_values": lambda: sd.unique_values(D.short),
    "full": lambda: sd.full(N, 2.5),
    "arange": lambda: sd.arange(float(N)),
    "linspace": lambda: sd.linspace(0.0, 1.0, N),
    # special and stats: the same iterator, heavier kernels
    "special.erf": lambda: sd.special.erf(D.x),
    "special.gammaln": lambda: sd.special.gammaln(D.u),
    "stats Normal cdf": lambda: D.normal.cdf(D.x),
    "stats Normal logpdf": lambda: D.normal.logpdf(D.x),
    # linalg: the batch driver (a batch, and one large matrix) and the long sums
    "linalg.det batch": lambda: sd.linalg.det(D.stack),
    "linalg.inv batch": lambda: sd.linalg.inv(D.stack),
    "linalg.inv one matrix": lambda: sd.linalg.inv(D.square),
    "linalg.vector_norm": lambda: sd.linalg.vector_norm(D.vectors),
    "linalg.vecdot": lambda: sd.linalg.vecdot(D.vectors, D.vectors),
    # regression: one large fit, and a batch through linalg's driver
    "regression.ols one fit": lambda: sd.regression.ols(D.response, D.design).params,
    "regression.ols batch": lambda: sd.regression.ols(D.responses, D.designs).params,
    "regression.glm one fit": lambda: sd.regression.glm(D.binary, D.design, family="binomial").params,
    "regression.glm batch": lambda: sd.regression.glm(D.binaries, D.designs, family="binomial").params,
    # nls: the model runs in Python; the kernel's QR and the model's large operations let go
    "regression.nls one fit": lambda: (
        sd.regression.nls(lambda x, a, b: a + b * sd.exp(-x * x), D.design[:, 0], D.response, p0=[0.5, 0.5]).params
    ),
}


@pytest.mark.parametrize("threads", [1, 10])
def test_long_calls_release_the_gil(threads):
    sd.set_num_threads(threads)
    for call in CALLS.values():
        call()  # first-use setup (tables, page faults of the allocator) outside the timed calls
    held = {}
    with Watcher() as w:
        for name, call in CALLS.items():
            # A pause from the machine being busy can hit one attempt: the best of a few counts.
            ratios = []
            for _ in range(4):
                pause, took = w.longest_pause_during(call)
                ratios.append((pause / took, took))
                if pause < 0.25 * took:
                    break
            else:
                held[name] = [f"{r:.2f} of {1e3 * t:.1f} ms" for r, t in ratios]
    assert not held, f"the watcher thread was kept waiting (threads={threads}): {held}"


def test_results_are_the_same_bits_with_1_and_10_threads():
    def bits(r):
        if isinstance(r, sd.ndarray):
            a = np.asarray(r)
            return (a.shape, a.dtype.str, np.ascontiguousarray(a).tobytes())
        return repr(r)

    skip = {"x += x (in place)", "overlapping in place"}  # they change their input
    results = {}
    for threads in (1, 10):
        sd.set_num_threads(threads)
        results[threads] = {name: bits(call()) for name, call in CALLS.items() if name not in skip}
    for name in results[1]:
        assert results[1][name] == results[10][name], name
    # the in-place ones, each from the same start
    for threads in (1, 10):
        sd.set_num_threads(threads)
        a = sd.asarray(np.arange(300_000, dtype=np.float64) * 0.37)
        a += a
        sd.add(a[1:], a[:-1], out=a[1:])
        sd.multiply(a[:-1], a[1:], out=a[:-1])
        results[threads] = bits(a)
    assert results[1] == results[10]
