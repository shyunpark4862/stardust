"""The plan cache (docs/design/fft.md §3.6, decision FT24): at most 32 plans and 256 MB, least recently
used out first, a plan rebuilt after eviction gives the same bits, and Python threads calling at once
(the same length and different lengths) get the results of a lone call."""

import threading

import numpy as np

import stardust as sd

from .._bits import bits_equal

C = sd._core.fft
F = sd.fft


def test_limits_and_lru():
    C._cache_clear()
    st = C._cache_state()
    assert st["plans"] == 0 and st["bytes"] == 0
    assert st["max_plans"] == 32 and st["max_bytes"] == 256 << 20
    lengths = list(range(100, 140))
    for n in lengths:
        F.fft(sd.ones(n))
    st = C._cache_state()
    assert st["plans"] == 32
    assert st["lengths"] == lengths[-32:]  # least recently used first
    F.fft(sd.ones(lengths[-32]))  # used again: now the most recent
    st = C._cache_state()
    assert st["lengths"][-1] == lengths[-32]
    F.fft(sd.ones(999))
    st = C._cache_state()
    assert st["lengths"][0] == lengths[-30] and 999 in st["lengths"] and lengths[-31] not in st["lengths"]
    assert st["bytes"] <= st["max_bytes"]
    C._cache_clear()


def test_byte_limit():
    C._cache_clear()
    sizes = [1 << 20, 3 << 19, 1 << 21, 5 << 19, 1 << 22, 3 << 20, 15 << 18]
    made = [C._plan_info(n)["bytes"] for n in sizes]
    assert sum(made) > 256 << 20  # more than the limit: the oldest went out by bytes, not by count
    st = C._cache_state()
    assert st["bytes"] <= 256 << 20 and st["plans"] < 32
    assert st["lengths"][-1] == sizes[-1] and sizes[0] not in st["lengths"]
    assert st["bytes"] == sum(made[sizes.index(n)] for n in st["lengths"])
    C._cache_clear()


def test_rebuilt_plan_same_bits():
    g = np.random.default_rng(1)
    for n in (1000, 1009, 4096):
        x = sd.asarray(g.uniform(-1, 1, n) + 1j * g.uniform(-1, 1, n))
        a = F.fft(x)
        C._cache_clear()
        assert bits_equal(F.fft(x), a)


def test_python_threads_at_once():
    g = np.random.default_rng(2)
    xs = [sd.asarray(g.uniform(-1, 1, (8, n)) + 1j * g.uniform(-1, 1, (8, n))) for n in (1009, 1024, 997, 1009, 1024)]
    want = [np.asarray(F.fft(x)) for x in xs]
    for _ in range(3):
        C._cache_clear()
        got = [None] * len(xs)
        barrier = threading.Barrier(len(xs))

        def run(i):
            barrier.wait()
            got[i] = np.asarray(F.fft(xs[i]))

        ts = [threading.Thread(target=run, args=(i,)) for i in range(len(xs))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        for i in range(len(xs)):
            assert bits_equal(got[i], want[i])
        assert C._cache_state()["plans"] == 3
