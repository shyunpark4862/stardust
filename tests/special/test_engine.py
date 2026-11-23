"""The n-ary elementwise engine sd.special runs on (docs/design/special.md §3.1, §3.3),
through the private test functions of stardust._core.special (csrc/special/ops/testing.hpp):

    _fma3(a, b, c)      a * b + c         three inputs
    _sum4(a, b, c, d)   ((a + b) + c) + d  four inputs: five operands, the iterator's most
    _powers4(x)         (x, x^2, x^3, x^4) four outputs, the shape of airy
    _cf_tan(x)          tan by a continued fraction, with a cost weight well above 1
"""

import math
import threading
import time

import numpy as np
import pytest

import stardust as sd
from tests._bits import bits_equal

C = sd.special._core


@pytest.fixture
def threads():
    before = sd.get_num_threads()
    yield sd.set_num_threads
    sd.set_num_threads(before)


def npy(x):
    return np.asarray(x)


# ---------------------------------------------------------------- operands and dtypes


def test_scalars_give_a_python_float():
    r = C._fma3(2.0, 3.0, 1.0)
    assert type(r) is float and r == 7.0
    r = C._sum4(1, 2, True, 4)  # int and bool are computed in float64
    assert type(r) is float and r == 8.0


def test_broadcast_three_inputs():
    rng = np.random.default_rng(0)
    a, b, c = rng.normal(size=(3, 1)), rng.normal(size=(1, 4)), rng.normal(size=(4,))
    r = C._fma3(sd.asarray(a), sd.asarray(b), sd.asarray(c))
    assert r.shape == (3, 4) and r.dtype == sd.float64
    np.testing.assert_array_equal(npy(r), a * b + c)  # same two roundings


def test_array_and_scalars_mixed():
    x = np.linspace(-2, 2, 7)
    r = C._fma3(sd.asarray(x), 2, 0.5)
    np.testing.assert_array_equal(npy(r), x * 2.0 + 0.5)


def test_bool_and_int_arrays_compute_in_float64():
    a = np.array([True, False, True])
    b = np.array([3, -4, 2**53 + 1], dtype=np.int64)  # 2**53 + 1 rounds on conversion, as in astype
    r = C._fma3(sd.asarray(a), sd.asarray(b), 1)
    assert r.dtype == sd.float64
    np.testing.assert_array_equal(npy(r), a.astype(float) * b.astype(float) + 1.0)


def test_five_operands_mixed_layouts_and_dtypes():
    # four inputs of four layouts and three dtypes (buffered casting on three of them) + the output
    rng = np.random.default_rng(1)
    base = rng.normal(size=(6, 8))
    a = np.asfortranarray(base)
    b = rng.integers(-5, 5, size=(8, 6)).T  # int64, transposed
    c = rng.normal(size=(6, 16))[:, ::-2]  # float64, reversed with a step
    d = rng.integers(0, 2, size=(8,)).astype(bool)  # bool, broadcast along the rows
    r = C._sum4(sd.asarray(a), sd.asarray(b), sd.asarray(c), sd.asarray(d))
    want = ((a + b.astype(float)) + c) + d.astype(float)
    np.testing.assert_array_equal(npy(r), want)


def test_complex_has_no_loop():
    with pytest.raises(TypeError, match=r"^_fma3\(\): not supported for complex128, int64 and int64$"):
        C._fma3(1j, 2, 3)
    with pytest.raises(TypeError, match=r"^_sum4\(\): not supported for float64, float64, complex128 and float64$"):
        C._sum4(1.0, 2.0, sd.asarray(np.array([1j])), 3.0)


def test_non_operands_name_the_argument():
    with pytest.raises(
        TypeError, match=r"^_fma3\(\): 'b' must be an sd.ndarray or a bool, int, float or complex, not list$"
    ):
        C._fma3(1.0, [2.0], 3.0)
    with pytest.raises(TypeError, match=r"^_sum4\(\): 'd' must be .*not NoneType$"):
        C._sum4(1.0, 2.0, 3.0, None)


def test_python_int_out_of_int64_range():
    # all operands bool/int64: the int must fit an int64, as for any elementwise function (docs/api/array.md §1.5)
    with pytest.raises(OverflowError):
        C._fma3(1, 2, 2**64)
    assert C._fma3(1.0, 2, 2**64) == 2.0**64 + 2.0  # next to a float it is a float


def test_shapes_that_do_not_broadcast():
    with pytest.raises(ValueError, match=r"^_fma3\(\): shapes could not be broadcast together"):
        C._fma3(sd.zeros(3), sd.zeros(4), 1.0)


def test_empty():
    r = C._fma3(sd.zeros((0, 3)), 1.0, sd.zeros(3))
    assert r.shape == (0, 3) and r.dtype == sd.float64


def test_result_follows_an_f_ordered_input():
    a = np.asfortranarray(np.arange(12.0).reshape(3, 4))
    r = C._fma3(sd.asarray(a), 2.0, 1.0)
    assert npy(r).flags.f_contiguous


# ---------------------------------------------------------------- out=


def test_out_is_written_and_returned():
    x = np.arange(5.0)
    out = sd.zeros(5)
    r = C._fma3(sd.asarray(x), 2.0, 1.0, out=out)
    assert r is out
    np.testing.assert_array_equal(npy(out), x * 2 + 1)


def test_out_may_be_an_input():
    x = sd.asarray(np.arange(5.0))
    C._fma3(x, x, x, out=x)
    np.testing.assert_array_equal(npy(x), np.arange(5.0) ** 2 + np.arange(5.0))


def test_out_partly_overlapping_an_input():
    base = np.arange(8.0)
    x = sd.asarray(base.copy())
    C._fma3(x[:-1], 1.0, 0.0, out=x[1:])  # shift right: input copied first
    np.testing.assert_array_equal(npy(x), np.concatenate([[0.0], base[:-1]]))


def test_out_dtype_rules():
    out = sd.zeros(3, dtype=sd.complex128)  # upward: computed in float64, then converted
    C._fma3(sd.ones(3), 2.0, 1.0, out=out)
    np.testing.assert_array_equal(npy(out), np.full(3, 3.0 + 0j))
    with pytest.raises(TypeError, match=r"^_fma3\(\): "):
        C._fma3(sd.ones(3), 2.0, 1.0, out=sd.zeros(3, dtype=sd.int64))


# ---------------------------------------------------------------- several outputs


def test_four_outputs():
    x = np.linspace(-3, 3, 11)
    r = C._powers4(sd.asarray(x))
    assert type(r) is tuple and len(r) == 4
    for _k, part in enumerate(r, start=1):
        assert part.dtype == sd.float64 and part.shape == x.shape
    x2 = x * x
    for got, want in zip(r, [x, x2, x2 * x, x2 * x2]):
        np.testing.assert_array_equal(npy(got), want)


def test_four_outputs_of_scalars_are_floats():
    assert C._powers4(2) == (2.0, 4.0, 8.0, 16.0)
    assert all(type(v) is float for v in C._powers4(2))


def test_four_outputs_take_no_out():
    with pytest.raises(TypeError):
        C._powers4(sd.ones(3), out=sd.zeros(3))


def test_four_outputs_strided_input():
    x = np.arange(20.0).reshape(4, 5)[:, ::2]
    r = C._powers4(sd.asarray(x))
    np.testing.assert_array_equal(npy(r[3]), (x * x) * (x * x))


# ---------------------------------------------------------------- bits and threads


def test_same_bits_for_any_thread_count(threads):
    rng = np.random.default_rng(2)
    x = sd.asarray(rng.uniform(-1.5, 1.5, size=200_000))
    a, b, c = (sd.asarray(rng.normal(size=200_000)) for _ in range(3))
    results = []
    for n in (1, 3, 10):
        threads(n)
        results.append((C._cf_tan(x), C._fma3(a, b, c), tuple(C._powers4(x))))
    for r in results[1:]:
        assert bits_equal(r, results[0])


def test_element_does_not_depend_on_its_neighbours():
    # f(x)[i] == f(x[i]): position, stride and layout do not change a value
    x = np.random.default_rng(3).uniform(-1.5, 1.5, size=101)
    whole = npy(C._cf_tan(sd.asarray(x)))
    strided = npy(C._cf_tan(sd.asarray(x)[::3]))
    for i in range(0, 101, 3):
        assert C._cf_tan(float(x[i])) == whole[i] == strided[i // 3]


# ---------------------------------------------------------------- the cost weight


def test_weight_moves_the_parallel_threshold():
    # weight 1 is the array's own rule: a float64 loop goes parallel from 2**15 elements
    assert C._parallel_tasks(2**15 - 1, 1) == 1
    assert C._parallel_tasks(2**15, 1) > 1
    # weight w splits at 2**15 / w elements, into tasks of about 2**12 / w elements
    assert C._parallel_tasks(2**15 // 64 - 1, 64) == 1
    assert C._parallel_tasks(2**15 // 64, 64) > 1
    assert C._parallel_tasks(10**6, 64) == pytest.approx(64 * C._parallel_tasks(10**6, 1), rel=0.05)


def test_functions_carry_their_weight():
    assert C._weight("_fma3") >= 1
    assert C._weight("_cf_tan") > 10 * C._weight("_fma3")


INF, NAN = math.inf, math.nan


S = sd.special


# ---- the GIL of a heavy call on few elements ----


@pytest.mark.serial
def test_heavy_call_on_few_elements_releases_the_gil():
    # yv costs 750 ns per element by the table (capped weight 256): 16 elements are below the
    # parallel threshold at the capped weight but above it at the uncapped one. These take
    # ~12 ms each (a 5e5-step recurrence), and another Python thread must keep running.
    v = sd.full((16,), 1.5e6)
    x = sd.full((16,), 1e6)
    old = sd.get_num_threads()
    sd.set_num_threads(1)
    longest = [0.0]
    stop = threading.Event()

    def watch():
        last = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            longest[0] = max(longest[0], now - last)
            last = now

    th = threading.Thread(target=watch)
    th.start()
    try:
        time.sleep(0.01)
        longest[0] = 0.0
        t = time.perf_counter()
        r = S.yv(v, x)
        took = time.perf_counter() - t
        time.sleep(0.01)
    finally:
        stop.set()
        th.join()
        sd.set_num_threads(old)
    assert np.all(np.asarray(r) == -INF)
    assert took > 0.05
    assert longest[0] < 0.5 * took, (longest[0], took)


@pytest.mark.serial
def test_heavy_scalar_call_releases_the_gil():
    # A scalar call takes the shorter way of nary_bind's call, by the same rule as an array: a
    # function whose one element is above the parallel threshold (nctdtrit, ~6.7 us by the
    # table) runs without the GIL. These take ~8 ms each (the walk runs out of steps: NaN).
    old = sd.get_num_threads()
    sd.set_num_threads(1)
    longest = [0.0]
    stop = threading.Event()

    def watch():
        last = time.perf_counter()
        while not stop.is_set():
            now = time.perf_counter()
            longest[0] = max(longest[0], now - last)
            last = now

    th = threading.Thread(target=watch)
    th.start()
    try:
        time.sleep(0.01)
        longest[0] = 0.0
        t = time.perf_counter()
        r = [S.nctdtrit(10.0, 3.9e4, 0.5) for _ in range(5)]
        each = (time.perf_counter() - t) / 5
        time.sleep(0.01)
    finally:
        stop.set()
        th.join()
        sd.set_num_threads(old)
    assert all(math.isnan(v) for v in r)
    assert each > 0.002
    assert longest[0] < 0.5 * each, (longest[0], each)
