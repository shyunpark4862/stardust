"""Regression tests for the array review fixes (2026-11-20): overflow guards, error paths,
layouts, the thread pool after fork, and the dtype rules decided with them."""

import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import stardust as sd

from .conftest import assert_same


def run_guarded(code, rss_limit=1 << 30, timeout=60):
    """Runs `code` in a fresh interpreter, killed past `rss_limit` bytes of RSS or `timeout`
    seconds (a regression here would eat memory or crash the process). Returns
    (returncode, output)."""
    p = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(code)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    t0 = time.monotonic()
    while p.poll() is None:
        try:
            rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(p.pid)]).strip() or 0) * 1024
        except subprocess.CalledProcessError, ValueError:
            rss = 0
        if rss > rss_limit or time.monotonic() - t0 > timeout:
            p.kill()
            p.wait()
            pytest.fail(f"killed: rss {rss / 1e6:.0f} MB after {time.monotonic() - t0:.1f} s")
        time.sleep(0.02)
    return p.returncode, p.stdout.read()


# ---- sizes that overflow int64 -----------------------------------------------


def test_concat_of_huge_broadcast_views_is_too_big():
    # Unchecked, the length sum wraps to 0 and the copies write past the allocation (SIGSEGV).
    rc, out = run_guarded(
        """
        import stardust as sd
        b = sd.broadcast_to(sd.ones(1, dtype=sd.bool), (2**62,))
        for f in [lambda: sd.concat([b] * 4), lambda: sd.concat([b] * 4, axis=None), lambda: sd.concat([b, b]),
                  lambda: sd.diff(b, prepend=b)]:
            try:
                f()
                print("returned")
            except ValueError as e:
                print(e)
        """
    )
    assert rc == 0, out
    assert out.split("\n")[:4] == [
        "concat(): array is too big",
        "concat(): array is too big",
        "concat(): array is too big",
        "diff(): array is too big",
    ]


def test_tile_product_overflow_is_too_big():
    for x, reps in [(sd.ones(4), (2**62,)), (sd.ones(2), (2**63 - 1,)), (sd.ones((4, 3)), (2**62, 1))]:
        with pytest.raises(ValueError, match=r"^tile\(\): array is too big$"):
            sd.tile(x, reps)


def test_bincount_and_histogram_at_int64_max():
    with pytest.raises(ValueError, match=r"^bincount\(\): array is too big$"):
        sd.bincount(sd.asarray([2**63 - 1]))
    with pytest.raises(ValueError, match=r"^histogram\(\): array is too big$"):  # not linspace's 'num'
        sd.histogram(sd.asarray([1.0]), bins=2**63 - 1)


def test_eye_k_far_outside_is_zero():
    for k in (-(2**63), -(2**63) + 1, -3, 3, 2**63 - 1):
        assert_same(sd.eye(3, k=k), np.eye(3, k=max(min(k, 3), -3)))
    assert_same(sd.eye(2, 4, k=-(2**63)), np.zeros((2, 4)))


def test_self_referencing_list_stops_at_32_dimensions():
    # Unchecked, the shape grows without end (853 MB in 0.3 s).
    rc, out = run_guarded(
        """
        import stardust as sd
        l = []
        l.append(l)
        for f in [lambda: sd.asarray(l), lambda: sd.ones(3)[l]]:
            try:
                f()
                print("returned")
            except ValueError as e:
                print(e)
        """,
        rss_limit=600 << 20,
    )
    assert rc == 0, out
    assert out.split("\n")[0] == "asarray(): arrays may have at most 32 dimensions"
    assert out.split("\n")[1].endswith("arrays may have at most 32 dimensions")


def test_new_array_strided_checks_the_shape():
    # outer of two 32-d arrays has 33 dimensions: unchecked, it writes past the shape array.
    rc, out = run_guarded(
        """
        import stardust as sd
        x = sd.ones((1,) * 31 + (2,))
        try:
            r = sd.linalg.outer(x, x)
            print("returned", r.ndim)
        except ValueError as e:
            print(e)
        """
    )
    assert rc == 0, out
    assert out.strip() == "outer(): arrays may have at most 32 dimensions"


def test_corrcoef_of_no_variables():
    for shape in [(0, 5), (0, 0)]:
        assert_same(sd.corrcoef(sd.zeros(shape)), np.zeros((0, 0)))
        assert_same(sd.cov(sd.zeros(shape)), np.zeros((0, 0)))


# ---- the thread pool -----------------------------------------------------------


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


# ---- DLPack ----------------------------------------------------------------------


class _Producer:
    def __init__(self, device):
        self.device = device

    def __dlpack__(self, **kw):
        return np.ones(3).__dlpack__(**kw)

    def __dlpack_device__(self):
        return self.device


def test_dlpack_argument_types():
    x = sd.ones(2)
    cases = [
        (
            lambda: sd.from_dlpack(np.ones(3), device=("cpu", 0)),
            r"^from_dlpack\(\): 'device' must be None or the CPU device \(1, 0\), not a tuple holding str$",
        ),
        (
            lambda: sd.from_dlpack(np.ones(3), device="cpu"),
            r"^from_dlpack\(\): 'device' must be None or the CPU device \(1, 0\), not str$",
        ),
        (
            lambda: x.__dlpack__(dl_device=("cpu", 0)),
            r"^__dlpack__\(\): 'dl_device' must be None or a \(device_type, device_id\) tuple of ints, "
            r"not a tuple holding str$",
        ),
        (
            lambda: x.__dlpack__(max_version=("a", "b")),
            r"^__dlpack__\(\): 'max_version' must be None or a \(major, minor\) tuple of ints, "
            r"not a tuple holding str$",
        ),
        (lambda: x.__dlpack__(max_version=(1.0, 0)), r"not a tuple holding float$"),
        (lambda: x.__dlpack__(max_version=(True, 0)), r"not a tuple holding bool$"),  # bool is no int (API.md §2)
        (lambda: x.__dlpack__(max_version=[1, 0]), r"tuple of ints, not list$"),
        (
            lambda: sd.from_dlpack(_Producer(("x", 0))),
            r"^from_dlpack\(\): __dlpack_device__ must return a \(device_type, device_id\) tuple of ints, "
            r"not a tuple holding str$",
        ),
    ]
    for f, msg in cases:
        with pytest.raises(TypeError, match=msg):
            f()
    # right types, wrong values: as before
    with pytest.raises(ValueError, match=r"'device' must be None or the CPU device \(1, 0\)$"):
        sd.from_dlpack(np.ones(3), device=(2, 0))
    with pytest.raises(BufferError):
        x.__dlpack__(dl_device=(2, 0))
    with pytest.raises(BufferError):
        sd.from_dlpack(_Producer((2, 0)))
    assert_same(sd.from_dlpack(np.ones(3), device=(1, 0)), np.ones(3))
    assert_same(sd.from_dlpack(np.ones(3), device=(np.int64(1), 0)), np.ones(3))
    assert type(x.__dlpack__(max_version=(2**70, 0))).__name__ == "PyCapsule"


def test_dlpack_bool_check_contiguous_and_strided():
    b = np.random.default_rng(1).random((300, 400)) < 0.5
    for v in [b, b[:, ::3], b[::-1], b.T]:
        assert_same(sd.from_dlpack(v), v)
    raw = np.frombuffer(bytes([0, 1, 2, 1] * 100_000), dtype=np.uint8).view(bool).copy()
    assert_same(sd.from_dlpack(raw), raw != 0)  # normalizing copy
    assert_same(sd.from_dlpack(raw.reshape(200, 2000)[:, ::2]), raw.reshape(200, 2000)[:, ::2] != 0)
    with pytest.raises(ValueError, match="holds bytes other than 0 and 1"):
        sd.from_dlpack(raw, copy=False)


# ---- numpy scalars, messages, indexing ----------------------------------------------


def test_numpy_float_scalars_are_floats_everywhere():
    x = sd.asarray([1.0, 2, 3])
    for f in (np.float16, np.float32, np.float64):
        assert sd.var(x, correction=f(1)) == 1.0
        assert sd.quantile(x, f(0.5)) == 2.0
        assert sd.cov(x, correction=f(1)) == 1.0
        assert_same(sd.histogram(x, range=(f(0), 3))[0], np.histogram([1.0, 2, 3], range=(0, 3))[0])
        assert sd.linalg.matrix_rank(sd.eye(2), rtol=f(0.1)) == 2
    for bad, shown in [(np.bool_(True), "numpy.bool"), (np.complex64(1), "numpy.complex64"), ("1", "str")]:
        with pytest.raises(TypeError, match=rf"^var\(\): 'correction' must be an int or a float, not {shown}$"):
            sd.var(x, correction=bad)


def test_messages_follow_the_rules():
    x = sd.ones(3)
    with pytest.raises(ValueError, match=r"^add\(\): operand of shape \(2, 2\) cannot be broadcast to \(2,\)$"):
        sd.add(sd.ones(2), sd.ones((2, 2)), out=sd.ones(2))
    with pytest.raises(ValueError, match=r"^operand of shape \(2, 3\) cannot be broadcast to \(3,\)$"):
        x += sd.ones((2, 3))
    with pytest.raises(MemoryError, match=r"^copy\(\): cannot allocate"):
        sd.broadcast_to(sd.ones(1), (2**40,)).copy()
    with pytest.raises(TypeError, match=r"^DType\(\): there is no constructor; use sd.bool"):
        type(sd.int64)()
    with pytest.raises(TypeError, match=r"^'==' not supported between instances of 'ndarray' and 'Series'$"):
        x == sd.frame.Series([1, 2])  # noqa: B015
    with pytest.raises(TypeError, match=r"^'!=' not supported between instances of 'ndarray' and 'Series'$"):
        x != sd.frame.Series([1, 2])  # noqa: B015
    with pytest.raises(TypeError, match=r"^asarray\(\): uint64 buffers cannot be converted to float64 without loss$"):
        sd.asarray(np.ones(2, dtype=np.uint64), dtype=sd.float64)
    with pytest.raises(TypeError, match=r"^asarray\(\): uint64 buffers cannot be converted to int64 without loss$"):
        sd.asarray(np.ones(2, dtype=np.uint64))
    with pytest.raises(
        TypeError, match=r"^asarray\(\): 'obj' must be an sd.ndarray, a list, a tuple, a range or a buffer, not str$"
    ):
        sd.asarray("abc")
    with pytest.raises(
        TypeError,
        match=r"^only a scalar, an sd.ndarray, a list, a tuple, a range or a buffer can be assigned, not NoneType$",
    ):
        x[...] = None


def test_index_errors():
    x = sd.asarray([1.0, 2, 3])
    with pytest.raises(IndexError, match=r"^a 0-dimensional array is not a valid index; use an int$"):
        x[np.array(1)]
    assert x[[[]]].shape == (1, 0) and x[[[]]].dtype == sd.float64  # as x[[]]: no positions, numpy too
    assert sd.ones((2, 3))[[[]], [[]]].shape == (1, 0)
    assert x[[]].shape == (0,)
    with pytest.raises(IndexError, match=r"^take_along_axis\(\): shapes could not be broadcast together"):
        sd.take_along_axis(sd.ones((2, 3)), sd.asarray([[0], [1], [0]]), axis=1)


def test_quantile_rejects_bool_arrays_too():
    with pytest.raises(TypeError, match=r"^quantile\(\): 'q' must be an int64 or float64 array, not bool$"):
        sd.quantile(sd.asarray([1.0, 2.0]), sd.asarray([True]))
    assert_same(sd.quantile(sd.asarray([1.0, 2.0, 3.0]), sd.asarray([0, 1])), np.array([1.0, 3.0]))


# ---- layouts, results ------------------------------------------------------------------


def test_broadcast_results_are_c_ordered():
    b = sd.broadcast_to(sd.ones(4), (3, 4))
    assert b.copy().strides == (32, 8) and (b + 1).strides == (32, 8)
    assert sd.astype(b, sd.int64).strides == (32, 8) and sd.sort(b).strides == (32, 8)
    assert (sd.broadcast_to(sd.ones(5), (2, 3, 5)) * 2).strides == (120, 40, 8)
    f = sd.asarray(np.asfortranarray(np.ones((3, 4))))
    assert (f + 1).strides == (8, 24)  # a real layout is still followed
    fb = sd.broadcast_to(sd.reshape(f, (3, 1, 4)), (3, 2, 4))
    assert (fb + 0).strides == (8, 24, 48)  # the stride-0 axis keeps its place


def test_diff_n_zero_is_x_itself():
    x = sd.asarray([1.0, 2.0])
    assert sd.diff(x, n=0, prepend=0.0) is x
    assert sd.diff(x, n=0, append=sd.asarray([9.0])) is x


def test_roll_shift_sequence_with_one_axis():
    a = np.arange(10.0).reshape(2, 5)
    for shift, axis in [((1, 2), 0), ((1, 2), 1), ((1, -1, 7), -1), ([2, 3], 1), ((), 1), ((2**62, 2**62), 1)]:
        assert_same(sd.roll(sd.asarray(a), shift, axis=axis), np.roll(a, shift, axis))
    assert_same(sd.roll(sd.arange(5.0), (1, 2), axis=0), np.roll(np.arange(5.0), 3))
    with pytest.raises(ValueError, match="repeated axis"):
        sd.roll(sd.asarray(a), 1, axis=(0, 0))


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


def test_unique_counts_matches_unique_all():
    rng = np.random.default_rng(3)
    for a in [
        np.array([0.0, -0.0, 1, np.nan, -0.0, np.nan, 1]),
        rng.integers(0, 1000, 300_000),
        rng.standard_normal(5000).round(1),
        np.array([1 + 1j, np.nan, 1 + 1j]),
        np.array([], float),
    ]:
        s = sd.asarray(a)
        c, r = sd.unique_counts(s), sd.unique_all(s)
        assert np.asarray(c.values).tobytes() == np.asarray(r.values).tobytes()
        assert_same(c.counts, np.asarray(r.counts))
        _, ni, ninv, nc = np.unique(a, return_index=True, return_inverse=True, return_counts=True, equal_nan=True)
        assert_same(c.counts, nc)
        assert_same(r.indices, ni)
        assert np.array_equal(np.asarray(r.inverse_indices).ravel(), ninv.ravel())


def test_searchsorted_without_copies():
    rng = np.random.default_rng(5)
    a = np.sort(rng.standard_normal(10_000))
    keys = rng.standard_normal(1000)
    for x1, x2 in [(a, keys), (a[::2], keys), (a, keys[::3]), (a.astype(np.int64), keys)]:
        assert_same(sd.searchsorted(sd.asarray(x1), sd.asarray(x2)), np.searchsorted(x1, x2))
        assert_same(sd.searchsorted(sd.asarray(x1), sd.asarray(x2), side="right"), np.searchsorted(x1, x2, "right"))
    assert sd.searchsorted(sd.asarray(a), 0.0) == np.searchsorted(a, 0.0)


# ---- decisions --------------------------------------------------------------------------


def test_linspace_with_an_overflowing_span_stays_finite():
    assert_same(sd.linspace(-1e308, 1e308, 5), np.array([-1e308, -5e307, 0.0, 5e307, 1e308]))
    assert_same(sd.linspace(-1e308, 1e308, 4, endpoint=False), np.array([-1e308, -5e307, 0.0, 5e307]))
    assert_same(sd.linspace(-1e308, 1e308, 1), np.array([-1e308]))
    assert_same(sd.linspace(0, 1, 5), np.linspace(0, 1, 5))  # the ordinary path is untouched
    assert np.isfinite(np.asarray(sd.linspace(-1e308 + 0j, 1e308j, 3)).view(float)).all()


def test_float_sort_keeps_the_stable_order_of_signed_zeros():
    rng = np.random.default_rng(6)
    for n in (100, 1023, 1024, 5000, 300_000):
        a = rng.choice([0.0, -0.0, 1.0, -1.0, np.nan, np.inf, -np.inf, 2.5], n)
        for desc in (False, True):
            s = sd.asarray(a)
            order = np.asarray(sd.argsort(s, descending=desc))  # the stable order
            assert np.asarray(sd.sort(s, descending=desc)).tobytes() == a[order].tobytes()
            assert np.asarray(sd.sort(s, descending=desc, stable=False)).tobytes() == a[order].tobytes()
        assert (
            np.asarray(sd.unique_values(sd.asarray(a))).tobytes()
            == np.asarray(sd.unique_all(sd.asarray(a)).values).tobytes()
        )


def test_int_mean_is_the_exact_sum_rounded_once():
    # base/int_sum.hpp: the exact integer sum, rounded once to double, divided once.
    rng = np.random.default_rng(7)
    x = rng.integers(-(2**62), 2**62, 1_000_003)
    exact = float(sum(int(v) for v in x)) / len(x)  # Python's int -> float rounds correctly
    m = rng.integers(-(2**62), 2**62, (20, 3001))
    rows = [float(sum(int(v) for v in r)) / m.shape[1] for r in m]
    old = sd.get_num_threads()
    try:
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            assert sd.mean(sd.asarray(x)) == exact
            assert sd.mean(sd.asarray(x[::-1])) == exact  # any layout: integer sums do not depend on the order
            assert sd.mean(sd.asarray(np.asfortranarray(m))) == sd.mean(sd.asarray(m))
            assert np.asarray(sd.mean(sd.asarray(m), axis=1)).tolist() == rows
            assert np.allclose(np.asarray(sd.mean(sd.asarray(m), axis=0)), m.mean(axis=0), rtol=1e-15)
    finally:
        sd.set_num_threads(old)
    assert sd.mean(sd.asarray([2**63 - 1, 2**63 - 1])) == 2.0**63  # no overflow
    b = rng.random((7, 9)) < 0.5
    assert_same(sd.mean(sd.asarray(b), axis=1), b.mean(axis=1))
    assert sd.mean(sd.asarray([True, False, True])) == 2 / 3
    assert np.isnan(sd.mean(sd.zeros(0, dtype=sd.int64)))


# ---- optimisations allowed by D-33 (BP-1, 2, 6, 7, 10) ------------------------------------


def test_weighted_histogram_with_edges_sums_each_bin():
    import math

    rng = np.random.default_rng(8)
    x = rng.standard_normal(200_000)
    w = rng.standard_normal(200_000) * 10.0 ** rng.integers(-3, 3, 200_000)
    edges = np.sort(rng.standard_normal(41))
    got = np.asarray(sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(w)).hist)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx[x == edges[-1]] = len(edges) - 2
    for b in range(len(edges) - 1):
        exact = math.fsum(w[idx == b])
        assert abs(got[b] - exact) <= 1e-13 * math.fsum(np.abs(w[idx == b])) + 1e-300
    wi = rng.integers(-(2**40), 2**40, 200_000)  # int64 weights: exact
    assert_same(
        sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(wi)).hist,
        np.histogram(x, bins=edges, weights=wi)[0],
    )
    old = sd.get_num_threads()
    try:
        res = []
        for n in (1, 10):
            sd.set_num_threads(n)
            res.append(np.asarray(sd.histogram(sd.asarray(x), bins=sd.asarray(edges), weights=sd.asarray(w)).hist))
        assert np.array_equal(res[0], res[1])
    finally:
        sd.set_num_threads(old)


def test_weighted_bincount_in_fixed_ranges():
    rng = np.random.default_rng(9)
    v = rng.integers(0, 500, 1_000_000)
    w = rng.standard_normal(1_000_000)
    old = sd.get_num_threads()
    try:
        res = []
        for n in (1, 3, 10):
            sd.set_num_threads(n)
            res.append(np.asarray(sd.bincount(sd.asarray(v), weights=sd.asarray(w))))
        assert all(np.array_equal(r, res[0]) for r in res)
    finally:
        sd.set_num_threads(old)
    assert np.allclose(res[0], np.bincount(v, w), rtol=1e-12, atol=1e-10)
    assert_same(sd.bincount(sd.asarray([0, 2, 2]), weights=sd.asarray([0.5, 1.0, 2.0])), np.array([0.5, 0.0, 3.0]))


def test_complex_abs_contract():
    import flint

    rng = np.random.default_rng(10)
    z = rng.standard_normal(2000) * np.exp(rng.uniform(-5, 5, 2000)) + 1j * rng.standard_normal(2000) * np.exp(
        rng.uniform(-5, 5, 2000)
    )
    z = np.concatenate([z, [3e-200 + 4e-200j, 1e-310 + 1e-311j, 1e300 + 1e300j, 5e-324 + 0j, 1e200 + 1e-200j]])
    got = np.asarray(sd.abs(sd.asarray(z)))
    prec = flint.ctx.prec
    flint.ctx.prec = 200
    try:
        for zi, g in zip(z, got):
            t = float((flint.arb(zi.real) ** 2 + flint.arb(zi.imag) ** 2).sqrt())
            assert abs(g - t) <= 1.5 * np.spacing(t), zi
    finally:
        flint.ctx.prec = prec
    special = np.asarray(
        sd.abs(sd.asarray([complex("inf+nanj"), complex("nan+1j"), 0j, -0.0 - 0.0j, 1.7e308 + 1e308j]))
    )
    assert (
        special[0] == np.inf and np.isnan(special[1]) and special[2] == 0 and special[3] == 0 and special[4] == np.inf
    )


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


def test_result_object_arrays_are_read_only():
    x = sd.asarray([3, 1, 3, 2])
    for r in (sd.unique_all(x), sd.unique_counts(x), sd.unique_inverse(x)):
        for a in r:
            with pytest.raises(ValueError, match="^assignment destination is read-only$"):
                a[0] = 0
            with pytest.raises(ValueError, match="read-only"):
                a[:1][0] = 0  # views of a field stay read-only
            assert np.asarray(a).flags.writeable is False
    edges = sd.asarray([0.0, 1.0, 2.0, 3.0])
    h = sd.histogram(x, bins=edges)
    for a in h:
        with pytest.raises(ValueError, match="read-only"):
            a[0] = 0
    edges[0] = -1.0  # the caller's own array stays writable
    assert h.bin_edges[0] == -1.0
    v = sd.unique_values(x)  # a plain array result is not a result object
    v[0] = 9


def test_searchsorted_sorter_positions_are_in_range():
    # Positions in [0, n): a negative one is out of range, as one past the end (one rule).
    x1 = sd.asarray([3.0, 1.0, 2.0])
    for bad, shown in [([-2, -1, 0], -2), ([1, 2, -3], -3), ([0, 1, 9], 9)]:
        with pytest.raises(
            IndexError, match=rf"^searchsorted\(\): index {shown} is out of bounds for axis 0 with size 3$"
        ):
            sd.searchsorted(x1, 2.5, sorter=sd.asarray(bad))
    assert sd.searchsorted(x1, 2.5, sorter=sd.asarray([1, 2, 0])) == 2
