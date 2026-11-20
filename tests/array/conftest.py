"""Shared helpers: numpy is the reference ("oracle") for every test."""

import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import stardust as sd
from bench.accuracy.common import ulp_distance
from tests._bits import bits_equal_any_nan

NP_DTYPE = {
    sd.bool: np.dtype(np.bool_),
    sd.int64: np.dtype(np.int64),
    sd.float64: np.dtype(np.float64),
    sd.complex128: np.dtype(np.complex128),
}
ALL_DTYPES = list(NP_DTYPE)


def assert_same(arr, expected):
    """`arr` (stardust) has exactly the shape, dtype and bits of `expected` (numpy)."""
    expected = np.asarray(expected)
    assert isinstance(arr, sd.ndarray), type(arr)
    assert arr.shape == expected.shape
    assert NP_DTYPE[arr.dtype] == expected.dtype, (arr.dtype, expected.dtype)
    got = np.asarray(arr)  # zero-copy via the buffer protocol
    assert bits_equal_any_nan(got, expected), f"\n got: {got!r}\nwant: {expected!r}"


@pytest.fixture(params=ALL_DTYPES, ids=str)
def dtype(request):
    return request.param


def _max_ulp(got, want):
    gn, wn = np.isnan(got).ravel(), np.isnan(want).ravel()
    assert np.array_equal(gn, wn), "nan positions differ"
    return int(np.max(ulp_distance(got, want), initial=0))


def assert_ulp(arr, expected, max_ulp):
    """Like assert_same, but floats may differ by up to `max_ulp` units in the
    last place (the accuracy contract of the vForce functions,
    docs/api/array.md §6.4)."""
    expected = np.asarray(expected)
    assert isinstance(arr, sd.ndarray), type(arr)
    assert arr.shape == expected.shape
    assert NP_DTYPE[arr.dtype] == expected.dtype, (arr.dtype, expected.dtype)
    got = np.asarray(arr)
    if expected.dtype.kind == "c":
        worst = max(_max_ulp(got.real, expected.real), _max_ulp(got.imag, expected.imag))
    else:
        worst = _max_ulp(got, expected)
    assert worst <= max_ulp, f"{worst} ULP apart\n got: {got!r}\nwant: {expected!r}"


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
