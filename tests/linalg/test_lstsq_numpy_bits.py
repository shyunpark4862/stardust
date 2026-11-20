"""lstsq's gelsd path (rtol >= eps) gives numpy's bits (2026-11-19): Accelerate's dgelsd
rounds differently on its underdetermined path when b is not 16-byte aligned, and numpy's b
sits 8 m n bytes after a; sd puts b at the same offset (docs/design/linalg.md §10.4)."""

import numpy as np
import pytest

import stardust as sd

SHAPES = [(m, n) for m in (1, 2, 3, 5, 8, 9, 13) for n in (1, 2, 3, 7, 8, 11, 20)] + [(30, 41), (41, 30), (7, 100)]


@pytest.mark.parametrize("m,n", SHAPES)
def test_matches_numpy_bitwise(m, n):
    for s in range(4):
        rng = np.random.default_rng(s)
        a = rng.standard_normal((m, n))
        if s == 3 and min(m, n) > 1:
            a[:, 1] = a[:, 0]  # rank deficient
        for b in (rng.standard_normal(m), rng.standard_normal((m, 3))):
            x = np.asarray(sd.linalg.lstsq(sd.asarray(a), sd.asarray(b)).solution.tolist())
            assert np.array_equal(x, np.linalg.lstsq(a, b, rcond=None)[0]), (m, n, s, b.shape)


@pytest.mark.parametrize("m,n", [(9, 11), (5, 7), (4, 9), (12, 5)])
def test_complex_and_batches(m, n):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    b = rng.standard_normal(m) + 0j
    x = np.asarray(sd.linalg.lstsq(sd.asarray(a), sd.asarray(b)).solution.tolist())
    assert np.array_equal(x, np.linalg.lstsq(a, b, rcond=None)[0])
    A = rng.standard_normal((3, m, n))
    B = rng.standard_normal((3, m, 2))
    xs = np.asarray(sd.linalg.lstsq(sd.asarray(A), sd.asarray(B)).solution.tolist())
    for i in range(3):
        assert np.array_equal(xs[i], np.linalg.lstsq(A[i], B[i], rcond=None)[0])


def test_gelsd_path_is_numpys_for_tiny_systems():
    # R63: rtol >= eps keeps gelsd (it scales a tiny matrix itself): numpy's bits, no NaN
    rng = np.random.default_rng(7)
    for _ in range(200):
        m, n = rng.integers(1, 5, size=2)
        a = rng.standard_normal((m, n)) * 10.0 ** rng.uniform(-323, -295)
        a[rng.random((m, n)) < 0.3] = 0.0
        b = rng.standard_normal(m)
        b[rng.random(m) < 0.3] = 0.0
        x = np.asarray(sd.linalg.lstsq(sd.asarray(a), sd.asarray(b)).solution.tolist())
        want = np.linalg.lstsq(a, b, rcond=None)[0]
        assert not np.isnan(x).any()
        assert np.array_equal(x, want), (a, b)
